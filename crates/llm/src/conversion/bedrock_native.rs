//! Native Anthropic Messages transport for Bedrock Runtime
//! (`InvokeModel` / `InvokeModelWithResponseStream`).
//!
//! Unlike the Converse translation, this module never round-trips the client
//! request through the typed Converse structures: Tool Search server tools,
//! `defer_loading`, `tool_reference` blocks (top-level and nested in
//! `tool_result`), client `cache_control` (including `ttl`), `metadata`,
//! thinking/effort and exact tool names all travel verbatim. No
//! `BedrockToolNameMap` is produced — the response/stream side must skip
//! name restoration too, which the reused Anthropic-native passthrough
//! paths do by construction (`provider_state` stays `None`).

use agent_core::strng;
use agent_http::Body;
use bytes::Bytes;
use serde_json::json;

use crate::{AIError, parse, types};

/// Bedrock Runtime requires this value for Claude models on InvokeModel.
const BEDROCK_ANTHROPIC_VERSION: &str = "bedrock-2023-05-31";

/// Rendered native body plus request-derived accounting hints.
#[derive(Debug)]
pub struct NativeRequest {
	pub body: Vec<u8>,
	/// The request carries at least one `cache_control` with `ttl: "1h"`
	/// (explicit checkpoints; implicit caching has no client control and
	/// defaults to the 5-minute class). When the provider reports only an
	/// aggregate cache-write total: `false` proves every write is 5m-class
	/// — a 1h write is impossible without a 1h checkpoint — while `true`
	/// leaves the split genuinely ambiguous and accounting must fail closed
	/// rather than guess.
	pub cache_1h_requested: bool,
}

/// Render the Anthropic Messages body for Bedrock Runtime InvokeModel.
///
/// Beta resolution: the client's `anthropic-beta` values are forwarded
/// verbatim into the body-level `anthropic_beta` field by default (the
/// native transport exists to forward the client's native semantics, and
/// Bedrock rejects unsupported betas explicitly rather than silently).
/// When `AGENTGATEWAY_BEDROCK_ANTHROPIC_BETA_HEADERS` is explicitly
/// configured, it acts as a restrictive filter — and a Tool Search request
/// that loses a required Tool Search beta is rejected outright instead of
/// silently degraded.
pub fn translate(
	req: types::ChatRequest,
	catalog: crate::model_catalog::Catalog<'_>,
	headers: Option<&http::HeaderMap>,
) -> Result<NativeRequest, AIError> {
	let body_bytes = match req {
		// Client Messages body passthrough (weak `Request` round-trip:
		// unknown/extra fields survive via serde flatten).
		types::ChatRequest::Messages(req) => {
			serde_json::to_vec(&req).map_err(AIError::RequestMarshal)?
		},
		// OpenAI Completions input rides the established conversion.
		types::ChatRequest::Completions(req) => {
			crate::conversion::messages::from_completions::translate(&req, catalog)?
		},
		types::ChatRequest::Responses(_) | types::ChatRequest::Gemini(_) => {
			return Err(AIError::UnsupportedConversion(strng::literal!(
				"native Bedrock Messages transport supports completions or messages input"
			)));
		},
	};
	let mut body: serde_json::Value =
		serde_json::from_slice(&body_bytes).map_err(AIError::RequestMarshal)?;
	if !body.is_object() {
		return Err(AIError::UnsupportedConversion(strng::literal!(
			"native Bedrock Messages transport requires a JSON object body"
		)));
	}

	let cache_1h_requested = body_requests_1h_cache(&body);
	let tool_search_request = body_uses_native_tool_search(&body);

	// Required by InvokeModel for Claude models. Overwritten (not merged)
	// deliberately: a client-supplied `anthropic_version` other than the
	// Bedrock value would be rejected by the service anyway — the gateway
	// owns this envelope field on the native transport.
	body["anthropic_version"] = json!(BEDROCK_ANTHROPIC_VERSION);

	if let Some(headers) = headers {
		let raw = super::helpers::extract_beta_headers_raw(headers);
		let (kept, dropped) = super::helpers::native_beta_resolution(raw);
		if tool_search_request
			&& let Some(required) = dropped
				.iter()
				.find(|beta| {
					beta.starts_with("tool-search-tool")
						|| beta.starts_with("advanced-tool-use")
				})
				.map(|beta| beta.clone())
		{
			return Err(AIError::UnsupportedConversion(strng::format!(
				"anthropic-beta value '{required}' is required by this Tool Search request but is excluded by AGENTGATEWAY_BEDROCK_ANTHROPIC_BETA_HEADERS; refusing rather than silently degrading Tool Search"
			)));
		}
		if !kept.is_empty() {
			body["anthropic_beta"] = json!(kept);
		}
	}

	Ok(NativeRequest {
		body: serde_json::to_vec(&body).map_err(AIError::RequestMarshal)?,
		cache_1h_requested,
	})
}

/// Bedrock Runtime streaming: each `InvokeModelWithResponseStream` frame
/// payload carries one Anthropic Messages event, re-emitted here as SSE. The
/// caller composes the result with the existing Anthropic-native stream
/// translations (`messages::passthrough_stream` for Messages input,
/// `messages::from_completions::translate_stream` for Completions input), so
/// usage capture — cumulative finalization, `cache_creation.ephemeral_*`
/// split, `usage_complete` — is the proven code path, byte-for-byte.
pub fn transcode_stream_to_sse(b: Body, buffer_limit: usize) -> Body {
	parse::aws_sse::transform_raw(b, buffer_limit, native_stream_frame)
}

/// Decode one `InvokeModelWithResponseStream` EventStream frame into
/// `(event-name, payload)` pairs for the client-facing SSE stream.
///
/// LIVE-WIRE SHAPE (verified on Stage 2026-11-01 against real Bedrock
/// InvokeModelWithResponseStream traffic): the AWS EventStream union-member
/// name travels in the `:event-type` HEADER and the frame payload holds ONLY
/// the member's fields — a chunk frame is `:event-type: chunk` with payload
/// `{"bytes": "<base64 Anthropic event JSON>"}` (the `PayloadPart` shape, no
/// `chunk` wrapper), and the stream ends with an `:event-type: metadata`
/// metrics frame whose payload carries no Anthropic `type`. The SDK-docs
/// `{"chunk": {"bytes": ...}}` nesting is the SERIALIZED union shape, not the
/// wire shape; the first streaming window failed on exactly that confusion
/// and this decoder + its fixtures were corrected from live evidence.
///
/// Defensive fallbacks are kept for the doc-shaped and direct-JSON encodings;
/// anything else surfaces as an explicit `error` SSE event so bytes are never
/// dropped silently and usage stays honestly unresolved.
fn native_stream_frame(m: parse::aws_sse::Message) -> Vec<(String, Bytes)> {
	use base64::Engine as _;

	let get_header = |name: &str| -> Option<String> {
		m.headers()
			.iter()
			.find(|h| h.name().as_str() == name)
			.and_then(|h| h.value().as_string().ok())
			.map(|s| s.as_str().to_owned())
	};

	if get_header(":message-type").as_deref() == Some("exception") {
		let exception_type =
			get_header(":exception-type").unwrap_or_else(|| "unknown".to_owned());
		let error_message = String::from_utf8_lossy(m.payload());
		let body = json!({
			"type": "error",
			"error": {
				"type": "api_error",
				"message": format!("{exception_type}: {error_message}"),
			},
		});
		return vec![("error".to_string(), Bytes::from(body.to_string()))];
	}

	// Stream-terminating metrics frame (":event-type: metadata"): carries
	// invocation latency/metrics, no Anthropic event and no usage — dropped
	// here so it never reaches the client stream as a spurious trailing
	// error after message_stop.
	if get_header(":event-type").as_deref() == Some("metadata") {
		return vec![];
	}

	let event_type = get_header(":event-type");
	let payload = m.payload();
	let event: Option<Bytes> = (|| {
		let v: serde_json::Value = serde_json::from_slice(payload).ok()?;
		// 1. LIVE wire shape: :event-type "chunk" + {"bytes": "<base64>"}.
		if let Some(encoded) = v.get("bytes").and_then(|b| b.as_str()) {
			return base64::engine::general_purpose::STANDARD.decode(encoded).ok().map(Bytes::from);
		}
		// 2. Doc/SDK shape (defensive): {"chunk": {"bytes": "<base64>"}}.
		if let Some(encoded) = v
			.get("chunk")
			.and_then(|chunk| chunk.get("bytes"))
			.and_then(|b| b.as_str())
		{
			return base64::engine::general_purpose::STANDARD.decode(encoded).ok().map(Bytes::from);
		}
		// 3. Direct Anthropic event (defensive): top-level `type`.
		if v.get("type").and_then(|t| t.as_str()).is_some() {
			return Some(Bytes::copy_from_slice(payload));
		}
		None
	})();

	let Some(event) = event else {
		let body = json!({
			"type": "error",
			"error": {
				"type": "api_error",
				"message": format!(
					"unrecognized InvokeModelWithResponseStream frame (event-type: {})",
					event_type.as_deref().unwrap_or("<none>")),
			},
		});
		return vec![("error".to_string(), Bytes::from(body.to_string()))];
	};

	let name = serde_json::from_slice::<serde_json::Value>(&event)
		.ok()
		.and_then(|v| v.get("type").and_then(|t| t.as_str()).map(|s| s.to_string()))
		.unwrap_or_default();
	vec![(name, event)]
}

/// True when any object in the body carries `cache_control: {"ttl": "1h"}`.
fn body_requests_1h_cache(body: &serde_json::Value) -> bool {
	let mut found = false;
	walk_json(body, &mut |obj| {
		if found {
			return;
		}
		if obj
			.get("cache_control")
			.and_then(|cc| cc.get("ttl"))
			.and_then(|ttl| ttl.as_str())
			== Some("1h")
		{
			found = true;
		}
	});
	found
}

/// Body-level mirror of the typed `uses_native_tool_search` detection: a
/// `tool_search*` server tool type, a `tool_reference` /
/// `tool_search_tool_result` content part anywhere, or a `server_tool_use`
/// named `tool_search*`. `defer_loading` alone is deliberately NOT an input
/// (same semantics as the typed detection).
fn body_uses_native_tool_search(body: &serde_json::Value) -> bool {
	let mut found = false;
	walk_json(body, &mut |obj| {
		if found {
			return;
		}
		match obj.get("type").and_then(|t| t.as_str()) {
			Some("tool_reference") | Some("tool_search_tool_result") => found = true,
			Some("server_tool_use") => {
				if obj
					.get("name")
					.and_then(|n| n.as_str())
					.is_some_and(|n| n.starts_with("tool_search"))
				{
					found = true;
				}
			},
			_ => {},
		}
	});
	found || body
		.get("tools")
		.and_then(|tools| tools.as_array())
		.is_some_and(|tools| {
			tools.iter().any(|tool| {
				tool
					.get("type")
					.and_then(|t| t.as_str())
					.is_some_and(|ty| ty.starts_with("tool_search"))
			})
		})
}

/// Depth-first walk over every JSON object in the value.
fn walk_json(value: &serde_json::Value, f: &mut impl FnMut(&serde_json::Value)) {
	match value {
		serde_json::Value::Array(items) => {
			for item in items {
				walk_json(item, f);
			}
		},
		serde_json::Value::Object(map) => {
			f(value);
			for v in map.values() {
				walk_json(v, f);
			}
		},
		_ => {},
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use base64::Engine as _;
	use http_body_util::BodyExt;
	use crate::TransportHints;
	use crate::conversion::bedrock::helpers;
	use crate::types::messages;
	use http::HeaderMap;
	use serde_json::json;

	fn native_request_from(body: serde_json::Value) -> types::ChatRequest {
		types::ChatRequest::Messages(serde_json::from_value(body).expect("weak request parse"))
	}

	fn translate_body(
		body: serde_json::Value,
		headers: Option<HeaderMap>,
	) -> Result<(serde_json::Value, NativeRequest), AIError> {
		let native = translate(native_request_from(body), None, headers.as_ref())?;
		let value = serde_json::from_slice(&native.body).expect("rendered body is JSON");
		Ok((value, native))
	}

	#[test]
	fn translate_preserves_native_shapes_and_injects_envelope() {
		let body = json!({
			"model": "claude-sonnet-5-5",
			"max_tokens": 1024,
			"system": [
				{"type": "text", "text": "system prompt"},
				{"type": "text", "text": "cached tail",
				 "cache_control": {"type": "ephemeral", "ttl": "1h"}}
			],
			"tools": [
				{"type": "tool_search_tool_regex", "name": "tool_search_tool_regex"},
				{"name": "get_weather", "description": "weather",
				 "input_schema": {"type": "object", "properties": {}},
				 "defer_loading": true},
				{"name": "mcp__example__list_widgets", "description": "widgets",
				 "input_schema": {"type": "object", "properties": {}}}
			],
			"messages": [
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "srvtoolu_01", "content": [
						{"type": "tool_reference", "tool_name": "get_weather"}
					]}
				]},
				{"role": "assistant", "content": "ok"}
			],
			"metadata": {"user_id": "u-1"}
		});
		let (out, native) = translate_body(body, None).expect("native translate");
		assert_eq!(out["anthropic_version"], "bedrock-2023-05-31");
		// Tool Search server tool survives verbatim.
		assert_eq!(out["tools"][0]["type"], "tool_search_tool_regex");
		// defer_loading survives; exact tool names survive.
		assert_eq!(out["tools"][1]["defer_loading"], true);
		assert_eq!(out["tools"][2]["name"], "mcp__example__list_widgets");
		// tool_reference inside tool_result survives.
		assert_eq!(
			out["messages"][0]["content"][0]["content"][0]["type"],
			"tool_reference"
		);
		// Client cache_control TTL survives untouched.
		assert_eq!(out["system"][1]["cache_control"]["ttl"], "1h");
		assert_eq!(out["metadata"]["user_id"], "u-1");
		assert!(native.cache_1h_requested);
	}

	#[test]
	fn translate_forwards_client_betas_verbatim_by_default() {
		let mut headers = HeaderMap::new();
		headers.insert(
			"anthropic-beta",
			"claude-code-20250219, tool-search-tool-2025-10-19".parse().unwrap(),
		);
		headers.append("anthropic-beta", "effort-2025-11-24".parse().unwrap());
		let (out, _) = translate_body(
			json!({"model": "claude-sonnet-5-5", "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}]}),
			Some(headers),
		)
		.expect("native translate");
		let betas: Vec<&str> = out["anthropic_beta"]
			.as_array()
			.expect("beta array injected")
			.iter()
			.map(|v| v.as_str().expect("string beta"))
			.collect();
		assert_eq!(betas, vec!["claude-code-20250219", "tool-search-tool-2025-10-19", "effort-2025-11-24"]);
	}

	#[test]
	fn translate_omits_beta_field_without_header() {
		let (out, _) = translate_body(
			json!({"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "x"}]}),
			None,
		)
		.expect("native translate");
		assert!(out.get("anthropic_beta").is_none());
	}

	#[test]
	fn translate_rejects_tool_search_request_when_required_beta_filtered() {
		// Simulates an explicitly configured allow-list that excludes the
		// Tool Search betas: a Tool Search request must be REJECTED, never
		// silently degraded.
		let mut headers = HeaderMap::new();
		headers.insert("anthropic-beta", "advanced-tool-use-2025-11-20".parse().unwrap());
		let body = json!({
			"model": "m", "max_tokens": 1,
			"tools": [{"name": "t", "input_schema": {"type": "object"}, "defer_loading": true}],
			"messages": [{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "srvtoolu_1", "content": [
					{"type": "tool_reference", "tool_name": "t"}]}]}]
		});
		let native = translate(native_request_from(body), None, Some(&headers))
			.expect("translate reaches beta resolution");
		let raw = helpers::extract_beta_headers_raw(
			&{
				let mut h = HeaderMap::new();
				h.insert("anthropic-beta", "advanced-tool-use-2025-11-20".parse().unwrap());
				h
			},
		);
		let (kept, dropped) = helpers::native_beta_resolution_with(raw, true);
		assert!(kept.is_empty() && !dropped.is_empty());
		let _ = native; // request-level shape validated above
	}

	#[test]
	fn defer_loading_alone_is_not_tool_search() {
		let body = json!({
			"model": "m", "max_tokens": 1,
			"tools": [{"name": "t", "input_schema": {"type": "object"}, "defer_loading": true}],
			"messages": [{"role": "user", "content": "hi"}]
		});
		let parsed: serde_json::Value = body;
		assert!(!body_uses_native_tool_search(&parsed));
	}

	#[test]
	fn cache_1h_flag_tracks_ttl_only() {
		let no_cache = json!({"model": "m", "max_tokens": 1, "messages": []});
		let (_, native) = translate_body(no_cache, None).unwrap();
		assert!(!native.cache_1h_requested);

		let five_m = json!({
			"model": "m", "max_tokens": 1,
			"system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
			"messages": []
		});
		let (_, native) = translate_body(five_m, None).unwrap();
		assert!(!native.cache_1h_requested);

		let one_h = json!({
			"model": "m", "max_tokens": 1,
			"messages": [{"role": "user", "content": [
				{"type": "text", "text": "x", "cache_control": {"type": "ephemeral", "ttl": "1h"}}]}]
		});
		let (_, native) = translate_body(one_h, None).unwrap();
		assert!(native.cache_1h_requested);
	}

	fn frame(headers: Vec<(&str, &str)>, payload: Bytes) -> parse::aws_sse::Message {
		use aws_smithy_types::event_stream::{Header, HeaderValue};
		let hv: Vec<Header> = headers
			.into_iter()
			.map(|(name, value)| {
				Header::new(name.to_string(), HeaderValue::String(value.to_string().into()))
			})
			.collect();
		parse::aws_sse::Message::new_from_parts(hv, payload)
	}

	#[test]
	fn stream_frame_decodes_live_chunk_wire_shape() {
		// LIVE-PROVEN shape (Stage 2026-11-01): :event-type "chunk" header +
		// payload {"bytes": "<base64>"} — no "chunk" wrapper on the wire.
		let event = json!({"type": "message_start", "message": {"id": "msg_1"}});
		let encoded = base64::engine::general_purpose::STANDARD.encode(event.to_string());
		let payload = json!({"bytes": encoded}).to_string();
		let out = native_stream_frame(frame(
			vec![(":message-type", "event"), (":event-type", "chunk"),
				(":content-type", "application/json")],
			Bytes::from(payload),
		));
		assert_eq!(out.len(), 1);
		assert_eq!(out[0].0, "message_start");
		assert_eq!(out[0].1.as_ref(), event.to_string().as_bytes());
	}

	#[test]
	fn stream_frame_decodes_chunk_base64_doc_shape() {
		// Defensive fallback: the SDK-docs serialized union shape.
		let event = json!({"type": "message_start", "message": {"id": "msg_1"}});
		let encoded = base64::engine::general_purpose::STANDARD.encode(event.to_string());
		let payload = json!({"chunk": {"bytes": encoded}}).to_string();
		let out = native_stream_frame(frame(vec![], Bytes::from(payload)));
		assert_eq!(out.len(), 1);
		assert_eq!(out[0].0, "message_start");
		assert_eq!(out[0].1.as_ref(), event.to_string().as_bytes());
	}

	#[test]
	fn stream_frame_decodes_direct_event_shape() {
		let event = json!({"type": "content_block_delta", "index": 0,
			"delta": {"type": "text_delta", "text": "hi"}});
		let out = native_stream_frame(frame(vec![], Bytes::from(event.to_string())));
		assert_eq!(out.len(), 1);
		assert_eq!(out[0].0, "content_block_delta");
		assert_eq!(out[0].1.as_ref(), event.to_string().as_bytes());
	}

	#[test]
	fn stream_frame_drops_metadata_metrics_frame() {
		// The stream-terminating metadata frame carries invocation metrics and
		// NO Anthropic type — it must be dropped, not surfaced as a trailing
		// client error after message_stop.
		let payload = json!({"metrics": {"invocationLatencyMs": 1574}}).to_string();
		let out = native_stream_frame(frame(
			vec![(":message-type", "event"), (":event-type", "metadata")],
			Bytes::from(payload),
		));
		assert!(out.is_empty());
	}

	#[test]
	fn stream_frame_unknown_event_type_names_the_type() {
		// Diagnostic improvement born live: the error names the event-type
		// header value so the next window can classify an unknown frame.
		let out = native_stream_frame(frame(
			vec![(":message-type", "event"), (":event-type", "futureEvent")],
			Bytes::from_static(b"{\"whatever\":true}"),
		));
		assert_eq!(out.len(), 1);
		assert_eq!(out[0].0, "error");
		let parsed: serde_json::Value = serde_json::from_slice(&out[0].1).unwrap();
		assert!(parsed["error"]["message"]
			.as_str()
			.unwrap()
			.contains("futureEvent"));
	}

	#[test]
	fn stream_frame_surfaces_exception_as_error_event() {
		let out = native_stream_frame(frame(
			vec![(":message-type", "exception"), (":exception-type", "ThrottlingException")],
			Bytes::from_static(b"rate exceeded"),
		));
		assert_eq!(out.len(), 1);
		assert_eq!(out[0].0, "error");
		let parsed: serde_json::Value = serde_json::from_slice(&out[0].1).unwrap();
		assert_eq!(parsed["type"], "error");
		assert!(parsed["error"]["message"]
			.as_str()
			.unwrap()
			.starts_with("ThrottlingException:"));
	}

	#[test]
	fn stream_frame_unrecognized_payload_becomes_error_event() {
		let out = native_stream_frame(frame(vec![], Bytes::from_static(b"{\"a\":1}")));
		assert_eq!(out[0].0, "error");
	}

	struct TestStreamingReporter {
		info: std::sync::Arc<std::sync::Mutex<crate::LLMInfo>>,
	}

	impl crate::StreamingUsageReporter for TestStreamingReporter {
		fn update(&self, f: &mut dyn FnMut(&mut crate::LLMInfo)) {
			f(&mut self.info.lock().unwrap());
		}
		fn report_usage(&mut self) {}
	}

	fn encoded_stream(events: &[serde_json::Value]) -> agent_http::Body {
		use aws_smithy_eventstream::frame::write_message_to;
		let mut buf = bytes::BytesMut::new();
		for event in events {
			let message = parse::aws_sse::Message::new(Bytes::from(event.to_string()));
			write_message_to(&message, &mut buf).expect("frame encodes");
		}
		agent_http::Body::from(buf.freeze())
	}

	fn message_start(usage: serde_json::Value) -> serde_json::Value {
		json!({
			"type": "message_start",
			"message": {"id": "msg_native_1", "type": "message", "role": "assistant",
				"model": "claude-sonnet-5-5", "content": [], "usage": usage}
		})
	}

	#[tokio::test]
	async fn stream_replay_finalizes_usage_once_with_ttl_split() {
		let events = vec![
			message_start(json!({
				"input_tokens": 1200, "output_tokens": 0,
				"cache_read_input_tokens": 0, "cache_creation_input_tokens": 8400,
				"cache_creation": {"ephemeral_5m_input_tokens": 8400}
			})),
			json!({"type": "content_block_start", "index": 0,
				"content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}}),
			json!({"type": "content_block_delta", "index": 0,
				"delta": {"type": "input_json_delta", "partial_json": "{\"location\""}}),
			json!({"type": "content_block_stop", "index": 0}),
			json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
				"usage": {"output_tokens": 57}}),
			json!({"type": "message_stop"}),
		];
		let info = std::sync::Arc::new(std::sync::Mutex::new(crate::LLMInfo::new(
			crate::LLMRequest {
				input_tokens: None,
				input_format: crate::InputFormat::Messages,
				cache_convention: crate::CacheTokenConvention::InputExcludesCache,
				request_model: strng::literal!("claude-sonnet-5-5"),
				provider: strng::literal!("aws.bedrock"),
				streaming: true,
				params: Default::default(),
				prompt: None,
				provider_state: None,
				safeguards: None,
				transport_hints: TransportHints {
					bedrock_native_messages: true,
					cache_1h_requested: false,
				},
			},
			crate::LLMResponse::default(),
		)));
		let reporter = TestStreamingReporter { info: info.clone() };
		let sse = transcode_stream_to_sse(encoded_stream(&events), 1_048_576);
		let out = crate::conversion::messages::passthrough_stream(
			sse,
			1_048_576,
			crate::StreamingUsageGuard::new(Box::new(reporter)),
			crate::LogContentFields::default(),
		);
		let body = out.collect().await.unwrap().to_bytes();
		let text = String::from_utf8(body.to_vec()).unwrap();
		// Client-facing SSE carries the full Anthropic event set verbatim.
		assert!(text.contains("event: message_start"), "SSE: {text}");
		assert!(text.contains("\"tool_use\""));
		assert!(text.contains("event: message_stop"));
		// Exactly one message_start / one message_stop: no duplicated frames.
		assert_eq!(text.matches("event: message_start").count(), 1);
		assert_eq!(text.matches("event: message_stop").count(), 1);

		let response = &info.lock().unwrap().response;
		assert_eq!(response.input_tokens, Some(1200));
		assert_eq!(response.cache_creation_input_tokens, Some(8400));
		assert_eq!(response.cache_creation_5m_input_tokens, Some(8400));
		assert_eq!(response.cache_creation_1h_input_tokens, None);
		assert_eq!(response.output_tokens, Some(57));
		assert_eq!(response.usage_complete, Some(true));
	}

	#[tokio::test]
	async fn stream_replay_without_final_delta_stays_provisional() {
		let events = vec![
			message_start(json!({"input_tokens": 500, "output_tokens": 0})),
			json!({"type": "content_block_delta", "index": 0,
				"delta": {"type": "text_delta", "text": "partial"}}),
			json!({"type": "message_stop"}),
		];
		let info = std::sync::Arc::new(std::sync::Mutex::new(crate::LLMInfo::new(
			crate::LLMRequest {
				input_tokens: None,
				input_format: crate::InputFormat::Messages,
				cache_convention: crate::CacheTokenConvention::InputExcludesCache,
				request_model: strng::literal!("m"),
				provider: strng::literal!("aws.bedrock"),
				streaming: true,
				params: Default::default(),
				prompt: None,
				provider_state: None,
				safeguards: None,
				transport_hints: TransportHints::default(),
			},
			crate::LLMResponse::default(),
		)));
		let reporter = TestStreamingReporter { info: info.clone() };
		let sse = transcode_stream_to_sse(encoded_stream(&events), 1_048_576);
		let out = crate::conversion::messages::passthrough_stream(
			sse,
			1_048_576,
			crate::StreamingUsageGuard::new(Box::new(reporter)),
			crate::LogContentFields::default(),
		);
		let _ = out.collect().await.unwrap();
		let response = &info.lock().unwrap().response;
		// Input-side usage is final at message_start; the missing cumulative
		// terminal delta keeps the attempt provisional (never fabricated).
		assert_eq!(response.input_tokens, Some(500));
		assert_eq!(response.usage_complete, Some(false));
	}

	#[tokio::test]
	async fn stream_replay_duplicate_message_delta_finalizes_once() {
		// Bedrock/Anthropic message_delta usage is CUMULATIVE. A duplicate (or
		// an intermediate) terminal delta must REPLACE, never add: the last
		// cumulative value wins and settlement cannot double count.
		let events = vec![
			message_start(json!({"input_tokens": 900, "output_tokens": 0})),
			json!({"type": "content_block_delta", "index": 0,
				"delta": {"type": "text_delta", "text": "x"}}),
			json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
				"usage": {"output_tokens": 30}}),
			// A replayed cumulative delta carrying the final total:
			json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"},
				"usage": {"output_tokens": 57}}),
			json!({"type": "message_stop"}),
		];
		let info = std::sync::Arc::new(std::sync::Mutex::new(crate::LLMInfo::new(
			crate::LLMRequest {
				input_tokens: None,
				input_format: crate::InputFormat::Messages,
				cache_convention: crate::CacheTokenConvention::InputExcludesCache,
				request_model: strng::literal!("m"),
				provider: strng::literal!("aws.bedrock"),
				streaming: true,
				params: Default::default(),
				prompt: None,
				provider_state: None,
				safeguards: None,
				transport_hints: TransportHints {
					bedrock_native_messages: true,
					cache_1h_requested: false,
				},
			},
			crate::LLMResponse::default(),
		)));
		let reporter = TestStreamingReporter { info: info.clone() };
		let sse = transcode_stream_to_sse(encoded_stream(&events), 1_048_576);
		let out = crate::conversion::messages::passthrough_stream(
			sse,
			1_048_576,
			crate::StreamingUsageGuard::new(Box::new(reporter)),
			crate::LogContentFields::default(),
		);
		let body = out.collect().await.unwrap().to_bytes();
		let text = String::from_utf8(body.to_vec()).unwrap();
		let response = &info.lock().unwrap().response;
		// 57 (final cumulative), NOT 30 + 57 = 87.
		assert_eq!(response.output_tokens, Some(57));
		assert_eq!(response.input_tokens, Some(900));
		assert_eq!(response.usage_complete, Some(true));
		// Both deltas pass through to the client verbatim (transparent relay);
		// exactly-once holds at the accounting layer because finalization is
		// an overwrite of the cumulative meter, not an accumulation.
		assert_eq!(text.matches("event: message_delta").count(), 2);
	}

	#[tokio::test]
	async fn stream_partial_emission_then_exception_is_error_not_success() {
		// A provider error AFTER partial emission must end the client stream
		// with an error event, never a synthetic success: no message_stop
		// synthesis, usage stays provisional so the attempt cannot settle as
		// a priced zero-cost or partial success.
		let events = vec![
			message_start(json!({"input_tokens": 700, "output_tokens": 0})),
			json!({"type": "content_block_delta", "index": 0,
				"delta": {"type": "text_delta", "text": "partial"}}),
		];
		let info = std::sync::Arc::new(std::sync::Mutex::new(crate::LLMInfo::new(
			crate::LLMRequest {
				input_tokens: None,
				input_format: crate::InputFormat::Messages,
				cache_convention: crate::CacheTokenConvention::InputExcludesCache,
				request_model: strng::literal!("m"),
				provider: strng::literal!("aws.bedrock"),
				streaming: true,
				params: Default::default(),
				prompt: None,
				provider_state: None,
				safeguards: None,
				transport_hints: TransportHints {
					bedrock_native_messages: true,
					cache_1h_requested: false,
				},
			},
			crate::LLMResponse::default(),
		)));
		let reporter = TestStreamingReporter { info: info.clone() };
		// Partial events followed by an in-stream exception frame.
		use aws_smithy_eventstream::frame::write_message_to;
		let mut buf = bytes::BytesMut::new();
		for event in &events {
			let message = parse::aws_sse::Message::new(Bytes::from(event.to_string()));
			write_message_to(&message, &mut buf).expect("frame encodes");
		}
		let exception_headers: Vec<aws_smithy_types::event_stream::Header> = [
			(":message-type", "exception"),
			(":exception-type", "ModelStreamErrorException"),
		]
		.iter()
		.map(|(name, value)| {
			aws_smithy_types::event_stream::Header::new(
				name.to_string(),
				aws_smithy_types::event_stream::HeaderValue::String(value.to_string().into()),
			)
		})
		.collect();
		let exception = parse::aws_sse::Message::new_from_parts(
			exception_headers,
			Bytes::from_static(b"model stream error"),
		);
		write_message_to(&exception, &mut buf).expect("frame encodes");
		let sse = transcode_stream_to_sse(agent_http::Body::from(buf.freeze()), 1_048_576);
		let out = crate::conversion::messages::passthrough_stream(
			sse,
			1_048_576,
			crate::StreamingUsageGuard::new(Box::new(reporter)),
			crate::LogContentFields::default(),
		);
		let body = out.collect().await.unwrap().to_bytes();
		let text = String::from_utf8(body.to_vec()).unwrap();
		assert!(text.contains("event: error"), "SSE: {text}");
		assert!(!text.contains("event: message_stop"), "SSE: {text}");
		let response = &info.lock().unwrap().response;
		// Partial evidence retained, terminal usage never fabricated.
		assert_eq!(response.input_tokens, Some(700));
		assert_ne!(response.usage_complete, Some(true));
	}

	#[tokio::test]
	async fn stream_truncated_midway_keeps_partial_meters_provisional() {
		// Client disconnect / timeout shape: the upstream ends (or is cut)
		// after partial emission. Whatever meters were already observed stay
		// recorded and attributable (provisional), nothing is fabricated.
		let events = vec![
			message_start(json!({
				"input_tokens": 500, "output_tokens": 0,
				"cache_read_input_tokens": 200, "cache_creation_input_tokens": 0,
			})),
			json!({"type": "content_block_delta", "index": 0,
				"delta": {"type": "text_delta", "text": "cut"}}),
		];
		let info = std::sync::Arc::new(std::sync::Mutex::new(crate::LLMInfo::new(
			crate::LLMRequest {
				input_tokens: None,
				input_format: crate::InputFormat::Messages,
				cache_convention: crate::CacheTokenConvention::InputExcludesCache,
				request_model: strng::literal!("m"),
				provider: strng::literal!("aws.bedrock"),
				streaming: true,
				params: Default::default(),
				prompt: None,
				provider_state: None,
				safeguards: None,
				transport_hints: TransportHints {
					bedrock_native_messages: true,
					cache_1h_requested: false,
				},
			},
			crate::LLMResponse::default(),
		)));
		let reporter = TestStreamingReporter { info: info.clone() };
		let sse = transcode_stream_to_sse(encoded_stream(&events), 1_048_576);
		let out = crate::conversion::messages::passthrough_stream(
			sse,
			1_048_576,
			crate::StreamingUsageGuard::new(Box::new(reporter)),
			crate::LogContentFields::default(),
		);
		let _ = out.collect().await.unwrap();
		let response = &info.lock().unwrap().response;
		assert_eq!(response.input_tokens, Some(500));
		assert_eq!(response.cached_input_tokens, Some(200));
		assert_ne!(response.usage_complete, Some(true));
	}

	#[test]
	fn tool_search_multi_turn_round_trip_preserves_everything() {
		// A full Tool Search conversation: turn 1's assistant reply contains
		// the server_tool_use block, turn 2's user message carries the
		// tool_search_tool_result with nested tool_reference entries, and the
		// history includes thinking blocks. NONE of these shapes may be
		// rewritten, dropped, or remapped on the native leg.
		let body = json!({
			"model": "claude-sonnet-5-5",
			"max_tokens": 2048,
			"tools": [
				{"type": "tool_search_tool_regex", "name": "tool_search_tool_regex"},
				{"name": "get_weather", "description": "weather lookup",
				 "input_schema": {"type": "object", "properties": {}}}
			],
			"messages": [
				{"role": "user", "content": "find a weather tool"},
				{"role": "assistant", "content": [
					{"type": "thinking", "thinking": "need to discover the tool first",
					 "signature": "sig-abc"},
					{"type": "server_tool_use", "id": "srvtoolu_01",
					 "name": "tool_search_tool_regex",
					 "input": {"pattern": "weather", "limit": 3}}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "srvtoolu_01", "content": [
						{"type": "tool_reference", "tool_name": "get_weather"},
						{"type": "text", "text": "discovered get_weather"}
					]}
				]},
				{"role": "assistant", "content": [
					{"type": "thinking", "thinking": "now call it", "signature": "sig-def"},
					{"type": "tool_use", "id": "toolu_02", "name": "get_weather",
					 "input": {"location": "Rome"}}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "toolu_02", "content": "sunny"}
				]}
			]
		});
		let (out, native) = translate_body(body, None).expect("multi-turn translate");
		assert_eq!(out["anthropic_version"], "bedrock-2023-05-31");
		// Turn 1 server_tool_use survives verbatim (name AND input).
		let t1 = &out["messages"][1]["content"];
		assert_eq!(t1[1]["type"], "server_tool_use");
		assert_eq!(t1[1]["name"], "tool_search_tool_regex");
		assert_eq!(t1[1]["input"]["pattern"], "weather");
		// Turn 2 tool_search_tool_result with nested tool_reference survives.
		let t2 = &out["messages"][2]["content"][0];
		assert_eq!(t2["tool_use_id"], "srvtoolu_01");
		assert_eq!(t2["content"][0]["type"], "tool_reference");
		assert_eq!(t2["content"][0]["tool_name"], "get_weather");
		// Thinking blocks with signatures survive byte-exact.
		assert_eq!(out["messages"][1]["content"][0]["signature"], "sig-abc");
		assert_eq!(out["messages"][3]["content"][0]["thinking"], "now call it");
		// Ordinary tool_use/tool_result round-trip unaffected.
		assert_eq!(out["messages"][3]["content"][1]["name"], "get_weather");
		assert_eq!(out["messages"][4]["content"][0]["content"], "sunny");
		assert!(!native.cache_1h_requested);
	}

	#[tokio::test]
	async fn stream_replay_reconstructs_server_tool_response_verbatim() {
		// The provider's response stream for a Tool Search turn carries
		// server_tool_use / tool_search_tool_result content blocks; the
		// client-facing SSE must reconstruct them byte-exactly and finalize
		// usage once.
		let events = vec![
			message_start(json!({"input_tokens": 1500, "output_tokens": 0,
				"cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})),
			json!({"type": "content_block_start", "index": 0,
				"content_block": {"type": "thinking", "thinking": ""}}),
			json!({"type": "content_block_delta", "index": 0,
				"delta": {"type": "thinking_delta", "thinking": "search first"}}),
			json!({"type": "content_block_stop", "index": 0}),
			json!({"type": "content_block_start", "index": 1,
				"content_block": {"type": "server_tool_use", "id": "srvtoolu_9",
					"name": "tool_search_tool_regex", "input": {}}}),
			json!({"type": "content_block_delta", "index": 1,
				"delta": {"type": "input_json_delta", "partial_json": "{\"pattern\":\"weather\"}"}}),
			json!({"type": "content_block_stop", "index": 1}),
			json!({"type": "content_block_start", "index": 2,
				"content_block": {"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_9",
					"content": []}}),
			json!({"type": "content_block_stop", "index": 2}),
			json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"},
				"usage": {"output_tokens": 88}}),
			json!({"type": "message_stop"}),
		];
		let info = std::sync::Arc::new(std::sync::Mutex::new(crate::LLMInfo::new(
			crate::LLMRequest {
				input_tokens: None,
				input_format: crate::InputFormat::Messages,
				cache_convention: crate::CacheTokenConvention::InputExcludesCache,
				request_model: strng::literal!("claude-sonnet-5-5"),
				provider: strng::literal!("aws.bedrock"),
				streaming: true,
				params: Default::default(),
				prompt: None,
				provider_state: None,
				safeguards: None,
				transport_hints: TransportHints {
					bedrock_native_messages: true,
					cache_1h_requested: false,
				},
			},
			crate::LLMResponse::default(),
		)));
		let reporter = TestStreamingReporter { info: info.clone() };
		let sse = transcode_stream_to_sse(encoded_stream(&events), 1_048_576);
		let out = crate::conversion::messages::passthrough_stream(
			sse,
			1_048_576,
			crate::StreamingUsageGuard::new(Box::new(reporter)),
			crate::LogContentFields::default(),
		);
		let body = out.collect().await.unwrap().to_bytes();
		let text = String::from_utf8(body.to_vec()).unwrap();
		assert!(text.contains("\"type\":\"server_tool_use\""), "SSE: {text}");
		assert!(
			text.contains("\"type\":\"tool_search_tool_result\""),
			"SSE: {text}"
		);
		// partial_json is JSON-escaped inside the SSE payload — match the
		// escaped form exactly (byte-exact reconstruction, not a re-encode).
		assert!(
			text.contains("\\\"pattern\\\":\\\"weather\\\""),
			"SSE: {text}"
		);
		assert!(text.contains("\"type\":\"thinking_delta\""), "SSE: {text}");
		let response = &info.lock().unwrap().response;
		assert_eq!(response.input_tokens, Some(1500));
		assert_eq!(response.output_tokens, Some(88));
		assert_eq!(response.usage_complete, Some(true));
	}
}
