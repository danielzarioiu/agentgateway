use std::collections::BTreeMap;

use agent_llm::types::safeguards::{SafeguardsRequest, SafeguardsReviewPlan};
use serde_json::{Value, json};

use super::super::typesafe::{Answer, TypeSafeClient, TypeSafeConfig};
use super::*;

const FIXTURES: &str = include_str!("fixtures/safety-eval-v1.json");

fn context_for(doc: &Value, case: &Value) -> Value {
	case
		.get("context")
		.cloned()
		.unwrap_or_else(|| doc["context_default"].clone())
}

fn plan_for(doc: &Value, case: &Value) -> SafeguardsReviewPlan {
	let safeguards = SafeguardsRequest {
		classifier_context: context_for(doc, case),
	};
	let transcript = case.get("transcript").cloned().unwrap_or_else(|| json!([]));
	SafeguardsReviewPlan::from_request("eval-model", &transcript, &safeguards)
}

fn review_for<'a>(plan: &'a SafeguardsReviewPlan, case: &'a Value) -> ToolCallReview<'a> {
	ToolCallReview {
		id: "toolu_eval",
		name: case["tool"]["name"].as_str().unwrap_or_default(),
		input: &case["tool"]["input"],
		plan,
	}
}

fn answers_from(value: &Value) -> BTreeMap<String, Answer> {
	let mut out = BTreeMap::new();
	let Some(object) = value.as_object() else {
		return out;
	};
	for (key, val) in object {
		match val {
			Value::Number(n) => {
				out.insert(
					key.clone(),
					Answer::Noul {
						noul: n.as_f64().unwrap_or(0.5),
					},
				);
			},
			Value::String(s) => {
				out.insert(
					key.clone(),
					Answer::Choice {
						choice: s.clone(),
						confidence: 0.9,
						probabilities: BTreeMap::new(),
					},
				);
			},
			_ => {},
		}
	}
	out
}

fn expects_flagged(expected: &str) -> bool {
	matches!(
		expected,
		"dangerous" | "needs_approval" | "flagged_local_display"
	)
}

fn verdict_matches(expect: &str, verdict: &Verdict) -> bool {
	matches!(
		(expect, verdict),
		("safe", Verdict::Safe)
			| ("dangerous", Verdict::Dangerous { .. })
			| ("needs_approval", Verdict::NeedsApproval { .. })
			| ("flagged_local_display", Verdict::Dangerous { .. })
			| ("unavailable", Verdict::Unavailable { .. })
	)
}

/// The deterministic harness: every fixture runs through `pre_decide` and, for
/// classifier-bound cases, `combine` with the scripted probabilities. False
/// negatives (expected flagged, classified safe) and false positives (expected
/// safe, classified flagged) are reported separately; an `unavailable` verdict
/// never counts as a safe classification.
#[test]
fn safety_eval_fixtures_deterministic() {
	let doc: Value = serde_json::from_str(FIXTURES).expect("fixture JSON parses");
	let cases = doc["cases"].as_array().expect("cases array");
	assert!(
		cases.len() >= 40,
		"fixture set unexpectedly small: {}",
		cases.len()
	);

	let mut false_negatives = Vec::new();
	let mut false_positives = Vec::new();
	let mut mismatches = Vec::new();

	for case in cases {
		let id = case["id"].as_str().expect("case id");
		let plan = plan_for(&doc, case);
		let review = review_for(&plan, case);
		let pre = pre_decide(&review);

		match case["expect_pre"].as_str().expect("expect_pre") {
			"mandatory_deny" => assert!(
				matches!(pre, PreDecision::MandatoryDeny { .. }),
				"{id}: expected mandatory_deny, got {pre:?}"
			),
			"fast_path_safe" => assert!(
				matches!(pre, PreDecision::FastPathSafe { .. }),
				"{id}: expected fast_path_safe, got {pre:?}"
			),
			"ask" => {
				let PreDecision::AskClassifier { floor } = pre else {
					panic!("{id}: expected ask, got {pre:?}");
				};
				let expected_floor = case["expect_floor"].as_str();
				assert_eq!(floor, expected_floor, "{id}: floor mismatch");
			},
			other => panic!("{id}: unknown expect_pre {other}"),
		}

		let Some(answers_value) = case.get("answers") else {
			// Mandatory/fast-path cases have a directly expected verdict.
			if let Some(expected) = case["expect_verdict"].as_str() {
				let verdict = match pre {
					PreDecision::MandatoryDeny { explanation, .. } => mandatory_denial_verdict(explanation),
					PreDecision::FastPathSafe { .. } => Verdict::Safe,
					PreDecision::AskClassifier { .. } => panic!("{id}: ask case without answers"),
				};
				assert!(
					verdict_matches(expected, &verdict),
					"{id}: expected {expected}, got {verdict:?}"
				);
			}
			continue;
		};

		let floor = match pre {
			PreDecision::AskClassifier { floor } => floor,
			_ => panic!("{id}: scripted-answers case must be an ask case"),
		};
		let verdict = combine(&answers_from(answers_value), floor);
		let expected = case["expect_verdict"].as_str().expect("expect_verdict");
		if !verdict_matches(expected, &verdict) {
			let entry = format!("{id}: expected {expected}, got {:?}", verdict.label());
			if expects_flagged(expected) && matches!(verdict, Verdict::Safe) {
				false_negatives.push(entry);
			} else if expected == "safe"
				&& matches!(
					verdict,
					Verdict::Dangerous { .. } | Verdict::NeedsApproval { .. }
				) {
				false_positives.push(entry);
			} else {
				mismatches.push(entry);
			}
		}
	}

	assert!(
		false_negatives.is_empty(),
		"FALSE NEGATIVES (expected flagged, classified SAFE — safety critical): {:#?}",
		false_negatives
	);
	assert!(
		false_positives.is_empty(),
		"FALSE POSITIVES (expected safe, flagged): {:#?}",
		false_positives
	);
	assert!(
		mismatches.is_empty(),
		"verdict class mismatches: {:#?}",
		mismatches
	);
}

/// Question set invariants: stable names, valid TypeSafe shapes.
#[test]
fn question_set_is_wellformed() {
	let questions = questions();
	let expected: &[&str] = &[
		"destructive",
		"secret_exposure",
		"network_exfiltration",
		"outside_trusted",
		"production_impact",
		"weakens_security",
		"instruction_origin_mismatch",
		"user_authorized",
		"overall",
	];
	assert_eq!(questions.len(), expected.len());
	for name in expected {
		assert!(questions.contains_key(*name), "missing question {name}");
	}
	for (name, question) in &questions {
		let rendered = serde_json::to_value(question).expect("serializable");
		if name == "overall" {
			assert_eq!(rendered["type"], "choice");
			continue;
		}
		assert_eq!(
			rendered["type"], "noul",
			"{name}: rubric v1 uses noul except overall"
		);
		assert!(
			rendered["criteria"]["true"].is_string(),
			"{name}: yes criterion"
		);
		assert!(
			rendered["criteria"]["false"].is_string(),
			"{name}: no criterion"
		);
	}
	assert!(
		serde_json::to_value(&questions["overall"]).unwrap()["criteria"]
			.as_object()
			.expect("choice criteria object")
			.contains_key("safe")
	);
}

#[test]
fn combine_disagreement_with_authorization_needs_approval() {
	// High risk, explicitly authorized, but the classifier's overall judgement
	// still says dangerous: never auto-approved, never silently denied — needs
	// approval (flagged with an approval explanation).
	let answers = answers_from(&json!({
		"destructive": 0.9, "secret_exposure": 0.05, "network_exfiltration": 0.05,
		"outside_trusted": 0.5, "production_impact": 0.85, "weakens_security": 0.05,
		"instruction_origin_mismatch": 0.05, "user_authorized": 0.95, "overall": "dangerous"
	}));
	match combine(&answers, Some("kubernetes-mutation")) {
		Verdict::NeedsApproval { explanation } => {
			assert!(explanation.contains("Manual approval"), "{explanation}");
		},
		other => panic!("expected needs_approval, got {other:?}"),
	}
}

#[test]
fn combine_partial_answer_is_unavailable_never_safe() {
	let answers = answers_from(&json!({
		"destructive": 0.1, "secret_exposure": 0.1, "network_exfiltration": 0.1,
		"outside_trusted": 0.1, "production_impact": 0.1, "weakens_security": 0.1,
		"instruction_origin_mismatch": 0.1, "user_authorized": 0.9
		// overall missing
	}));
	assert!(matches!(
		combine(&answers, None),
		Verdict::Unavailable { .. }
	));
	let mut empty = BTreeMap::new();
	empty.insert(
		"overall".to_string(),
		Answer::Choice {
			choice: "safe".to_string(),
			confidence: 1.0,
			probabilities: BTreeMap::new(),
		},
	);
	assert!(matches!(combine(&empty, None), Verdict::Unavailable { .. }));
}

#[test]
fn trusted_directory_containment() {
	let trusted = vec!["/work/repo".to_string()];
	let (cwd, home) = (Some("/work/repo"), Some("/Users/dev"));
	assert!(is_within_trusted(
		"/work/repo/src/a.rs",
		&trusted,
		cwd,
		home
	));
	assert!(is_within_trusted("src/a.rs", &trusted, cwd, home));
	assert!(is_within_trusted(
		"~/dev/x",
		&["~/dev".to_string()],
		cwd,
		home
	));
	assert!(
		!is_within_trusted("/work/repo-secret/x", &trusted, cwd, home),
		"prefix must not match sibling dirs"
	);
	assert!(!is_within_trusted("../escape", &trusted, cwd, home));
	assert!(!is_within_trusted("/etc/hosts", &trusted, cwd, home));
	assert!(!is_within_trusted("relative/unknown", &trusted, None, home));
	assert!(!is_within_trusted("~/x", &trusted, cwd, None));
}

/// LIVE evaluation against the real TypeSafe classifier. Skipped unless both
/// `AGENTGATEWAY_SAFEGUARDS_LIVE_EVAL=1` and `TYPESAFE_API_KEY` are set.
/// Runs every `ask` case through the real Jev model and reports false
/// negatives (fails) and false positives (reported) separately.
#[tokio::test]
#[ignore = "requires TYPESAFE_API_KEY; enable with AGENTGATEWAY_SAFEGUARDS_LIVE_EVAL=1"]
async fn live_typesafe_safety_eval() {
	if std::env::var("AGENTGATEWAY_SAFEGUARDS_LIVE_EVAL")
		.ok()
		.as_deref()
		!= Some("1")
	{
		eprintln!("set AGENTGATEWAY_SAFEGUARDS_LIVE_EVAL=1 to run the live evaluation");
		return;
	}
	let Ok(api_key) = std::env::var("TYPESAFE_API_KEY") else {
		panic!("TYPESAFE_API_KEY is not set");
	};
	let client = TypeSafeClient::new(TypeSafeConfig {
		base_url: std::env::var("TYPESAFE_BASE_URL")
			.unwrap_or_else(|_| super::super::typesafe::DEFAULT_BASE_URL.to_string()),
		api_key: secrecy::SecretString::from(api_key),
		model: std::env::var("TYPESAFE_MODEL")
			.unwrap_or_else(|_| super::super::typesafe::DEFAULT_MODEL.to_string()),
		timeout: std::time::Duration::from_secs(60),
		long_context: false,
	})
	.expect("client");

	let verification = client.verify_model().await.expect("models verification");
	assert!(
		verification.available,
		"model {} is not served; listed: {:?} — never substitute an alias",
		verification.requested, verification.listed
	);

	let doc: Value = serde_json::from_str(FIXTURES).expect("fixture JSON");
	let cases = doc["cases"].as_array().expect("cases");
	let mut false_negatives = Vec::new();
	let mut false_positives = Vec::new();
	let mut unavailable = Vec::new();
	let mut evaluated = 0usize;

	for case in cases {
		if case.get("answers").is_none() {
			continue;
		}
		let id = case["id"].as_str().unwrap_or_default();
		let plan = plan_for(&doc, case);
		let review = review_for(&plan, case);
		let PreDecision::AskClassifier { floor } = pre_decide(&review) else {
			continue;
		};
		evaluated += 1;
		let state = build_state(&review, &[]);
		let result = tokio::time::timeout(
			std::time::Duration::from_secs(90),
			client.system_one(state, questions()),
		)
		.await
		.expect("per-case timeout")
		.expect("system_one call");
		let verdict = combine(&result.response.answers, floor);
		let expected = case["expect_verdict"].as_str().unwrap_or_default();
		eprintln!(
			"{id}: expected={expected} actual={:?} answers={:?}",
			verdict.label(),
			result.response.answers
		);
		if matches!(verdict, Verdict::Unavailable { .. }) {
			unavailable.push(id.to_string());
			continue;
		}
		if !verdict_matches(expected, &verdict) {
			if expects_flagged(expected) && matches!(verdict, Verdict::Safe) {
				false_negatives.push(id.to_string());
			} else if expected == "safe" && expects_flagged(verdict.label()) {
				false_positives.push(id.to_string());
			}
		}
	}

	eprintln!(
		"LIVE EVAL: evaluated={evaluated} false_negatives={} false_positives={} unavailable={:#?}",
		false_negatives.len(),
		false_positives.len(),
		unavailable
	);
	assert!(
		false_negatives.is_empty(),
		"FALSE NEGATIVES with the live classifier: {false_negatives:?}"
	);
}

#[test]
fn verdict_to_protocol_mapping() {
	assert_eq!(
		Verdict::Safe.to_protocol(),
		agent_llm::types::safeguards::ToolUseVerdict::not_flagged()
	);
	let Verdict::Dangerous { explanation } = mandatory_denial_verdict("why".to_string()) else {
		panic!()
	};
	match (Verdict::Dangerous { explanation }).to_protocol() {
		agent_llm::types::safeguards::ToolUseVerdict::Evaluated {
			outcome: agent_llm::types::safeguards::Outcome::Flagged,
			explanation: Some(text),
		} => assert_eq!(text, "why"),
		other => panic!("expected flagged verdict, got {other:?}"),
	}
	assert!(matches!(
		Verdict::NeedsApproval {
			explanation: "x".into()
		}
		.to_protocol(),
		agent_llm::types::safeguards::ToolUseVerdict::Evaluated {
			outcome: agent_llm::types::safeguards::Outcome::Flagged,
			..
		}
	));
}
