//! Blind code reviewer agent.

use rig::client::CompletionClient;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::prompts;

use super::coder::OaiAgent;

/// Build the blind reviewer agent.
///
/// NO tools — the reviewer only sees a diff passed via prompt.
/// Returns PASS or FAIL on the first line.
pub fn build_reviewer(client: &openai::CompletionsClient, model: &str) -> OaiAgent {
    build_reviewer_named(client, model, "reviewer")
}

/// Build the blind reviewer with a custom agent name.
pub fn build_reviewer_named(
    client: &openai::CompletionsClient,
    model: &str,
    name: &str,
) -> OaiAgent {
    let mut builder = client
        .agent(model)
        .name(name)
        .description("Blind code reviewer. Returns strict JSON validation output.")
        .preamble(prompts::REVIEWER_PREAMBLE)
        .temperature(0.1);

    // Attach GBNF grammar for structured review output when enabled.
    // This forces llama-server to produce valid JSON matching the
    // StructuredReview schema, eliminating fail-closed fallbacks from
    // malformed output.
    if let Some(params) =
        crate::grammars::params_if_enabled(crate::grammars::Grammar::ReviewVerdict)
    {
        builder = builder.additional_params(params);
    }

    builder.build()
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum ReviewVerdict {
    Pass,
    Fail,
    NeedsEscalation,
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StructuredReview {
    verdict: ReviewVerdict,
    confidence: f32,
    blocking_issues: Vec<String>,
    suggested_next_action: String,
    touched_files: Vec<String>,
}

/// Parse a reviewer response into pass/fail + feedback.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReviewResult {
    pub passed: bool,
    pub schema_valid: bool,
    pub confidence: f32,
    pub blocking_issues: Vec<String>,
    pub suggested_next_action: String,
    pub touched_files: Vec<String>,
    pub feedback: String,
}

impl ReviewResult {
    pub fn parse(response: &str) -> Self {
        match serde_json::from_str::<StructuredReview>(response) {
            Ok(parsed) => {
                let passed = matches!(parsed.verdict, ReviewVerdict::Pass);
                Self {
                    passed,
                    schema_valid: true,
                    confidence: parsed.confidence,
                    blocking_issues: parsed.blocking_issues,
                    suggested_next_action: parsed.suggested_next_action,
                    touched_files: parsed.touched_files,
                    feedback: response.to_string(),
                }
            }
            Err(err) => Self {
                passed: false,
                schema_valid: false,
                confidence: 0.0,
                blocking_issues: vec![format!("Invalid reviewer schema: {err}")],
                suggested_next_action:
                    "Return strict JSON with verdict/confidence/blocking_issues/suggested_next_action/touched_files.".to_string(),
                touched_files: Vec::new(),
                feedback: response.to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pass() {
        let result = ReviewResult::parse(
            r#"{
                "verdict": "pass",
                "confidence": 0.93,
                "blocking_issues": [],
                "suggested_next_action": "merge",
                "touched_files": ["src/main.rs"]
            }"#,
        );
        assert!(result.passed);
        assert!(result.schema_valid);
        assert_eq!(result.touched_files, vec!["src/main.rs".to_string()]);
    }

    #[test]
    fn test_parse_fail() {
        let result = ReviewResult::parse(
            r#"{
                "verdict": "fail",
                "confidence": 0.7,
                "blocking_issues": ["missing error handling"],
                "suggested_next_action": "fix and re-run",
                "touched_files": ["src/lib.rs"]
            }"#,
        );
        assert!(!result.passed);
        assert!(result.schema_valid);
        assert_eq!(result.blocking_issues.len(), 1);
    }

    #[test]
    fn test_parse_invalid_schema_fails_closed() {
        let result = ReviewResult::parse("PASS\nlooks okay");
        assert!(!result.passed);
        assert!(!result.schema_valid);
        assert!(result
            .blocking_issues
            .first()
            .is_some_and(|s| s.contains("Invalid reviewer schema")));
    }
}
