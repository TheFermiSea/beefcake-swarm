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
    build_reviewer_named(client, model, "reviewer", None)
}

/// Build the blind reviewer with a custom agent name.
pub fn build_reviewer_named(
    client: &openai::CompletionsClient,
    model: &str,
    name: &str,
    wt_path: Option<&std::path::Path>,
) -> OaiAgent {
    let preamble = match wt_path {
        Some(path) => prompts::load_prompt("reviewer", path, prompts::REVIEWER_PREAMBLE),
        None => prompts::REVIEWER_PREAMBLE.to_string(),
    };
    client
        .agent(model)
        .name(name)
        .description("Blind code reviewer. Returns strict JSON validation output.")
        .preamble(&preamble)
        .temperature(0.1)
        .build()
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum ReviewVerdict {
    Pass,
    Fail,
    NeedsEscalation,
}

/// Scored criteria dimensions (0–3 each).
///
/// Verdict is PASS only if ALL scores are ≥ 2. Scores are produced by
/// the reviewer and used by the orchestrator to diagnose failure patterns.
/// These are surfaced in `HarnessComponentTrace` for load-bearing audits.
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct CriteriaScores {
    /// Does the code do what was asked? Are error paths handled? (0–3)
    pub correctness: u8,
    /// Is the implementation complete? No stubs / todo! / unimplemented!? (0–3)
    pub completeness: u8,
    /// Are errors propagated with `?`? No unwrap on fallible ops? (0–3)
    pub robustness: u8,
    /// Idiomatic Rust, no new clippy warnings, consistent with codebase? (0–3)
    pub conventions: u8,
}

impl CriteriaScores {
    /// Whether all criteria meet the pass threshold (≥ 2).
    pub fn all_passing(&self) -> bool {
        self.correctness >= 2
            && self.completeness >= 2
            && self.robustness >= 2
            && self.conventions >= 2
    }

    /// Return names of criteria that scored below 2.
    pub fn failing_criteria(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.correctness < 2 {
            out.push("correctness");
        }
        if self.completeness < 2 {
            out.push("completeness");
        }
        if self.robustness < 2 {
            out.push("robustness");
        }
        if self.conventions < 2 {
            out.push("conventions");
        }
        out
    }
}

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct StructuredReview {
    verdict: ReviewVerdict,
    #[serde(default)]
    scores: Option<CriteriaScoresRaw>,
    confidence: f32,
    blocking_issues: Vec<String>,
    suggested_next_action: String,
    touched_files: Vec<String>,
}

/// Raw scored dimensions as returned by the LLM (may contain values > 3 from
/// a misbehaving model — clamped on parse).
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
struct CriteriaScoresRaw {
    correctness: u8,
    completeness: u8,
    robustness: u8,
    conventions: u8,
}

/// Parse a reviewer response into pass/fail + feedback.
#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ReviewResult {
    pub passed: bool,
    pub schema_valid: bool,
    pub confidence: f32,
    /// Scored criteria dimensions from the calibrated rubric.
    /// Present when the reviewer returned the new schema format; `None` for
    /// legacy responses that predate the scored rubric.
    pub scores: Option<CriteriaScores>,
    pub blocking_issues: Vec<String>,
    pub suggested_next_action: String,
    pub touched_files: Vec<String>,
    pub feedback: String,
    /// Auto-detected patterns that may indicate reviewer leniency.
    /// Non-empty means the orchestrator should treat this result with extra
    /// skepticism (e.g. escalate even if verdict is pass).
    pub leniency_flags: Vec<String>,
}

impl ReviewResult {
    pub fn parse(response: &str) -> Self {
        // Strip markdown code fences that LLMs sometimes wrap around JSON.
        // e.g. "```json\n{...}\n```" → "{...}"
        let cleaned = strip_markdown_fences(response);
        match serde_json::from_str::<StructuredReview>(&cleaned) {
            Ok(parsed) => {
                let passed = matches!(parsed.verdict, ReviewVerdict::Pass);
                let scores = parsed.scores.map(|raw| CriteriaScores {
                    correctness: raw.correctness.min(3),
                    completeness: raw.completeness.min(3),
                    robustness: raw.robustness.min(3),
                    conventions: raw.conventions.min(3),
                });

                // Cross-check: if verdict is pass but scores say otherwise, flag it.
                let leniency_flags = detect_leniency_flags(passed, scores.as_ref());

                Self {
                    passed,
                    schema_valid: true,
                    confidence: parsed.confidence,
                    scores,
                    blocking_issues: parsed.blocking_issues,
                    suggested_next_action: parsed.suggested_next_action,
                    touched_files: parsed.touched_files,
                    feedback: response.to_string(),
                    leniency_flags,
                }
            }
            Err(err) => Self {
                passed: false,
                schema_valid: false,
                confidence: 0.0,
                scores: None,
                blocking_issues: vec![format!("Invalid reviewer schema: {err}")],
                suggested_next_action:
                    "Return strict JSON with verdict/scores/confidence/blocking_issues/suggested_next_action/touched_files.".to_string(),
                touched_files: Vec::new(),
                feedback: response.to_string(),
                leniency_flags: Vec::new(),
            },
        }
    }
}

/// Detect patterns that indicate the reviewer may have been too lenient.
///
/// Returns a list of flag descriptions. The orchestrator surfaces these
/// in `HarnessComponentTrace.reviewer_leniency_flags` for audit telemetry.
fn detect_leniency_flags(passed: bool, scores: Option<&CriteriaScores>) -> Vec<String> {
    let mut flags = Vec::new();

    if let Some(s) = scores {
        // Verdict says pass but scored criteria say fail
        if passed && !s.all_passing() {
            let failing = s.failing_criteria();
            flags.push(format!("verdict=pass but criteria score <2: {:?}", failing));
        }
        // Perfect scores with no explanation is suspicious for complex diffs
        if passed
            && s.correctness == 3
            && s.completeness == 3
            && s.robustness == 3
            && s.conventions == 3
        {
            flags.push("all scores=3 (maximum) — verify evaluator was not over-generous".into());
        }
    }

    flags
}

/// Strip markdown code fences from LLM responses.
///
/// LLMs sometimes wrap JSON output in ` ```json ... ``` `. This extracts
/// the content between the fences, or returns the original string if no
/// fences are found.
fn strip_markdown_fences(s: &str) -> String {
    let trimmed = s.trim();
    if let Some(rest) = trimmed.strip_prefix("```") {
        // Skip the optional language tag on the opening fence line
        let after_lang = rest.find('\n').map(|i| &rest[i + 1..]).unwrap_or(rest);
        // Remove the closing fence
        if let Some(end) = after_lang.rfind("```") {
            return after_lang[..end].trim().to_string();
        }
    }
    trimmed.to_string()
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
    fn test_parse_pass_with_scores() {
        let result = ReviewResult::parse(
            r#"{
                "verdict": "pass",
                "scores": {"correctness": 2, "completeness": 2, "robustness": 3, "conventions": 2},
                "confidence": 0.93,
                "blocking_issues": [],
                "suggested_next_action": "merge",
                "touched_files": ["src/main.rs"]
            }"#,
        );
        assert!(result.passed);
        assert!(result.schema_valid);
        let scores = result.scores.unwrap();
        assert!(scores.all_passing());
        assert!(result.leniency_flags.is_empty());
    }

    #[test]
    fn test_leniency_flag_pass_with_low_score() {
        let result = ReviewResult::parse(
            r#"{
                "verdict": "pass",
                "scores": {"correctness": 2, "completeness": 1, "robustness": 2, "conventions": 2},
                "confidence": 0.8,
                "blocking_issues": [],
                "suggested_next_action": "merge",
                "touched_files": ["src/lib.rs"]
            }"#,
        );
        // completeness < 2 but verdict is pass — should flag leniency
        assert!(result.passed);
        assert!(
            !result.leniency_flags.is_empty(),
            "should have leniency flag"
        );
        assert!(result.leniency_flags[0].contains("completeness"));
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
    fn test_criteria_scores_failing_criteria() {
        let scores = CriteriaScores {
            correctness: 2,
            completeness: 1,
            robustness: 0,
            conventions: 2,
        };
        let failing = scores.failing_criteria();
        assert_eq!(failing, vec!["completeness", "robustness"]);
        assert!(!scores.all_passing());
    }

    #[test]
    fn test_parse_markdown_fenced_json() {
        let fenced = "```json\n{\n  \"verdict\": \"pass\",\n  \"confidence\": 0.95,\n  \"blocking_issues\": [],\n  \"suggested_next_action\": \"merge\",\n  \"touched_files\": [\"src/runtime_adapter.rs\"]\n}\n```";
        let result = ReviewResult::parse(fenced);
        assert!(result.passed);
        assert!(result.schema_valid);
        assert_eq!(result.confidence, 0.95);
    }

    #[test]
    fn test_strip_markdown_fences() {
        assert_eq!(strip_markdown_fences("```json\n{}\n```"), "{}");
        assert_eq!(strip_markdown_fences("```\n{}\n```"), "{}");
        assert_eq!(strip_markdown_fences("{}"), "{}");
        assert_eq!(strip_markdown_fences("  ```json\n{}\n```  "), "{}");
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
