//! Optional LLM judge for resolution quality scoring.
//!
//! Evaluates whether a diff actually addresses the issue, is minimal,
//! and avoids side effects. Produces a 0-100 quality score that can
//! be blended with the deterministic verifier health score.
//!
//! **Not wired into the orchestrator yet** -- this module provides the
//! scoring primitives only.  Integration (after verifier, before
//! experiment_db record) is a follow-up.

use regex::Regex;

/// Everything the judge needs to evaluate a resolution.
#[derive(Debug, Clone)]
pub struct JudgeInput {
    pub issue_title: String,
    pub issue_description: String,
    pub diff_summary: String,
    pub verifier_report: String,
}

/// Build the system+user prompt sent to the judge LLM.
///
/// The prompt asks for a single numeric 0-100 quality score and a
/// one-line rationale.  We intentionally keep the expected output
/// format simple so that even small local models can comply.
pub fn build_judge_prompt(input: &JudgeInput) -> String {
    format!(
        r#"You are a code-review judge. Score the following resolution on a 0-100 scale.

Criteria (equal weight):
1. **Relevance** — Does the diff address the issue described?
2. **Minimality** — Are changes limited to what is necessary?
3. **Side-effects** — Does the diff introduce unrelated changes or regressions?
4. **Correctness** — Does the verifier report indicate a healthy build?

## Issue
**Title:** {title}
**Description:** {description}

## Diff summary
```
{diff}
```

## Verifier report
```
{verifier}
```

Respond with ONLY a line in this format:
Score: <number 0-100>

Do not include any other text."#,
        title = input.issue_title,
        description = input.issue_description,
        diff = input.diff_summary,
        verifier = input.verifier_report,
    )
}

/// Extract a numeric 0-100 score from the judge LLM's response.
///
/// Handles several common formats:
/// - `Score: 85`
/// - `85/100`
/// - `{"score": 85}` (JSON)
/// - Plain number `85`
pub fn parse_judge_score(response: &str) -> Option<f64> {
    let trimmed = response.trim();

    // Try "Score: <N>" (case-insensitive)
    let score_prefix = Regex::new(r"(?i)score\s*:\s*(\d{1,3})").expect("judge score-prefix regex");
    if let Some(caps) = score_prefix.captures(trimmed) {
        if let Some(val) = caps.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) {
            return clamp_score(val);
        }
    }

    // Try "<N>/100"
    let fraction = Regex::new(r"(\d{1,3})\s*/\s*100").expect("judge fraction regex");
    if let Some(caps) = fraction.captures(trimmed) {
        if let Some(val) = caps.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) {
            return clamp_score(val);
        }
    }

    // Try JSON: {"score": N} or {"score":N}
    let json_score = Regex::new(r#""score"\s*:\s*(\d{1,3})"#).expect("judge json-score regex");
    if let Some(caps) = json_score.captures(trimmed) {
        if let Some(val) = caps.get(1).and_then(|m| m.as_str().parse::<f64>().ok()) {
            return clamp_score(val);
        }
    }

    // Try plain number (entire response is just a number, possibly with whitespace)
    if let Ok(val) = trimmed.parse::<f64>() {
        return clamp_score(val);
    }

    None
}

fn clamp_score(val: f64) -> Option<f64> {
    if (0.0..=100.0).contains(&val) {
        Some(val)
    } else {
        None
    }
}

/// Blend the deterministic verifier score with the LLM judge score.
///
/// `judge_ratio` controls the weight given to the judge (0.0 = verifier
/// only, 1.0 = judge only).  Default is 0.20 (80% verifier, 20% judge).
pub fn blend_scores(verifier_score: f64, judge_score: f64, judge_ratio: f64) -> f64 {
    let ratio = judge_ratio.clamp(0.0, 1.0);
    let blended = (1.0 - ratio) * verifier_score + ratio * judge_score;
    blended.clamp(0.0, 100.0)
}

/// Default judge-to-verifier weight ratio (20% judge).
pub const DEFAULT_JUDGE_RATIO: f64 = 0.20;

#[cfg(test)]
mod tests {
    use super::*;

    // ── prompt building ─────────────────────────────────────────────

    #[test]
    fn prompt_contains_issue_and_diff() {
        let input = JudgeInput {
            issue_title: "Fix clippy warnings".into(),
            issue_description: "Remove unused imports".into(),
            diff_summary: "--- a/lib.rs\n+++ b/lib.rs".into(),
            verifier_report: "All checks passed".into(),
        };
        let prompt = build_judge_prompt(&input);
        assert!(prompt.contains("Fix clippy warnings"));
        assert!(prompt.contains("Remove unused imports"));
        assert!(prompt.contains("--- a/lib.rs"));
        assert!(prompt.contains("All checks passed"));
        assert!(prompt.contains("Score:"));
    }

    // ── score parsing ───────────────────────────────────────────────

    #[test]
    fn parse_score_prefix() {
        assert_eq!(parse_judge_score("Score: 85"), Some(85.0));
        assert_eq!(parse_judge_score("score:90"), Some(90.0));
        assert_eq!(parse_judge_score("SCORE : 42"), Some(42.0));
    }

    #[test]
    fn parse_fraction() {
        assert_eq!(parse_judge_score("85/100"), Some(85.0));
        assert_eq!(parse_judge_score("  70 / 100  "), Some(70.0));
    }

    #[test]
    fn parse_json() {
        assert_eq!(parse_judge_score(r#"{"score": 75}"#), Some(75.0));
        assert_eq!(parse_judge_score(r#"{"score":100}"#), Some(100.0));
    }

    #[test]
    fn parse_plain_number() {
        assert_eq!(parse_judge_score("88"), Some(88.0));
        assert_eq!(parse_judge_score("  55  "), Some(55.0));
    }

    #[test]
    fn parse_out_of_range_returns_none() {
        assert_eq!(parse_judge_score("Score: 150"), None);
        assert_eq!(parse_judge_score("-5"), None);
        assert_eq!(parse_judge_score("101"), None);
    }

    #[test]
    fn parse_garbage_returns_none() {
        assert_eq!(parse_judge_score("looks good to me"), None);
        assert_eq!(parse_judge_score(""), None);
    }

    #[test]
    fn parse_zero_and_hundred() {
        assert_eq!(parse_judge_score("0"), Some(0.0));
        assert_eq!(parse_judge_score("100"), Some(100.0));
        assert_eq!(parse_judge_score("Score: 0"), Some(0.0));
        assert_eq!(parse_judge_score("100/100"), Some(100.0));
    }

    // ── score blending ──────────────────────────────────────────────

    #[test]
    fn blend_default_ratio() {
        // 80% of 100 + 20% of 50 = 90
        let blended = blend_scores(100.0, 50.0, DEFAULT_JUDGE_RATIO);
        assert!((blended - 90.0).abs() < f64::EPSILON);
    }

    #[test]
    fn blend_verifier_only() {
        let blended = blend_scores(75.0, 25.0, 0.0);
        assert!((blended - 75.0).abs() < f64::EPSILON);
    }

    #[test]
    fn blend_judge_only() {
        let blended = blend_scores(75.0, 25.0, 1.0);
        assert!((blended - 25.0).abs() < f64::EPSILON);
    }

    #[test]
    fn blend_clamps_ratio() {
        // ratio > 1.0 clamped to 1.0
        let blended = blend_scores(80.0, 60.0, 5.0);
        assert!((blended - 60.0).abs() < f64::EPSILON);
        // ratio < 0.0 clamped to 0.0
        let blended = blend_scores(80.0, 60.0, -1.0);
        assert!((blended - 80.0).abs() < f64::EPSILON);
    }

    #[test]
    fn blend_clamps_output() {
        // Even with extreme inputs, output stays in [0, 100]
        let blended = blend_scores(200.0, 200.0, 0.5);
        assert!((blended - 100.0).abs() < f64::EPSILON);
    }
}
