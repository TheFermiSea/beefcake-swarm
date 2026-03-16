//! Validation quality gates: cloud and local reviewer execution + feedback extraction.
//!
//! Runs external LLM models as code reviewers on git diffs, parsing their
//! structured JSON feedback into actionable `ValidatorFeedback` entries
//! (the TextGrad pattern).

use std::path::Path;

use rig::providers::openai;
use tracing::{info, warn};

use crate::agents::reviewer;
use coordination::{ValidatorFeedback, ValidatorIssueType};

use super::helpers::timeout_from_env;

/// Default timeout for each cloud validation call.
const DEFAULT_VALIDATION_TIMEOUT_SECS: u64 = 120; // 2 minutes

/// Result of a single cloud model validation.
pub(crate) struct CloudValidationResult {
    pub(crate) model: String,
    pub(crate) passed: bool,
    pub(crate) feedback: String,
}

/// Result of a local validator review (blocking gate).
pub(crate) struct LocalValidationResult {
    pub(crate) model: String,
    pub(crate) passed: bool,
    #[allow(dead_code)] // kept for diagnostics/future logging
    pub(crate) schema_valid: bool,
    pub(crate) feedback: String,
    pub(crate) blocking_issues: Vec<String>,
    pub(crate) suggested_next_action: String,
    pub(crate) touched_files: Vec<String>,
}

/// Classify a blocking issue description into a `ValidatorIssueType`.
fn classify_issue(description: &str) -> ValidatorIssueType {
    let lower = description.to_lowercase();
    if lower.contains("logic") || lower.contains("incorrect") || lower.contains("wrong") {
        ValidatorIssueType::LogicError
    } else if lower.contains("safety")
        || lower.contains("error handling")
        || lower.contains("unwrap")
        || lower.contains("panic")
    {
        ValidatorIssueType::MissingSafetyCheck
    } else if lower.contains("edge case")
        || lower.contains("boundary")
        || lower.contains("overflow")
        || lower.contains("empty")
    {
        ValidatorIssueType::UnhandledEdgeCase
    } else if lower.contains("style") || lower.contains("naming") || lower.contains("format") {
        ValidatorIssueType::StyleViolation
    } else if lower.contains("behavior")
        || lower.contains("specification")
        || lower.contains("spec")
    {
        ValidatorIssueType::IncorrectBehavior
    } else {
        ValidatorIssueType::Other
    }
}

/// Convert a cloud validation result into structured validator feedback entries.
///
/// Parses the reviewer's JSON response to extract blocking_issues and
/// touched_files, converting prose feedback into actionable deltas (TextGrad pattern).
pub(crate) fn extract_validator_feedback(result: &CloudValidationResult) -> Vec<ValidatorFeedback> {
    if result.passed {
        return vec![];
    }

    let review = reviewer::ReviewResult::parse(&result.feedback);

    if review.blocking_issues.is_empty() {
        // Unstructured feedback — wrap as a single entry
        return vec![ValidatorFeedback {
            file: None,
            line_range: None,
            issue_type: ValidatorIssueType::Other,
            description: result
                .feedback
                .lines()
                .take(5)
                .collect::<Vec<_>>()
                .join(" "),
            suggested_fix: None,
            source_model: Some(result.model.clone()),
        }];
    }

    review
        .blocking_issues
        .iter()
        .map(|issue| {
            // Try to classify the issue type from keywords
            let issue_type = classify_issue(issue);

            // Try to extract file reference from touched_files
            let file = review.touched_files.first().cloned();

            ValidatorFeedback {
                file,
                line_range: None,
                issue_type,
                description: issue.clone(),
                suggested_fix: if review.suggested_next_action.is_empty() {
                    None
                } else {
                    Some(review.suggested_next_action.clone())
                },
                source_model: Some(result.model.clone()),
            }
        })
        .collect()
}

/// Convert a local validation result into structured validator feedback entries.
///
/// Similar to `extract_validator_feedback` but operates on `LocalValidationResult`.
pub(crate) fn extract_local_validator_feedback(
    result: &LocalValidationResult,
) -> Vec<ValidatorFeedback> {
    if result.passed {
        return vec![];
    }

    if result.blocking_issues.is_empty() {
        // Unstructured feedback — wrap as a single entry
        return vec![ValidatorFeedback {
            file: None,
            line_range: None,
            issue_type: ValidatorIssueType::Other,
            description: result
                .feedback
                .lines()
                .take(5)
                .collect::<Vec<_>>()
                .join(" "),
            suggested_fix: None,
            source_model: Some(result.model.clone()),
        }];
    }

    result
        .blocking_issues
        .iter()
        .map(|issue| {
            let issue_type = classify_issue(issue);
            let file = result.touched_files.first().cloned();

            ValidatorFeedback {
                file,
                line_range: None,
                issue_type,
                description: issue.clone(),
                suggested_fix: if result.suggested_next_action.is_empty() {
                    None
                } else {
                    Some(result.suggested_next_action.clone())
                },
                source_model: Some(result.model.clone()),
            }
        })
        .collect()
}

/// Build the reviewer prompt for a given diff.
///
/// Shared between cloud and local validation to prevent prompt drift.
fn build_reviewer_prompt(diff_for_review: &str) -> String {
    format!(
        "You are reviewing a Rust code change from an autonomous coding agent. \
         The change has already passed all deterministic gates (cargo fmt, clippy, \
         cargo check, cargo test). Your job is to catch logic errors, edge cases, \
         and design issues that the compiler cannot detect.\n\n\
         Respond with STRICT JSON ONLY using schema: \
         {{\"verdict\":\"pass|fail|needs_escalation\",\"confidence\":0.0-1.0,\
         \"blocking_issues\":[...],\"suggested_next_action\":\"...\",\
         \"touched_files\":[...]}}.\n\n\
         ```diff\n{diff_for_review}\n```"
    )
}

/// Run cloud validation on the worktree diff using external high-end models.
///
/// Sends the git diff (since initial commit) to each configured cloud validator
/// model for blind PASS/FAIL review. This is **advisory** — the orchestrator
/// logs results but doesn't block on FAIL to avoid subjective LLM feedback loops.
///
/// Validator models are configured via env vars:
/// - `SWARM_VALIDATOR_MODEL_1` (default: `gemini-3-pro-preview`)
/// - `SWARM_VALIDATOR_MODEL_2` (default: `claude-sonnet-4-5-20250929`)
pub(crate) async fn cloud_validate(
    cloud_client: &openai::CompletionsClient,
    wt_path: &Path,
    initial_commit: &str,
) -> Vec<CloudValidationResult> {
    // Get the full diff since the initial commit
    let diff = match std::process::Command::new("git")
        .args(["diff", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).to_string()
        }
        Ok(output) => {
            warn!(
                "git diff failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Vec::new();
        }
        Err(e) => {
            warn!("Failed to run git diff: {e}");
            return Vec::new();
        }
    };

    if diff.trim().is_empty() {
        info!("No diff to validate — skipping cloud validation");
        return Vec::new();
    }

    // Truncate very large diffs to avoid token limits
    let max_diff_chars = 32_000;
    let diff_for_review = if diff.len() > max_diff_chars {
        format!(
            "{}\n\n... [truncated — {} total chars, showing first {}]",
            &diff[..max_diff_chars],
            diff.len(),
            max_diff_chars,
        )
    } else {
        diff
    };

    let models = [
        std::env::var("SWARM_VALIDATOR_MODEL_1")
            .unwrap_or_else(|_| "gemini-3.1-pro-preview".into()),
        std::env::var("SWARM_VALIDATOR_MODEL_2").unwrap_or_else(|_| "claude-sonnet-4-6".into()),
    ];

    let review_prompt = build_reviewer_prompt(&diff_for_review);
    let validation_timeout = timeout_from_env(
        "SWARM_VALIDATION_TIMEOUT_SECS",
        DEFAULT_VALIDATION_TIMEOUT_SECS,
    );

    let mut results = Vec::new();
    for model in &models {
        info!(model, "Running cloud validation");
        let validator = reviewer::build_reviewer(cloud_client, model);
        match tokio::time::timeout(
            validation_timeout,
            super::prompt_with_retry(&validator, &review_prompt, 3),
        )
        .await
        {
            Ok(Ok(response)) => {
                let review = reviewer::ReviewResult::parse(&response);
                if !review.schema_valid {
                    warn!(
                        model,
                        "Cloud validation response was invalid schema; treating as FAIL"
                    );
                }
                let status = if review.passed { "PASS" } else { "FAIL" };
                info!(model, status, "Cloud validation complete");
                results.push(CloudValidationResult {
                    model: model.clone(),
                    passed: review.passed,
                    feedback: review.feedback,
                });
            }
            Ok(Err(e)) => {
                warn!(model, "Cloud validation error: {e}");
            }
            Err(_) => {
                warn!(
                    model,
                    "Cloud validation timed out ({}s)",
                    validation_timeout.as_secs()
                );
            }
        }
    }

    results
}

/// Run local validation via the reviewer agent (vasp-02/HydraCoder).
///
/// Generates a diff, sends it to the reviewer, and parses the structured JSON response.
/// - **Fail-open** on infrastructure errors (diff failure, timeout, LLM error) — deterministic gates already passed.
/// - **Fail-closed** on invalid JSON schema — malformed reviewer output counts as failure.
pub(crate) async fn local_validate(
    reviewer: &crate::agents::coder::OaiAgent,
    wt_path: &Path,
    initial_commit: &str,
    model_name: &str,
) -> LocalValidationResult {
    let validation_timeout = timeout_from_env("SWARM_LOCAL_VALIDATION_TIMEOUT_SECS", 60);

    // Generate diff (async to avoid blocking the tokio runtime)
    let diff = match tokio::process::Command::new("git")
        .args(["diff", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .await
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).to_string()
        }
        Ok(output) => {
            warn!(
                "local_validate: git diff failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            // Fail-open: infra error
            return LocalValidationResult {
                model: model_name.to_string(),
                passed: true,
                schema_valid: true,
                feedback: "git diff failed — fail-open".to_string(),
                blocking_issues: vec![],
                suggested_next_action: String::new(),
                touched_files: vec![],
            };
        }
        Err(e) => {
            warn!("local_validate: Failed to run git diff: {e}");
            return LocalValidationResult {
                model: model_name.to_string(),
                passed: true,
                schema_valid: true,
                feedback: format!("git diff error — fail-open: {e}"),
                blocking_issues: vec![],
                suggested_next_action: String::new(),
                touched_files: vec![],
            };
        }
    };

    if diff.trim().is_empty() {
        info!("local_validate: No diff to validate — pass");
        return LocalValidationResult {
            model: model_name.to_string(),
            passed: true,
            schema_valid: true,
            feedback: "No diff".to_string(),
            blocking_issues: vec![],
            suggested_next_action: String::new(),
            touched_files: vec![],
        };
    }

    // Truncate large diffs (on a valid char boundary to avoid panics)
    let max_diff_bytes = 32_000;
    let diff_for_review = if diff.len() > max_diff_bytes {
        let boundary = diff.floor_char_boundary(max_diff_bytes);
        format!(
            "{}\n\n... [truncated — {} total bytes, showing first {}]",
            &diff[..boundary],
            diff.len(),
            boundary,
        )
    } else {
        diff
    };

    let review_prompt = build_reviewer_prompt(&diff_for_review);

    // Call reviewer with timeout and retry
    match tokio::time::timeout(
        validation_timeout,
        super::prompt_with_retry(reviewer, &review_prompt, 2),
    )
    .await
    {
        Ok(Ok(response)) => {
            let review = reviewer::ReviewResult::parse(&response);
            if !review.schema_valid {
                // Fail-closed: bad JSON schema
                warn!(
                    model = model_name,
                    "Local validation: invalid schema — fail-closed"
                );
                return LocalValidationResult {
                    model: model_name.to_string(),
                    passed: false,
                    schema_valid: false,
                    feedback: response,
                    blocking_issues: review.blocking_issues,
                    suggested_next_action: review.suggested_next_action,
                    touched_files: review.touched_files,
                };
            }
            let status = if review.passed { "PASS" } else { "FAIL" };
            info!(model = model_name, status, "Local validation complete");
            LocalValidationResult {
                model: model_name.to_string(),
                passed: review.passed,
                schema_valid: true,
                feedback: response,
                blocking_issues: review.blocking_issues,
                suggested_next_action: review.suggested_next_action,
                touched_files: review.touched_files,
            }
        }
        Ok(Err(e)) => {
            // Fail-open: LLM error
            warn!(model = model_name, error = %e, "Local validation LLM error — fail-open");
            LocalValidationResult {
                model: model_name.to_string(),
                passed: true,
                schema_valid: true,
                feedback: format!("LLM error — fail-open: {e}"),
                blocking_issues: vec![],
                suggested_next_action: String::new(),
                touched_files: vec![],
            }
        }
        Err(_) => {
            // Fail-open: timeout
            warn!(
                model = model_name,
                timeout_secs = validation_timeout.as_secs(),
                "Local validation timed out — fail-open"
            );
            LocalValidationResult {
                model: model_name.to_string(),
                passed: true,
                schema_valid: true,
                feedback: "Timed out — fail-open".to_string(),
                blocking_issues: vec![],
                suggested_next_action: String::new(),
                touched_files: vec![],
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_validator_feedback_pass_returns_empty() {
        let result = CloudValidationResult {
            model: "test-model".into(),
            passed: true,
            feedback: r#"{"verdict":"pass","confidence":0.95,"blocking_issues":[],"suggested_next_action":"merge","touched_files":["src/lib.rs"]}"#.into(),
        };
        assert!(extract_validator_feedback(&result).is_empty());
    }

    #[test]
    fn test_extract_validator_feedback_fail_with_blocking_issues() {
        let result = CloudValidationResult {
            model: "gemini-3-pro".into(),
            passed: false,
            feedback: r#"{"verdict":"fail","confidence":0.7,"blocking_issues":["missing error handling for edge case","logic error in loop termination"],"suggested_next_action":"add bounds checking","touched_files":["src/main.rs"]}"#.into(),
        };
        let feedback = extract_validator_feedback(&result);
        assert_eq!(feedback.len(), 2);
        // "missing error handling" matches safety check before "edge case"
        assert_eq!(
            feedback[0].issue_type,
            ValidatorIssueType::MissingSafetyCheck
        );
        assert_eq!(feedback[1].issue_type, ValidatorIssueType::LogicError);
        assert_eq!(feedback[0].source_model.as_deref(), Some("gemini-3-pro"));
        assert!(feedback[0].suggested_fix.is_some());
    }

    #[test]
    fn test_extract_validator_feedback_malformed_falls_back() {
        let result = CloudValidationResult {
            model: "test-model".into(),
            passed: false,
            feedback: "FAIL\nThis code has issues".into(),
        };
        let feedback = extract_validator_feedback(&result);
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].issue_type, ValidatorIssueType::Other);
    }

    #[test]
    fn test_classify_issue_keywords() {
        assert_eq!(
            classify_issue("missing error handling for None case"),
            ValidatorIssueType::MissingSafetyCheck
        );
        assert_eq!(
            classify_issue("logic error in loop"),
            ValidatorIssueType::LogicError
        );
        assert_eq!(
            classify_issue("edge case when input is empty"),
            ValidatorIssueType::UnhandledEdgeCase
        );
        assert_eq!(
            classify_issue("naming convention violated"),
            ValidatorIssueType::StyleViolation
        );
        assert_eq!(
            classify_issue("behavior differs from specification"),
            ValidatorIssueType::IncorrectBehavior
        );
        assert_eq!(
            classify_issue("something else entirely"),
            ValidatorIssueType::Other
        );
    }

    #[test]
    fn test_extract_local_validator_feedback_pass_returns_empty() {
        let result = LocalValidationResult {
            model: "test-model".into(),
            passed: true,
            schema_valid: true,
            feedback: "looks good".into(),
            blocking_issues: vec![],
            suggested_next_action: String::new(),
            touched_files: vec![],
        };
        let feedback = extract_local_validator_feedback(&result);
        assert!(feedback.is_empty());
    }

    #[test]
    fn test_extract_local_validator_feedback_fail_with_issues() {
        let result = LocalValidationResult {
            model: "HydraCoder".into(),
            passed: false,
            schema_valid: true,
            feedback: "structured review".into(),
            blocking_issues: vec![
                "missing error handling in parse_config".into(),
                "logic error in boundary check".into(),
            ],
            suggested_next_action: "fix and re-run".into(),
            touched_files: vec!["src/config.rs".into()],
        };
        let feedback = extract_local_validator_feedback(&result);
        assert_eq!(feedback.len(), 2);
        assert_eq!(
            feedback[0].issue_type,
            ValidatorIssueType::MissingSafetyCheck
        );
        assert_eq!(feedback[1].issue_type, ValidatorIssueType::LogicError);
        assert_eq!(feedback[0].file.as_deref(), Some("src/config.rs"));
        assert_eq!(feedback[0].suggested_fix.as_deref(), Some("fix and re-run"));
        assert_eq!(feedback[0].source_model.as_deref(), Some("HydraCoder"));
    }

    #[test]
    fn test_extract_local_validator_feedback_malformed_wraps_as_single() {
        let result = LocalValidationResult {
            model: "test-model".into(),
            passed: false,
            schema_valid: false,
            feedback: "PASS\nlooks okay\nbut it was malformed".into(),
            blocking_issues: vec![],
            suggested_next_action: String::new(),
            touched_files: vec![],
        };
        let feedback = extract_local_validator_feedback(&result);
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].issue_type, ValidatorIssueType::Other);
        assert!(feedback[0].description.contains("PASS"));
        assert!(feedback[0].description.contains("looks okay"));
    }
}
