//! Structured specialist response contracts and validation.
//!
//! Every specialist (planner, fixer, coder, reviewer) produces a response that
//! the orchestrator must parse into a typed contract before consuming. Malformed
//! responses are rejected (fail-closed) and trigger escalation or retry.
//!
//! ## Contract schema
//!
//! ```text
//! SpecialistResponse {
//!     objective_status: Success | Partial | Failed | Blocked,
//!     patch_plan:       Option<ImplementationPlan>,   // planner output
//!     risks:            Vec<Risk>,                    // identified risks
//!     required_followups: Vec<Followup>,              // what still needs work
//!     files_changed:    Vec<String>,                  // files modified
//!     raw_response:     String,                       // original text
//!     schema_valid:     bool,                         // false → fail-closed
//! }
//! ```

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::agents::reviewer::ReviewResult;
use crate::pipeline::ImplementationPlan;

/// Status of the specialist's objective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ObjectiveStatus {
    /// Specialist completed its objective fully.
    Success,
    /// Partially complete — some work done but followups needed.
    Partial,
    /// Could not complete — needs escalation or different approach.
    Failed,
    /// Blocked on an external dependency or unresolvable issue.
    Blocked,
}

impl std::fmt::Display for ObjectiveStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Success => write!(f, "success"),
            Self::Partial => write!(f, "partial"),
            Self::Failed => write!(f, "failed"),
            Self::Blocked => write!(f, "blocked"),
        }
    }
}

/// A risk identified by a specialist.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Risk {
    pub severity: RiskSeverity,
    pub description: String,
    pub file: Option<String>,
}

/// Risk severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
}

/// A required followup action.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Followup {
    pub action: String,
    pub target: FollowupTarget,
}

/// Which specialist should handle a followup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum FollowupTarget {
    Planner,
    Fixer,
    Coder,
    Reviewer,
    Verifier,
    Human,
}

impl std::fmt::Display for FollowupTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Planner => write!(f, "planner"),
            Self::Fixer => write!(f, "fixer"),
            Self::Coder => write!(f, "coder"),
            Self::Reviewer => write!(f, "reviewer"),
            Self::Verifier => write!(f, "verifier"),
            Self::Human => write!(f, "human"),
        }
    }
}

/// Unified typed output from any specialist.
///
/// The orchestrator MUST parse raw responses into this struct before making
/// routing decisions. Invalid/unparseable responses produce `schema_valid: false`
/// and `objective_status: Failed` (fail-closed).
#[derive(Debug, Clone)]
pub struct SpecialistResponse {
    /// How the specialist fared on its objective.
    pub objective_status: ObjectiveStatus,
    /// Structured plan (only present for planner responses).
    pub patch_plan: Option<ImplementationPlan>,
    /// Risks identified by the specialist.
    pub risks: Vec<Risk>,
    /// What still needs to be done.
    pub required_followups: Vec<Followup>,
    /// Files mentioned as changed or relevant.
    pub files_changed: Vec<String>,
    /// The raw response text.
    pub raw_response: String,
    /// Whether the response conformed to the expected schema.
    pub schema_valid: bool,
}

impl SpecialistResponse {
    /// Create a fail-closed response for unparseable output.
    pub fn fail_closed(raw: String) -> Self {
        Self {
            objective_status: ObjectiveStatus::Failed,
            patch_plan: None,
            risks: vec![Risk {
                severity: RiskSeverity::High,
                description: "Specialist response failed schema validation".into(),
                file: None,
            }],
            required_followups: vec![Followup {
                action: "Retry with clearer instructions or escalate".into(),
                target: FollowupTarget::Human,
            }],
            files_changed: Vec::new(),
            raw_response: raw,
            schema_valid: false,
        }
    }

    /// Check if this response indicates the objective was fully achieved.
    pub fn is_success(&self) -> bool {
        self.objective_status == ObjectiveStatus::Success && self.schema_valid
    }

    /// Check if this response needs followup work.
    pub fn needs_followup(&self) -> bool {
        !self.required_followups.is_empty()
    }

    /// Check if this response indicates the specialist is blocked.
    pub fn is_blocked(&self) -> bool {
        self.objective_status == ObjectiveStatus::Blocked
    }
}

// ---------------------------------------------------------------------------
// Parsers — one per specialist type
// ---------------------------------------------------------------------------

/// Parse a planner response into a typed contract.
///
/// The planner is expected to return strict JSON matching `ImplementationPlan`.
/// On parse failure: fail-closed (schema_valid=false, status=Failed).
pub fn parse_planner_response(raw: &str) -> SpecialistResponse {
    // Try to extract JSON from response (may have surrounding text)
    let json_str = extract_json_block(raw).unwrap_or(raw);

    match serde_json::from_str::<ImplementationPlan>(json_str) {
        Ok(plan) => {
            // Validate plan bounds
            if plan.steps.is_empty() {
                return SpecialistResponse {
                    objective_status: ObjectiveStatus::Failed,
                    patch_plan: None,
                    risks: vec![Risk {
                        severity: RiskSeverity::Medium,
                        description: "Plan has no steps".into(),
                        file: None,
                    }],
                    required_followups: vec![Followup {
                        action: "Re-plan with specific steps".into(),
                        target: FollowupTarget::Planner,
                    }],
                    files_changed: Vec::new(),
                    raw_response: raw.to_string(),
                    schema_valid: false,
                };
            }

            let risks = plan_risks(&plan);
            let files = plan.target_files.clone();

            SpecialistResponse {
                objective_status: ObjectiveStatus::Success,
                patch_plan: Some(plan),
                risks,
                required_followups: vec![Followup {
                    action: "Implement the plan".into(),
                    target: FollowupTarget::Fixer,
                }],
                files_changed: files,
                raw_response: raw.to_string(),
                schema_valid: true,
            }
        }
        Err(_) => SpecialistResponse::fail_closed(raw.to_string()),
    }
}

/// Parse a fixer or coder response into a typed contract.
///
/// Fixer/coder agents return freeform text describing what they did.
/// We extract status from known patterns (BLOCKED, DISCOVERED, etc.).
pub fn parse_coder_response(raw: &str) -> SpecialistResponse {
    let status = infer_status_from_text(raw);
    let files = extract_file_paths(raw);
    let followups = infer_followups_from_text(raw);
    let risks = infer_risks_from_text(raw);

    SpecialistResponse {
        objective_status: status,
        patch_plan: None,
        risks,
        required_followups: followups,
        files_changed: files,
        raw_response: raw.to_string(),
        // Coder responses are freeform — always considered valid if non-empty
        schema_valid: !raw.trim().is_empty(),
    }
}

/// Parse a reviewer response into a typed contract.
///
/// Delegates to the existing `ReviewResult::parse()` fail-closed parser,
/// then wraps into the unified contract.
pub fn parse_reviewer_response(raw: &str) -> SpecialistResponse {
    let review = ReviewResult::parse(raw);

    let status = if review.passed {
        ObjectiveStatus::Success
    } else if !review.schema_valid {
        ObjectiveStatus::Failed
    } else {
        ObjectiveStatus::Partial
    };

    let risks: Vec<Risk> = review
        .blocking_issues
        .iter()
        .map(|issue| Risk {
            severity: RiskSeverity::High,
            description: issue.clone(),
            file: None,
        })
        .collect();

    let followups = if review.passed {
        vec![Followup {
            action: "Proceed to merge".into(),
            target: FollowupTarget::Verifier,
        }]
    } else {
        vec![Followup {
            action: review.suggested_next_action.clone(),
            target: if review.schema_valid {
                FollowupTarget::Fixer
            } else {
                FollowupTarget::Reviewer
            },
        }]
    };

    SpecialistResponse {
        objective_status: status,
        patch_plan: None,
        risks,
        required_followups: followups,
        files_changed: review.touched_files,
        raw_response: raw.to_string(),
        schema_valid: review.schema_valid,
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Validate that a specialist response meets minimum contract requirements.
///
/// Fail-closed: returns `false` for any contract violation.
pub fn validate_response(response: &SpecialistResponse) -> Vec<String> {
    let mut violations = Vec::new();

    if !response.schema_valid {
        violations.push("Response failed schema validation".into());
    }

    if response.raw_response.trim().is_empty() {
        violations.push("Empty raw response".into());
    }

    // Success requires no high-severity risks
    if response.objective_status == ObjectiveStatus::Success {
        let has_high_risk = response
            .risks
            .iter()
            .any(|r| r.severity == RiskSeverity::High);
        if has_high_risk {
            violations.push("Success status with high-severity risk".into());
        }
    }

    // Failed/Blocked should have followups
    if matches!(
        response.objective_status,
        ObjectiveStatus::Failed | ObjectiveStatus::Blocked
    ) && response.required_followups.is_empty()
    {
        violations.push("Failed/Blocked status without followup actions".into());
    }

    violations
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Try to extract a JSON block from a response that may contain surrounding text.
fn extract_json_block(text: &str) -> Option<&str> {
    // Look for ```json ... ``` fenced blocks
    if let Some(start) = text.find("```json") {
        let json_start = start + 7;
        if let Some(end) = text[json_start..].find("```") {
            return Some(text[json_start..json_start + end].trim());
        }
    }

    // Look for first { to last }
    let start = text.find('{')?;
    let end = text.rfind('}')?;
    if end > start {
        Some(&text[start..=end])
    } else {
        None
    }
}

/// Convert plan risk level to contract risks.
fn plan_risks(plan: &ImplementationPlan) -> Vec<Risk> {
    use crate::pipeline::PlanRisk;
    match plan.risk {
        PlanRisk::High => vec![Risk {
            severity: RiskSeverity::High,
            description: "Plan assessed as high risk".into(),
            file: None,
        }],
        PlanRisk::Medium => vec![Risk {
            severity: RiskSeverity::Medium,
            description: "Plan assessed as medium risk".into(),
            file: None,
        }],
        PlanRisk::Low => vec![],
    }
}

/// Infer objective status from freeform coder response text.
fn infer_status_from_text(text: &str) -> ObjectiveStatus {
    let upper = text.to_uppercase();

    if upper.contains("BLOCKED") {
        return ObjectiveStatus::Blocked;
    }

    // Check for explicit failure indicators
    if upper.contains("COULD NOT") || upper.contains("UNABLE TO") || upper.contains("FAILED TO") {
        return ObjectiveStatus::Failed;
    }

    // Check for partial completion indicators
    if upper.contains("PARTIAL") || upper.contains("DISCOVERED:") || upper.contains("TODO:") {
        return ObjectiveStatus::Partial;
    }

    // Default: if the response is non-empty, assume success (coder made edits)
    if !text.trim().is_empty() {
        ObjectiveStatus::Success
    } else {
        ObjectiveStatus::Failed
    }
}

/// Extract file paths mentioned in coder response text.
fn extract_file_paths(text: &str) -> Vec<String> {
    let mut files = Vec::new();
    for word in text.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| {
            matches!(c, '`' | '"' | '\'' | ',' | '.' | '(' | ')' | '[' | ']')
        });
        if looks_like_file_path(trimmed) && !files.contains(&trimmed.to_string()) {
            files.push(trimmed.to_string());
        }
    }
    files
}

fn looks_like_file_path(s: &str) -> bool {
    // Must contain a dot and a slash, or end with .rs/.toml/.lock
    if s.len() < 4 {
        return false;
    }
    let has_extension = s.ends_with(".rs")
        || s.ends_with(".toml")
        || s.ends_with(".md")
        || s.ends_with(".json")
        || s.ends_with(".lock");
    let has_path_sep = s.contains('/');
    has_extension && has_path_sep
}

/// Infer followup actions from coder response patterns.
fn infer_followups_from_text(text: &str) -> Vec<Followup> {
    let mut followups = Vec::new();
    let upper = text.to_uppercase();

    if upper.contains("DISCOVERED:") {
        followups.push(Followup {
            action: "Track discovered issue".into(),
            target: FollowupTarget::Planner,
        });
    }

    if upper.contains("BLOCKED") {
        followups.push(Followup {
            action: "Resolve blocker".into(),
            target: FollowupTarget::Human,
        });
    }

    if upper.contains("TODO:") {
        followups.push(Followup {
            action: "Complete remaining TODO items".into(),
            target: FollowupTarget::Fixer,
        });
    }

    followups
}

/// Infer risks from coder response text.
fn infer_risks_from_text(text: &str) -> Vec<Risk> {
    let mut risks = Vec::new();
    let upper = text.to_uppercase();

    if upper.contains("UNSAFE") {
        risks.push(Risk {
            severity: RiskSeverity::High,
            description: "Response mentions unsafe code".into(),
            file: None,
        });
    }

    if upper.contains("BREAKING CHANGE") || upper.contains("API CHANGE") {
        risks.push(Risk {
            severity: RiskSeverity::High,
            description: "Response mentions breaking/API change".into(),
            file: None,
        });
    }

    if upper.contains("WORKAROUND") || upper.contains("HACK") {
        risks.push(Risk {
            severity: RiskSeverity::Medium,
            description: "Response uses workaround/hack".into(),
            file: None,
        });
    }

    risks
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ObjectiveStatus --

    #[test]
    fn test_objective_status_display() {
        assert_eq!(ObjectiveStatus::Success.to_string(), "success");
        assert_eq!(ObjectiveStatus::Blocked.to_string(), "blocked");
    }

    #[test]
    fn test_objective_status_serde_roundtrip() {
        for status in [
            ObjectiveStatus::Success,
            ObjectiveStatus::Partial,
            ObjectiveStatus::Failed,
            ObjectiveStatus::Blocked,
        ] {
            let json = serde_json::to_string(&status).unwrap();
            let restored: ObjectiveStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(restored, status);
        }
    }

    // -- SpecialistResponse --

    #[test]
    fn test_fail_closed_response() {
        let resp = SpecialistResponse::fail_closed("garbage output".into());
        assert_eq!(resp.objective_status, ObjectiveStatus::Failed);
        assert!(!resp.schema_valid);
        assert!(!resp.is_success());
        assert!(resp.needs_followup());
        assert!(!resp.is_blocked());
    }

    #[test]
    fn test_specialist_response_is_success() {
        let resp = SpecialistResponse {
            objective_status: ObjectiveStatus::Success,
            patch_plan: None,
            risks: vec![],
            required_followups: vec![],
            files_changed: vec![],
            raw_response: "done".into(),
            schema_valid: true,
        };
        assert!(resp.is_success());
        assert!(!resp.needs_followup());
    }

    #[test]
    fn test_specialist_response_invalid_schema_not_success() {
        let resp = SpecialistResponse {
            objective_status: ObjectiveStatus::Success,
            patch_plan: None,
            risks: vec![],
            required_followups: vec![],
            files_changed: vec![],
            raw_response: "done".into(),
            schema_valid: false, // invalid!
        };
        assert!(!resp.is_success());
    }

    // -- Planner parsing --

    #[test]
    fn test_parse_planner_valid_json() {
        let raw = r#"{
            "approach": "Fix the borrow checker error by adding a clone",
            "steps": [{"description": "Clone the value in handler()", "file": "src/lib.rs"}],
            "target_files": ["src/lib.rs"],
            "risk": "low"
        }"#;
        let resp = parse_planner_response(raw);
        assert!(resp.schema_valid);
        assert_eq!(resp.objective_status, ObjectiveStatus::Success);
        assert!(resp.patch_plan.is_some());
        assert_eq!(resp.files_changed, vec!["src/lib.rs"]);
    }

    #[test]
    fn test_parse_planner_with_fenced_json() {
        let raw = "Here is the plan:\n```json\n{\n\"approach\": \"Add lifetime\",\n\"steps\": [{\"description\": \"Add 'a\", \"file\": \"src/lib.rs\"}],\n\"target_files\": [\"src/lib.rs\"],\n\"risk\": \"low\"\n}\n```\nLet me know.";
        let resp = parse_planner_response(raw);
        assert!(resp.schema_valid);
        assert!(resp.patch_plan.is_some());
    }

    #[test]
    fn test_parse_planner_empty_steps_rejected() {
        let raw = r#"{
            "approach": "Do nothing",
            "steps": [],
            "target_files": [],
            "risk": "low"
        }"#;
        let resp = parse_planner_response(raw);
        assert!(!resp.schema_valid);
        assert_eq!(resp.objective_status, ObjectiveStatus::Failed);
    }

    #[test]
    fn test_parse_planner_malformed_json() {
        let resp = parse_planner_response("this is not json at all");
        assert!(!resp.schema_valid);
        assert_eq!(resp.objective_status, ObjectiveStatus::Failed);
    }

    #[test]
    fn test_parse_planner_high_risk_plan() {
        let raw = r#"{
            "approach": "Restructure the module",
            "steps": [{"description": "Move types", "file": "src/lib.rs"}],
            "target_files": ["src/lib.rs"],
            "risk": "high"
        }"#;
        let resp = parse_planner_response(raw);
        assert!(resp.schema_valid);
        assert!(resp.risks.iter().any(|r| r.severity == RiskSeverity::High));
    }

    // -- Coder parsing --

    #[test]
    fn test_parse_coder_success() {
        let raw = "I fixed the borrow checker error in `src/lib.rs` by adding .clone().\n\
                    The edit_file call updated the handler function.";
        let resp = parse_coder_response(raw);
        assert!(resp.schema_valid);
        assert_eq!(resp.objective_status, ObjectiveStatus::Success);
        assert!(resp.files_changed.contains(&"src/lib.rs".to_string()));
    }

    #[test]
    fn test_parse_coder_blocked() {
        let raw = "BLOCKED: The dependency crate does not export the required trait.";
        let resp = parse_coder_response(raw);
        assert!(resp.is_blocked());
        assert!(resp.needs_followup());
    }

    #[test]
    fn test_parse_coder_partial_with_discovered() {
        let raw = "Fixed the type error. DISCOVERED: There's also a missing test for edge cases.";
        let resp = parse_coder_response(raw);
        assert_eq!(resp.objective_status, ObjectiveStatus::Partial);
    }

    #[test]
    fn test_parse_coder_empty_response() {
        let resp = parse_coder_response("");
        assert!(!resp.schema_valid);
        assert_eq!(resp.objective_status, ObjectiveStatus::Failed);
    }

    #[test]
    fn test_parse_coder_unsafe_risk() {
        let raw = "Added an unsafe block to transmute the pointer.";
        let resp = parse_coder_response(raw);
        assert!(resp.risks.iter().any(|r| r.severity == RiskSeverity::High));
    }

    // -- Reviewer parsing --

    #[test]
    fn test_parse_reviewer_pass() {
        let raw = r#"{
            "verdict": "pass",
            "confidence": 0.95,
            "blocking_issues": [],
            "suggested_next_action": "merge",
            "touched_files": ["src/lib.rs"]
        }"#;
        let resp = parse_reviewer_response(raw);
        assert!(resp.schema_valid);
        assert_eq!(resp.objective_status, ObjectiveStatus::Success);
        assert!(resp.is_success());
    }

    #[test]
    fn test_parse_reviewer_fail() {
        let raw = r#"{
            "verdict": "fail",
            "confidence": 0.8,
            "blocking_issues": ["Missing error handling on line 42"],
            "suggested_next_action": "Add error handling",
            "touched_files": ["src/handler.rs"]
        }"#;
        let resp = parse_reviewer_response(raw);
        assert!(resp.schema_valid);
        assert_eq!(resp.objective_status, ObjectiveStatus::Partial);
        assert!(!resp.is_success());
        assert!(resp.risks.iter().any(|r| r.severity == RiskSeverity::High));
    }

    #[test]
    fn test_parse_reviewer_malformed() {
        let resp = parse_reviewer_response("PASS: looks good");
        assert!(!resp.schema_valid);
        assert_eq!(resp.objective_status, ObjectiveStatus::Failed);
    }

    // -- Validation --

    #[test]
    fn test_validate_success_response() {
        let resp = SpecialistResponse {
            objective_status: ObjectiveStatus::Success,
            patch_plan: None,
            risks: vec![],
            required_followups: vec![],
            files_changed: vec![],
            raw_response: "done".into(),
            schema_valid: true,
        };
        let violations = validate_response(&resp);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_validate_invalid_schema() {
        let resp = SpecialistResponse::fail_closed("bad".into());
        let violations = validate_response(&resp);
        assert!(violations.iter().any(|v| v.contains("schema validation")));
    }

    #[test]
    fn test_validate_success_with_high_risk() {
        let resp = SpecialistResponse {
            objective_status: ObjectiveStatus::Success,
            patch_plan: None,
            risks: vec![Risk {
                severity: RiskSeverity::High,
                description: "Dangerous".into(),
                file: None,
            }],
            required_followups: vec![],
            files_changed: vec![],
            raw_response: "done".into(),
            schema_valid: true,
        };
        let violations = validate_response(&resp);
        assert!(violations.iter().any(|v| v.contains("high-severity risk")));
    }

    #[test]
    fn test_validate_failed_without_followups() {
        let resp = SpecialistResponse {
            objective_status: ObjectiveStatus::Failed,
            patch_plan: None,
            risks: vec![],
            required_followups: vec![], // no followups!
            files_changed: vec![],
            raw_response: "failed".into(),
            schema_valid: true,
        };
        let violations = validate_response(&resp);
        assert!(violations.iter().any(|v| v.contains("followup")));
    }

    #[test]
    fn test_validate_empty_response() {
        let resp = SpecialistResponse {
            objective_status: ObjectiveStatus::Success,
            patch_plan: None,
            risks: vec![],
            required_followups: vec![],
            files_changed: vec![],
            raw_response: "  ".into(),
            schema_valid: true,
        };
        let violations = validate_response(&resp);
        assert!(violations.iter().any(|v| v.contains("Empty")));
    }

    // -- Helpers --

    #[test]
    fn test_extract_json_block_fenced() {
        let text = "Here:\n```json\n{\"a\": 1}\n```\nDone.";
        assert_eq!(extract_json_block(text), Some("{\"a\": 1}"));
    }

    #[test]
    fn test_extract_json_block_bare() {
        let text = "Result: {\"a\": 1} end";
        assert_eq!(extract_json_block(text), Some("{\"a\": 1}"));
    }

    #[test]
    fn test_extract_json_block_none() {
        assert_eq!(extract_json_block("no json here"), None);
    }

    #[test]
    fn test_extract_file_paths() {
        let text = "Modified `src/lib.rs` and `crates/foo/src/bar.rs`.";
        let files = extract_file_paths(text);
        assert!(files.contains(&"src/lib.rs".to_string()));
        assert!(files.contains(&"crates/foo/src/bar.rs".to_string()));
    }

    #[test]
    fn test_extract_file_paths_no_matches() {
        let files = extract_file_paths("just some text without paths");
        assert!(files.is_empty());
    }

    #[test]
    fn test_infer_status_blocked() {
        assert_eq!(
            infer_status_from_text("BLOCKED: missing dep"),
            ObjectiveStatus::Blocked
        );
    }

    #[test]
    fn test_infer_status_failed() {
        assert_eq!(
            infer_status_from_text("Could not resolve the error"),
            ObjectiveStatus::Failed
        );
    }

    #[test]
    fn test_infer_status_partial() {
        assert_eq!(
            infer_status_from_text("Fixed part of it. TODO: handle edge case"),
            ObjectiveStatus::Partial
        );
    }

    #[test]
    fn test_infer_status_success() {
        assert_eq!(
            infer_status_from_text("Applied the fix successfully"),
            ObjectiveStatus::Success
        );
    }

    #[test]
    fn test_followup_target_display() {
        assert_eq!(FollowupTarget::Human.to_string(), "human");
        assert_eq!(FollowupTarget::Planner.to_string(), "planner");
    }
}
