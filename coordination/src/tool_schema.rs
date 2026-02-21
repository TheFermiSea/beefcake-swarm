//! Tool Input/Output Schemas — Strict Contracts for Reviewer Analysis Tools
//!
//! Defines typed schemas for all tool interactions in the reviewer pipeline.
//! Eliminates freeform blob contracts — every tool input and output has a
//! defined structure with validation.
//!
//! # Schemas
//!
//! ```text
//! AstGrepRequest/AstGrepResult     — AST pattern matching
//! DependencyCheckRequest/Result     — Import/dependency impact analysis
//! VerifierGateRequest/Result        — Quality gate pipeline
//! ReviewDecisionRequest/Result      — Final reviewer verdict
//! ```

use crate::feedback::error_parser::ErrorCategory;
use serde::{Deserialize, Serialize};

// ── AST Analysis ──────────────────────────────────────────────────────

/// Request for AST pattern analysis via ast-grep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstGrepRequest {
    /// Pattern to match (ast-grep pattern syntax).
    pub pattern: String,
    /// Language to parse (e.g., "rust", "typescript").
    pub language: String,
    /// Files or directories to search.
    pub paths: Vec<String>,
    /// Maximum results to return (prevents unbounded output).
    pub max_results: usize,
    /// Optional rule ID from sgconfig for anti-pattern checks.
    pub rule_id: Option<String>,
}

/// A single AST match from ast-grep.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstMatch {
    /// File path containing the match.
    pub file: String,
    /// Line number of the match start.
    pub line: usize,
    /// Column number of the match start.
    pub column: usize,
    /// The matched source text.
    pub matched_text: String,
    /// Surrounding context (a few lines before/after).
    pub context: String,
    /// Rule ID if matched against a named rule.
    pub rule_id: Option<String>,
}

/// Result of AST pattern analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstGrepResult {
    /// Whether the analysis completed successfully.
    pub success: bool,
    /// Matches found.
    pub matches: Vec<AstMatch>,
    /// Total matches (may exceed returned if truncated).
    pub total_matches: usize,
    /// Whether results were truncated by max_results.
    pub truncated: bool,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Error message if analysis failed.
    pub error: Option<String>,
}

impl AstGrepResult {
    /// Create a successful result.
    pub fn ok(matches: Vec<AstMatch>, total: usize, truncated: bool, duration_ms: u64) -> Self {
        Self {
            success: true,
            matches,
            total_matches: total,
            truncated,
            duration_ms,
            error: None,
        }
    }

    /// Create a failed result.
    pub fn err(error: &str, duration_ms: u64) -> Self {
        Self {
            success: false,
            matches: Vec::new(),
            total_matches: 0,
            truncated: false,
            duration_ms,
            error: Some(error.to_string()),
        }
    }
}

// ── Dependency Impact ─────────────────────────────────────────────────

/// Request for dependency impact analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyCheckRequest {
    /// Changed files to analyze.
    pub changed_files: Vec<String>,
    /// Package/crate to scope analysis to.
    pub package: Option<String>,
    /// Whether to include transitive dependents.
    pub include_transitive: bool,
}

/// Impact level of a dependency change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// No downstream impact (leaf node change).
    None,
    /// Limited impact (1-3 direct dependents).
    Low,
    /// Moderate impact (4-10 dependents or public API change).
    Medium,
    /// High impact (>10 dependents, trait change, or breaking API).
    High,
}

impl std::fmt::Display for ImpactLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
        }
    }
}

/// A file affected by the dependency change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AffectedFile {
    /// File path.
    pub file: String,
    /// How this file is affected.
    pub reason: String,
    /// Whether this is a direct or transitive dependent.
    pub direct: bool,
}

/// Result of dependency impact analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyCheckResult {
    /// Whether the analysis completed.
    pub success: bool,
    /// Overall impact level.
    pub impact_level: ImpactLevel,
    /// Files affected by the change.
    pub affected_files: Vec<AffectedFile>,
    /// Number of direct dependents.
    pub direct_dependents: usize,
    /// Number of transitive dependents.
    pub transitive_dependents: usize,
    /// Whether any public API signatures changed.
    pub api_change_detected: bool,
    /// Execution duration in milliseconds.
    pub duration_ms: u64,
    /// Error message if analysis failed.
    pub error: Option<String>,
}

impl DependencyCheckResult {
    /// Create a no-impact result.
    pub fn no_impact(duration_ms: u64) -> Self {
        Self {
            success: true,
            impact_level: ImpactLevel::None,
            affected_files: Vec::new(),
            direct_dependents: 0,
            transitive_dependents: 0,
            api_change_detected: false,
            duration_ms,
            error: None,
        }
    }
}

// ── Verifier Gate ─────────────────────────────────────────────────────

/// Request for the verifier quality gate pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierGateRequest {
    /// Workspace root path.
    pub workspace_root: String,
    /// Specific packages to check (empty = all).
    pub packages: Vec<String>,
    /// Gates to run (empty = all: fmt, clippy, check, test).
    pub gates: Vec<String>,
    /// Whether to stop on first failure.
    pub fail_fast: bool,
}

/// Result of a single quality gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCheckResult {
    /// Gate name (e.g., "fmt", "clippy", "check", "test").
    pub gate: String,
    /// Whether this gate passed.
    pub passed: bool,
    /// Number of errors.
    pub error_count: usize,
    /// Number of warnings.
    pub warning_count: usize,
    /// Dominant error category, if any.
    pub dominant_category: Option<ErrorCategory>,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

/// Result of the full verifier gate pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierGateResult {
    /// Whether all gates passed.
    pub all_passed: bool,
    /// Number of gates passed.
    pub gates_passed: usize,
    /// Total number of gates.
    pub gates_total: usize,
    /// Per-gate results.
    pub gates: Vec<GateCheckResult>,
    /// Total error count.
    pub total_errors: usize,
    /// First failing gate name.
    pub first_failure: Option<String>,
    /// Total pipeline duration in milliseconds.
    pub duration_ms: u64,
}

impl VerifierGateResult {
    /// All-green result.
    pub fn green(gates: Vec<GateCheckResult>, duration_ms: u64) -> Self {
        let gates_total = gates.len();
        Self {
            all_passed: true,
            gates_passed: gates_total,
            gates_total,
            gates,
            total_errors: 0,
            first_failure: None,
            duration_ms,
        }
    }
}

// ── Review Decision ───────────────────────────────────────────────────

/// Verdict from the reviewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReviewVerdict {
    /// Code is acceptable.
    Pass,
    /// Code has issues that must be fixed.
    Fail,
    /// Reviewer cannot decide — needs human or escalation.
    NeedsEscalation,
}

impl std::fmt::Display for ReviewVerdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pass => write!(f, "pass"),
            Self::Fail => write!(f, "fail"),
            Self::NeedsEscalation => write!(f, "needs_escalation"),
        }
    }
}

/// Input for the final review decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewDecisionRequest {
    /// Diff to review.
    pub diff: String,
    /// Verifier gate results.
    pub verifier_result: VerifierGateResult,
    /// AST analysis results (if available).
    pub ast_result: Option<AstGrepResult>,
    /// Dependency impact results (if available).
    pub dependency_result: Option<DependencyCheckResult>,
    /// Issue context (title, description).
    pub issue_context: String,
}

/// A specific issue found by the reviewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewIssue {
    /// Whether this issue blocks merging.
    pub blocking: bool,
    /// File path (if applicable).
    pub file: Option<String>,
    /// Line number (if applicable).
    pub line: Option<usize>,
    /// Description of the issue.
    pub description: String,
    /// Suggested fix.
    pub suggestion: Option<String>,
}

/// Structured review decision output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewDecisionResult {
    /// Overall verdict.
    pub verdict: ReviewVerdict,
    /// Confidence in the verdict (0.0–1.0).
    pub confidence: f64,
    /// Issues found during review.
    pub issues: Vec<ReviewIssue>,
    /// Number of blocking issues.
    pub blocking_count: usize,
    /// Suggested next action.
    pub next_action: String,
    /// Files that were touched in the diff.
    pub touched_files: Vec<String>,
}

impl ReviewDecisionResult {
    /// Quick pass result with high confidence.
    pub fn pass(confidence: f64, touched_files: Vec<String>) -> Self {
        Self {
            verdict: ReviewVerdict::Pass,
            confidence: confidence.clamp(0.0, 1.0),
            issues: Vec::new(),
            blocking_count: 0,
            next_action: "merge".to_string(),
            touched_files,
        }
    }

    /// Quick fail result.
    pub fn fail(issues: Vec<ReviewIssue>, touched_files: Vec<String>) -> Self {
        let blocking_count = issues.iter().filter(|i| i.blocking).count();
        Self {
            verdict: ReviewVerdict::Fail,
            confidence: 0.8,
            issues,
            blocking_count,
            next_action: "fix blocking issues and re-submit".to_string(),
            touched_files,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ast_grep_result_ok() {
        let result = AstGrepResult::ok(
            vec![AstMatch {
                file: "src/lib.rs".to_string(),
                line: 42,
                column: 5,
                matched_text: "unwrap()".to_string(),
                context: "let x = foo.unwrap();".to_string(),
                rule_id: Some("no-unwrap".to_string()),
            }],
            1,
            false,
            50,
        );
        assert!(result.success);
        assert_eq!(result.matches.len(), 1);
        assert!(!result.truncated);
    }

    #[test]
    fn test_ast_grep_result_err() {
        let result = AstGrepResult::err("ast-grep binary not found", 0);
        assert!(!result.success);
        assert!(result.matches.is_empty());
        assert!(result.error.is_some());
    }

    #[test]
    fn test_dependency_no_impact() {
        let result = DependencyCheckResult::no_impact(100);
        assert!(result.success);
        assert_eq!(result.impact_level, ImpactLevel::None);
        assert_eq!(result.direct_dependents, 0);
    }

    #[test]
    fn test_impact_level_display() {
        assert_eq!(ImpactLevel::None.to_string(), "none");
        assert_eq!(ImpactLevel::Low.to_string(), "low");
        assert_eq!(ImpactLevel::Medium.to_string(), "medium");
        assert_eq!(ImpactLevel::High.to_string(), "high");
    }

    #[test]
    fn test_verifier_gate_green() {
        let gates = vec![
            GateCheckResult {
                gate: "fmt".to_string(),
                passed: true,
                error_count: 0,
                warning_count: 0,
                dominant_category: None,
                duration_ms: 50,
            },
            GateCheckResult {
                gate: "clippy".to_string(),
                passed: true,
                error_count: 0,
                warning_count: 0,
                dominant_category: None,
                duration_ms: 200,
            },
        ];
        let result = VerifierGateResult::green(gates, 250);
        assert!(result.all_passed);
        assert_eq!(result.gates_passed, 2);
        assert_eq!(result.gates_total, 2);
        assert!(result.first_failure.is_none());
    }

    #[test]
    fn test_review_verdict_display() {
        assert_eq!(ReviewVerdict::Pass.to_string(), "pass");
        assert_eq!(ReviewVerdict::Fail.to_string(), "fail");
        assert_eq!(
            ReviewVerdict::NeedsEscalation.to_string(),
            "needs_escalation"
        );
    }

    #[test]
    fn test_review_decision_pass() {
        let result = ReviewDecisionResult::pass(0.95, vec!["src/lib.rs".to_string()]);
        assert_eq!(result.verdict, ReviewVerdict::Pass);
        assert_eq!(result.confidence, 0.95);
        assert_eq!(result.blocking_count, 0);
        assert_eq!(result.next_action, "merge");
    }

    #[test]
    fn test_review_decision_fail() {
        let issues = vec![
            ReviewIssue {
                blocking: true,
                file: Some("src/main.rs".to_string()),
                line: Some(42),
                description: "Missing error handling".to_string(),
                suggestion: Some("Use ? operator".to_string()),
            },
            ReviewIssue {
                blocking: false,
                file: None,
                line: None,
                description: "Consider adding docs".to_string(),
                suggestion: None,
            },
        ];
        let result = ReviewDecisionResult::fail(issues, vec!["src/main.rs".to_string()]);
        assert_eq!(result.verdict, ReviewVerdict::Fail);
        assert_eq!(result.blocking_count, 1);
        assert_eq!(result.issues.len(), 2);
    }

    #[test]
    fn test_review_decision_confidence_clamping() {
        let result = ReviewDecisionResult::pass(1.5, vec![]);
        assert_eq!(result.confidence, 1.0);

        let result = ReviewDecisionResult::pass(-0.5, vec![]);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_ast_grep_request_serialize() {
        let req = AstGrepRequest {
            pattern: "$EXPR.unwrap()".to_string(),
            language: "rust".to_string(),
            paths: vec!["src/".to_string()],
            max_results: 50,
            rule_id: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        let parsed: AstGrepRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.pattern, "$EXPR.unwrap()");
        assert_eq!(parsed.max_results, 50);
    }

    #[test]
    fn test_review_verdict_serde_roundtrip() {
        let json = serde_json::to_string(&ReviewVerdict::NeedsEscalation).unwrap();
        assert_eq!(json, "\"needs_escalation\"");
        let parsed: ReviewVerdict = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, ReviewVerdict::NeedsEscalation);
    }

    #[test]
    fn test_full_review_pipeline_roundtrip() {
        let decision = ReviewDecisionResult {
            verdict: ReviewVerdict::Fail,
            confidence: 0.85,
            issues: vec![ReviewIssue {
                blocking: true,
                file: Some("src/lib.rs".to_string()),
                line: Some(10),
                description: "Unsafe unwrap".to_string(),
                suggestion: Some("Use expect or ?".to_string()),
            }],
            blocking_count: 1,
            next_action: "fix".to_string(),
            touched_files: vec!["src/lib.rs".to_string()],
        };
        let json = serde_json::to_string_pretty(&decision).unwrap();
        let parsed: ReviewDecisionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.verdict, ReviewVerdict::Fail);
        assert_eq!(parsed.issues.len(), 1);
        assert!(parsed.issues[0].blocking);
    }
}
