//! Reviewer Workflow Policy — Enforced Stage Ordering
//!
//! Defines and enforces the reviewer pipeline: verifier gates must pass
//! before AST analysis, which must pass before dependency impact checks.
//! Each stage produces an auditable trace entry.
//!
//! # Pipeline
//!
//! ```text
//! Stage 1: Verifier Gates (fmt → clippy → check → test)
//!    │ FAIL → short-circuit, return failure
//!    ▼
//! Stage 2: AST Pattern Analysis (anti-pattern rules)
//!    │ FAIL → flag issues, continue to stage 3
//!    ▼
//! Stage 3: Dependency Impact Check (affected files, API changes)
//!    │
//!    ▼
//! Aggregate → ReviewDecision
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Stages in the reviewer pipeline, in execution order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReviewStage {
    /// Quality gate pipeline (fmt, clippy, check, test).
    VerifierGates,
    /// AST pattern matching for anti-patterns.
    AstAnalysis,
    /// Dependency impact analysis.
    DependencyCheck,
    /// Final reviewer decision (aggregation of prior stages).
    Decision,
}

impl ReviewStage {
    /// Return all stages in execution order.
    pub fn ordered() -> &'static [Self] {
        &[
            Self::VerifierGates,
            Self::AstAnalysis,
            Self::DependencyCheck,
            Self::Decision,
        ]
    }

    /// Whether this stage short-circuits the pipeline on failure.
    pub fn is_blocking(self) -> bool {
        matches!(self, Self::VerifierGates)
    }
}

impl std::fmt::Display for ReviewStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerifierGates => write!(f, "verifier_gates"),
            Self::AstAnalysis => write!(f, "ast_analysis"),
            Self::DependencyCheck => write!(f, "dependency_check"),
            Self::Decision => write!(f, "decision"),
        }
    }
}

/// Outcome of a single stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageOutcome {
    /// Stage passed.
    Passed,
    /// Stage found issues but pipeline continues.
    Warning,
    /// Stage failed and pipeline may short-circuit.
    Failed,
    /// Stage was skipped (due to prior failure or policy).
    Skipped,
}

impl std::fmt::Display for StageOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Passed => write!(f, "passed"),
            Self::Warning => write!(f, "warning"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

/// Audit trace entry for one pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEntry {
    /// Which stage this entry is for.
    pub stage: ReviewStage,
    /// Outcome of the stage.
    pub outcome: StageOutcome,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Number of issues found.
    pub issue_count: usize,
    /// When the stage started.
    pub started_at: DateTime<Utc>,
    /// Summary message.
    pub summary: String,
}

/// Complete audit trace for a reviewer pipeline run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewTrace {
    /// Pipeline execution ID.
    pub trace_id: String,
    /// Individual stage entries.
    pub entries: Vec<TraceEntry>,
    /// Whether the pipeline short-circuited.
    pub short_circuited: bool,
    /// Stage that caused short-circuit, if any.
    pub short_circuit_stage: Option<ReviewStage>,
    /// When the pipeline started.
    pub started_at: DateTime<Utc>,
    /// Total duration in milliseconds.
    pub total_duration_ms: u64,
}

impl ReviewTrace {
    /// Create a new trace.
    pub fn new(trace_id: &str) -> Self {
        Self {
            trace_id: trace_id.to_string(),
            entries: Vec::new(),
            short_circuited: false,
            short_circuit_stage: None,
            started_at: Utc::now(),
            total_duration_ms: 0,
        }
    }

    /// Record a stage outcome.
    pub fn record(
        &mut self,
        stage: ReviewStage,
        outcome: StageOutcome,
        duration_ms: u64,
        issue_count: usize,
        summary: &str,
    ) {
        self.entries.push(TraceEntry {
            stage,
            outcome,
            duration_ms,
            issue_count,
            started_at: Utc::now(),
            summary: summary.to_string(),
        });
        self.total_duration_ms += duration_ms;
    }

    /// Mark the pipeline as short-circuited at a stage.
    pub fn mark_short_circuit(&mut self, stage: ReviewStage) {
        self.short_circuited = true;
        self.short_circuit_stage = Some(stage);
    }

    /// Number of stages that have been executed.
    pub fn stages_executed(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| e.outcome != StageOutcome::Skipped)
            .count()
    }

    /// Whether all executed stages passed.
    pub fn all_passed(&self) -> bool {
        self.entries
            .iter()
            .all(|e| matches!(e.outcome, StageOutcome::Passed | StageOutcome::Skipped))
    }

    /// Total issues across all stages.
    pub fn total_issues(&self) -> usize {
        self.entries.iter().map(|e| e.issue_count).sum()
    }

    /// Compact text summary for logging.
    ///
    /// Example: `[PASS] 3/4 stages | 0 issues | 1200ms`
    pub fn compact_summary(&self) -> String {
        let status = if self.all_passed() { "PASS" } else { "FAIL" };
        let total = ReviewStage::ordered().len();
        let executed = self.stages_executed();
        let issues = self.total_issues();
        let mut parts = vec![format!("[{}] {}/{} stages", status, executed, total)];
        if issues > 0 {
            parts.push(format!("{} issues", issues));
        }
        if self.short_circuited {
            if let Some(stage) = self.short_circuit_stage {
                parts.push(format!("short-circuited at {}", stage));
            }
        }
        parts.push(format!("{}ms", self.total_duration_ms));
        parts.join(" | ")
    }
}

/// Policy engine that enforces the reviewer pipeline ordering.
///
/// The policy determines which stages to execute and whether to
/// short-circuit on failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerPolicy {
    /// Whether to short-circuit on verifier gate failure.
    pub fail_fast_on_verifier: bool,
    /// Whether AST analysis is required (vs. optional).
    pub require_ast_analysis: bool,
    /// Whether dependency check is required (vs. optional).
    pub require_dependency_check: bool,
    /// Maximum total duration for the pipeline in ms (0 = unlimited).
    pub max_duration_ms: u64,
}

impl ReviewerPolicy {
    /// Determine which stages should be executed given the current trace state.
    ///
    /// Returns the next stage to execute, or None if the pipeline is complete
    /// or should stop.
    pub fn next_stage(&self, trace: &ReviewTrace) -> Option<ReviewStage> {
        let completed: Vec<ReviewStage> = trace.entries.iter().map(|e| e.stage).collect();

        for &stage in ReviewStage::ordered() {
            if completed.contains(&stage) {
                continue;
            }

            // Check if we should skip this stage
            if stage == ReviewStage::AstAnalysis && !self.require_ast_analysis {
                // Skip but allow — caller should record as Skipped
                return Some(stage);
            }
            if stage == ReviewStage::DependencyCheck && !self.require_dependency_check {
                return Some(stage);
            }

            // Check if we should short-circuit
            if self.fail_fast_on_verifier && trace.short_circuited {
                return None;
            }

            // Check timeout
            if self.max_duration_ms > 0 && trace.total_duration_ms >= self.max_duration_ms {
                return None;
            }

            return Some(stage);
        }

        None
    }

    /// Whether a stage failure should short-circuit the pipeline.
    pub fn should_short_circuit(&self, stage: ReviewStage) -> bool {
        self.fail_fast_on_verifier && stage.is_blocking()
    }

    /// Validate that a trace conforms to the policy ordering.
    ///
    /// Returns Ok if ordering is correct, Err with violation description.
    pub fn validate_ordering(&self, trace: &ReviewTrace) -> Result<(), String> {
        let ordered = ReviewStage::ordered();
        let mut last_idx = None;

        for entry in &trace.entries {
            if entry.outcome == StageOutcome::Skipped {
                continue;
            }
            let idx = ordered.iter().position(|&s| s == entry.stage);
            if let (Some(current), Some(prev)) = (idx, last_idx) {
                if current <= prev {
                    return Err(format!(
                        "stage '{}' executed after '{}' (out of order)",
                        entry.stage,
                        trace
                            .entries
                            .iter()
                            .find(|e| ordered.iter().position(|&s| s == e.stage) == Some(prev))
                            .map(|e| e.stage.to_string())
                            .unwrap_or_default()
                    ));
                }
            }
            last_idx = idx;
        }

        Ok(())
    }
}

impl Default for ReviewerPolicy {
    fn default() -> Self {
        Self {
            fail_fast_on_verifier: true,
            require_ast_analysis: true,
            require_dependency_check: true,
            max_duration_ms: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stage_ordering() {
        let stages = ReviewStage::ordered();
        assert_eq!(stages[0], ReviewStage::VerifierGates);
        assert_eq!(stages[1], ReviewStage::AstAnalysis);
        assert_eq!(stages[2], ReviewStage::DependencyCheck);
        assert_eq!(stages[3], ReviewStage::Decision);
    }

    #[test]
    fn test_stage_blocking() {
        assert!(ReviewStage::VerifierGates.is_blocking());
        assert!(!ReviewStage::AstAnalysis.is_blocking());
        assert!(!ReviewStage::DependencyCheck.is_blocking());
        assert!(!ReviewStage::Decision.is_blocking());
    }

    #[test]
    fn test_trace_all_passed() {
        let mut trace = ReviewTrace::new("test-001");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Passed,
            100,
            0,
            "4/4 gates",
        );
        trace.record(
            ReviewStage::AstAnalysis,
            StageOutcome::Passed,
            50,
            0,
            "clean",
        );
        trace.record(
            ReviewStage::DependencyCheck,
            StageOutcome::Passed,
            30,
            0,
            "no impact",
        );
        trace.record(ReviewStage::Decision, StageOutcome::Passed, 10, 0, "pass");

        assert!(trace.all_passed());
        assert_eq!(trace.stages_executed(), 4);
        assert_eq!(trace.total_issues(), 0);
        assert_eq!(trace.total_duration_ms, 190);
        assert!(!trace.short_circuited);
    }

    #[test]
    fn test_trace_short_circuit() {
        let mut trace = ReviewTrace::new("test-002");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Failed,
            200,
            5,
            "2/4 gates failed",
        );
        trace.mark_short_circuit(ReviewStage::VerifierGates);

        assert!(!trace.all_passed());
        assert_eq!(trace.stages_executed(), 1);
        assert_eq!(trace.total_issues(), 5);
        assert!(trace.short_circuited);
        assert_eq!(trace.short_circuit_stage, Some(ReviewStage::VerifierGates));
    }

    #[test]
    fn test_trace_compact_summary_pass() {
        let mut trace = ReviewTrace::new("test-003");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Passed,
            100,
            0,
            "ok",
        );
        trace.record(ReviewStage::AstAnalysis, StageOutcome::Passed, 50, 0, "ok");
        trace.record(
            ReviewStage::DependencyCheck,
            StageOutcome::Passed,
            30,
            0,
            "ok",
        );
        trace.record(ReviewStage::Decision, StageOutcome::Passed, 10, 0, "ok");

        let summary = trace.compact_summary();
        assert!(summary.contains("[PASS]"));
        assert!(summary.contains("4/4 stages"));
        assert!(summary.contains("190ms"));
    }

    #[test]
    fn test_trace_compact_summary_fail() {
        let mut trace = ReviewTrace::new("test-004");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Failed,
            200,
            3,
            "failed",
        );
        trace.mark_short_circuit(ReviewStage::VerifierGates);

        let summary = trace.compact_summary();
        assert!(summary.contains("[FAIL]"));
        assert!(summary.contains("3 issues"));
        assert!(summary.contains("short-circuited"));
    }

    #[test]
    fn test_policy_next_stage_normal() {
        let policy = ReviewerPolicy::default();
        let trace = ReviewTrace::new("test");

        assert_eq!(policy.next_stage(&trace), Some(ReviewStage::VerifierGates));
    }

    #[test]
    fn test_policy_next_stage_after_verifier() {
        let policy = ReviewerPolicy::default();
        let mut trace = ReviewTrace::new("test");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Passed,
            100,
            0,
            "ok",
        );

        assert_eq!(policy.next_stage(&trace), Some(ReviewStage::AstAnalysis));
    }

    #[test]
    fn test_policy_short_circuits_on_verifier_fail() {
        let policy = ReviewerPolicy::default();
        let mut trace = ReviewTrace::new("test");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Failed,
            100,
            3,
            "fail",
        );
        trace.mark_short_circuit(ReviewStage::VerifierGates);

        assert_eq!(policy.next_stage(&trace), None);
    }

    #[test]
    fn test_policy_timeout() {
        let policy = ReviewerPolicy {
            max_duration_ms: 500,
            ..Default::default()
        };
        let mut trace = ReviewTrace::new("test");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Passed,
            300,
            0,
            "ok",
        );
        trace.record(ReviewStage::AstAnalysis, StageOutcome::Passed, 250, 0, "ok");

        // Over budget — should stop
        assert_eq!(policy.next_stage(&trace), None);
    }

    #[test]
    fn test_policy_validate_ordering_ok() {
        let policy = ReviewerPolicy::default();
        let mut trace = ReviewTrace::new("test");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Passed,
            100,
            0,
            "ok",
        );
        trace.record(ReviewStage::AstAnalysis, StageOutcome::Passed, 50, 0, "ok");
        trace.record(
            ReviewStage::DependencyCheck,
            StageOutcome::Passed,
            30,
            0,
            "ok",
        );

        assert!(policy.validate_ordering(&trace).is_ok());
    }

    #[test]
    fn test_policy_validate_ordering_violation() {
        let policy = ReviewerPolicy::default();
        let mut trace = ReviewTrace::new("test");
        // Wrong order: AST before Verifier
        trace.record(ReviewStage::AstAnalysis, StageOutcome::Passed, 50, 0, "ok");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Passed,
            100,
            0,
            "ok",
        );

        assert!(policy.validate_ordering(&trace).is_err());
    }

    #[test]
    fn test_should_short_circuit() {
        let policy = ReviewerPolicy::default();
        assert!(policy.should_short_circuit(ReviewStage::VerifierGates));
        assert!(!policy.should_short_circuit(ReviewStage::AstAnalysis));
        assert!(!policy.should_short_circuit(ReviewStage::DependencyCheck));
    }

    #[test]
    fn test_policy_json_roundtrip() {
        let policy = ReviewerPolicy {
            fail_fast_on_verifier: true,
            require_ast_analysis: false,
            require_dependency_check: true,
            max_duration_ms: 30000,
        };
        let json = serde_json::to_string(&policy).unwrap();
        let parsed: ReviewerPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.require_ast_analysis, false);
        assert_eq!(parsed.max_duration_ms, 30000);
    }

    #[test]
    fn test_stage_display() {
        assert_eq!(ReviewStage::VerifierGates.to_string(), "verifier_gates");
        assert_eq!(ReviewStage::AstAnalysis.to_string(), "ast_analysis");
        assert_eq!(ReviewStage::DependencyCheck.to_string(), "dependency_check");
        assert_eq!(ReviewStage::Decision.to_string(), "decision");
    }

    #[test]
    fn test_outcome_display() {
        assert_eq!(StageOutcome::Passed.to_string(), "passed");
        assert_eq!(StageOutcome::Warning.to_string(), "warning");
        assert_eq!(StageOutcome::Failed.to_string(), "failed");
        assert_eq!(StageOutcome::Skipped.to_string(), "skipped");
    }

    #[test]
    fn test_trace_with_skipped_stages() {
        let mut trace = ReviewTrace::new("test");
        trace.record(
            ReviewStage::VerifierGates,
            StageOutcome::Passed,
            100,
            0,
            "ok",
        );
        trace.record(
            ReviewStage::AstAnalysis,
            StageOutcome::Skipped,
            0,
            0,
            "not required",
        );
        trace.record(
            ReviewStage::DependencyCheck,
            StageOutcome::Passed,
            30,
            0,
            "ok",
        );
        trace.record(ReviewStage::Decision, StageOutcome::Passed, 10, 0, "pass");

        assert!(trace.all_passed()); // Skipped counts as passed
        assert_eq!(trace.stages_executed(), 3); // Skipped not counted
    }
}
