use serde::{Deserialize, Serialize};

/// The action performed on a file artifact during an iteration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactAction {
    /// File was read but not modified.
    Read,
    /// File was modified (existed before the iteration).
    Modified,
    /// File was created during this iteration.
    Created,
    /// File was deleted during this iteration.
    Deleted,
}

/// A record of a single file artifact touched during an iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactRecord {
    /// Relative path to the file within the worktree.
    pub path: String,
    /// The action performed on the file.
    pub action: ArtifactAction,
    /// Optional line range affected (start, end), inclusive.
    pub line_range: Option<(u32, u32)>,
    /// Net change in file size in bytes (positive = grew, negative = shrank).
    pub size_delta: Option<i64>,
}

/// Current schema version for execution artifacts.
/// Bump when adding/removing/renaming fields.
pub const ARTIFACT_SCHEMA_VERSION: u8 = 1;

/// Snapshot of the routing decision for an iteration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDecision {
    /// Which coder was selected (e.g. "RustCoder", "GeneralCoder").
    pub coder: String,
    /// Error categories that influenced the routing decision.
    pub input_error_categories: Vec<String>,
    /// The tier at the time of routing.
    pub tier: String,
    /// Free-text rationale (e.g. "borrow errors → RustCoder").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rationale: Option<String>,
}

/// Snapshot of the routing decision for an iteration.
/// Compact snapshot of verifier gate results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierSnapshot {
    /// Whether all gates passed.
    pub all_green: bool,
    /// Per-gate results: (gate_name, passed, error_count).
    pub gates: Vec<GateSnapshot>,
    /// Total errors across all gates.
    pub total_errors: usize,
    /// Top error categories from the verifier.
    pub error_categories: Vec<String>,
}

/// Result of a single verifier gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSnapshot {
    pub name: String,
    pub passed: bool,
    pub error_count: usize,
    /// First few error messages (truncated for space).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sample_errors: Vec<String>,
}

/// Snapshot of the cloud evaluator (validator) result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluatorSnapshot {
    /// Model used for evaluation.
    pub model: String,
    /// The verdict: pass, fail, or needs_escalation.
    pub verdict: String,
    /// Confidence score (0.0–1.0).
    pub confidence: f32,
    /// Whether the schema was valid (fail-closed on invalid).
    pub schema_valid: bool,
    /// Blocking issues identified by the evaluator.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub blocking_issues: Vec<String>,
    /// Suggested next action from the evaluator.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub suggested_next_action: Option<String>,
}

/// Why the system decided to retry, escalate, or stop.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetryAction {
    /// Continue at the same tier.
    Retry,
    /// Escalate to a higher tier.
    Escalate { from_tier: String, to_tier: String },
    /// Issue resolved — merge and close.
    Resolved,
    /// Give up — stuck or budget exhausted.
    GiveUp { reason: String },
}

/// Rationale for the retry/escalate/stop decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryRationale {
    /// What action was taken.
    pub action: RetryAction,
    /// Error count before this decision.
    pub error_count_before: usize,
    /// Error count after this iteration's changes.
    pub error_count_after: usize,
    /// Whether a regression was detected.
    pub regression: bool,
    /// Number of consecutive no-change iterations.
    pub consecutive_no_change: u32,
    /// Remaining iteration budget.
    pub budget_remaining: u32,
}

/// Complete typed execution artifact for a single iteration.
///
/// Captures every decision point for offline replay and root-cause analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionArtifact {
    /// Schema version for backward compatibility.
    pub schema_version: u8,
    /// Routing decision snapshot.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub route_decision: Option<RouteDecision>,
    /// Verifier gate results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verifier_snapshot: Option<VerifierSnapshot>,
    /// Cloud evaluator result.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evaluator_snapshot: Option<EvaluatorSnapshot>,
    /// Retry/escalate/stop rationale.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retry_rationale: Option<RetryRationale>,
}

impl ExecutionArtifact {
    /// Create a new empty artifact with the current schema version.
    pub fn new() -> Self {
        Self {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            route_decision: None,
            verifier_snapshot: None,
            evaluator_snapshot: None,
            retry_rationale: None,
        }
    }
}

impl Default for ExecutionArtifact {
    fn default() -> Self {
        Self::new()
    }
}

/// Metrics for a single iteration within a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationMetrics {
    pub iteration: u32,
    pub tier: String,
    pub agent_model: String,
    pub agent_prompt_tokens: u32,
    pub agent_completion_tokens: u32,
    pub agent_response_ms: u64,
    pub verifier_ms: u64,
    pub error_count: usize,
    pub error_categories: Vec<String>,
    pub no_change: bool,
    pub auto_fix_applied: bool,
    pub regression_detected: bool,
    pub rollback_performed: bool,
    pub escalated: bool,
    pub coder_route: Option<String>,
    /// File-level footprint of this iteration.
    pub artifacts: Vec<ArtifactRecord>,
    /// Typed execution artifact for replay and diagnostics.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_artifact: Option<ExecutionArtifact>,
    /// Hill-climbing progress score: 1.0 - (error_count / best_error_count).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress_score: Option<f64>,
    /// Best error count seen so far across all iterations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub best_error_count: Option<usize>,
}

/// Metrics for a complete orchestrator session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMetrics {
    pub session_id: String,
    pub issue_id: String,
    pub issue_title: String,
    pub success: bool,
    pub total_iterations: u32,
    pub final_tier: String,
    pub elapsed_ms: u64,
    pub total_no_change_iterations: u32,
    pub no_change_rate: f64,
    pub cloud_validations: Vec<ValidationMetric>,
    #[serde(default)]
    pub local_validations: Vec<ValidationMetric>,
    pub iterations: Vec<IterationMetrics>,
    pub timestamp: String,
    /// Active stack profile.
    pub stack_profile: String,
    /// Repository identifier.
    pub repo_id: Option<String>,
    /// Adapter identifier.
    pub adapter_id: Option<String>,
    /// Number of iterations until first file write.
    pub turns_until_first_write: Option<u32>,
    /// Whether a file write occurred by iteration 2.
    pub write_by_turn_2: bool,
    /// Version of the role model map.
    pub role_map_version: String,
    /// TensorZero episode ID for linking inferences to feedback.
    /// Format: `{issue_id}_{session_short_id}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tensorzero_episode_id: Option<String>,
    /// Per-session harness component trace for load-bearing audit.
    ///
    /// Which components fired and whether they caught real issues.
    /// Use this over many sessions to identify components that are never
    /// load-bearing and can be simplified away.
    #[serde(default)]
    pub harness_trace: HarnessComponentTrace,
    /// Total input tokens consumed across all TZ inferences in this session.
    #[serde(default)]
    pub input_tokens: u64,
    /// Total output tokens generated across all TZ inferences in this session.
    #[serde(default)]
    pub output_tokens: u64,
    /// Estimated cost in USD based on cloud pricing rates.
    #[serde(default)]
    pub estimated_cost_usd: f64,
}

/// Per-session record of which harness components fired and whether they
/// contributed to quality.
///
/// Over many sessions this reveals which components are genuinely load-bearing
/// (see Anthropic's harness-design article: "periodically audit each assumption").
/// A component showing `fired: true, caught_issue: false` over N sessions is
/// a candidate for removal or reformulation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HarnessComponentTrace {
    /// Whether the blind reviewer ran on this session.
    pub reviewer_fired: bool,
    /// Reviewer said FAIL where verifier said PASS — i.e., reviewer caught
    /// a quality issue the verifier missed.
    pub reviewer_caught_issue: bool,
    /// Whether the adversary red-team agent ran.
    pub adversary_fired: bool,
    /// Adversary found an issue (generated a failing test case or security concern).
    pub adversary_caught_issue: bool,
    /// Whether the planner ran an acceptance-criteria generation step.
    pub planner_fired: bool,
    /// Whether a sprint contract was negotiated and used.
    pub sprint_contract_used: bool,
    /// Whether a pivot (strategy change) was triggered mid-session.
    pub pivot_triggered: bool,
    /// Whether a context reset (clean-slate handoff) was triggered.
    pub context_reset_triggered: bool,
    /// How many intra-session dead-end records were injected into prompts.
    pub dead_end_injection_count: u32,
    /// Whether context anxiety was detected (proactive reset signal).
    pub context_anxiety_detected: bool,
    /// Reviewer leniency flags raised (non-zero = reviewer may have been too lenient).
    pub reviewer_leniency_flag_count: u32,
}

/// Result of a single cloud validation call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetric {
    pub model: String,
    pub passed: bool,
}

/// In-flight state for the current iteration.
pub struct IterationBuilder {
    pub iteration: u32,
    pub tier: String,
    pub agent_model: String,
    pub agent_prompt_tokens: u32,
    pub agent_completion_tokens: u32,
    pub agent_response_ms: u64,
    pub verifier_ms: u64,
    pub error_count: usize,
    pub error_categories: Vec<String>,
    pub no_change: bool,
    pub auto_fix_applied: bool,
    pub regression_detected: bool,
    pub rollback_performed: bool,
    pub escalated: bool,
    pub coder_route: Option<String>,
    pub artifacts: Vec<ArtifactRecord>,
    pub execution_artifact: ExecutionArtifact,
    pub progress_score: Option<f64>,
    pub best_error_count: Option<usize>,
}

/// A single entry in the failure ledger (JSONL format).
///
/// Captures both failures and successes (per ECC continuous learning pattern)
/// for trajectory analysis and pattern detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureLedgerEntry {
    pub tool: String,
    pub error_class: String,
    pub signal_traced: String,
    pub file_path: Option<String>,
    pub iteration: usize,
    pub timestamp: String,
    pub success: bool,
}

/// SLO status: whether the metric is within target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloStatus {
    /// Metric meets or exceeds target.
    Met,
    /// Metric is within warning threshold but not meeting target.
    Warning,
    /// Metric is below acceptable threshold.
    Breached,
}

/// A single SLO measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloMeasurement {
    /// SLO name.
    pub name: String,
    /// Measured value.
    pub value: f64,
    /// Target value (must meet or exceed).
    pub target: f64,
    /// Warning threshold (below target but acceptable).
    pub warning: f64,
    /// Current status.
    pub status: SloStatus,
}

/// SLO targets for swarm operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloTargets {
    /// Target success rate (fraction, 0.0–1.0). Default: 0.80.
    pub success_rate: f64,
    /// Target average iterations-to-green. Default: 2.5.
    pub avg_iterations_to_green: f64,
    /// Target stuck rate (fraction, lower is better). Default: 0.10.
    pub stuck_rate: f64,
    /// Target no-change rate (fraction, lower is better). Default: 0.15.
    pub no_change_rate: f64,
}

impl Default for SloTargets {
    fn default() -> Self {
        Self {
            success_rate: 0.80,
            avg_iterations_to_green: 2.5,
            stuck_rate: 0.10,
            no_change_rate: 0.15,
        }
    }
}

/// Complete SLO report across a cohort of sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SloReport {
    /// Number of sessions analyzed.
    pub session_count: usize,
    /// Individual SLO measurements.
    pub measurements: Vec<SloMeasurement>,
    /// Overall status (worst of all measurements).
    pub overall_status: SloStatus,
}

impl SloReport {
    /// Whether all SLOs are met.
    pub fn all_met(&self) -> bool {
        self.overall_status == SloStatus::Met
    }

    /// Compact summary for logging.
    pub fn summary(&self) -> String {
        let items: Vec<String> = self
            .measurements
            .iter()
            .map(|m| format!("{}={:.2}/{:.2}({:?})", m.name, m.value, m.target, m.status))
            .collect();
        format!(
            "[{:?}] {} sessions | {}",
            self.overall_status,
            self.session_count,
            items.join(" | ")
        )
    }
}

/// Approximate per-token costs in USD (per million tokens).
///
/// Cloud costs are based on Claude Opus 4.6 thinking pricing.
/// Local costs are $0 since we self-host on HPC.
pub mod cost_rates {
    /// Cloud input: ~$15/M tokens (Opus 4.6 thinking via CLIAPIProxy)
    pub const CLOUD_INPUT_PER_M: f64 = 15.0;
    /// Cloud output: ~$75/M tokens (Opus 4.6 thinking via CLIAPIProxy)
    pub const CLOUD_OUTPUT_PER_M: f64 = 75.0;
    /// Local: $0 (self-hosted Qwen3.5 on V100S cluster)
    pub const LOCAL_INPUT_PER_M: f64 = 0.0;
    pub const LOCAL_OUTPUT_PER_M: f64 = 0.0;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::*;

    #[test]
    fn test_execution_artifact_default() {
        let artifact = ExecutionArtifact::new();
        assert_eq!(artifact.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert!(artifact.route_decision.is_none());
        assert!(artifact.verifier_snapshot.is_none());
        assert!(artifact.evaluator_snapshot.is_none());
        assert!(artifact.retry_rationale.is_none());
    }

    #[test]
    fn test_execution_artifact_serde_roundtrip() {
        let artifact = ExecutionArtifact {
            schema_version: ARTIFACT_SCHEMA_VERSION,
            route_decision: Some(RouteDecision {
                coder: "RustCoder".into(),
                input_error_categories: vec!["BorrowChecker".into(), "Lifetime".into()],
                tier: "Integrator".into(),
                rationale: Some("borrow errors → RustCoder".into()),
            }),
            verifier_snapshot: Some(VerifierSnapshot {
                all_green: false,
                gates: vec![
                    GateSnapshot {
                        name: "fmt".into(),
                        passed: true,
                        error_count: 0,
                        sample_errors: vec![],
                    },
                    GateSnapshot {
                        name: "clippy".into(),
                        passed: false,
                        error_count: 2,
                        sample_errors: vec!["unused variable `x`".into()],
                    },
                ],
                total_errors: 2,
                error_categories: vec!["Clippy".into()],
            }),
            evaluator_snapshot: Some(EvaluatorSnapshot {
                model: "claude-sonnet-4-5".into(),
                verdict: "fail".into(),
                confidence: 0.85,
                schema_valid: true,
                blocking_issues: vec!["clippy warnings remain".into()],
                suggested_next_action: Some("fix clippy warnings".into()),
            }),
            retry_rationale: Some(RetryRationale {
                action: RetryAction::Retry,
                error_count_before: 5,
                error_count_after: 2,
                regression: false,
                consecutive_no_change: 0,
                budget_remaining: 4,
            }),
        };

        let json = serde_json::to_string_pretty(&artifact).unwrap();
        let restored: ExecutionArtifact = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.schema_version, ARTIFACT_SCHEMA_VERSION);
        let rd = restored.route_decision.unwrap();
        assert_eq!(rd.coder, "RustCoder");
        assert_eq!(rd.input_error_categories.len(), 2);

        let vs = restored.verifier_snapshot.unwrap();
        assert!(!vs.all_green);
        assert_eq!(vs.gates.len(), 2);
        assert_eq!(vs.total_errors, 2);

        let es = restored.evaluator_snapshot.unwrap();
        assert_eq!(es.verdict, "fail");
        assert_eq!(es.confidence, 0.85);
        assert!(es.schema_valid);

        let rr = restored.retry_rationale.unwrap();
        assert!(matches!(rr.action, RetryAction::Retry));
        assert_eq!(rr.budget_remaining, 4);
    }

    #[test]
    fn test_execution_artifact_backward_compat_missing_fields() {
        // Simulate loading an older JSON that doesn't have execution_artifact
        let json = r#"{
            "iteration": 1, "tier": "Integrator", "agent_model": "m1",
            "agent_prompt_tokens": 100, "agent_completion_tokens": 50,
            "agent_response_ms": 1000, "verifier_ms": 500,
            "error_count": 0, "error_categories": [],
            "no_change": false, "auto_fix_applied": false,
            "regression_detected": false, "rollback_performed": false,
            "escalated": false, "artifacts": []
        }"#;
        let metrics: IterationMetrics = serde_json::from_str(json).unwrap();
        assert!(metrics.execution_artifact.is_none());
    }

    #[test]
    fn test_collector_records_execution_artifacts() {
        let mut collector = MetricsCollector::new(
            "sess-ea",
            "issue-ea",
            "Artifact test",
            "hybrid_balanced_v1",
            None,
            None,
            "v1",
        );

        collector.start_iteration(1, "Integrator");
        collector.record_route_decision(RouteDecision {
            coder: "GeneralCoder".into(),
            input_error_categories: vec![],
            tier: "Integrator".into(),
            rationale: None,
        });
        collector.record_verifier_snapshot(VerifierSnapshot {
            all_green: true,
            gates: vec![GateSnapshot {
                name: "fmt".into(),
                passed: true,
                error_count: 0,
                sample_errors: vec![],
            }],
            total_errors: 0,
            error_categories: vec![],
        });
        collector.record_retry_rationale(RetryRationale {
            action: RetryAction::Resolved,
            error_count_before: 3,
            error_count_after: 0,
            regression: false,
            consecutive_no_change: 0,
            budget_remaining: 5,
        });
        collector.finish_iteration();

        let metrics = collector.finalize(true, "Integrator");
        let artifact = metrics.iterations[0].execution_artifact.as_ref().unwrap();
        assert!(artifact.route_decision.is_some());
        assert!(artifact.verifier_snapshot.is_some());
        assert!(artifact.retry_rationale.is_some());
        assert!(artifact.evaluator_snapshot.is_none());
    }

    #[test]
    fn test_collector_omits_empty_artifact() {
        let mut collector = MetricsCollector::new(
            "sess-empty",
            "issue-empty",
            "Empty artifact",
            "hybrid_balanced_v1",
            None,
            None,
            "v1",
        );
        collector.start_iteration(1, "Worker");
        // Don't record any artifact components
        collector.finish_iteration();

        let metrics = collector.finalize(true, "Worker");
        // No artifact attached when nothing was recorded
        assert!(metrics.iterations[0].execution_artifact.is_none());
    }

    #[test]
    fn test_retry_action_escalate_serde() {
        let rationale = RetryRationale {
            action: RetryAction::Escalate {
                from_tier: "Integrator".into(),
                to_tier: "Cloud".into(),
            },
            error_count_before: 5,
            error_count_after: 5,
            regression: false,
            consecutive_no_change: 2,
            budget_remaining: 3,
        };
        let json = serde_json::to_string(&rationale).unwrap();
        let restored: RetryRationale = serde_json::from_str(&json).unwrap();
        match restored.action {
            RetryAction::Escalate {
                from_tier, to_tier, ..
            } => {
                assert_eq!(from_tier, "Integrator");
                assert_eq!(to_tier, "Cloud");
            }
            _ => panic!("Expected Escalate variant"),
        }
    }

    #[test]
    fn test_retry_action_give_up_serde() {
        let rationale = RetryRationale {
            action: RetryAction::GiveUp {
                reason: "budget exhausted".into(),
            },
            error_count_before: 10,
            error_count_after: 10,
            regression: false,
            consecutive_no_change: 3,
            budget_remaining: 0,
        };
        let json = serde_json::to_string(&rationale).unwrap();
        assert!(json.contains("give_up"));
        assert!(json.contains("budget exhausted"));
    }
}
