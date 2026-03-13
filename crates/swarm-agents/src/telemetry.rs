//! Structured telemetry for dogfooding the swarm.
//!
//! Captures per-session and per-iteration metrics during orchestrator runs.
//! Two output sinks:
//! - `.swarm-metrics.json` (in worktree): complete session snapshot, overwritten each session
//! - `.swarm-telemetry.jsonl` (in repo root): append-only log of all sessions

use std::path::Path;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

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

// ──────────────────────────────────────────────────────────────────────────────
// Typed Execution Artifacts — structured decision records for replay/diagnostics
// ──────────────────────────────────────────────────────────────────────────────

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
}

/// Result of a single cloud validation call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationMetric {
    pub model: String,
    pub passed: bool,
}

/// Builder that accumulates metrics during the orchestrator loop.
///
/// Call `start_iteration()` / `finish_iteration()` around each loop body,
/// then `finalize()` at the end to produce the complete `SessionMetrics`.
pub struct MetricsCollector {
    session_id: String,
    issue_id: String,
    issue_title: String,
    session_start: Instant,
    current_iteration: Option<IterationBuilder>,
    iterations: Vec<IterationMetrics>,
    cloud_validations: Vec<ValidationMetric>,
    local_validations: Vec<ValidationMetric>,
    stack_profile: String,
    repo_id: Option<String>,
    adapter_id: Option<String>,
    role_map_version: String,
}

/// In-flight state for the current iteration.
struct IterationBuilder {
    iteration: u32,
    tier: String,
    agent_model: String,
    agent_prompt_tokens: u32,
    agent_completion_tokens: u32,
    agent_response_ms: u64,
    verifier_ms: u64,
    error_count: usize,
    error_categories: Vec<String>,
    no_change: bool,
    auto_fix_applied: bool,
    regression_detected: bool,
    rollback_performed: bool,
    escalated: bool,
    coder_route: Option<String>,
    artifacts: Vec<ArtifactRecord>,
    execution_artifact: ExecutionArtifact,
    progress_score: Option<f64>,
    best_error_count: Option<usize>,
}

impl MetricsCollector {
    pub fn new(
        session_id: &str,
        issue_id: &str,
        issue_title: &str,
        stack_profile: &str,
        repo_id: Option<String>,
        adapter_id: Option<String>,
        role_map_version: &str,
    ) -> Self {
        Self {
            session_id: session_id.to_string(),
            issue_id: issue_id.to_string(),
            issue_title: issue_title.to_string(),
            session_start: Instant::now(),
            current_iteration: None,
            iterations: Vec::new(),
            cloud_validations: Vec::new(),
            local_validations: Vec::new(),
            stack_profile: stack_profile.to_string(),
            repo_id,
            adapter_id,
            role_map_version: role_map_version.to_string(),
        }
    }

    /// Begin tracking a new iteration.
    pub fn start_iteration(&mut self, iteration: u32, tier: &str) {
        self.current_iteration = Some(IterationBuilder {
            iteration,
            tier: tier.to_string(),
            agent_model: String::new(),
            agent_prompt_tokens: 0,
            agent_completion_tokens: 0,
            agent_response_ms: 0,
            verifier_ms: 0,
            error_count: 0,
            error_categories: Vec::new(),
            no_change: false,
            auto_fix_applied: false,
            regression_detected: false,
            rollback_performed: false,
            escalated: false,
            coder_route: None,
            artifacts: Vec::new(),
            execution_artifact: ExecutionArtifact::new(),
            progress_score: None,
            best_error_count: None,
        });
    }

    /// Record agent prompt wall-clock time.
    pub fn record_agent_time(&mut self, duration: Duration) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.agent_response_ms = duration.as_millis() as u64;
        }
    }

    /// Record agent model and token usage.
    pub fn record_agent_metrics(
        &mut self,
        model: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
    ) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.agent_model = model.to_string();
            iter.agent_prompt_tokens = prompt_tokens;
            iter.agent_completion_tokens = completion_tokens;
        }
    }

    /// Record verifier pipeline wall-clock time.
    pub fn record_verifier_time(&mut self, duration: Duration) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.verifier_ms = duration.as_millis() as u64;
        }
    }

    /// Record verifier results for this iteration.
    pub fn record_verifier_results(&mut self, error_count: usize, categories: Vec<String>) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.error_count = error_count;
            iter.error_categories = categories;
        }
    }

    /// Record that this iteration produced no file changes.
    pub fn record_no_change(&mut self) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.no_change = true;
        }
    }

    /// Record that auto-fix was applied this iteration.
    pub fn record_auto_fix(&mut self) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.auto_fix_applied = true;
        }
    }

    /// Record regression detection this iteration.
    pub fn record_regression(&mut self, rolled_back: bool) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.regression_detected = true;
            iter.rollback_performed = rolled_back;
        }
    }

    /// Record an escalation event this iteration.
    pub fn record_escalation(&mut self) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.escalated = true;
        }
    }

    /// Record which coder was routed to this iteration.
    pub fn record_coder_route(&mut self, route: &str) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.coder_route = Some(route.to_string());
        }
    }

    /// Record the hill-climbing progress score for this iteration.
    pub fn record_progress_score(&mut self, error_count: usize, best_error_count: usize) {
        if let Some(ref mut iter) = self.current_iteration {
            let score = if best_error_count == 0 {
                1.0
            } else {
                1.0 - (error_count as f64 / best_error_count as f64)
            };
            iter.progress_score = Some(score);
            iter.best_error_count = Some(best_error_count);
        }
    }

    /// Record a file artifact touched during this iteration.
    pub fn record_artifact(&mut self, artifact: ArtifactRecord) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.artifacts.push(artifact);
        }
    }

    /// Record the routing decision for this iteration.
    pub fn record_route_decision(&mut self, decision: RouteDecision) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.execution_artifact.route_decision = Some(decision);
        }
    }

    /// Record the verifier snapshot for this iteration.
    pub fn record_verifier_snapshot(&mut self, snapshot: VerifierSnapshot) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.execution_artifact.verifier_snapshot = Some(snapshot);
        }
    }

    /// Record the evaluator (cloud validator) snapshot for this iteration.
    pub fn record_evaluator_snapshot(&mut self, snapshot: EvaluatorSnapshot) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.execution_artifact.evaluator_snapshot = Some(snapshot);
        }
    }

    /// Record the retry/escalate rationale for this iteration.
    pub fn record_retry_rationale(&mut self, rationale: RetryRationale) {
        if let Some(ref mut iter) = self.current_iteration {
            iter.execution_artifact.retry_rationale = Some(rationale);
        }
    }

    /// Finish the current iteration and store its metrics.
    pub fn finish_iteration(&mut self) {
        if let Some(iter) = self.current_iteration.take() {
            // Only attach the artifact if any decision was recorded
            let artifact = if iter.execution_artifact.route_decision.is_some()
                || iter.execution_artifact.verifier_snapshot.is_some()
                || iter.execution_artifact.evaluator_snapshot.is_some()
                || iter.execution_artifact.retry_rationale.is_some()
            {
                Some(iter.execution_artifact)
            } else {
                None
            };

            self.iterations.push(IterationMetrics {
                iteration: iter.iteration,
                tier: iter.tier,
                agent_model: iter.agent_model,
                agent_prompt_tokens: iter.agent_prompt_tokens,
                agent_completion_tokens: iter.agent_completion_tokens,
                agent_response_ms: iter.agent_response_ms,
                verifier_ms: iter.verifier_ms,
                error_count: iter.error_count,
                error_categories: iter.error_categories,
                no_change: iter.no_change,
                auto_fix_applied: iter.auto_fix_applied,
                regression_detected: iter.regression_detected,
                rollback_performed: iter.rollback_performed,
                escalated: iter.escalated,
                coder_route: iter.coder_route,
                artifacts: iter.artifacts,
                execution_artifact: artifact,
                progress_score: iter.progress_score,
                best_error_count: iter.best_error_count,
            });
        }
    }

    /// Record cloud validation results.
    pub fn record_cloud_validation(&mut self, model: &str, passed: bool) {
        self.cloud_validations.push(ValidationMetric {
            model: model.to_string(),
            passed,
        });
    }

    /// Record local validation results.
    pub fn record_local_validation(&mut self, model: &str, passed: bool) {
        self.local_validations.push(ValidationMetric {
            model: model.to_string(),
            passed,
        });
    }

    /// Build a `LoopMetrics` snapshot from the current in-progress iteration.
    ///
    /// Returns `None` if no iteration is in progress.
    pub fn build_loop_metrics(&self, all_green: bool) -> Option<LoopMetrics> {
        self.current_iteration.as_ref().map(|iter| LoopMetrics {
            iteration: iter.iteration,
            tier: iter.tier.clone(),
            agent_ms: iter.agent_response_ms,
            verifier_ms: iter.verifier_ms,
            error_count: iter.error_count,
            all_green,
            escalated: iter.escalated,
            no_change: iter.no_change,
        })
    }

    /// Finalize and produce the complete session metrics.
    pub fn finalize(mut self, success: bool, final_tier: &str) -> SessionMetrics {
        // Flush any in-progress iteration
        self.finish_iteration();

        let total = self.iterations.len() as u32;
        let no_change_count = self.iterations.iter().filter(|i| i.no_change).count() as u32;
        let no_change_rate = if total > 0 {
            no_change_count as f64 / total as f64
        } else {
            0.0
        };

        let turns_until_first_write = self
            .iterations
            .iter()
            .find(|i| {
                i.artifacts.iter().any(|a| {
                    a.action == ArtifactAction::Modified || a.action == ArtifactAction::Created
                })
            })
            .map(|i| i.iteration);

        let write_by_turn_2 = self
            .iterations
            .iter()
            .take(2)
            .any(|i| {
                i.artifacts.iter().any(|a| {
                    a.action == ArtifactAction::Modified || a.action == ArtifactAction::Created
                })
            });

        SessionMetrics {
            session_id: self.session_id,
            issue_id: self.issue_id,
            issue_title: self.issue_title,
            success,
            total_iterations: total,
            final_tier: final_tier.to_string(),
            elapsed_ms: self.session_start.elapsed().as_millis() as u64,
            total_no_change_iterations: no_change_count,
            no_change_rate,
            cloud_validations: self.cloud_validations,
            local_validations: self.local_validations,
            iterations: self.iterations,
            timestamp: chrono::Utc::now().to_rfc3339(),
            stack_profile: self.stack_profile,
            repo_id: self.repo_id,
            adapter_id: self.adapter_id,
            turns_until_first_write,
            write_by_turn_2,
            role_map_version: self.role_map_version,
        }
    }
}

/// Write session metrics to `.swarm-metrics.json` in the worktree.
pub fn write_session_metrics(metrics: &SessionMetrics, wt_path: &Path) {
    let path = wt_path.join(".swarm-metrics.json");
    match serde_json::to_string_pretty(metrics) {
        Ok(json) => match std::fs::write(&path, json) {
            Ok(()) => info!(path = %path.display(), "Wrote session metrics"),
            Err(e) => warn!("Failed to write session metrics: {e}"),
        },
        Err(e) => warn!("Failed to serialize session metrics: {e}"),
    }
}

/// Append session metrics to `.swarm-telemetry.jsonl` in the repo root.
///
/// Each line is a complete JSON object (JSONL format) for easy streaming analysis.
pub fn append_telemetry(metrics: &SessionMetrics, repo_root: &Path) {
    let path = repo_root.join(".swarm-telemetry.jsonl");
    match serde_json::to_string(metrics) {
        Ok(json) => {
            use std::io::Write;
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(mut file) => {
                    if let Err(e) = writeln!(file, "{json}") {
                        warn!("Failed to append telemetry: {e}");
                    } else {
                        info!(path = %path.display(), "Appended session telemetry");
                    }
                }
                Err(e) => warn!("Failed to open telemetry file: {e}"),
            }
        }
        Err(e) => warn!("Failed to serialize telemetry: {e}"),
    }
}

/// Append a row to `experiments.tsv` in the worktree.
///
/// Each row captures a single iteration decision point for trajectory analysis.
/// Header is written on first call. The TSV format enables easy `sort | uniq -c`
/// analysis without JSON parsing.
pub fn append_experiment_tsv(
    worktree_path: &Path,
    commit: &str,
    error_count: usize,
    gates_passed: &[&str],
    status: &str,
    description: &str,
) {
    use std::io::Write;
    let tsv_path = worktree_path.join("experiments.tsv");
    let needs_header = !tsv_path.exists();

    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&tsv_path)
    {
        Ok(mut file) => {
            if needs_header {
                let _ = writeln!(
                    file,
                    "timestamp\tcommit\terror_count\tgates_passed\tstatus\tdescription"
                );
            }
            let ts = chrono::Utc::now().to_rfc3339();
            let gates = gates_passed.join(",");
            let _ = writeln!(
                file,
                "{ts}\t{commit}\t{error_count}\t{gates}\t{status}\t{description}"
            );
        }
        Err(e) => warn!("Failed to write experiments.tsv: {e}"),
    }
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

/// Append an entry to `.swarm-failure-ledger.jsonl` in the worktree.
pub fn append_failure_ledger(worktree_path: &Path, entry: &FailureLedgerEntry) {
    use std::io::Write;
    let path = worktree_path.join(".swarm-failure-ledger.jsonl");
    match serde_json::to_string(entry) {
        Ok(json) => {
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(mut file) => {
                    if let Err(e) = writeln!(file, "{json}") {
                        warn!("Failed to append failure ledger: {e}");
                    }
                }
                Err(e) => warn!("Failed to open failure ledger: {e}"),
            }
        }
        Err(e) => warn!("Failed to serialize failure ledger entry: {e}"),
    }
}

/// Write execution artifacts from session metrics to `.swarm-artifacts/` directory.
///
/// Creates one JSON file per iteration: `iteration-001.json`, `iteration-002.json`, etc.
/// Only writes files for iterations that have execution artifacts attached.
/// Supports retention: if `max_sessions` is set, prunes oldest session directories.
pub fn write_execution_artifacts(
    metrics: &SessionMetrics,
    wt_path: &Path,
    max_sessions: Option<usize>,
) {
    let artifacts_dir = wt_path.join(".swarm-artifacts").join(&metrics.session_id);

    // Create the session directory
    if let Err(e) = std::fs::create_dir_all(&artifacts_dir) {
        warn!("Failed to create artifacts directory: {e}");
        return;
    }

    let mut written = 0usize;
    for iter_metrics in &metrics.iterations {
        if let Some(ref artifact) = iter_metrics.execution_artifact {
            let filename = format!("iteration-{:03}.json", iter_metrics.iteration);
            let path = artifacts_dir.join(&filename);
            match serde_json::to_string_pretty(artifact) {
                Ok(json) => match std::fs::write(&path, json) {
                    Ok(()) => written += 1,
                    Err(e) => warn!("Failed to write artifact {filename}: {e}"),
                },
                Err(e) => warn!("Failed to serialize artifact {filename}: {e}"),
            }
        }
    }

    if written > 0 {
        info!(
            path = %artifacts_dir.display(),
            count = written,
            "Wrote execution artifacts"
        );
    }

    // Retention: prune old session directories if max_sessions is set
    if let Some(max) = max_sessions {
        let parent = wt_path.join(".swarm-artifacts");
        prune_artifact_sessions(&parent, max);
    }
}

/// Remove oldest session artifact directories to stay within the retention limit.
fn prune_artifact_sessions(artifacts_root: &Path, max_sessions: usize) {
    let entries: Vec<_> = match std::fs::read_dir(artifacts_root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .collect(),
        Err(_) => return,
    };

    if entries.len() <= max_sessions {
        return;
    }

    // Sort by modification time (oldest first)
    let mut sorted: Vec<_> = entries
        .into_iter()
        .filter_map(|e| {
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((mtime, e.path()))
        })
        .collect();
    sorted.sort_by_key(|(mtime, _)| *mtime);

    let to_remove = sorted.len() - max_sessions;
    for (_, path) in sorted.into_iter().take(to_remove) {
        if let Err(e) = std::fs::remove_dir_all(&path) {
            warn!("Failed to prune artifact session {}: {e}", path.display());
        } else {
            info!(path = %path.display(), "Pruned old artifact session");
        }
    }
}

/// Compute the churn score for a set of artifact records.
///
/// Churn score is the ratio of modification actions (Modified + Deleted) to
/// total artifact touches. A score of 1.0 means every touched file was
/// modified or deleted; 0.0 means only reads and creates occurred.
///
/// Returns 0.0 when `artifacts` is empty.
pub fn artifact_churn_score(artifacts: &[ArtifactRecord]) -> f64 {
    if artifacts.is_empty() {
        return 0.0;
    }
    let modifications = artifacts
        .iter()
        .filter(|a| matches!(a.action, ArtifactAction::Modified | ArtifactAction::Deleted))
        .count();
    modifications as f64 / artifacts.len() as f64
}

/// Aggregate analytics computed from multiple sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateAnalytics {
    pub total_sessions: usize,
    pub success_rate: f64,
    pub average_iterations: f64,
    pub average_elapsed_ms: f64,
    pub total_prompt_tokens: u64,
    pub total_completion_tokens: u64,
    pub error_category_frequencies: std::collections::HashMap<String, usize>,
}

/// Per-iteration loop metrics emitted as a structured tracing event.
///
/// Compatible with OpenTelemetry exporters via `tracing-opentelemetry`.
#[derive(Debug, Clone)]
pub struct LoopMetrics {
    pub iteration: u32,
    pub tier: String,
    pub agent_ms: u64,
    pub verifier_ms: u64,
    pub error_count: usize,
    pub all_green: bool,
    pub escalated: bool,
    pub no_change: bool,
}

impl LoopMetrics {
    /// Emit this as a structured tracing event (OpenTelemetry-compatible).
    pub fn emit(&self) {
        tracing::info!(
            target: "swarm.metrics",
            iteration = self.iteration,
            tier = %self.tier,
            agent_ms = self.agent_ms,
            verifier_ms = self.verifier_ms,
            error_count = self.error_count,
            all_green = self.all_green,
            escalated = self.escalated,
            no_change = self.no_change,
            "loop_iteration_complete"
        );
    }
}

/// Reads and analyzes telemetry data from `.swarm-telemetry.jsonl` files.
pub struct TelemetryReader {
    sessions: Vec<SessionMetrics>,
}

impl TelemetryReader {
    /// Read telemetry sessions from a JSONL file.
    pub fn read_from_file(path: &Path) -> std::io::Result<Self> {
        use std::fs::File;
        use std::io::{BufRead, BufReader};

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut sessions = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let session: SessionMetrics = serde_json::from_str(&line)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            sessions.push(session);
        }

        Ok(Self { sessions })
    }

    /// Get the parsed sessions.
    pub fn sessions(&self) -> &[SessionMetrics] {
        &self.sessions
    }

    /// Compute aggregate analytics across all loaded sessions.
    pub fn aggregate_analytics(&self) -> AggregateAnalytics {
        let total_sessions = self.sessions.len();
        if total_sessions == 0 {
            return AggregateAnalytics {
                total_sessions: 0,
                success_rate: 0.0,
                average_iterations: 0.0,
                average_elapsed_ms: 0.0,
                total_prompt_tokens: 0,
                total_completion_tokens: 0,
                error_category_frequencies: std::collections::HashMap::new(),
            };
        }

        let mut successful_sessions = 0;
        let mut total_iterations = 0;
        let mut total_elapsed_ms = 0;
        let mut total_prompt_tokens = 0;
        let mut total_completion_tokens = 0;
        let mut error_category_frequencies = std::collections::HashMap::new();

        for session in &self.sessions {
            if session.success {
                successful_sessions += 1;
            }
            total_iterations += session.total_iterations;
            total_elapsed_ms += session.elapsed_ms;

            for iter in &session.iterations {
                total_prompt_tokens += iter.agent_prompt_tokens as u64;
                total_completion_tokens += iter.agent_completion_tokens as u64;

                for cat in &iter.error_categories {
                    *error_category_frequencies.entry(cat.clone()).or_insert(0) += 1;
                }
            }
        }

        AggregateAnalytics {
            total_sessions,
            success_rate: successful_sessions as f64 / total_sessions as f64,
            average_iterations: total_iterations as f64 / total_sessions as f64,
            average_elapsed_ms: total_elapsed_ms as f64 / total_sessions as f64,
            total_prompt_tokens,
            total_completion_tokens,
            error_category_frequencies,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// SLO (Service Level Objective) definitions and computation
// ──────────────────────────────────────────────────────────────────────────────

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

impl TelemetryReader {
    /// Compute SLO measurements against the given targets.
    pub fn compute_slos(&self, targets: &SloTargets) -> SloReport {
        let sessions = &self.sessions;
        let count = sessions.len();
        if count == 0 {
            return SloReport {
                session_count: 0,
                measurements: vec![],
                overall_status: SloStatus::Met,
            };
        }

        let success_count = sessions.iter().filter(|s| s.success).count();
        let success_rate = success_count as f64 / count as f64;

        // Average iterations-to-green (only for successful sessions)
        let successful: Vec<&SessionMetrics> = sessions.iter().filter(|s| s.success).collect();
        let avg_iters = if successful.is_empty() {
            f64::MAX
        } else {
            successful
                .iter()
                .map(|s| s.total_iterations as f64)
                .sum::<f64>()
                / successful.len() as f64
        };

        // Stuck rate: sessions that failed AND had max iterations used
        let stuck_count = sessions
            .iter()
            .filter(|s| !s.success && s.total_iterations >= 6)
            .count();
        let stuck_rate = stuck_count as f64 / count as f64;

        // No-change rate: average across all sessions
        let no_change_rate = sessions.iter().map(|s| s.no_change_rate).sum::<f64>() / count as f64;

        let mut measurements = vec![
            measure(
                "success_rate",
                success_rate,
                targets.success_rate,
                targets.success_rate * 0.9,
                true,
            ),
            measure(
                "avg_iterations_to_green",
                avg_iters,
                targets.avg_iterations_to_green,
                targets.avg_iterations_to_green * 1.5,
                false,
            ),
            measure(
                "stuck_rate",
                stuck_rate,
                targets.stuck_rate,
                targets.stuck_rate * 1.5,
                false,
            ),
            measure(
                "no_change_rate",
                no_change_rate,
                targets.no_change_rate,
                targets.no_change_rate * 1.5,
                false,
            ),
        ];

        let overall_status = measurements
            .iter()
            .map(|m| m.status)
            .max_by_key(|s| match s {
                SloStatus::Met => 0,
                SloStatus::Warning => 1,
                SloStatus::Breached => 2,
            })
            .unwrap_or(SloStatus::Met);

        // Sort to put breached first for quick triage.
        measurements.sort_by_key(|m| match m.status {
            SloStatus::Breached => 0,
            SloStatus::Warning => 1,
            SloStatus::Met => 2,
        });

        SloReport {
            session_count: count,
            measurements,
            overall_status,
        }
    }
}

/// Build a single SLO measurement.
///
/// `higher_is_better`: if true, value >= target is Met; if false, value <= target is Met.
fn measure(
    name: &str,
    value: f64,
    target: f64,
    warning: f64,
    higher_is_better: bool,
) -> SloMeasurement {
    let status = if higher_is_better {
        if value >= target {
            SloStatus::Met
        } else if value >= warning {
            SloStatus::Warning
        } else {
            SloStatus::Breached
        }
    } else {
        // Lower is better (stuck_rate, no_change_rate, iterations)
        if value <= target {
            SloStatus::Met
        } else if value <= warning {
            SloStatus::Warning
        } else {
            SloStatus::Breached
        }
    };
    SloMeasurement {
        name: name.to_string(),
        value,
        target,
        warning,
        status,
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Cost tracking and budget enforcement
// ──────────────────────────────────────────────────────────────────────────────

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

/// Tracks estimated token costs across iterations and enforces a per-issue budget.
#[derive(Debug, Clone)]
pub struct CostTracker {
    /// Maximum allowed cost in USD (0.0 = unlimited).
    budget: f64,
    /// Accumulated estimated cost in USD.
    accumulated: f64,
    /// Total prompt tokens across all iterations.
    total_prompt_tokens: u64,
    /// Total completion tokens across all iterations.
    total_completion_tokens: u64,
}

impl CostTracker {
    /// Create a new cost tracker with the given budget.
    ///
    /// A budget of 0.0 means no cost limit (unlimited).
    pub fn new(budget: f64) -> Self {
        Self {
            budget,
            accumulated: 0.0,
            total_prompt_tokens: 0,
            total_completion_tokens: 0,
        }
    }

    /// Record token usage for an iteration and update estimated cost.
    ///
    /// `is_cloud` determines which pricing tier to apply.
    pub fn record_usage(&mut self, prompt_tokens: u32, completion_tokens: u32, is_cloud: bool) {
        self.total_prompt_tokens += prompt_tokens as u64;
        self.total_completion_tokens += completion_tokens as u64;

        let (input_rate, output_rate) = if is_cloud {
            (
                cost_rates::CLOUD_INPUT_PER_M,
                cost_rates::CLOUD_OUTPUT_PER_M,
            )
        } else {
            (
                cost_rates::LOCAL_INPUT_PER_M,
                cost_rates::LOCAL_OUTPUT_PER_M,
            )
        };

        self.accumulated += (prompt_tokens as f64 / 1_000_000.0) * input_rate
            + (completion_tokens as f64 / 1_000_000.0) * output_rate;
    }

    /// Check if the accumulated cost exceeds the budget.
    ///
    /// Returns `Some(reason)` if over budget, `None` if within budget or unlimited.
    pub fn check_budget(&self) -> Option<String> {
        if self.budget > 0.0 && self.accumulated >= self.budget {
            Some(format!(
                "cost budget exceeded: ${:.4} >= ${:.4} limit",
                self.accumulated, self.budget
            ))
        } else {
            None
        }
    }

    /// Current accumulated cost in USD.
    pub fn accumulated_cost(&self) -> f64 {
        self.accumulated
    }

    /// Total prompt tokens recorded.
    pub fn total_prompt_tokens(&self) -> u64 {
        self.total_prompt_tokens
    }

    /// Total completion tokens recorded.
    pub fn total_completion_tokens(&self) -> u64 {
        self.total_completion_tokens
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Context pruning for cost optimization
// ──────────────────────────────────────────────────────────────────────────────

/// Prune a task prompt to reduce token usage after multiple iterations.
///
/// After `prune_after` iterations, the prompt is trimmed to keep only:
/// - The original task description (first paragraph)
/// - The last `keep_recent` iteration results
/// - The latest verifier output
///
/// Returns the original prompt unchanged if pruning is not applicable.
pub fn prune_task_prompt(
    prompt: &str,
    current_iteration: u32,
    prune_after: u32,
    keep_recent: usize,
) -> String {
    if current_iteration <= prune_after {
        return prompt.to_string();
    }

    let sections: Vec<&str> = prompt.split("\n---\n").collect();
    if sections.len() <= keep_recent + 1 {
        return prompt.to_string();
    }

    // Keep: first section (task description) + last `keep_recent` sections
    let mut pruned = Vec::with_capacity(keep_recent + 2);
    pruned.push(sections[0]);
    pruned.push("[Earlier iterations pruned for context efficiency]");
    for section in sections.iter().rev().take(keep_recent).rev() {
        pruned.push(section);
    }
    pruned.join("\n---\n")
}

// ──────────────────────────────────────────────────────────────────────────────
// Structured event emitter for real-time observability
// ──────────────────────────────────────────────────────────────────────────────

/// A structured swarm event emitted in real-time during orchestration.
///
/// Each event is a self-contained JSON record written to `telemetry.jsonl` and
/// optionally POSTed to a webhook URL. This supplements the batch
/// `SessionMetrics` with live, per-action granularity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmEvent {
    /// Event type (dot-notation, e.g. `swarm.issue.started`).
    pub event: String,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Issue ID this event relates to.
    pub issue_id: String,
    /// Typed payload.
    #[serde(flatten)]
    pub payload: SwarmEventPayload,
}

/// Typed payloads for each event kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SwarmEventPayload {
    IssueStarted {
        title: String,
        priority: Option<u8>,
        tier: String,
    },
    IterationCompleted {
        iteration: u32,
        tier: String,
        error_count: usize,
        no_change: bool,
        elapsed_ms: u64,
    },
    WorkerFailed {
        model: String,
        error: String,
        iteration: u32,
    },
    HealthCheck {
        endpoint: String,
        healthy: bool,
        latency_ms: Option<u64>,
    },
    IssueResolved {
        success: bool,
        total_iterations: u32,
        elapsed_ms: u64,
    },
}

/// Emits structured events to a JSONL file and optional webhook.
pub struct SwarmEventEmitter {
    /// Path to the JSONL event log (typically `<repo_root>/.swarm-events.jsonl`).
    event_log_path: std::path::PathBuf,
    /// Optional webhook URL for critical events.
    webhook_url: Option<String>,
    /// HTTP client for webhook delivery (reused across calls).
    http_client: Option<reqwest::Client>,
}

impl SwarmEventEmitter {
    /// Create a new emitter writing to the given repo root.
    ///
    /// Reads `SWARM_WEBHOOK_URL` from the environment for webhook delivery.
    pub fn new(repo_root: &Path) -> Self {
        let webhook_url = std::env::var("SWARM_WEBHOOK_URL")
            .ok()
            .filter(|u| !u.is_empty());
        let http_client = webhook_url.as_ref().map(|_| reqwest::Client::new());
        Self {
            event_log_path: repo_root.join(".swarm-events.jsonl"),
            webhook_url,
            http_client,
        }
    }

    /// Emit a structured event. Writes to the event log and optionally fires a webhook.
    pub fn emit(&self, event: SwarmEvent) {
        // Write to JSONL
        if let Ok(json) = serde_json::to_string(&event) {
            use std::io::Write;
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.event_log_path)
            {
                Ok(mut file) => {
                    if let Err(e) = writeln!(file, "{json}") {
                        tracing::warn!(
                            error = %e,
                            path = %self.event_log_path.display(),
                            "Failed to write JSONL event"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        path = %self.event_log_path.display(),
                        "Failed to open JSONL event log"
                    );
                }
            }

            // Also emit as a structured tracing event for log aggregation
            tracing::info!(
                target: "swarm.events",
                event_type = %event.event,
                issue_id = %event.issue_id,
                "structured_event"
            );
        }

        // Webhook delivery for critical events (non-blocking fire-and-forget)
        if let (Some(url), Some(client)) = (&self.webhook_url, &self.http_client) {
            if Self::is_critical(&event) {
                if tokio::runtime::Handle::try_current().is_ok() {
                    let url = url.clone();
                    let client = client.clone();
                    let event_clone = event;
                    tokio::spawn(async move {
                        let _ = client
                            .post(&url)
                            .json(&event_clone)
                            .timeout(std::time::Duration::from_secs(5))
                            .send()
                            .await;
                    });
                } else {
                    tracing::debug!("No Tokio runtime — skipping webhook delivery");
                }
            }
        }
    }

    /// Helper to emit an issue-started event.
    pub fn issue_started(&self, issue_id: &str, title: &str, priority: Option<u8>, tier: &str) {
        self.emit(SwarmEvent {
            event: "swarm.issue.started".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::IssueStarted {
                title: title.into(),
                priority,
                tier: tier.into(),
            },
        });
    }

    /// Helper to emit an iteration-completed event.
    pub fn iteration_completed(
        &self,
        issue_id: &str,
        iteration: u32,
        tier: &str,
        error_count: usize,
        no_change: bool,
        elapsed_ms: u64,
    ) {
        self.emit(SwarmEvent {
            event: "swarm.iteration.completed".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::IterationCompleted {
                iteration,
                tier: tier.into(),
                error_count,
                no_change,
                elapsed_ms,
            },
        });
    }

    /// Helper to emit a worker-failed event.
    pub fn worker_failed(&self, issue_id: &str, model: &str, error: &str, iteration: u32) {
        self.emit(SwarmEvent {
            event: "swarm.worker.failed".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::WorkerFailed {
                model: model.into(),
                error: error.into(),
                iteration,
            },
        });
    }

    /// Helper to emit a health-check event.
    pub fn health_check(
        &self,
        issue_id: &str,
        endpoint: &str,
        healthy: bool,
        latency_ms: Option<u64>,
    ) {
        self.emit(SwarmEvent {
            event: "swarm.health.check".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::HealthCheck {
                endpoint: endpoint.into(),
                healthy,
                latency_ms,
            },
        });
    }

    /// Helper to emit an issue-resolved event.
    pub fn issue_resolved(
        &self,
        issue_id: &str,
        success: bool,
        total_iterations: u32,
        elapsed_ms: u64,
    ) {
        self.emit(SwarmEvent {
            event: "swarm.issue.resolved".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::IssueResolved {
                success,
                total_iterations,
                elapsed_ms,
            },
        });
    }

    /// Whether an event is critical enough to warrant webhook delivery.
    fn is_critical(event: &SwarmEvent) -> bool {
        matches!(
            event.payload,
            SwarmEventPayload::WorkerFailed { .. }
                | SwarmEventPayload::IssueResolved { .. }
                | SwarmEventPayload::HealthCheck { healthy: false, .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_metrics_collector_basic_flow() {
        let mut collector = MetricsCollector::new("sess-1", "issue-1", "Fix bug");

        collector.start_iteration(1, "Integrator");
        collector.record_agent_time(Duration::from_secs(30));
        collector.record_agent_metrics("test-model-1", 100, 50);
        collector.record_verifier_time(Duration::from_secs(45));
        collector.record_verifier_results(3, vec!["BorrowChecker".into(), "Lifetime".into()]);
        collector.record_coder_route("RustCoder");
        collector.finish_iteration();

        collector.start_iteration(2, "Integrator");
        collector.record_agent_time(Duration::from_secs(25));
        collector.record_agent_metrics("test-model-2", 80, 40);
        collector.record_verifier_time(Duration::from_secs(40));
        collector.record_verifier_results(0, vec![]);
        collector.record_auto_fix();
        collector.finish_iteration();

        collector.record_cloud_validation("gemini-3-pro-preview", true);
        collector.record_cloud_validation("claude-sonnet-4-5", true);

        let metrics = collector.finalize(true, "Integrator");

        assert_eq!(metrics.session_id, "sess-1");
        assert_eq!(metrics.issue_id, "issue-1");
        assert!(metrics.success);
        assert_eq!(metrics.total_iterations, 2);
        assert_eq!(metrics.iterations.len(), 2);
        assert_eq!(metrics.iterations[0].error_count, 3);
        assert_eq!(metrics.iterations[0].agent_response_ms, 30_000);
        assert_eq!(metrics.iterations[0].agent_model, "test-model-1");
        assert_eq!(metrics.iterations[0].agent_prompt_tokens, 100);
        assert_eq!(metrics.iterations[0].agent_completion_tokens, 50);
        assert_eq!(metrics.iterations[1].error_count, 0);
        assert_eq!(metrics.iterations[1].agent_model, "test-model-2");
        assert_eq!(metrics.iterations[1].agent_prompt_tokens, 80);
        assert_eq!(metrics.iterations[1].agent_completion_tokens, 40);
        assert!(metrics.iterations[1].auto_fix_applied);
        assert_eq!(metrics.cloud_validations.len(), 2);
    }

    #[test]
    fn test_metrics_collector_regression_tracking() {
        let mut collector = MetricsCollector::new("sess-2", "issue-2", "Fix regression");

        collector.start_iteration(1, "Implementer");
        collector.record_regression(true);
        collector.record_escalation();
        collector.finish_iteration();

        let metrics = collector.finalize(false, "Cloud");

        assert!(!metrics.success);
        assert!(metrics.iterations[0].regression_detected);
        assert!(metrics.iterations[0].rollback_performed);
        assert!(metrics.iterations[0].escalated);
    }

    #[test]
    fn test_finalize_flushes_in_progress_iteration() {
        let mut collector = MetricsCollector::new("sess-3", "issue-3", "Test flush");

        collector.start_iteration(1, "Integrator");
        collector.record_agent_time(Duration::from_secs(10));
        // Don't call finish_iteration — finalize should flush it

        let metrics = collector.finalize(true, "Integrator");
        assert_eq!(metrics.total_iterations, 1);
        assert_eq!(metrics.iterations[0].agent_response_ms, 10_000);
    }

    #[test]
    fn test_write_session_metrics_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = SessionMetrics {
            session_id: "test-session".into(),
            issue_id: "test-issue".into(),
            issue_title: "Test".into(),
            success: true,
            total_iterations: 1,
            final_tier: "Integrator".into(),
            elapsed_ms: 5000,
            total_no_change_iterations: 0,
            no_change_rate: 0.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2024-01-01T00:00:00Z".into(),
        };

        write_session_metrics(&metrics, dir.path());

        let path = dir.path().join(".swarm-metrics.json");
        assert!(path.exists());

        let contents = std::fs::read_to_string(&path).unwrap();
        let loaded: SessionMetrics = serde_json::from_str(&contents).unwrap();
        assert_eq!(loaded.session_id, "test-session");
        assert!(loaded.success);
    }

    #[test]
    fn test_append_telemetry_jsonl() {
        let dir = tempfile::tempdir().unwrap();

        let metrics1 = SessionMetrics {
            session_id: "sess-1".into(),
            issue_id: "issue-1".into(),
            issue_title: "First".into(),
            success: true,
            total_iterations: 1,
            final_tier: "Integrator".into(),
            elapsed_ms: 3000,
            total_no_change_iterations: 0,
            no_change_rate: 0.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2024-01-01T00:00:00Z".into(),
        };
        let metrics2 = SessionMetrics {
            session_id: "sess-2".into(),
            issue_id: "issue-2".into(),
            issue_title: "Second".into(),
            success: false,
            total_iterations: 3,
            final_tier: "Cloud".into(),
            elapsed_ms: 15000,
            total_no_change_iterations: 1,
            no_change_rate: 1.0 / 3.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2024-01-01T01:00:00Z".into(),
        };

        append_telemetry(&metrics1, dir.path());
        append_telemetry(&metrics2, dir.path());

        let path = dir.path().join(".swarm-telemetry.jsonl");
        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let loaded1: SessionMetrics = serde_json::from_str(lines[0]).unwrap();
        let loaded2: SessionMetrics = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(loaded1.session_id, "sess-1");
        assert_eq!(loaded2.session_id, "sess-2");
        assert!(loaded1.success);
        assert!(!loaded2.success);
    }

    #[test]
    fn test_telemetry_reader_and_analytics() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".swarm-telemetry.jsonl");

        let mut metrics1 = SessionMetrics {
            session_id: "sess-1".into(),
            issue_id: "issue-1".into(),
            issue_title: "First".into(),
            success: true,
            total_iterations: 2,
            final_tier: "Integrator".into(),
            elapsed_ms: 3000,
            total_no_change_iterations: 0,
            no_change_rate: 0.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2024-01-01T00:00:00Z".into(),
        };
        metrics1.iterations.push(IterationMetrics {
            iteration: 1,
            tier: "Integrator".into(),
            agent_model: "model-1".into(),
            agent_prompt_tokens: 100,
            agent_completion_tokens: 50,
            agent_response_ms: 1000,
            verifier_ms: 500,
            error_count: 1,
            error_categories: vec!["Syntax".into()],
            no_change: false,
            auto_fix_applied: false,
            regression_detected: false,
            rollback_performed: false,
            escalated: false,
            coder_route: None,
            artifacts: vec![],
            execution_artifact: None,
            progress_score: None,
            best_error_count: None,
        });
        metrics1.iterations.push(IterationMetrics {
            iteration: 2,
            tier: "Integrator".into(),
            agent_model: "model-1".into(),
            agent_prompt_tokens: 120,
            agent_completion_tokens: 60,
            agent_response_ms: 1200,
            verifier_ms: 600,
            error_count: 0,
            error_categories: vec![],
            no_change: false,
            auto_fix_applied: false,
            regression_detected: false,
            rollback_performed: false,
            escalated: false,
            coder_route: None,
            artifacts: vec![],
            execution_artifact: None,
            progress_score: None,
            best_error_count: None,
        });

        let mut metrics2 = SessionMetrics {
            session_id: "sess-2".into(),
            issue_id: "issue-2".into(),
            issue_title: "Second".into(),
            success: false,
            total_iterations: 1,
            final_tier: "Cloud".into(),
            elapsed_ms: 15000,
            total_no_change_iterations: 0,
            no_change_rate: 0.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2024-01-01T01:00:00Z".into(),
        };
        metrics2.iterations.push(IterationMetrics {
            iteration: 1,
            tier: "Cloud".into(),
            agent_model: "model-2".into(),
            agent_prompt_tokens: 200,
            agent_completion_tokens: 100,
            agent_response_ms: 2000,
            verifier_ms: 1000,
            error_count: 2,
            error_categories: vec!["Syntax".into(), "Type".into()],
            no_change: false,
            auto_fix_applied: false,
            regression_detected: false,
            rollback_performed: false,
            escalated: false,
            coder_route: None,
            artifacts: vec![],
            execution_artifact: None,
            progress_score: None,
            best_error_count: None,
        });

        append_telemetry(&metrics1, dir.path());
        append_telemetry(&metrics2, dir.path());

        let reader = TelemetryReader::read_from_file(&path).unwrap();
        assert_eq!(reader.sessions().len(), 2);

        let analytics = reader.aggregate_analytics();
        assert_eq!(analytics.total_sessions, 2);
        assert_eq!(analytics.success_rate, 0.5); // 1 success out of 2
        assert_eq!(analytics.average_iterations, 1.5); // (2 + 1) / 2
        assert_eq!(analytics.average_elapsed_ms, 9000.0); // (3000 + 15000) / 2
        assert_eq!(analytics.total_prompt_tokens, 420); // 100 + 120 + 200
        assert_eq!(analytics.total_completion_tokens, 210); // 50 + 60 + 100

        let mut expected_errors = std::collections::HashMap::new();
        expected_errors.insert("Syntax".to_string(), 2);
        expected_errors.insert("Type".to_string(), 1);
        assert_eq!(analytics.error_category_frequencies, expected_errors);
    }

    #[test]
    fn test_artifact_churn_score_empty() {
        assert_eq!(artifact_churn_score(&[]), 0.0);
    }

    #[test]
    fn test_artifact_churn_score_all_reads() {
        let artifacts = vec![
            ArtifactRecord {
                path: "src/lib.rs".into(),
                action: ArtifactAction::Read,
                line_range: None,
                size_delta: None,
            },
            ArtifactRecord {
                path: "src/main.rs".into(),
                action: ArtifactAction::Read,
                line_range: None,
                size_delta: None,
            },
        ];
        assert_eq!(artifact_churn_score(&artifacts), 0.0);
    }

    #[test]
    fn test_artifact_churn_score_all_modifications() {
        let artifacts = vec![
            ArtifactRecord {
                path: "src/lib.rs".into(),
                action: ArtifactAction::Modified,
                line_range: Some((1, 50)),
                size_delta: Some(100),
            },
            ArtifactRecord {
                path: "src/old.rs".into(),
                action: ArtifactAction::Deleted,
                line_range: None,
                size_delta: Some(-200),
            },
        ];
        assert_eq!(artifact_churn_score(&artifacts), 1.0);
    }

    #[test]
    fn test_artifact_churn_score_mixed() {
        // 2 modifications out of 4 total = 0.5
        let artifacts = vec![
            ArtifactRecord {
                path: "src/a.rs".into(),
                action: ArtifactAction::Read,
                line_range: None,
                size_delta: None,
            },
            ArtifactRecord {
                path: "src/b.rs".into(),
                action: ArtifactAction::Modified,
                line_range: Some((10, 20)),
                size_delta: Some(50),
            },
            ArtifactRecord {
                path: "src/c.rs".into(),
                action: ArtifactAction::Created,
                line_range: None,
                size_delta: Some(300),
            },
            ArtifactRecord {
                path: "src/d.rs".into(),
                action: ArtifactAction::Deleted,
                line_range: None,
                size_delta: Some(-150),
            },
        ];
        assert_eq!(artifact_churn_score(&artifacts), 0.5);
    }

    #[test]
    fn test_record_artifact_stored_in_iteration() {
        let mut collector = MetricsCollector::new("sess-art", "issue-art", "Artifact test");

        collector.start_iteration(1, "Worker");
        collector.record_artifact(ArtifactRecord {
            path: "src/foo.rs".into(),
            action: ArtifactAction::Modified,
            line_range: Some((1, 30)),
            size_delta: Some(42),
        });
        collector.record_artifact(ArtifactRecord {
            path: "src/bar.rs".into(),
            action: ArtifactAction::Created,
            line_range: None,
            size_delta: Some(100),
        });
        collector.finish_iteration();

        let metrics = collector.finalize(true, "Worker");
        assert_eq!(metrics.iterations[0].artifacts.len(), 2);
        assert_eq!(metrics.iterations[0].artifacts[0].path, "src/foo.rs");
        assert_eq!(
            metrics.iterations[0].artifacts[0].action,
            ArtifactAction::Modified
        );
        assert_eq!(metrics.iterations[0].artifacts[0].line_range, Some((1, 30)));
        assert_eq!(metrics.iterations[0].artifacts[0].size_delta, Some(42));
        assert_eq!(metrics.iterations[0].artifacts[1].path, "src/bar.rs");
        assert_eq!(
            metrics.iterations[0].artifacts[1].action,
            ArtifactAction::Created
        );
    }

    fn test_session(success: bool, iterations: u32, no_change_rate: f64) -> SessionMetrics {
        SessionMetrics {
            session_id: format!("sess-{}", iterations),
            issue_id: "issue-1".into(),
            issue_title: "Test".into(),
            success,
            total_iterations: iterations,
            final_tier: "Integrator".into(),
            elapsed_ms: iterations as u64 * 5000,
            total_no_change_iterations: 0,
            no_change_rate,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2026-01-01T00:00:00Z".into(),
        }
    }

    fn reader_from_sessions(sessions: Vec<SessionMetrics>) -> TelemetryReader {
        TelemetryReader { sessions }
    }

    #[test]
    fn test_slo_all_met() {
        let reader = reader_from_sessions(vec![
            test_session(true, 1, 0.0),
            test_session(true, 2, 0.0),
            test_session(true, 1, 0.05),
        ]);
        let report = reader.compute_slos(&SloTargets::default());
        assert!(report.all_met());
        assert_eq!(report.overall_status, SloStatus::Met);
        assert_eq!(report.session_count, 3);
    }

    #[test]
    fn test_slo_success_rate_breached() {
        let reader = reader_from_sessions(vec![
            test_session(true, 1, 0.0),
            test_session(false, 6, 0.5),
            test_session(false, 6, 0.3),
            test_session(false, 6, 0.2),
        ]);
        let report = reader.compute_slos(&SloTargets::default());
        assert_eq!(report.overall_status, SloStatus::Breached);
        let sr = report
            .measurements
            .iter()
            .find(|m| m.name == "success_rate")
            .unwrap();
        assert_eq!(sr.status, SloStatus::Breached);
        assert!((sr.value - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_slo_stuck_rate_warning() {
        // 1 stuck out of 5 = 0.20 > target 0.10 but <= warning 0.15
        // Actually 0.20 > 0.15 so this is Breached
        let reader = reader_from_sessions(vec![
            test_session(true, 1, 0.0),
            test_session(true, 2, 0.0),
            test_session(true, 1, 0.0),
            test_session(true, 1, 0.0),
            test_session(false, 6, 0.5),
        ]);
        let report = reader.compute_slos(&SloTargets::default());
        let stuck = report
            .measurements
            .iter()
            .find(|m| m.name == "stuck_rate")
            .unwrap();
        // 1/5 = 0.20 > warning threshold of 0.15
        assert_eq!(stuck.status, SloStatus::Breached);
    }

    #[test]
    fn test_slo_empty_sessions() {
        let reader = reader_from_sessions(vec![]);
        let report = reader.compute_slos(&SloTargets::default());
        assert_eq!(report.session_count, 0);
        assert!(report.all_met());
        assert!(report.measurements.is_empty());
    }

    #[test]
    fn test_slo_iterations_to_green() {
        let reader =
            reader_from_sessions(vec![test_session(true, 4, 0.0), test_session(true, 5, 0.0)]);
        let report = reader.compute_slos(&SloTargets::default());
        let iters = report
            .measurements
            .iter()
            .find(|m| m.name == "avg_iterations_to_green")
            .unwrap();
        // avg = 4.5 > target 2.5 → Breached (lower is better)
        assert!((iters.value - 4.5).abs() < 0.01);
        assert_eq!(iters.status, SloStatus::Breached);
    }

    #[test]
    fn test_slo_report_summary() {
        let reader = reader_from_sessions(vec![test_session(true, 1, 0.0)]);
        let report = reader.compute_slos(&SloTargets::default());
        let summary = report.summary();
        assert!(summary.contains("Met"));
        assert!(summary.contains("1 sessions"));
        assert!(summary.contains("success_rate"));
    }

    #[test]
    fn test_slo_custom_targets() {
        let targets = SloTargets {
            success_rate: 0.50,
            avg_iterations_to_green: 5.0,
            stuck_rate: 0.30,
            no_change_rate: 0.40,
        };
        let reader = reader_from_sessions(vec![
            test_session(true, 3, 0.1),
            test_session(false, 6, 0.2),
        ]);
        let report = reader.compute_slos(&targets);
        let sr = report
            .measurements
            .iter()
            .find(|m| m.name == "success_rate")
            .unwrap();
        assert_eq!(sr.status, SloStatus::Met); // 0.50 >= 0.50
    }

    // ──────────────────────────────────────────────────────────────────────
    // Typed Execution Artifact Tests
    // ──────────────────────────────────────────────────────────────────────

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
        let mut collector = MetricsCollector::new("sess-ea", "issue-ea", "Artifact test");

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
        let mut collector = MetricsCollector::new("sess-empty", "issue-empty", "Empty artifact");
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

    #[test]
    fn test_write_execution_artifacts_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = SessionMetrics {
            session_id: "test-session-art".into(),
            issue_id: "test-issue".into(),
            issue_title: "Test".into(),
            success: true,
            total_iterations: 2,
            final_tier: "Integrator".into(),
            elapsed_ms: 5000,
            total_no_change_iterations: 0,
            no_change_rate: 0.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![
                IterationMetrics {
                    iteration: 1,
                    tier: "Integrator".into(),
                    agent_model: "m1".into(),
                    agent_prompt_tokens: 0,
                    agent_completion_tokens: 0,
                    agent_response_ms: 0,
                    verifier_ms: 0,
                    error_count: 0,
                    error_categories: vec![],
                    no_change: false,
                    auto_fix_applied: false,
                    regression_detected: false,
                    rollback_performed: false,
                    escalated: false,
                    coder_route: None,
                    artifacts: vec![],
                    execution_artifact: Some(ExecutionArtifact {
                        schema_version: ARTIFACT_SCHEMA_VERSION,
                        route_decision: Some(RouteDecision {
                            coder: "RustCoder".into(),
                            input_error_categories: vec![],
                            tier: "Integrator".into(),
                            rationale: None,
                        }),
                        verifier_snapshot: None,
                        evaluator_snapshot: None,
                        retry_rationale: None,
                    }),
                    progress_score: None,
                    best_error_count: None,
                },
                IterationMetrics {
                    iteration: 2,
                    tier: "Integrator".into(),
                    agent_model: "m1".into(),
                    agent_prompt_tokens: 0,
                    agent_completion_tokens: 0,
                    agent_response_ms: 0,
                    verifier_ms: 0,
                    error_count: 0,
                    error_categories: vec![],
                    no_change: false,
                    auto_fix_applied: false,
                    regression_detected: false,
                    rollback_performed: false,
                    escalated: false,
                    coder_route: None,
                    artifacts: vec![],
                    // No artifact for this iteration
                    execution_artifact: None,
                    progress_score: None,
                    best_error_count: None,
                },
            ],
            timestamp: "2026-01-01T00:00:00Z".into(),
        };

        write_execution_artifacts(&metrics, dir.path(), None);

        let art_dir = dir.path().join(".swarm-artifacts").join("test-session-art");
        assert!(art_dir.exists());

        // Only iteration 1 has an artifact
        let iter1 = art_dir.join("iteration-001.json");
        assert!(iter1.exists());
        let iter2 = art_dir.join("iteration-002.json");
        assert!(!iter2.exists());

        // Verify content is valid JSON
        let content = std::fs::read_to_string(&iter1).unwrap();
        let loaded: ExecutionArtifact = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert_eq!(loaded.route_decision.unwrap().coder, "RustCoder");
    }

    #[test]
    fn test_artifact_retention_pruning() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts_root = dir.path().join(".swarm-artifacts");

        // Create 5 session directories with staggered modification times
        for i in 1..=5 {
            let session_dir = artifacts_root.join(format!("session-{i}"));
            std::fs::create_dir_all(&session_dir).unwrap();
            std::fs::write(session_dir.join("iteration-001.json"), "{}").unwrap();
            // Small delay to ensure different modification times
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Prune to keep only 3
        prune_artifact_sessions(&artifacts_root, 3);

        let remaining: Vec<_> = std::fs::read_dir(&artifacts_root)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(remaining.len(), 3);
    }

    #[test]
    fn test_swarm_event_serialization() {
        let event = SwarmEvent {
            event: "swarm.issue.started".into(),
            timestamp: "2026-03-03T00:00:00Z".into(),
            issue_id: "test-123".into(),
            payload: SwarmEventPayload::IssueStarted {
                title: "Fix borrow checker error".into(),
                priority: Some(1),
                tier: "Worker".into(),
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("swarm.issue.started"));
        assert!(json.contains("issue_started"));
        assert!(json.contains("test-123"));

        // Round-trip
        let deserialized: SwarmEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event, "swarm.issue.started");
        assert_eq!(deserialized.issue_id, "test-123");
    }

    #[test]
    fn test_event_emitter_writes_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let emitter = SwarmEventEmitter::new(dir.path());

        emitter.issue_started("test-456", "Add feature X", Some(2), "Worker");
        emitter.iteration_completed("test-456", 1, "Worker", 3, false, 5000);
        emitter.issue_resolved("test-456", true, 2, 10000);

        let content = std::fs::read_to_string(dir.path().join(".swarm-events.jsonl")).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);

        // Each line is valid JSON
        for line in &lines {
            let _: SwarmEvent = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn test_is_critical() {
        let critical = SwarmEvent {
            event: "swarm.worker.failed".into(),
            timestamp: "2026-03-03T00:00:00Z".into(),
            issue_id: "test".into(),
            payload: SwarmEventPayload::WorkerFailed {
                model: "Qwen3.5".into(),
                error: "timeout".into(),
                iteration: 1,
            },
        };
        assert!(SwarmEventEmitter::is_critical(&critical));

        let non_critical = SwarmEvent {
            event: "swarm.iteration.completed".into(),
            timestamp: "2026-03-03T00:00:00Z".into(),
            issue_id: "test".into(),
            payload: SwarmEventPayload::IterationCompleted {
                iteration: 1,
                tier: "Worker".into(),
                error_count: 0,
                no_change: false,
                elapsed_ms: 5000,
            },
        };
        assert!(!SwarmEventEmitter::is_critical(&non_critical));

        let unhealthy = SwarmEvent {
            event: "swarm.health.check".into(),
            timestamp: "2026-03-03T00:00:00Z".into(),
            issue_id: "test".into(),
            payload: SwarmEventPayload::HealthCheck {
                endpoint: "vasp-03:8081".into(),
                healthy: false,
                latency_ms: None,
            },
        };
        assert!(SwarmEventEmitter::is_critical(&unhealthy));
    }

    #[test]
    fn test_cost_tracker_basic() {
        let mut tracker = CostTracker::new(0.0); // unlimited
        tracker.record_usage(1000, 500, false); // local, $0
        assert_eq!(tracker.accumulated_cost(), 0.0);
        assert_eq!(tracker.total_prompt_tokens(), 1000);
        assert_eq!(tracker.total_completion_tokens(), 500);
        assert!(tracker.check_budget().is_none());
    }

    #[test]
    fn test_cost_tracker_cloud_pricing() {
        let mut tracker = CostTracker::new(1.0); // $1 budget
                                                 // 100K prompt tokens = $1.50, should exceed $1 budget
        tracker.record_usage(100_000, 0, true);
        let cost = tracker.accumulated_cost();
        assert!(cost > 0.0);
        // $15/M * 100K = $1.50
        assert!((cost - 1.5).abs() < 0.01);
        assert!(tracker.check_budget().is_some());
    }

    #[test]
    fn test_cost_tracker_budget_enforcement() {
        let mut tracker = CostTracker::new(0.5); // $0.50 budget
                                                 // Small usage, under budget
        tracker.record_usage(10_000, 1_000, true);
        assert!(tracker.check_budget().is_none());
        // Large usage, over budget
        tracker.record_usage(100_000, 10_000, true);
        assert!(tracker.check_budget().is_some());
    }

    #[test]
    fn test_prune_task_prompt_no_prune() {
        let prompt = "Task description\n---\nIteration 1\n---\nIteration 2";
        // Iteration 2, prune_after 3 → no pruning
        assert_eq!(prune_task_prompt(prompt, 2, 3, 2), prompt);
    }

    #[test]
    fn test_prune_task_prompt_prunes() {
        let prompt = "Task description\n---\nIteration 1\n---\nIteration 2\n---\nIteration 3\n---\nIteration 4";
        let pruned = prune_task_prompt(prompt, 5, 3, 2);
        assert!(pruned.contains("Task description"));
        assert!(pruned.contains("Iteration 4"));
        assert!(pruned.contains("Iteration 3"));
        assert!(!pruned.contains("Iteration 1"));
        assert!(pruned.contains("[Earlier iterations pruned"));
    }

    #[test]
    fn test_append_experiment_tsv_creates_header() {
        let dir = tempfile::TempDir::new().unwrap();
        append_experiment_tsv(
            dir.path(),
            "abc123",
            5,
            &["fmt", "clippy"],
            "keep",
            "partial progress",
        );
        let content = std::fs::read_to_string(dir.path().join("experiments.tsv")).unwrap();
        assert!(content.starts_with("timestamp\t"));
        assert!(content.contains("abc123"));
        assert!(content.contains("keep"));
    }

    #[test]
    fn test_append_experiment_tsv_appends() {
        let dir = tempfile::TempDir::new().unwrap();
        append_experiment_tsv(dir.path(), "abc", 5, &["fmt"], "keep", "first");
        append_experiment_tsv(dir.path(), "def", 3, &["fmt", "clippy"], "revert", "second");
        let content = std::fs::read_to_string(dir.path().join("experiments.tsv")).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3); // header + 2 rows
    }

    #[test]
    fn test_append_failure_ledger() {
        let dir = tempfile::TempDir::new().unwrap();
        let entry = FailureLedgerEntry {
            tool: "edit_file".to_string(),
            error_class: "match_failure".to_string(),
            signal_traced: "old_content not found".to_string(),
            file_path: Some("src/main.rs".to_string()),
            iteration: 1,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            success: false,
        };
        append_failure_ledger(dir.path(), &entry);
        let content =
            std::fs::read_to_string(dir.path().join(".swarm-failure-ledger.jsonl")).unwrap();
        assert!(content.contains("edit_file"));
        assert!(content.contains("match_failure"));
    }

    #[test]
    fn test_failure_ledger_success_entry() {
        let dir = tempfile::TempDir::new().unwrap();
        let entry = FailureLedgerEntry {
            tool: "edit_file".to_string(),
            error_class: "anchor_edit".to_string(),
            signal_traced: "lines 10-15 replaced".to_string(),
            file_path: Some("src/lib.rs".to_string()),
            iteration: 2,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
            success: true,
        };
        append_failure_ledger(dir.path(), &entry);
        let content =
            std::fs::read_to_string(dir.path().join(".swarm-failure-ledger.jsonl")).unwrap();
        let parsed: FailureLedgerEntry = serde_json::from_str(content.trim()).unwrap();
        assert!(parsed.success);
    }
}
