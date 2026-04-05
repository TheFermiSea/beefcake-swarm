use super::*;

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};

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
    tensorzero_episode_id: Option<String>,
    harness_trace: HarnessComponentTrace,
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
            tensorzero_episode_id: None,
            harness_trace: HarnessComponentTrace::default(),
        }
    }

    /// Set the TensorZero episode ID for this session.
    ///
    /// Episode IDs link all inferences for a single issue run,
    /// enabling TensorZero to correlate feedback with specific
    /// prompt variants and model configurations.
    pub fn set_episode_id(&mut self, episode_id: String) {
        self.tensorzero_episode_id = Some(episode_id);
    }

    /// Get the TensorZero episode ID, if set.
    pub fn episode_id(&self) -> Option<&str> {
        self.tensorzero_episode_id.as_deref()
    }

    /// Record a harness component event.
    ///
    /// Call these at the point the component fires. The trace is accumulated
    /// throughout the session and written into `SessionMetrics.harness_trace`
    /// on `finalize()`.
    pub fn record_reviewer_fired(&mut self, caught_issue: bool) {
        self.harness_trace.reviewer_fired = true;
        if caught_issue {
            self.harness_trace.reviewer_caught_issue = true;
        }
    }

    pub fn record_adversary_fired(&mut self, caught_issue: bool) {
        self.harness_trace.adversary_fired = true;
        if caught_issue {
            self.harness_trace.adversary_caught_issue = true;
        }
    }

    pub fn record_planner_fired(&mut self) {
        self.harness_trace.planner_fired = true;
    }

    pub fn record_sprint_contract_used(&mut self) {
        self.harness_trace.sprint_contract_used = true;
    }

    pub fn record_pivot_triggered(&mut self) {
        self.harness_trace.pivot_triggered = true;
    }

    pub fn record_context_reset(&mut self) {
        self.harness_trace.context_reset_triggered = true;
    }

    pub fn record_context_anxiety(&mut self) {
        self.harness_trace.context_anxiety_detected = true;
    }

    pub fn record_dead_end_injected(&mut self) {
        self.harness_trace.dead_end_injection_count += 1;
    }

    pub fn record_reviewer_leniency_flags(&mut self, count: u32) {
        self.harness_trace.reviewer_leniency_flag_count += count;
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

        let write_by_turn_2 = self.iterations.iter().take(2).any(|i| {
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
            tensorzero_episode_id: self.tensorzero_episode_id,
            harness_trace: self.harness_trace,
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector_basic_flow() {
        let mut collector = MetricsCollector::new(
            "sess-1",
            "issue-1",
            "Fix bug",
            "hybrid_balanced_v1",
            None,
            None,
            "v1",
        );

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
        let mut collector = MetricsCollector::new(
            "sess-2",
            "issue-2",
            "Fix regression",
            "hybrid_balanced_v1",
            None,
            None,
            "v1",
        );

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
        let mut collector = MetricsCollector::new(
            "sess-3",
            "issue-3",
            "Test flush",
            "hybrid_balanced_v1",
            None,
            None,
            "v1",
        );

        collector.start_iteration(1, "Integrator");
        collector.record_agent_time(Duration::from_secs(10));
        // Don't call finish_iteration — finalize should flush it

        let metrics = collector.finalize(true, "Integrator");
        assert_eq!(metrics.total_iterations, 1);
        assert_eq!(metrics.iterations[0].agent_response_ms, 10_000);
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
            stack_profile: "hybrid_balanced_v1".into(),
            repo_id: None,
            adapter_id: None,
            turns_until_first_write: None,
            write_by_turn_2: false,
            role_map_version: "v1".into(),
            tensorzero_episode_id: None,
            harness_trace: HarnessComponentTrace::default(),
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
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
            stack_profile: "hybrid_balanced_v1".into(),
            repo_id: None,
            adapter_id: None,
            turns_until_first_write: None,
            write_by_turn_2: false,
            role_map_version: "v1".into(),
            tensorzero_episode_id: None,
            harness_trace: HarnessComponentTrace::default(),
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
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
        let mut collector = MetricsCollector::new(
            "sess-art",
            "issue-art",
            "Artifact test",
            "hybrid_balanced_v1",
            None,
            None,
            "v1",
        );

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
            stack_profile: "hybrid_balanced_v1".into(),
            repo_id: None,
            adapter_id: None,
            turns_until_first_write: None,
            write_by_turn_2: false,
            role_map_version: "v1".into(),
            tensorzero_episode_id: None,
            harness_trace: HarnessComponentTrace::default(),
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
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
}
