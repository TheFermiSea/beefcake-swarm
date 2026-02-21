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
    pub iterations: Vec<IterationMetrics>,
    pub timestamp: String,
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
}

impl MetricsCollector {
    pub fn new(session_id: &str, issue_id: &str, issue_title: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            issue_id: issue_id.to_string(),
            issue_title: issue_title.to_string(),
            session_start: Instant::now(),
            current_iteration: None,
            iterations: Vec::new(),
            cloud_validations: Vec::new(),
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

    /// Finish the current iteration and store its metrics.
    pub fn finish_iteration(&mut self) {
        if let Some(iter) = self.current_iteration.take() {
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
            iterations: self.iterations,
            timestamp: chrono::Utc::now().to_rfc3339(),
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
}
