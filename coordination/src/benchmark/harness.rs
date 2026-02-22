//! Benchmark harness for router and orchestration outcomes.
//!
//! Measures quality, latency, and cost metrics from orchestration sessions:
//! - First-pass verifier success rate
//! - Iterations-to-green (how many loops before all gates pass)
//! - Escalation frequency (how often tasks escalate from Worker to Council)
//! - p50/p95 latency per session
//! - Token/cost envelopes
//!
//! Supports baseline-vs-post-change comparison to evaluate the impact
//! of routing or orchestration changes.

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A single orchestration session record for benchmarking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    /// Session identifier.
    pub session_id: String,
    /// Whether the verifier passed on the first iteration.
    pub first_pass_success: bool,
    /// Total iterations until green (None if never reached green).
    pub iterations_to_green: Option<u32>,
    /// Total iterations attempted.
    pub total_iterations: u32,
    /// Whether the task escalated to a higher tier.
    pub escalated: bool,
    /// Number of escalation events.
    pub escalation_count: u32,
    /// Wall-clock duration of the session.
    pub duration: Duration,
    /// Total tokens consumed.
    pub tokens_used: u64,
    /// Estimated cost (in arbitrary units).
    pub estimated_cost: f64,
    /// Final outcome.
    pub outcome: SessionOutcome,
}

/// Final outcome of a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionOutcome {
    /// Verifier passed, task completed.
    Success,
    /// Max iterations reached without passing.
    MaxIterations,
    /// Task was stuck and required human intervention.
    Stuck,
    /// Task failed for other reasons.
    Failed,
}

/// Aggregated metrics from a set of session records.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationMetrics {
    /// Number of sessions measured.
    pub session_count: usize,
    /// First-pass verifier success rate (0.0 to 1.0).
    pub first_pass_rate: f64,
    /// Overall success rate (verifier eventually passed).
    pub overall_success_rate: f64,
    /// Average iterations to green (for sessions that succeeded).
    pub avg_iterations_to_green: f64,
    /// Median iterations to green.
    pub median_iterations_to_green: f64,
    /// Escalation frequency (fraction of sessions that escalated).
    pub escalation_rate: f64,
    /// Average escalations per session.
    pub avg_escalations: f64,
    /// Latency percentiles.
    pub latency_p50: Duration,
    pub latency_p95: Duration,
    pub latency_max: Duration,
    /// Token usage statistics.
    pub tokens_p50: u64,
    pub tokens_p95: u64,
    pub tokens_total: u64,
    /// Cost statistics.
    pub cost_total: f64,
    pub cost_avg: f64,
    /// Stuck rate (sessions requiring human intervention).
    pub stuck_rate: f64,
}

/// Comparison between baseline and post-change metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsDelta {
    /// Baseline metrics.
    pub baseline: OrchestrationMetrics,
    /// Post-change metrics.
    pub post_change: OrchestrationMetrics,
    /// First-pass rate delta (positive = improvement).
    pub first_pass_delta: f64,
    /// Overall success rate delta.
    pub success_rate_delta: f64,
    /// Iterations-to-green delta (negative = improvement).
    pub iterations_delta: f64,
    /// Escalation rate delta (negative = improvement).
    pub escalation_rate_delta: f64,
    /// Latency p50 delta (negative = improvement).
    pub latency_p50_delta: Duration,
    /// Token p50 delta (negative = improvement).
    pub tokens_p50_delta: i64,
    /// Cost delta (negative = improvement).
    pub cost_delta: f64,
    /// Stuck rate delta (negative = improvement).
    pub stuck_rate_delta: f64,
}

/// Computes aggregated orchestration metrics from session records.
pub fn compute_metrics(records: &[SessionRecord]) -> OrchestrationMetrics {
    if records.is_empty() {
        return OrchestrationMetrics {
            session_count: 0,
            first_pass_rate: 0.0,
            overall_success_rate: 0.0,
            avg_iterations_to_green: 0.0,
            median_iterations_to_green: 0.0,
            escalation_rate: 0.0,
            avg_escalations: 0.0,
            latency_p50: Duration::ZERO,
            latency_p95: Duration::ZERO,
            latency_max: Duration::ZERO,
            tokens_p50: 0,
            tokens_p95: 0,
            tokens_total: 0,
            cost_total: 0.0,
            cost_avg: 0.0,
            stuck_rate: 0.0,
        };
    }

    let n = records.len();

    // First-pass success rate.
    let first_pass = records.iter().filter(|r| r.first_pass_success).count();
    let first_pass_rate = first_pass as f64 / n as f64;

    // Overall success rate.
    let successes = records
        .iter()
        .filter(|r| r.outcome == SessionOutcome::Success)
        .count();
    let overall_success_rate = successes as f64 / n as f64;

    // Iterations to green (only for successful sessions).
    let mut itg: Vec<u32> = records
        .iter()
        .filter_map(|r| r.iterations_to_green)
        .collect();
    itg.sort();
    let avg_itg = if itg.is_empty() {
        0.0
    } else {
        itg.iter().sum::<u32>() as f64 / itg.len() as f64
    };
    let median_itg = percentile_u32(&itg, 50);

    // Escalation metrics.
    let escalated = records.iter().filter(|r| r.escalated).count();
    let escalation_rate = escalated as f64 / n as f64;
    let total_escalations: u32 = records.iter().map(|r| r.escalation_count).sum();
    let avg_escalations = total_escalations as f64 / n as f64;

    // Latency percentiles.
    let mut durations: Vec<Duration> = records.iter().map(|r| r.duration).collect();
    durations.sort();
    let latency_p50 = percentile_duration(&durations, 50);
    let latency_p95 = percentile_duration(&durations, 95);
    let latency_max = durations.last().copied().unwrap_or(Duration::ZERO);

    // Token percentiles.
    let mut tokens: Vec<u64> = records.iter().map(|r| r.tokens_used).collect();
    tokens.sort();
    let tokens_p50 = percentile_u64(&tokens, 50);
    let tokens_p95 = percentile_u64(&tokens, 95);
    let tokens_total: u64 = tokens.iter().sum();

    // Cost.
    let cost_total: f64 = records.iter().map(|r| r.estimated_cost).sum();
    let cost_avg = cost_total / n as f64;

    // Stuck rate.
    let stuck = records
        .iter()
        .filter(|r| r.outcome == SessionOutcome::Stuck)
        .count();
    let stuck_rate = stuck as f64 / n as f64;

    OrchestrationMetrics {
        session_count: n,
        first_pass_rate,
        overall_success_rate,
        avg_iterations_to_green: avg_itg,
        median_iterations_to_green: median_itg as f64,
        escalation_rate,
        avg_escalations,
        latency_p50,
        latency_p95,
        latency_max,
        tokens_p50,
        tokens_p95,
        tokens_total,
        cost_total,
        cost_avg,
        stuck_rate,
    }
}

/// Compare baseline metrics against post-change metrics.
pub fn compare_metrics(baseline: &[SessionRecord], post_change: &[SessionRecord]) -> MetricsDelta {
    let b = compute_metrics(baseline);
    let p = compute_metrics(post_change);

    let latency_p50_delta = p.latency_p50.abs_diff(b.latency_p50);

    MetricsDelta {
        first_pass_delta: p.first_pass_rate - b.first_pass_rate,
        success_rate_delta: p.overall_success_rate - b.overall_success_rate,
        iterations_delta: p.avg_iterations_to_green - b.avg_iterations_to_green,
        escalation_rate_delta: p.escalation_rate - b.escalation_rate,
        latency_p50_delta,
        tokens_p50_delta: p.tokens_p50 as i64 - b.tokens_p50 as i64,
        cost_delta: p.cost_total - b.cost_total,
        stuck_rate_delta: p.stuck_rate - b.stuck_rate,
        baseline: b,
        post_change: p,
    }
}

/// Format a MetricsDelta as a human-readable comparison report.
pub fn format_comparison(delta: &MetricsDelta) -> String {
    let mut report = String::new();

    report.push_str("# Orchestration Benchmark Comparison\n\n");

    report.push_str("## Quality Metrics\n\n");
    report.push_str("| Metric | Baseline | Post-Change | Delta |\n");
    report.push_str("|--------|----------|-------------|-------|\n");
    report.push_str(&format!(
        "| First-pass rate | {:.1}% | {:.1}% | {:+.1}% |\n",
        delta.baseline.first_pass_rate * 100.0,
        delta.post_change.first_pass_rate * 100.0,
        delta.first_pass_delta * 100.0,
    ));
    report.push_str(&format!(
        "| Overall success | {:.1}% | {:.1}% | {:+.1}% |\n",
        delta.baseline.overall_success_rate * 100.0,
        delta.post_change.overall_success_rate * 100.0,
        delta.success_rate_delta * 100.0,
    ));
    report.push_str(&format!(
        "| Avg iterations | {:.2} | {:.2} | {:+.2} |\n",
        delta.baseline.avg_iterations_to_green,
        delta.post_change.avg_iterations_to_green,
        delta.iterations_delta,
    ));
    report.push_str(&format!(
        "| Escalation rate | {:.1}% | {:.1}% | {:+.1}% |\n",
        delta.baseline.escalation_rate * 100.0,
        delta.post_change.escalation_rate * 100.0,
        delta.escalation_rate_delta * 100.0,
    ));
    report.push_str(&format!(
        "| Stuck rate | {:.1}% | {:.1}% | {:+.1}% |\n\n",
        delta.baseline.stuck_rate * 100.0,
        delta.post_change.stuck_rate * 100.0,
        delta.stuck_rate_delta * 100.0,
    ));

    report.push_str("## Latency\n\n");
    report.push_str("| Percentile | Baseline | Post-Change |\n");
    report.push_str("|------------|----------|-------------|\n");
    report.push_str(&format!(
        "| p50 | {:.1}s | {:.1}s |\n",
        delta.baseline.latency_p50.as_secs_f64(),
        delta.post_change.latency_p50.as_secs_f64(),
    ));
    report.push_str(&format!(
        "| p95 | {:.1}s | {:.1}s |\n",
        delta.baseline.latency_p95.as_secs_f64(),
        delta.post_change.latency_p95.as_secs_f64(),
    ));
    report.push_str(&format!(
        "| max | {:.1}s | {:.1}s |\n\n",
        delta.baseline.latency_max.as_secs_f64(),
        delta.post_change.latency_max.as_secs_f64(),
    ));

    report.push_str("## Cost\n\n");
    report.push_str("| Metric | Baseline | Post-Change | Delta |\n");
    report.push_str("|--------|----------|-------------|-------|\n");
    report.push_str(&format!(
        "| Token p50 | {} | {} | {:+} |\n",
        delta.baseline.tokens_p50, delta.post_change.tokens_p50, delta.tokens_p50_delta,
    ));
    report.push_str(&format!(
        "| Total tokens | {} | {} | {:+} |\n",
        delta.baseline.tokens_total,
        delta.post_change.tokens_total,
        delta.post_change.tokens_total as i64 - delta.baseline.tokens_total as i64,
    ));
    report.push_str(&format!(
        "| Total cost | {:.2} | {:.2} | {:+.2} |\n",
        delta.baseline.cost_total, delta.post_change.cost_total, delta.cost_delta,
    ));

    report
}

/// Compute the p-th percentile from a sorted slice of u32.
fn percentile_u32(sorted: &[u32], p: usize) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (p * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}

/// Compute the p-th percentile from a sorted slice of u64.
fn percentile_u64(sorted: &[u64], p: usize) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (p * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}

/// Compute the p-th percentile from a sorted slice of Duration.
fn percentile_duration(sorted: &[Duration], p: usize) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = (p * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        id: &str,
        first_pass: bool,
        itg: Option<u32>,
        total_iter: u32,
        escalated: bool,
        escalation_count: u32,
        duration_secs: u64,
        tokens: u64,
        cost: f64,
        outcome: SessionOutcome,
    ) -> SessionRecord {
        SessionRecord {
            session_id: id.to_string(),
            first_pass_success: first_pass,
            iterations_to_green: itg,
            total_iterations: total_iter,
            escalated,
            escalation_count,
            duration: Duration::from_secs(duration_secs),
            tokens_used: tokens,
            estimated_cost: cost,
            outcome,
        }
    }

    #[test]
    fn test_compute_metrics_empty() {
        let metrics = compute_metrics(&[]);
        assert_eq!(metrics.session_count, 0);
        assert_eq!(metrics.first_pass_rate, 0.0);
    }

    #[test]
    fn test_compute_metrics_all_first_pass() {
        let records = vec![
            make_record(
                "s1",
                true,
                Some(1),
                1,
                false,
                0,
                60,
                500,
                0.1,
                SessionOutcome::Success,
            ),
            make_record(
                "s2",
                true,
                Some(1),
                1,
                false,
                0,
                45,
                400,
                0.08,
                SessionOutcome::Success,
            ),
        ];
        let metrics = compute_metrics(&records);

        assert_eq!(metrics.session_count, 2);
        assert_eq!(metrics.first_pass_rate, 1.0);
        assert_eq!(metrics.overall_success_rate, 1.0);
        assert_eq!(metrics.avg_iterations_to_green, 1.0);
        assert_eq!(metrics.escalation_rate, 0.0);
        assert_eq!(metrics.stuck_rate, 0.0);
    }

    #[test]
    fn test_compute_metrics_mixed_outcomes() {
        let records = vec![
            make_record(
                "s1",
                true,
                Some(1),
                1,
                false,
                0,
                60,
                500,
                0.1,
                SessionOutcome::Success,
            ),
            make_record(
                "s2",
                false,
                Some(3),
                3,
                true,
                1,
                180,
                2000,
                0.5,
                SessionOutcome::Success,
            ),
            make_record(
                "s3",
                false,
                None,
                6,
                true,
                2,
                600,
                5000,
                1.0,
                SessionOutcome::MaxIterations,
            ),
            make_record(
                "s4",
                false,
                None,
                4,
                false,
                0,
                300,
                3000,
                0.7,
                SessionOutcome::Stuck,
            ),
        ];
        let metrics = compute_metrics(&records);

        assert_eq!(metrics.session_count, 4);
        assert_eq!(metrics.first_pass_rate, 0.25); // 1/4
        assert_eq!(metrics.overall_success_rate, 0.5); // 2/4
        assert_eq!(metrics.avg_iterations_to_green, 2.0); // (1+3)/2
        assert_eq!(metrics.escalation_rate, 0.5); // 2/4
        assert_eq!(metrics.stuck_rate, 0.25); // 1/4
        assert_eq!(metrics.tokens_total, 10500);
    }

    #[test]
    fn test_compute_metrics_latency_percentiles() {
        let records = vec![
            make_record(
                "s1",
                true,
                Some(1),
                1,
                false,
                0,
                10,
                100,
                0.01,
                SessionOutcome::Success,
            ),
            make_record(
                "s2",
                true,
                Some(1),
                1,
                false,
                0,
                20,
                200,
                0.02,
                SessionOutcome::Success,
            ),
            make_record(
                "s3",
                true,
                Some(2),
                2,
                false,
                0,
                30,
                300,
                0.03,
                SessionOutcome::Success,
            ),
            make_record(
                "s4",
                true,
                Some(1),
                1,
                false,
                0,
                40,
                400,
                0.04,
                SessionOutcome::Success,
            ),
            make_record(
                "s5",
                false,
                None,
                5,
                true,
                1,
                500,
                5000,
                1.0,
                SessionOutcome::Failed,
            ),
        ];
        let metrics = compute_metrics(&records);

        // Sorted durations: [10, 20, 30, 40, 500]
        assert_eq!(metrics.latency_p50, Duration::from_secs(30));
        assert_eq!(metrics.latency_p95, Duration::from_secs(500));
        assert_eq!(metrics.latency_max, Duration::from_secs(500));
    }

    #[test]
    fn test_compare_metrics_improvement() {
        let baseline = vec![
            make_record(
                "b1",
                false,
                Some(3),
                3,
                true,
                1,
                180,
                2000,
                0.5,
                SessionOutcome::Success,
            ),
            make_record(
                "b2",
                false,
                None,
                6,
                true,
                2,
                600,
                5000,
                1.0,
                SessionOutcome::MaxIterations,
            ),
        ];
        let post = vec![
            make_record(
                "p1",
                true,
                Some(1),
                1,
                false,
                0,
                60,
                500,
                0.1,
                SessionOutcome::Success,
            ),
            make_record(
                "p2",
                false,
                Some(2),
                2,
                false,
                0,
                120,
                1000,
                0.2,
                SessionOutcome::Success,
            ),
        ];

        let delta = compare_metrics(&baseline, &post);

        // Baseline: first_pass=0%, success=50%, avg_itg=3.0, escalation=100%
        // Post:     first_pass=50%, success=100%, avg_itg=1.5, escalation=0%
        assert!(delta.first_pass_delta > 0.0); // Improved
        assert!(delta.success_rate_delta > 0.0); // Improved
        assert!(delta.iterations_delta < 0.0); // Fewer iterations
        assert!(delta.escalation_rate_delta < 0.0); // Fewer escalations
    }

    #[test]
    fn test_compare_metrics_regression() {
        let baseline = vec![make_record(
            "b1",
            true,
            Some(1),
            1,
            false,
            0,
            60,
            500,
            0.1,
            SessionOutcome::Success,
        )];
        let post = vec![make_record(
            "p1",
            false,
            Some(4),
            4,
            true,
            1,
            300,
            3000,
            0.8,
            SessionOutcome::Success,
        )];

        let delta = compare_metrics(&baseline, &post);

        assert!(delta.first_pass_delta < 0.0); // Regressed
        assert!(delta.iterations_delta > 0.0); // More iterations
        assert!(delta.escalation_rate_delta > 0.0); // More escalations
    }

    #[test]
    fn test_format_comparison() {
        let baseline = vec![make_record(
            "b1",
            true,
            Some(1),
            1,
            false,
            0,
            60,
            500,
            0.1,
            SessionOutcome::Success,
        )];
        let post = vec![make_record(
            "p1",
            true,
            Some(1),
            1,
            false,
            0,
            45,
            400,
            0.08,
            SessionOutcome::Success,
        )];

        let delta = compare_metrics(&baseline, &post);
        let report = format_comparison(&delta);

        assert!(report.contains("Orchestration Benchmark Comparison"));
        assert!(report.contains("First-pass rate"));
        assert!(report.contains("Escalation rate"));
        assert!(report.contains("p50"));
        assert!(report.contains("p95"));
        assert!(report.contains("Total cost"));
    }

    #[test]
    fn test_percentile_edge_cases() {
        assert_eq!(percentile_u32(&[], 50), 0);
        assert_eq!(percentile_u32(&[42], 50), 42);
        assert_eq!(percentile_u32(&[1, 2, 3, 4, 5], 0), 1);
        assert_eq!(percentile_u32(&[1, 2, 3, 4, 5], 100), 5);
    }

    #[test]
    fn test_session_record_serialization() {
        let record = make_record(
            "test",
            true,
            Some(1),
            1,
            false,
            0,
            60,
            500,
            0.1,
            SessionOutcome::Success,
        );
        let json = serde_json::to_string(&record).unwrap();
        let parsed: SessionRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.session_id, "test");
        assert!(parsed.first_pass_success);
        assert_eq!(parsed.outcome, SessionOutcome::Success);
    }

    #[test]
    fn test_metrics_serialization() {
        let records = vec![make_record(
            "s1",
            true,
            Some(1),
            1,
            false,
            0,
            60,
            500,
            0.1,
            SessionOutcome::Success,
        )];
        let metrics = compute_metrics(&records);
        let json = serde_json::to_string(&metrics).unwrap();
        let parsed: OrchestrationMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.session_count, 1);
        assert_eq!(parsed.first_pass_rate, 1.0);
    }
}
