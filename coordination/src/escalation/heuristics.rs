//! Structured escalation heuristics derived from telemetry data
//!
//! Analyzes historical session metrics to compute optimal escalation thresholds.
//! When telemetry data is available, these heuristics replace the static defaults
//! in `EscalationConfig`, allowing the escalation engine to adapt to observed
//! swarm behavior patterns.

use crate::escalation::engine::EscalationConfig;
use serde::{Deserialize, Serialize};

// NOTE: SessionMetrics lives in the swarm-agents crate, not in coordination.
// We use a lightweight mirror struct to avoid a circular dependency.

/// Lightweight mirror of per-session telemetry data used for heuristic computation.
///
/// This avoids a dependency on the `swarm-agents` crate from `coordination`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSample {
    /// Whether the session succeeded
    pub success: bool,
    /// Total iterations used
    pub total_iterations: u32,
    /// Total no-change iterations
    pub total_no_change_iterations: u32,
    /// No-change rate (0.0–1.0)
    pub no_change_rate: f64,
    /// Final tier reached
    pub final_tier: String,
}

/// Structured escalation heuristics derived from historical telemetry.
///
/// Computed from a corpus of `SessionSample` records. When no telemetry is
/// available, `TelemetryHeuristics::default()` returns the same values as
/// `EscalationConfig::default()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryHeuristics {
    /// Number of sessions used to compute these heuristics (0 = defaults)
    pub sample_count: usize,
    /// Derived repeat threshold (escalate when same error repeats N times)
    pub repeat_threshold: u32,
    /// Derived failure threshold (escalate after N total failures)
    pub failure_threshold: u32,
    /// Derived no-change threshold (stuck after N consecutive no-change iterations)
    pub no_change_threshold: u32,
    /// Multi-file complexity threshold (unchanged from default)
    pub multi_file_threshold: usize,
    /// Whether adversary review is required before close
    pub require_adversary_review: bool,
    /// Observed average no-change rate across sessions
    pub observed_no_change_rate: f64,
    /// Observed average iterations per session
    pub observed_avg_iterations: f64,
    /// Observed success rate across sessions
    pub observed_success_rate: f64,
}

impl Default for TelemetryHeuristics {
    fn default() -> Self {
        let defaults = EscalationConfig::default();
        Self {
            sample_count: 0,
            repeat_threshold: defaults.repeat_threshold,
            failure_threshold: defaults.failure_threshold,
            no_change_threshold: defaults.no_change_threshold,
            multi_file_threshold: defaults.multi_file_threshold,
            require_adversary_review: defaults.require_adversary_review,
            observed_no_change_rate: 0.0,
            observed_avg_iterations: 0.0,
            observed_success_rate: 0.0,
        }
    }
}

impl TelemetryHeuristics {
    /// Compute heuristics from a slice of historical session samples.
    ///
    /// Returns `TelemetryHeuristics::default()` when `sessions` is empty.
    pub fn from_sessions(sessions: &[SessionSample]) -> Self {
        if sessions.is_empty() {
            return Self::default();
        }

        let n = sessions.len() as f64;

        let avg_no_change_rate = sessions.iter().map(|s| s.no_change_rate).sum::<f64>() / n;
        let avg_iterations = sessions
            .iter()
            .map(|s| s.total_iterations as f64)
            .sum::<f64>()
            / n;
        let avg_no_change = sessions
            .iter()
            .map(|s| s.total_no_change_iterations as f64)
            .sum::<f64>()
            / n;
        let success_rate = sessions.iter().filter(|s| s.success).count() as f64 / n;

        // repeat_threshold: lower when no-change rate is high (escalate sooner)
        let repeat_threshold = if avg_no_change_rate > 0.4 {
            1
        } else if avg_no_change_rate > 0.25 {
            2
        } else {
            3
        }
        .clamp(1, 4);

        // failure_threshold: lower when avg iterations is high (escalate sooner)
        let failure_threshold = if avg_iterations > 5.0 {
            2
        } else if avg_iterations > 3.5 {
            3
        } else {
            4
        }
        .clamp(2, 6);

        // no_change_threshold: lower when avg no-change count is high (escalate sooner)
        let no_change_threshold = if avg_no_change > 3.0 {
            2
        } else if avg_no_change > 1.5 {
            3
        } else {
            4
        }
        .clamp(2, 5);

        Self {
            sample_count: sessions.len(),
            repeat_threshold,
            failure_threshold,
            no_change_threshold,
            multi_file_threshold: 8,
            require_adversary_review: true,
            observed_no_change_rate: avg_no_change_rate,
            observed_avg_iterations: avg_iterations,
            observed_success_rate: success_rate,
        }
    }

    /// Convert these heuristics into an `EscalationConfig`.
    pub fn to_escalation_config(&self) -> EscalationConfig {
        EscalationConfig {
            repeat_threshold: self.repeat_threshold,
            failure_threshold: self.failure_threshold,
            no_change_threshold: self.no_change_threshold,
            multi_file_threshold: self.multi_file_threshold,
            require_adversary_review: self.require_adversary_review,
        }
    }
}

/// Compute escalation heuristics from a slice of historical session samples.
///
/// Convenience free function that delegates to `TelemetryHeuristics::from_sessions`.
pub fn compute_heuristics(sessions: &[SessionSample]) -> TelemetryHeuristics {
    TelemetryHeuristics::from_sessions(sessions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(
        success: bool,
        total_iterations: u32,
        total_no_change_iterations: u32,
        no_change_rate: f64,
    ) -> SessionSample {
        SessionSample {
            success,
            total_iterations,
            total_no_change_iterations,
            no_change_rate,
            final_tier: "worker".to_string(),
        }
    }

    #[test]
    fn test_default_heuristics() {
        let h = TelemetryHeuristics::default();
        let cfg = EscalationConfig::default();

        assert_eq!(h.repeat_threshold, cfg.repeat_threshold);
        assert_eq!(h.failure_threshold, cfg.failure_threshold);
        assert_eq!(h.no_change_threshold, cfg.no_change_threshold);
        assert_eq!(h.multi_file_threshold, cfg.multi_file_threshold);
        assert_eq!(h.require_adversary_review, cfg.require_adversary_review);
        assert_eq!(h.sample_count, 0);
    }

    #[test]
    fn test_from_empty_sessions() {
        let h = TelemetryHeuristics::from_sessions(&[]);
        let defaults = TelemetryHeuristics::default();

        assert_eq!(h.repeat_threshold, defaults.repeat_threshold);
        assert_eq!(h.failure_threshold, defaults.failure_threshold);
        assert_eq!(h.no_change_threshold, defaults.no_change_threshold);
        assert_eq!(h.sample_count, 0);
    }

    #[test]
    fn test_high_no_change_rate_lowers_repeat_threshold() {
        // avg_no_change_rate = 0.5 → repeat_threshold should be 1
        let sessions = vec![
            make_session(false, 4, 2, 0.5),
            make_session(false, 4, 2, 0.5),
            make_session(false, 4, 2, 0.5),
        ];
        let h = TelemetryHeuristics::from_sessions(&sessions);
        assert_eq!(
            h.repeat_threshold, 1,
            "High no-change rate should yield repeat_threshold=1"
        );
    }

    #[test]
    fn test_high_iterations_lowers_failure_threshold() {
        // avg_iterations = 6 → failure_threshold should be 2
        let sessions = vec![
            make_session(false, 6, 1, 0.1),
            make_session(false, 6, 1, 0.1),
            make_session(false, 6, 1, 0.1),
        ];
        let h = TelemetryHeuristics::from_sessions(&sessions);
        assert_eq!(
            h.failure_threshold, 2,
            "High avg iterations should yield failure_threshold=2"
        );
    }

    #[test]
    fn test_to_escalation_config() {
        let sessions = vec![make_session(true, 3, 1, 0.1), make_session(true, 2, 0, 0.0)];
        let h = TelemetryHeuristics::from_sessions(&sessions);
        let cfg = h.to_escalation_config();

        assert_eq!(cfg.repeat_threshold, h.repeat_threshold);
        assert_eq!(cfg.failure_threshold, h.failure_threshold);
        assert_eq!(cfg.no_change_threshold, h.no_change_threshold);
        assert_eq!(cfg.multi_file_threshold, h.multi_file_threshold);
        assert_eq!(cfg.require_adversary_review, h.require_adversary_review);
        assert!(cfg.require_adversary_review);
    }

    #[test]
    fn test_compute_heuristics_free_function() {
        let sessions = vec![
            make_session(true, 3, 1, 0.2),
            make_session(false, 5, 2, 0.3),
        ];
        let h1 = TelemetryHeuristics::from_sessions(&sessions);
        let h2 = compute_heuristics(&sessions);

        assert_eq!(h1.repeat_threshold, h2.repeat_threshold);
        assert_eq!(h1.failure_threshold, h2.failure_threshold);
        assert_eq!(h1.no_change_threshold, h2.no_change_threshold);
        assert_eq!(h1.sample_count, h2.sample_count);
    }
}
