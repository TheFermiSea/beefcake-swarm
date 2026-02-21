//! Compaction observability — metrics, spans, and instrumentation hooks
//! for monitoring compaction health in dashboards and debugging.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Snapshot of compaction metrics for a single compaction event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionMetrics {
    /// Monotonic compaction sequence number.
    pub compaction_seq: u64,
    /// Trigger that initiated compaction.
    pub trigger: String,
    /// Entries before compaction.
    pub entries_before: usize,
    /// Entries after compaction.
    pub entries_after: usize,
    /// Entries compacted (removed + replaced with summary).
    pub entries_compacted: usize,
    /// Tokens before compaction.
    pub tokens_before: u32,
    /// Tokens after compaction.
    pub tokens_after: u32,
    /// Tokens freed by compaction.
    pub tokens_freed: u32,
    /// Compression ratio (tokens_freed / tokens_before).
    pub compression_ratio: f64,
    /// Summary token count.
    pub summary_tokens: u32,
    /// Compaction duration in microseconds.
    pub duration_us: u64,
    /// Whether the compaction succeeded.
    pub success: bool,
    /// Error message if failed.
    pub error: Option<String>,
    /// Timestamp (Unix epoch seconds).
    pub timestamp: u64,
}

impl CompactionMetrics {
    /// Create a successful compaction metrics snapshot.
    #[allow(clippy::too_many_arguments)]
    pub fn success(
        seq: u64,
        trigger: &str,
        entries_before: usize,
        entries_after: usize,
        tokens_before: u32,
        tokens_after: u32,
        summary_tokens: u32,
        duration_us: u64,
    ) -> Self {
        let entries_compacted = entries_before.saturating_sub(entries_after);
        let tokens_freed = tokens_before.saturating_sub(tokens_after);
        let compression_ratio = if tokens_before > 0 {
            tokens_freed as f64 / tokens_before as f64
        } else {
            0.0
        };

        Self {
            compaction_seq: seq,
            trigger: trigger.to_string(),
            entries_before,
            entries_after,
            entries_compacted,
            tokens_before,
            tokens_after,
            tokens_freed,
            compression_ratio,
            summary_tokens,
            duration_us,
            success: true,
            error: None,
            timestamp: 0, // caller sets this
        }
    }

    /// Create a failed compaction metrics snapshot.
    pub fn failure(seq: u64, trigger: &str, error: &str, duration_us: u64) -> Self {
        Self {
            compaction_seq: seq,
            trigger: trigger.to_string(),
            entries_before: 0,
            entries_after: 0,
            entries_compacted: 0,
            tokens_before: 0,
            tokens_after: 0,
            tokens_freed: 0,
            compression_ratio: 0.0,
            summary_tokens: 0,
            duration_us,
            success: false,
            error: Some(error.to_string()),
            timestamp: 0,
        }
    }

    /// Set the timestamp.
    pub fn at_time(mut self, timestamp: u64) -> Self {
        self.timestamp = timestamp;
        self
    }
}

/// Aggregate statistics across multiple compaction events.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CompactionStats {
    /// Total compactions attempted.
    pub total_compactions: u64,
    /// Successful compactions.
    pub successful_compactions: u64,
    /// Failed compactions.
    pub failed_compactions: u64,
    /// Total entries compacted across all events.
    pub total_entries_compacted: usize,
    /// Total tokens freed across all events.
    pub total_tokens_freed: u64,
    /// Average compression ratio.
    pub avg_compression_ratio: f64,
    /// Average compaction duration in microseconds.
    pub avg_duration_us: f64,
    /// Maximum compaction duration in microseconds.
    pub max_duration_us: u64,
    /// Average summary size in tokens.
    pub avg_summary_tokens: f64,
}

/// Rolling window of compaction metrics with aggregate statistics.
pub struct CompactionObserver {
    /// Recent metrics (bounded ring buffer).
    history: VecDeque<CompactionMetrics>,
    /// Maximum history size.
    max_history: usize,
    /// Running aggregate statistics.
    stats: CompactionStats,
    /// Next compaction sequence number.
    next_seq: u64,
}

impl CompactionObserver {
    /// Create a new observer with the given history capacity.
    pub fn new(max_history: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(max_history),
            max_history,
            stats: CompactionStats::default(),
            next_seq: 1,
        }
    }

    /// Get the next sequence number (auto-increments).
    pub fn next_seq(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }

    /// Record a compaction event.
    pub fn record(&mut self, metrics: CompactionMetrics) {
        self.stats.total_compactions += 1;

        if metrics.success {
            self.stats.successful_compactions += 1;
            self.stats.total_entries_compacted += metrics.entries_compacted;
            self.stats.total_tokens_freed += metrics.tokens_freed as u64;

            // Update running averages
            let n = self.stats.successful_compactions as f64;
            self.stats.avg_compression_ratio = self.stats.avg_compression_ratio
                + (metrics.compression_ratio - self.stats.avg_compression_ratio) / n;
            self.stats.avg_duration_us = self.stats.avg_duration_us
                + (metrics.duration_us as f64 - self.stats.avg_duration_us) / n;
            self.stats.avg_summary_tokens = self.stats.avg_summary_tokens
                + (metrics.summary_tokens as f64 - self.stats.avg_summary_tokens) / n;

            if metrics.duration_us > self.stats.max_duration_us {
                self.stats.max_duration_us = metrics.duration_us;
            }
        } else {
            self.stats.failed_compactions += 1;
        }

        // Add to ring buffer
        if self.history.len() >= self.max_history {
            self.history.pop_front();
        }
        self.history.push_back(metrics);
    }

    /// Get aggregate statistics.
    pub fn stats(&self) -> &CompactionStats {
        &self.stats
    }

    /// Get recent metrics history.
    pub fn history(&self) -> &VecDeque<CompactionMetrics> {
        &self.history
    }

    /// Get the most recent metrics entry.
    pub fn last(&self) -> Option<&CompactionMetrics> {
        self.history.back()
    }

    /// Get metrics for the last N compactions.
    pub fn recent(&self, n: usize) -> Vec<&CompactionMetrics> {
        self.history.iter().rev().take(n).collect()
    }

    /// Check if compaction health is degraded.
    /// Returns a warning message if concerning patterns are detected.
    pub fn health_check(&self) -> Option<String> {
        if self.stats.total_compactions < 3 {
            return None; // Not enough data
        }

        // High failure rate
        let failure_rate =
            self.stats.failed_compactions as f64 / self.stats.total_compactions as f64;
        if failure_rate > 0.5 {
            return Some(format!(
                "High compaction failure rate: {:.0}% ({}/{})",
                failure_rate * 100.0,
                self.stats.failed_compactions,
                self.stats.total_compactions
            ));
        }

        // Very low compression ratio (summaries almost as big as originals)
        if self.stats.avg_compression_ratio < 0.1 && self.stats.successful_compactions > 2 {
            return Some(format!(
                "Low compression ratio: {:.2} — summaries may be too large",
                self.stats.avg_compression_ratio
            ));
        }

        // Slow compaction
        if self.stats.max_duration_us > 5_000_000 {
            return Some(format!(
                "Slow compaction detected: max {}ms",
                self.stats.max_duration_us / 1000
            ));
        }

        None
    }

    /// Format a summary line for logging.
    pub fn summary_line(&self) -> String {
        format!(
            "compactions={}/{} ({}% success), tokens_freed={}, avg_ratio={:.2}, avg_duration={}us",
            self.stats.successful_compactions,
            self.stats.total_compactions,
            if self.stats.total_compactions > 0 {
                self.stats.successful_compactions * 100 / self.stats.total_compactions
            } else {
                0
            },
            self.stats.total_tokens_freed,
            self.stats.avg_compression_ratio,
            self.stats.avg_duration_us as u64
        )
    }
}

impl Default for CompactionObserver {
    fn default() -> Self {
        Self::new(100)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_success_metrics(seq: u64, tokens_before: u32, tokens_after: u32) -> CompactionMetrics {
        CompactionMetrics::success(
            seq,
            "budget_threshold",
            20,
            5,
            tokens_before,
            tokens_after,
            50,
            1000,
        )
    }

    #[test]
    fn test_metrics_success() {
        let m = CompactionMetrics::success(1, "budget", 20, 5, 1000, 300, 50, 500);
        assert!(m.success);
        assert_eq!(m.entries_compacted, 15);
        assert_eq!(m.tokens_freed, 700);
        assert!((m.compression_ratio - 0.7).abs() < 0.01);
        assert!(m.error.is_none());
    }

    #[test]
    fn test_metrics_failure() {
        let m = CompactionMetrics::failure(1, "event", "empty store", 100);
        assert!(!m.success);
        assert_eq!(m.error.as_deref(), Some("empty store"));
        assert_eq!(m.entries_compacted, 0);
    }

    #[test]
    fn test_metrics_timestamp() {
        let m = CompactionMetrics::success(1, "test", 10, 5, 500, 200, 30, 100).at_time(1234567890);
        assert_eq!(m.timestamp, 1234567890);
    }

    #[test]
    fn test_observer_record_and_stats() {
        let mut obs = CompactionObserver::new(10);

        let seq = obs.next_seq();
        obs.record(make_success_metrics(seq, 1000, 300));
        let seq = obs.next_seq();
        obs.record(make_success_metrics(seq, 800, 200));

        let stats = obs.stats();
        assert_eq!(stats.total_compactions, 2);
        assert_eq!(stats.successful_compactions, 2);
        assert_eq!(stats.failed_compactions, 0);
        assert_eq!(stats.total_entries_compacted, 30); // 15 + 15
        assert_eq!(stats.total_tokens_freed, 1300); // 700 + 600
    }

    #[test]
    fn test_observer_failure_tracking() {
        let mut obs = CompactionObserver::new(10);

        let seq = obs.next_seq();
        obs.record(make_success_metrics(seq, 500, 200));
        let seq = obs.next_seq();
        obs.record(CompactionMetrics::failure(seq, "event", "empty", 50));

        let stats = obs.stats();
        assert_eq!(stats.total_compactions, 2);
        assert_eq!(stats.successful_compactions, 1);
        assert_eq!(stats.failed_compactions, 1);
    }

    #[test]
    fn test_observer_ring_buffer() {
        let mut obs = CompactionObserver::new(3);

        for _i in 0..5 {
            let seq = obs.next_seq();
            obs.record(make_success_metrics(seq, 1000, 500));
        }

        assert_eq!(obs.history().len(), 3); // capped at max_history
        assert_eq!(obs.history().front().unwrap().compaction_seq, 3); // oldest retained
        assert_eq!(obs.history().back().unwrap().compaction_seq, 5); // newest
    }

    #[test]
    fn test_observer_recent() {
        let mut obs = CompactionObserver::new(10);

        for _ in 0..5 {
            let seq = obs.next_seq();
            obs.record(make_success_metrics(seq, 1000, 500));
        }

        let recent = obs.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].compaction_seq, 5); // most recent first
        assert_eq!(recent[1].compaction_seq, 4);
    }

    #[test]
    fn test_observer_last() {
        let mut obs = CompactionObserver::new(10);
        assert!(obs.last().is_none());

        let seq = obs.next_seq();
        obs.record(make_success_metrics(seq, 1000, 500));
        assert_eq!(obs.last().unwrap().compaction_seq, 1);
    }

    #[test]
    fn test_health_check_healthy() {
        let mut obs = CompactionObserver::new(10);
        for _ in 0..5 {
            let seq = obs.next_seq();
            obs.record(make_success_metrics(seq, 1000, 300));
        }
        assert!(obs.health_check().is_none());
    }

    #[test]
    fn test_health_check_high_failure_rate() {
        let mut obs = CompactionObserver::new(10);
        let seq = obs.next_seq();
        obs.record(make_success_metrics(seq, 1000, 300));
        for _ in 0..3 {
            let seq = obs.next_seq();
            obs.record(CompactionMetrics::failure(seq, "event", "fail", 100));
        }

        let warning = obs.health_check();
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("failure rate"));
    }

    #[test]
    fn test_health_check_low_compression() {
        let mut obs = CompactionObserver::new(10);
        for _ in 0..5 {
            let seq = obs.next_seq();
            // Very low compression: 1000 → 950
            obs.record(make_success_metrics(seq, 1000, 950));
        }

        let warning = obs.health_check();
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("compression ratio"));
    }

    #[test]
    fn test_health_check_slow_compaction() {
        let mut obs = CompactionObserver::new(10);
        for _ in 0..3 {
            let seq = obs.next_seq();
            let mut m = make_success_metrics(seq, 1000, 300);
            m.duration_us = 6_000_000; // 6 seconds
            obs.record(m);
        }

        let warning = obs.health_check();
        assert!(warning.is_some());
        assert!(warning.unwrap().contains("Slow compaction"));
    }

    #[test]
    fn test_summary_line() {
        let mut obs = CompactionObserver::new(10);
        let seq = obs.next_seq();
        obs.record(make_success_metrics(seq, 1000, 300));
        let line = obs.summary_line();
        assert!(line.contains("compactions=1/1"));
        assert!(line.contains("100% success"));
        assert!(line.contains("tokens_freed=700"));
    }

    #[test]
    fn test_metrics_serde() {
        let m = CompactionMetrics::success(1, "budget", 20, 5, 1000, 300, 50, 500);
        let json = serde_json::to_string(&m).unwrap();
        let parsed: CompactionMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.compaction_seq, 1);
        assert_eq!(parsed.tokens_freed, 700);
        assert!(parsed.success);
    }

    #[test]
    fn test_stats_serde() {
        let stats = CompactionStats {
            total_compactions: 10,
            successful_compactions: 8,
            failed_compactions: 2,
            total_entries_compacted: 100,
            total_tokens_freed: 5000,
            avg_compression_ratio: 0.65,
            avg_duration_us: 800.0,
            max_duration_us: 2000,
            avg_summary_tokens: 40.0,
        };
        let json = serde_json::to_string(&stats).unwrap();
        let parsed: CompactionStats = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_compactions, 10);
        assert_eq!(parsed.total_tokens_freed, 5000);
    }
}
