//! Performance Controls — Timeout, Retry, and Truncation for Analysis Requests
//!
//! Provides configurable budgets for long-running analysis operations like
//! graph queries, AST scans, and dependency traversals. Ensures operations
//! stay within latency and token budgets.
//!
//! # Usage
//!
//! ```rust,ignore
//! use coordination::perf_control::{PerfBudget, RetryPolicy, TruncationPolicy};
//!
//! let budget = PerfBudget::new()
//!     .timeout_ms(5000)
//!     .max_results(100)
//!     .max_output_bytes(32_768);
//!
//! if budget.is_exceeded(elapsed_ms, result_count, output_bytes) {
//!     // Truncate and return partial results
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

/// Performance budget for a single operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfBudget {
    /// Maximum wall-clock time in milliseconds (0 = unlimited).
    pub timeout_ms: u64,
    /// Maximum number of results to return (0 = unlimited).
    pub max_results: usize,
    /// Maximum output size in bytes (0 = unlimited).
    pub max_output_bytes: usize,
    /// Maximum depth for recursive/graph traversals (0 = unlimited).
    pub max_depth: usize,
}

impl PerfBudget {
    /// Create a new budget with all limits unlimited.
    pub fn new() -> Self {
        Self {
            timeout_ms: 0,
            max_results: 0,
            max_output_bytes: 0,
            max_depth: 0,
        }
    }

    /// Set timeout in milliseconds.
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Set maximum result count.
    pub fn max_results(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }

    /// Set maximum output size in bytes.
    pub fn max_output_bytes(mut self, n: usize) -> Self {
        self.max_output_bytes = n;
        self
    }

    /// Set maximum traversal depth.
    pub fn max_depth(mut self, n: usize) -> Self {
        self.max_depth = n;
        self
    }

    /// Check if any budget limit has been exceeded.
    pub fn is_exceeded(&self, elapsed_ms: u64, result_count: usize, output_bytes: usize) -> bool {
        (self.timeout_ms > 0 && elapsed_ms >= self.timeout_ms)
            || (self.max_results > 0 && result_count >= self.max_results)
            || (self.max_output_bytes > 0 && output_bytes >= self.max_output_bytes)
    }

    /// Which limit was exceeded first, if any.
    pub fn exceeded_limit(
        &self,
        elapsed_ms: u64,
        result_count: usize,
        output_bytes: usize,
    ) -> Option<BudgetLimit> {
        if self.timeout_ms > 0 && elapsed_ms >= self.timeout_ms {
            return Some(BudgetLimit::Timeout);
        }
        if self.max_results > 0 && result_count >= self.max_results {
            return Some(BudgetLimit::ResultCount);
        }
        if self.max_output_bytes > 0 && output_bytes >= self.max_output_bytes {
            return Some(BudgetLimit::OutputSize);
        }
        None
    }

    /// Get the timeout as a Duration, or None if unlimited.
    pub fn timeout_duration(&self) -> Option<Duration> {
        if self.timeout_ms > 0 {
            Some(Duration::from_millis(self.timeout_ms))
        } else {
            None
        }
    }
}

impl Default for PerfBudget {
    /// Default budget: 10s timeout, 200 results, 64KB output, 10 depth.
    fn default() -> Self {
        Self {
            timeout_ms: 10_000,
            max_results: 200,
            max_output_bytes: 65_536,
            max_depth: 10,
        }
    }
}

/// Which budget limit was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetLimit {
    /// Wall-clock timeout.
    Timeout,
    /// Maximum result count.
    ResultCount,
    /// Maximum output size.
    OutputSize,
    /// Maximum traversal depth.
    Depth,
}

impl std::fmt::Display for BudgetLimit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Timeout => write!(f, "timeout"),
            Self::ResultCount => write!(f, "result_count"),
            Self::OutputSize => write!(f, "output_size"),
            Self::Depth => write!(f, "depth"),
        }
    }
}

/// Retry policy for transient failures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (0 = no retries).
    pub max_retries: u32,
    /// Initial backoff delay in milliseconds.
    pub initial_backoff_ms: u64,
    /// Backoff multiplier (e.g., 2.0 for exponential).
    pub backoff_multiplier: f64,
    /// Maximum backoff delay in milliseconds.
    pub max_backoff_ms: u64,
}

impl RetryPolicy {
    /// Calculate the backoff delay for a given attempt number (0-indexed).
    pub fn backoff_ms(&self, attempt: u32) -> u64 {
        if attempt == 0 {
            return 0;
        }
        let delay =
            self.initial_backoff_ms as f64 * self.backoff_multiplier.powi(attempt as i32 - 1);
        (delay as u64).min(self.max_backoff_ms)
    }

    /// Whether another retry is allowed given the attempt count.
    pub fn should_retry(&self, attempt: u32) -> bool {
        attempt < self.max_retries
    }

    /// Get the backoff as a Duration for a given attempt.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        Duration::from_millis(self.backoff_ms(attempt))
    }
}

impl Default for RetryPolicy {
    /// Default: 2 retries, 500ms initial backoff, 2x multiplier, 5s max.
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_backoff_ms: 500,
            backoff_multiplier: 2.0,
            max_backoff_ms: 5_000,
        }
    }
}

/// Truncation policy for oversized results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationPolicy {
    /// Strategy for truncating results.
    pub strategy: TruncationStrategy,
    /// Whether to include a truncation warning in output.
    pub include_warning: bool,
}

/// How to truncate oversized results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TruncationStrategy {
    /// Keep first N results (default).
    Head,
    /// Keep last N results.
    Tail,
    /// Keep highest-priority results (requires scoring).
    Priority,
}

impl Default for TruncationPolicy {
    fn default() -> Self {
        Self {
            strategy: TruncationStrategy::Head,
            include_warning: true,
        }
    }
}

/// Result of applying truncation to a collection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationResult {
    /// Whether truncation was applied.
    pub truncated: bool,
    /// Number of items before truncation.
    pub original_count: usize,
    /// Number of items after truncation.
    pub retained_count: usize,
    /// Which limit caused truncation.
    pub reason: Option<BudgetLimit>,
}

impl TruncationResult {
    /// No truncation was needed.
    pub fn none(count: usize) -> Self {
        Self {
            truncated: false,
            original_count: count,
            retained_count: count,
            reason: None,
        }
    }

    /// Truncation was applied.
    pub fn applied(original: usize, retained: usize, reason: BudgetLimit) -> Self {
        Self {
            truncated: true,
            original_count: original,
            retained_count: retained,
            reason: Some(reason),
        }
    }

    /// Warning message for the consumer.
    pub fn warning(&self) -> Option<String> {
        if self.truncated {
            Some(format!(
                "Results truncated: {} of {} retained ({})",
                self.retained_count,
                self.original_count,
                self.reason
                    .map(|r| r.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            ))
        } else {
            None
        }
    }
}

/// Combined performance guard that tracks budget consumption in real-time.
pub struct PerfGuard {
    budget: PerfBudget,
    start: Instant,
    result_count: usize,
    output_bytes: usize,
    depth: usize,
}

impl PerfGuard {
    /// Create a new guard with a budget.
    pub fn new(budget: PerfBudget) -> Self {
        Self {
            budget,
            start: Instant::now(),
            result_count: 0,
            output_bytes: 0,
            depth: 0,
        }
    }

    /// Record results being added.
    pub fn add_results(&mut self, count: usize, bytes: usize) {
        self.result_count += count;
        self.output_bytes += bytes;
    }

    /// Set current traversal depth.
    pub fn set_depth(&mut self, depth: usize) {
        self.depth = depth;
    }

    /// Elapsed time since guard creation.
    pub fn elapsed_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Check if any budget limit has been exceeded.
    pub fn is_exceeded(&self) -> bool {
        self.budget
            .is_exceeded(self.elapsed_ms(), self.result_count, self.output_bytes)
            || (self.budget.max_depth > 0 && self.depth >= self.budget.max_depth)
    }

    /// Which limit was exceeded, if any.
    pub fn exceeded_limit(&self) -> Option<BudgetLimit> {
        if self.budget.max_depth > 0 && self.depth >= self.budget.max_depth {
            return Some(BudgetLimit::Depth);
        }
        self.budget
            .exceeded_limit(self.elapsed_ms(), self.result_count, self.output_bytes)
    }

    /// Current result count.
    pub fn result_count(&self) -> usize {
        self.result_count
    }

    /// Current output bytes.
    pub fn output_bytes(&self) -> usize {
        self.output_bytes
    }
}

/// Truncate a vector according to budget limits.
pub fn truncate_results<T>(items: Vec<T>, budget: &PerfBudget) -> (Vec<T>, TruncationResult) {
    if budget.max_results == 0 || items.len() <= budget.max_results {
        let count = items.len();
        return (items, TruncationResult::none(count));
    }

    let original = items.len();
    let retained = budget.max_results;
    let truncated_items: Vec<T> = items.into_iter().take(retained).collect();
    (
        truncated_items,
        TruncationResult::applied(original, retained, BudgetLimit::ResultCount),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_not_exceeded() {
        let budget = PerfBudget::default();
        assert!(!budget.is_exceeded(100, 10, 1000));
    }

    #[test]
    fn test_budget_timeout_exceeded() {
        let budget = PerfBudget::default(); // 10s timeout
        assert!(budget.is_exceeded(10_001, 0, 0));
        assert_eq!(
            budget.exceeded_limit(10_001, 0, 0),
            Some(BudgetLimit::Timeout)
        );
    }

    #[test]
    fn test_budget_result_count_exceeded() {
        let budget = PerfBudget::default(); // 200 results
        assert!(budget.is_exceeded(0, 200, 0));
        assert_eq!(
            budget.exceeded_limit(0, 200, 0),
            Some(BudgetLimit::ResultCount)
        );
    }

    #[test]
    fn test_budget_output_size_exceeded() {
        let budget = PerfBudget::default(); // 64KB output
        assert!(budget.is_exceeded(0, 0, 65_536));
        assert_eq!(
            budget.exceeded_limit(0, 0, 65_536),
            Some(BudgetLimit::OutputSize)
        );
    }

    #[test]
    fn test_budget_unlimited() {
        let budget = PerfBudget::new(); // All unlimited
        assert!(!budget.is_exceeded(999_999, 999_999, 999_999_999));
        assert_eq!(budget.exceeded_limit(999_999, 999_999, 999_999_999), None);
    }

    #[test]
    fn test_budget_builder() {
        let budget = PerfBudget::new()
            .timeout_ms(5000)
            .max_results(50)
            .max_output_bytes(16_384)
            .max_depth(5);
        assert_eq!(budget.timeout_ms, 5000);
        assert_eq!(budget.max_results, 50);
        assert_eq!(budget.max_output_bytes, 16_384);
        assert_eq!(budget.max_depth, 5);
    }

    #[test]
    fn test_budget_timeout_duration() {
        let budget = PerfBudget::new().timeout_ms(5000);
        assert_eq!(budget.timeout_duration(), Some(Duration::from_millis(5000)));

        let unlimited = PerfBudget::new();
        assert_eq!(unlimited.timeout_duration(), None);
    }

    #[test]
    fn test_retry_policy_backoff() {
        let policy = RetryPolicy::default();
        assert_eq!(policy.backoff_ms(0), 0); // First attempt, no delay
        assert_eq!(policy.backoff_ms(1), 500); // First retry
        assert_eq!(policy.backoff_ms(2), 1000); // Second retry
    }

    #[test]
    fn test_retry_policy_max_backoff() {
        let policy = RetryPolicy {
            max_retries: 10,
            initial_backoff_ms: 1000,
            backoff_multiplier: 3.0,
            max_backoff_ms: 5000,
        };
        assert_eq!(policy.backoff_ms(1), 1000);
        assert_eq!(policy.backoff_ms(2), 3000);
        assert_eq!(policy.backoff_ms(3), 5000); // Capped
        assert_eq!(policy.backoff_ms(4), 5000); // Still capped
    }

    #[test]
    fn test_retry_policy_should_retry() {
        let policy = RetryPolicy::default(); // 2 retries
        assert!(policy.should_retry(0));
        assert!(policy.should_retry(1));
        assert!(!policy.should_retry(2));
        assert!(!policy.should_retry(3));
    }

    #[test]
    fn test_truncation_no_truncation() {
        let items = vec![1, 2, 3];
        let budget = PerfBudget::new().max_results(10);
        let (result, info) = truncate_results(items, &budget);
        assert_eq!(result.len(), 3);
        assert!(!info.truncated);
        assert!(info.warning().is_none());
    }

    #[test]
    fn test_truncation_applied() {
        let items: Vec<i32> = (0..100).collect();
        let budget = PerfBudget::new().max_results(10);
        let (result, info) = truncate_results(items, &budget);
        assert_eq!(result.len(), 10);
        assert!(info.truncated);
        assert_eq!(info.original_count, 100);
        assert_eq!(info.retained_count, 10);
        assert_eq!(info.reason, Some(BudgetLimit::ResultCount));
        let warning = info.warning().unwrap();
        assert!(warning.contains("10 of 100"));
    }

    #[test]
    fn test_truncation_unlimited() {
        let items: Vec<i32> = (0..1000).collect();
        let budget = PerfBudget::new(); // No limits
        let (result, info) = truncate_results(items, &budget);
        assert_eq!(result.len(), 1000);
        assert!(!info.truncated);
    }

    #[test]
    fn test_perf_guard_basic() {
        let budget = PerfBudget::new().max_results(5);
        let mut guard = PerfGuard::new(budget);

        assert!(!guard.is_exceeded());
        guard.add_results(3, 100);
        assert!(!guard.is_exceeded());
        guard.add_results(3, 100);
        assert!(guard.is_exceeded());
        assert_eq!(guard.result_count(), 6);
        assert_eq!(guard.output_bytes(), 200);
    }

    #[test]
    fn test_perf_guard_depth() {
        let budget = PerfBudget::new().max_depth(3);
        let mut guard = PerfGuard::new(budget);

        guard.set_depth(2);
        assert!(!guard.is_exceeded());
        guard.set_depth(3);
        assert!(guard.is_exceeded());
        assert_eq!(guard.exceeded_limit(), Some(BudgetLimit::Depth));
    }

    #[test]
    fn test_budget_limit_display() {
        assert_eq!(BudgetLimit::Timeout.to_string(), "timeout");
        assert_eq!(BudgetLimit::ResultCount.to_string(), "result_count");
        assert_eq!(BudgetLimit::OutputSize.to_string(), "output_size");
        assert_eq!(BudgetLimit::Depth.to_string(), "depth");
    }

    #[test]
    fn test_budget_json_roundtrip() {
        let budget = PerfBudget::default();
        let json = serde_json::to_string(&budget).unwrap();
        let parsed: PerfBudget = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.timeout_ms, 10_000);
        assert_eq!(parsed.max_results, 200);
        assert_eq!(parsed.max_output_bytes, 65_536);
    }

    #[test]
    fn test_retry_policy_json_roundtrip() {
        let policy = RetryPolicy::default();
        let json = serde_json::to_string(&policy).unwrap();
        let parsed: RetryPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_retries, 2);
        assert_eq!(parsed.initial_backoff_ms, 500);
    }
}
