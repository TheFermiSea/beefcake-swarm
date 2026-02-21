//! Resilience — Tool Failure Degraded Mode
//!
//! Provides types and logic for graceful degradation when tools or services
//! fail. Instead of hard errors, tools can return degraded responses with
//! confidence levels, warnings, and fallback metadata.
//!
//! # Design
//!
//! ```text
//! Tool call
//!   ├─ Primary succeeds → DegradedResponse { level: Full, ... }
//!   ├─ Primary fails, fallback succeeds → DegradedResponse { level: Partial, warnings, ... }
//!   └─ All fail → DegradedResponse { level: Unavailable, ... }
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use coordination::resilience::{DegradedResponse, DegradationLevel, FallbackChain};
//!
//! let chain = FallbackChain::new("ast_analysis")
//!     .add_tier("ast_grep", 1.0)
//!     .add_tier("regex_grep", 0.7)
//!     .add_tier("text_search", 0.4);
//!
//! // Try each tier, return first success with appropriate confidence
//! let response = chain.execute(|tier| try_tool(tier));
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// How much of the tool's capability is available.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum DegradationLevel {
    /// All capabilities available, primary tool succeeded.
    Full,
    /// Reduced capabilities — using a fallback with lower fidelity.
    Partial,
    /// Tool completely unavailable — returning best-effort or empty result.
    Unavailable,
}

impl std::fmt::Display for DegradationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Full => write!(f, "full"),
            Self::Partial => write!(f, "partial"),
            Self::Unavailable => write!(f, "unavailable"),
        }
    }
}

/// A tool response wrapped with degradation metadata.
///
/// Consumers can check `level` and `confidence` to decide how much to
/// trust the result, and `warnings` for user-facing diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradedResponse<T> {
    /// The actual response payload.
    pub payload: T,
    /// Current degradation level.
    pub level: DegradationLevel,
    /// Confidence in the response (0.0–1.0).
    /// - Full: 1.0
    /// - Partial: depends on fallback tier
    /// - Unavailable: 0.0
    pub confidence: f64,
    /// Which service tier produced this response.
    pub served_by: String,
    /// Warning messages for the consumer.
    pub warnings: Vec<String>,
    /// When this response was produced.
    pub timestamp: DateTime<Utc>,
}

impl<T> DegradedResponse<T> {
    /// Create a full-confidence response from a primary tool.
    pub fn full(payload: T, served_by: &str) -> Self {
        Self {
            payload,
            level: DegradationLevel::Full,
            confidence: 1.0,
            served_by: served_by.to_string(),
            warnings: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    /// Create a partial-confidence response from a fallback.
    pub fn partial(payload: T, served_by: &str, confidence: f64, warning: &str) -> Self {
        Self {
            payload,
            level: DegradationLevel::Partial,
            confidence: confidence.clamp(0.0, 1.0),
            served_by: served_by.to_string(),
            warnings: vec![warning.to_string()],
            timestamp: Utc::now(),
        }
    }

    /// Create an unavailable response with a best-effort payload.
    pub fn unavailable(payload: T, warning: &str) -> Self {
        Self {
            payload,
            level: DegradationLevel::Unavailable,
            confidence: 0.0,
            served_by: "none".to_string(),
            warnings: vec![warning.to_string()],
            timestamp: Utc::now(),
        }
    }

    /// Whether this response is at full confidence.
    pub fn is_full(&self) -> bool {
        self.level == DegradationLevel::Full
    }

    /// Whether any degradation has occurred.
    pub fn is_degraded(&self) -> bool {
        self.level != DegradationLevel::Full
    }
}

/// A tier in a fallback chain.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackTier {
    /// Identifier for this tier (e.g., "ast_grep", "regex_search").
    pub name: String,
    /// Confidence factor when this tier serves the response (0.0–1.0).
    pub confidence: f64,
}

/// Ordered chain of fallback service tiers for a tool.
///
/// When the primary tier fails, the chain tries each subsequent tier
/// in order, returning the first success with adjusted confidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackChain {
    /// Name of the tool this chain serves.
    pub tool_name: String,
    /// Ordered tiers from highest to lowest fidelity.
    pub tiers: Vec<FallbackTier>,
}

impl FallbackChain {
    /// Create a new chain for a tool.
    pub fn new(tool_name: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            tiers: Vec::new(),
        }
    }

    /// Add a fallback tier with a confidence factor.
    pub fn add_tier(mut self, name: &str, confidence: f64) -> Self {
        self.tiers.push(FallbackTier {
            name: name.to_string(),
            confidence: confidence.clamp(0.0, 1.0),
        });
        self
    }

    /// Execute the fallback chain with a closure that tries each tier.
    ///
    /// The closure receives the tier name and returns Ok(result) or Err(reason).
    /// Returns a DegradedResponse wrapping the first successful result.
    pub fn execute<T, F>(&self, mut try_fn: F) -> DegradedResponse<Option<T>>
    where
        F: FnMut(&str) -> Result<T, String>,
    {
        let mut warnings = Vec::new();

        for (idx, tier) in self.tiers.iter().enumerate() {
            match try_fn(&tier.name) {
                Ok(result) => {
                    if idx == 0 {
                        return DegradedResponse::full(Some(result), &tier.name);
                    }
                    let warning = format!(
                        "{}: primary tier(s) failed, using fallback '{}'",
                        self.tool_name, tier.name
                    );
                    warnings.push(warning.clone());
                    let mut resp =
                        DegradedResponse::partial(Some(result), &tier.name, tier.confidence, "");
                    resp.warnings = warnings;
                    return resp;
                }
                Err(reason) => {
                    warnings.push(format!(
                        "{} '{}' failed: {}",
                        self.tool_name, tier.name, reason
                    ));
                }
            }
        }

        // All tiers failed
        let mut resp = DegradedResponse::unavailable(
            None,
            &format!(
                "{}: all {} tiers exhausted",
                self.tool_name,
                self.tiers.len()
            ),
        );
        resp.warnings = warnings;
        resp
    }

    /// Number of tiers in the chain.
    pub fn tier_count(&self) -> usize {
        self.tiers.len()
    }
}

/// Health status of a tool, tracked over time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolHealth {
    /// Tool identifier.
    pub tool_name: String,
    /// Current degradation level.
    pub level: DegradationLevel,
    /// Consecutive successes since last failure.
    pub consecutive_successes: u32,
    /// Consecutive failures since last success.
    pub consecutive_failures: u32,
    /// Total calls made.
    pub total_calls: u64,
    /// Total failures.
    pub total_failures: u64,
    /// Last observed error message.
    pub last_error: Option<String>,
    /// When last status change occurred.
    pub last_change: DateTime<Utc>,
}

impl ToolHealth {
    /// Create a new healthy tool tracker.
    pub fn new(tool_name: &str) -> Self {
        Self {
            tool_name: tool_name.to_string(),
            level: DegradationLevel::Full,
            consecutive_successes: 0,
            consecutive_failures: 0,
            total_calls: 0,
            total_failures: 0,
            last_error: None,
            last_change: Utc::now(),
        }
    }

    /// Record a successful call.
    pub fn record_success(&mut self) {
        self.total_calls += 1;
        self.consecutive_successes += 1;
        self.consecutive_failures = 0;

        // Recover from degraded state after 3 consecutive successes
        if self.level != DegradationLevel::Full && self.consecutive_successes >= 3 {
            self.level = DegradationLevel::Full;
            self.last_change = Utc::now();
            self.last_error = None;
        }
    }

    /// Record a failed call.
    pub fn record_failure(&mut self, error: &str) {
        self.total_calls += 1;
        self.total_failures += 1;
        self.consecutive_failures += 1;
        self.consecutive_successes = 0;
        self.last_error = Some(error.to_string());

        let new_level = if self.consecutive_failures >= 3 {
            DegradationLevel::Unavailable
        } else if self.consecutive_failures >= 1 {
            DegradationLevel::Partial
        } else {
            DegradationLevel::Full
        };

        if new_level != self.level {
            self.level = new_level;
            self.last_change = Utc::now();
        }
    }

    /// Failure rate as a fraction (0.0–1.0).
    pub fn failure_rate(&self) -> f64 {
        if self.total_calls == 0 {
            0.0
        } else {
            self.total_failures as f64 / self.total_calls as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_degraded_response_full() {
        let resp = DegradedResponse::full("hello", "primary");
        assert!(resp.is_full());
        assert!(!resp.is_degraded());
        assert_eq!(resp.confidence, 1.0);
        assert_eq!(resp.served_by, "primary");
        assert!(resp.warnings.is_empty());
    }

    #[test]
    fn test_degraded_response_partial() {
        let resp = DegradedResponse::partial("fallback result", "regex", 0.7, "using fallback");
        assert!(!resp.is_full());
        assert!(resp.is_degraded());
        assert_eq!(resp.confidence, 0.7);
        assert_eq!(resp.warnings.len(), 1);
    }

    #[test]
    fn test_degraded_response_unavailable() {
        let resp: DegradedResponse<String> =
            DegradedResponse::unavailable("empty".to_string(), "all failed");
        assert!(resp.is_degraded());
        assert_eq!(resp.level, DegradationLevel::Unavailable);
        assert_eq!(resp.confidence, 0.0);
    }

    #[test]
    fn test_confidence_clamping() {
        let resp = DegradedResponse::partial(42, "tier", 1.5, "over 1");
        assert_eq!(resp.confidence, 1.0);

        let resp = DegradedResponse::partial(42, "tier", -0.5, "under 0");
        assert_eq!(resp.confidence, 0.0);
    }

    #[test]
    fn test_fallback_chain_primary_succeeds() {
        let chain = FallbackChain::new("test_tool")
            .add_tier("primary", 1.0)
            .add_tier("fallback", 0.5);

        let resp = chain.execute(|tier| {
            if tier == "primary" {
                Ok("primary result")
            } else {
                Err("not called".to_string())
            }
        });

        assert!(resp.is_full());
        assert_eq!(resp.payload, Some("primary result"));
        assert_eq!(resp.served_by, "primary");
        assert!(resp.warnings.is_empty());
    }

    #[test]
    fn test_fallback_chain_falls_to_second() {
        let chain = FallbackChain::new("analysis")
            .add_tier("ast_grep", 1.0)
            .add_tier("regex", 0.7)
            .add_tier("text", 0.3);

        let resp = chain.execute(|tier| match tier {
            "ast_grep" => Err("ast_grep unavailable".to_string()),
            "regex" => Ok("regex result"),
            _ => Err("not reached".to_string()),
        });

        assert!(resp.is_degraded());
        assert_eq!(resp.level, DegradationLevel::Partial);
        assert_eq!(resp.payload, Some("regex result"));
        assert_eq!(resp.served_by, "regex");
        assert_eq!(resp.confidence, 0.7);
        assert_eq!(resp.warnings.len(), 2); // ast_grep failure + fallback notice
    }

    #[test]
    fn test_fallback_chain_all_fail() {
        let chain = FallbackChain::new("tool")
            .add_tier("a", 1.0)
            .add_tier("b", 0.5);

        let resp: DegradedResponse<Option<String>> = chain.execute(|_| Err("fail".to_string()));

        assert_eq!(resp.level, DegradationLevel::Unavailable);
        assert!(resp.payload.is_none());
        assert_eq!(resp.confidence, 0.0);
        assert_eq!(resp.warnings.len(), 2); // One per failed tier
    }

    #[test]
    fn test_fallback_chain_empty() {
        let chain: FallbackChain = FallbackChain::new("empty_tool");

        let resp: DegradedResponse<Option<String>> = chain.execute(|_| Ok("never".to_string()));

        assert_eq!(resp.level, DegradationLevel::Unavailable);
        assert!(resp.payload.is_none());
    }

    #[test]
    fn test_tool_health_new() {
        let health = ToolHealth::new("ast_grep");
        assert_eq!(health.level, DegradationLevel::Full);
        assert_eq!(health.consecutive_failures, 0);
        assert_eq!(health.failure_rate(), 0.0);
    }

    #[test]
    fn test_tool_health_degrades_on_failure() {
        let mut health = ToolHealth::new("tool");

        health.record_failure("timeout");
        assert_eq!(health.level, DegradationLevel::Partial);
        assert_eq!(health.consecutive_failures, 1);

        health.record_failure("timeout");
        assert_eq!(health.level, DegradationLevel::Partial);

        health.record_failure("timeout");
        assert_eq!(health.level, DegradationLevel::Unavailable);
        assert_eq!(health.consecutive_failures, 3);
    }

    #[test]
    fn test_tool_health_recovers() {
        let mut health = ToolHealth::new("tool");

        // Degrade to unavailable
        for _ in 0..3 {
            health.record_failure("err");
        }
        assert_eq!(health.level, DegradationLevel::Unavailable);

        // Recover after 3 successes
        health.record_success();
        assert_eq!(health.level, DegradationLevel::Unavailable); // Not yet
        health.record_success();
        assert_eq!(health.level, DegradationLevel::Unavailable); // Not yet
        health.record_success();
        assert_eq!(health.level, DegradationLevel::Full); // Recovered
        assert!(health.last_error.is_none());
    }

    #[test]
    fn test_tool_health_failure_rate() {
        let mut health = ToolHealth::new("tool");

        health.record_success();
        health.record_failure("err");
        health.record_success();
        health.record_failure("err");

        assert_eq!(health.total_calls, 4);
        assert_eq!(health.total_failures, 2);
        assert!((health.failure_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tool_health_interleaved() {
        let mut health = ToolHealth::new("tool");

        health.record_failure("first");
        assert_eq!(health.level, DegradationLevel::Partial);

        health.record_success();
        assert_eq!(health.consecutive_failures, 0);
        assert_eq!(health.consecutive_successes, 1);
        // Still partial because we need 3 consecutive successes
        assert_eq!(health.level, DegradationLevel::Partial);
    }

    #[test]
    fn test_degradation_level_ordering() {
        assert!(DegradationLevel::Full < DegradationLevel::Partial);
        assert!(DegradationLevel::Partial < DegradationLevel::Unavailable);
    }

    #[test]
    fn test_degradation_level_display() {
        assert_eq!(DegradationLevel::Full.to_string(), "full");
        assert_eq!(DegradationLevel::Partial.to_string(), "partial");
        assert_eq!(DegradationLevel::Unavailable.to_string(), "unavailable");
    }

    #[test]
    fn test_fallback_chain_tier_count() {
        let chain = FallbackChain::new("tool")
            .add_tier("a", 1.0)
            .add_tier("b", 0.5)
            .add_tier("c", 0.2);
        assert_eq!(chain.tier_count(), 3);
    }

    #[test]
    fn test_degraded_response_json_roundtrip() {
        let resp = DegradedResponse::partial("test data".to_string(), "fallback", 0.7, "warning");
        let json = serde_json::to_string(&resp).unwrap();
        let parsed: DegradedResponse<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.payload, "test data");
        assert_eq!(parsed.confidence, 0.7);
        assert_eq!(parsed.level, DegradationLevel::Partial);
    }
}
