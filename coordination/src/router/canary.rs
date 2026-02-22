//! Speculative dual-route canary mode.
//!
//! For high-risk tasks, launches two candidate model routes in parallel.
//! The "winner" is the first route to achieve a passing verifier result
//! (or the route with fewer errors if neither passes). The loser is
//! early-stopped when the winner's confidence threshold is reached.
//!
//! Gated behind `SWARM_CANARY_ENABLED=1` feature flag with a strict
//! per-session budget cap.
//!
//! # Usage
//!
//! ```ignore
//! use coordination::router::canary::{CanaryConfig, CanarySession, CanaryRoute};
//!
//! if CanaryConfig::from_env().enabled {
//!     let session = CanarySession::new(config);
//!     session.record_route_result(route_a, result_a);
//!     session.record_route_result(route_b, result_b);
//!     let outcome = session.evaluate();
//! }
//! ```

use std::fmt;
use std::time::Duration;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use super::task_classifier::ModelTier;

/// Default budget cap per canary session (in estimated tokens).
const DEFAULT_BUDGET_CAP: u64 = 50_000;

/// Default confidence threshold to early-stop the loser route.
const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.85;

/// Configuration for canary mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryConfig {
    /// Whether canary mode is enabled.
    pub enabled: bool,
    /// Maximum token budget for both routes combined.
    pub budget_cap: u64,
    /// Confidence threshold — if one route achieves this pass rate,
    /// the other route is considered the loser and should be stopped.
    pub confidence_threshold: f64,
    /// Minimum risk level to trigger canary mode.
    /// Only tasks at or above this risk level use dual routing.
    pub min_risk_level: CanaryRiskThreshold,
}

/// Minimum risk level for canary activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CanaryRiskThreshold {
    /// Only critical-risk tasks.
    Critical,
    /// High and critical risk tasks.
    High,
    /// Medium, high, and critical (default).
    Medium,
}

impl CanaryConfig {
    /// Load canary configuration from environment variables.
    ///
    /// - `SWARM_CANARY_ENABLED`: "1" or "true" to enable (default: disabled)
    /// - `SWARM_CANARY_BUDGET_CAP`: max tokens for both routes (default: 50000)
    /// - `SWARM_CANARY_CONFIDENCE`: threshold 0.0-1.0 (default: 0.85)
    /// - `SWARM_CANARY_MIN_RISK`: "critical"|"high"|"medium" (default: "high")
    pub fn from_env() -> Self {
        let enabled = std::env::var("SWARM_CANARY_ENABLED")
            .ok()
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let budget_cap = std::env::var("SWARM_CANARY_BUDGET_CAP")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(DEFAULT_BUDGET_CAP);

        let confidence_threshold = std::env::var("SWARM_CANARY_CONFIDENCE")
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            .unwrap_or(DEFAULT_CONFIDENCE_THRESHOLD);

        let min_risk_level = std::env::var("SWARM_CANARY_MIN_RISK")
            .ok()
            .and_then(|v| match v.to_lowercase().as_str() {
                "critical" => Some(CanaryRiskThreshold::Critical),
                "high" => Some(CanaryRiskThreshold::High),
                "medium" => Some(CanaryRiskThreshold::Medium),
                _ => None,
            })
            .unwrap_or(CanaryRiskThreshold::High);

        Self {
            enabled,
            budget_cap,
            confidence_threshold,
            min_risk_level,
        }
    }

    /// Check if canary mode should activate for a given risk level.
    pub fn should_activate(&self, risk: &super::classifier::RiskLevel) -> bool {
        if !self.enabled {
            return false;
        }
        match self.min_risk_level {
            CanaryRiskThreshold::Critical => {
                matches!(risk, super::classifier::RiskLevel::Critical)
            }
            CanaryRiskThreshold::High => matches!(
                risk,
                super::classifier::RiskLevel::Critical | super::classifier::RiskLevel::High
            ),
            CanaryRiskThreshold::Medium => matches!(
                risk,
                super::classifier::RiskLevel::Critical
                    | super::classifier::RiskLevel::High
                    | super::classifier::RiskLevel::Medium
            ),
        }
    }
}

impl Default for CanaryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            budget_cap: DEFAULT_BUDGET_CAP,
            confidence_threshold: DEFAULT_CONFIDENCE_THRESHOLD,
            min_risk_level: CanaryRiskThreshold::High,
        }
    }
}

/// Identifies which route in a canary session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RouteLabel {
    /// The primary (default) route.
    Primary,
    /// The alternative (canary) route.
    Canary,
}

impl fmt::Display for RouteLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Primary => write!(f, "primary"),
            Self::Canary => write!(f, "canary"),
        }
    }
}

/// A candidate route in the canary session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryRoute {
    /// Which route this is.
    pub label: RouteLabel,
    /// Model tier used.
    pub tier: ModelTier,
    /// Model identifier.
    pub model_id: String,
    /// Temperature setting.
    pub temperature: f32,
}

/// Result of running a route.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResult {
    /// Which route produced this result.
    pub label: RouteLabel,
    /// Whether the verifier passed.
    pub verifier_passed: bool,
    /// Number of compilation errors remaining.
    pub error_count: u32,
    /// Estimated tokens consumed by this route.
    pub tokens_used: u64,
    /// Wall-clock duration of this route.
    pub duration: Duration,
    /// Number of iterations used.
    pub iterations: u32,
}

/// Outcome of the canary comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CanaryOutcome {
    /// One route clearly won.
    Winner {
        winner: RouteLabel,
        loser: RouteLabel,
        reason: String,
    },
    /// Both routes achieved the same result.
    Tie { reason: String },
    /// Budget was exceeded before either route completed.
    BudgetExceeded { tokens_used: u64, budget_cap: u64 },
    /// Canary was not activated (risk too low or disabled).
    Skipped,
}

/// Telemetry record for a canary session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanaryTelemetry {
    /// Session identifier.
    pub session_id: String,
    /// Task description.
    pub task_summary: String,
    /// Routes configured.
    pub routes: Vec<CanaryRoute>,
    /// Results from each route.
    pub results: Vec<RouteResult>,
    /// Final outcome.
    pub outcome: CanaryOutcome,
    /// Total tokens consumed across both routes.
    pub total_tokens: u64,
    /// Total wall-clock time.
    pub total_duration: Duration,
    /// Timestamp.
    pub timestamp: DateTime<Utc>,
}

/// Manages a single canary comparison session.
pub struct CanarySession {
    config: CanaryConfig,
    session_id: String,
    task_summary: String,
    routes: Vec<CanaryRoute>,
    results: Vec<RouteResult>,
    started_at: DateTime<Utc>,
}

impl CanarySession {
    /// Create a new canary session.
    pub fn new(config: CanaryConfig, session_id: String, task_summary: String) -> Self {
        Self {
            config,
            session_id,
            task_summary,
            routes: Vec::new(),
            results: Vec::new(),
            started_at: Utc::now(),
        }
    }

    /// Register a route in this session.
    pub fn add_route(&mut self, route: CanaryRoute) {
        self.routes.push(route);
    }

    /// Record the result of running a route.
    pub fn record_result(&mut self, result: RouteResult) {
        self.results.push(result);
    }

    /// Check if the combined budget has been exceeded.
    pub fn budget_exceeded(&self) -> bool {
        let total: u64 = self.results.iter().map(|r| r.tokens_used).sum();
        total > self.config.budget_cap
    }

    /// Check if either route has met the confidence threshold
    /// and the other should be early-stopped.
    ///
    /// Returns `Some(winner_label)` if a winner is identified.
    pub fn check_early_stop(&self) -> Option<RouteLabel> {
        // Need at least one result to evaluate.
        if self.results.is_empty() {
            return None;
        }

        for result in &self.results {
            if result.verifier_passed {
                return Some(result.label);
            }
        }

        None
    }

    /// Evaluate the final canary outcome after both routes complete.
    pub fn evaluate(&self) -> CanaryOutcome {
        if self.budget_exceeded() {
            let total: u64 = self.results.iter().map(|r| r.tokens_used).sum();
            return CanaryOutcome::BudgetExceeded {
                tokens_used: total,
                budget_cap: self.config.budget_cap,
            };
        }

        match self.results.len() {
            0 => CanaryOutcome::Skipped,
            1 => {
                let r = &self.results[0];
                let other = if r.label == RouteLabel::Primary {
                    RouteLabel::Canary
                } else {
                    RouteLabel::Primary
                };
                CanaryOutcome::Winner {
                    winner: r.label,
                    loser: other,
                    reason: "Only one route completed".to_string(),
                }
            }
            _ => self.compare_results(),
        }
    }

    /// Generate telemetry record for this session.
    pub fn telemetry(&self) -> CanaryTelemetry {
        let total_tokens: u64 = self.results.iter().map(|r| r.tokens_used).sum();
        let total_duration: Duration = self.results.iter().map(|r| r.duration).sum();

        CanaryTelemetry {
            session_id: self.session_id.clone(),
            task_summary: self.task_summary.clone(),
            routes: self.routes.clone(),
            results: self.results.clone(),
            outcome: self.evaluate(),
            total_tokens,
            total_duration,
            timestamp: self.started_at,
        }
    }

    /// Compare two route results and determine a winner.
    fn compare_results(&self) -> CanaryOutcome {
        let a = &self.results[0];
        let b = &self.results[1];

        // 1. Verifier pass takes priority.
        match (a.verifier_passed, b.verifier_passed) {
            (true, false) => {
                return CanaryOutcome::Winner {
                    winner: a.label,
                    loser: b.label,
                    reason: format!("{} passed verifier, {} did not", a.label, b.label),
                };
            }
            (false, true) => {
                return CanaryOutcome::Winner {
                    winner: b.label,
                    loser: a.label,
                    reason: format!("{} passed verifier, {} did not", b.label, a.label),
                };
            }
            _ => {}
        }

        // 2. Both passed or both failed — compare error counts.
        if a.error_count != b.error_count {
            let (winner, loser) = if a.error_count < b.error_count {
                (a, b)
            } else {
                (b, a)
            };
            return CanaryOutcome::Winner {
                winner: winner.label,
                loser: loser.label,
                reason: format!(
                    "{} had {} errors vs {} had {} errors",
                    winner.label, winner.error_count, loser.label, loser.error_count
                ),
            };
        }

        // 3. Same error count — compare tokens (prefer cheaper).
        if a.tokens_used != b.tokens_used {
            let (winner, loser) = if a.tokens_used < b.tokens_used {
                (a, b)
            } else {
                (b, a)
            };
            return CanaryOutcome::Winner {
                winner: winner.label,
                loser: loser.label,
                reason: format!(
                    "{} used {} tokens vs {} used {} tokens (cheaper wins)",
                    winner.label, winner.tokens_used, loser.label, loser.tokens_used
                ),
            };
        }

        // 4. Identical — tie.
        CanaryOutcome::Tie {
            reason: "Both routes achieved identical results".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(enabled: bool) -> CanaryConfig {
        CanaryConfig {
            enabled,
            budget_cap: 10_000,
            confidence_threshold: 0.85,
            min_risk_level: CanaryRiskThreshold::High,
        }
    }

    fn make_route(label: RouteLabel, tier: ModelTier) -> CanaryRoute {
        CanaryRoute {
            label,
            tier,
            model_id: format!("{}-model", label),
            temperature: 0.3,
        }
    }

    fn make_result(label: RouteLabel, passed: bool, errors: u32, tokens: u64) -> RouteResult {
        RouteResult {
            label,
            verifier_passed: passed,
            error_count: errors,
            tokens_used: tokens,
            duration: Duration::from_secs(10),
            iterations: 1,
        }
    }

    #[test]
    fn test_config_defaults() {
        let config = CanaryConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.budget_cap, DEFAULT_BUDGET_CAP);
        assert_eq!(config.confidence_threshold, DEFAULT_CONFIDENCE_THRESHOLD);
    }

    #[test]
    fn test_config_from_env_disabled_by_default() {
        std::env::remove_var("SWARM_CANARY_ENABLED");
        let config = CanaryConfig::from_env();
        assert!(!config.enabled);
    }

    #[test]
    fn test_should_activate_disabled() {
        let config = make_config(false);
        assert!(!config.should_activate(&super::super::classifier::RiskLevel::Critical));
    }

    #[test]
    fn test_should_activate_high_threshold() {
        let config = make_config(true);
        assert!(config.should_activate(&super::super::classifier::RiskLevel::Critical));
        assert!(config.should_activate(&super::super::classifier::RiskLevel::High));
        assert!(!config.should_activate(&super::super::classifier::RiskLevel::Medium));
        assert!(!config.should_activate(&super::super::classifier::RiskLevel::Low));
    }

    #[test]
    fn test_should_activate_medium_threshold() {
        let mut config = make_config(true);
        config.min_risk_level = CanaryRiskThreshold::Medium;
        assert!(config.should_activate(&super::super::classifier::RiskLevel::Critical));
        assert!(config.should_activate(&super::super::classifier::RiskLevel::High));
        assert!(config.should_activate(&super::super::classifier::RiskLevel::Medium));
        assert!(!config.should_activate(&super::super::classifier::RiskLevel::Low));
    }

    #[test]
    fn test_should_activate_critical_threshold() {
        let mut config = make_config(true);
        config.min_risk_level = CanaryRiskThreshold::Critical;
        assert!(config.should_activate(&super::super::classifier::RiskLevel::Critical));
        assert!(!config.should_activate(&super::super::classifier::RiskLevel::High));
    }

    #[test]
    fn test_session_evaluate_no_results() {
        let session = CanarySession::new(
            make_config(true),
            "test-1".to_string(),
            "test task".to_string(),
        );
        assert!(matches!(session.evaluate(), CanaryOutcome::Skipped));
    }

    #[test]
    fn test_session_evaluate_single_result() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-2".to_string(),
            "test task".to_string(),
        );
        session.add_route(make_route(RouteLabel::Primary, ModelTier::Worker));
        session.record_result(make_result(RouteLabel::Primary, true, 0, 500));

        match session.evaluate() {
            CanaryOutcome::Winner { winner, .. } => {
                assert_eq!(winner, RouteLabel::Primary);
            }
            other => panic!("Expected Winner, got {:?}", other),
        }
    }

    #[test]
    fn test_session_evaluate_verifier_winner() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-3".to_string(),
            "test task".to_string(),
        );
        session.add_route(make_route(RouteLabel::Primary, ModelTier::Worker));
        session.add_route(make_route(RouteLabel::Canary, ModelTier::Council));
        session.record_result(make_result(RouteLabel::Primary, false, 3, 1000));
        session.record_result(make_result(RouteLabel::Canary, true, 0, 2000));

        match session.evaluate() {
            CanaryOutcome::Winner { winner, loser, .. } => {
                assert_eq!(winner, RouteLabel::Canary);
                assert_eq!(loser, RouteLabel::Primary);
            }
            other => panic!("Expected Winner, got {:?}", other),
        }
    }

    #[test]
    fn test_session_evaluate_error_count_winner() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-4".to_string(),
            "test task".to_string(),
        );
        session.add_route(make_route(RouteLabel::Primary, ModelTier::Worker));
        session.add_route(make_route(RouteLabel::Canary, ModelTier::Council));
        // Both fail, but primary has fewer errors.
        session.record_result(make_result(RouteLabel::Primary, false, 1, 1000));
        session.record_result(make_result(RouteLabel::Canary, false, 5, 1000));

        match session.evaluate() {
            CanaryOutcome::Winner { winner, .. } => {
                assert_eq!(winner, RouteLabel::Primary);
            }
            other => panic!("Expected Winner, got {:?}", other),
        }
    }

    #[test]
    fn test_session_evaluate_token_tiebreaker() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-5".to_string(),
            "test task".to_string(),
        );
        session.add_route(make_route(RouteLabel::Primary, ModelTier::Worker));
        session.add_route(make_route(RouteLabel::Canary, ModelTier::Council));
        // Both pass with 0 errors, but canary used fewer tokens.
        session.record_result(make_result(RouteLabel::Primary, true, 0, 5000));
        session.record_result(make_result(RouteLabel::Canary, true, 0, 2000));

        match session.evaluate() {
            CanaryOutcome::Winner { winner, .. } => {
                assert_eq!(winner, RouteLabel::Canary);
            }
            other => panic!("Expected Winner, got {:?}", other),
        }
    }

    #[test]
    fn test_session_evaluate_tie() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-6".to_string(),
            "test task".to_string(),
        );
        session.add_route(make_route(RouteLabel::Primary, ModelTier::Worker));
        session.add_route(make_route(RouteLabel::Canary, ModelTier::Council));
        session.record_result(make_result(RouteLabel::Primary, true, 0, 1000));
        session.record_result(make_result(RouteLabel::Canary, true, 0, 1000));

        assert!(matches!(session.evaluate(), CanaryOutcome::Tie { .. }));
    }

    #[test]
    fn test_session_budget_exceeded() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-7".to_string(),
            "test task".to_string(),
        );
        // Budget is 10_000.
        session.record_result(make_result(RouteLabel::Primary, false, 5, 6000));
        session.record_result(make_result(RouteLabel::Canary, false, 3, 6000));

        assert!(session.budget_exceeded());
        assert!(matches!(
            session.evaluate(),
            CanaryOutcome::BudgetExceeded { .. }
        ));
    }

    #[test]
    fn test_session_early_stop_on_verifier_pass() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-8".to_string(),
            "test task".to_string(),
        );
        session.record_result(make_result(RouteLabel::Primary, true, 0, 500));

        assert_eq!(session.check_early_stop(), Some(RouteLabel::Primary));
    }

    #[test]
    fn test_session_no_early_stop_when_failing() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-9".to_string(),
            "test task".to_string(),
        );
        session.record_result(make_result(RouteLabel::Primary, false, 3, 500));

        assert_eq!(session.check_early_stop(), None);
    }

    #[test]
    fn test_telemetry_generation() {
        let mut session = CanarySession::new(
            make_config(true),
            "test-10".to_string(),
            "compare workers".to_string(),
        );
        session.add_route(make_route(RouteLabel::Primary, ModelTier::Worker));
        session.add_route(make_route(RouteLabel::Canary, ModelTier::Council));
        session.record_result(make_result(RouteLabel::Primary, true, 0, 1000));
        session.record_result(make_result(RouteLabel::Canary, true, 0, 2000));

        let telemetry = session.telemetry();
        assert_eq!(telemetry.session_id, "test-10");
        assert_eq!(telemetry.routes.len(), 2);
        assert_eq!(telemetry.results.len(), 2);
        assert_eq!(telemetry.total_tokens, 3000);
    }

    #[test]
    fn test_route_label_display() {
        assert_eq!(format!("{}", RouteLabel::Primary), "primary");
        assert_eq!(format!("{}", RouteLabel::Canary), "canary");
    }
}
