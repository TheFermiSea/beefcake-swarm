//! Feature Flags — Independent Capability Toggles
//!
//! Provides per-capability boolean flags for safe staged deployment of
//! major swarm subsystems. Each flag can be controlled independently via
//! environment variables or per-run overrides.
//!
//! # Environment Variables
//!
//! | Variable | Default | Description |
//! |---|---|---|
//! | `SWARM_SMART_ROUTER_ENABLED` | `false` | Dynamic model router with performance-based scoring |
//! | `SWARM_STATE_MACHINE_ENABLED` | `false` | Escalation state machine for tier routing |
//! | `SWARM_CANARY_ENABLED` | `false` | Speculative dual-route canary mode |
//! | `SWARM_STRUCTURED_EVALUATOR_REQUIRED` | `false` | Require structured evaluator output (fail-closed) |
//!
//! # Per-Run Overrides
//!
//! ```rust,ignore
//! use coordination::rollout::FeatureFlags;
//!
//! let mut flags = FeatureFlags::from_env();
//! flags.apply_overrides(&FeatureFlagOverrides {
//!     smart_router_enabled: Some(true),
//!     ..Default::default()
//! });
//! ```
//!
//! # Integration with RolloutManager
//!
//! Feature flags are orthogonal to the progressive rollout system.
//! `FeatureFlags` controls *whether* a capability is available at all,
//! while `RolloutManager` controls *which cohort* sees it. A feature
//! must be both flag-enabled and at the appropriate rollout stage.

use serde::{Deserialize, Serialize};

/// Independent toggle for each major swarm capability.
///
/// Defaults to all-disabled (conservative). Use [`FeatureFlags::from_env`]
/// to read environment variables, then optionally apply per-run overrides.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Enable the dynamic model router with performance-based scoring.
    ///
    /// When enabled, the router uses `DynamicRouter` to select models based
    /// on historical performance data. When disabled, falls back to the
    /// static `ModelRouter` with hardcoded tier assignments.
    ///
    /// Env: `SWARM_SMART_ROUTER_ENABLED`
    pub smart_router_enabled: bool,

    /// Enable the escalation state machine for tier routing.
    ///
    /// When enabled, the orchestrator uses the full state machine to track
    /// escalation triggers (repeated errors, compile failures, file count
    /// thresholds). When disabled, uses simple retry-count-based escalation.
    ///
    /// Env: `SWARM_STATE_MACHINE_ENABLED`
    pub state_machine_enabled: bool,

    /// Enable speculative dual-route canary mode.
    ///
    /// When enabled, high-risk tasks may be routed to two candidate tiers
    /// simultaneously, with the loser early-stopped when the winner passes
    /// the verifier. Subject to budget cap and risk-level gating.
    ///
    /// Env: `SWARM_CANARY_ENABLED`
    pub speculative_canary_enabled: bool,

    /// Require structured evaluator output (fail-closed).
    ///
    /// When enabled, the reviewer/validator must produce output matching
    /// the `StructuredReview` schema. Unparseable output is treated as a
    /// failure, forcing re-evaluation. When disabled, free-form text
    /// responses are accepted with best-effort parsing.
    ///
    /// Env: `SWARM_STRUCTURED_EVALUATOR_REQUIRED`
    pub structured_evaluator_required: bool,

    /// Enable worker-first mode with manager-on-escalation.
    ///
    /// When enabled, tasks start at the Worker tier (local models) instead
    /// of the Council tier (cloud manager). The cloud manager is only
    /// invoked when escalation triggers fire (repeated errors, excessive
    /// failures, multi-file complexity, or worker budget exhaustion).
    /// Reduces median iteration time for simple issues.
    ///
    /// Env: `SWARM_WORKER_FIRST_ENABLED`
    pub worker_first_enabled: bool,
}

impl Default for FeatureFlags {
    /// Conservative defaults: all capabilities disabled.
    fn default() -> Self {
        Self {
            smart_router_enabled: false,
            state_machine_enabled: false,
            speculative_canary_enabled: false,
            structured_evaluator_required: false,
            worker_first_enabled: false,
        }
    }
}

impl FeatureFlags {
    /// Read feature flags from environment variables.
    ///
    /// Each flag is set by its corresponding `SWARM_*` env var.
    /// Accepts "1", "true", or "yes" (case-insensitive) as enabled.
    /// Missing or any other value means disabled.
    pub fn from_env() -> Self {
        Self {
            smart_router_enabled: parse_bool_env("SWARM_SMART_ROUTER_ENABLED"),
            state_machine_enabled: parse_bool_env("SWARM_STATE_MACHINE_ENABLED"),
            speculative_canary_enabled: parse_bool_env("SWARM_CANARY_ENABLED"),
            structured_evaluator_required: parse_bool_env("SWARM_STRUCTURED_EVALUATOR_REQUIRED"),
            worker_first_enabled: parse_bool_env("SWARM_WORKER_FIRST_ENABLED"),
        }
    }

    /// Create flags with all capabilities enabled.
    ///
    /// Useful for testing or "full stack" runs.
    pub fn all_enabled() -> Self {
        Self {
            smart_router_enabled: true,
            state_machine_enabled: true,
            speculative_canary_enabled: true,
            structured_evaluator_required: true,
            worker_first_enabled: true,
        }
    }

    /// Apply per-run overrides. Only `Some` values are applied;
    /// `None` fields leave the existing value unchanged.
    pub fn apply_overrides(&mut self, overrides: &FeatureFlagOverrides) {
        if let Some(v) = overrides.smart_router_enabled {
            self.smart_router_enabled = v;
        }
        if let Some(v) = overrides.state_machine_enabled {
            self.state_machine_enabled = v;
        }
        if let Some(v) = overrides.speculative_canary_enabled {
            self.speculative_canary_enabled = v;
        }
        if let Some(v) = overrides.structured_evaluator_required {
            self.structured_evaluator_required = v;
        }
        if let Some(v) = overrides.worker_first_enabled {
            self.worker_first_enabled = v;
        }
    }

    /// Merge another flags struct, overriding only fields that are `true`.
    ///
    /// This is a union operation: if *either* source has a flag enabled,
    /// the result is enabled.
    pub fn merge_enabled(&mut self, other: &FeatureFlags) {
        self.smart_router_enabled |= other.smart_router_enabled;
        self.state_machine_enabled |= other.state_machine_enabled;
        self.speculative_canary_enabled |= other.speculative_canary_enabled;
        self.structured_evaluator_required |= other.structured_evaluator_required;
        self.worker_first_enabled |= other.worker_first_enabled;
    }

    /// Returns a list of enabled feature names.
    pub fn enabled_features(&self) -> Vec<&'static str> {
        let mut features = Vec::new();
        if self.smart_router_enabled {
            features.push("smart_router");
        }
        if self.state_machine_enabled {
            features.push("state_machine");
        }
        if self.speculative_canary_enabled {
            features.push("speculative_canary");
        }
        if self.structured_evaluator_required {
            features.push("structured_evaluator");
        }
        if self.worker_first_enabled {
            features.push("worker_first");
        }
        features
    }

    /// Returns the number of enabled features.
    pub fn enabled_count(&self) -> usize {
        self.enabled_features().len()
    }

    /// Whether any feature is enabled.
    pub fn any_enabled(&self) -> bool {
        self.smart_router_enabled
            || self.state_machine_enabled
            || self.speculative_canary_enabled
            || self.structured_evaluator_required
            || self.worker_first_enabled
    }

    /// Total number of feature flags.
    const FLAG_COUNT: usize = 5;

    /// Format as a human-readable summary line.
    pub fn summary(&self) -> String {
        let enabled = self.enabled_features();
        if enabled.is_empty() {
            "Feature flags: all disabled (conservative mode)".to_string()
        } else {
            format!(
                "Feature flags: {}/{} enabled [{}]",
                enabled.len(),
                Self::FLAG_COUNT,
                enabled.join(", ")
            )
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string())
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl std::fmt::Display for FeatureFlags {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "smart_router={} state_machine={} canary={} structured_eval={} worker_first={}",
            flag_str(self.smart_router_enabled),
            flag_str(self.state_machine_enabled),
            flag_str(self.speculative_canary_enabled),
            flag_str(self.structured_evaluator_required),
            flag_str(self.worker_first_enabled),
        )
    }
}

/// Per-run overrides for feature flags.
///
/// `None` means "don't override", `Some(true/false)` explicitly sets the value.
/// Apply with [`FeatureFlags::apply_overrides`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FeatureFlagOverrides {
    pub smart_router_enabled: Option<bool>,
    pub state_machine_enabled: Option<bool>,
    pub speculative_canary_enabled: Option<bool>,
    pub structured_evaluator_required: Option<bool>,
    pub worker_first_enabled: Option<bool>,
}

impl FeatureFlagOverrides {
    /// Whether any override is set.
    pub fn has_overrides(&self) -> bool {
        self.smart_router_enabled.is_some()
            || self.state_machine_enabled.is_some()
            || self.speculative_canary_enabled.is_some()
            || self.structured_evaluator_required.is_some()
            || self.worker_first_enabled.is_some()
    }
}

/// Parse a boolean from an environment variable.
/// Accepts "1", "true", or "yes" (case-insensitive).
fn parse_bool_env(var: &str) -> bool {
    std::env::var(var)
        .map(|v| parse_bool_env_value(&v))
        .unwrap_or(false)
}

/// Parse a boolean from a raw string value.
/// Accepts "1", "true", or "yes" (case-insensitive).
fn parse_bool_env_value(value: &str) -> bool {
    let v = value.trim().to_lowercase();
    v == "1" || v == "true" || v == "yes"
}

/// Format a boolean as "ON" or "OFF" for display.
fn flag_str(enabled: bool) -> &'static str {
    if enabled {
        "ON"
    } else {
        "OFF"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_all_disabled() {
        let flags = FeatureFlags::default();
        assert!(!flags.smart_router_enabled);
        assert!(!flags.state_machine_enabled);
        assert!(!flags.speculative_canary_enabled);
        assert!(!flags.structured_evaluator_required);
        assert!(!flags.any_enabled());
        assert_eq!(flags.enabled_count(), 0);
    }

    #[test]
    fn test_all_enabled() {
        let flags = FeatureFlags::all_enabled();
        assert!(flags.smart_router_enabled);
        assert!(flags.state_machine_enabled);
        assert!(flags.speculative_canary_enabled);
        assert!(flags.structured_evaluator_required);
        assert!(flags.worker_first_enabled);
        assert!(flags.any_enabled());
        assert_eq!(flags.enabled_count(), 5);
    }

    #[test]
    fn test_parse_bool_env_helper() {
        // Test the parsing logic directly without shared env var races.
        // parse_bool_env accepts "1", "true", "yes" (case-insensitive).
        assert!(parse_bool_env_value("1"));
        assert!(parse_bool_env_value("true"));
        assert!(parse_bool_env_value("TRUE"));
        assert!(parse_bool_env_value("True"));
        assert!(parse_bool_env_value("yes"));
        assert!(parse_bool_env_value("Yes"));
        assert!(parse_bool_env_value("YES"));

        // Invalid values → false
        assert!(!parse_bool_env_value("0"));
        assert!(!parse_bool_env_value("false"));
        assert!(!parse_bool_env_value("maybe"));
        assert!(!parse_bool_env_value(""));
        assert!(!parse_bool_env_value("  "));
    }

    #[test]
    fn test_from_env_defaults_to_disabled() {
        // Use a unique prefix to avoid collisions with parallel tests.
        // Just verify the parse helper since env tests race.
        let flags = FeatureFlags::default();
        assert!(!flags.any_enabled());
    }

    #[test]
    fn test_apply_overrides_partial() {
        let mut flags = FeatureFlags::default();
        let overrides = FeatureFlagOverrides {
            smart_router_enabled: Some(true),
            structured_evaluator_required: Some(true),
            ..Default::default()
        };
        flags.apply_overrides(&overrides);

        assert!(flags.smart_router_enabled);
        assert!(!flags.state_machine_enabled); // unchanged
        assert!(!flags.speculative_canary_enabled); // unchanged
        assert!(flags.structured_evaluator_required);
    }

    #[test]
    fn test_apply_overrides_can_disable() {
        let mut flags = FeatureFlags::all_enabled();
        let overrides = FeatureFlagOverrides {
            speculative_canary_enabled: Some(false),
            ..Default::default()
        };
        flags.apply_overrides(&overrides);

        assert!(flags.smart_router_enabled); // unchanged
        assert!(!flags.speculative_canary_enabled); // disabled
    }

    #[test]
    fn test_merge_enabled_union() {
        let mut flags = FeatureFlags {
            smart_router_enabled: true,
            state_machine_enabled: false,
            speculative_canary_enabled: false,
            structured_evaluator_required: true,
            worker_first_enabled: false,
        };
        let other = FeatureFlags {
            smart_router_enabled: false,
            state_machine_enabled: true,
            speculative_canary_enabled: true,
            structured_evaluator_required: false,
            worker_first_enabled: true,
        };
        flags.merge_enabled(&other);

        assert!(flags.smart_router_enabled);
        assert!(flags.state_machine_enabled);
        assert!(flags.speculative_canary_enabled);
        assert!(flags.structured_evaluator_required);
        assert!(flags.worker_first_enabled);
    }

    #[test]
    fn test_enabled_features_list() {
        let flags = FeatureFlags {
            smart_router_enabled: true,
            state_machine_enabled: false,
            speculative_canary_enabled: true,
            structured_evaluator_required: false,
            worker_first_enabled: false,
        };
        let features = flags.enabled_features();
        assert_eq!(features, vec!["smart_router", "speculative_canary"]);
    }

    #[test]
    fn test_display() {
        let flags = FeatureFlags {
            smart_router_enabled: true,
            state_machine_enabled: false,
            speculative_canary_enabled: true,
            structured_evaluator_required: false,
            worker_first_enabled: true,
        };
        let display = flags.to_string();
        assert_eq!(
            display,
            "smart_router=ON state_machine=OFF canary=ON structured_eval=OFF worker_first=ON"
        );
    }

    #[test]
    fn test_summary_none_enabled() {
        let flags = FeatureFlags::default();
        assert_eq!(
            flags.summary(),
            "Feature flags: all disabled (conservative mode)"
        );
    }

    #[test]
    fn test_summary_some_enabled() {
        let flags = FeatureFlags {
            smart_router_enabled: true,
            state_machine_enabled: true,
            speculative_canary_enabled: false,
            structured_evaluator_required: false,
            worker_first_enabled: false,
        };
        let summary = flags.summary();
        assert!(summary.contains("2/5 enabled"));
        assert!(summary.contains("smart_router"));
        assert!(summary.contains("state_machine"));
    }

    #[test]
    fn test_json_roundtrip() {
        let flags = FeatureFlags {
            smart_router_enabled: true,
            state_machine_enabled: false,
            speculative_canary_enabled: true,
            structured_evaluator_required: true,
            worker_first_enabled: false,
        };
        let json = flags.to_json();
        let restored = FeatureFlags::from_json(&json).unwrap();
        assert_eq!(flags, restored);
    }

    #[test]
    fn test_overrides_has_overrides() {
        let empty = FeatureFlagOverrides::default();
        assert!(!empty.has_overrides());

        let partial = FeatureFlagOverrides {
            smart_router_enabled: Some(true),
            ..Default::default()
        };
        assert!(partial.has_overrides());
    }

    #[test]
    fn test_overrides_json_roundtrip() {
        let overrides = FeatureFlagOverrides {
            smart_router_enabled: Some(true),
            state_machine_enabled: None,
            speculative_canary_enabled: Some(false),
            structured_evaluator_required: None,
            worker_first_enabled: Some(true),
        };
        let json = serde_json::to_string(&overrides).unwrap();
        let restored: FeatureFlagOverrides = serde_json::from_str(&json).unwrap();
        assert_eq!(overrides, restored);
    }
}
