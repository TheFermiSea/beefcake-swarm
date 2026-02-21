//! Canary Rollout — Feature-Flagged Progressive Deployment
//!
//! Deterministic state machine for progressive feature rollout with
//! cohort-based controls. Supports advance/rollback by cohort with
//! safety gates at each stage.
//!
//! # Rollout Stages
//!
//! ```text
//! Disabled → Canary (5%) → Staging (25%) → Production (100%)
//!    ↑          │               │               │
//!    └──────────┴───────────────┴───────────────┘
//!                    (rollback at any stage)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use coordination::rollout::{RolloutManager, FeatureFlag, Cohort};
//!
//! let mut mgr = RolloutManager::new();
//! mgr.register("debate_loop", "Debate-based review loop");
//! mgr.advance("debate_loop").unwrap();  // Disabled → Canary
//! assert!(mgr.is_enabled("debate_loop", &Cohort::Canary));
//! assert!(!mgr.is_enabled("debate_loop", &Cohort::Production));
//! ```

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Deployment stage in the rollout pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum RolloutStage {
    /// Feature is off for all cohorts.
    Disabled,
    /// Enabled for canary cohort only (~5% of traffic).
    Canary,
    /// Enabled for canary + staging cohorts (~25%).
    Staging,
    /// Enabled for all cohorts (100%).
    Production,
}

impl RolloutStage {
    /// Next stage in the pipeline, if any.
    pub fn next(self) -> Option<Self> {
        match self {
            Self::Disabled => Some(Self::Canary),
            Self::Canary => Some(Self::Staging),
            Self::Staging => Some(Self::Production),
            Self::Production => None,
        }
    }

    /// Previous stage in the pipeline, if any.
    pub fn prev(self) -> Option<Self> {
        match self {
            Self::Disabled => None,
            Self::Canary => Some(Self::Disabled),
            Self::Staging => Some(Self::Canary),
            Self::Production => Some(Self::Staging),
        }
    }
}

impl std::fmt::Display for RolloutStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disabled => write!(f, "disabled"),
            Self::Canary => write!(f, "canary"),
            Self::Staging => write!(f, "staging"),
            Self::Production => write!(f, "production"),
        }
    }
}

/// Traffic cohort for feature targeting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cohort {
    /// Early adopter cohort (~5%).
    Canary,
    /// Broader validation cohort (~25%).
    Staging,
    /// All remaining traffic.
    Production,
}

impl Cohort {
    /// Minimum rollout stage required for this cohort to see a feature.
    pub fn required_stage(self) -> RolloutStage {
        match self {
            Self::Canary => RolloutStage::Canary,
            Self::Staging => RolloutStage::Staging,
            Self::Production => RolloutStage::Production,
        }
    }
}

impl std::fmt::Display for Cohort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Canary => write!(f, "canary"),
            Self::Staging => write!(f, "staging"),
            Self::Production => write!(f, "production"),
        }
    }
}

/// A feature flag with rollout state and history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlag {
    /// Unique feature identifier (e.g., "debate_loop").
    pub id: String,
    /// Human-readable description.
    pub description: String,
    /// Current rollout stage.
    pub stage: RolloutStage,
    /// When the flag was created.
    pub created_at: DateTime<Utc>,
    /// When the stage last changed.
    pub updated_at: DateTime<Utc>,
    /// History of stage transitions.
    pub transitions: Vec<StageTransition>,
}

/// Record of a stage transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTransition {
    /// Previous stage.
    pub from: RolloutStage,
    /// New stage.
    pub to: RolloutStage,
    /// When the transition occurred.
    pub timestamp: DateTime<Utc>,
    /// Reason for the transition.
    pub reason: String,
}

/// Errors from rollout operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RolloutError {
    /// Feature not found.
    NotFound(String),
    /// Feature already registered.
    AlreadyExists(String),
    /// Cannot advance past Production.
    AlreadyFullyRolledOut(String),
    /// Cannot rollback from Disabled.
    AlreadyDisabled(String),
    /// Safety gate check failed.
    GateCheckFailed(String),
}

impl std::fmt::Display for RolloutError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "feature '{}' not found", id),
            Self::AlreadyExists(id) => write!(f, "feature '{}' already registered", id),
            Self::AlreadyFullyRolledOut(id) => {
                write!(f, "feature '{}' is already at production stage", id)
            }
            Self::AlreadyDisabled(id) => write!(f, "feature '{}' is already disabled", id),
            Self::GateCheckFailed(msg) => write!(f, "safety gate failed: {}", msg),
        }
    }
}

impl std::error::Error for RolloutError {}

/// Safety gate that must pass before advancing a rollout stage.
pub trait SafetyGate {
    /// Check whether it's safe to advance from the current stage.
    /// Returns Ok(()) if safe, Err with reason if not.
    fn check(&self, feature: &FeatureFlag) -> Result<(), String>;
}

/// Rollout summary for reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutSummary {
    /// Total registered features.
    pub total: usize,
    /// Count by stage.
    pub by_stage: HashMap<String, usize>,
    /// Features currently in canary or staging (actively rolling out).
    pub in_flight: Vec<String>,
}

/// Manages feature flags and progressive rollout.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutManager {
    /// Registered feature flags.
    features: HashMap<String, FeatureFlag>,
}

impl RolloutManager {
    /// Create a new empty manager.
    pub fn new() -> Self {
        Self {
            features: HashMap::new(),
        }
    }

    /// Register a new feature flag (starts as Disabled).
    pub fn register(&mut self, id: &str, description: &str) -> Result<(), RolloutError> {
        if self.features.contains_key(id) {
            return Err(RolloutError::AlreadyExists(id.to_string()));
        }
        let now = Utc::now();
        self.features.insert(
            id.to_string(),
            FeatureFlag {
                id: id.to_string(),
                description: description.to_string(),
                stage: RolloutStage::Disabled,
                created_at: now,
                updated_at: now,
                transitions: Vec::new(),
            },
        );
        Ok(())
    }

    /// Get a feature flag by ID.
    pub fn get(&self, id: &str) -> Option<&FeatureFlag> {
        self.features.get(id)
    }

    /// Check if a feature is enabled for a given cohort.
    pub fn is_enabled(&self, feature_id: &str, cohort: &Cohort) -> bool {
        self.features
            .get(feature_id)
            .map(|f| f.stage >= cohort.required_stage())
            .unwrap_or(false)
    }

    /// Advance a feature to the next rollout stage.
    pub fn advance(&mut self, feature_id: &str) -> Result<RolloutStage, RolloutError> {
        self.advance_with_reason(feature_id, "manual advance")
    }

    /// Advance with a reason recorded in the transition history.
    pub fn advance_with_reason(
        &mut self,
        feature_id: &str,
        reason: &str,
    ) -> Result<RolloutStage, RolloutError> {
        let feature = self
            .features
            .get_mut(feature_id)
            .ok_or_else(|| RolloutError::NotFound(feature_id.to_string()))?;

        let next = feature
            .stage
            .next()
            .ok_or_else(|| RolloutError::AlreadyFullyRolledOut(feature_id.to_string()))?;

        let now = Utc::now();
        feature.transitions.push(StageTransition {
            from: feature.stage,
            to: next,
            timestamp: now,
            reason: reason.to_string(),
        });
        feature.stage = next;
        feature.updated_at = now;

        Ok(next)
    }

    /// Advance only if a safety gate passes.
    pub fn advance_with_gate(
        &mut self,
        feature_id: &str,
        gate: &dyn SafetyGate,
        reason: &str,
    ) -> Result<RolloutStage, RolloutError> {
        let feature = self
            .features
            .get(feature_id)
            .ok_or_else(|| RolloutError::NotFound(feature_id.to_string()))?;

        gate.check(feature)
            .map_err(RolloutError::GateCheckFailed)?;

        self.advance_with_reason(feature_id, reason)
    }

    /// Rollback a feature to the previous stage.
    pub fn rollback(&mut self, feature_id: &str) -> Result<RolloutStage, RolloutError> {
        self.rollback_with_reason(feature_id, "manual rollback")
    }

    /// Rollback with a reason recorded in the transition history.
    pub fn rollback_with_reason(
        &mut self,
        feature_id: &str,
        reason: &str,
    ) -> Result<RolloutStage, RolloutError> {
        let feature = self
            .features
            .get_mut(feature_id)
            .ok_or_else(|| RolloutError::NotFound(feature_id.to_string()))?;

        let prev = feature
            .stage
            .prev()
            .ok_or_else(|| RolloutError::AlreadyDisabled(feature_id.to_string()))?;

        let now = Utc::now();
        feature.transitions.push(StageTransition {
            from: feature.stage,
            to: prev,
            timestamp: now,
            reason: reason.to_string(),
        });
        feature.stage = prev;
        feature.updated_at = now;

        Ok(prev)
    }

    /// Emergency rollback: set a feature directly to Disabled.
    pub fn emergency_disable(
        &mut self,
        feature_id: &str,
        reason: &str,
    ) -> Result<(), RolloutError> {
        let feature = self
            .features
            .get_mut(feature_id)
            .ok_or_else(|| RolloutError::NotFound(feature_id.to_string()))?;

        if feature.stage == RolloutStage::Disabled {
            return Ok(());
        }

        let now = Utc::now();
        feature.transitions.push(StageTransition {
            from: feature.stage,
            to: RolloutStage::Disabled,
            timestamp: now,
            reason: format!("EMERGENCY: {}", reason),
        });
        feature.stage = RolloutStage::Disabled;
        feature.updated_at = now;

        Ok(())
    }

    /// Get a summary of all feature rollout states.
    pub fn summary(&self) -> RolloutSummary {
        let mut by_stage: HashMap<String, usize> = HashMap::new();
        let mut in_flight = Vec::new();

        for feature in self.features.values() {
            *by_stage.entry(feature.stage.to_string()).or_insert(0) += 1;
            if feature.stage == RolloutStage::Canary || feature.stage == RolloutStage::Staging {
                in_flight.push(feature.id.clone());
            }
        }
        in_flight.sort();

        RolloutSummary {
            total: self.features.len(),
            by_stage,
            in_flight,
        }
    }

    /// List all feature IDs.
    pub fn feature_ids(&self) -> Vec<&str> {
        let mut ids: Vec<&str> = self.features.keys().map(|k| k.as_str()).collect();
        ids.sort();
        ids
    }

    /// Serialize the manager state to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string())
    }

    /// Deserialize manager state from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

impl Default for RolloutManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_get() {
        let mut mgr = RolloutManager::new();
        mgr.register("debate_loop", "Debate-based review").unwrap();
        let flag = mgr.get("debate_loop").unwrap();
        assert_eq!(flag.stage, RolloutStage::Disabled);
        assert_eq!(flag.description, "Debate-based review");
    }

    #[test]
    fn test_register_duplicate_errors() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "first").unwrap();
        let err = mgr.register("f1", "second").unwrap_err();
        assert_eq!(err, RolloutError::AlreadyExists("f1".to_string()));
    }

    #[test]
    fn test_advance_through_stages() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test").unwrap();

        assert_eq!(mgr.advance("f1").unwrap(), RolloutStage::Canary);
        assert_eq!(mgr.advance("f1").unwrap(), RolloutStage::Staging);
        assert_eq!(mgr.advance("f1").unwrap(), RolloutStage::Production);

        let err = mgr.advance("f1").unwrap_err();
        assert_eq!(err, RolloutError::AlreadyFullyRolledOut("f1".to_string()));
    }

    #[test]
    fn test_rollback() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test").unwrap();
        mgr.advance("f1").unwrap(); // Canary
        mgr.advance("f1").unwrap(); // Staging

        assert_eq!(mgr.rollback("f1").unwrap(), RolloutStage::Canary);
        assert_eq!(mgr.rollback("f1").unwrap(), RolloutStage::Disabled);

        let err = mgr.rollback("f1").unwrap_err();
        assert_eq!(err, RolloutError::AlreadyDisabled("f1".to_string()));
    }

    #[test]
    fn test_is_enabled_by_cohort() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test").unwrap();

        // Disabled — nothing enabled
        assert!(!mgr.is_enabled("f1", &Cohort::Canary));
        assert!(!mgr.is_enabled("f1", &Cohort::Staging));
        assert!(!mgr.is_enabled("f1", &Cohort::Production));

        // Canary — only canary sees it
        mgr.advance("f1").unwrap();
        assert!(mgr.is_enabled("f1", &Cohort::Canary));
        assert!(!mgr.is_enabled("f1", &Cohort::Staging));
        assert!(!mgr.is_enabled("f1", &Cohort::Production));

        // Staging — canary + staging
        mgr.advance("f1").unwrap();
        assert!(mgr.is_enabled("f1", &Cohort::Canary));
        assert!(mgr.is_enabled("f1", &Cohort::Staging));
        assert!(!mgr.is_enabled("f1", &Cohort::Production));

        // Production — all
        mgr.advance("f1").unwrap();
        assert!(mgr.is_enabled("f1", &Cohort::Canary));
        assert!(mgr.is_enabled("f1", &Cohort::Staging));
        assert!(mgr.is_enabled("f1", &Cohort::Production));
    }

    #[test]
    fn test_is_enabled_unknown_feature() {
        let mgr = RolloutManager::new();
        assert!(!mgr.is_enabled("nonexistent", &Cohort::Canary));
    }

    #[test]
    fn test_emergency_disable() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test").unwrap();
        mgr.advance("f1").unwrap(); // Canary
        mgr.advance("f1").unwrap(); // Staging

        mgr.emergency_disable("f1", "SLO breach detected").unwrap();

        let flag = mgr.get("f1").unwrap();
        assert_eq!(flag.stage, RolloutStage::Disabled);
        let last = flag.transitions.last().unwrap();
        assert_eq!(last.from, RolloutStage::Staging);
        assert_eq!(last.to, RolloutStage::Disabled);
        assert!(last.reason.contains("EMERGENCY"));
    }

    #[test]
    fn test_emergency_disable_already_disabled() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test").unwrap();
        // Should be a no-op, not an error
        mgr.emergency_disable("f1", "precautionary").unwrap();
        assert_eq!(mgr.get("f1").unwrap().transitions.len(), 0);
    }

    #[test]
    fn test_transition_history() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test").unwrap();
        mgr.advance_with_reason("f1", "initial canary").unwrap();
        mgr.advance_with_reason("f1", "metrics look good").unwrap();
        mgr.rollback_with_reason("f1", "error spike").unwrap();

        let flag = mgr.get("f1").unwrap();
        assert_eq!(flag.transitions.len(), 3);
        assert_eq!(flag.transitions[0].from, RolloutStage::Disabled);
        assert_eq!(flag.transitions[0].to, RolloutStage::Canary);
        assert_eq!(flag.transitions[0].reason, "initial canary");
        assert_eq!(flag.transitions[1].to, RolloutStage::Staging);
        assert_eq!(flag.transitions[2].from, RolloutStage::Staging);
        assert_eq!(flag.transitions[2].to, RolloutStage::Canary);
        assert_eq!(flag.transitions[2].reason, "error spike");
    }

    #[test]
    fn test_safety_gate() {
        struct RequireCanaryFirst;
        impl SafetyGate for RequireCanaryFirst {
            fn check(&self, feature: &FeatureFlag) -> Result<(), String> {
                if feature.stage == RolloutStage::Canary
                    && feature
                        .transitions
                        .iter()
                        .any(|t| t.to == RolloutStage::Canary)
                {
                    Ok(())
                } else {
                    Err("must have been in canary first".to_string())
                }
            }
        }

        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test").unwrap();
        mgr.advance("f1").unwrap(); // Canary

        let gate = RequireCanaryFirst;
        let result = mgr.advance_with_gate("f1", &gate, "gate passed");
        assert_eq!(result.unwrap(), RolloutStage::Staging);
    }

    #[test]
    fn test_safety_gate_blocks_advance() {
        struct AlwaysFails;
        impl SafetyGate for AlwaysFails {
            fn check(&self, _feature: &FeatureFlag) -> Result<(), String> {
                Err("SLO breached: stuck_rate > 15%".to_string())
            }
        }

        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test").unwrap();
        mgr.advance("f1").unwrap(); // Canary

        let gate = AlwaysFails;
        let err = mgr
            .advance_with_gate("f1", &gate, "should fail")
            .unwrap_err();
        assert!(matches!(err, RolloutError::GateCheckFailed(_)));

        // Stage unchanged
        assert_eq!(mgr.get("f1").unwrap().stage, RolloutStage::Canary);
    }

    #[test]
    fn test_summary() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "one").unwrap();
        mgr.register("f2", "two").unwrap();
        mgr.register("f3", "three").unwrap();
        mgr.advance("f1").unwrap(); // Canary
        mgr.advance("f2").unwrap(); // Canary
        mgr.advance("f2").unwrap(); // Staging

        let summary = mgr.summary();
        assert_eq!(summary.total, 3);
        assert_eq!(summary.by_stage.get("disabled"), Some(&1));
        assert_eq!(summary.by_stage.get("canary"), Some(&1));
        assert_eq!(summary.by_stage.get("staging"), Some(&1));
        assert_eq!(summary.in_flight, vec!["f1", "f2"]);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut mgr = RolloutManager::new();
        mgr.register("f1", "test feature").unwrap();
        mgr.advance("f1").unwrap();

        let json = mgr.to_json();
        let restored = RolloutManager::from_json(&json).unwrap();
        let flag = restored.get("f1").unwrap();
        assert_eq!(flag.stage, RolloutStage::Canary);
        assert_eq!(flag.description, "test feature");
        assert_eq!(flag.transitions.len(), 1);
    }

    #[test]
    fn test_not_found_errors() {
        let mut mgr = RolloutManager::new();
        assert!(matches!(
            mgr.advance("nope"),
            Err(RolloutError::NotFound(_))
        ));
        assert!(matches!(
            mgr.rollback("nope"),
            Err(RolloutError::NotFound(_))
        ));
        assert!(matches!(
            mgr.emergency_disable("nope", "x"),
            Err(RolloutError::NotFound(_))
        ));
    }

    #[test]
    fn test_stage_display() {
        assert_eq!(RolloutStage::Disabled.to_string(), "disabled");
        assert_eq!(RolloutStage::Canary.to_string(), "canary");
        assert_eq!(RolloutStage::Staging.to_string(), "staging");
        assert_eq!(RolloutStage::Production.to_string(), "production");
    }

    #[test]
    fn test_cohort_display() {
        assert_eq!(Cohort::Canary.to_string(), "canary");
        assert_eq!(Cohort::Staging.to_string(), "staging");
        assert_eq!(Cohort::Production.to_string(), "production");
    }

    #[test]
    fn test_feature_ids() {
        let mut mgr = RolloutManager::new();
        mgr.register("beta", "b").unwrap();
        mgr.register("alpha", "a").unwrap();
        assert_eq!(mgr.feature_ids(), vec!["alpha", "beta"]);
    }
}
