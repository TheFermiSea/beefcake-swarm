//! Escalation State — Tracks iteration history and tier budgets

use crate::feedback::error_parser::ErrorCategory;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model tiers in the swarm hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmTier {
    /// Strand-Rust-Coder 14B — fast code generation
    Implementer,
    /// OR1-Behemoth 72B — multi-file refactors, complex debugging
    Integrator,
    /// Qwen3-Coder-Next 80B MoE — adversarial review
    Adversary,
    /// Cloud models (Opus 4.6, Gemini 3 Pro, GPT-5.x) via PAL MCP
    Cloud,
}

impl SwarmTier {
    /// Get the model identifier for this tier
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Implementer => "strand-rust-coder-14b-q8_0",
            Self::Integrator => "or1-behemoth-q4_k_m",
            Self::Adversary => "Qwen3-Coder-Next-UD-Q4_K_XL.gguf",
            Self::Cloud => "cloud-brain-trust",
        }
    }

    /// Get the default budget for this tier per issue
    pub fn default_budget(&self) -> TierBudget {
        match self {
            Self::Implementer => TierBudget {
                max_iterations: 6,
                max_consultations: 6,
            },
            Self::Integrator => TierBudget {
                max_iterations: 2,
                max_consultations: 2,
            },
            Self::Adversary => TierBudget {
                max_iterations: 1,
                max_consultations: 1,
            },
            Self::Cloud => TierBudget {
                max_iterations: 2, // 1 architecture + 1 review
                max_consultations: 2,
            },
        }
    }
}

impl std::fmt::Display for SwarmTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Implementer => write!(f, "implementer"),
            Self::Integrator => write!(f, "integrator"),
            Self::Adversary => write!(f, "adversary"),
            Self::Cloud => write!(f, "cloud"),
        }
    }
}

/// Budget limits per tier per issue
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TierBudget {
    /// Maximum iterations (compile-fix loops)
    pub max_iterations: u32,
    /// Maximum consultations (requests to this tier)
    pub max_consultations: u32,
}

/// Record of a single iteration attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationRecord {
    /// Which iteration this was (1-indexed)
    pub iteration: u32,
    /// Which tier handled this iteration
    pub tier: SwarmTier,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Error categories present in the verifier report
    pub error_categories: Vec<ErrorCategory>,
    /// Total error count from verifier
    pub error_count: usize,
    /// Whether the verifier reported all-green
    pub all_green: bool,
    /// Whether errors changed from previous iteration
    pub progress_made: bool,
}

/// Record of an escalation event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRecord {
    /// From which tier
    pub from_tier: SwarmTier,
    /// To which tier
    pub to_tier: SwarmTier,
    /// Why escalation was triggered
    pub reason: EscalationReason,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
    /// Iteration number when escalation occurred
    pub at_iteration: u32,
}

/// Reasons for escalation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationReason {
    /// Same error category repeated N times
    RepeatedErrorCategory { category: ErrorCategory, count: u32 },
    /// Total compile failures exceeded threshold
    TotalFailuresExceeded { count: u32, threshold: u32 },
    /// Tier budget exhausted
    BudgetExhausted { tier: SwarmTier },
    /// Multi-file change detected (>8 files)
    MultiFileComplexity { file_count: usize },
    /// Integrator still stuck after consultations
    IntegratorStuck { consultations: u32 },
    /// Explicit escalation by higher tier
    Explicit { reason: String },
}

impl std::fmt::Display for EscalationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RepeatedErrorCategory { category, count } => {
                write!(f, "{} error repeated {}x", category, count)
            }
            Self::TotalFailuresExceeded { count, threshold } => {
                write!(f, "{} failures (threshold: {})", count, threshold)
            }
            Self::BudgetExhausted { tier } => {
                write!(f, "{} budget exhausted", tier)
            }
            Self::MultiFileComplexity { file_count } => {
                write!(f, "{} files touched (>8)", file_count)
            }
            Self::IntegratorStuck { consultations } => {
                write!(f, "integrator stuck after {} consultations", consultations)
            }
            Self::Explicit { reason } => write!(f, "explicit: {}", reason),
        }
    }
}

/// Full escalation state for a single beads issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationState {
    /// Beads issue ID being worked on
    pub bead_id: String,
    /// Current active tier
    pub current_tier: SwarmTier,
    /// Total iteration count across all tiers
    pub total_iterations: u32,
    /// Iterations spent at each tier
    pub tier_iterations: HashMap<SwarmTier, u32>,
    /// Consultations used at each tier
    pub tier_consultations: HashMap<SwarmTier, u32>,
    /// Budget for each tier
    pub tier_budgets: HashMap<SwarmTier, TierBudget>,
    /// History of all iterations
    pub iteration_history: Vec<IterationRecord>,
    /// History of escalation events
    pub escalation_history: Vec<EscalationRecord>,
    /// Error categories seen in the last N iterations (for repeat detection)
    pub recent_error_categories: Vec<Vec<ErrorCategory>>,
    /// Whether the issue has been resolved (all-green)
    pub resolved: bool,
    /// Whether the issue is stuck (all budgets exhausted, needs human)
    pub stuck: bool,
    /// Timestamp of last activity
    pub last_activity: DateTime<Utc>,
}

impl EscalationState {
    /// Create a new escalation state for a beads issue
    pub fn new(bead_id: impl Into<String>) -> Self {
        let mut tier_budgets = HashMap::new();
        tier_budgets.insert(
            SwarmTier::Implementer,
            SwarmTier::Implementer.default_budget(),
        );
        tier_budgets.insert(
            SwarmTier::Integrator,
            SwarmTier::Integrator.default_budget(),
        );
        tier_budgets.insert(SwarmTier::Adversary, SwarmTier::Adversary.default_budget());
        tier_budgets.insert(SwarmTier::Cloud, SwarmTier::Cloud.default_budget());

        Self {
            bead_id: bead_id.into(),
            current_tier: SwarmTier::Implementer,
            total_iterations: 0,
            tier_iterations: HashMap::new(),
            tier_consultations: HashMap::new(),
            tier_budgets,
            iteration_history: Vec::new(),
            escalation_history: Vec::new(),
            recent_error_categories: Vec::new(),
            resolved: false,
            stuck: false,
            last_activity: Utc::now(),
        }
    }

    /// Record an iteration result
    pub fn record_iteration(
        &mut self,
        error_categories: Vec<ErrorCategory>,
        error_count: usize,
        all_green: bool,
    ) {
        self.total_iterations += 1;
        *self.tier_iterations.entry(self.current_tier).or_insert(0) += 1;

        let progress_made = self.check_progress(&error_categories, error_count);

        let record = IterationRecord {
            iteration: self.total_iterations,
            tier: self.current_tier,
            timestamp: Utc::now(),
            error_categories: error_categories.clone(),
            error_count,
            all_green,
            progress_made,
        };

        self.iteration_history.push(record);
        self.recent_error_categories.push(error_categories);

        // Keep sliding window of last 6 iterations
        if self.recent_error_categories.len() > 6 {
            self.recent_error_categories.remove(0);
        }

        if all_green {
            self.resolved = true;
        }

        self.last_activity = Utc::now();
    }

    /// Record a consultation with a tier
    pub fn record_consultation(&mut self, tier: SwarmTier) {
        *self.tier_consultations.entry(tier).or_insert(0) += 1;
        self.last_activity = Utc::now();
    }

    /// Record an escalation event
    pub fn record_escalation(&mut self, to_tier: SwarmTier, reason: EscalationReason) {
        let record = EscalationRecord {
            from_tier: self.current_tier,
            to_tier,
            reason,
            timestamp: Utc::now(),
            at_iteration: self.total_iterations,
        };
        self.escalation_history.push(record);
        self.current_tier = to_tier;
        self.last_activity = Utc::now();
    }

    /// Check if progress was made (error categories changed or count decreased)
    fn check_progress(&self, new_categories: &[ErrorCategory], new_count: usize) -> bool {
        if let Some(prev) = self.iteration_history.last() {
            // Progress if error count decreased
            if new_count < prev.error_count {
                return true;
            }
            // Progress if error categories changed
            if new_categories != prev.error_categories {
                return true;
            }
            false
        } else {
            // First iteration — always "progress"
            true
        }
    }

    /// Get remaining budget for a tier
    pub fn remaining_budget(&self, tier: SwarmTier) -> u32 {
        let budget = self
            .tier_budgets
            .get(&tier)
            .map(|b| b.max_iterations)
            .unwrap_or(0);
        let used = self.tier_iterations.get(&tier).copied().unwrap_or(0);
        budget.saturating_sub(used)
    }

    /// Get remaining consultations for a tier
    pub fn remaining_consultations(&self, tier: SwarmTier) -> u32 {
        let budget = self
            .tier_budgets
            .get(&tier)
            .map(|b| b.max_consultations)
            .unwrap_or(0);
        let used = self.tier_consultations.get(&tier).copied().unwrap_or(0);
        budget.saturating_sub(used)
    }

    /// Check if a specific error category has repeated N times in recent iterations
    pub fn error_category_repeat_count(&self, category: ErrorCategory) -> u32 {
        self.recent_error_categories
            .iter()
            .rev()
            .take_while(|cats| cats.contains(&category))
            .count() as u32
    }

    /// Get the most recently repeated error category (if any repeats >= 2)
    pub fn most_repeated_category(&self) -> Option<(ErrorCategory, u32)> {
        if self.recent_error_categories.len() < 2 {
            return None;
        }

        let last = self.recent_error_categories.last()?;
        let mut best: Option<(ErrorCategory, u32)> = None;

        for cat in last {
            let count = self.error_category_repeat_count(*cat);
            if count >= 2 {
                match &best {
                    Some((_, best_count)) if count > *best_count => {
                        best = Some((*cat, count));
                    }
                    None => {
                        best = Some((*cat, count));
                    }
                    _ => {}
                }
            }
        }

        best
    }

    /// Total compile failures (non-green iterations)
    pub fn total_failures(&self) -> u32 {
        self.iteration_history
            .iter()
            .filter(|r| !r.all_green)
            .count() as u32
    }

    /// Get a summary for logging
    pub fn summary(&self) -> String {
        format!(
            "bead={} tier={} iter={} failures={} resolved={} stuck={}",
            self.bead_id,
            self.current_tier,
            self.total_iterations,
            self.total_failures(),
            self.resolved,
            self.stuck,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_tier_budgets() {
        assert_eq!(SwarmTier::Implementer.default_budget().max_iterations, 6);
        assert_eq!(SwarmTier::Integrator.default_budget().max_iterations, 2);
        assert_eq!(SwarmTier::Cloud.default_budget().max_iterations, 2);
    }

    #[test]
    fn test_escalation_state_new() {
        let state = EscalationState::new("beads-123");
        assert_eq!(state.current_tier, SwarmTier::Implementer);
        assert_eq!(state.total_iterations, 0);
        assert!(!state.resolved);
        assert!(!state.stuck);
    }

    #[test]
    fn test_record_iteration() {
        let mut state = EscalationState::new("beads-123");

        state.record_iteration(vec![ErrorCategory::Lifetime], 3, false);
        assert_eq!(state.total_iterations, 1);
        assert_eq!(state.remaining_budget(SwarmTier::Implementer), 5);

        state.record_iteration(vec![ErrorCategory::Lifetime], 3, false);
        assert_eq!(state.total_iterations, 2);
        assert_eq!(
            state.error_category_repeat_count(ErrorCategory::Lifetime),
            2
        );
    }

    #[test]
    fn test_error_category_repeat_detection() {
        let mut state = EscalationState::new("beads-123");

        // Different categories — no repeat
        state.record_iteration(vec![ErrorCategory::TypeMismatch], 2, false);
        state.record_iteration(vec![ErrorCategory::Lifetime], 1, false);
        assert_eq!(state.most_repeated_category(), None);

        // Now lifetime repeats
        state.record_iteration(vec![ErrorCategory::Lifetime], 1, false);
        let repeated = state.most_repeated_category();
        assert!(repeated.is_some());
        assert_eq!(repeated.unwrap().0, ErrorCategory::Lifetime);
        assert_eq!(repeated.unwrap().1, 2);
    }

    #[test]
    fn test_resolved_on_all_green() {
        let mut state = EscalationState::new("beads-123");
        state.record_iteration(vec![], 0, true);
        assert!(state.resolved);
    }

    #[test]
    fn test_escalation_record() {
        let mut state = EscalationState::new("beads-123");
        state.record_escalation(
            SwarmTier::Integrator,
            EscalationReason::RepeatedErrorCategory {
                category: ErrorCategory::Lifetime,
                count: 2,
            },
        );
        assert_eq!(state.current_tier, SwarmTier::Integrator);
        assert_eq!(state.escalation_history.len(), 1);
    }
}
