//! Escalation State — Tracks iteration history and tier budgets

use crate::feedback::error_parser::ErrorCategory;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Model tiers in the swarm hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmTier {
    /// HydraCoder — local worker for code generation and fixes
    Worker,
    /// Manager Council (Opus 4.5, Gemini 3 Pro, Qwen 3.5) — escalated coordination
    Council,
    /// Human intervention — blocking beads issue
    Human,
}

impl SwarmTier {
    /// Get the model identifier for this tier
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Worker => "HydraCoder-Q6_K",
            Self::Council => "manager-council",
            Self::Human => "human",
        }
    }

    /// Get the default budget for this tier per issue
    pub fn default_budget(&self) -> TierBudget {
        match self {
            Self::Worker => TierBudget {
                max_iterations: 4,
                max_consultations: 4,
            },
            Self::Council => TierBudget {
                max_iterations: 6,
                max_consultations: 6,
            },
            Self::Human => TierBudget {
                max_iterations: 0,
                max_consultations: 0,
            },
        }
    }
}

impl std::fmt::Display for SwarmTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Worker => write!(f, "worker"),
            Self::Council => write!(f, "council"),
            Self::Human => write!(f, "human"),
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

/// Per-tier turn and timeout policy.
///
/// Centralises the agent turn limits and wall-clock timeouts that were
/// previously scattered across `coder.rs`, `manager.rs`, and `orchestrator.rs`.
/// The orchestrator queries `TurnPolicy::for_tier()` instead of hard-coding
/// constants, making calibration a single-file change.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TurnPolicy {
    /// Maximum tool-call turns the agent may take per invocation.
    pub max_turns: usize,
    /// Wall-clock timeout (seconds) for a single agent invocation.
    pub timeout_secs: u64,
}

impl TurnPolicy {
    /// Calibrated defaults per tier.
    ///
    /// | Tier    | max_turns | timeout  | Rationale                              |
    /// |---------|-----------|----------|----------------------------------------|
    /// | Worker  | 15        | 30 min   | Fast coder, bounded tool loops         |
    /// | Council | 20        | 45 min   | Manager delegates to workers via tools |
    /// | Human   | 0         | 0        | No automated agent                     |
    pub fn for_tier(tier: SwarmTier) -> Self {
        match tier {
            SwarmTier::Worker => Self {
                max_turns: 15,
                timeout_secs: 30 * 60,
            },
            SwarmTier::Council => Self {
                max_turns: 20,
                timeout_secs: 45 * 60,
            },
            SwarmTier::Human => Self {
                max_turns: 0,
                timeout_secs: 0,
            },
        }
    }
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
    /// Council still stuck after consultations
    CouncilStuck { consultations: u32 },
    /// Explicit escalation by higher tier
    Explicit { reason: String },
    /// Consecutive no-change iterations exceeded threshold
    ConsecutiveNoChange { count: u32, threshold: u32 },
    /// Friction detector triggered (oscillation, plateau, churn)
    FrictionDetected { description: String },
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
            Self::CouncilStuck { consultations } => {
                write!(f, "council stuck after {} consultations", consultations)
            }
            Self::Explicit { reason } => write!(f, "explicit: {}", reason),
            Self::ConsecutiveNoChange { count, threshold } => {
                write!(
                    f,
                    "{} consecutive no-change iterations (threshold: {})",
                    count, threshold
                )
            }
            Self::FrictionDetected { description } => {
                write!(f, "friction: {}", description)
            }
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
    /// Consecutive iterations where no file changes were produced.
    /// Persists across SLURM preemptions for accurate no-change detection.
    pub consecutive_no_change: u32,
    /// Timestamp of last activity
    pub last_activity: DateTime<Utc>,
}

impl EscalationState {
    /// Create a new escalation state for a beads issue
    pub fn new(bead_id: impl Into<String>) -> Self {
        let mut tier_budgets = HashMap::new();
        tier_budgets.insert(SwarmTier::Worker, SwarmTier::Worker.default_budget());
        tier_budgets.insert(SwarmTier::Council, SwarmTier::Council.default_budget());
        tier_budgets.insert(SwarmTier::Human, SwarmTier::Human.default_budget());

        Self {
            bead_id: bead_id.into(),
            current_tier: SwarmTier::Worker,
            total_iterations: 0,
            tier_iterations: HashMap::new(),
            tier_consultations: HashMap::new(),
            tier_budgets,
            iteration_history: Vec::new(),
            escalation_history: Vec::new(),
            recent_error_categories: Vec::new(),
            resolved: false,
            stuck: false,
            consecutive_no_change: 0,
            last_activity: Utc::now(),
        }
    }

    /// Set the initial tier (builder pattern).
    ///
    /// Allows the orchestrator to override the default starting tier.
    pub fn with_initial_tier(mut self, tier: SwarmTier) -> Self {
        self.current_tier = tier;
        self
    }

    /// Override a tier's budget (builder pattern).
    ///
    /// Allows the orchestrator to customize iteration/consultation limits.
    pub fn with_budget(mut self, tier: SwarmTier, budget: TierBudget) -> Self {
        self.tier_budgets.insert(tier, budget);
        self
    }

    /// Record a no-change iteration (agent produced no file edits).
    pub fn record_no_change(&mut self) {
        self.consecutive_no_change += 1;
    }

    /// Reset the no-change counter (agent produced file edits).
    pub fn reset_no_change(&mut self) {
        self.consecutive_no_change = 0;
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
        assert_eq!(SwarmTier::Worker.default_budget().max_iterations, 4);
        assert_eq!(SwarmTier::Council.default_budget().max_iterations, 6);
        assert_eq!(SwarmTier::Human.default_budget().max_iterations, 0);
    }

    #[test]
    fn test_escalation_state_new() {
        let state = EscalationState::new("beads-123");
        assert_eq!(state.current_tier, SwarmTier::Worker);
        assert_eq!(state.total_iterations, 0);
        assert!(!state.resolved);
        assert!(!state.stuck);
    }

    #[test]
    fn test_record_iteration() {
        let mut state = EscalationState::new("beads-123");

        state.record_iteration(vec![ErrorCategory::Lifetime], 3, false);
        assert_eq!(state.total_iterations, 1);
        assert_eq!(state.remaining_budget(SwarmTier::Worker), 3);

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
            SwarmTier::Council,
            EscalationReason::RepeatedErrorCategory {
                category: ErrorCategory::Lifetime,
                count: 2,
            },
        );
        assert_eq!(state.current_tier, SwarmTier::Council);
        assert_eq!(state.escalation_history.len(), 1);
    }

    #[test]
    fn test_turn_policy_defaults() {
        let worker = TurnPolicy::for_tier(SwarmTier::Worker);
        assert_eq!(worker.max_turns, 15);
        assert_eq!(worker.timeout_secs, 30 * 60);

        let council = TurnPolicy::for_tier(SwarmTier::Council);
        assert_eq!(council.max_turns, 20);
        assert_eq!(council.timeout_secs, 45 * 60);

        let human = TurnPolicy::for_tier(SwarmTier::Human);
        assert_eq!(human.max_turns, 0);
        assert_eq!(human.timeout_secs, 0);
    }
}
