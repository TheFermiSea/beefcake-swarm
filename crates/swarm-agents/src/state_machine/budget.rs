use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use super::OrchestratorState;

// ──────────────────────────────────────────────────────────────────────────────
// Per-State Timeout and Cancellation Budgets
// ──────────────────────────────────────────────────────────────────────────────

/// Why a state was cancelled (deterministic reason codes).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CancellationReason {
    /// Wall-clock timeout for this state was exceeded.
    Timeout {
        state: OrchestratorState,
        elapsed_ms: u64,
        limit_ms: u64,
    },
    /// Iteration budget for this state was exhausted.
    BudgetExhausted {
        state: OrchestratorState,
        used: u32,
        limit: u32,
    },
    /// Global iteration limit reached across all states.
    GlobalBudgetExhausted { total_iterations: u32, limit: u32 },
    /// External cancellation (e.g., operator signal).
    External { reason: String },
}

impl fmt::Display for CancellationReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Timeout {
                state,
                elapsed_ms,
                limit_ms,
            } => {
                write!(f, "Timeout in {state}: {elapsed_ms}ms > {limit_ms}ms limit")
            }
            Self::BudgetExhausted { state, used, limit } => {
                write!(f, "Budget exhausted in {state}: {used}/{limit} iterations")
            }
            Self::GlobalBudgetExhausted {
                total_iterations,
                limit,
            } => {
                write!(
                    f,
                    "Global budget exhausted: {total_iterations}/{limit} iterations"
                )
            }
            Self::External { reason } => write!(f, "External cancellation: {reason}"),
        }
    }
}

/// Budget configuration for a single state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateBudget {
    /// Maximum wall-clock time in this state (milliseconds).
    /// `None` means no timeout.
    pub timeout_ms: Option<u64>,
    /// Maximum iterations allowed in this state.
    /// `None` means unlimited (bounded by global budget).
    pub max_iterations: Option<u32>,
}

impl StateBudget {
    /// Create a budget with both timeout and iteration limit.
    pub fn new(timeout: Duration, max_iterations: u32) -> Self {
        Self {
            timeout_ms: Some(timeout.as_millis() as u64),
            max_iterations: Some(max_iterations),
        }
    }

    /// Create a timeout-only budget.
    pub fn timeout_only(timeout: Duration) -> Self {
        Self {
            timeout_ms: Some(timeout.as_millis() as u64),
            max_iterations: None,
        }
    }

    /// Create an iteration-only budget.
    pub fn iterations_only(max: u32) -> Self {
        Self {
            timeout_ms: None,
            max_iterations: Some(max),
        }
    }

    /// Unlimited budget (no timeout, no iteration limit).
    pub fn unlimited() -> Self {
        Self {
            timeout_ms: None,
            max_iterations: None,
        }
    }
}

/// Per-state budget configuration for the state machine.
///
/// Default budgets match the existing orchestrator behavior:
/// - Implementing: 45 min timeout, 6 iterations
/// - Verifying: 5 min timeout
/// - Validating: 10 min timeout
/// - Escalating: 2 min timeout
/// - Merging: 5 min timeout
/// - Others: no budget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetConfig {
    /// Per-state budgets. States not in the map have no budget.
    pub budgets: HashMap<OrchestratorState, StateBudget>,
    /// Global iteration limit across all states.
    pub global_max_iterations: u32,
}

impl Default for BudgetConfig {
    fn default() -> Self {
        let mut budgets = HashMap::new();
        budgets.insert(
            OrchestratorState::Implementing,
            StateBudget::new(Duration::from_secs(45 * 60), 6),
        );
        budgets.insert(
            OrchestratorState::Verifying,
            StateBudget::timeout_only(Duration::from_secs(5 * 60)),
        );
        budgets.insert(
            OrchestratorState::Validating,
            StateBudget::timeout_only(Duration::from_secs(10 * 60)),
        );
        budgets.insert(
            OrchestratorState::Escalating,
            StateBudget::timeout_only(Duration::from_secs(2 * 60)),
        );
        budgets.insert(
            OrchestratorState::Merging,
            StateBudget::timeout_only(Duration::from_secs(5 * 60)),
        );
        Self {
            budgets,
            global_max_iterations: 10,
        }
    }
}

/// Tracks per-state time and iteration counts for budget enforcement.
#[derive(Debug)]
pub struct BudgetTracker {
    config: BudgetConfig,
    /// When each state was last entered.
    state_entered_at: Option<Instant>,
    /// Count of times each state has been entered (for iteration budgets).
    state_entry_counts: HashMap<OrchestratorState, u32>,
    /// Total iterations across all states.
    total_iterations: u32,
}

impl BudgetTracker {
    /// Create a new tracker with the given budget configuration.
    pub fn new(config: BudgetConfig) -> Self {
        Self {
            config,
            state_entered_at: None,
            state_entry_counts: HashMap::new(),
            total_iterations: 0,
        }
    }

    /// Create a tracker with default budgets.
    pub fn with_defaults() -> Self {
        Self::new(BudgetConfig::default())
    }

    /// Notify the tracker that a state transition occurred.
    ///
    /// Call this after each successful `StateMachine::advance()`.
    pub fn on_state_entered(&mut self, state: OrchestratorState) {
        self.state_entered_at = Some(Instant::now());
        *self.state_entry_counts.entry(state).or_insert(0) += 1;
        // Only count Planning entries as global iterations — each Planning entry
        // marks the start of a new solve attempt (plan → implement → verify cycle).
        // Per-state entry counts still track all states for per-state budget checks.
        if state == OrchestratorState::Planning {
            self.total_iterations += 1;
        }
    }

    /// Check if the current state has exceeded its budget.
    ///
    /// Returns `Some(CancellationReason)` if the budget is exceeded.
    pub fn check_budget(&self, current_state: OrchestratorState) -> Option<CancellationReason> {
        // Global iteration check
        if self.total_iterations > self.config.global_max_iterations {
            return Some(CancellationReason::GlobalBudgetExhausted {
                total_iterations: self.total_iterations,
                limit: self.config.global_max_iterations,
            });
        }

        // Per-state checks
        if let Some(budget) = self.config.budgets.get(&current_state) {
            // Timeout check
            if let (Some(limit_ms), Some(entered_at)) = (budget.timeout_ms, self.state_entered_at) {
                let elapsed_ms = entered_at.elapsed().as_millis() as u64;
                if elapsed_ms > limit_ms {
                    return Some(CancellationReason::Timeout {
                        state: current_state,
                        elapsed_ms,
                        limit_ms,
                    });
                }
            }

            // Iteration count check
            if let Some(max_iters) = budget.max_iterations {
                let used = self
                    .state_entry_counts
                    .get(&current_state)
                    .copied()
                    .unwrap_or(0);
                if used > max_iters {
                    return Some(CancellationReason::BudgetExhausted {
                        state: current_state,
                        used,
                        limit: max_iters,
                    });
                }
            }
        }

        None
    }

    /// Get the number of times a state has been entered.
    pub fn entry_count(&self, state: OrchestratorState) -> u32 {
        self.state_entry_counts.get(&state).copied().unwrap_or(0)
    }

    /// Get the total iterations across all states.
    pub fn total_iterations(&self) -> u32 {
        self.total_iterations
    }

    /// Get the remaining iteration budget for a state, if configured.
    pub fn remaining_iterations(&self, state: OrchestratorState) -> Option<u32> {
        self.config
            .budgets
            .get(&state)
            .and_then(|b| b.max_iterations)
            .map(|max| {
                let used = self.state_entry_counts.get(&state).copied().unwrap_or(0);
                max.saturating_sub(used)
            })
    }

    /// Get the budget configuration.
    pub fn config(&self) -> &BudgetConfig {
        &self.config
    }
}
