//! Orchestrator State Machine — explicit states and legal transition guards.
//!
//! Provides a typed state model for the orchestration loop so that:
//! 1. Every state transition is auditable and logged.
//! 2. Illegal transitions are caught at compile time (via `advance()` guards).
//! 3. Offline replay can reconstruct the exact sequence of states.
//!
//! The orchestrator loop calls `advance()` to move between states. Each call
//! validates the transition is legal and records it in the transition log.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// The set of orchestrator states.
///
/// States follow the invariant: every run starts at `SelectingIssue` and
/// terminates at either `Resolved` or `Failed`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrchestratorState {
    /// Picking the next issue from beads.
    SelectingIssue,
    /// Creating or resuming a git worktree for the issue.
    PreparingWorktree,
    /// Building initial context / work packet before entering the loop.
    Planning,
    /// Calling the implementer agent (coder) to produce changes.
    Implementing,
    /// Running deterministic quality gates (fmt, clippy, check, test).
    Verifying,
    /// Cloud-based blind validation of the changes.
    Validating,
    /// Deciding whether to retry, escalate tier, or give up.
    Escalating,
    /// Merging the worktree back to main.
    Merging,
    /// Issue successfully resolved — terminal state.
    Resolved,
    /// Stuck or budget exhausted — terminal state.
    Failed,
}

impl OrchestratorState {
    /// Whether this is a terminal state (no further transitions allowed).
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Resolved | Self::Failed)
    }
}

impl fmt::Display for OrchestratorState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SelectingIssue => write!(f, "SelectingIssue"),
            Self::PreparingWorktree => write!(f, "PreparingWorktree"),
            Self::Planning => write!(f, "Planning"),
            Self::Implementing => write!(f, "Implementing"),
            Self::Verifying => write!(f, "Verifying"),
            Self::Validating => write!(f, "Validating"),
            Self::Escalating => write!(f, "Escalating"),
            Self::Merging => write!(f, "Merging"),
            Self::Resolved => write!(f, "Resolved"),
            Self::Failed => write!(f, "Failed"),
        }
    }
}

/// Legal transitions between orchestrator states.
///
/// The transition table encodes the valid edges in the state graph:
/// ```text
/// SelectingIssue → PreparingWorktree | Failed
/// PreparingWorktree → Planning | Failed
/// Planning → Implementing | Failed
/// Implementing → Verifying | Failed
/// Verifying → Validating | Implementing | Escalating | Merging | Failed
/// Validating → Merging | Implementing | Escalating | Failed
/// Escalating → Implementing | Failed
/// Merging → Resolved | Failed
/// ```
fn is_legal_transition(from: OrchestratorState, to: OrchestratorState) -> bool {
    use OrchestratorState::*;

    // Any non-terminal state can transition to Failed.
    if to == Failed && !from.is_terminal() {
        return true;
    }

    matches!(
        (from, to),
        (SelectingIssue, PreparingWorktree)
            | (PreparingWorktree, Planning)
            | (Planning, Implementing)
            | (Implementing, Verifying)
            // After verifying: green → validate or merge; errors → retry or escalate
            | (Verifying, Validating)
            | (Verifying, Implementing)
            | (Verifying, Escalating)
            | (Verifying, Merging)
            // After validating: pass → merge; fail → retry or escalate
            | (Validating, Merging)
            | (Validating, Implementing)
            | (Validating, Escalating)
            // After escalating: re-enter implementation at new tier
            | (Escalating, Implementing)
            // Merge → resolved
            | (Merging, Resolved)
    )
}

/// A single recorded state transition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionRecord {
    /// The state transitioned from.
    pub from: OrchestratorState,
    /// The state transitioned to.
    pub to: OrchestratorState,
    /// Iteration number at the time of transition (0 for pre-loop states).
    pub iteration: u32,
    /// Milliseconds since the state machine was created.
    pub elapsed_ms: u64,
    /// Optional context about why this transition happened.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// Error returned when an illegal transition is attempted.
#[derive(Debug, Clone)]
pub struct IllegalTransition {
    pub from: OrchestratorState,
    pub to: OrchestratorState,
}

impl fmt::Display for IllegalTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Illegal state transition: {} → {}", self.from, self.to)
    }
}

impl std::error::Error for IllegalTransition {}

/// The orchestrator state machine.
///
/// Tracks the current state, enforces legal transitions, and maintains
/// a complete log of all transitions for replay and diagnostics.
#[derive(Debug)]
pub struct StateMachine {
    current: OrchestratorState,
    iteration: u32,
    created_at: Instant,
    transitions: Vec<TransitionRecord>,
}

impl StateMachine {
    /// Create a new state machine starting at `SelectingIssue`.
    pub fn new() -> Self {
        Self {
            current: OrchestratorState::SelectingIssue,
            iteration: 0,
            created_at: Instant::now(),
            transitions: Vec::new(),
        }
    }

    /// Get the current state.
    pub fn current(&self) -> OrchestratorState {
        self.current
    }

    /// Get the current iteration number.
    pub fn iteration(&self) -> u32 {
        self.iteration
    }

    /// Set the iteration counter (called by the orchestrator loop).
    pub fn set_iteration(&mut self, iteration: u32) {
        self.iteration = iteration;
    }

    /// Attempt to advance to the next state.
    ///
    /// Returns `Ok(())` if the transition is legal, or `Err(IllegalTransition)`
    /// if the transition would violate the state graph.
    pub fn advance(
        &mut self,
        to: OrchestratorState,
        reason: Option<&str>,
    ) -> Result<(), IllegalTransition> {
        if !is_legal_transition(self.current, to) {
            return Err(IllegalTransition {
                from: self.current,
                to,
            });
        }

        let record = TransitionRecord {
            from: self.current,
            to,
            iteration: self.iteration,
            elapsed_ms: self.created_at.elapsed().as_millis() as u64,
            reason: reason.map(String::from),
        };

        tracing::debug!(
            from = %self.current,
            to = %to,
            iteration = self.iteration,
            "State transition"
        );

        self.transitions.push(record);
        self.current = to;
        Ok(())
    }

    /// Transition to `Failed` state from any non-terminal state.
    ///
    /// Convenience method — always legal from non-terminal states.
    pub fn fail(&mut self, reason: &str) -> Result<(), IllegalTransition> {
        self.advance(OrchestratorState::Failed, Some(reason))
    }

    /// Whether the state machine is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.current.is_terminal()
    }

    /// Get the full transition log.
    pub fn transitions(&self) -> &[TransitionRecord] {
        &self.transitions
    }

    /// Get a summary string of the state machine's history.
    pub fn summary(&self) -> String {
        let states: Vec<String> = self.transitions.iter().map(|t| t.to.to_string()).collect();
        format!(
            "{} → {} ({}ms, {} transitions)",
            OrchestratorState::SelectingIssue,
            self.current,
            self.created_at.elapsed().as_millis(),
            self.transitions.len(),
        ) + if states.is_empty() {
            String::new()
        } else {
            format!(" [{}]", states.join(" → "))
        }
        .as_str()
    }
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Transition Audit Log — structured export for post-run reasoning
// ──────────────────────────────────────────────────────────────────────────────

/// A structured audit report of a state machine run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Final state of the run.
    pub final_state: OrchestratorState,
    /// Total number of transitions.
    pub transition_count: usize,
    /// Final iteration number.
    pub iteration: u32,
    /// Total wall-clock duration in milliseconds.
    pub total_elapsed_ms: u64,
    /// Number of times each state was entered.
    pub state_visit_counts: HashMap<OrchestratorState, u32>,
    /// Number of retry loops (Verifying/Validating → Implementing).
    pub retry_count: u32,
    /// Number of escalations (→ Escalating).
    pub escalation_count: u32,
    /// The full ordered transition log.
    pub transitions: Vec<TransitionRecord>,
    /// Any invariant violations detected.
    pub invariant_violations: Vec<String>,
}

impl StateMachine {
    /// Generate a structured audit report from the state machine's history.
    pub fn audit_report(&self) -> AuditReport {
        let mut state_visits: HashMap<OrchestratorState, u32> = HashMap::new();
        let mut retry_count = 0u32;
        let mut escalation_count = 0u32;

        for t in &self.transitions {
            *state_visits.entry(t.to).or_insert(0) += 1;

            // Count retries: going back to Implementing from Verifying/Validating
            if t.to == OrchestratorState::Implementing
                && (t.from == OrchestratorState::Verifying
                    || t.from == OrchestratorState::Validating)
            {
                retry_count += 1;
            }

            // Count escalations
            if t.to == OrchestratorState::Escalating {
                escalation_count += 1;
            }
        }

        let violations = check_invariants(&self.transitions, self.current);

        AuditReport {
            final_state: self.current,
            transition_count: self.transitions.len(),
            iteration: self.iteration,
            total_elapsed_ms: self.created_at.elapsed().as_millis() as u64,
            state_visit_counts: state_visits,
            retry_count,
            escalation_count,
            transitions: self.transitions.clone(),
            invariant_violations: violations,
        }
    }

    /// Export the transition log as a JSON string.
    pub fn export_transitions_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.transitions)
    }
}

impl fmt::Display for AuditReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== State Machine Audit Report ===")?;
        writeln!(f, "Final state: {}", self.final_state)?;
        writeln!(f, "Transitions: {}", self.transition_count)?;
        writeln!(f, "Iterations: {}", self.iteration)?;
        writeln!(f, "Retries: {}", self.retry_count)?;
        writeln!(f, "Escalations: {}", self.escalation_count)?;

        if !self.invariant_violations.is_empty() {
            writeln!(f, "VIOLATIONS ({}):", self.invariant_violations.len())?;
            for v in &self.invariant_violations {
                writeln!(f, "  - {v}")?;
            }
        } else {
            writeln!(f, "Invariants: all passed")?;
        }

        writeln!(f, "--- Transition Log ---")?;
        for (i, t) in self.transitions.iter().enumerate() {
            let reason = t.reason.as_deref().unwrap_or("");
            writeln!(
                f,
                "  [{i}] {} → {} (iter={}, +{}ms) {reason}",
                t.from, t.to, t.iteration, t.elapsed_ms
            )?;
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// State Invariants — assertions that should hold for any valid run
// ──────────────────────────────────────────────────────────────────────────────

/// Check structural invariants on a completed transition log.
///
/// Returns a list of violation descriptions (empty = all invariants hold).
pub fn check_invariants(
    transitions: &[TransitionRecord],
    final_state: OrchestratorState,
) -> Vec<String> {
    let mut violations = Vec::new();

    // INV-1: Every transition must be legal according to the transition table.
    for (i, t) in transitions.iter().enumerate() {
        if !is_legal_transition(t.from, t.to) {
            violations.push(format!(
                "INV-1 (legal transitions): transition [{i}] {} → {} is illegal",
                t.from, t.to
            ));
        }
    }

    // INV-2: Terminal states are absorbing — no transitions after Resolved/Failed.
    let mut saw_terminal = false;
    for (i, t) in transitions.iter().enumerate() {
        if saw_terminal {
            violations.push(format!(
                "INV-2 (terminal absorbing): transition [{i}] {} → {} occurs after terminal state",
                t.from, t.to
            ));
        }
        if t.to.is_terminal() {
            saw_terminal = true;
        }
    }

    // INV-3: If run ended in terminal, the last transition should land on it.
    if final_state.is_terminal() && !transitions.is_empty() {
        if let Some(last) = transitions.last() {
            if last.to != final_state {
                violations.push(format!(
                    "INV-3 (terminal consistency): final_state={final_state} but last transition lands on {}",
                    last.to
                ));
            }
        }
    }

    // INV-4: Consecutive transitions must chain (each from == previous to).
    for window in transitions.windows(2) {
        if window[1].from != window[0].to {
            violations.push(format!(
                "INV-4 (chain continuity): expected from={} but got from={} at transition to {}",
                window[0].to, window[1].from, window[1].to
            ));
        }
    }

    // INV-5: First transition (if any) must start from SelectingIssue.
    if let Some(first) = transitions.first() {
        if first.from != OrchestratorState::SelectingIssue {
            violations.push(format!(
                "INV-5 (initial state): first transition starts from {} instead of SelectingIssue",
                first.from
            ));
        }
    }

    // INV-6: Iteration counter must be non-decreasing.
    for window in transitions.windows(2) {
        if window[1].iteration < window[0].iteration {
            violations.push(format!(
                "INV-6 (iteration monotonic): iteration decreased from {} to {} at {} → {}",
                window[0].iteration, window[1].iteration, window[1].from, window[1].to
            ));
        }
    }

    // INV-7: elapsed_ms must be non-decreasing.
    for window in transitions.windows(2) {
        if window[1].elapsed_ms < window[0].elapsed_ms {
            violations.push(format!(
                "INV-7 (time monotonic): elapsed_ms decreased from {} to {} at {} → {}",
                window[0].elapsed_ms, window[1].elapsed_ms, window[1].from, window[1].to
            ));
        }
    }

    violations
}

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
        self.total_iterations += 1;
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

// ──────────────────────────────────────────────────────────────────────────────
// Checkpoint / Resume — typed state snapshots for crash-safe recovery
// ──────────────────────────────────────────────────────────────────────────────

/// Current checkpoint schema version. Bump on breaking changes.
pub const CHECKPOINT_SCHEMA_VERSION: u8 = 1;

/// A typed snapshot of the state machine at a stable transition point.
///
/// Written to disk after every stable transition (states where it's safe
/// to resume: after Verifying, after Implementing, after Escalating).
/// On restart, the orchestrator loads the checkpoint and rebuilds the
/// state machine from it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateCheckpoint {
    /// Schema version for forward-compatibility detection.
    pub schema_version: u8,
    /// Unique ID for this checkpoint (monotonically increasing).
    pub checkpoint_id: u64,
    /// The state at checkpoint time.
    pub state: OrchestratorState,
    /// Current iteration number.
    pub iteration: u32,
    /// Complete transition history up to this point.
    pub transitions: Vec<TransitionRecord>,
    /// ISO 8601 timestamp when the checkpoint was created.
    pub created_at: String,
    /// Git commit hash at checkpoint time (for worktree state verification).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_hash: Option<String>,
    /// Issue ID being processed.
    pub issue_id: String,
}

/// Result of attempting to resume from a checkpoint.
#[derive(Debug)]
pub enum ResumeResult {
    /// Successfully restored state machine from checkpoint.
    Restored(StateMachine),
    /// Checkpoint is from an incompatible schema version.
    IncompatibleSchema {
        checkpoint_version: u8,
        current_version: u8,
    },
    /// Checkpoint is stale (git hash doesn't match worktree).
    StaleCheckpoint {
        expected_hash: String,
        actual_hash: String,
    },
}

/// States that are safe to checkpoint at (stable transition points).
fn is_checkpointable(state: OrchestratorState) -> bool {
    matches!(
        state,
        OrchestratorState::Implementing
            | OrchestratorState::Verifying
            | OrchestratorState::Escalating
            | OrchestratorState::Validating
    )
}

impl StateMachine {
    /// Create a checkpoint of the current state.
    ///
    /// Returns `None` if the current state is not a stable checkpoint point
    /// (terminal states and pre-loop states are not checkpointable).
    pub fn checkpoint(&self, issue_id: &str, git_hash: Option<&str>) -> Option<StateCheckpoint> {
        if !is_checkpointable(self.current) {
            return None;
        }

        Some(StateCheckpoint {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id: self.transitions.len() as u64,
            state: self.current,
            iteration: self.iteration,
            transitions: self.transitions.clone(),
            created_at: chrono::Utc::now().to_rfc3339(),
            git_hash: git_hash.map(String::from),
            issue_id: issue_id.to_string(),
        })
    }

    /// Resume a state machine from a checkpoint.
    ///
    /// Validates schema version compatibility. If `expected_git_hash` is
    /// provided, verifies it matches the checkpoint's git hash (detects
    /// stale checkpoints from a different worktree state).
    pub fn resume_from(
        checkpoint: &StateCheckpoint,
        expected_git_hash: Option<&str>,
    ) -> ResumeResult {
        // Schema compatibility check
        if checkpoint.schema_version != CHECKPOINT_SCHEMA_VERSION {
            return ResumeResult::IncompatibleSchema {
                checkpoint_version: checkpoint.schema_version,
                current_version: CHECKPOINT_SCHEMA_VERSION,
            };
        }

        // Staleness check: if both hashes are available, they must match
        if let (Some(expected), Some(checkpoint_hash)) =
            (expected_git_hash, checkpoint.git_hash.as_deref())
        {
            if expected != checkpoint_hash {
                return ResumeResult::StaleCheckpoint {
                    expected_hash: expected.to_string(),
                    actual_hash: checkpoint_hash.to_string(),
                };
            }
        }

        let sm = StateMachine {
            current: checkpoint.state,
            iteration: checkpoint.iteration,
            created_at: Instant::now(), // Reset wall-clock (can't restore Instant)
            transitions: checkpoint.transitions.clone(),
        };

        tracing::info!(
            state = %sm.current,
            iteration = sm.iteration,
            transitions = sm.transitions.len(),
            "Resumed state machine from checkpoint"
        );

        ResumeResult::Restored(sm)
    }
}

/// Write a state checkpoint to disk.
pub fn save_checkpoint(checkpoint: &StateCheckpoint, path: &std::path::Path) {
    match serde_json::to_string_pretty(checkpoint) {
        Ok(json) => match std::fs::write(path, json) {
            Ok(()) => tracing::info!(
                path = %path.display(),
                state = %checkpoint.state,
                iteration = checkpoint.iteration,
                "Saved state checkpoint"
            ),
            Err(e) => tracing::warn!("Failed to write checkpoint: {e}"),
        },
        Err(e) => tracing::warn!("Failed to serialize checkpoint: {e}"),
    }
}

/// Load a state checkpoint from disk.
pub fn load_checkpoint(path: &std::path::Path) -> Option<StateCheckpoint> {
    match std::fs::read_to_string(path) {
        Ok(contents) => match serde_json::from_str::<StateCheckpoint>(&contents) {
            Ok(cp) => {
                tracing::info!(
                    path = %path.display(),
                    state = %cp.state,
                    iteration = cp.iteration,
                    "Loaded state checkpoint"
                );
                Some(cp)
            }
            Err(e) => {
                tracing::warn!("Failed to parse checkpoint: {e}");
                None
            }
        },
        Err(e) => {
            tracing::debug!("No checkpoint file at {}: {e}", path.display());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let sm = StateMachine::new();
        assert_eq!(sm.current(), OrchestratorState::SelectingIssue);
        assert!(!sm.is_terminal());
        assert_eq!(sm.transitions().len(), 0);
    }

    #[test]
    fn test_happy_path_transitions() {
        let mut sm = StateMachine::new();

        // Full success path
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Validating, Some("all gates green"))
            .unwrap();
        sm.advance(OrchestratorState::Merging, Some("validator passed"))
            .unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        assert!(sm.is_terminal());
        assert_eq!(sm.current(), OrchestratorState::Resolved);
        assert_eq!(sm.transitions().len(), 7);
    }

    #[test]
    fn test_retry_loop() {
        let mut sm = StateMachine::new();

        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();

        // Verifier found errors → retry
        sm.advance(
            OrchestratorState::Implementing,
            Some("errors found, retrying"),
        )
        .unwrap();
        sm.set_iteration(2);
        sm.advance(OrchestratorState::Verifying, None).unwrap();

        // Now green → validate → merge
        sm.advance(OrchestratorState::Validating, None).unwrap();
        sm.advance(OrchestratorState::Merging, None).unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        assert!(sm.is_terminal());
        assert_eq!(sm.transitions().len(), 9);
    }

    #[test]
    fn test_escalation_path() {
        let mut sm = StateMachine::new();

        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();

        // Errors persist → escalate
        sm.advance(
            OrchestratorState::Escalating,
            Some("repeated borrow errors"),
        )
        .unwrap();
        sm.advance(OrchestratorState::Implementing, Some("escalated to Cloud"))
            .unwrap();
        sm.set_iteration(2);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(
            OrchestratorState::Merging,
            Some("all green after escalation"),
        )
        .unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        assert!(sm.is_terminal());
    }

    #[test]
    fn test_failure_from_any_state() {
        for state in [
            OrchestratorState::SelectingIssue,
            OrchestratorState::PreparingWorktree,
            OrchestratorState::Planning,
            OrchestratorState::Implementing,
            OrchestratorState::Verifying,
            OrchestratorState::Validating,
            OrchestratorState::Escalating,
            OrchestratorState::Merging,
        ] {
            let mut sm = StateMachine {
                current: state,
                iteration: 0,
                created_at: Instant::now(),
                transitions: Vec::new(),
            };
            assert!(sm.fail("test failure").is_ok());
            assert_eq!(sm.current(), OrchestratorState::Failed);
            assert!(sm.is_terminal());
        }
    }

    #[test]
    fn test_cannot_transition_from_terminal() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Merging, None).unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        // Cannot transition from Resolved
        let err = sm
            .advance(OrchestratorState::Implementing, None)
            .unwrap_err();
        assert_eq!(err.from, OrchestratorState::Resolved);
        assert_eq!(err.to, OrchestratorState::Implementing);

        // Cannot fail from terminal either
        assert!(sm.fail("nope").is_err());
    }

    #[test]
    fn test_illegal_skip_transition() {
        let mut sm = StateMachine::new();

        // Can't skip directly to Implementing without PreparingWorktree
        let err = sm
            .advance(OrchestratorState::Implementing, None)
            .unwrap_err();
        assert_eq!(err.from, OrchestratorState::SelectingIssue);
        assert_eq!(err.to, OrchestratorState::Implementing);
    }

    #[test]
    fn test_illegal_backward_transition() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();

        // Can't go backward to SelectingIssue
        assert!(sm.advance(OrchestratorState::SelectingIssue, None).is_err());
    }

    #[test]
    fn test_transition_record_has_reason() {
        let mut sm = StateMachine::new();
        sm.advance(
            OrchestratorState::PreparingWorktree,
            Some("issue-123 selected"),
        )
        .unwrap();

        let record = &sm.transitions()[0];
        assert_eq!(record.from, OrchestratorState::SelectingIssue);
        assert_eq!(record.to, OrchestratorState::PreparingWorktree);
        assert_eq!(record.reason.as_deref(), Some("issue-123 selected"));
    }

    #[test]
    fn test_transition_record_serde_roundtrip() {
        let record = TransitionRecord {
            from: OrchestratorState::Verifying,
            to: OrchestratorState::Escalating,
            iteration: 3,
            elapsed_ms: 12345,
            reason: Some("repeated borrow errors".into()),
        };

        let json = serde_json::to_string(&record).unwrap();
        let restored: TransitionRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.from, OrchestratorState::Verifying);
        assert_eq!(restored.to, OrchestratorState::Escalating);
        assert_eq!(restored.iteration, 3);
        assert_eq!(restored.elapsed_ms, 12345);
    }

    #[test]
    fn test_state_display() {
        assert_eq!(
            OrchestratorState::SelectingIssue.to_string(),
            "SelectingIssue"
        );
        assert_eq!(OrchestratorState::Failed.to_string(), "Failed");
    }

    #[test]
    fn test_summary() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.fail("test").unwrap();
        let summary = sm.summary();
        assert!(summary.contains("Failed"));
        assert!(summary.contains("2 transitions"));
    }

    #[test]
    fn test_verifying_can_skip_to_merging() {
        // When verifier is green and no cloud validation needed
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(
            OrchestratorState::Merging,
            Some("all green, no cloud validation needed"),
        )
        .unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();
        assert!(sm.is_terminal());
    }

    #[test]
    fn test_validator_can_trigger_escalation() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Validating, None).unwrap();
        // Validator says needs_escalation
        sm.advance(
            OrchestratorState::Escalating,
            Some("validator: needs_escalation"),
        )
        .unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        assert_eq!(sm.current(), OrchestratorState::Implementing);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Checkpoint / Resume Tests
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_checkpoint_at_verifying() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();

        let cp = sm.checkpoint("issue-123", Some("abc1234")).unwrap();
        assert_eq!(cp.schema_version, CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(cp.state, OrchestratorState::Verifying);
        assert_eq!(cp.iteration, 1);
        assert_eq!(cp.issue_id, "issue-123");
        assert_eq!(cp.git_hash.as_deref(), Some("abc1234"));
        assert_eq!(cp.transitions.len(), 4);
    }

    #[test]
    fn test_checkpoint_not_allowed_at_terminal() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.fail("test").unwrap();

        // Terminal states are not checkpointable
        assert!(sm.checkpoint("issue", None).is_none());
    }

    #[test]
    fn test_checkpoint_not_allowed_at_pre_loop() {
        let sm = StateMachine::new();
        // SelectingIssue is not checkpointable
        assert!(sm.checkpoint("issue", None).is_none());
    }

    #[test]
    fn test_resume_from_checkpoint() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(2);
        sm.advance(OrchestratorState::Verifying, None).unwrap();

        let cp = sm.checkpoint("issue-456", Some("def5678")).unwrap();

        // Resume from checkpoint
        match StateMachine::resume_from(&cp, Some("def5678")) {
            ResumeResult::Restored(restored) => {
                assert_eq!(restored.current(), OrchestratorState::Verifying);
                assert_eq!(restored.iteration(), 2);
                assert_eq!(restored.transitions().len(), 4);
                // Can continue from restored state
                // (verify we can actually transition)
            }
            other => panic!("Expected Restored, got {other:?}"),
        }
    }

    #[test]
    fn test_resume_continues_transitions() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();

        let cp = sm.checkpoint("issue", None).unwrap();

        match StateMachine::resume_from(&cp, None) {
            ResumeResult::Restored(mut restored) => {
                // Can advance from restored state
                restored
                    .advance(OrchestratorState::Implementing, Some("resumed — retrying"))
                    .unwrap();
                assert_eq!(restored.current(), OrchestratorState::Implementing);
                // Transition log includes both original and new transitions
                assert_eq!(restored.transitions().len(), 5);
            }
            other => panic!("Expected Restored, got {other:?}"),
        }
    }

    #[test]
    fn test_resume_incompatible_schema() {
        let cp = StateCheckpoint {
            schema_version: 99, // Future version
            checkpoint_id: 0,
            state: OrchestratorState::Verifying,
            iteration: 1,
            transitions: vec![],
            created_at: "2026-01-01T00:00:00Z".into(),
            git_hash: None,
            issue_id: "issue".into(),
        };

        match StateMachine::resume_from(&cp, None) {
            ResumeResult::IncompatibleSchema {
                checkpoint_version,
                current_version,
            } => {
                assert_eq!(checkpoint_version, 99);
                assert_eq!(current_version, CHECKPOINT_SCHEMA_VERSION);
            }
            other => panic!("Expected IncompatibleSchema, got {other:?}"),
        }
    }

    #[test]
    fn test_resume_stale_checkpoint() {
        let cp = StateCheckpoint {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id: 0,
            state: OrchestratorState::Verifying,
            iteration: 1,
            transitions: vec![],
            created_at: "2026-01-01T00:00:00Z".into(),
            git_hash: Some("old_hash".into()),
            issue_id: "issue".into(),
        };

        match StateMachine::resume_from(&cp, Some("new_hash")) {
            ResumeResult::StaleCheckpoint {
                expected_hash,
                actual_hash,
            } => {
                assert_eq!(expected_hash, "new_hash");
                assert_eq!(actual_hash, "old_hash");
            }
            other => panic!("Expected StaleCheckpoint, got {other:?}"),
        }
    }

    #[test]
    fn test_checkpoint_serde_roundtrip() {
        let cp = StateCheckpoint {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id: 5,
            state: OrchestratorState::Implementing,
            iteration: 3,
            transitions: vec![TransitionRecord {
                from: OrchestratorState::Verifying,
                to: OrchestratorState::Implementing,
                iteration: 2,
                elapsed_ms: 5000,
                reason: Some("retry after errors".into()),
            }],
            created_at: "2026-02-21T00:00:00Z".into(),
            git_hash: Some("abc123".into()),
            issue_id: "beefcake-xyz".into(),
        };

        let json = serde_json::to_string_pretty(&cp).unwrap();
        let restored: StateCheckpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.schema_version, CHECKPOINT_SCHEMA_VERSION);
        assert_eq!(restored.state, OrchestratorState::Implementing);
        assert_eq!(restored.iteration, 3);
        assert_eq!(restored.transitions.len(), 1);
        assert_eq!(restored.issue_id, "beefcake-xyz");
    }

    #[test]
    fn test_save_and_load_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(".swarm-state-checkpoint.json");

        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();

        let cp = sm.checkpoint("test-issue", Some("deadbeef")).unwrap();
        save_checkpoint(&cp, &path);
        assert!(path.exists());

        let loaded = load_checkpoint(&path).unwrap();
        assert_eq!(loaded.state, OrchestratorState::Verifying);
        assert_eq!(loaded.iteration, 1);
        assert_eq!(loaded.issue_id, "test-issue");
        assert_eq!(loaded.git_hash.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn test_load_nonexistent_checkpoint() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("no-such-file.json");
        assert!(load_checkpoint(&path).is_none());
    }

    #[test]
    fn test_resume_no_git_hash_skips_staleness() {
        // When checkpoint has no git hash, staleness check is skipped
        let cp = StateCheckpoint {
            schema_version: CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id: 0,
            state: OrchestratorState::Implementing,
            iteration: 1,
            transitions: vec![],
            created_at: "2026-01-01T00:00:00Z".into(),
            git_hash: None,
            issue_id: "issue".into(),
        };

        // Even with a provided expected hash, no staleness error
        match StateMachine::resume_from(&cp, Some("any_hash")) {
            ResumeResult::Restored(sm) => {
                assert_eq!(sm.current(), OrchestratorState::Implementing);
            }
            other => panic!("Expected Restored, got {other:?}"),
        }
    }

    // ──────────────────────────────────────────────────────────────────────
    // Budget / Timeout Tests
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_budget_config_defaults() {
        let config = BudgetConfig::default();
        assert_eq!(config.global_max_iterations, 10);

        // Implementing has both timeout and iteration limit
        let imp = config
            .budgets
            .get(&OrchestratorState::Implementing)
            .unwrap();
        assert_eq!(imp.timeout_ms, Some(45 * 60 * 1000));
        assert_eq!(imp.max_iterations, Some(6));

        // Verifying has timeout only
        let ver = config.budgets.get(&OrchestratorState::Verifying).unwrap();
        assert!(ver.timeout_ms.is_some());
        assert!(ver.max_iterations.is_none());

        // SelectingIssue has no budget
        assert!(config
            .budgets
            .get(&OrchestratorState::SelectingIssue)
            .is_none());
    }

    #[test]
    fn test_budget_tracker_no_violation() {
        let mut tracker = BudgetTracker::with_defaults();
        tracker.on_state_entered(OrchestratorState::Implementing);

        // Fresh entry — no violations
        assert!(tracker
            .check_budget(OrchestratorState::Implementing)
            .is_none());
        assert_eq!(tracker.entry_count(OrchestratorState::Implementing), 1);
        assert_eq!(tracker.total_iterations(), 1);
    }

    #[test]
    fn test_budget_tracker_iteration_exhaustion() {
        let config = BudgetConfig {
            budgets: {
                let mut m = HashMap::new();
                m.insert(
                    OrchestratorState::Implementing,
                    StateBudget::iterations_only(2),
                );
                m
            },
            global_max_iterations: 100,
        };
        let mut tracker = BudgetTracker::new(config);

        // Enter state 3 times (limit is 2)
        tracker.on_state_entered(OrchestratorState::Implementing);
        assert!(tracker
            .check_budget(OrchestratorState::Implementing)
            .is_none());

        tracker.on_state_entered(OrchestratorState::Implementing);
        assert!(tracker
            .check_budget(OrchestratorState::Implementing)
            .is_none());

        tracker.on_state_entered(OrchestratorState::Implementing);
        match tracker.check_budget(OrchestratorState::Implementing) {
            Some(CancellationReason::BudgetExhausted { state, used, limit }) => {
                assert_eq!(state, OrchestratorState::Implementing);
                assert_eq!(used, 3);
                assert_eq!(limit, 2);
            }
            other => panic!("Expected BudgetExhausted, got {other:?}"),
        }
    }

    #[test]
    fn test_budget_tracker_global_exhaustion() {
        let config = BudgetConfig {
            budgets: HashMap::new(),
            global_max_iterations: 3,
        };
        let mut tracker = BudgetTracker::new(config);

        tracker.on_state_entered(OrchestratorState::Implementing);
        tracker.on_state_entered(OrchestratorState::Verifying);
        tracker.on_state_entered(OrchestratorState::Implementing);
        assert!(tracker
            .check_budget(OrchestratorState::Implementing)
            .is_none());

        // 4th entry exceeds global limit of 3
        tracker.on_state_entered(OrchestratorState::Verifying);
        match tracker.check_budget(OrchestratorState::Verifying) {
            Some(CancellationReason::GlobalBudgetExhausted {
                total_iterations,
                limit,
            }) => {
                assert_eq!(total_iterations, 4);
                assert_eq!(limit, 3);
            }
            other => panic!("Expected GlobalBudgetExhausted, got {other:?}"),
        }
    }

    #[test]
    fn test_budget_tracker_remaining_iterations() {
        let config = BudgetConfig {
            budgets: {
                let mut m = HashMap::new();
                m.insert(
                    OrchestratorState::Implementing,
                    StateBudget::iterations_only(5),
                );
                m
            },
            global_max_iterations: 100,
        };
        let mut tracker = BudgetTracker::new(config);

        assert_eq!(
            tracker.remaining_iterations(OrchestratorState::Implementing),
            Some(5)
        );

        tracker.on_state_entered(OrchestratorState::Implementing);
        tracker.on_state_entered(OrchestratorState::Implementing);
        assert_eq!(
            tracker.remaining_iterations(OrchestratorState::Implementing),
            Some(3)
        );

        // State without configured budget returns None
        assert!(tracker
            .remaining_iterations(OrchestratorState::Verifying)
            .is_none());
    }

    #[test]
    fn test_budget_tracker_unconfigured_state() {
        let tracker = BudgetTracker::with_defaults();
        // SelectingIssue has no budget — always OK (global check still runs)
        assert!(tracker
            .check_budget(OrchestratorState::SelectingIssue)
            .is_none());
    }

    #[test]
    fn test_state_budget_constructors() {
        let full = StateBudget::new(Duration::from_secs(300), 5);
        assert_eq!(full.timeout_ms, Some(300_000));
        assert_eq!(full.max_iterations, Some(5));

        let timeout = StateBudget::timeout_only(Duration::from_secs(60));
        assert_eq!(timeout.timeout_ms, Some(60_000));
        assert!(timeout.max_iterations.is_none());

        let iters = StateBudget::iterations_only(10);
        assert!(iters.timeout_ms.is_none());
        assert_eq!(iters.max_iterations, Some(10));

        let unlimited = StateBudget::unlimited();
        assert!(unlimited.timeout_ms.is_none());
        assert!(unlimited.max_iterations.is_none());
    }

    #[test]
    fn test_cancellation_reason_display() {
        let timeout = CancellationReason::Timeout {
            state: OrchestratorState::Implementing,
            elapsed_ms: 5000,
            limit_ms: 3000,
        };
        assert!(timeout.to_string().contains("Timeout"));
        assert!(timeout.to_string().contains("5000ms"));

        let budget = CancellationReason::BudgetExhausted {
            state: OrchestratorState::Implementing,
            used: 7,
            limit: 6,
        };
        assert!(budget.to_string().contains("7/6"));

        let global = CancellationReason::GlobalBudgetExhausted {
            total_iterations: 11,
            limit: 10,
        };
        assert!(global.to_string().contains("11/10"));

        let external = CancellationReason::External {
            reason: "operator signal".into(),
        };
        assert!(external.to_string().contains("operator signal"));
    }

    #[test]
    fn test_cancellation_reason_serde_roundtrip() {
        let reasons = vec![
            CancellationReason::Timeout {
                state: OrchestratorState::Verifying,
                elapsed_ms: 12345,
                limit_ms: 10000,
            },
            CancellationReason::BudgetExhausted {
                state: OrchestratorState::Implementing,
                used: 7,
                limit: 6,
            },
            CancellationReason::GlobalBudgetExhausted {
                total_iterations: 11,
                limit: 10,
            },
            CancellationReason::External {
                reason: "test".into(),
            },
        ];

        for reason in &reasons {
            let json = serde_json::to_string(reason).unwrap();
            let restored: CancellationReason = serde_json::from_str(&json).unwrap();
            assert_eq!(&restored, reason);
        }
    }

    #[test]
    fn test_budget_config_serde_roundtrip() {
        let config = BudgetConfig::default();
        let json = serde_json::to_string_pretty(&config).unwrap();
        let restored: BudgetConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.global_max_iterations, config.global_max_iterations);
        assert_eq!(restored.budgets.len(), config.budgets.len());
    }

    #[test]
    fn test_budget_tracker_config_accessor() {
        let tracker = BudgetTracker::with_defaults();
        let config = tracker.config();
        assert_eq!(config.global_max_iterations, 10);
        assert!(config
            .budgets
            .contains_key(&OrchestratorState::Implementing));
    }

    // ──────────────────────────────────────────────────────────────────────
    // Audit Report Tests
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_audit_report_happy_path() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Validating, None).unwrap();
        sm.advance(OrchestratorState::Merging, None).unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        let report = sm.audit_report();
        assert_eq!(report.final_state, OrchestratorState::Resolved);
        assert_eq!(report.transition_count, 7);
        assert_eq!(report.retry_count, 0);
        assert_eq!(report.escalation_count, 0);
        assert!(report.invariant_violations.is_empty());
    }

    #[test]
    fn test_audit_report_with_retries() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        // Retry from verifying
        sm.advance(OrchestratorState::Implementing, Some("retry"))
            .unwrap();
        sm.set_iteration(2);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Validating, None).unwrap();
        // Retry from validating
        sm.advance(OrchestratorState::Implementing, Some("retry"))
            .unwrap();
        sm.set_iteration(3);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Merging, None).unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        let report = sm.audit_report();
        assert_eq!(report.retry_count, 2);
        assert_eq!(report.escalation_count, 0);
        assert!(report.invariant_violations.is_empty());
    }

    #[test]
    fn test_audit_report_with_escalation() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Escalating, Some("stuck"))
            .unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(2);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Merging, None).unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        let report = sm.audit_report();
        assert_eq!(report.escalation_count, 1);
        assert!(report.invariant_violations.is_empty());
    }

    #[test]
    fn test_audit_report_counts_state_visits() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(2);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Merging, None).unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        let report = sm.audit_report();
        assert_eq!(
            report
                .state_visit_counts
                .get(&OrchestratorState::Implementing),
            Some(&2)
        );
        assert_eq!(
            report.state_visit_counts.get(&OrchestratorState::Verifying),
            Some(&2)
        );
        assert_eq!(
            report.state_visit_counts.get(&OrchestratorState::Merging),
            Some(&1)
        );
    }

    #[test]
    fn test_audit_report_display() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.fail("test").unwrap();

        let report = sm.audit_report();
        let display = report.to_string();
        assert!(display.contains("Audit Report"));
        assert!(display.contains("Failed"));
        assert!(display.contains("Invariants: all passed"));
    }

    #[test]
    fn test_audit_report_serde_roundtrip() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.fail("test failure").unwrap();

        let report = sm.audit_report();
        let json = serde_json::to_string_pretty(&report).unwrap();
        let restored: AuditReport = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.final_state, OrchestratorState::Failed);
        assert_eq!(restored.transition_count, 3);
    }

    #[test]
    fn test_export_transitions_json() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, Some("test"))
            .unwrap();

        let json = sm.export_transitions_json().unwrap();
        let parsed: Vec<TransitionRecord> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].to, OrchestratorState::PreparingWorktree);
    }

    // ──────────────────────────────────────────────────────────────────────
    // Invariant Tests
    // ──────────────────────────────────────────────────────────────────────

    #[test]
    fn test_invariants_happy_path_clean() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Merging, None).unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        let violations = check_invariants(sm.transitions(), sm.current());
        assert!(violations.is_empty(), "Violations: {violations:?}");
    }

    #[test]
    fn test_invariant_detects_illegal_transition() {
        // Manually construct an illegal log
        let transitions = vec![TransitionRecord {
            from: OrchestratorState::SelectingIssue,
            to: OrchestratorState::Merging, // Illegal skip
            iteration: 0,
            elapsed_ms: 0,
            reason: None,
        }];

        let violations = check_invariants(&transitions, OrchestratorState::Merging);
        assert!(violations.iter().any(|v| v.contains("INV-1")));
    }

    #[test]
    fn test_invariant_detects_post_terminal_transition() {
        let transitions = vec![
            TransitionRecord {
                from: OrchestratorState::SelectingIssue,
                to: OrchestratorState::Failed,
                iteration: 0,
                elapsed_ms: 0,
                reason: None,
            },
            TransitionRecord {
                from: OrchestratorState::Failed,
                to: OrchestratorState::Implementing, // After terminal
                iteration: 1,
                elapsed_ms: 100,
                reason: None,
            },
        ];

        let violations = check_invariants(&transitions, OrchestratorState::Implementing);
        assert!(violations.iter().any(|v| v.contains("INV-2")));
    }

    #[test]
    fn test_invariant_detects_chain_discontinuity() {
        let transitions = vec![
            TransitionRecord {
                from: OrchestratorState::SelectingIssue,
                to: OrchestratorState::PreparingWorktree,
                iteration: 0,
                elapsed_ms: 0,
                reason: None,
            },
            TransitionRecord {
                from: OrchestratorState::Implementing, // Discontinuity
                to: OrchestratorState::Verifying,
                iteration: 1,
                elapsed_ms: 100,
                reason: None,
            },
        ];

        let violations = check_invariants(&transitions, OrchestratorState::Verifying);
        assert!(violations.iter().any(|v| v.contains("INV-4")));
    }

    #[test]
    fn test_invariant_detects_wrong_initial_state() {
        let transitions = vec![TransitionRecord {
            from: OrchestratorState::Implementing, // Wrong start
            to: OrchestratorState::Verifying,
            iteration: 1,
            elapsed_ms: 0,
            reason: None,
        }];

        let violations = check_invariants(&transitions, OrchestratorState::Verifying);
        assert!(violations.iter().any(|v| v.contains("INV-5")));
    }

    #[test]
    fn test_invariant_detects_iteration_decrease() {
        let transitions = vec![
            TransitionRecord {
                from: OrchestratorState::SelectingIssue,
                to: OrchestratorState::PreparingWorktree,
                iteration: 5,
                elapsed_ms: 0,
                reason: None,
            },
            TransitionRecord {
                from: OrchestratorState::PreparingWorktree,
                to: OrchestratorState::Planning,
                iteration: 3, // Decreased
                elapsed_ms: 100,
                reason: None,
            },
        ];

        let violations = check_invariants(&transitions, OrchestratorState::Planning);
        assert!(violations.iter().any(|v| v.contains("INV-6")));
    }

    #[test]
    fn test_invariant_detects_time_decrease() {
        let transitions = vec![
            TransitionRecord {
                from: OrchestratorState::SelectingIssue,
                to: OrchestratorState::PreparingWorktree,
                iteration: 0,
                elapsed_ms: 500,
                reason: None,
            },
            TransitionRecord {
                from: OrchestratorState::PreparingWorktree,
                to: OrchestratorState::Planning,
                iteration: 0,
                elapsed_ms: 200, // Decreased
                reason: None,
            },
        ];

        let violations = check_invariants(&transitions, OrchestratorState::Planning);
        assert!(violations.iter().any(|v| v.contains("INV-7")));
    }

    #[test]
    fn test_invariants_empty_log() {
        let violations = check_invariants(&[], OrchestratorState::SelectingIssue);
        assert!(violations.is_empty());
    }

    // ──────────────────────────────────────────────────────────────────────
    // Property-Style Tests — exhaustive/systematic scenario coverage
    // ──────────────────────────────────────────────────────────────────────

    /// All non-terminal states can transition to Failed.
    #[test]
    fn test_property_any_non_terminal_can_fail() {
        let non_terminal = [
            OrchestratorState::SelectingIssue,
            OrchestratorState::PreparingWorktree,
            OrchestratorState::Planning,
            OrchestratorState::Implementing,
            OrchestratorState::Verifying,
            OrchestratorState::Validating,
            OrchestratorState::Escalating,
            OrchestratorState::Merging,
        ];

        for state in non_terminal {
            assert!(
                is_legal_transition(state, OrchestratorState::Failed),
                "{state} → Failed should be legal"
            );
        }
    }

    /// Terminal states cannot transition to anything.
    #[test]
    fn test_property_terminal_states_absorbing() {
        let terminals = [OrchestratorState::Resolved, OrchestratorState::Failed];
        let all_states = [
            OrchestratorState::SelectingIssue,
            OrchestratorState::PreparingWorktree,
            OrchestratorState::Planning,
            OrchestratorState::Implementing,
            OrchestratorState::Verifying,
            OrchestratorState::Validating,
            OrchestratorState::Escalating,
            OrchestratorState::Merging,
            OrchestratorState::Resolved,
            OrchestratorState::Failed,
        ];

        for terminal in terminals {
            for target in all_states {
                assert!(
                    !is_legal_transition(terminal, target),
                    "{terminal} → {target} should be illegal (terminal is absorbing)"
                );
            }
        }
    }

    /// Every retry loop through the state machine is bounded by budget.
    #[test]
    fn test_property_retry_loop_bounded_by_budget() {
        let config = BudgetConfig {
            budgets: {
                let mut m = HashMap::new();
                m.insert(
                    OrchestratorState::Implementing,
                    StateBudget::iterations_only(3),
                );
                m
            },
            global_max_iterations: 100,
        };
        let mut tracker = BudgetTracker::new(config);
        let mut sm = StateMachine::new();

        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();

        let mut retries = 0u32;
        for iter in 1..=10 {
            sm.advance(OrchestratorState::Implementing, None).unwrap();
            tracker.on_state_entered(OrchestratorState::Implementing);
            sm.set_iteration(iter);

            if let Some(_reason) = tracker.check_budget(OrchestratorState::Implementing) {
                sm.fail("budget exhausted").unwrap();
                break;
            }

            sm.advance(OrchestratorState::Verifying, None).unwrap();
            tracker.on_state_entered(OrchestratorState::Verifying);

            // Simulate failure: go back to implementing
            if !sm.is_terminal() {
                retries += 1;
            }
        }

        // Budget was 3, so we should have been stopped
        assert!(sm.is_terminal() || retries <= 3);
        let report = sm.audit_report();
        assert!(report.invariant_violations.is_empty());
    }

    /// Escalation always returns to Implementing (deterministic trigger).
    #[test]
    fn test_property_escalation_deterministic_reentry() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(1);
        sm.advance(OrchestratorState::Verifying, None).unwrap();

        // Escalate
        sm.advance(OrchestratorState::Escalating, Some("error repeat"))
            .unwrap();

        // The only legal transition from Escalating (besides Failed) is Implementing
        assert!(is_legal_transition(
            OrchestratorState::Escalating,
            OrchestratorState::Implementing
        ));

        // No other non-fail transitions from Escalating
        for state in [
            OrchestratorState::SelectingIssue,
            OrchestratorState::PreparingWorktree,
            OrchestratorState::Planning,
            OrchestratorState::Verifying,
            OrchestratorState::Validating,
            OrchestratorState::Escalating,
            OrchestratorState::Merging,
            OrchestratorState::Resolved,
        ] {
            assert!(
                !is_legal_transition(OrchestratorState::Escalating, state),
                "Escalating → {state} should be illegal"
            );
        }
    }

    /// Multiple escalations in a single run maintain invariants.
    #[test]
    fn test_property_multiple_escalations_maintain_invariants() {
        let mut sm = StateMachine::new();
        sm.advance(OrchestratorState::PreparingWorktree, None)
            .unwrap();
        sm.advance(OrchestratorState::Planning, None).unwrap();

        for iter in 1..=3 {
            sm.advance(OrchestratorState::Implementing, None).unwrap();
            sm.set_iteration(iter);
            sm.advance(OrchestratorState::Verifying, None).unwrap();
            sm.advance(OrchestratorState::Escalating, Some("stuck"))
                .unwrap();
        }

        // Final attempt succeeds
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(4);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Merging, None).unwrap();
        sm.advance(OrchestratorState::Resolved, None).unwrap();

        let report = sm.audit_report();
        assert_eq!(report.escalation_count, 3);
        assert_eq!(report.retry_count, 0); // Escalations are not retries
        assert!(report.invariant_violations.is_empty());
    }

    /// Global budget caps total iterations across all states.
    #[test]
    fn test_property_global_budget_caps_all_states() {
        let config = BudgetConfig {
            budgets: HashMap::new(),
            global_max_iterations: 5,
        };
        let mut tracker = BudgetTracker::new(config);

        let states = [
            OrchestratorState::Implementing,
            OrchestratorState::Verifying,
            OrchestratorState::Implementing,
            OrchestratorState::Verifying,
            OrchestratorState::Implementing,
        ];

        for &s in &states {
            tracker.on_state_entered(s);
        }
        assert!(tracker
            .check_budget(OrchestratorState::Implementing)
            .is_none());

        // One more pushes over
        tracker.on_state_entered(OrchestratorState::Verifying);
        assert!(matches!(
            tracker.check_budget(OrchestratorState::Verifying),
            Some(CancellationReason::GlobalBudgetExhausted { .. })
        ));
    }
}
