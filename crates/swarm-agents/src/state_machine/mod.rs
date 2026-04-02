//! Orchestrator State Machine — explicit states and legal transition guards.
//!
//! Provides a typed state model for the orchestration loop so that:
//! 1. Every state transition is auditable and logged.
//! 2. Illegal transitions are caught at compile time (via `advance()` guards).
//! 3. Offline replay can reconstruct the exact sequence of states.
//!
//! The orchestrator loop calls `advance()` to move between states. Each call
//! validates the transition is legal and records it in the transition log.

use std::fmt;
use std::time::Instant;

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

pub mod audit;
pub mod budget;
pub mod checkpoint;

pub use audit::*;
pub use budget::*;
pub use checkpoint::*;

#[cfg(test)]
mod tests;
