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
}
