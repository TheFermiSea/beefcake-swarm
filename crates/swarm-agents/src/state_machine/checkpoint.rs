use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::{OrchestratorState, StateMachine, TransitionRecord};

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
