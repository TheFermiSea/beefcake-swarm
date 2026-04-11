//! Orchestrator event types for the session log.
//!
//! Every significant action in the orchestrator loop emits an event.
//! Events are append-only, immutable, and form the source of truth
//! for crash recovery via `wake()`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::config::Tier;
use crate::state_machine::OrchestratorState;

/// Monotonically increasing event identifier within a session.
pub type EventId = u64;

/// A single event in the orchestrator session log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Monotonically increasing ID (1-based).
    pub id: EventId,
    /// Wall-clock timestamp.
    pub timestamp: DateTime<Utc>,
    /// The event payload.
    pub kind: EventKind,
}

/// All possible orchestrator events.
///
/// Each variant captures the minimum information needed to reconstruct
/// state on replay. Large payloads (full LLM responses, file contents)
/// are stored as truncated summaries — the session log is for recovery
/// and debugging, not full transcript archival.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    /// Session started for an issue.
    SessionStarted {
        issue_id: String,
        objective: String,
        /// Git commit on main when session started.
        base_commit: Option<String>,
    },

    /// State machine transitioned between states.
    StateTransition {
        from: OrchestratorState,
        to: OrchestratorState,
        iteration: u32,
        reason: Option<String>,
    },

    /// A new iteration of the implement→verify loop started.
    IterationStarted { number: u32, tier: Tier },

    /// Git worktree was provisioned for this session.
    WorktreeProvisioned {
        path: String,
        branch: String,
        commit: String,
    },

    /// An LLM turn completed (manager or worker).
    LlmTurnCompleted {
        agent: String,
        model: String,
        turn: u32,
        tokens_in: Option<u64>,
        tokens_out: Option<u64>,
        duration_ms: u64,
    },

    /// A tool was called by the LLM.
    ToolCallCompleted {
        agent: String,
        tool_name: String,
        /// Whether the tool call succeeded or errored.
        success: bool,
        duration_ms: u64,
        /// First 200 chars of the result (for debugging, not replay).
        result_preview: String,
    },

    /// Worker was delegated a subtask.
    WorkerDelegated {
        role: String,
        model: String,
        /// First 200 chars of the prompt.
        prompt_preview: String,
    },

    /// Worker returned a result.
    WorkerCompleted {
        role: String,
        model: String,
        success: bool,
        /// Number of files the worker modified.
        files_changed: u32,
        duration_ms: u64,
    },

    /// Deterministic verifier ran quality gates.
    VerifierResult {
        passed: bool,
        /// e.g. ["fmt", "clippy", "check", "test"]
        gates_passed: Vec<String>,
        gates_failed: Vec<String>,
        error_count: u32,
        warning_count: u32,
    },

    /// Tier escalation triggered.
    EscalationTriggered {
        from_tier: Tier,
        to_tier: Tier,
        reason: String,
    },

    /// An iteration of the implement→verify loop completed.
    IterationCompleted {
        number: u32,
        /// Did the verifier pass this iteration?
        verified: bool,
        files_changed: Vec<String>,
        duration_ms: u64,
    },

    /// Context was pruned or rebuilt for the next iteration.
    ContextRebuilt {
        strategy: String,
        tokens_used: u64,
        tokens_budget: u64,
    },

    /// No-change circuit breaker incremented.
    NoChangeDetected {
        consecutive_count: u32,
        max_allowed: u32,
    },

    /// Session completed (terminal).
    SessionCompleted {
        resolved: bool,
        total_iterations: u32,
        duration_ms: u64,
        /// If resolved, the merge commit hash.
        merge_commit: Option<String>,
        /// If failed, the reason.
        failure_reason: Option<String>,
    },

    /// Checkpoint was written to disk.
    CheckpointWritten {
        checkpoint_id: u64,
        state: OrchestratorState,
        iteration: u32,
    },

    /// Free-form annotation (for debugging, not replay).
    Note { message: String },
}

impl SessionEvent {
    /// Create a new event with the given kind and auto-assigned timestamp.
    /// The `id` must be set by the `SessionLog` on append.
    pub fn new(id: EventId, kind: EventKind) -> Self {
        Self {
            id,
            timestamp: Utc::now(),
            kind,
        }
    }
}

impl std::fmt::Display for EventKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EventKind::SessionStarted { issue_id, .. } => {
                write!(f, "session_started({})", issue_id)
            }
            EventKind::StateTransition { from, to, .. } => {
                write!(f, "transition({:?} → {:?})", from, to)
            }
            EventKind::IterationStarted { number, tier } => {
                write!(f, "iteration_started(#{}, {:?})", number, tier)
            }
            EventKind::WorktreeProvisioned { branch, .. } => {
                write!(f, "worktree_provisioned({})", branch)
            }
            EventKind::LlmTurnCompleted { agent, turn, .. } => {
                write!(f, "llm_turn({}, #{})", agent, turn)
            }
            EventKind::ToolCallCompleted {
                tool_name, success, ..
            } => {
                write!(f, "tool_call({}, ok={})", tool_name, success)
            }
            EventKind::WorkerDelegated { role, model, .. } => {
                write!(f, "worker_delegated({}, {})", role, model)
            }
            EventKind::WorkerCompleted { role, success, .. } => {
                write!(f, "worker_completed({}, ok={})", role, success)
            }
            EventKind::VerifierResult { passed, .. } => {
                write!(f, "verifier(passed={})", passed)
            }
            EventKind::EscalationTriggered {
                from_tier, to_tier, ..
            } => {
                write!(f, "escalation({:?} → {:?})", from_tier, to_tier)
            }
            EventKind::IterationCompleted {
                number, verified, ..
            } => {
                write!(f, "iteration_completed(#{}, verified={})", number, verified)
            }
            EventKind::ContextRebuilt { strategy, .. } => {
                write!(f, "context_rebuilt({})", strategy)
            }
            EventKind::NoChangeDetected {
                consecutive_count, ..
            } => {
                write!(f, "no_change({})", consecutive_count)
            }
            EventKind::SessionCompleted { resolved, .. } => {
                write!(f, "session_completed(resolved={})", resolved)
            }
            EventKind::CheckpointWritten { checkpoint_id, .. } => {
                write!(f, "checkpoint(#{})", checkpoint_id)
            }
            EventKind::Note { message } => {
                // Truncate at a char boundary to avoid panic on multi-byte UTF-8.
                let end = message
                    .char_indices()
                    .take_while(|(i, _)| *i < 50)
                    .last()
                    .map(|(i, c)| i + c.len_utf8())
                    .unwrap_or(0);
                write!(f, "note({})", &message[..end])
            }
        }
    }
}
