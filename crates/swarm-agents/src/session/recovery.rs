//! Crash recovery: reconstruct orchestrator state from session events.
//!
//! Implements the `wake()` pattern from Anthropic's Managed Agents: on
//! restart, the session log is replayed to reconstruct the state machine,
//! iteration count, and other mutable state — enabling the harness to
//! resume from where it left off.

use anyhow::{bail, Context, Result};
use tracing::{info, warn};

use crate::session::events::{EventKind, SessionEvent};
use crate::state_machine::OrchestratorState;

/// Tool names (and proxy-prefixed variants) that produce file writes.
/// Checked by name rather than substring to avoid false positives from
/// tools like "delegate_worker" whose name contains no write-related words
/// but whose inner worker may write files.
fn is_write_tool(name: &str) -> bool {
    matches!(
        name,
        "write_file"
            | "edit_file"
            | "patch"
            | "apply_plan"
            | "proxy_write_file"
            | "proxy_edit_file"
            // Worker delegation tools always produce file changes when they succeed.
            | "rust_coder"
            | "general_coder"
            | "fixer"
            | "delegate_worker"
            | "proxy_rust_coder"
            | "proxy_general_coder"
            | "proxy_fixer"
    )
}

/// State recovered from replaying session events.
///
/// Contains enough information to reconstruct an `OrchestratorContext`
/// without re-running initialization. The caller provides the external
/// dependencies (config, factory, agents); this struct provides the
/// internal state (state machine position, iteration count, etc.).
#[derive(Debug)]
pub struct RecoveredState {
    /// The issue ID this session is processing.
    pub issue_id: String,
    /// The objective/title of the issue.
    pub objective: String,
    /// The worktree path (from WorktreeProvisioned event).
    pub worktree_path: String,
    /// The branch name (from WorktreeProvisioned event).
    pub branch: String,
    /// The current orchestrator state (from last StateTransition).
    pub current_state: OrchestratorState,
    /// The current iteration number.
    pub iteration: u32,
    /// All state transitions replayed (for StateMachine reconstruction).
    pub transitions: Vec<crate::state_machine::TransitionRecord>,
    /// Total tool calls observed (for HarnessState reconstruction).
    pub tool_call_count: u32,
    /// Total LLM turns observed.
    pub llm_turn_count: u32,
    /// Whether any write/edit tool was called.
    pub has_written: bool,
    /// Last verifier result (if any).
    pub last_verifier_passed: Option<bool>,
    /// Number of consecutive no-change detections.
    pub consecutive_no_change: u32,
    /// The highest event ID seen (for resume position).
    pub last_event_id: u64,
}

/// Replay session events to recover orchestrator state.
///
/// Returns `None` if the session has already completed (terminal state).
/// Returns `Err` if the session log is empty or inconsistent.
pub fn recover_from_events(events: &[SessionEvent]) -> Result<Option<RecoveredState>> {
    if events.is_empty() {
        bail!("session log is empty — nothing to recover from");
    }

    // Extract session metadata from SessionStarted event.
    let (issue_id, objective) = events
        .iter()
        .find_map(|e| match &e.kind {
            EventKind::SessionStarted {
                issue_id,
                objective,
                ..
            } => Some((issue_id.clone(), objective.clone())),
            _ => None,
        })
        .context("no SessionStarted event found in session log")?;

    // Extract worktree info from WorktreeProvisioned event.
    let (worktree_path, branch) = events
        .iter()
        .find_map(|e| match &e.kind {
            EventKind::WorktreeProvisioned { path, branch, .. } => {
                Some((path.clone(), branch.clone()))
            }
            _ => None,
        })
        .unwrap_or_else(|| ("".into(), format!("swarm/{}", issue_id)));

    // Replay state transitions to find current state and iteration.
    let mut current_state = OrchestratorState::SelectingIssue;
    let mut iteration: u32 = 0;
    let mut transitions = Vec::new();

    // Replay tool/LLM metrics for HarnessState.
    let mut tool_call_count: u32 = 0;
    let mut llm_turn_count: u32 = 0;
    let mut has_written = false;
    let mut last_verifier_passed: Option<bool> = None;
    let mut consecutive_no_change: u32 = 0;
    let mut last_event_id: u64 = 0;

    for event in events {
        last_event_id = event.id;

        match &event.kind {
            EventKind::StateTransition {
                from,
                to,
                iteration: iter,
                reason,
            } => {
                current_state = *to;
                iteration = *iter;
                transitions.push(crate::state_machine::TransitionRecord {
                    from: *from,
                    to: *to,
                    iteration: *iter,
                    elapsed_ms: 0, // Can't reconstruct wall-clock from events
                    reason: reason.clone(),
                });
            }

            EventKind::IterationStarted { number, .. } => {
                iteration = *number;
            }

            EventKind::ToolCallCompleted {
                tool_name, success, ..
            } => {
                tool_call_count += 1;
                if *success && is_write_tool(tool_name) {
                    has_written = true;
                }
            }

            EventKind::LlmTurnCompleted { .. } => {
                llm_turn_count += 1;
            }

            EventKind::VerifierResult { passed, .. } => {
                last_verifier_passed = Some(*passed);
            }

            EventKind::NoChangeDetected {
                consecutive_count, ..
            } => {
                consecutive_no_change = *consecutive_count;
            }

            EventKind::IterationCompleted { .. } => {
                // Reset per-iteration counters.
                has_written = false;
                consecutive_no_change = 0;
            }

            EventKind::SessionCompleted { .. } => {
                // Session already finished — nothing to resume.
                info!(
                    issue = %issue_id,
                    state = %current_state,
                    "Session already completed — nothing to resume"
                );
                return Ok(None);
            }

            _ => {}
        }
    }

    // If we're in a terminal state, nothing to resume.
    if current_state.is_terminal() {
        info!(
            issue = %issue_id,
            state = %current_state,
            "Session reached terminal state — nothing to resume"
        );
        return Ok(None);
    }

    info!(
        issue = %issue_id,
        state = %current_state,
        iteration = iteration,
        transitions = transitions.len(),
        tool_calls = tool_call_count,
        llm_turns = llm_turn_count,
        last_event = last_event_id,
        "Recovered state from session events"
    );

    Ok(Some(RecoveredState {
        issue_id,
        objective,
        worktree_path,
        branch,
        current_state,
        iteration,
        transitions,
        tool_call_count,
        llm_turn_count,
        has_written,
        last_verifier_passed,
        consecutive_no_change,
        last_event_id,
    }))
}

/// Check whether a worktree has a resumable session.
///
/// Returns true if:
/// 1. A session log exists with events
/// 2. The session hasn't reached a terminal state
pub fn has_resumable_session(worktree_path: &std::path::Path) -> bool {
    let log_path = worktree_path.join(crate::session::SESSION_LOG_FILENAME);
    if !crate::session::SessionLog::exists_with_events(&log_path) {
        return false;
    }

    // Quick check: load events and see if session is still active.
    match crate::session::SessionLog::load_from_path(&log_path) {
        Ok(events) => {
            // Check if any SessionCompleted event exists.
            let completed = events
                .iter()
                .any(|e| matches!(e.kind, EventKind::SessionCompleted { .. }));
            if completed {
                return false;
            }

            // Check if the last state transition went to a terminal state.
            let last_state = events.iter().rev().find_map(|e| match &e.kind {
                EventKind::StateTransition { to, .. } => Some(*to),
                _ => None,
            });

            match last_state {
                Some(state) => !state.is_terminal(),
                None => true, // No transitions yet — session just started, resumable.
            }
        }
        Err(e) => {
            warn!(error = %e, "Failed to check session log for resume");
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Tier;
    use crate::session::events::{EventKind, SessionEvent};
    use chrono::Utc;

    fn make_event(id: u64, kind: EventKind) -> SessionEvent {
        SessionEvent {
            id,
            timestamp: Utc::now(),
            kind,
        }
    }

    #[test]
    fn test_recover_basic_session() {
        let events = vec![
            make_event(
                1,
                EventKind::SessionStarted {
                    issue_id: "test-123".into(),
                    objective: "Fix the bug".into(),
                    base_commit: Some("abc".into()),
                },
            ),
            make_event(
                2,
                EventKind::WorktreeProvisioned {
                    path: "/tmp/wt/test-123".into(),
                    branch: "swarm/test-123".into(),
                    commit: "abc".into(),
                },
            ),
            make_event(
                3,
                EventKind::StateTransition {
                    from: OrchestratorState::SelectingIssue,
                    to: OrchestratorState::PreparingWorktree,
                    iteration: 0,
                    reason: Some("issue selected".into()),
                },
            ),
            make_event(
                4,
                EventKind::StateTransition {
                    from: OrchestratorState::PreparingWorktree,
                    to: OrchestratorState::Planning,
                    iteration: 0,
                    reason: Some("worktree ready".into()),
                },
            ),
            make_event(
                5,
                EventKind::IterationStarted {
                    number: 1,
                    tier: Tier::Coder,
                },
            ),
            make_event(
                6,
                EventKind::StateTransition {
                    from: OrchestratorState::Planning,
                    to: OrchestratorState::Implementing,
                    iteration: 1,
                    reason: Some("plan ready".into()),
                },
            ),
        ];

        let recovered = recover_from_events(&events).unwrap().unwrap();
        assert_eq!(recovered.issue_id, "test-123");
        assert_eq!(recovered.current_state, OrchestratorState::Implementing);
        assert_eq!(recovered.iteration, 1);
        assert_eq!(recovered.transitions.len(), 3);
        assert_eq!(recovered.worktree_path, "/tmp/wt/test-123");
    }

    #[test]
    fn test_recover_completed_session_returns_none() {
        let events = vec![
            make_event(
                1,
                EventKind::SessionStarted {
                    issue_id: "test-456".into(),
                    objective: "Done".into(),
                    base_commit: None,
                },
            ),
            make_event(
                2,
                EventKind::SessionCompleted {
                    resolved: true,
                    total_iterations: 2,
                    duration_ms: 30000,
                    merge_commit: Some("xyz".into()),
                    failure_reason: None,
                },
            ),
        ];

        let result = recover_from_events(&events).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_recover_failed_terminal_returns_none() {
        let events = vec![
            make_event(
                1,
                EventKind::SessionStarted {
                    issue_id: "test-789".into(),
                    objective: "Will fail".into(),
                    base_commit: None,
                },
            ),
            make_event(
                2,
                EventKind::StateTransition {
                    from: OrchestratorState::Implementing,
                    to: OrchestratorState::Failed,
                    iteration: 3,
                    reason: Some("budget exhausted".into()),
                },
            ),
        ];

        let result = recover_from_events(&events).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_recover_empty_events_errors() {
        let result = recover_from_events(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_recover_tracks_tool_calls() {
        let events = vec![
            make_event(
                1,
                EventKind::SessionStarted {
                    issue_id: "test-tc".into(),
                    objective: "Tool tracking".into(),
                    base_commit: None,
                },
            ),
            make_event(
                2,
                EventKind::StateTransition {
                    from: OrchestratorState::SelectingIssue,
                    to: OrchestratorState::Implementing,
                    iteration: 1,
                    reason: None,
                },
            ),
            make_event(
                3,
                EventKind::ToolCallCompleted {
                    agent: "coder".into(),
                    tool_name: "read_file".into(),
                    success: true,
                    duration_ms: 50,
                    result_preview: "...".into(),
                },
            ),
            make_event(
                4,
                EventKind::ToolCallCompleted {
                    agent: "coder".into(),
                    tool_name: "write_file".into(),
                    success: true,
                    duration_ms: 100,
                    result_preview: "ok".into(),
                },
            ),
            make_event(
                5,
                EventKind::LlmTurnCompleted {
                    agent: "manager".into(),
                    model: "claude-opus-4-6".into(),
                    turn: 1,
                    tokens_in: Some(1000),
                    tokens_out: Some(500),
                    duration_ms: 2000,
                },
            ),
        ];

        let recovered = recover_from_events(&events).unwrap().unwrap();
        assert_eq!(recovered.tool_call_count, 2);
        assert_eq!(recovered.llm_turn_count, 1);
        assert!(recovered.has_written);
    }
}
