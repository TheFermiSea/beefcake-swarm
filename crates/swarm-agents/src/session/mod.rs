//! Event-sourced session log for orchestrator crash recovery.
//!
//! Implements the "durable session" pattern from Anthropic's Managed Agents
//! architecture: an append-only event log that records every significant
//! orchestrator action, enabling crash recovery via `wake()` and queryable
//! context management.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────┐
//! │  Harness (driver.rs)            │  ← stateless, can crash & restart
//! │  emits events via SessionLog    │
//! └──────────┬──────────────────────┘
//!            │ append()
//!            ▼
//! ┌─────────────────────────────────┐
//! │  SessionLog (.swarm-session.jsonl)  │  ← durable, append-only
//! │  survives crashes                    │
//! │  queryable (load_since, by_type)     │
//! └─────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use swarm_agents::session::{SessionLog, EventKind};
//!
//! let log = SessionLog::open("/tmp/wt/issue-123/.swarm-session.jsonl")?;
//!
//! // Emit events during orchestration.
//! log.append(EventKind::SessionStarted {
//!     issue_id: "issue-123".into(),
//!     objective: "Fix the borrow checker error".into(),
//!     base_commit: Some("abc123".into()),
//! })?;
//!
//! // On crash recovery, replay events to reconstruct state.
//! let events = log.load_all()?;
//! ```

pub mod events;
pub mod log;
pub mod recovery;

pub use events::{EventId, EventKind, SessionEvent};
pub use log::SessionLog;
pub use recovery::{has_resumable_session, recover_from_events, RecoveredState};

/// Subdirectory holding harness state within a worktree — kept out of agent tool reach.
pub const SWARM_STATE_DIR: &str = ".swarm";

pub const SESSION_LOG_FILENAME: &str = ".swarm/session.jsonl";
pub const PROGRESS_FILENAME: &str = ".swarm/progress.txt";
pub const CHECKPOINT_FILENAME: &str = ".swarm/checkpoint.json";
pub const CHECKPOINT_TMP_FILENAME: &str = ".swarm/checkpoint.json.tmp";
pub const SESSION_STATE_FILENAME: &str = ".swarm/session-state.json";

pub fn session_log_path(wt_root: &std::path::Path) -> std::path::PathBuf {
    wt_root.join(SESSION_LOG_FILENAME)
}

pub fn progress_path(wt_root: &std::path::Path) -> std::path::PathBuf {
    wt_root.join(PROGRESS_FILENAME)
}

pub fn checkpoint_path(wt_root: &std::path::Path) -> std::path::PathBuf {
    wt_root.join(CHECKPOINT_FILENAME)
}

pub fn checkpoint_tmp_path(wt_root: &std::path::Path) -> std::path::PathBuf {
    wt_root.join(CHECKPOINT_TMP_FILENAME)
}

pub fn session_state_path(wt_root: &std::path::Path) -> std::path::PathBuf {
    wt_root.join(SESSION_STATE_FILENAME)
}
