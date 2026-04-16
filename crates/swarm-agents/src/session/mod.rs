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

/// Default subdirectory for harness state within a worktree.
///
/// Historically, files lived at the worktree root (`.swarm-session.jsonl`,
/// `.swarm-progress.txt`, `.swarm-checkpoint.json`). Agents observed burning
/// 1,705 tool calls across 20 runs poking at these top-level files instead of
/// the actual target source. Moving them under `.swarm/` combined with the
/// FORBIDDEN_PREFIXES entry keeps them out of `list_files` output AND out of
/// agent reach.
pub const SWARM_STATE_DIR: &str = ".swarm";

/// Default filename for the session log within the swarm state dir.
/// Full relative path: `.swarm/session.jsonl`.
pub const SESSION_LOG_FILENAME: &str = ".swarm/session.jsonl";

/// Full relative path helper for callers that need the path from a worktree root.
pub fn session_log_path(wt_root: &std::path::Path) -> std::path::PathBuf {
    wt_root.join(SESSION_LOG_FILENAME)
}
