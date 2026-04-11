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

pub use events::{EventId, EventKind, SessionEvent};
pub use log::SessionLog;

/// Default filename for the session log within a worktree.
pub const SESSION_LOG_FILENAME: &str = ".swarm-session.jsonl";
