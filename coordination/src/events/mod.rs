//! Event-driven coordination module for multi-agent ensemble
//!
//! This module provides the pub/sub messaging infrastructure for
//! coordinating between models and persisting event history for replay.
//!
//! # Architecture
//!
//! The event system consists of three main components:
//!
//! 1. **Event Types** (`types.rs`): Defines the 13 event types that
//!    drive ensemble coordination, from task creation to arbitration.
//!
//! 2. **Event Bus** (`bus.rs`): Tokio broadcast-based pub/sub with
//!    optional persistence to RocksDB.
//!
//! 3. **Event History** (`history.rs`): Query and replay capabilities
//!    for debugging and recovery.
//!
//! # Event Flow
//!
//! ```text
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │   Producer   │────▶│  Event Bus   │────▶│  Subscribers │
//! │  (publish)   │     │  (broadcast) │     │   (recv)     │
//! └──────────────┘     └──────┬───────┘     └──────────────┘
//!                             │
//!                             ▼
//!                      ┌──────────────┐
//!                      │   RocksDB    │
//!                      │  (persist)   │
//!                      └──────────────┘
//! ```
//!
//! # Usage
//!
//! ```ignore
//! use rust_cluster_mcp::events::{EventBus, EnsembleEvent, EventHistory};
//! use chrono::Utc;
//!
//! // Create event bus with persistence
//! let bus = EventBus::with_persistence(store.clone()).shared();
//!
//! // Subscribe to events
//! let mut receiver = bus.subscribe();
//!
//! // Publish an event
//! bus.publish(EnsembleEvent::TaskCreated {
//!     task_id: "task-1".to_string(),
//!     session_id: "session-1".to_string(),
//!     prompt_preview: "Analyze...".to_string(),
//!     require_consensus: true,
//!     timestamp: Utc::now(),
//! })?;
//!
//! // Receive event
//! let event = receiver.recv().await?;
//!
//! // Replay history
//! let history = EventHistory::new(store);
//! let recent = history.get_recent_events(60)?; // Last hour
//! ```

pub mod bus;
pub mod history;
pub mod types;

// Re-export core types
pub use bus::{
    EventBus, EventBusError, EventBusExt, EventBusResult, EventFilter, FilteredReceiver,
    SharedEventBus,
};
pub use history::{
    EventHistory, EventStats, HistoryError, HistoryResult, ReplayBuilder, ReplayStats,
};
pub use types::{
    ArbitrationReason, ContextUpdater, EnsembleEvent, EventId, SessionEndReason, UnloadReason,
    VoteSummary,
};
