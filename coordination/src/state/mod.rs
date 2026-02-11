//! State persistence module for multi-agent ensemble coordination
//!
//! This module provides RocksDB-backed persistent storage for:
//! - Ensemble sessions that survive model swaps
//! - Tasks and their results from multiple models
//! - Voting records for consensus decisions
//! - Shared context across model executions
//! - Event history for replay and debugging
//!
//! # Architecture
//!
//! The state store uses RocksDB column families to logically separate different
//! data types while sharing a single database instance:
//!
//! - `sessions`: EnsembleSession tracking overall coordination
//! - `tasks`: EnsembleTask for work items
//! - `results`: ModelResult from each LLM execution
//! - `voting`: VoteRecord for consensus decisions
//! - `context`: SharedContext maintained across model swaps
//! - `events`: Event history for replay
//!
//! # Usage
//!
//! ```ignore
//! use rust_cluster_mcp::state::{StateStore, EnsembleSession, EnsembleTask};
//!
//! // Open or create the state store
//! let store = StateStore::open("./ensemble-state")?;
//!
//! // Create and store a session
//! let session = EnsembleSession::new();
//! store.put_session(&session)?;
//!
//! // Create a task for ensemble processing
//! let task = EnsembleTask::new(
//!     session.id.clone(),
//!     "Analyze this Rust code".to_string(),
//!     true, // require consensus
//! );
//! store.put_task(&task)?;
//! ```

pub mod schema;
pub mod store;
pub mod types;

// Re-export core types
pub use store::{SharedStateStore, StateStore, StoreError, StoreResult};
pub use types::{
    EnsembleSession, EnsembleTask, ModelId, ModelResult, SessionId, SharedContext, TaskId,
    TaskStatus, VoteRecord, VotingStrategy,
};
