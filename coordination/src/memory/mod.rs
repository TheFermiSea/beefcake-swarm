//! Swarm Memory — Abstraction for conversation context management.
//!
//! Provides a memory abstraction compatible with existing message/session
//! structures, supporting compaction, token budgeting, and summary sentinels.
//!
//! # Modules
//!
//! - [`store`] — SwarmMemory trait, MemoryEntry, in-memory implementation
//! - [`errors`] — Typed error taxonomy for compaction and summarization
//! - [`budget`] — Token budgeting with pluggable estimators and compaction triggers

pub mod budget;
pub mod errors;
pub mod store;

pub use budget::{
    BudgetDecision, CompactionTrigger, TokenBudget, TokenEstimator, WordCountEstimator,
};
pub use errors::{CompactionError, CompactionErrorKind, SummarizationError};
pub use store::{MemoryEntry, MemoryEntryKind, MemorySnapshot, SwarmMemory, SwarmMemoryStore};
