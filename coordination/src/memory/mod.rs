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
//! - [`summarizer`] — Bounded summarizer contract and mock implementation
//! - [`compactor`] — Compaction orchestrator with event-driven triggers

pub mod budget;
pub mod compactor;
pub mod errors;
pub mod observability;
pub mod store;
pub mod summarizer;

pub use budget::{
    BudgetDecision, CompactionTrigger, TokenBudget, TokenEstimator, WordCountEstimator,
};
pub use compactor::{
    CompactionEvent, CompactionPolicy, CompactionResult, CompactionTriggerKind, MemoryCompactor,
};
pub use errors::{CompactionError, CompactionErrorKind, SummarizationError};
pub use observability::{CompactionMetrics, CompactionObserver, CompactionStats};
pub use store::{MemoryEntry, MemoryEntryKind, MemorySnapshot, SwarmMemory, SwarmMemoryStore};
pub use summarizer::{MockSummarizer, Summarizer, SummaryRequest, SummaryResponse};
