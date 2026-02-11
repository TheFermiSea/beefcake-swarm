//! Multi-agent ensemble coordination module
//!
//! This module provides the orchestration layer for coordinating multiple
//! Rust-specialized LLMs (OR1-Behemoth, Strand-Rust-Coder, HydraCoder) with
//! Claude as the overseer.
//!
//! # Architecture
//!
//! ```text
//! ┌───────────────────────────────────────────────────────────┐
//! │                     Claude (Overseer)                      │
//! │  • Submits tasks via MCP tools                            │
//! │  • Arbitrates disputes                                     │
//! │  • Updates context                                         │
//! └─────────────────────────┬─────────────────────────────────┘
//!                           │
//!                           ▼
//! ┌───────────────────────────────────────────────────────────┐
//! │                 EnsembleCoordinator                        │
//! │  • Manages sessions and tasks                              │
//! │  • Orchestrates model execution                            │
//! │  • Triggers voting                                         │
//! └─────────────────────────┬─────────────────────────────────┘
//!                           │
//!           ┌───────────────┼───────────────┐
//!           ▼               ▼               ▼
//!     ┌───────────┐   ┌───────────┐   ┌───────────┐
//!     │  Voting   │   │  Context  │   │Arbitration│
//!     │ Protocol  │   │  Manager  │   │  Manager  │
//!     └───────────┘   └───────────┘   └───────────┘
//! ```
//!
//! # Components
//!
//! - **Coordinator**: Central orchestrator managing the ensemble workflow
//! - **VotingProtocol**: Implements majority, weighted, and unanimous voting
//! - **ContextManager**: Maintains shared context across model swaps
//! - **ArbitrationManager**: Handles disputes and Claude intervention
//!
//! # Workflow
//!
//! 1. Claude calls `ensemble_start` to create a session
//! 2. Claude calls `ensemble_submit` with a task (prompt + require_consensus)
//! 3. Coordinator executes models sequentially:
//!    - Load model → execute → store result → unload
//! 4. When all models complete, voting is triggered
//! 5. If consensus fails (tie or low confidence), arbitration is requested
//! 6. Claude makes final decision via `ensemble_arbitrate`
//! 7. Context is updated for future tasks
//!
//! # Usage
//!
//! ```ignore
//! use rust_cluster_mcp::ensemble::{EnsembleCoordinator, EnsembleConfig};
//! use rust_cluster_mcp::state::StateStore;
//! use rust_cluster_mcp::events::EventBus;
//!
//! // Setup
//! let store = StateStore::open("./ensemble-state")?.shared();
//! let bus = EventBus::with_persistence(store.clone()).shared();
//! let config = EnsembleConfig::default();
//!
//! let coordinator = EnsembleCoordinator::new(store, bus, config)?;
//!
//! // Start session
//! let session = coordinator.start_session(None).await?;
//!
//! // Submit task requiring consensus
//! let task = coordinator.submit_task(
//!     "Implement error handling for API client".to_string(),
//!     Some(existing_code),
//!     true, // require_consensus
//! ).await?;
//!
//! // Execute task (runs all models)
//! let task = coordinator.execute_task(&task.id).await?;
//!
//! // Vote on results
//! let outcome = coordinator.vote_on_task(&task.id, None).await?;
//! println!("Winner: {:?}", outcome.winner);
//! ```

pub mod arbitration;
pub mod context;
pub mod coordinator;
pub mod voting;

// Re-export core types
pub use arbitration::{
    ArbitrationDecision, ArbitrationError, ArbitrationManager, ArbitrationRequest,
    ArbitrationResult, ModelResponseSummary,
};
pub use context::{ContextError, ContextManager, ContextResult, ContextSnapshot};
pub use coordinator::{
    CoordinatorError, CoordinatorResult, EnsembleConfig, EnsembleCoordinator, EnsembleStatus,
    SharedEnsembleCoordinator,
};
pub use voting::{VoteOutcome, VotingError, VotingProtocol, VotingResult};
