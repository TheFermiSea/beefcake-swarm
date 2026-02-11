//! Agent Harness Module
//!
//! Implements Anthropic's patterns for effective long-running agents:
//! - Session state persistence across context windows
//! - Feature specification registry (JSON-based)
//! - Progress tracking (claude-progress.txt pattern)
//! - Git-based state management for rollback/recovery
//! - Startup ritual automation
//!
//! Reference: <https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents>

pub mod error;
pub mod feature_registry;
pub mod git_manager;
pub mod progress;
pub mod session;
pub mod startup;
pub mod tools;
pub mod types;

pub use error::{HarnessError, HarnessResult};
pub use tools::{create_shared_state, HarnessState, SharedHarnessState};
pub use types::*;
