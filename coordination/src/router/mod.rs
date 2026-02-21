//! Model Router Module
//!
//! Routes tasks to appropriate LLM models based on:
//! - Error classification (from feedback module)
//! - Task complexity
//! - Previous failure patterns
//!
//! # Model Selection Strategy
//!
//! ```text
//! Error Type         | First Choice      | Escalation
//! -------------------|-------------------|-------------------
//! Type mismatch      | Strand (fast)     | Hydra → OR1
//! Import errors      | Strand (fast)     | Hydra → OR1
//! Borrow checker     | Hydra (special)   | OR1
//! Lifetime           | Hydra (special)   | OR1
//! Trait bounds       | Hydra (special)   | OR1
//! Async patterns     | Hydra (special)   | OR1
//! Complex/Multi      | OR1 (reasoning)   | -
//! ```

pub mod prompts;
pub mod task_classifier;

pub use prompts::{FixPromptBuilder, PromptTemplate};
pub use task_classifier::{
    DynamicRouter, ModelRouter, ModelSelection, ModelTier, PerformanceHistory, PerformanceRecord,
    TaskClassification, TaskType,
};
