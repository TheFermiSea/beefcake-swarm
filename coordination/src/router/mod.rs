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

pub mod circuit_breaker;
pub mod classifier;
pub mod prompts;
pub mod task_classifier;

pub use circuit_breaker::{CircuitBreaker, CircuitState, FallbackLadder};
pub use classifier::{
    ComplexityFactors, PreRoutingAnalysis, PreRoutingClassifier, RiskFactor, RiskKind, RiskLevel,
};
pub use prompts::{FixPromptBuilder, PromptTemplate};
pub use task_classifier::{
    DynamicRouter, ModelRouter, ModelSelection, ModelTier, PerformanceHistory, PerformanceRecord,
    ScoringWeights, SmartScore, TaskClassification, TaskType,
};
