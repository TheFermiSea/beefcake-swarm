//! Task classification and model routing
//!
//! Analyzes tasks and errors to select the optimal model.

use crate::feedback::error_parser::{ErrorCategory, ErrorSummary, ParsedError};
use serde::{Deserialize, Serialize};

/// Types of tasks that can be routed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    /// Generate code from scratch based on description
    CodeGeneration,
    /// Fix compilation errors
    ErrorFix,
    /// Refactor existing code
    Refactor,
    /// Explain code behavior
    Explain,
    /// Review code for issues
    Review,
    /// Architecture/design questions
    Architecture,
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CodeGeneration => write!(f, "code_generation"),
            Self::ErrorFix => write!(f, "error_fix"),
            Self::Refactor => write!(f, "refactor"),
            Self::Explain => write!(f, "explain"),
            Self::Review => write!(f, "review"),
            Self::Architecture => write!(f, "architecture"),
        }
    }
}

/// Task classification result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskClassification {
    /// Type of task
    pub task_type: TaskType,
    /// Complexity score (1-5)
    pub complexity: u8,
    /// Keywords found in the task
    pub keywords: Vec<String>,
    /// Error categories if this is an error fix task
    pub error_categories: Vec<ErrorCategory>,
    /// Recommended model tier
    pub recommended_tier: ModelTier,
}

/// Model tier for routing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Strand-Rust-Coder 14B - fast, good for simple fixes
    Fast,
    /// HydraCoder 31B MoE - specialized Rust knowledge
    Specialized,
    /// OR1-Behemoth 73B - deep reasoning, complex problems
    Reasoning,
}

impl ModelTier {
    /// Get the model identifier
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Fast => "Strand-Rust-Coder-14B-v1-Q8_0",
            Self::Specialized => "HydraCoder.Q6_K",
            Self::Reasoning => "OR1-Behemoth.Q8_0",
        }
    }

    /// Get expected tokens per second
    pub fn expected_speed(&self) -> u32 {
        match self {
            Self::Fast => 53,
            Self::Specialized => 50,
            Self::Reasoning => 11,
        }
    }

    /// Escalate to next tier
    pub fn escalate(&self) -> Self {
        match self {
            Self::Fast => Self::Specialized,
            Self::Specialized => Self::Reasoning,
            Self::Reasoning => Self::Reasoning,
        }
    }

    /// Get temperature for this tier
    pub fn default_temperature(&self) -> f32 {
        match self {
            Self::Fast => 0.3,
            Self::Specialized => 0.2,
            Self::Reasoning => 0.7,
        }
    }
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fast => write!(f, "fast"),
            Self::Specialized => write!(f, "specialized"),
            Self::Reasoning => write!(f, "reasoning"),
        }
    }
}

/// Model selection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelection {
    /// Selected model tier
    pub tier: ModelTier,
    /// Model identifier
    pub model_id: String,
    /// Recommended temperature
    pub temperature: f32,
    /// Maximum tokens for response
    pub max_tokens: u32,
    /// Reason for selection
    pub reason: String,
}

impl ModelSelection {
    /// Create a new selection
    pub fn new(tier: ModelTier, reason: impl Into<String>) -> Self {
        Self {
            tier,
            model_id: tier.model_id().to_string(),
            temperature: tier.default_temperature(),
            max_tokens: match tier {
                ModelTier::Fast => 1024,
                ModelTier::Specialized => 2048,
                ModelTier::Reasoning => 4096,
            },
            reason: reason.into(),
        }
    }

    /// With custom temperature
    pub fn with_temperature(mut self, temp: f32) -> Self {
        self.temperature = temp;
        self
    }

    /// With custom max tokens
    pub fn with_max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = tokens;
        self
    }
}

/// Model router for selecting the best model for a task
pub struct ModelRouter {
    /// Current failure count for escalation tracking
    failure_count: u32,
    /// Threshold for escalation
    escalation_threshold: u32,
    /// Last used tier (for tracking escalation)
    last_tier: Option<ModelTier>,
}

impl ModelRouter {
    /// Create a new router
    pub fn new() -> Self {
        Self {
            failure_count: 0,
            escalation_threshold: 2,
            last_tier: None,
        }
    }

    /// Create with custom escalation threshold
    pub fn with_escalation_threshold(mut self, threshold: u32) -> Self {
        self.escalation_threshold = threshold;
        self
    }

    /// Select model for code generation task
    pub fn select_for_generation(&self, description: &str) -> ModelSelection {
        let complexity = self.estimate_complexity(description);

        let tier = if complexity >= 4 {
            ModelTier::Reasoning
        } else if complexity >= 2 {
            ModelTier::Specialized
        } else {
            ModelTier::Fast
        };

        ModelSelection::new(
            tier,
            format!("Code generation with complexity {}/5", complexity),
        )
    }

    /// Select model for error fixing
    pub fn select_for_errors(&self, errors: &[ParsedError]) -> ModelSelection {
        if errors.is_empty() {
            return ModelSelection::new(ModelTier::Fast, "No errors to fix");
        }

        let summary = crate::feedback::error_parser::RustcErrorParser::summarize(errors);
        self.select_from_summary(&summary)
    }

    /// Select model based on error summary
    pub fn select_from_summary(&self, summary: &ErrorSummary) -> ModelSelection {
        // Check if we should escalate due to failures
        let base_tier = self.determine_base_tier(summary);
        let escalated_tier = self.apply_escalation(base_tier);

        let reason = self.build_reason(summary, base_tier, escalated_tier);

        ModelSelection::new(escalated_tier, reason)
    }

    /// Determine base tier from error summary
    fn determine_base_tier(&self, summary: &ErrorSummary) -> ModelTier {
        // Complex scenarios always use reasoning
        if summary.total >= 5 {
            return ModelTier::Reasoning;
        }

        // Lifetime and async errors need specialized handling
        if summary.has_lifetime_errors || summary.has_async_errors {
            return ModelTier::Specialized;
        }

        // Borrow checker errors benefit from specialization
        if summary.has_borrow_errors {
            return ModelTier::Specialized;
        }

        // High complexity errors
        if summary.max_complexity >= 3 {
            return ModelTier::Specialized;
        }

        // Simple errors
        ModelTier::Fast
    }

    /// Apply escalation based on failure history
    fn apply_escalation(&self, base_tier: ModelTier) -> ModelTier {
        if self.failure_count >= self.escalation_threshold {
            base_tier.escalate()
        } else {
            base_tier
        }
    }

    /// Build reason string for selection
    fn build_reason(&self, summary: &ErrorSummary, base: ModelTier, selected: ModelTier) -> String {
        let mut parts = vec![];

        parts.push(format!("{} errors", summary.total));

        if summary.has_lifetime_errors {
            parts.push("lifetime issues".to_string());
        }
        if summary.has_borrow_errors {
            parts.push("borrow checker".to_string());
        }
        if summary.has_async_errors {
            parts.push("async patterns".to_string());
        }

        if base != selected {
            parts.push(format!(
                "escalated from {} due to {} failures",
                base, self.failure_count
            ));
        }

        parts.join(", ")
    }

    /// Record a successful fix
    pub fn record_success(&mut self) {
        self.failure_count = 0;
    }

    /// Record a failed fix attempt
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
    }

    /// Reset failure tracking
    pub fn reset(&mut self) {
        self.failure_count = 0;
        self.last_tier = None;
    }

    /// Estimate complexity from description text
    fn estimate_complexity(&self, description: &str) -> u8 {
        let desc_lower = description.to_lowercase();
        let mut complexity = 1u8;

        // Keywords that indicate higher complexity
        let complex_keywords = [
            "lifetime",
            "borrow",
            "async",
            "await",
            "trait",
            "generic",
            "macro",
            "unsafe",
            "concurrency",
            "parallel",
            "lock",
            "mutex",
            "arc",
            "pin",
            "future",
        ];

        for keyword in complex_keywords {
            if desc_lower.contains(keyword) {
                complexity = complexity.saturating_add(1);
            }
        }

        // Length-based complexity
        if description.len() > 500 {
            complexity = complexity.saturating_add(1);
        }

        complexity.min(5)
    }

    /// Classify a task from its description
    pub fn classify_task(&self, description: &str) -> TaskClassification {
        let desc_lower = description.to_lowercase();

        // Detect task type from keywords
        let task_type = if desc_lower.contains("fix")
            || desc_lower.contains("error")
            || desc_lower.contains("compile")
        {
            TaskType::ErrorFix
        } else if desc_lower.contains("refactor") || desc_lower.contains("clean up") {
            TaskType::Refactor
        } else if desc_lower.contains("explain") || desc_lower.contains("what does") {
            TaskType::Explain
        } else if desc_lower.contains("review") || desc_lower.contains("check") {
            TaskType::Review
        } else if desc_lower.contains("design")
            || desc_lower.contains("architect")
            || desc_lower.contains("structure")
        {
            TaskType::Architecture
        } else {
            TaskType::CodeGeneration
        };

        let complexity = self.estimate_complexity(description);

        let recommended_tier = match task_type {
            TaskType::Architecture => ModelTier::Reasoning,
            TaskType::Explain | TaskType::Review => ModelTier::Reasoning,
            _ if complexity >= 4 => ModelTier::Reasoning,
            _ if complexity >= 2 => ModelTier::Specialized,
            _ => ModelTier::Fast,
        };

        TaskClassification {
            task_type,
            complexity,
            keywords: self.extract_keywords(&desc_lower),
            error_categories: vec![],
            recommended_tier,
        }
    }

    /// Extract relevant keywords from description
    fn extract_keywords(&self, description: &str) -> Vec<String> {
        let keywords = [
            "lifetime", "borrow", "async", "await", "trait", "generic", "macro", "unsafe", "error",
            "type", "struct", "enum", "impl", "fn", "mut", "ref",
        ];

        keywords
            .iter()
            .filter(|k| description.contains(*k))
            .map(|k| k.to_string())
            .collect()
    }
}

impl Default for ModelRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_tier_escalation() {
        assert_eq!(ModelTier::Fast.escalate(), ModelTier::Specialized);
        assert_eq!(ModelTier::Specialized.escalate(), ModelTier::Reasoning);
        assert_eq!(ModelTier::Reasoning.escalate(), ModelTier::Reasoning);
    }

    #[test]
    fn test_complexity_estimation() {
        let router = ModelRouter::new();

        // Simple task
        assert!(router.estimate_complexity("add two numbers") <= 2);

        // Complex task
        assert!(router.estimate_complexity("implement async trait with lifetime bounds") >= 3);
    }

    #[test]
    fn test_task_classification() {
        let router = ModelRouter::new();

        let fix_task = router.classify_task("fix the compilation error");
        assert_eq!(fix_task.task_type, TaskType::ErrorFix);

        let arch_task = router.classify_task("design the system architecture");
        assert_eq!(arch_task.task_type, TaskType::Architecture);
        assert_eq!(arch_task.recommended_tier, ModelTier::Reasoning);
    }

    #[test]
    fn test_failure_escalation() {
        let mut router = ModelRouter::new();

        // Initial selection should be based on complexity alone
        let selection = router.select_for_generation("simple function");
        assert_eq!(selection.tier, ModelTier::Fast);

        // Record failures
        router.record_failure();
        router.record_failure();

        // Should escalate after threshold
        // (Note: escalation applies to error-based selection)
    }

    #[test]
    fn test_model_selection() {
        let selection = ModelSelection::new(ModelTier::Specialized, "test reason")
            .with_temperature(0.5)
            .with_max_tokens(3000);

        assert_eq!(selection.tier, ModelTier::Specialized);
        assert_eq!(selection.temperature, 0.5);
        assert_eq!(selection.max_tokens, 3000);
    }
}
