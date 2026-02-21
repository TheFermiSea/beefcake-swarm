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
    /// HydraCoder 30B-A3B MoE — simple Rust fixes, fast
    Worker,
    /// Manager Council — complex tasks, architecture, review
    Council,
}

impl ModelTier {
    /// Get the model identifier
    pub fn model_id(&self) -> &'static str {
        match self {
            Self::Worker => "HydraCoder-Q6_K",
            Self::Council => "manager-council",
        }
    }

    /// Get expected tokens per second
    pub fn expected_speed(&self) -> u32 {
        match self {
            Self::Worker => 40,
            Self::Council => 10,
        }
    }

    /// Escalate to next tier
    pub fn escalate(&self) -> Self {
        match self {
            Self::Worker => Self::Council,
            Self::Council => Self::Council,
        }
    }

    /// Get temperature for this tier
    pub fn default_temperature(&self) -> f32 {
        match self {
            Self::Worker => 0.3,
            Self::Council => 0.5,
        }
    }
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Worker => write!(f, "worker"),
            Self::Council => write!(f, "council"),
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
                ModelTier::Worker => 2048,
                ModelTier::Council => 4096,
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
///
/// Stateless router — tier selection is purely deterministic based on
/// task characteristics. Escalation state lives in the correction loop.
pub struct ModelRouter;

impl ModelRouter {
    /// Create a new router
    pub fn new() -> Self {
        Self {}
    }

    /// Select model for code generation task
    pub fn select_for_generation(&self, description: &str) -> ModelSelection {
        let complexity = self.estimate_complexity(description);

        let tier = if complexity >= 3 {
            ModelTier::Council
        } else {
            ModelTier::Worker
        };

        ModelSelection::new(
            tier,
            format!("Code generation with complexity {}/5", complexity),
        )
    }

    /// Select model for error fixing
    pub fn select_for_errors(&self, errors: &[ParsedError]) -> ModelSelection {
        if errors.is_empty() {
            return ModelSelection::new(ModelTier::Worker, "No errors to fix");
        }

        let summary = crate::feedback::error_parser::RustcErrorParser::summarize(errors);
        self.select_from_summary(&summary)
    }

    /// Select model based on error summary
    pub fn select_from_summary(&self, summary: &ErrorSummary) -> ModelSelection {
        let tier = self.determine_base_tier(summary);
        let reason = self.build_reason(summary, tier);

        ModelSelection::new(tier, reason)
    }

    /// Determine base tier from error summary
    ///
    /// Council: lifetime, async, trait bounds, >= 5 errors, complexity >= 3
    /// Worker: type mismatch, import resolution, simple single-file borrow errors, complexity < 3
    fn determine_base_tier(&self, summary: &ErrorSummary) -> ModelTier {
        // Many errors need council coordination
        if summary.total >= 5 {
            return ModelTier::Council;
        }

        // Lifetime and async errors need council
        if summary.has_lifetime_errors || summary.has_async_errors {
            return ModelTier::Council;
        }

        // Trait bound errors need council
        if summary.by_category.contains_key(&ErrorCategory::TraitBound) {
            return ModelTier::Council;
        }

        // High complexity errors
        if summary.max_complexity >= 3 {
            return ModelTier::Council;
        }

        // Simple borrow errors on single files stay with worker
        // (multi-file borrow issues would show higher complexity)

        // Simple errors: type mismatch, imports, low-complexity borrow
        ModelTier::Worker
    }

    /// Build reason string for selection
    fn build_reason(&self, summary: &ErrorSummary, tier: ModelTier) -> String {
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

        parts.push(format!("routed to {}", tier));

        parts.join(", ")
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
            TaskType::Architecture | TaskType::Explain | TaskType::Review => ModelTier::Council,
            _ if complexity >= 3 => ModelTier::Council,
            _ => ModelTier::Worker,
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

// ── Dynamic routing with historical performance ──────────────────────────────

use std::collections::HashMap;

/// Tracks success/failure counts for a single routing slot
#[derive(Debug, Clone, Default)]
pub struct PerformanceRecord {
    /// Number of successful completions
    pub successes: u32,
    /// Number of failures
    pub failures: u32,
}

impl PerformanceRecord {
    /// Record a successful outcome
    pub fn record_success(&mut self) {
        self.successes += 1;
    }

    /// Record a failed outcome
    pub fn record_failure(&mut self) {
        self.failures += 1;
    }

    /// Success rate in [0.0, 1.0]. Returns 0.5 when no data is available.
    pub fn success_rate(&self) -> f32 {
        let total = self.total();
        if total == 0 {
            0.5
        } else {
            self.successes as f32 / total as f32
        }
    }

    /// Total number of recorded outcomes
    pub fn total(&self) -> u32 {
        self.successes + self.failures
    }
}

/// Historical performance data used by `DynamicRouter`
#[derive(Debug, Clone, Default)]
pub struct PerformanceHistory {
    /// Per-tier success rates
    pub tier_performance: HashMap<ModelTier, PerformanceRecord>,
    /// Per-(tier, category) success rates
    pub category_performance: HashMap<(ModelTier, String), PerformanceRecord>,
}

impl PerformanceHistory {
    /// Create a new, empty history
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a routing outcome
    pub fn record_outcome(&mut self, tier: ModelTier, category: Option<&str>, success: bool) {
        let tier_record = self.tier_performance.entry(tier).or_default();
        if success {
            tier_record.record_success();
        } else {
            tier_record.record_failure();
        }

        if let Some(cat) = category {
            let cat_record = self
                .category_performance
                .entry((tier, cat.to_string()))
                .or_default();
            if success {
                cat_record.record_success();
            } else {
                cat_record.record_failure();
            }
        }
    }

    /// Success rate for a tier (0.5 if no data)
    pub fn success_rate_for_tier(&self, tier: ModelTier) -> f32 {
        self.tier_performance
            .get(&tier)
            .map(|r| r.success_rate())
            .unwrap_or(0.5)
    }

    /// Success rate for a (tier, category) pair, falling back to tier rate
    pub fn success_rate_for_category(&self, tier: ModelTier, category: &str) -> f32 {
        self.category_performance
            .get(&(tier, category.to_string()))
            .map(|r| r.success_rate())
            .unwrap_or_else(|| self.success_rate_for_tier(tier))
    }

    /// Return the tier with better historical performance for this category.
    ///
    /// The non-base tier is preferred only when it has **≥ 10 % better** success
    /// rate **and** at least 3 total attempts for that (tier, category) slot.
    /// Otherwise the base tier is returned unchanged.
    pub fn preferred_tier_for_category(&self, category: &str, base_tier: ModelTier) -> ModelTier {
        let other_tier = match base_tier {
            ModelTier::Worker => ModelTier::Council,
            ModelTier::Council => ModelTier::Worker,
        };

        let other_record = self
            .category_performance
            .get(&(other_tier, category.to_string()));

        // If there is no data for the other tier, stay with base
        let other_record = match other_record {
            Some(r) => r,
            None => return base_tier,
        };

        // Require at least 3 attempts before trusting the data
        if other_record.total() < 3 {
            return base_tier;
        }

        let base_rate = self.success_rate_for_category(base_tier, category);
        let other_rate = other_record.success_rate();

        // Prefer the other tier only if it is meaningfully better (≥ 10 %)
        if other_rate >= base_rate + 0.10 {
            other_tier
        } else {
            base_tier
        }
    }
}

/// Wraps `ModelRouter` and adjusts routing decisions based on `PerformanceHistory`
pub struct DynamicRouter {
    base_router: ModelRouter,
    history: PerformanceHistory,
}

impl DynamicRouter {
    /// Create a new dynamic router with empty history
    pub fn new() -> Self {
        Self {
            base_router: ModelRouter::new(),
            history: PerformanceHistory::new(),
        }
    }

    /// Like `ModelRouter::select_for_errors` but may override the tier based on history
    pub fn select_for_errors_dynamic(&self, errors: &[ParsedError]) -> ModelSelection {
        let base = self.base_router.select_for_errors(errors);

        // Nothing to adjust when there are no errors
        if errors.is_empty() {
            return base;
        }

        // Use the category of the first (primary) error as the routing key
        let category = errors[0].category.to_string();
        let preferred = self
            .history
            .preferred_tier_for_category(&category, base.tier);

        if preferred != base.tier {
            let rate = self.history.success_rate_for_category(preferred, &category);
            let reason = format!("historical performance: {:.0}% success rate", rate * 100.0);
            ModelSelection::new(preferred, reason)
        } else {
            base
        }
    }

    /// Record the outcome of a routing decision
    pub fn record_outcome(
        &mut self,
        selection: &ModelSelection,
        category: Option<&str>,
        success: bool,
    ) {
        self.history
            .record_outcome(selection.tier, category, success);
    }

    /// Access the underlying performance history
    pub fn history(&self) -> &PerformanceHistory {
        &self.history
    }
}

impl Default for DynamicRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_tier_escalation() {
        assert_eq!(ModelTier::Worker.escalate(), ModelTier::Council);
        assert_eq!(ModelTier::Council.escalate(), ModelTier::Council);
    }

    #[test]
    fn test_model_tier_properties() {
        assert_eq!(ModelTier::Worker.model_id(), "HydraCoder-Q6_K");
        assert_eq!(ModelTier::Council.model_id(), "manager-council");

        assert_eq!(ModelTier::Worker.expected_speed(), 40);
        assert_eq!(ModelTier::Council.expected_speed(), 10);

        assert!((ModelTier::Worker.default_temperature() - 0.3).abs() < f32::EPSILON);
        assert!((ModelTier::Council.default_temperature() - 0.5).abs() < f32::EPSILON);

        assert_eq!(format!("{}", ModelTier::Worker), "worker");
        assert_eq!(format!("{}", ModelTier::Council), "council");
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
        assert_eq!(arch_task.recommended_tier, ModelTier::Council);
    }

    #[test]
    fn test_simple_task_routes_to_worker() {
        let router = ModelRouter::new();

        let selection = router.select_for_generation("simple function");
        assert_eq!(selection.tier, ModelTier::Worker);
    }

    #[test]
    fn test_complex_task_routes_to_council() {
        let router = ModelRouter::new();

        let selection = router.select_for_generation("implement async trait with lifetime bounds");
        assert_eq!(selection.tier, ModelTier::Council);
    }

    #[test]
    fn test_review_explain_route_to_council() {
        let router = ModelRouter::new();

        let explain = router.classify_task("explain the borrow checker behavior");
        assert_eq!(explain.recommended_tier, ModelTier::Council);

        let review = router.classify_task("review this module for issues");
        assert_eq!(review.recommended_tier, ModelTier::Council);
    }

    #[test]
    fn test_model_selection() {
        let selection = ModelSelection::new(ModelTier::Worker, "test reason")
            .with_temperature(0.5)
            .with_max_tokens(3000);

        assert_eq!(selection.tier, ModelTier::Worker);
        assert_eq!(selection.temperature, 0.5);
        assert_eq!(selection.max_tokens, 3000);
    }

    #[test]
    fn test_model_selection_defaults() {
        let worker = ModelSelection::new(ModelTier::Worker, "worker task");
        assert_eq!(worker.max_tokens, 2048);
        assert_eq!(worker.model_id, "HydraCoder-Q6_K");

        let council = ModelSelection::new(ModelTier::Council, "council task");
        assert_eq!(council.max_tokens, 4096);
        assert_eq!(council.model_id, "manager-council");
    }

    #[test]
    fn test_stateless_router() {
        // Router is stateless — two identical calls produce identical results
        let router = ModelRouter::new();

        let first = router.select_for_generation("simple function");
        let second = router.select_for_generation("simple function");
        assert_eq!(first.tier, second.tier);
        assert_eq!(first.model_id, second.model_id);
    }

    // ── PerformanceRecord tests ──────────────────────────────────────────────

    #[test]
    fn test_performance_record_no_data() {
        let record = PerformanceRecord::default();
        assert_eq!(record.total(), 0);
        // No data → neutral 0.5
        assert!((record.success_rate() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_performance_record_success_rate() {
        let mut record = PerformanceRecord::default();
        record.record_success();
        record.record_success();
        record.record_success();
        record.record_failure();
        // 3 successes out of 4 → 0.75
        assert_eq!(record.total(), 4);
        assert!((record.success_rate() - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_performance_record_all_failures() {
        let mut record = PerformanceRecord::default();
        record.record_failure();
        record.record_failure();
        assert!((record.success_rate() - 0.0).abs() < f32::EPSILON);
    }

    // ── PerformanceHistory tests ─────────────────────────────────────────────

    #[test]
    fn test_performance_history_no_data() {
        let history = PerformanceHistory::new();
        // Both tiers return neutral rate when empty
        assert!((history.success_rate_for_tier(ModelTier::Worker) - 0.5).abs() < f32::EPSILON);
        assert!((history.success_rate_for_tier(ModelTier::Council) - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_performance_history_record_and_retrieve() {
        let mut history = PerformanceHistory::new();
        history.record_outcome(ModelTier::Worker, Some("borrow_checker"), true);
        history.record_outcome(ModelTier::Worker, Some("borrow_checker"), true);
        history.record_outcome(ModelTier::Worker, Some("borrow_checker"), false);

        // Tier rate: 2/3 ≈ 0.667
        let tier_rate = history.success_rate_for_tier(ModelTier::Worker);
        assert!((tier_rate - 2.0 / 3.0).abs() < 1e-5);

        // Category rate: same data → 2/3
        let cat_rate = history.success_rate_for_category(ModelTier::Worker, "borrow_checker");
        assert!((cat_rate - 2.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_performance_history_category_fallback() {
        let mut history = PerformanceHistory::new();
        // Record tier-level data only (no category)
        history.record_outcome(ModelTier::Council, None, true);
        history.record_outcome(ModelTier::Council, None, false);

        // Category with no specific data falls back to tier rate (0.5)
        let rate = history.success_rate_for_category(ModelTier::Council, "lifetime");
        assert!((rate - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_preferred_tier_no_data_returns_base() {
        let history = PerformanceHistory::new();
        // No data → always return base tier
        assert_eq!(
            history.preferred_tier_for_category("type_mismatch", ModelTier::Worker),
            ModelTier::Worker
        );
        assert_eq!(
            history.preferred_tier_for_category("lifetime", ModelTier::Council),
            ModelTier::Council
        );
    }

    #[test]
    fn test_preferred_tier_insufficient_data_returns_base() {
        let mut history = PerformanceHistory::new();
        // Only 2 attempts for Council on "lifetime" — below the 3-attempt threshold
        history.record_outcome(ModelTier::Council, Some("lifetime"), true);
        history.record_outcome(ModelTier::Council, Some("lifetime"), true);

        assert_eq!(
            history.preferred_tier_for_category("lifetime", ModelTier::Worker),
            ModelTier::Worker
        );
    }

    #[test]
    fn test_preferred_tier_switches_when_better() {
        let mut history = PerformanceHistory::new();
        // Council has 4/4 = 100 % on "lifetime"; Worker has 0 data → 0.5 fallback
        for _ in 0..4 {
            history.record_outcome(ModelTier::Council, Some("lifetime"), true);
        }

        // Council is ≥ 10 % better than Worker's neutral 0.5, so prefer Council
        assert_eq!(
            history.preferred_tier_for_category("lifetime", ModelTier::Worker),
            ModelTier::Council
        );
    }

    // ── DynamicRouter tests ──────────────────────────────────────────────────

    #[test]
    fn test_dynamic_router_no_history_falls_back_to_base() {
        let router = DynamicRouter::new();
        // No errors → Worker (same as base router)
        let selection = router.select_for_errors_dynamic(&[]);
        assert_eq!(selection.tier, ModelTier::Worker);
    }

    #[test]
    fn test_dynamic_router_with_history_overrides_tier() {
        use crate::feedback::error_parser::{ErrorCategory, ParsedError};

        let mut router = DynamicRouter::new();

        // Simulate Council succeeding 4 times on "lifetime" errors
        let category_str = ErrorCategory::Lifetime.to_string();
        for _ in 0..4 {
            let sel = ModelSelection::new(ModelTier::Council, "test");
            router.record_outcome(&sel, Some(&category_str), true);
        }

        // Build a fake lifetime ParsedError using the actual struct fields
        let fake_error = ParsedError {
            category: ErrorCategory::Lifetime,
            code: None,
            message: "lifetime error".to_string(),
            file: None,
            line: None,
            column: None,
            suggestion: None,
            rendered: "lifetime error".to_string(),
            labels: vec![],
        };

        // Base router would pick Council for lifetime anyway, but let's verify
        // the dynamic path works end-to-end without panicking
        let selection = router.select_for_errors_dynamic(&[fake_error]);
        // Council should be selected (either by base logic or by history)
        assert_eq!(selection.tier, ModelTier::Council);
    }

    #[test]
    fn test_dynamic_router_history_accessor() {
        let mut router = DynamicRouter::new();
        router.record_outcome(
            &ModelSelection::new(ModelTier::Worker, "test"),
            Some("type_mismatch"),
            true,
        );
        let history = router.history();
        assert_eq!(history.success_rate_for_tier(ModelTier::Worker), 1.0);
    }
}
