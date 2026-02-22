//! Worker-First Policy — Intelligent Initial Tier Selection
//!
//! When worker-first mode is enabled, tasks start at the Worker tier
//! (local models) instead of the Council tier (cloud manager). The
//! cloud manager is only invoked when escalation triggers fire.
//!
//! # Classification
//!
//! The classifier analyzes task descriptions and error categories to
//! determine whether a task is simple enough for direct worker handling:
//!
//! ```text
//! Task → WorkerFirstClassifier → InitialTierRecommendation
//!   │                                    │
//!   │  Simple (lint, type fix, imports)   │→ Worker
//!   │  Medium (borrow, lifetime)         │→ Worker (with escalation readiness)
//!   │  Complex (multi-file, arch)        │→ Council
//!   │  Unknown (first attempt)           │→ Worker (default for worker-first)
//! ```
//!
//! # Integration
//!
//! The orchestrator checks `FeatureFlags::worker_first_enabled` and
//! calls `classify_initial_tier()` to determine the starting tier
//! instead of defaulting to Council.

use crate::escalation::state::SwarmTier;
use crate::feedback::error_parser::ErrorCategory;
use serde::{Deserialize, Serialize};

/// Complexity assessment for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskComplexity {
    /// Single-file fixes: lint, formatting, type mismatches, imports.
    Simple,
    /// Moderate Rust-specific: borrow checker, lifetimes, trait bounds.
    Medium,
    /// Multi-file changes, architectural issues, async patterns.
    Complex,
    /// Not enough information to classify.
    Unknown,
}

impl std::fmt::Display for TaskComplexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Simple => write!(f, "simple"),
            Self::Medium => write!(f, "medium"),
            Self::Complex => write!(f, "complex"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// Recommendation for the initial tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialTierRecommendation {
    /// Recommended starting tier.
    pub tier: SwarmTier,
    /// Assessed complexity.
    pub complexity: TaskComplexity,
    /// Reason for the recommendation.
    pub reason: String,
    /// Confidence score (0.0 to 1.0).
    pub confidence: f64,
}

/// Keywords that indicate simple tasks solvable by workers.
const SIMPLE_KEYWORDS: &[&str] = &[
    "lint",
    "format",
    "formatting",
    "clippy",
    "unused",
    "dead_code",
    "import",
    "use statement",
    "type mismatch",
    "typo",
    "rename",
    "visibility",
    "pub(",
    "derive",
    "missing field",
    "add field",
    "remove field",
    "cfg(",
    "feature gate",
    "documentation",
    "doc comment",
    "test fix",
    "update test",
];

/// Keywords that indicate medium-complexity tasks.
const MEDIUM_KEYWORDS: &[&str] = &[
    "borrow",
    "lifetime",
    "trait bound",
    "generic",
    "where clause",
    "impl ",
    "dyn ",
    "ownership",
    "move ",
    "reference",
    "&mut",
    "clone",
    "copy",
];

/// Keywords that indicate complex tasks needing manager.
const COMPLEX_KEYWORDS: &[&str] = &[
    "refactor",
    "redesign",
    "architecture",
    "multi-file",
    "cross-module",
    "async",
    "tokio",
    "concurrency",
    "parallel",
    "migration",
    "breaking change",
    "api change",
    "new module",
    "new crate",
    "state machine",
    "orchestrat",
    "distributed",
];

/// Classify a task based on its description.
pub fn classify_from_description(description: &str) -> TaskComplexity {
    let lower = description.to_lowercase();

    let simple_hits = SIMPLE_KEYWORDS
        .iter()
        .filter(|kw| lower.contains(*kw))
        .count();
    let medium_hits = MEDIUM_KEYWORDS
        .iter()
        .filter(|kw| lower.contains(*kw))
        .count();
    let complex_hits = COMPLEX_KEYWORDS
        .iter()
        .filter(|kw| lower.contains(*kw))
        .count();

    // Complex signals dominate
    if complex_hits >= 2 || (complex_hits >= 1 && simple_hits == 0 && medium_hits == 0) {
        return TaskComplexity::Complex;
    }

    // Medium with no simple counterweight
    if medium_hits >= 2 || (medium_hits >= 1 && simple_hits == 0) {
        return TaskComplexity::Medium;
    }

    // Simple wins when present
    if simple_hits >= 1 {
        return TaskComplexity::Simple;
    }

    TaskComplexity::Unknown
}

/// Classify a task based on known error categories from a previous attempt.
pub fn classify_from_errors(categories: &[ErrorCategory]) -> TaskComplexity {
    if categories.is_empty() {
        return TaskComplexity::Unknown;
    }

    let has_simple = categories.iter().any(|c| {
        matches!(
            c,
            ErrorCategory::TypeMismatch | ErrorCategory::ImportResolution | ErrorCategory::Syntax
        )
    });

    let has_medium = categories.iter().any(|c| {
        matches!(
            c,
            ErrorCategory::BorrowChecker | ErrorCategory::Lifetime | ErrorCategory::TraitBound
        )
    });

    let has_complex = categories.iter().any(|c| matches!(c, ErrorCategory::Async));

    // Multiple categories = likely complex
    let unique_cats: std::collections::HashSet<_> = categories.iter().collect();
    if unique_cats.len() >= 4 {
        return TaskComplexity::Complex;
    }

    if has_complex {
        return TaskComplexity::Complex;
    }

    if has_medium {
        return TaskComplexity::Medium;
    }

    if has_simple {
        return TaskComplexity::Simple;
    }

    TaskComplexity::Unknown
}

/// Determine the initial tier for a task.
///
/// This is the main entry point for worker-first routing. It combines
/// description analysis with any known error categories to produce a
/// tier recommendation.
pub fn classify_initial_tier(
    description: &str,
    known_errors: &[ErrorCategory],
) -> InitialTierRecommendation {
    let desc_complexity = classify_from_description(description);
    let error_complexity = classify_from_errors(known_errors);

    // Combine: take the more complex assessment
    let (complexity, source) = match (&desc_complexity, &error_complexity) {
        // Complex from either source → Council
        (TaskComplexity::Complex, _) => (TaskComplexity::Complex, "description"),
        (_, TaskComplexity::Complex) => (TaskComplexity::Complex, "error categories"),

        // Medium from either → Worker (medium tasks are worker-solvable)
        (TaskComplexity::Medium, _) => (TaskComplexity::Medium, "description"),
        (_, TaskComplexity::Medium) => (TaskComplexity::Medium, "error categories"),

        // Simple from either → Worker
        (TaskComplexity::Simple, _) => (TaskComplexity::Simple, "description"),
        (_, TaskComplexity::Simple) => (TaskComplexity::Simple, "error categories"),

        // Both unknown → Worker (worker-first default)
        (TaskComplexity::Unknown, TaskComplexity::Unknown) => {
            (TaskComplexity::Unknown, "default (worker-first)")
        }
    };

    let (tier, reason, confidence) = match complexity {
        TaskComplexity::Simple => (
            SwarmTier::Worker,
            format!(
                "Simple task detected from {} — direct worker routing",
                source
            ),
            0.85,
        ),
        TaskComplexity::Medium => (
            SwarmTier::Worker,
            format!(
                "Medium-complexity task from {} — worker with escalation readiness",
                source
            ),
            0.65,
        ),
        TaskComplexity::Complex => (
            SwarmTier::Council,
            format!(
                "Complex task detected from {} — requires manager coordination",
                source
            ),
            0.80,
        ),
        TaskComplexity::Unknown => (
            SwarmTier::Worker,
            format!(
                "Unknown complexity ({}) — worker-first default, will escalate if needed",
                source
            ),
            0.50,
        ),
    };

    InitialTierRecommendation {
        tier,
        complexity,
        reason,
        confidence,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_simple_from_description() {
        assert_eq!(
            classify_from_description("Fix unused import in lib.rs"),
            TaskComplexity::Simple,
        );
        assert_eq!(
            classify_from_description("Fix clippy warning about dead_code"),
            TaskComplexity::Simple,
        );
        assert_eq!(
            classify_from_description("Add missing field to struct"),
            TaskComplexity::Simple,
        );
    }

    #[test]
    fn test_classify_medium_from_description() {
        assert_eq!(
            classify_from_description("Fix borrow checker error with lifetime annotation"),
            TaskComplexity::Medium,
        );
        assert_eq!(
            classify_from_description("Add trait bound for generic parameter"),
            TaskComplexity::Medium,
        );
    }

    #[test]
    fn test_classify_complex_from_description() {
        assert_eq!(
            classify_from_description("Refactor the async orchestration layer to use tokio tasks"),
            TaskComplexity::Complex,
        );
        assert_eq!(
            classify_from_description("Architecture redesign for multi-file state machine"),
            TaskComplexity::Complex,
        );
    }

    #[test]
    fn test_classify_unknown_from_description() {
        assert_eq!(
            classify_from_description("Implement feature X"),
            TaskComplexity::Unknown,
        );
        assert_eq!(
            classify_from_description("Add per-agent performance tracking"),
            TaskComplexity::Unknown,
        );
    }

    #[test]
    fn test_classify_simple_from_errors() {
        assert_eq!(
            classify_from_errors(&[ErrorCategory::TypeMismatch]),
            TaskComplexity::Simple,
        );
        assert_eq!(
            classify_from_errors(&[ErrorCategory::ImportResolution]),
            TaskComplexity::Simple,
        );
    }

    #[test]
    fn test_classify_medium_from_errors() {
        assert_eq!(
            classify_from_errors(&[ErrorCategory::BorrowChecker]),
            TaskComplexity::Medium,
        );
        assert_eq!(
            classify_from_errors(&[ErrorCategory::Lifetime]),
            TaskComplexity::Medium,
        );
    }

    #[test]
    fn test_classify_complex_from_errors() {
        assert_eq!(
            classify_from_errors(&[ErrorCategory::Async]),
            TaskComplexity::Complex,
        );
        // Many unique error categories → complex
        assert_eq!(
            classify_from_errors(&[
                ErrorCategory::TypeMismatch,
                ErrorCategory::BorrowChecker,
                ErrorCategory::Lifetime,
                ErrorCategory::Async,
            ]),
            TaskComplexity::Complex,
        );
    }

    #[test]
    fn test_classify_empty_errors_is_unknown() {
        assert_eq!(classify_from_errors(&[]), TaskComplexity::Unknown);
    }

    #[test]
    fn test_initial_tier_simple_goes_to_worker() {
        let rec = classify_initial_tier("Fix unused import", &[]);
        assert_eq!(rec.tier, SwarmTier::Worker);
        assert_eq!(rec.complexity, TaskComplexity::Simple);
        assert!(rec.confidence > 0.7);
    }

    #[test]
    fn test_initial_tier_complex_goes_to_council() {
        let rec = classify_initial_tier(
            "Refactor the async orchestration layer",
            &[ErrorCategory::Async],
        );
        assert_eq!(rec.tier, SwarmTier::Council);
        assert_eq!(rec.complexity, TaskComplexity::Complex);
    }

    #[test]
    fn test_initial_tier_unknown_defaults_to_worker() {
        let rec = classify_initial_tier("Implement feature X", &[]);
        assert_eq!(rec.tier, SwarmTier::Worker);
        assert_eq!(rec.complexity, TaskComplexity::Unknown);
        assert!(rec.confidence <= 0.5);
    }

    #[test]
    fn test_initial_tier_errors_override_description() {
        // Description says simple, but errors indicate medium
        let rec = classify_initial_tier("Fix lint warning", &[ErrorCategory::BorrowChecker]);
        assert_eq!(rec.tier, SwarmTier::Worker);
        // Medium from errors overrides simple from description
        assert_eq!(rec.complexity, TaskComplexity::Medium);
    }

    #[test]
    fn test_medium_still_routes_to_worker() {
        let rec =
            classify_initial_tier("Fix borrow checker issue", &[ErrorCategory::BorrowChecker]);
        assert_eq!(rec.tier, SwarmTier::Worker);
        assert_eq!(rec.complexity, TaskComplexity::Medium);
    }

    #[test]
    fn test_display_complexity() {
        assert_eq!(TaskComplexity::Simple.to_string(), "simple");
        assert_eq!(TaskComplexity::Medium.to_string(), "medium");
        assert_eq!(TaskComplexity::Complex.to_string(), "complex");
        assert_eq!(TaskComplexity::Unknown.to_string(), "unknown");
    }

    #[test]
    fn test_recommendation_serialization() {
        let rec = classify_initial_tier("Fix type mismatch", &[ErrorCategory::TypeMismatch]);
        let json = serde_json::to_string(&rec).unwrap();
        let restored: InitialTierRecommendation = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.tier, rec.tier);
        assert_eq!(restored.complexity, rec.complexity);
    }
}
