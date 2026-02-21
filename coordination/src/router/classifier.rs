//! Pre-routing classifier-extractor for complexity and risk analysis.
//!
//! Runs BEFORE the router makes a model tier selection.

use serde::{Deserialize, Serialize};

// Re-import types available in the lib crate.
// NOTE: The binary target (main.rs) does not declare all modules from lib.rs,
// so we define lightweight local types to keep this module self-contained.

/// Error category for classification (mirrors feedback::error_parser::ErrorCategory).
pub use crate::feedback::error_parser::ErrorCategory;
pub use crate::router::task_classifier::ModelTier;

/// A failure signal from verification (self-contained version).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignal {
    pub gate: String,
    pub category: ErrorCategory,
    pub code: Option<String>,
    pub file: Option<String>,
    pub line: Option<u32>,
    pub message: String,
}

/// A constraint on the task (self-contained version).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    pub kind: ConstraintKind,
    pub description: String,
}

/// Kinds of constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    NoBreakingApi,
    NoDeps,
    MaxPatchLoc,
    Other,
}

/// Lightweight work packet for pre-routing analysis (self-contained version).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkPacket {
    pub objective: String,
    pub files_touched: Vec<String>,
    pub constraints: Vec<Constraint>,
    pub iteration: u32,
    pub failure_signals: Vec<FailureSignal>,
}

/// Risk severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Kind of risk factor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskKind {
    BreakingApiChange,
    UnsafeCode,
    SecuritySensitive,
    DataLoss,
    ConcurrencyHazard,
    ExternalDependency,
    LargeScope,
    RepeatedFailures,
}

/// A single identified risk factor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    pub kind: RiskKind,
    pub description: String,
    pub severity: RiskLevel,
}

/// Extracted complexity factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityFactors {
    pub score: u8,
    pub file_count: usize,
    pub error_count: usize,
    pub has_lifetime_errors: bool,
    pub has_async_errors: bool,
    pub has_trait_errors: bool,
    pub iteration_depth: u32,
    pub dominant_categories: Vec<ErrorCategory>,
}

/// Full pre-routing analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreRoutingAnalysis {
    pub complexity: ComplexityFactors,
    pub risk_level: RiskLevel,
    pub risk_factors: Vec<RiskFactor>,
    pub recommended_tier: ModelTier,
    pub rationale: String,
}

impl PreRoutingAnalysis {
    /// Whether this task should be escalated immediately
    pub fn should_escalate(&self) -> bool {
        matches!(self.recommended_tier, ModelTier::Council)
    }

    /// Get a compact summary for logging
    pub fn summary(&self) -> String {
        format!(
            "complexity={}/5 risk={:?} tier={} factors={}",
            self.complexity.score,
            self.risk_level,
            self.recommended_tier,
            self.risk_factors.len()
        )
    }
}

/// Pre-routing classifier that analyzes tasks before model tier selection
pub struct PreRoutingClassifier;

impl PreRoutingClassifier {
    pub fn new() -> Self {
        Self
    }

    /// Analyze a task from its components
    pub fn analyze_task(
        &self,
        description: &str,
        files_touched: &[String],
        constraints: &[Constraint],
        iteration: u32,
        failure_signals: &[FailureSignal],
    ) -> PreRoutingAnalysis {
        let desc = description.to_lowercase();
        let has_cat = |cat: ErrorCategory| failure_signals.iter().any(|s| s.category == cat);
        let has_lifetime = has_cat(ErrorCategory::Lifetime);
        let has_async = has_cat(ErrorCategory::Async);
        let has_trait = has_cat(ErrorCategory::TraitBound);

        // ── Complexity scoring ────────────────────────────────────────────────
        let mut score: u8 = 1;
        if files_touched.len() > 3 {
            score += 1;
        }
        if files_touched.len() > 8 {
            score += 1;
        }
        if has_lifetime || has_async {
            score += 1;
        }
        if has_trait {
            score += 1;
        }
        if iteration > 2 {
            score += 1;
        }
        if [
            "lifetime",
            "async",
            "trait",
            "generic",
            "unsafe",
            "macro",
            "concurrency",
        ]
        .iter()
        .any(|k| desc.contains(k))
        {
            score += 1;
        }
        let score = score.min(5);

        let mut cats: Vec<ErrorCategory> = failure_signals
            .iter()
            .map(|s| s.category)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        cats.sort_by_key(|c| c.to_string());

        let complexity = ComplexityFactors {
            score,
            file_count: files_touched.len(),
            error_count: failure_signals.len(),
            has_lifetime_errors: has_lifetime,
            has_async_errors: has_async,
            has_trait_errors: has_trait,
            iteration_depth: iteration,
            dominant_categories: cats,
        };

        // ── Risk detection ────────────────────────────────────────────────────
        let has_no_breaking = constraints
            .iter()
            .any(|c| c.kind == ConstraintKind::NoBreakingApi);
        let has_no_deps = constraints.iter().any(|c| c.kind == ConstraintKind::NoDeps);
        let contains = |kws: &[&str]| kws.iter().any(|k| desc.contains(k));

        let candidates: &[(bool, RiskKind, &str, RiskLevel)] = &[
            (
                has_no_breaking || contains(&["pub ", "public api", "breaking"]),
                RiskKind::BreakingApiChange,
                "Potential breaking API change",
                RiskLevel::High,
            ),
            (
                desc.contains("unsafe"),
                RiskKind::UnsafeCode,
                "Unsafe code present",
                RiskLevel::High,
            ),
            (
                contains(&["auth", "password", "token", "secret", "crypto", "encrypt"]),
                RiskKind::SecuritySensitive,
                "Security-sensitive operation",
                RiskLevel::Critical,
            ),
            (
                contains(&["delete", "drop", "truncate", "remove all"]),
                RiskKind::DataLoss,
                "Potential data loss",
                RiskLevel::High,
            ),
            (
                has_async || contains(&["mutex", "lock", "arc", "concurrent", "parallel"]),
                RiskKind::ConcurrencyHazard,
                "Concurrency hazard detected",
                RiskLevel::High,
            ),
            (
                has_no_deps && contains(&["add", "depend", "crate", "library"]),
                RiskKind::ExternalDependency,
                "External dependency conflicts NoDeps constraint",
                RiskLevel::Medium,
            ),
            (
                files_touched.len() > 8,
                RiskKind::LargeScope,
                "Large scope: many files touched",
                RiskLevel::Medium,
            ),
            (
                iteration > 3,
                RiskKind::RepeatedFailures,
                "Repeated failures indicate stuck pattern",
                RiskLevel::High,
            ),
        ];

        let risk_factors: Vec<RiskFactor> = candidates
            .iter()
            .filter(|(cond, ..)| *cond)
            .map(|(_, kind, desc, sev)| RiskFactor {
                kind: *kind,
                description: (*desc).to_string(),
                severity: *sev,
            })
            .collect();

        let risk_level = risk_factors
            .iter()
            .map(|r| r.severity)
            .max()
            .unwrap_or(RiskLevel::Low);

        // ── Tier recommendation ───────────────────────────────────────────────
        let recommended_tier = if matches!(risk_level, RiskLevel::Critical | RiskLevel::High)
            || score >= 4
            || (score >= 3 && risk_level >= RiskLevel::Medium)
        {
            ModelTier::Council
        } else {
            ModelTier::Worker
        };

        let mut parts = vec![format!("complexity={}/5", score)];
        if !risk_factors.is_empty() {
            parts.push(format!("risk={:?}({})", risk_level, risk_factors.len()));
        }
        if iteration > 2 {
            parts.push(format!("iteration={}", iteration));
        }
        parts.push(format!("→ {}", recommended_tier));

        PreRoutingAnalysis {
            complexity,
            risk_level,
            risk_factors,
            recommended_tier,
            rationale: parts.join(", "),
        }
    }

    /// Analyze from a WorkPacket directly
    pub fn analyze_packet(&self, packet: &WorkPacket) -> PreRoutingAnalysis {
        self.analyze_task(
            &packet.objective,
            &packet.files_touched,
            &packet.constraints,
            packet.iteration,
            &packet.failure_signals,
        )
    }
}

impl Default for PreRoutingClassifier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sig(category: ErrorCategory) -> FailureSignal {
        FailureSignal {
            gate: "check".into(),
            category,
            code: None,
            file: None,
            line: None,
            message: "e".into(),
        }
    }

    #[test]
    fn test_low_complexity_low_risk_routes_to_worker() {
        let a = PreRoutingClassifier::new().analyze_task(
            "add a helper function",
            &["src/lib.rs".into()],
            &[],
            0,
            &[],
        );
        assert_eq!(a.recommended_tier, ModelTier::Worker);
        assert!(!a.should_escalate());
        assert_eq!(a.complexity.score, 1);
    }

    #[test]
    fn test_high_complexity_routes_to_council() {
        let files: Vec<String> = (0..10).map(|i| format!("src/f{}.rs", i)).collect();
        let a = PreRoutingClassifier::new().analyze_task(
            "fix errors",
            &files,
            &[],
            0,
            &[sig(ErrorCategory::Lifetime)],
        );
        assert_eq!(a.recommended_tier, ModelTier::Council);
        assert!(a.complexity.score >= 4);
    }

    #[test]
    fn test_unsafe_code_routes_to_council() {
        let a =
            PreRoutingClassifier::new().analyze_task("add unsafe block for FFI", &[], &[], 0, &[]);
        assert_eq!(a.recommended_tier, ModelTier::Council);
        assert!(a
            .risk_factors
            .iter()
            .any(|r| r.kind == RiskKind::UnsafeCode));
    }

    #[test]
    fn test_repeated_failures_routes_to_council() {
        let a = PreRoutingClassifier::new().analyze_task("fix compilation", &[], &[], 5, &[]);
        assert_eq!(a.recommended_tier, ModelTier::Council);
        assert!(a
            .risk_factors
            .iter()
            .any(|r| r.kind == RiskKind::RepeatedFailures));
    }

    #[test]
    fn test_analyze_packet_works() {
        let packet = WorkPacket {
            objective: "simple refactor".into(),
            files_touched: vec!["src/lib.rs".into()],
            failure_signals: vec![],
            constraints: vec![],
            iteration: 1,
        };
        let a = PreRoutingClassifier::new().analyze_packet(&packet);
        assert_eq!(a.recommended_tier, ModelTier::Worker);
        assert!(!a.summary().is_empty());
    }
}
