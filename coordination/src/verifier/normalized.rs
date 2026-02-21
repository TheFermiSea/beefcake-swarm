//! Normalized, machine-readable verifier output.
//!
//! Compact enough for inclusion in work packets and reviewer prompts.
//! Approximately 500–2000 bytes serialized vs 5–50 KB for full [`VerifierReport`].

use crate::feedback::error_parser::ErrorCategory;
use crate::verifier::report::VerifierReport;
use serde::{Deserialize, Serialize};

/// Compact summary of a single gate result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSummary {
    pub gate: String,
    pub passed: bool,
    pub error_count: usize,
    pub warning_count: usize,
    pub duration_ms: u64,
}

/// Bucketed error counts by category.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBucket {
    pub category: ErrorCategory,
    pub count: usize,
    /// Representative error message (first occurrence).
    pub sample_message: String,
    /// Representative file (first occurrence).
    pub sample_file: Option<String>,
}

/// Normalized, machine-readable verifier output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedOutput {
    /// Overall pass/fail.
    pub all_green: bool,
    /// Number of gates passed / total gates.
    pub gates_passed: usize,
    pub gates_total: usize,
    /// Per-gate compact summaries.
    pub gates: Vec<GateSummary>,
    /// Bucketed error counts by category.
    pub error_buckets: Vec<ErrorBucket>,
    /// Total error count across all gates.
    pub total_errors: usize,
    /// Total warning count across all gates.
    pub total_warnings: usize,
    /// Dominant error category (most frequent), if any.
    pub dominant_category: Option<ErrorCategory>,
    /// Total pipeline duration in ms.
    pub duration_ms: u64,
    /// First failing gate name, if any.
    pub first_failure: Option<String>,
}

impl NormalizedOutput {
    /// Normalize a full [`VerifierReport`] into compact form.
    pub fn from_report(report: &VerifierReport) -> Self {
        let gates: Vec<GateSummary> = report
            .gates
            .iter()
            .map(|g| GateSummary {
                gate: g.gate.clone(),
                passed: g.outcome.is_passed(),
                error_count: g.error_count,
                warning_count: g.warning_count,
                duration_ms: g.duration_ms,
            })
            .collect();

        // Aggregate errors by category from report.error_categories
        let mut buckets: Vec<ErrorBucket> = report
            .error_categories
            .iter()
            .map(|(cat, count)| {
                // Find first error of this category for sample
                let sample = report
                    .gates
                    .iter()
                    .flat_map(|g| g.errors.iter())
                    .find(|e| e.category == *cat);
                ErrorBucket {
                    category: *cat,
                    count: *count,
                    sample_message: sample.map(|e| e.message.clone()).unwrap_or_default(),
                    sample_file: sample.and_then(|e| e.file.clone()),
                }
            })
            .collect();
        // Most frequent first
        buckets.sort_by(|a, b| b.count.cmp(&a.count));

        let total_errors: usize = gates.iter().map(|g| g.error_count).sum();
        let total_warnings: usize = gates.iter().map(|g| g.warning_count).sum();
        let dominant_category = buckets.first().map(|b| b.category);
        let first_failure = gates.iter().find(|g| !g.passed).map(|g| g.gate.clone());

        NormalizedOutput {
            all_green: report.all_green,
            gates_passed: report.gates_passed,
            gates_total: report.gates_total,
            gates,
            error_buckets: buckets,
            total_errors,
            total_warnings,
            dominant_category,
            duration_ms: report.total_duration_ms,
            first_failure,
        }
    }

    /// Compact text summary for inclusion in prompts.
    ///
    /// Example: `[FAIL] 2/4 gates | 5 errors (borrow_checker:3, lifetime:2) | first_fail=check | 1200ms`
    pub fn compact_text(&self) -> String {
        let status = if self.all_green { "GREEN" } else { "FAIL" };
        let mut parts = vec![format!(
            "[{}] {}/{} gates",
            status, self.gates_passed, self.gates_total
        )];
        if self.total_errors > 0 {
            let cats: Vec<String> = self
                .error_buckets
                .iter()
                .take(3) // Top 3 categories
                .map(|b| format!("{}:{}", b.category, b.count))
                .collect();
            parts.push(format!(
                "{} errors ({})",
                self.total_errors,
                cats.join(", ")
            ));
        }
        if let Some(ref fail) = self.first_failure {
            parts.push(format!("first_fail={}", fail));
        }
        parts.push(format!("{}ms", self.duration_ms));
        parts.join(" | ")
    }

    /// JSON string for machine consumption.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self)
            .unwrap_or_else(|_| r#"{"error":"serialization failed"}"#.to_string())
    }

    /// Whether there are errors in a specific category.
    pub fn has_category(&self, cat: ErrorCategory) -> bool {
        self.error_buckets.iter().any(|b| b.category == cat)
    }

    /// Count of errors in a specific category.
    pub fn category_count(&self, cat: ErrorCategory) -> usize {
        self.error_buckets
            .iter()
            .find(|b| b.category == cat)
            .map(|b| b.count)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::error_parser::ParsedError;
    use crate::verifier::report::{GateOutcome, GateResult};
    use std::time::Duration;

    fn green_gate(name: &str) -> GateResult {
        GateResult {
            gate: name.to_string(),
            outcome: GateOutcome::Passed,
            duration_ms: 100,
            exit_code: Some(0),
            error_count: 0,
            warning_count: 0,
            errors: vec![],
            stderr_excerpt: None,
        }
    }

    fn failing_gate(name: &str, errors: Vec<ParsedError>) -> GateResult {
        let error_count = errors.len();
        GateResult {
            gate: name.to_string(),
            outcome: GateOutcome::Failed,
            duration_ms: 2000,
            exit_code: Some(1),
            error_count,
            warning_count: 0,
            errors,
            stderr_excerpt: None,
        }
    }

    fn parsed_error(cat: ErrorCategory, code: &str, msg: &str, file: &str) -> ParsedError {
        ParsedError {
            category: cat,
            code: Some(code.to_string()),
            message: msg.to_string(),
            file: Some(file.to_string()),
            line: Some(42),
            column: Some(10),
            suggestion: None,
            rendered: String::new(),
            labels: vec![],
        }
    }

    fn all_green_report() -> VerifierReport {
        let mut report = VerifierReport::new("/tmp/test".to_string());
        for gate in &["fmt", "clippy", "check", "test"] {
            report.add_gate(green_gate(gate));
        }
        report.finalize(Duration::from_millis(400));
        report
    }

    fn failing_report() -> VerifierReport {
        let mut report = VerifierReport::new("/tmp/test".to_string());
        report.add_gate(green_gate("fmt"));
        report.add_gate(failing_gate(
            "check",
            vec![
                parsed_error(
                    ErrorCategory::BorrowChecker,
                    "E0502",
                    "cannot borrow as mutable",
                    "src/lib.rs",
                ),
                parsed_error(
                    ErrorCategory::BorrowChecker,
                    "E0505",
                    "moved value",
                    "src/lib.rs",
                ),
                parsed_error(
                    ErrorCategory::Lifetime,
                    "E0106",
                    "missing lifetime",
                    "src/main.rs",
                ),
            ],
        ));
        report.finalize(Duration::from_millis(2100));
        report
    }

    #[test]
    fn test_normalize_all_green_report() {
        let norm = NormalizedOutput::from_report(&all_green_report());
        assert!(norm.all_green);
        assert_eq!(norm.total_errors, 0);
        assert_eq!(norm.gates_passed, 4);
        assert_eq!(norm.gates_total, 4);
        assert!(norm.error_buckets.is_empty());
        assert!(norm.first_failure.is_none());
        assert!(norm.dominant_category.is_none());
    }

    #[test]
    fn test_normalize_failing_report() {
        let norm = NormalizedOutput::from_report(&failing_report());
        assert!(!norm.all_green);
        assert_eq!(norm.gates_passed, 1);
        assert_eq!(norm.gates_total, 2);
        assert_eq!(norm.total_errors, 3);
        assert_eq!(norm.error_buckets.len(), 2);
        // Buckets sorted by count descending: BorrowChecker(2) > Lifetime(1)
        assert_eq!(norm.error_buckets[0].category, ErrorCategory::BorrowChecker);
        assert_eq!(norm.error_buckets[0].count, 2);
        assert_eq!(norm.error_buckets[1].category, ErrorCategory::Lifetime);
        assert_eq!(norm.error_buckets[1].count, 1);
    }

    #[test]
    fn test_compact_text_green() {
        let norm = NormalizedOutput::from_report(&all_green_report());
        let text = norm.compact_text();
        assert!(text.contains("[GREEN]"));
        assert!(text.contains("4/4 gates"));
        assert!(!text.contains("errors"));
    }

    #[test]
    fn test_compact_text_fail() {
        let norm = NormalizedOutput::from_report(&failing_report());
        let text = norm.compact_text();
        assert!(text.contains("[FAIL]"));
        assert!(text.contains("3 errors"));
        assert!(text.contains("first_fail=check"));
    }

    #[test]
    fn test_error_buckets_sorted() {
        let norm = NormalizedOutput::from_report(&failing_report());
        for w in norm.error_buckets.windows(2) {
            assert!(w[0].count >= w[1].count, "buckets not sorted descending");
        }
    }

    #[test]
    fn test_dominant_category() {
        let norm = NormalizedOutput::from_report(&failing_report());
        assert_eq!(norm.dominant_category, Some(ErrorCategory::BorrowChecker));
    }

    #[test]
    fn test_has_category() {
        let norm = NormalizedOutput::from_report(&failing_report());
        assert!(norm.has_category(ErrorCategory::BorrowChecker));
        assert!(norm.has_category(ErrorCategory::Lifetime));
        assert!(!norm.has_category(ErrorCategory::Async));
    }

    #[test]
    fn test_to_json_roundtrip() {
        let norm = NormalizedOutput::from_report(&failing_report());
        let json = norm.to_json();
        let parsed: NormalizedOutput = serde_json::from_str(&json).expect("roundtrip failed");
        assert_eq!(parsed.all_green, norm.all_green);
        assert_eq!(parsed.total_errors, norm.total_errors);
        assert_eq!(parsed.gates_passed, norm.gates_passed);
        assert_eq!(parsed.error_buckets.len(), norm.error_buckets.len());
    }

    #[test]
    fn test_first_failure_identification() {
        let mut report = VerifierReport::new("/tmp/test".to_string());
        report.add_gate(green_gate("fmt"));
        report.add_gate(failing_gate(
            "check",
            vec![parsed_error(
                ErrorCategory::TypeMismatch,
                "E0308",
                "mismatched types",
                "src/main.rs",
            )],
        ));
        report.finalize(Duration::from_millis(500));

        let norm = NormalizedOutput::from_report(&report);
        assert_eq!(norm.first_failure, Some("check".to_string()));
    }

    #[test]
    fn test_category_count() {
        let norm = NormalizedOutput::from_report(&failing_report());
        assert_eq!(norm.category_count(ErrorCategory::BorrowChecker), 2);
        assert_eq!(norm.category_count(ErrorCategory::Lifetime), 1);
        assert_eq!(norm.category_count(ErrorCategory::Async), 0);
    }
}
