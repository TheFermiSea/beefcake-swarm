//! Verifier Report — Structured output from the verification pipeline
//!
//! Contains classified errors, gate results, and failure signals
//! suitable for consumption by the Escalation Engine.

use crate::feedback::error_parser::{ErrorCategory, ErrorSummary, ParsedError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Outcome of a single verification gate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GateOutcome {
    /// Gate passed successfully
    Passed,
    /// Gate failed with errors
    Failed,
    /// Gate was skipped (previous gate failed and pipeline is sequential)
    Skipped,
    /// Gate produced warnings but did not block the pipeline
    Warning,
}

impl GateOutcome {
    pub fn is_passed(&self) -> bool {
        matches!(self, Self::Passed | Self::Warning)
    }
}

impl std::fmt::Display for GateOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Passed => write!(f, "PASS"),
            Self::Failed => write!(f, "FAIL"),
            Self::Skipped => write!(f, "SKIP"),
            Self::Warning => write!(f, "WARN"),
        }
    }
}

/// Result of a single verification gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// Gate name (fmt, clippy, check, test)
    pub gate: String,
    /// Whether the gate passed, failed, or was skipped
    pub outcome: GateOutcome,
    /// Duration of this gate
    pub duration_ms: u64,
    /// Exit code from the command
    pub exit_code: Option<i32>,
    /// Number of errors found
    pub error_count: usize,
    /// Number of warnings found
    pub warning_count: usize,
    /// Parsed errors (only for check/clippy gates)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<ParsedError>,
    /// Raw stderr output (truncated to 4KB for context efficiency)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stderr_excerpt: Option<String>,
}

/// Failure signal for the Escalation Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailureSignal {
    /// Which gate produced this signal
    pub gate: String,
    /// Error category classification
    pub category: ErrorCategory,
    /// Rustc error code (e.g., "E0106")
    pub code: Option<String>,
    /// File where the error occurred
    pub file: Option<String>,
    /// Line number
    pub line: Option<usize>,
    /// Brief error message
    pub message: String,
}

impl FailureSignal {
    /// Create from a ParsedError
    pub fn from_parsed_error(gate: &str, error: &ParsedError) -> Self {
        Self {
            gate: gate.to_string(),
            category: error.category,
            code: error.code.clone(),
            file: error.file.clone(),
            line: error.line,
            message: error.message.clone(),
        }
    }
}

/// Classification of a validator-reported issue (TextGrad pattern).
///
/// Used to structure reviewer/validator feedback into actionable deltas
/// rather than unstructured prose, enabling tighter feedback loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValidatorIssueType {
    /// Logic error in the implementation
    LogicError,
    /// Missing safety or error handling check
    MissingSafetyCheck,
    /// Unhandled edge case or boundary condition
    UnhandledEdgeCase,
    /// Style or idiom violation
    StyleViolation,
    /// Incorrect behavior vs specification
    IncorrectBehavior,
    /// Uncategorized issue
    Other,
}

impl std::fmt::Display for ValidatorIssueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LogicError => write!(f, "logic_error"),
            Self::MissingSafetyCheck => write!(f, "missing_safety_check"),
            Self::UnhandledEdgeCase => write!(f, "unhandled_edge_case"),
            Self::StyleViolation => write!(f, "style_violation"),
            Self::IncorrectBehavior => write!(f, "incorrect_behavior"),
            Self::Other => write!(f, "other"),
        }
    }
}

/// Structured validator feedback entry (TextGrad pattern).
///
/// Converts subjective reviewer prose into actionable code locations + fixes.
/// Fed back into the next iteration's WorkPacket so the implementer gets
/// specific, targeted guidance instead of unstructured critique.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatorFeedback {
    /// File where the issue was found (from diff or reviewer annotation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file: Option<String>,
    /// Line range (start, end) if identifiable
    #[serde(skip_serializing_if = "Option::is_none")]
    pub line_range: Option<(usize, usize)>,
    /// Classification of the issue
    pub issue_type: ValidatorIssueType,
    /// Description of the problem
    pub description: String,
    /// Suggested fix if available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggested_fix: Option<String>,
    /// Which model produced this feedback
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_model: Option<String>,
}

/// Complete verification report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierReport {
    /// Timestamp of the verification run
    pub timestamp: DateTime<Utc>,
    /// Total pipeline duration
    pub total_duration_ms: u64,
    /// Number of gates that passed
    pub gates_passed: usize,
    /// Total number of gates attempted
    pub gates_total: usize,
    /// Whether ALL gates passed (clean build)
    pub all_green: bool,
    /// Individual gate results
    pub gates: Vec<GateResult>,
    /// Classified failure signals for the Escalation Engine
    pub failure_signals: Vec<FailureSignal>,
    /// Error summary across all gates
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_summary: Option<ErrorSummary>,
    /// Error categories present, with counts
    pub error_categories: HashMap<ErrorCategory, usize>,
    /// First failing gate (for quick triage)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_failure: Option<String>,
    /// Working directory that was verified
    pub working_dir: String,
    /// Git branch (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// Git commit SHA (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Pre-gate safety scan warnings (dangerous patterns in agent diff)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safety_warnings: Vec<super::safety_scan::SafetyWarning>,
}

impl VerifierReport {
    /// Create a new empty report
    pub fn new(working_dir: String) -> Self {
        Self {
            timestamp: Utc::now(),
            total_duration_ms: 0,
            gates_passed: 0,
            gates_total: 0,
            all_green: false,
            gates: Vec::new(),
            failure_signals: Vec::new(),
            error_summary: None,
            error_categories: HashMap::new(),
            first_failure: None,
            working_dir,
            branch: None,
            commit: None,
            safety_warnings: Vec::new(),
        }
    }

    /// Add a gate result to the report
    pub fn add_gate(&mut self, result: GateResult) {
        if result.outcome.is_passed() {
            self.gates_passed += 1;
        } else if result.outcome == GateOutcome::Failed && self.first_failure.is_none() {
            self.first_failure = Some(result.gate.clone());
        }

        // Aggregate error categories from parsed errors
        for error in &result.errors {
            *self.error_categories.entry(error.category).or_insert(0) += 1;
        }

        // Generate failure signals
        for error in &result.errors {
            self.failure_signals
                .push(FailureSignal::from_parsed_error(&result.gate, error));
        }

        self.gates_total += 1;
        self.gates.push(result);
    }

    /// Finalize the report after all gates have run
    pub fn finalize(&mut self, total_duration: Duration) {
        self.total_duration_ms = total_duration.as_millis() as u64;
        self.all_green = self.gates_passed == self.gates_total && self.gates_total > 0;

        // Build error summary from all failure signals
        if !self.failure_signals.is_empty() {
            let parsed_errors: Vec<ParsedError> =
                self.gates.iter().flat_map(|g| g.errors.clone()).collect();

            if !parsed_errors.is_empty() {
                self.error_summary = Some(
                    crate::feedback::error_parser::RustcErrorParser::summarize(&parsed_errors),
                );
            }
        }
    }

    /// Get a compact summary for logging
    pub fn summary(&self) -> String {
        let gate_statuses: Vec<String> = self
            .gates
            .iter()
            .map(|g| format!("{}:{}", g.gate, g.outcome))
            .collect();

        format!(
            "[{}] {}/{} gates passed ({}ms) [{}]",
            if self.all_green { "GREEN" } else { "RED" },
            self.gates_passed,
            self.gates_total,
            self.total_duration_ms,
            gate_statuses.join(" → "),
        )
    }

    /// Get unique error categories present
    pub fn unique_error_categories(&self) -> Vec<ErrorCategory> {
        let mut cats: Vec<ErrorCategory> = self.error_categories.keys().copied().collect();
        cats.sort_by_key(|c| c.to_string());
        cats
    }

    /// Check if a specific error category is present
    pub fn has_error_category(&self, category: ErrorCategory) -> bool {
        self.error_categories.contains_key(&category)
    }

    /// Get the dominant error category (most frequent)
    pub fn dominant_error_category(&self) -> Option<ErrorCategory> {
        self.error_categories
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(cat, _)| *cat)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gate_outcome_display() {
        assert_eq!(format!("{}", GateOutcome::Passed), "PASS");
        assert_eq!(format!("{}", GateOutcome::Failed), "FAIL");
        assert_eq!(format!("{}", GateOutcome::Skipped), "SKIP");
        assert_eq!(format!("{}", GateOutcome::Warning), "WARN");
    }

    #[test]
    fn test_gate_outcome_warning_counts_as_passed() {
        assert!(GateOutcome::Passed.is_passed());
        assert!(GateOutcome::Warning.is_passed());
        assert!(!GateOutcome::Failed.is_passed());
        assert!(!GateOutcome::Skipped.is_passed());
    }

    #[test]
    fn test_report_warning_gate_does_not_block_all_green() {
        let mut report = VerifierReport::new("/tmp/test".to_string());
        report.add_gate(GateResult {
            gate: "fmt".to_string(),
            outcome: GateOutcome::Passed,
            duration_ms: 100,
            exit_code: Some(0),
            error_count: 0,
            warning_count: 0,
            errors: vec![],
            stderr_excerpt: None,
        });
        report.add_gate(GateResult {
            gate: "sg".to_string(),
            outcome: GateOutcome::Warning,
            duration_ms: 50,
            exit_code: Some(1),
            error_count: 0,
            warning_count: 3,
            errors: vec![],
            stderr_excerpt: Some("3 diagnostics".to_string()),
        });
        report.finalize(Duration::from_millis(150));
        assert!(report.all_green, "Warning gate should not block all_green");
        assert_eq!(report.gates_passed, 2);
    }

    #[test]
    fn test_report_add_gate() {
        let mut report = VerifierReport::new("/tmp/test".to_string());

        report.add_gate(GateResult {
            gate: "fmt".to_string(),
            outcome: GateOutcome::Passed,
            duration_ms: 100,
            exit_code: Some(0),
            error_count: 0,
            warning_count: 0,
            errors: vec![],
            stderr_excerpt: None,
        });

        report.add_gate(GateResult {
            gate: "check".to_string(),
            outcome: GateOutcome::Failed,
            duration_ms: 2000,
            exit_code: Some(1),
            error_count: 1,
            warning_count: 0,
            errors: vec![ParsedError {
                category: ErrorCategory::Lifetime,
                code: Some("E0106".to_string()),
                message: "missing lifetime specifier".to_string(),
                file: Some("src/main.rs".to_string()),
                line: Some(42),
                column: Some(10),
                suggestion: None,
                rendered: "error[E0106]: missing lifetime specifier".to_string(),
                labels: vec![],
            }],
            stderr_excerpt: None,
        });

        assert_eq!(report.gates_passed, 1);
        assert_eq!(report.gates_total, 2);
        assert_eq!(report.first_failure, Some("check".to_string()));
        assert!(report.has_error_category(ErrorCategory::Lifetime));
        assert_eq!(
            report.dominant_error_category(),
            Some(ErrorCategory::Lifetime)
        );
    }

    #[test]
    fn test_report_all_green() {
        let mut report = VerifierReport::new("/tmp/test".to_string());

        for gate in &["fmt", "clippy", "check", "test"] {
            report.add_gate(GateResult {
                gate: gate.to_string(),
                outcome: GateOutcome::Passed,
                duration_ms: 100,
                exit_code: Some(0),
                error_count: 0,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: None,
            });
        }

        report.finalize(Duration::from_millis(400));
        assert!(report.all_green);
        assert_eq!(report.gates_passed, 4);
        assert!(report.summary().contains("GREEN"));
    }

    #[test]
    fn test_failure_signal_from_parsed_error() {
        let error = ParsedError {
            category: ErrorCategory::BorrowChecker,
            code: Some("E0502".to_string()),
            message: "cannot borrow as mutable".to_string(),
            file: Some("src/lib.rs".to_string()),
            line: Some(15),
            column: Some(5),
            suggestion: None,
            rendered: String::new(),
            labels: vec![],
        };

        let signal = FailureSignal::from_parsed_error("check", &error);
        assert_eq!(signal.gate, "check");
        assert_eq!(signal.category, ErrorCategory::BorrowChecker);
        assert_eq!(signal.code, Some("E0502".to_string()));
    }
}
