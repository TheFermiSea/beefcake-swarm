//! Rustc error classification and parsing
//!
//! Extracts structured information from compiler errors to guide model selection
//! and prompt construction.

use crate::feedback::compiler::{CargoMessage, DiagnosticMessage};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::sync::LazyLock;

/// Compiled regex patterns for error classification
static BORROW_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(borrow|cannot move|already borrowed|mutable|immutable)").unwrap()
});

static LIFETIME_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(lifetime|'[a-z]+|does not live long enough|outlive)").unwrap()
});

static TYPE_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(type|mismatched types|expected .*, found|E0308)").unwrap());

static TRAIT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(trait|impl|bound|does not implement|E0277|E0599)").unwrap());

static ASYNC_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(async|await|future|pin|send|sync)").unwrap());

static MACRO_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)(macro|procedural|derive|#\[)").unwrap());

static IMPORT_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(unresolved|cannot find|not found|use of undeclared|E0432|E0433)").unwrap()
});

/// Error categories for routing to appropriate models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCategory {
    /// Type mismatch errors (E0308, etc.)
    TypeMismatch,
    /// Borrow checker errors (E0382, E0502, etc.)
    BorrowChecker,
    /// Lifetime errors (E0106, E0621, etc.)
    Lifetime,
    /// Trait bound errors (E0277, etc.)
    TraitBound,
    /// Async/await related errors
    Async,
    /// Macro expansion errors
    Macro,
    /// Import/module resolution errors
    ImportResolution,
    /// Syntax errors
    Syntax,
    /// Other/unknown errors
    Other,
}

impl ErrorCategory {
    /// Determine complexity level (1-3, higher = more complex)
    pub fn complexity(&self) -> u8 {
        match self {
            Self::TypeMismatch => 1,
            Self::ImportResolution => 1,
            Self::Syntax => 1,
            Self::BorrowChecker => 2,
            Self::TraitBound => 2,
            Self::Lifetime => 3,
            Self::Async => 3,
            Self::Macro => 3,
            Self::Other => 2,
        }
    }

    /// Get recommended model tier for this error category
    /// Returns: "fast" (Strand), "specialized" (Hydra), "reasoning" (OR1)
    pub fn recommended_tier(&self) -> &'static str {
        match self {
            Self::TypeMismatch | Self::ImportResolution | Self::Syntax => "fast",
            Self::BorrowChecker | Self::TraitBound => "specialized",
            Self::Lifetime | Self::Async | Self::Macro | Self::Other => "specialized",
        }
    }
}

impl std::fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TypeMismatch => write!(f, "type_mismatch"),
            Self::BorrowChecker => write!(f, "borrow_checker"),
            Self::Lifetime => write!(f, "lifetime"),
            Self::TraitBound => write!(f, "trait_bound"),
            Self::Async => write!(f, "async"),
            Self::Macro => write!(f, "macro"),
            Self::ImportResolution => write!(f, "import"),
            Self::Syntax => write!(f, "syntax"),
            Self::Other => write!(f, "other"),
        }
    }
}

/// Parsed and classified error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedError {
    /// Error category
    pub category: ErrorCategory,
    /// Error code (e.g., "E0308")
    pub code: Option<String>,
    /// Error message
    pub message: String,
    /// File location
    pub file: Option<String>,
    /// Line number
    pub line: Option<usize>,
    /// Column number
    pub column: Option<usize>,
    /// Suggested fix from compiler
    pub suggestion: Option<String>,
    /// The rendered error for LLM consumption
    pub rendered: String,
    /// Labels from spans
    pub labels: Vec<String>,
}

impl ParsedError {
    /// Create from a diagnostic message
    pub fn from_diagnostic(diag: &DiagnosticMessage) -> Self {
        let category = Self::classify_error(diag);
        let code = diag.error_code().map(String::from);

        let (file, line, column) = diag
            .primary_span()
            .map(|s| {
                (
                    Some(s.file_name.clone()),
                    Some(s.line_start),
                    Some(s.column_start),
                )
            })
            .unwrap_or((None, None, None));

        let suggestion = diag.suggested_replacement().map(String::from);

        let labels: Vec<String> = diag.spans.iter().filter_map(|s| s.label.clone()).collect();

        let rendered = diag.format();

        Self {
            category,
            code,
            message: diag.message.clone(),
            file,
            line,
            column,
            suggestion,
            rendered,
            labels,
        }
    }

    /// Classify error based on message content and error code
    fn classify_error(diag: &DiagnosticMessage) -> ErrorCategory {
        let text = format!(
            "{} {} {}",
            diag.message,
            diag.code.as_ref().map(|c| c.code.as_str()).unwrap_or(""),
            diag.spans
                .iter()
                .filter_map(|s| s.label.as_ref())
                .cloned()
                .collect::<Vec<_>>()
                .join(" ")
        );

        // Check error codes first (most reliable)
        if let Some(code) = diag.error_code() {
            match code {
                // Type errors
                "E0308" | "E0271" | "E0369" | "E0277" => {}
                // Borrow checker
                "E0382" | "E0502" | "E0503" | "E0505" | "E0507" => {
                    return ErrorCategory::BorrowChecker
                }
                // Lifetimes
                "E0106" | "E0621" | "E0700" | "E0495" => return ErrorCategory::Lifetime,
                // Traits
                "E0599" => return ErrorCategory::TraitBound,
                // Imports
                "E0432" | "E0433" | "E0412" => return ErrorCategory::ImportResolution,
                _ => {}
            }
        }

        // Pattern-based classification
        if LIFETIME_PATTERN.is_match(&text) {
            return ErrorCategory::Lifetime;
        }
        if BORROW_PATTERN.is_match(&text) {
            return ErrorCategory::BorrowChecker;
        }
        if ASYNC_PATTERN.is_match(&text) {
            return ErrorCategory::Async;
        }
        if MACRO_PATTERN.is_match(&text) {
            return ErrorCategory::Macro;
        }
        if TRAIT_PATTERN.is_match(&text) {
            return ErrorCategory::TraitBound;
        }
        if TYPE_PATTERN.is_match(&text) {
            return ErrorCategory::TypeMismatch;
        }
        if IMPORT_PATTERN.is_match(&text) {
            return ErrorCategory::ImportResolution;
        }

        ErrorCategory::Other
    }

    /// Format for LLM fix prompt
    pub fn format_for_fix_prompt(&self) -> String {
        let mut result = format!("Error: {}\n", self.message);

        if let Some(code) = &self.code {
            result.push_str(&format!("Code: {}\n", code));
        }

        if let Some(file) = &self.file {
            result.push_str(&format!(
                "Location: {}:{}:{}\n",
                file,
                self.line.unwrap_or(0),
                self.column.unwrap_or(0)
            ));
        }

        if !self.labels.is_empty() {
            result.push_str(&format!("Labels: {}\n", self.labels.join(", ")));
        }

        if let Some(suggestion) = &self.suggestion {
            result.push_str(&format!("Compiler suggestion: {}\n", suggestion));
        }

        result.push_str(&format!("\nFull diagnostic:\n{}", self.rendered));

        result
    }
}

/// Parser for rustc errors
pub struct RustcErrorParser;

impl RustcErrorParser {
    /// Parse errors from cargo messages
    pub fn parse_cargo_messages(messages: &[CargoMessage]) -> Vec<ParsedError> {
        messages
            .iter()
            .filter_map(|m| m.as_diagnostic())
            .filter(|d| d.level == "error")
            .map(ParsedError::from_diagnostic)
            .collect()
    }

    /// Get the highest complexity category from a list of errors
    pub fn max_complexity(errors: &[ParsedError]) -> u8 {
        errors
            .iter()
            .map(|e| e.category.complexity())
            .max()
            .unwrap_or(0)
    }

    /// Group errors by category
    pub fn group_by_category(
        errors: &[ParsedError],
    ) -> std::collections::HashMap<ErrorCategory, Vec<&ParsedError>> {
        let mut groups = std::collections::HashMap::new();
        for error in errors {
            groups
                .entry(error.category)
                .or_insert_with(Vec::new)
                .push(error);
        }
        groups
    }

    /// Get summary statistics for errors
    pub fn summarize(errors: &[ParsedError]) -> ErrorSummary {
        let groups = Self::group_by_category(errors);

        ErrorSummary {
            total: errors.len(),
            by_category: groups.iter().map(|(k, v)| (*k, v.len())).collect(),
            max_complexity: Self::max_complexity(errors),
            has_borrow_errors: groups.contains_key(&ErrorCategory::BorrowChecker),
            has_lifetime_errors: groups.contains_key(&ErrorCategory::Lifetime),
            has_async_errors: groups.contains_key(&ErrorCategory::Async),
        }
    }
}

/// Summary of parsed errors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorSummary {
    /// Total error count
    pub total: usize,
    /// Count by category
    pub by_category: std::collections::HashMap<ErrorCategory, usize>,
    /// Maximum complexity level
    pub max_complexity: u8,
    /// Whether borrow checker errors are present
    pub has_borrow_errors: bool,
    /// Whether lifetime errors are present
    pub has_lifetime_errors: bool,
    /// Whether async errors are present
    pub has_async_errors: bool,
}

impl ErrorSummary {
    /// Get recommended model tier based on error composition
    pub fn recommended_tier(&self) -> &'static str {
        if self.has_lifetime_errors || self.has_async_errors || self.max_complexity >= 3 {
            "reasoning"
        } else if self.has_borrow_errors || self.max_complexity >= 2 {
            "specialized"
        } else {
            "fast"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_category_complexity() {
        assert_eq!(ErrorCategory::TypeMismatch.complexity(), 1);
        assert_eq!(ErrorCategory::BorrowChecker.complexity(), 2);
        assert_eq!(ErrorCategory::Lifetime.complexity(), 3);
    }

    #[test]
    fn test_classify_borrow_error() {
        use crate::feedback::compiler::ErrorCode;

        let diag = DiagnosticMessage {
            message: "cannot move out of borrowed content".to_string(),
            code: Some(ErrorCode {
                code: "E0507".to_string(),
                explanation: None,
            }),
            level: "error".to_string(),
            spans: vec![],
            children: vec![],
            rendered: None,
        };

        let parsed = ParsedError::from_diagnostic(&diag);
        assert_eq!(parsed.category, ErrorCategory::BorrowChecker);
    }

    #[test]
    fn test_classify_lifetime_error() {
        use crate::feedback::compiler::ErrorCode;

        let diag = DiagnosticMessage {
            message: "missing lifetime specifier".to_string(),
            code: Some(ErrorCode {
                code: "E0106".to_string(),
                explanation: None,
            }),
            level: "error".to_string(),
            spans: vec![],
            children: vec![],
            rendered: None,
        };

        let parsed = ParsedError::from_diagnostic(&diag);
        assert_eq!(parsed.category, ErrorCategory::Lifetime);
    }

    #[test]
    fn test_error_summary() {
        use crate::feedback::compiler::ErrorCode;

        let errors = vec![
            ParsedError::from_diagnostic(&DiagnosticMessage {
                message: "type mismatch".to_string(),
                code: Some(ErrorCode {
                    code: "E0308".to_string(),
                    explanation: None,
                }),
                level: "error".to_string(),
                spans: vec![],
                children: vec![],
                rendered: None,
            }),
            ParsedError::from_diagnostic(&DiagnosticMessage {
                message: "cannot borrow as mutable".to_string(),
                code: Some(ErrorCode {
                    code: "E0502".to_string(),
                    explanation: None,
                }),
                level: "error".to_string(),
                spans: vec![],
                children: vec![],
                rendered: None,
            }),
        ];

        let summary = RustcErrorParser::summarize(&errors);
        assert_eq!(summary.total, 2);
        assert!(summary.has_borrow_errors);
        assert!(!summary.has_lifetime_errors);
        assert_eq!(summary.recommended_tier(), "specialized");
    }
}
