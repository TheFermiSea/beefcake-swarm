//! Compaction and summarization error taxonomy.
//!
//! Explicit typed errors for every failure class in the compaction
//! and summarization paths. No broad catches.

use serde::{Deserialize, Serialize};

/// High-level error kind for compaction failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionErrorKind {
    /// No entries available to compact.
    EmptyInput,
    /// Token budget insufficient for even the summary.
    BudgetExhausted,
    /// Summarization model call failed.
    SummarizationFailed,
    /// Summary output exceeded token budget.
    SummaryTooLarge,
    /// Integrity check failed after compaction.
    IntegrityViolation,
    /// Concurrent modification detected.
    ConcurrentModification,
    /// Persistence (save/load) failed.
    PersistenceFailed,
    /// Sequence gap detected in entries.
    SequenceGap,
    /// Compaction range is invalid.
    InvalidRange,
}

impl CompactionErrorKind {
    /// Whether this error is retryable.
    pub fn is_retryable(self) -> bool {
        matches!(
            self,
            Self::SummarizationFailed | Self::ConcurrentModification | Self::PersistenceFailed
        )
    }

    /// Suggested action for this error kind.
    pub fn suggested_action(self) -> &'static str {
        match self {
            Self::EmptyInput => "skip compaction — nothing to compact",
            Self::BudgetExhausted => "increase token budget or reduce retention",
            Self::SummarizationFailed => "retry with backoff or use fallback summarizer",
            Self::SummaryTooLarge => "request shorter summary or increase budget",
            Self::IntegrityViolation => "discard summary and retry from clean state",
            Self::ConcurrentModification => "re-read state and retry compaction",
            Self::PersistenceFailed => "check storage health and retry",
            Self::SequenceGap => "rebuild sequence index before compacting",
            Self::InvalidRange => "verify compaction range parameters",
        }
    }
}

impl std::fmt::Display for CompactionErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty_input"),
            Self::BudgetExhausted => write!(f, "budget_exhausted"),
            Self::SummarizationFailed => write!(f, "summarization_failed"),
            Self::SummaryTooLarge => write!(f, "summary_too_large"),
            Self::IntegrityViolation => write!(f, "integrity_violation"),
            Self::ConcurrentModification => write!(f, "concurrent_modification"),
            Self::PersistenceFailed => write!(f, "persistence_failed"),
            Self::SequenceGap => write!(f, "sequence_gap"),
            Self::InvalidRange => write!(f, "invalid_range"),
        }
    }
}

/// Compaction error with full context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionError {
    /// Error kind.
    pub kind: CompactionErrorKind,
    /// Human-readable detail.
    pub detail: String,
    /// Sequence range that was being compacted (if applicable).
    pub seq_range: Option<(u64, u64)>,
    /// Token count context (if applicable).
    pub token_context: Option<TokenContext>,
}

/// Token count context for budget-related errors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenContext {
    /// Current token count.
    pub current: u64,
    /// Budget limit.
    pub budget: u64,
    /// Summary size (if applicable).
    pub summary_size: Option<u64>,
}

impl CompactionError {
    /// Create a new compaction error.
    pub fn new(kind: CompactionErrorKind, detail: &str) -> Self {
        Self {
            kind,
            detail: detail.to_string(),
            seq_range: None,
            token_context: None,
        }
    }

    /// Add sequence range context.
    pub fn with_range(mut self, start: u64, end: u64) -> Self {
        self.seq_range = Some((start, end));
        self
    }

    /// Add token context.
    pub fn with_tokens(mut self, current: u64, budget: u64) -> Self {
        self.token_context = Some(TokenContext {
            current,
            budget,
            summary_size: None,
        });
        self
    }

    /// Whether this error is retryable.
    pub fn is_retryable(&self) -> bool {
        self.kind.is_retryable()
    }
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "compaction error [{}]: {}", self.kind, self.detail)?;
        if let Some((start, end)) = self.seq_range {
            write!(f, " (seq {}..{})", start, end)?;
        }
        if let Some(ref ctx) = self.token_context {
            write!(f, " (tokens: {}/{})", ctx.current, ctx.budget)?;
        }
        Ok(())
    }
}

impl std::error::Error for CompactionError {}

/// Summarization-specific error.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummarizationError {
    /// The model that was used.
    pub model: String,
    /// What went wrong.
    pub reason: String,
    /// Number of entries that were being summarized.
    pub entry_count: usize,
    /// Total tokens in input.
    pub input_tokens: u64,
    /// Whether a retry is recommended.
    pub retryable: bool,
}

impl SummarizationError {
    /// Create a new summarization error.
    pub fn new(model: &str, reason: &str, entry_count: usize, input_tokens: u64) -> Self {
        Self {
            model: model.to_string(),
            reason: reason.to_string(),
            entry_count,
            input_tokens,
            retryable: true,
        }
    }

    /// Mark as non-retryable.
    pub fn non_retryable(mut self) -> Self {
        self.retryable = false;
        self
    }

    /// Convert to a CompactionError.
    pub fn into_compaction_error(self) -> CompactionError {
        CompactionError::new(CompactionErrorKind::SummarizationFailed, &self.reason)
    }
}

impl std::fmt::Display for SummarizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "summarization failed [{}]: {} ({} entries, {} tokens)",
            self.model, self.reason, self.entry_count, self.input_tokens
        )
    }
}

impl std::error::Error for SummarizationError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compaction_error_display() {
        let err = CompactionError::new(CompactionErrorKind::BudgetExhausted, "over limit")
            .with_range(1, 10)
            .with_tokens(5000, 4000);
        let display = err.to_string();
        assert!(display.contains("budget_exhausted"));
        assert!(display.contains("over limit"));
        assert!(display.contains("seq 1..10"));
        assert!(display.contains("5000/4000"));
    }

    #[test]
    fn test_compaction_error_retryable() {
        assert!(CompactionErrorKind::SummarizationFailed.is_retryable());
        assert!(CompactionErrorKind::ConcurrentModification.is_retryable());
        assert!(CompactionErrorKind::PersistenceFailed.is_retryable());

        assert!(!CompactionErrorKind::EmptyInput.is_retryable());
        assert!(!CompactionErrorKind::BudgetExhausted.is_retryable());
        assert!(!CompactionErrorKind::IntegrityViolation.is_retryable());
    }

    #[test]
    fn test_suggested_actions() {
        assert!(CompactionErrorKind::EmptyInput
            .suggested_action()
            .contains("skip"));
        assert!(CompactionErrorKind::SummarizationFailed
            .suggested_action()
            .contains("retry"));
    }

    #[test]
    fn test_summarization_error() {
        let err = SummarizationError::new("or1-behemoth", "timeout", 5, 2000);
        assert!(err.retryable);
        let display = err.to_string();
        assert!(display.contains("or1-behemoth"));
        assert!(display.contains("timeout"));
        assert!(display.contains("5 entries"));

        let err2 = SummarizationError::new("strand-14b", "bad output", 3, 1000).non_retryable();
        assert!(!err2.retryable);
    }

    #[test]
    fn test_summarization_to_compaction_error() {
        let sum_err = SummarizationError::new("model", "fail", 1, 100);
        let comp_err = sum_err.into_compaction_error();
        assert_eq!(comp_err.kind, CompactionErrorKind::SummarizationFailed);
        assert!(comp_err.detail.contains("fail"));
    }

    #[test]
    fn test_error_kind_display() {
        assert_eq!(CompactionErrorKind::EmptyInput.to_string(), "empty_input");
        assert_eq!(
            CompactionErrorKind::BudgetExhausted.to_string(),
            "budget_exhausted"
        );
        assert_eq!(
            CompactionErrorKind::SummarizationFailed.to_string(),
            "summarization_failed"
        );
        assert_eq!(
            CompactionErrorKind::SummaryTooLarge.to_string(),
            "summary_too_large"
        );
        assert_eq!(
            CompactionErrorKind::IntegrityViolation.to_string(),
            "integrity_violation"
        );
        assert_eq!(
            CompactionErrorKind::ConcurrentModification.to_string(),
            "concurrent_modification"
        );
        assert_eq!(CompactionErrorKind::SequenceGap.to_string(), "sequence_gap");
        assert_eq!(
            CompactionErrorKind::InvalidRange.to_string(),
            "invalid_range"
        );
    }

    #[test]
    fn test_compaction_error_serde() {
        let err = CompactionError::new(CompactionErrorKind::BudgetExhausted, "over")
            .with_range(1, 5)
            .with_tokens(3000, 2000);
        let json = serde_json::to_string(&err).unwrap();
        let parsed: CompactionError = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kind, CompactionErrorKind::BudgetExhausted);
        assert_eq!(parsed.seq_range, Some((1, 5)));
        let ctx = parsed.token_context.unwrap();
        assert_eq!(ctx.current, 3000);
        assert_eq!(ctx.budget, 2000);
    }

    #[test]
    fn test_summarization_error_serde() {
        let err = SummarizationError::new("model-x", "timeout", 10, 5000);
        let json = serde_json::to_string(&err).unwrap();
        let parsed: SummarizationError = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.model, "model-x");
        assert_eq!(parsed.entry_count, 10);
        assert_eq!(parsed.input_tokens, 5000);
    }

    #[test]
    fn test_error_kind_serde() {
        let kind = CompactionErrorKind::IntegrityViolation;
        let json = serde_json::to_string(&kind).unwrap();
        assert_eq!(json, "\"integrity_violation\"");
        let parsed: CompactionErrorKind = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, CompactionErrorKind::IntegrityViolation);
    }

    #[test]
    fn test_compaction_error_no_context() {
        let err = CompactionError::new(CompactionErrorKind::EmptyInput, "nothing to do");
        let display = err.to_string();
        assert!(display.contains("empty_input"));
        assert!(!display.contains("seq"));
        assert!(!display.contains("tokens"));
    }
}
