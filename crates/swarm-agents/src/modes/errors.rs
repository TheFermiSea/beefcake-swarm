//! NS-1.4: Orchestration error taxonomy with retry classification.
//!
//! Every error in the mode orchestration layer is represented here. Callers
//! can query `is_retriable()` / `retry_category()` without string matching.
//!
//! ## Retry categories
//!
//! | Category           | Retriable | Max retries |
//! |--------------------|-----------|-------------|
//! | Transient          | yes       | configurable |
//! | RateLimit          | yes       | configurable with backoff |
//! | ContextExhausted   | yes       | 1 (after compaction) |
//! | ParseFailure       | yes       | 2 |
//! | ToolFailure        | yes       | 2 |
//! | PolicyViolation    | no        | — |
//! | MaxIterations      | no        | — |
//! | Cancelled          | no        | — |

use std::fmt;

use thiserror::Error;

/// Classification used by the orchestrator to decide whether to retry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryCategory {
    /// Transient network / inference backend error — safe to retry immediately.
    Transient,
    /// API rate limit — retry with exponential back-off.
    RateLimit,
    /// Context window would be exceeded — compact first, then retry.
    ContextExhausted,
    /// LLM returned output that failed schema parsing — retry with stricter prompt.
    ParseFailure,
    /// Deterministic tool (diff apply, cargo, etc.) returned an error — retry.
    ToolFailure,
    /// A safety or policy constraint was violated — do not retry; escalate.
    PolicyViolation,
    /// Budget of iterations / time exhausted — terminal.
    MaxIterations,
    /// Explicitly cancelled by caller or watchdog — terminal.
    Cancelled,
}

impl RetryCategory {
    pub fn is_retriable(self) -> bool {
        matches!(
            self,
            Self::Transient
                | Self::RateLimit
                | Self::ContextExhausted
                | Self::ParseFailure
                | Self::ToolFailure
        )
    }

    /// Suggested max retry attempts for retriable categories.
    ///
    /// Returns `None` for non-retriable categories.
    pub fn default_max_retries(self) -> Option<u32> {
        match self {
            Self::Transient => Some(3),
            Self::RateLimit => Some(5),
            Self::ContextExhausted => Some(1),
            Self::ParseFailure => Some(2),
            Self::ToolFailure => Some(2),
            _ => None,
        }
    }
}

impl fmt::Display for RetryCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient => write!(f, "transient"),
            Self::RateLimit => write!(f, "rate_limit"),
            Self::ContextExhausted => write!(f, "context_exhausted"),
            Self::ParseFailure => write!(f, "parse_failure"),
            Self::ToolFailure => write!(f, "tool_failure"),
            Self::PolicyViolation => write!(f, "policy_violation"),
            Self::MaxIterations => write!(f, "max_iterations"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Unified error type for all mode orchestration operations.
#[derive(Debug, Error)]
pub enum OrchestrationError {
    // ── Retriable ─────────────────────────────────────────────────────────
    /// LLM inference request failed (network, timeout, backend crash).
    #[error("Inference failure: {0}")]
    InferenceFailure(String),

    /// API rate limit from a cloud provider.
    #[error("Rate limit: {0}")]
    RateLimit(String),

    /// Prompt + history would exceed the model's context window.
    #[error("Context window exhausted: {0} tokens used, limit {1}")]
    ContextExhausted(u64, u64),

    /// LLM returned output that could not be parsed into the expected schema.
    #[error("Parse failure: {0}")]
    ParseFailure(String),

    /// A deterministic tool returned a non-success result.
    #[error("Tool failure [{tool}]: {message}")]
    ToolFailure { tool: String, message: String },

    // ── Non-retriable ─────────────────────────────────────────────────────
    /// A safety, policy, or blast-radius guard rejected the operation.
    #[error("Policy violation: {0}")]
    PolicyViolation(String),

    /// Maximum iteration budget consumed without reaching a terminal state.
    #[error("Max iterations ({0}) exceeded")]
    MaxIterations(u32),

    /// The operation was explicitly cancelled.
    #[error("Cancelled: {0}")]
    Cancelled(String),

    /// Configuration is invalid or missing required fields.
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Any other error that doesn't fit the above categories.
    #[error("Internal error: {0}")]
    Internal(#[from] anyhow::Error),
}

impl OrchestrationError {
    /// Classify this error for retry logic.
    pub fn retry_category(&self) -> RetryCategory {
        match self {
            Self::InferenceFailure(_) => RetryCategory::Transient,
            Self::RateLimit(_) => RetryCategory::RateLimit,
            Self::ContextExhausted(_, _) => RetryCategory::ContextExhausted,
            Self::ParseFailure(_) => RetryCategory::ParseFailure,
            Self::ToolFailure { .. } => RetryCategory::ToolFailure,
            Self::PolicyViolation(_) => RetryCategory::PolicyViolation,
            Self::MaxIterations(_) => RetryCategory::MaxIterations,
            Self::Cancelled(_) => RetryCategory::Cancelled,
            Self::Configuration(_) => RetryCategory::PolicyViolation,
            Self::Internal(_) => RetryCategory::Transient,
        }
    }

    /// Returns `true` if the orchestrator may retry after this error.
    pub fn is_retriable(&self) -> bool {
        self.retry_category().is_retriable()
    }

    /// Build a `ToolFailure` variant conveniently.
    pub fn tool(tool: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ToolFailure {
            tool: tool.into(),
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inference_failure_is_retriable() {
        let err = OrchestrationError::InferenceFailure("timeout".into());
        assert!(err.is_retriable());
        assert_eq!(err.retry_category(), RetryCategory::Transient);
        assert_eq!(err.retry_category().default_max_retries(), Some(3));
    }

    #[test]
    fn max_iterations_is_terminal() {
        let err = OrchestrationError::MaxIterations(10);
        assert!(!err.is_retriable());
        assert_eq!(err.retry_category().default_max_retries(), None);
    }

    #[test]
    fn context_exhausted_retries_once() {
        let err = OrchestrationError::ContextExhausted(32768, 32768);
        assert!(err.is_retriable());
        assert_eq!(err.retry_category().default_max_retries(), Some(1));
    }

    #[test]
    fn policy_violation_not_retriable() {
        let err = OrchestrationError::PolicyViolation("path traversal".into());
        assert!(!err.is_retriable());
    }

    #[test]
    fn tool_failure_is_retriable() {
        let err = OrchestrationError::tool("apply_diff", "hunk mismatch at line 42");
        assert!(err.is_retriable());
    }
}
