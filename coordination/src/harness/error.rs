//! Harness error types
//!
//! Provides structured error handling for all harness operations.
//! Includes agent-friendly error formatting with recovery actions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for harness operations
pub type HarnessResult<T> = Result<T, HarnessError>;

// ============================================================================
// Structured Error Response (Agent-Friendly)
// ============================================================================

/// Structured error response for MCP tools that helps agents self-recover.
///
/// This format is designed based on best practices from:
/// - Block's MCP Server Playbook: "Errors should tell the agent what to do"
/// - Hugo Bowne's Agent Harness Principles: "Recovery actions beat error messages"
///
/// # Example Response
/// ```json
/// {
///   "code": "REGISTRY_CORRUPTED",
///   "message": "Feature registry JSON is invalid",
///   "recovery_action": "Call harness_start with force_recovery=true to reset",
///   "context": {
///     "iteration": 5,
///     "session_id": "abc123",
///     "feature_id": "my-feature"
///   }
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredError {
    /// Machine-readable error code (e.g., "REGISTRY_CORRUPTED", "MAX_ITERATIONS")
    pub code: String,

    /// Human-readable error message
    pub message: String,

    /// Actionable recovery instruction for the agent
    pub recovery_action: String,

    /// Relevant context for debugging and recovery
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub context: HashMap<String, serde_json::Value>,

    /// Whether this error is retryable (transient failure)
    #[serde(default)]
    pub retryable: bool,
}

impl StructuredError {
    /// Create a new structured error
    pub fn new(
        code: impl Into<String>,
        message: impl Into<String>,
        recovery_action: impl Into<String>,
    ) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            recovery_action: recovery_action.into(),
            context: HashMap::new(),
            retryable: false,
        }
    }

    /// Add context key-value pair
    pub fn with_context(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.context.insert(key.into(), value.into());
        self
    }

    /// Mark as retryable
    pub fn retryable(mut self) -> Self {
        self.retryable = true;
        self
    }

    /// Add session context (common pattern)
    pub fn with_session(self, session_id: &str, iteration: u32) -> Self {
        self.with_context("session_id", session_id.to_string())
            .with_context("iteration", iteration)
    }

    /// Add feature context
    pub fn with_feature(self, feature_id: &str) -> Self {
        self.with_context("feature_id", feature_id.to_string())
    }
}

impl std::fmt::Display for StructuredError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}", self.code, self.message)
    }
}

impl std::error::Error for StructuredError {}

/// Errors that can occur during harness operations
#[derive(Error, Debug)]
pub enum HarnessError {
    /// Feature registry file not found
    #[error("Feature registry not found at {path}")]
    RegistryNotFound { path: PathBuf },

    /// Feature registry contains invalid JSON
    #[error("Invalid feature registry JSON: {message}")]
    InvalidRegistry { message: String },

    /// Feature not found in registry
    #[error("Feature not found: {feature_id}")]
    FeatureNotFound { feature_id: String },

    /// Progress file operation failed
    #[error("Progress file error: {message}")]
    ProgressFileError { message: String },

    /// Git operation failed
    #[error("Git operation failed: {operation} - {message}")]
    GitError { operation: String, message: String },

    /// Session error
    #[error("Session error: {message}")]
    SessionError { message: String },

    /// Maximum iterations reached
    #[error("Maximum iterations ({max}) reached without completion")]
    MaxIterationsReached { max: u32 },

    /// Startup ritual failed
    #[error("Startup ritual failed at step '{step}': {message}")]
    StartupFailed { step: String, message: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError { message: String },

    /// IO error wrapper
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Working directory mismatch
    #[error("Working directory mismatch: expected {expected}, got {actual}")]
    WorkingDirectoryMismatch { expected: PathBuf, actual: PathBuf },

    /// Uncommitted changes prevent operation
    #[error("Uncommitted changes detected. Commit or stash changes before proceeding.")]
    UncommittedChanges,

    /// Invalid state transition
    #[error("Invalid state transition from {from} to {to}")]
    InvalidStateTransition { from: String, to: String },

    /// Validation error (for invalid input parameters)
    #[error("Validation error: {message}")]
    ValidationError { message: String },

    /// Blocked by pending intervention (Phase 5: Human Intervention Points)
    #[error(
        "Blocked by pending intervention(s): {message}. Resolve intervention(s) before continuing."
    )]
    BlockedByIntervention { message: String },
}

impl HarnessError {
    /// Create a registry not found error
    pub fn registry_not_found(path: impl Into<PathBuf>) -> Self {
        Self::RegistryNotFound { path: path.into() }
    }

    /// Create an invalid registry error
    pub fn invalid_registry(message: impl Into<String>) -> Self {
        Self::InvalidRegistry {
            message: message.into(),
        }
    }

    /// Create a feature not found error
    pub fn feature_not_found(feature_id: impl Into<String>) -> Self {
        Self::FeatureNotFound {
            feature_id: feature_id.into(),
        }
    }

    /// Create a git error
    pub fn git(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::GitError {
            operation: operation.into(),
            message: message.into(),
        }
    }

    /// Create a session error
    pub fn session(message: impl Into<String>) -> Self {
        Self::SessionError {
            message: message.into(),
        }
    }

    /// Create a startup failed error
    pub fn startup_failed(step: impl Into<String>, message: impl Into<String>) -> Self {
        Self::StartupFailed {
            step: step.into(),
            message: message.into(),
        }
    }

    /// Create a config error
    pub fn config(message: impl Into<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    /// Create a progress file error
    pub fn progress(message: impl Into<String>) -> Self {
        Self::ProgressFileError {
            message: message.into(),
        }
    }

    /// Create a validation error (for invalid input parameters)
    pub fn validation(message: impl Into<String>) -> Self {
        Self::ValidationError {
            message: message.into(),
        }
    }

    /// Create a blocked by intervention error (Phase 5: Human Intervention Points)
    pub fn blocked_by_intervention(message: impl Into<String>) -> Self {
        Self::BlockedByIntervention {
            message: message.into(),
        }
    }

    /// Check if this error is retryable (transient failure)
    pub fn is_retryable(&self) -> bool {
        match self {
            // Git operations that might succeed on retry
            Self::GitError { message, .. } => {
                let lower = message.to_lowercase();
                // Lock file conflicts
                lower.contains("lock") ||
                // Timeout or network issues
                lower.contains("timeout") ||
                lower.contains("connection") ||
                lower.contains("network") ||
                // Repository busy
                lower.contains("could not lock") ||
                lower.contains("another git process")
            }
            // IO errors that might be transient
            Self::Io(e) => matches!(
                e.kind(),
                std::io::ErrorKind::WouldBlock
                    | std::io::ErrorKind::Interrupted
                    | std::io::ErrorKind::TimedOut
            ),
            // Validation errors are not retryable
            Self::ValidationError { .. } => false,
            // Blocked by intervention is not retryable - must resolve intervention first
            Self::BlockedByIntervention { .. } => false,
            _ => false,
        }
    }

    /// Get recovery suggestion for this error
    pub fn recovery_suggestion(&self) -> Option<&'static str> {
        match self {
            Self::RegistryNotFound { .. } => Some(
                "Create a new features.json with `harness_init` or copy from a template. \
                 Example: echo '[]' > features.json",
            ),
            Self::InvalidRegistry { .. } => Some(
                "The features.json file is corrupted. Try restoring from .features.json.backup \
                 or git: `git checkout -- features.json`. If no backup exists, create empty: \
                 echo '[]' > features.json",
            ),
            Self::FeatureNotFound { .. } => Some(
                "Verify the feature ID exists in features.json. Use `harness_status` to list \
                 available features.",
            ),
            Self::GitError { operation, message } => {
                let lower = message.to_lowercase();
                if lower.contains("lock") || lower.contains("another git process") {
                    Some("Git lock file conflict. Wait a moment and retry, or remove stale lock: \
                          rm -f .git/index.lock")
                } else if lower.contains("nothing to commit") {
                    Some("No changes to commit. Make changes first before creating a checkpoint.")
                } else if lower.contains("not a git repository") {
                    Some("Initialize git repository: git init")
                } else if operation.contains("commit") {
                    Some("Ensure there are staged changes: git add -A")
                } else {
                    Some("Check git status and repository state: git status")
                }
            }
            Self::SessionError { .. } => Some(
                "Session state may be corrupted. Try `harness_start` with a fresh session or \
                 remove .harness-session.json to reset.",
            ),
            Self::MaxIterationsReached { max } => {
                // Can't include max in static str, but provide general guidance
                let _ = max;
                Some(
                    "Maximum iterations reached. Increase max_iterations in config or complete \
                     the current feature before continuing.",
                )
            }
            Self::StartupFailed { step, .. } => {
                let _ = step;
                Some(
                    "Startup ritual failed. Check git state, features.json, and progress file. \
                     Try: git status && cat features.json",
                )
            }
            Self::ConfigError { .. } => Some(
                "Check configuration settings. Verify paths exist and environment variables \
                 are set correctly.",
            ),
            Self::ProgressFileError { .. } => Some(
                "Progress file may be corrupted. Safe to delete and restart: \
                 rm claude-progress.txt && harness_start",
            ),
            Self::Io(e) => match e.kind() {
                std::io::ErrorKind::NotFound => {
                    Some("File or directory not found. Check the path exists.")
                }
                std::io::ErrorKind::PermissionDenied => {
                    Some("Permission denied. Check file permissions: ls -la <path>")
                }
                std::io::ErrorKind::AlreadyExists => {
                    Some("File already exists. Remove or rename the existing file.")
                }
                _ => Some("IO error occurred. Check disk space and file permissions."),
            },
            Self::Json(_) => Some(
                "JSON parsing failed. Validate JSON syntax: python -m json.tool < file.json",
            ),
            Self::WorkingDirectoryMismatch { .. } => Some(
                "Working directory changed unexpectedly. Ensure you're in the project root.",
            ),
            Self::UncommittedChanges => Some(
                "Commit or stash changes first: git stash or git commit -am 'WIP'",
            ),
            Self::InvalidStateTransition { .. } => Some(
                "Invalid operation for current session state. Check session status with \
                 `harness_status`.",
            ),
            Self::ValidationError { .. } => Some(
                "Invalid input parameters. Check the request parameters and try again.",
            ),
            Self::BlockedByIntervention { .. } => Some(
                "Work is blocked by pending intervention(s). Call harness_status to see pending \
                 interventions, then use harness_resolve_intervention to resolve them before continuing.",
            ),
        }
    }

    /// Get error with recovery suggestion formatted
    pub fn with_suggestion(&self) -> String {
        match self.recovery_suggestion() {
            Some(suggestion) => format!("{}\n\nRecovery: {}", self, suggestion),
            None => self.to_string(),
        }
    }

    /// Convert to structured error for MCP tool responses
    ///
    /// Returns a StructuredError with:
    /// - Machine-readable error code
    /// - Human-readable message
    /// - Actionable recovery instruction
    /// - Retryable flag for transient errors
    pub fn to_structured(&self) -> StructuredError {
        let (code, recovery) = match self {
            Self::RegistryNotFound { path } => (
                "REGISTRY_NOT_FOUND",
                format!(
                    "Create features.json: echo '[]' > {} or call harness_start with force_recovery=true",
                    path.display()
                ),
            ),
            Self::InvalidRegistry { .. } => (
                "REGISTRY_CORRUPTED",
                "Restore from backup: cp .features.json.backup features.json or call harness_start with force_recovery=true".to_string(),
            ),
            Self::FeatureNotFound { feature_id } => (
                "FEATURE_NOT_FOUND",
                format!(
                    "Verify feature '{}' exists by calling harness_status with include_features=true",
                    feature_id
                ),
            ),
            Self::ProgressFileError { .. } => (
                "PROGRESS_FILE_ERROR",
                "Delete and restart: rm claude-progress.txt && call harness_start".to_string(),
            ),
            Self::GitError { operation, message } => {
                let lower = message.to_lowercase();
                if lower.contains("lock") || lower.contains("another git process") {
                    (
                        "GIT_LOCK_CONFLICT",
                        "Wait and retry, or remove stale lock: rm -f .git/index.lock".to_string(),
                    )
                } else if lower.contains("nothing to commit") {
                    (
                        "GIT_NOTHING_TO_COMMIT",
                        "Make changes before creating checkpoint, or skip checkpoint".to_string(),
                    )
                } else if lower.contains("not a git repository") {
                    (
                        "GIT_NOT_INITIALIZED",
                        "Initialize git: git init && git add -A && git commit -m 'Initial'".to_string(),
                    )
                } else {
                    (
                        "GIT_ERROR",
                        format!("Check git state with: git status. Operation '{}' failed.", operation),
                    )
                }
            }
            Self::SessionError { .. } => (
                "SESSION_ERROR",
                "Call harness_start to reinitialize session, or rm .harness-session.json for fresh start".to_string(),
            ),
            Self::MaxIterationsReached { max } => (
                "MAX_ITERATIONS_REACHED",
                format!(
                    "Session limit ({}) reached. Call harness_end, then harness_start with higher max_iterations, or complete current feature first.",
                    max
                ),
            ),
            Self::StartupFailed { step, .. } => (
                "STARTUP_FAILED",
                format!(
                    "Startup failed at '{}'. Check git status, features.json, and working directory. Try: harness_start with force_recovery=true",
                    step
                ),
            ),
            Self::ConfigError { .. } => (
                "CONFIG_ERROR",
                "Check configuration: verify paths exist and environment variables are set correctly".to_string(),
            ),
            Self::Io(e) => match e.kind() {
                std::io::ErrorKind::NotFound => (
                    "FILE_NOT_FOUND",
                    "Verify path exists before operation".to_string(),
                ),
                std::io::ErrorKind::PermissionDenied => (
                    "PERMISSION_DENIED",
                    "Check file permissions: ls -la <path>".to_string(),
                ),
                _ => (
                    "IO_ERROR",
                    "Check disk space and file permissions".to_string(),
                ),
            },
            Self::Json(_) => (
                "JSON_PARSE_ERROR",
                "Validate JSON syntax or restore from backup".to_string(),
            ),
            Self::WorkingDirectoryMismatch { expected, actual } => (
                "WORKING_DIRECTORY_MISMATCH",
                format!(
                    "Change to correct directory: cd {} (currently in {})",
                    expected.display(),
                    actual.display()
                ),
            ),
            Self::UncommittedChanges => (
                "UNCOMMITTED_CHANGES",
                "Commit or stash changes: git stash or git commit -am 'WIP'".to_string(),
            ),
            Self::InvalidStateTransition { from, to } => (
                "INVALID_STATE_TRANSITION",
                format!(
                    "Cannot transition from '{}' to '{}'. Call harness_status to check current state.",
                    from, to
                ),
            ),
            Self::ValidationError { message } => (
                "VALIDATION_ERROR",
                format!(
                    "Invalid input: {}. Check parameters and try again.",
                    message
                ),
            ),
            Self::BlockedByIntervention { message } => (
                "BLOCKED_BY_INTERVENTION",
                format!(
                    "Blocked by intervention(s): {}. Call harness_status to see pending interventions, then harness_resolve_intervention to resolve them.",
                    message
                ),
            ),
        };

        let mut structured = StructuredError::new(code, self.to_string(), recovery);

        if self.is_retryable() {
            structured = structured.retryable();
        }

        structured
    }

    /// Convert to structured error JSON string for MCP responses
    pub fn to_structured_json(&self) -> String {
        serde_json::to_string_pretty(&self.to_structured())
            .unwrap_or_else(|_| format!(r#"{{"code":"SERIALIZATION_ERROR","message":"{}"}}"#, self))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = HarnessError::registry_not_found("/path/to/features.json");
        assert!(err.to_string().contains("Feature registry not found"));

        let err = HarnessError::MaxIterationsReached { max: 20 };
        assert!(err.to_string().contains("20"));

        let err = HarnessError::git("commit", "nothing to commit");
        assert!(err.to_string().contains("commit"));
        assert!(err.to_string().contains("nothing to commit"));
    }

    #[test]
    fn test_error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let harness_err: HarnessError = io_err.into();
        assert!(matches!(harness_err, HarnessError::Io(_)));
    }

    #[test]
    fn test_is_retryable() {
        // Lock file errors should be retryable
        let err = HarnessError::git("add", "fatal: Unable to create lock file");
        assert!(err.is_retryable());

        // Another git process error should be retryable
        let err = HarnessError::git("commit", "another git process seems to be running");
        assert!(err.is_retryable());

        // Network errors should be retryable
        let err = HarnessError::git("fetch", "connection timed out");
        assert!(err.is_retryable());

        // Regular git errors should not be retryable
        let err = HarnessError::git("commit", "nothing to commit");
        assert!(!err.is_retryable());

        // Non-git errors should not be retryable
        let err = HarnessError::feature_not_found("test");
        assert!(!err.is_retryable());

        // IO interrupt should be retryable
        let io_err = std::io::Error::new(std::io::ErrorKind::Interrupted, "interrupted");
        let err: HarnessError = io_err.into();
        assert!(err.is_retryable());
    }

    #[test]
    fn test_recovery_suggestions() {
        // Registry not found should have suggestion
        let err = HarnessError::registry_not_found("/path/to/features.json");
        assert!(err.recovery_suggestion().is_some());
        assert!(err.recovery_suggestion().unwrap().contains("features.json"));

        // Invalid registry should mention backup
        let err = HarnessError::invalid_registry("parse error");
        assert!(err.recovery_suggestion().unwrap().contains("backup"));

        // Git lock error should suggest removing lock
        let err = HarnessError::git("add", "unable to create lock file");
        assert!(err.recovery_suggestion().unwrap().contains("lock"));

        // Max iterations should mention increasing limit
        let err = HarnessError::MaxIterationsReached { max: 20 };
        assert!(err
            .recovery_suggestion()
            .unwrap()
            .contains("max_iterations"));

        // with_suggestion should format nicely
        let err = HarnessError::UncommittedChanges;
        let formatted = err.with_suggestion();
        assert!(formatted.contains("Uncommitted changes"));
        assert!(formatted.contains("Recovery:"));
        assert!(formatted.contains("stash"));
    }

    // ========================================================================
    // StructuredError Tests
    // ========================================================================

    #[test]
    fn test_structured_error_creation() {
        let err = StructuredError::new(
            "TEST_ERROR",
            "Something went wrong",
            "Try again with correct parameters",
        );
        assert_eq!(err.code, "TEST_ERROR");
        assert_eq!(err.message, "Something went wrong");
        assert!(!err.retryable);
        assert!(err.context.is_empty());
    }

    #[test]
    fn test_structured_error_with_context() {
        let err = StructuredError::new("TEST_ERROR", "Error", "Fix it")
            .with_context("session_id", "abc123")
            .with_context("iteration", 5)
            .with_feature("my-feature");

        assert_eq!(err.context.len(), 3);
        assert_eq!(err.context.get("session_id").unwrap(), "abc123");
        assert_eq!(err.context.get("iteration").unwrap(), &5);
        assert_eq!(err.context.get("feature_id").unwrap(), "my-feature");
    }

    #[test]
    fn test_structured_error_retryable() {
        let err = StructuredError::new("TRANSIENT", "Retry please", "Wait and retry").retryable();
        assert!(err.retryable);
    }

    #[test]
    fn test_structured_error_with_session() {
        let err = StructuredError::new("TEST", "Test", "Test").with_session("session-123", 10);
        assert_eq!(err.context.get("session_id").unwrap(), "session-123");
        assert_eq!(err.context.get("iteration").unwrap(), &10);
    }

    #[test]
    fn test_harness_error_to_structured() {
        // Test registry not found
        let err = HarnessError::registry_not_found("/path/to/features.json");
        let structured = err.to_structured();
        assert_eq!(structured.code, "REGISTRY_NOT_FOUND");
        assert!(structured.recovery_action.contains("features.json"));
        assert!(!structured.retryable);

        // Test git lock error (should be retryable)
        let err = HarnessError::git("add", "fatal: Unable to create lock file");
        let structured = err.to_structured();
        assert_eq!(structured.code, "GIT_LOCK_CONFLICT");
        assert!(structured.retryable);

        // Test max iterations
        let err = HarnessError::MaxIterationsReached { max: 20 };
        let structured = err.to_structured();
        assert_eq!(structured.code, "MAX_ITERATIONS_REACHED");
        assert!(structured.recovery_action.contains("20"));
    }

    #[test]
    fn test_harness_error_to_structured_json() {
        let err = HarnessError::feature_not_found("my-feature");
        let json = err.to_structured_json();

        // Should be valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["code"], "FEATURE_NOT_FOUND");
        assert!(parsed["recovery_action"]
            .as_str()
            .unwrap()
            .contains("my-feature"));
    }

    #[test]
    fn test_structured_error_serialization() {
        let err = StructuredError::new("TEST_CODE", "Test message", "Test recovery")
            .with_context("key", "value")
            .retryable();

        let json = serde_json::to_string(&err).unwrap();
        let restored: StructuredError = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.code, "TEST_CODE");
        assert_eq!(restored.message, "Test message");
        assert_eq!(restored.recovery_action, "Test recovery");
        assert!(restored.retryable);
        assert_eq!(restored.context.get("key").unwrap(), "value");
    }
}
