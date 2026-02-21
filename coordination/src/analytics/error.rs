//! Structured error types for the analytics module.
//!
//! Replaces `anyhow::Result` with domain-specific errors so consumers
//! can match on specific failure variants.

use std::path::PathBuf;

/// Errors from the analytics subsystem (skill library, experience replay).
#[derive(Debug, thiserror::Error)]
pub enum AnalyticsError {
    /// Failed to read a file from disk.
    #[error("Failed to read {path}: {source}")]
    FileRead {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Failed to write a file to disk.
    #[error("Failed to write {path}: {source}")]
    FileWrite {
        path: PathBuf,
        source: std::io::Error,
    },

    /// Failed to parse JSON data.
    #[error("Failed to parse JSON: {0}")]
    JsonParse(#[from] serde_json::Error),
}

/// Result type alias for analytics operations.
pub type AnalyticsResult<T> = Result<T, AnalyticsError>;
