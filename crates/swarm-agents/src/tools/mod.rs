//! Rig-compatible tools for the Manager-Worker swarm.
//!
//! Each tool implements `rig::tool::Tool` and can be attached to agents
//! via `AgentBuilder::tool()`. Tools are sandboxed to a worktree root.

pub mod exec_tool;
pub mod fs_tools;
pub mod patch_tool;
pub mod verifier_tool;

use std::path::{Path, PathBuf};

/// Errors that can occur during tool execution.
#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("path `{0}` escapes sandbox")]
    Sandbox(String),

    #[error("command `{command}` not in allowlist")]
    CommandNotAllowed { command: String },

    #[error("command timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("command failed (exit {code}): {stderr}")]
    CommandFailed { code: i32, stderr: String },

    #[error("verifier error: {0}")]
    Verifier(String),
}

/// Validate that a resolved path stays within the sandbox root.
///
/// Returns the canonicalized path on success.
pub fn sandbox_check(working_dir: &Path, relative_path: &str) -> Result<PathBuf, ToolError> {
    let candidate = working_dir.join(relative_path);
    let resolved = candidate
        .canonicalize()
        .or_else(|_| {
            // File might not exist yet (for writes) — canonicalize parent
            if let Some(parent) = candidate.parent() {
                let canon_parent = parent.canonicalize()?;
                Ok(canon_parent.join(candidate.file_name().unwrap_or_default()))
            } else {
                Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    "cannot resolve path",
                ))
            }
        })
        .map_err(ToolError::Io)?;

    let canon_root = working_dir.canonicalize().map_err(ToolError::Io)?;

    if !resolved.starts_with(&canon_root) {
        return Err(ToolError::Sandbox(relative_path.to_string()));
    }
    Ok(resolved)
}
