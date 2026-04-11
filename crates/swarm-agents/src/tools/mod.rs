//! Rig-compatible tools for the Manager-Worker swarm.
//!
//! Each tool implements `rig::tool::Tool` and can be attached to agents
//! via `AgentBuilder::tool()`. Tools are sandboxed to a worktree root.

pub mod apply_plan_tool;
pub mod astgrep_tool;
pub mod batch;
pub mod bundles;
pub mod cargo_metadata_tool;
pub mod colgrep_tool;
pub mod delegate_worker;
pub mod exec_tool;
pub mod file_exists_tool;
pub mod fs_tools;
pub mod git_tools;
pub mod graph_context_tool;
pub mod migration_matrix;
pub mod notebook_tool;
pub mod patch_tool;
pub mod plan_parallel_tool;
pub mod proxy_wrappers;
pub mod quick_check;
pub mod search_code_tool;
pub mod shared;
pub mod submit_plan_tool;
pub mod tz_mcp_tool;
pub mod verifier_tool;
pub mod workpad_tool;

use std::path::{Path, PathBuf};
use std::time::Duration;

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

    #[error("notebook error: {0}")]
    Notebook(String),

    #[error("policy violation: {0}")]
    Policy(String),

    #[error("parse error: {0}")]
    Parse(String),

    #[error("validation error: {0}")]
    Validation(String),

    #[error("external service error: {0}")]
    External(String),
}

/// Run a command with timeout, returning stdout or a formatted error.
pub(crate) async fn run_command_with_timeout(
    program: &str,
    args: &[&str],
    working_dir: &Path,
    timeout_secs: u64,
) -> Result<String, ToolError> {
    let program = program.to_string();
    let args: Vec<String> = args.iter().map(|s| s.to_string()).collect();
    let wd = working_dir.to_path_buf();

    let result = tokio::task::spawn_blocking(move || {
        use std::os::unix::process::CommandExt;
        std::process::Command::new(&program)
            .args(&args)
            .current_dir(&wd)
            // Isolate in its own process group so we can kill the entire tree
            // on timeout — prevents zombie accumulation from nested cargo/rustc.
            .process_group(0)
            .output()
            .map_err(ToolError::Io)
    });

    match tokio::time::timeout(Duration::from_secs(timeout_secs), result).await {
        Ok(Ok(Ok(output))) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            if output.status.success() {
                Ok(stdout)
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                Ok(format!(
                    "Exit code: {}\nstdout:\n{}\nstderr:\n{}",
                    output.status, stdout, stderr
                ))
            }
        }
        Ok(Ok(Err(e))) => Err(e),
        Ok(Err(e)) => Err(ToolError::Io(std::io::Error::other(format!(
            "task joined with error (panic): {e}"
        )))),
        Err(_) => Err(ToolError::Timeout {
            seconds: timeout_secs,
        }),
    }
}

/// Directories that agents must never read or modify.
///
/// `.beads/` contains the issue tracker database — agents deleting it caused
/// 100% dogfood failure rate on 2026-03-14. `.git/` is obviously off-limits.
const FORBIDDEN_PREFIXES: &[&str] = &[".beads", ".git", ".dolt"];

/// Validate that a resolved path stays within the sandbox root and does not
/// touch forbidden directories (`.beads/`, `.git/`, etc.).
///
/// Returns the canonicalized path on success.
pub fn sandbox_check(working_dir: &Path, relative_path: &str) -> Result<PathBuf, ToolError> {
    // Block access to forbidden directories before any filesystem operations.
    // Normalize the path to catch tricks like "foo/../.beads/config.yaml".
    let normalized = Path::new(relative_path);
    for component in normalized.components() {
        if let std::path::Component::Normal(seg) = component {
            let seg_str = seg.to_string_lossy();
            for prefix in FORBIDDEN_PREFIXES {
                if seg_str == *prefix || seg_str.starts_with(&format!("{prefix}/")) {
                    return Err(ToolError::Sandbox(format!(
                        "path `{relative_path}` touches forbidden directory `{prefix}/`"
                    )));
                }
            }
        }
    }

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
