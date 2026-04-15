//! Rig-compatible tools for the Manager-Worker swarm.
//!
//! Each tool implements `rig::tool::Tool` and can be attached to agents
//! via `AgentBuilder::tool()`. Tools are sandboxed to a worktree root.

pub mod apply_plan_tool;
pub mod astgrep_tool;
pub mod batch;
pub mod blast_radius_tool;
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

/// Root-level files that agents must never read OR write.
/// Session log and event files — workers corrupting these causes context loss.
const FORBIDDEN_FILES: &[&str] = &[
    ".swarm-session.jsonl",
    ".swarm-progress.txt",
    ".swarm-events.jsonl",
    ".swarm-telemetry.jsonl",
    ".swarm-hook-events.jsonl",
];

/// Root-level files that agents may read but must not write.
/// Workers were observed writing `"{}"` to `.swarm-checkpoint.json` in a loop
/// (observed 2026-04-15). Reads are harmless — the model probes for a resume
/// checkpoint on turn 1 and moves on when it gets a sensible response.
const WRITE_FORBIDDEN_FILES: &[&str] = &[
    ".swarm-checkpoint.json",
    ".swarm-checkpoint.json.tmp",
];

/// Validate that a resolved path stays within the sandbox root and does not
/// touch forbidden directories (`.beads/`, `.git/`, etc.) or protected files.
///
/// `is_write` — when `true`, also enforces `WRITE_FORBIDDEN_FILES` (files
/// that may be read but never mutated).
///
/// Returns the canonicalized path on success.
pub fn sandbox_check(
    working_dir: &Path,
    relative_path: &str,
    is_write: bool,
) -> Result<PathBuf, ToolError> {
    // Normalize the path to catch tricks like "./.swarm-session.jsonl" or
    // "foo/../.beads/config.yaml". Component iteration strips `.` and
    // resolves `..`, and Normal segments never contain `/`.
    let normalized = Path::new(relative_path);

    // Block access to forbidden files (checked against the final filename
    // component, not the raw string, to catch "./", "../" prefix bypasses).
    if let Some(filename) = normalized.file_name() {
        let filename_str = filename.to_string_lossy();
        for forbidden in FORBIDDEN_FILES {
            if filename_str == *forbidden {
                return Err(ToolError::Sandbox(format!(
                    "path `{relative_path}` is a protected swarm infrastructure file"
                )));
            }
        }
        if is_write {
            for forbidden in WRITE_FORBIDDEN_FILES {
                if filename_str == *forbidden {
                    return Err(ToolError::Sandbox(format!(
                        "path `{relative_path}` is a read-only swarm infrastructure file"
                    )));
                }
            }
        }
    }

    // Block access to forbidden directories. Normal components never contain
    // `/` (separators are stripped by Path::components), so the equality
    // check is sufficient — no need for starts_with("prefix/").
    for component in normalized.components() {
        if let std::path::Component::Normal(seg) = component {
            let seg_str = seg.to_string_lossy();
            for prefix in FORBIDDEN_PREFIXES {
                if seg_str == *prefix {
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
