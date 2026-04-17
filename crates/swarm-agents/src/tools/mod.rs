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

/// Directories agents must never read or modify:
/// - `.beads/` / `.dolt` — issue tracker database (corruption → 100% dogfood failure)
/// - `.git/` — repo metadata
/// - `.swarm/` — harness state (session log, checkpoint, progress)
const FORBIDDEN_PREFIXES: &[&str] = &[".beads", ".git", ".dolt", ".swarm"];

/// Top-level filenames starting with any of these prefixes are blocked.
/// Catches legacy `.swarm-session.jsonl`, `.swarm-checkpoint.json`, etc.
const FORBIDDEN_FILENAME_PREFIXES: &[&str] = &[".swarm-"];

/// Exact filenames always blocked (redundant with prefix, but kept explicit for
/// well-known names so errors can be more helpful).
const FORBIDDEN_FILES: &[&str] = &[
    ".swarm-session.jsonl",
    ".swarm-progress.txt",
    ".swarm-events.jsonl",
    ".swarm-telemetry.jsonl",
    ".swarm-hook-events.jsonl",
];

/// Legacy swarm files that agents may inspect for recovery context but must
/// never mutate directly.
const READ_ONLY_SWARM_FILES: &[&str] = &[".swarm-checkpoint.json", ".swarm-checkpoint.json.tmp"];

/// Substrings in shell commands that reach into harness state, rejected by `sandbox_command`.
pub const FORBIDDEN_COMMAND_SUBSTRINGS: &[&str] = &[".swarm-", ".swarm/", ".beads/", ".git/"];

/// Marker substring embedded in every sandbox error message. `runtime_adapter`
/// greps tool result strings for this to fail-fast when agents repeatedly hit
/// the sandbox — changing this breaks that coupling.
pub const HARNESS_SANDBOX_MARKER: &str = "harness state";

/// Reject shell commands that reference harness state paths. Closes the
/// `proxy_run_command` hole where agents bypassed the file-tool sandbox
/// by running `cat .swarm-session.jsonl` directly.
pub fn sandbox_command(command: &str) -> Result<(), ToolError> {
    for needle in FORBIDDEN_COMMAND_SUBSTRINGS {
        if command.contains(needle) {
            return Err(ToolError::Sandbox(format!(
                "command references {HARNESS_SANDBOX_MARKER} (`{needle}`): \
                 these are orchestrator internals, not your target. \
                 Read the actual source file instead."
            )));
        }
    }
    Ok(())
}

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

    // Filename-component check catches "./.swarm-session.jsonl", "../.beads/x", etc.
    if let Some(filename) = normalized.file_name() {
        let filename_str = filename.to_string_lossy();
        let is_read_only_swarm_file = READ_ONLY_SWARM_FILES.contains(&filename_str.as_ref());

        if is_write && is_read_only_swarm_file {
            return Err(ToolError::Sandbox(format!(
                "path `{relative_path}` is {HARNESS_SANDBOX_MARKER} \
                 (checkpoint state is readable for recovery but never writable by agents)"
            )));
        }

        for forbidden in FORBIDDEN_FILES {
            if filename_str == *forbidden {
                return Err(ToolError::Sandbox(format!(
                    "path `{relative_path}` is {HARNESS_SANDBOX_MARKER} \
                     (session/checkpoint/progress logs), NOT your target. \
                     Read the actual source file you need to modify instead."
                )));
            }
        }
        for prefix in FORBIDDEN_FILENAME_PREFIXES {
            if filename_str.starts_with(prefix) && !is_read_only_swarm_file {
                return Err(ToolError::Sandbox(format!(
                    "path `{relative_path}` starts with `{prefix}` — {HARNESS_SANDBOX_MARKER}, \
                     not your target. Read the actual source file instead."
                )));
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn sandbox_command_blocks_harness_paths() {
        assert!(sandbox_command("cat .swarm-session.jsonl").is_err());
        assert!(sandbox_command("cat .swarm/session.jsonl").is_err());
        assert!(sandbox_command("grep foo .swarm-checkpoint.json").is_err());
        assert!(sandbox_command("ls .beads/").is_err());
        assert!(sandbox_command("cat .git/HEAD").is_err());
    }

    #[test]
    fn sandbox_command_allows_legitimate_commands() {
        assert!(sandbox_command("cargo check --workspace").is_ok());
        assert!(sandbox_command("grep foo src/main.rs").is_ok());
        assert!(sandbox_command("ls").is_ok());
        // Substring false-positive guard: bare "swarm" must not match.
        assert!(sandbox_command("grep swarm src/lib.rs").is_ok());
    }

    #[test]
    fn sandbox_check_blocks_legacy_dotswarm_filename_prefix() {
        let tmp = tempdir().unwrap();
        let err = sandbox_check(tmp.path(), ".swarm-new-future-file.log", false).unwrap_err();
        assert!(matches!(err, ToolError::Sandbox(_)));
    }

    #[test]
    fn sandbox_check_blocks_dotswarm_subdir() {
        let tmp = tempdir().unwrap();
        std::fs::create_dir_all(tmp.path().join(".swarm")).unwrap();
        let err = sandbox_check(tmp.path(), ".swarm/anything.json", false).unwrap_err();
        assert!(matches!(err, ToolError::Sandbox(_)));
    }

    #[test]
    fn sandbox_check_error_message_contains_marker() {
        // runtime_adapter detects HARNESS_SANDBOX_MARKER in tool results
        // to fail-fast agents fixated on harness paths — changing the
        // marker breaks that contract.
        let tmp = tempdir().unwrap();
        let err = sandbox_check(tmp.path(), ".swarm-session.jsonl", false).unwrap_err();
        assert!(format!("{err}").contains(HARNESS_SANDBOX_MARKER));
    }
}
