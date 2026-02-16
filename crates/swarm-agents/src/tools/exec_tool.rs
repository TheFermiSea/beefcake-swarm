//! Sandboxed command execution tool.
//!
//! Only allows a specific set of commands (cargo, git, etc.) and enforces
//! a timeout to prevent runaway processes.

use std::path::{Path, PathBuf};
use std::time::Duration;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::ToolError;

/// Commands that are allowed to be executed.
///
/// Prefer modern Rust CLI tools (rg, fd, bat, sd, delta) over their classic
/// Unix counterparts. Both are allowed since LLMs may default to classic syntax.
const ALLOWED_COMMANDS: &[&str] = &[
    // Core build/vcs/tracking
    "cargo", "git", "bd", // Modern Rust CLI tools (preferred)
    "rg", "fd", "bat", "sd", "delta", // Classic Unix fallbacks (LLMs default to these)
    "ls", "wc", "find", "grep", "cat", "head", "tail", "sed", "awk", "sort", "uniq", "diff",
    // File operations
    "touch", "mkdir",
];

/// Default timeout for command execution.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Timeout for cargo test specifically.
const TEST_TIMEOUT_SECS: u64 = 300;

#[derive(Deserialize)]
pub struct RunCommandArgs {
    /// The command to run (e.g. "cargo test -p coordination").
    pub command: String,
}

/// Execute a shell command within the worktree, subject to allowlist and timeout.
pub struct RunCommandTool {
    pub working_dir: PathBuf,
}

impl RunCommandTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for RunCommandTool {
    const NAME: &'static str = "run_command";
    type Error = ToolError;
    type Args = RunCommandArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "run_command".into(),
            description: "Run a shell command in the workspace. \
                          Prefer modern tools: rg (not grep), fd (not find), bat (not cat), \
                          sd (not sed), delta (not diff). \
                          Also allowed: cargo, git, bd, ls, wc, head, tail, awk, sort, uniq, \
                          touch, mkdir. Use bd for beads issue tracking."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The command to run (e.g. 'cargo build', 'git diff')"
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let parts: Vec<&str> = args.command.split_whitespace().collect();
        let program = parts.first().ok_or_else(|| ToolError::CommandNotAllowed {
            command: String::new(),
        })?;

        // Allowlist check
        if !ALLOWED_COMMANDS.contains(program) {
            return Err(ToolError::CommandNotAllowed {
                command: program.to_string(),
            });
        }

        // Determine timeout
        let timeout_secs = if args.command.contains("cargo test") {
            TEST_TIMEOUT_SECS
        } else {
            DEFAULT_TIMEOUT_SECS
        };

        let working_dir = self.working_dir.clone();
        let command = args.command.clone();

        // Run in a blocking task to avoid blocking the async runtime
        let result = tokio::task::spawn_blocking(move || {
            let output = std::process::Command::new("sh")
                .args(["-c", &command])
                .current_dir(&working_dir)
                .output();

            match output {
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout);
                    let stderr = String::from_utf8_lossy(&out.stderr);

                    if out.status.success() {
                        Ok(format!("{stdout}{stderr}"))
                    } else {
                        let code = out.status.code().unwrap_or(-1);
                        // Return stderr as output (not an error) so the agent can see it
                        Ok(format!(
                            "EXIT CODE: {code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                        ))
                    }
                }
                Err(e) => Err(ToolError::Io(e)),
            }
        });

        // Apply timeout
        match tokio::time::timeout(Duration::from_secs(timeout_secs), result).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => Err(ToolError::Io(std::io::Error::other(format!(
                "task join error: {e}"
            )))),
            Err(_) => Err(ToolError::Timeout {
                seconds: timeout_secs,
            }),
        }
    }
}
