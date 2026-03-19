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
    "touch", "mkdir", // Shell utilities (safe output/pipeline helpers)
    "echo", "printf", "true", "false",
];

/// Characters that indicate command chaining or injection intent.
/// Blocked in ALL execution modes (direct and shell).
const DANGEROUS_METACHARACTERS: &[char] = &[';', '`', '\n', '\r'];

/// Patterns that enable command substitution — blocked when using shell mode.
/// Without a shell these are harmless literals, but with `sh -c` they'd execute.
const SHELL_SUBSTITUTION_PATTERNS: &[&str] = &["$(", "${"];

/// Shell-like tokens that are no-ops in our tool (we always capture both
/// stdout and stderr). Stripped from direct-execution args so LLMs can write
/// natural shell commands like `cargo clippy 2>&1` without errors.
const NOOP_REDIRECTIONS: &[&str] = &["2>&1", "2>/dev/null"];

/// Default timeout for command execution.
/// Set high to accommodate fresh worktree builds (RocksDB C++ compilation
/// takes 10-15 min on ai-proxy).
const DEFAULT_TIMEOUT_SECS: u64 = 1800;

/// Timeout for cargo test specifically.
const TEST_TIMEOUT_SECS: u64 = 1800;

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
                          touch, mkdir, echo, printf. Use bd for beads issue tracking."
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
        // Reject characters that indicate command chaining or injection intent.
        if let Some(bad) = args
            .command
            .chars()
            .find(|c| DANGEROUS_METACHARACTERS.contains(c))
        {
            return Err(ToolError::CommandNotAllowed {
                command: format!("shell metacharacter '{bad}' not allowed in commands"),
            });
        }

        let has_pipe = args.command.contains('|');

        if has_pipe {
            // ── Shell pipeline mode ──
            // LLMs naturally write `cargo clippy 2>&1 | grep pattern | head -20`.
            // We validate every program in the pipeline against the allowlist,
            // then execute the full command via `sh -c`.

            // Block command substitution — safe as literal args but dangerous in sh.
            for pat in SHELL_SUBSTITUTION_PATTERNS {
                if args.command.contains(pat) {
                    return Err(ToolError::CommandNotAllowed {
                        command: format!(
                            "command substitution '{pat}' not allowed in shell pipelines"
                        ),
                    });
                }
            }

            // Validate every pipeline segment's program against the allowlist.
            for segment in split_pipeline_segments(&args.command)? {
                let seg = segment.trim();
                if seg.is_empty() {
                    continue;
                }
                let seg_parts = shlex::split(seg).ok_or_else(|| ToolError::CommandNotAllowed {
                    command: "invalid quoting in pipeline segment".to_string(),
                })?;
                let prog = seg_parts
                    .first()
                    .ok_or_else(|| ToolError::CommandNotAllowed {
                        command: "empty pipeline segment".to_string(),
                    })?;
                if !ALLOWED_COMMANDS.contains(&prog.as_str()) {
                    return Err(ToolError::CommandNotAllowed {
                        command: prog.to_string(),
                    });
                }
            }

            // Determine timeout from the first command in the pipeline.
            let timeout_secs = if args.command.starts_with("cargo") && args.command.contains("test")
            {
                TEST_TIMEOUT_SECS
            } else {
                DEFAULT_TIMEOUT_SECS
            };

            let cmd = args.command.clone();
            let working_dir = self.working_dir.clone();

            let result = tokio::task::spawn_blocking(move || {
                let output = std::process::Command::new("sh")
                    .args(["-c", &cmd])
                    .current_dir(&working_dir)
                    .output();

                format_output(output)
            });

            return match tokio::time::timeout(Duration::from_secs(timeout_secs), result).await {
                Ok(Ok(r)) => r,
                Ok(Err(e)) => Err(ToolError::Io(std::io::Error::other(format!(
                    "task join error: {e}"
                )))),
                Err(_) => Err(ToolError::Timeout {
                    seconds: timeout_secs,
                }),
            };
        }

        // ── Direct execution mode (no shell) ──
        // Use shell-words parsing to properly handle quoted arguments
        // (e.g., `rg "foo bar"` → ["rg", "foo bar"] instead of ["rg", "\"foo", "bar\""])
        let parts = shlex::split(&args.command).ok_or_else(|| ToolError::CommandNotAllowed {
            command: "invalid quoting in command".to_string(),
        })?;
        let program = parts.first().ok_or_else(|| ToolError::CommandNotAllowed {
            command: String::new(),
        })?;

        // Allowlist check
        if !ALLOWED_COMMANDS.contains(&program.as_str()) {
            return Err(ToolError::CommandNotAllowed {
                command: program.to_string(),
            });
        }

        // Determine timeout
        let timeout_secs = if program == "cargo" && parts.iter().any(|p| p == "test") {
            TEST_TIMEOUT_SECS
        } else {
            DEFAULT_TIMEOUT_SECS
        };

        let working_dir = self.working_dir.clone();
        let program_owned = program.to_string();

        // Strip no-op redirections (2>&1, 2>/dev/null) that LLMs add out of habit.
        // We already capture both stdout and stderr, so these are meaningless
        // without a shell and would be passed as literal args to the program.
        let arg_list: Vec<String> = parts[1..]
            .iter()
            .filter(|a| !NOOP_REDIRECTIONS.contains(&a.as_str()))
            .cloned()
            .collect();

        // Run in a blocking task to avoid blocking the async runtime.
        // Execute directly (no shell) to prevent metacharacter injection.
        let result = tokio::task::spawn_blocking(move || {
            let output = std::process::Command::new(&program_owned)
                .args(&arg_list)
                .current_dir(&working_dir)
                .output();

            format_output(output)
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

fn split_pipeline_segments(command: &str) -> Result<Vec<String>, ToolError> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut quote: Option<char> = None;
    let mut escaped = false;

    for ch in command.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
            continue;
        }

        match ch {
            '\\' => {
                current.push(ch);
                escaped = true;
            }
            '\'' | '"' => {
                if quote == Some(ch) {
                    quote = None;
                } else if quote.is_none() {
                    quote = Some(ch);
                }
                current.push(ch);
            }
            '|' if quote.is_none() => {
                let segment = current.trim();
                if segment.is_empty() {
                    return Err(ToolError::CommandNotAllowed {
                        command: "empty pipeline segment".to_string(),
                    });
                }
                segments.push(segment.to_string());
                current.clear();
            }
            _ => current.push(ch),
        }
    }

    if escaped || quote.is_some() {
        return Err(ToolError::CommandNotAllowed {
            command: "invalid quoting in pipeline segment".to_string(),
        });
    }

    let tail = current.trim();
    if tail.is_empty() {
        return Err(ToolError::CommandNotAllowed {
            command: "empty pipeline segment".to_string(),
        });
    }
    segments.push(tail.to_string());

    Ok(segments)
}

/// Format command output into a string for the agent.
fn format_output(
    output: Result<std::process::Output, std::io::Error>,
) -> Result<String, ToolError> {
    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let stderr = String::from_utf8_lossy(&out.stderr);

            if out.status.success() {
                Ok(format!("{stdout}{stderr}"))
            } else {
                let code = out.status.code().unwrap_or(-1);
                Ok(format!(
                    "EXIT CODE: {code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                ))
            }
        }
        Err(e) => Err(ToolError::Io(e)),
    }
}

#[cfg(test)]
mod tests {
    use super::split_pipeline_segments;

    #[test]
    fn pipeline_splitter_keeps_quoted_regex_pipes() {
        let command = r#"cargo test --lib 2>&1 | grep -E "FAILED|failures:""#;
        let segments = split_pipeline_segments(command).expect("split quoted pipeline");
        assert_eq!(
            segments,
            vec![
                "cargo test --lib 2>&1".to_string(),
                r#"grep -E "FAILED|failures:""#.to_string()
            ]
        );
    }

    #[test]
    fn pipeline_splitter_keeps_escaped_pipe_literals() {
        let command = r#"printf FAILED\|failures: | head -n 1"#;
        let segments = split_pipeline_segments(command).expect("split escaped pipe");
        assert_eq!(
            segments,
            vec![
                r#"printf FAILED\|failures:"#.to_string(),
                "head -n 1".to_string()
            ]
        );
    }

    #[test]
    fn pipeline_splitter_rejects_unterminated_quotes() {
        let err = split_pipeline_segments(r#"grep "FAILED|failures:"#).unwrap_err();
        assert!(
            format!("{err:?}").contains("invalid quoting"),
            "unexpected error: {err:?}"
        );
    }
}
