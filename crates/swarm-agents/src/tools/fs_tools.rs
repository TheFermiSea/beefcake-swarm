//! File system tools: read, write, and list files within a sandboxed worktree.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{sandbox_check, ToolError};

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ReadFileArgs {
    /// Relative path within the workspace.
    pub path: String,
    /// First line to read (1-indexed, inclusive). Optional.
    /// When provided with `end_line`, only that line range is returned.
    pub start_line: Option<u32>,
    /// Last line to read (1-indexed, inclusive). Optional.
    /// When provided with `start_line`, only that line range is returned.
    pub end_line: Option<u32>,
}

/// Read a file from the worktree. Path must stay within the sandbox.
///
/// When `max_output_chars` is set, large files are truncated with a
/// `[...N lines truncated...]` marker. This keeps tool results small enough
/// for small models (HydraCoder 30B MoE) to stay in tool-calling mode on
/// subsequent turns. Controlled by `SWARM_READ_FILE_MAX_CHARS` (default: 0 = unlimited).
pub struct ReadFileTool {
    pub working_dir: PathBuf,
    /// Maximum characters to return. 0 = unlimited.
    pub max_output_chars: usize,
}

/// Default max chars for read_file output (env override: `SWARM_READ_FILE_MAX_CHARS`).
/// 6000 chars ≈ 1500 tokens — keeps total context under HydraCoder's reliable zone.
fn default_read_file_max_chars() -> usize {
    std::env::var("SWARM_READ_FILE_MAX_CHARS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(6000)
}

impl ReadFileTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
            max_output_chars: default_read_file_max_chars(),
        }
    }
}

impl Tool for ReadFileTool {
    const NAME: &'static str = "read_file";
    type Error = ToolError;
    type Args = ReadFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".into(),
            description: "Read the contents of a file in the workspace. \
                          Use start_line/end_line to read a specific range when the file is large."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the workspace"
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed, inclusive). Omit to start from line 1."
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, inclusive). Omit to read to end of file."
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let full_path = sandbox_check(&self.working_dir, &args.path)?;
        let content = std::fs::read_to_string(&full_path)?;

        // Apply line-range slicing when requested.
        let content = if args.start_line.is_some() || args.end_line.is_some() {
            let lines: Vec<&str> = content.lines().collect();
            let total = lines.len();
            // Convert 1-indexed user input to 0-indexed bounds, clamped to file size.
            let start = args.start_line.map(|n| (n as usize).saturating_sub(1)).unwrap_or(0).min(total);
            let end = args.end_line.map(|n| (n as usize).min(total)).unwrap_or(total);
            if start >= end {
                return Ok(format!(
                    "[Empty range: start_line={} end_line={} total_lines={total}]",
                    start + 1,
                    end
                ));
            }
            // Annotate with line numbers so the model knows where it is in the file.
            let annotated: String = lines[start..end]
                .iter()
                .enumerate()
                .map(|(i, line)| format!("{:>5}: {}", start + i + 1, line))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "[Lines {}-{} of {total} total]\n{annotated}",
                start + 1,
                end
            )
        } else {
            content
        };

        // Truncate large files to keep context small for local models.
        if self.max_output_chars > 0 && content.len() > self.max_output_chars {
            let lines: Vec<&str> = content.lines().collect();
            let total_lines = lines.len();
            let mut truncated = String::with_capacity(self.max_output_chars + 100);
            let mut chars = 0;
            let mut included_lines = 0;
            for line in &lines {
                let line_len = line.len() + 1; // +1 for newline
                if chars + line_len > self.max_output_chars {
                    break;
                }
                truncated.push_str(line);
                truncated.push('\n');
                chars += line_len;
                included_lines += 1;
            }
            let remaining = total_lines - included_lines;
            truncated.push_str(&format!(
                "\n[...{remaining} more lines truncated. Use start_line/end_line to read a specific range.]\n"
            ));
            Ok(truncated)
        } else {
            Ok(content)
        }
    }
}

// ---------------------------------------------------------------------------
// WriteFileTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct WriteFileArgs {
    /// Relative path within the workspace.
    pub path: String,
    /// The content to write.
    pub content: String,
}

/// Write content to a file in the worktree. Creates parent directories.
pub struct WriteFileTool {
    pub working_dir: PathBuf,
}

impl WriteFileTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for WriteFileTool {
    const NAME: &'static str = "write_file";
    type Error = ToolError;
    type Args = WriteFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "write_file".into(),
            description:
                "Write content to a file in the workspace. Creates parent directories if needed."
                    .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the workspace"
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let full_path = sandbox_check(&self.working_dir, &args.path)?;
        if !args.path.contains('/') {
            tracing::warn!(
                path = %args.path,
                "write_file: path has no directory component, writing to worktree root"
            );
        }

        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Heuristic: detect double-JSON-encoded content from local models.
        // Qwen3.5 sometimes wraps the entire file in quotes with
        // escaped characters. After rig's JSON parse the content arrives as
        // a valid Rust string that starts/ends with `"` and contains escape
        // sequences like `\n`, `\t`, `\"`. Only unescape if escape sequences
        // are present — otherwise `"hello"` would be silently stripped to `hello`.
        let content = if args.content.starts_with('"')
            && args.content.ends_with('"')
            && has_json_escape_sequences(&args.content)
        {
            match serde_json::from_str::<String>(&args.content) {
                Ok(unescaped) if unescaped != args.content => {
                    tracing::warn!(
                        path = %args.path,
                        orig_len = args.content.len(),
                        unescaped_len = unescaped.len(),
                        "write_file: detected double-escaped content, unescaping"
                    );
                    unescaped
                }
                _ => args.content,
            }
        } else {
            args.content
        };

        // Blast-radius guard: reject writes that shrink an existing file by >50%.
        // Prevents catastrophic file corruption from truncated model output
        // (e.g., job 1653: 500-line pipeline.rs replaced with 1 line of garbage).
        // Runs AFTER unescape so the size comparison uses the final content length.
        // Uses fs::metadata to avoid reading the entire file and to work with non-UTF8 files.
        if full_path.exists() {
            let existing_len = std::fs::metadata(&full_path)?.len() as usize;
            let new_len = content.len();
            if existing_len > 100 && new_len < existing_len / 2 {
                tracing::error!(
                    path = %args.path,
                    existing_bytes = existing_len,
                    new_bytes = new_len,
                    shrink_pct = (100 - (new_len * 100 / existing_len)),
                    "write_file: BLAST-RADIUS GUARD triggered — refusing destructive write"
                );
                return Err(ToolError::Policy(format!(
                    "Blast-radius guard: refusing to write {new_len} bytes to {} \
                     (currently {existing_len} bytes, {:.0}% shrink). \
                     Use edit_file for targeted changes instead of rewriting the entire file.",
                    args.path,
                    100.0 - (new_len as f64 * 100.0 / existing_len as f64)
                )));
            }
        }

        let bytes = content.len();
        std::fs::write(&full_path, &content)?;
        Ok(format!("Wrote {bytes} bytes to {}", args.path))
    }
}

/// Check if a quoted string contains JSON escape sequences (e.g., `\n`, `\t`, `\"`).
/// Used to distinguish double-encoded content from legitimate quoted text.
fn has_json_escape_sequences(s: &str) -> bool {
    // Look inside the outer quotes for backslash-escaped characters
    let inner = &s[1..s.len() - 1];
    inner.contains("\\n")
        || inner.contains("\\t")
        || inner.contains("\\r")
        || inner.contains("\\\"")
        || inner.contains("\\\\")
        || inner.contains("\\u")
}

// ── Unit tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_read_file_args_deserialize() {
        let json = r#"{"path": "src/main.rs"}"#;
        let args: ReadFileArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.path, "src/main.rs");
    }

    #[test]
    fn test_write_file_args_deserialize() {
        let json = r#"{"path": "src/lib.rs", "content": "fn main() {}"}"#;
        let args: WriteFileArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.path, "src/lib.rs");
        assert_eq!(args.content, "fn main() {}");
    }

    #[test]
    fn test_write_file_args_missing_content_fails() {
        let json = r#"{"path": "src/lib.rs"}"#;
        let result = serde_json::from_str::<WriteFileArgs>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_has_json_escape_sequences_with_newlines() {
        assert!(has_json_escape_sequences(r#""hello\nworld""#));
    }

    #[test]
    fn test_has_json_escape_sequences_with_tabs() {
        assert!(has_json_escape_sequences(r#""hello\tworld""#));
    }

    #[test]
    fn test_has_json_escape_sequences_plain_quoted() {
        // No escape sequences — just a quoted string
        assert!(!has_json_escape_sequences(r#""hello world""#));
    }

    #[test]
    fn test_has_json_escape_sequences_with_unicode() {
        assert!(has_json_escape_sequences(r#""hello\u0020world""#));
    }

    #[test]
    fn test_list_files_args_deserialize() {
        let json = r#"{"path": ""}"#;
        let args: ListFilesArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.path, "");
    }

    #[test]
    fn test_list_files_args_with_subdir() {
        let json = r#"{"path": "src/tools"}"#;
        let args: ListFilesArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.path, "src/tools");
    }
}

// ---------------------------------------------------------------------------
// ListFilesTool
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ListFilesArgs {
    /// Relative directory path within the workspace (empty string = root).
    pub path: String,
}

/// List files and directories at a path within the worktree.
pub struct ListFilesTool {
    pub working_dir: PathBuf,
}

impl ListFilesTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for ListFilesTool {
    const NAME: &'static str = "list_files";
    type Error = ToolError;
    type Args = ListFilesArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "list_files".into(),
            description: "List files and directories at a path in the workspace.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative directory path (empty string for workspace root)"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let dir = if args.path.is_empty() {
            self.working_dir.clone()
        } else {
            sandbox_check(&self.working_dir, &args.path)?
        };

        let mut entries = Vec::new();
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().to_string();
            // Skip hidden and target dirs
            if name.starts_with('.') || name == "target" {
                continue;
            }
            let kind = if entry.file_type()?.is_dir() {
                "dir"
            } else {
                "file"
            };
            entries.push(format!("{kind}\t{name}"));
        }
        entries.sort();
        Ok(entries.join("\n"))
    }
}
