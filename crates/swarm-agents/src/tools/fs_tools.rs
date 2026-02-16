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
}

/// Read a file from the worktree. Path must stay within the sandbox.
pub struct ReadFileTool {
    pub working_dir: PathBuf,
}

impl ReadFileTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
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
            description: "Read the contents of a file in the workspace.".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the workspace"
                    }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let full_path = sandbox_check(&self.working_dir, &args.path)?;
        let content = std::fs::read_to_string(&full_path)?;
        Ok(content)
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
        if let Some(parent) = full_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Heuristic: detect double-JSON-encoded content from local models.
        // If content is wrapped in quotes and contains literal \n but no real
        // newlines, it was likely JSON-stringified by the model.
        let content = if args.content.starts_with('"')
            && args.content.ends_with('"')
            && args.content.contains("\\n")
            && !args.content[1..args.content.len() - 1].contains('\n')
        {
            match serde_json::from_str::<String>(&args.content) {
                Ok(unescaped) => {
                    tracing::warn!("write_file: detected double-escaped content, unescaping");
                    unescaped
                }
                Err(_) => args.content,
            }
        } else {
            args.content
        };

        let bytes = content.len();
        std::fs::write(&full_path, &content)?;
        Ok(format!("Wrote {bytes} bytes to {}", args.path))
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
