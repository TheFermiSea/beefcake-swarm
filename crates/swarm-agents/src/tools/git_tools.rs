//! Git awareness tools for agent situational awareness.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{run_command_with_timeout, ToolError};

// ── GetDiffTool ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct GetDiffArgs {
    /// Base ref to diff against (default: HEAD~1).
    pub base_ref: Option<String>,
    /// Show only file names (like --name-only). Default false.
    pub name_only: Option<bool>,
}

/// Show `git diff` output so agents know what they've changed.
pub struct GetDiffTool {
    pub working_dir: PathBuf,
}

impl GetDiffTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for GetDiffTool {
    const NAME: &'static str = "get_diff";
    type Error = ToolError;
    type Args = GetDiffArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "get_diff".into(),
            description: "Show git diff output. Use to see what has changed in the worktree. \
                          Defaults to diff against HEAD~1. Use name_only=true for just filenames."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "base_ref": {
                        "type": "string",
                        "description": "Base ref to diff against (default: HEAD~1)"
                    },
                    "name_only": {
                        "type": "boolean",
                        "description": "Show only changed file names (default: false)"
                    }
                },
                "required": []
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let base = args.base_ref.as_deref().unwrap_or("HEAD~1");
        let name_only = args.name_only.unwrap_or(false);

        let stat_arg = if name_only { "--name-only" } else { "--stat" };
        let args = ["diff", base, stat_arg];

        let output = run_command_with_timeout("git", &args, &self.working_dir, 30).await?;

        if output.trim().is_empty() {
            Ok("No changes".to_string())
        } else {
            Ok(output)
        }
    }
}

// ── ListChangedFilesTool ─────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ListChangedFilesArgs {}

/// Show `git status --short` for situational awareness.
pub struct ListChangedFilesTool {
    pub working_dir: PathBuf,
}

impl ListChangedFilesTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for ListChangedFilesTool {
    const NAME: &'static str = "list_changed_files";
    type Error = ToolError;
    type Args = ListChangedFilesArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "list_changed_files".into(),
            description: "List files with uncommitted changes (git status --short). \
                          Shows modified, added, deleted, and untracked files."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let output =
            run_command_with_timeout("git", &["status", "--short"], &self.working_dir, 10).await?;

        if output.trim().is_empty() {
            Ok("No changes (working tree clean)".to_string())
        } else {
            Ok(output)
        }
    }
}
