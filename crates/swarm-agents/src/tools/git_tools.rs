//! Git awareness tools for agent situational awareness.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{run_command_with_timeout, ToolError};
use crate::git_ops::{filter_meaningful_diff_output, filter_meaningful_status};

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
        let output = filter_meaningful_diff_output(&output, name_only);

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
        let output = filter_meaningful_status(&output);

        if output.trim().is_empty() {
            Ok("No changes (working tree clean)".to_string())
        } else {
            Ok(output)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::process::Command;
    use tempfile::tempdir;

    fn setup_git_repo(dir: &Path) {
        Command::new("git")
            .arg("init")
            .current_dir(dir)
            .output()
            .expect("Failed to initialize git repository");

        Command::new("git")
            .args(["config", "user.name", "Test User"])
            .current_dir(dir)
            .output()
            .expect("Failed to set git user name");

        Command::new("git")
            .args(["config", "user.email", "test@example.com"])
            .current_dir(dir)
            .output()
            .expect("Failed to set git user email");

        // Initial commit so HEAD exists
        fs::write(dir.join("initial.txt"), "initial content").unwrap();
        Command::new("git")
            .args(["add", "initial.txt"])
            .current_dir(dir)
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "Initial commit"])
            .current_dir(dir)
            .output()
            .unwrap();
    }

    #[tokio::test]
    async fn test_get_diff_no_changes() {
        let dir = tempdir().unwrap();
        setup_git_repo(dir.path());

        let tool = GetDiffTool::new(dir.path());

        // There's only one commit, HEAD~1 won't exist. We should test diffing against HEAD
        let args_head = GetDiffArgs {
            base_ref: Some("HEAD".to_string()),
            name_only: None,
        };

        let result = tool.call(args_head).await.expect("Tool call failed");
        assert_eq!(result, "No changes");
    }

    #[tokio::test]
    async fn test_get_diff_with_changes() {
        let dir = tempdir().unwrap();
        setup_git_repo(dir.path());

        fs::write(dir.path().join("test_file.txt"), "hello world").unwrap();
        Command::new("git")
            .args(["add", "test_file.txt"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "Second commit"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let tool = GetDiffTool::new(dir.path());

        let args = GetDiffArgs {
            base_ref: Some("HEAD~1".to_string()),
            name_only: None,
        };

        let result = tool.call(args).await.expect("Tool call failed");

        assert!(result.contains("test_file.txt"));
    }

    #[tokio::test]
    async fn test_get_diff_name_only() {
        let dir = tempdir().unwrap();
        setup_git_repo(dir.path());

        fs::write(dir.path().join("test_file_name_only.txt"), "hello world").unwrap();
        Command::new("git")
            .args(["add", "test_file_name_only.txt"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "Second commit"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let tool = GetDiffTool::new(dir.path());

        let args = GetDiffArgs {
            base_ref: Some("HEAD~1".to_string()),
            name_only: Some(true),
        };
        let result = tool.call(args).await.expect("Tool call failed");
        // Output should just be the filename
        assert_eq!(result.trim(), "test_file_name_only.txt");
    }
}
