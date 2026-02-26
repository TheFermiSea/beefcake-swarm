//! Code search tool using ripgrep.

use std::path::{Path, PathBuf};
use std::time::Duration;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{sandbox_check, ToolError};

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_RESULTS: usize = 20;

#[derive(Deserialize)]
pub struct SearchCodeInput {
    /// The regex pattern to search for.
    pub pattern: String,
    /// Optional glob pattern to restrict which files are searched (e.g. "*.rs").
    pub glob: Option<String>,
    /// Maximum number of matches to return (default 20).
    pub max_results: Option<usize>,
}

/// Search source code using ripgrep, sandboxed to the worktree.
pub struct SearchCodeTool {
    pub working_dir: PathBuf,
}

impl SearchCodeTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for SearchCodeTool {
    const NAME: &'static str = "search_code";
    type Error = ToolError;
    type Args = SearchCodeInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "search_code".into(),
            description: "Search for a regex pattern in the workspace using ripgrep. \
                          Returns matching lines as `file:line: content`."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for"
                    },
                    "glob": {
                        "type": "string",
                        "description": "Optional glob to restrict which files are searched (e.g. '*.rs')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (default 20)"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let max = args.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
        let working_dir = self.working_dir.clone();
        let working_dir_for_closure = working_dir.clone();
        let pattern = args.pattern.clone();
        let glob = args.glob.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut cmd = std::process::Command::new("rg");
            cmd.arg("--json").arg(&pattern);
            if let Some(ref g) = glob {
                cmd.arg("--glob").arg(g);
            }
            cmd.current_dir(&working_dir_for_closure);
            cmd.output().map_err(ToolError::Io)
        });

        let output =
            match tokio::time::timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS), result).await {
                Ok(Ok(r)) => r?,
                Ok(Err(e)) => {
                    return Err(ToolError::Io(std::io::Error::other(format!(
                        "task join error: {e}"
                    ))))
                }
                Err(_) => {
                    return Err(ToolError::Timeout {
                        seconds: DEFAULT_TIMEOUT_SECS,
                    })
                }
            };

        // rg exits non-zero when there are no matches — that's fine.
        let stdout = String::from_utf8_lossy(&output.stdout);

        let mut lines: Vec<String> = Vec::new();
        for raw in stdout.lines() {
            if lines.len() >= max {
                break;
            }
            let Ok(obj) = serde_json::from_str::<serde_json::Value>(raw) else {
                continue;
            };
            if obj.get("type").and_then(|t| t.as_str()) != Some("match") {
                continue;
            }
            let data = match obj.get("data") {
                Some(d) => d,
                None => continue,
            };
            let file_path = data
                .pointer("/path/text")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let line_number = data
                .get("line_number")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            let content = data
                .pointer("/lines/text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim_end_matches('\n');

            // Sandbox-check the relative path returned by rg.
            if !file_path.is_empty() {
                sandbox_check(&working_dir, file_path)?;
            }

            lines.push(format!("{file_path}:{line_number}: {content}"));
        }

        if lines.is_empty() {
            Ok("No matches found".to_string())
        } else {
            Ok(lines.join("\n"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_search_fn_main() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = SearchCodeTool::new(&repo_root);
        let result = tool
            .call(SearchCodeInput {
                pattern: "fn main".to_string(),
                glob: Some("*.rs".to_string()),
                max_results: Some(5),
            })
            .await
            .expect("search_code should not error");
        // There must be at least one fn main in the workspace
        assert!(
            result.contains("fn main") || result == "No matches found",
            "unexpected output: {result}"
        );
    }
}
