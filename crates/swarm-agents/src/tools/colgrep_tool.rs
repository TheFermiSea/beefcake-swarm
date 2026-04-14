//! Semantic code search tool using colgrep.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{run_command_with_timeout, sandbox_check, ToolError};

const DEFAULT_TIMEOUT_SECS: u64 = 60; // ColGrep can take a bit longer
const DEFAULT_MAX_RESULTS: usize = 15;

#[derive(Deserialize)]
pub struct ColGrepInput {
    /// The natural language query or concept to search for.
    pub query: String,
    /// Optional regex pattern to pre-filter before semantic ranking.
    pub regex_filter: Option<String>,
    /// Optional glob pattern to restrict which files are searched (e.g. "*.rs").
    pub include_pattern: Option<String>,
    /// Maximum number of matches to return (default 15).
    pub max_results: Option<usize>,
}

/// Search source code by meaning using colgrep, sandboxed to the worktree.
pub struct ColGrepTool {
    pub working_dir: PathBuf,
}

impl ColGrepTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }

    fn parse_output(&self, output: &str) -> String {
        // Basic validation that output is valid JSON (if not, it might be an error message we should return)
        if !output.trim().starts_with('[') {
            return output.to_string();
        }

        let Ok(results) = serde_json::from_str::<Vec<serde_json::Value>>(output) else {
            return output.to_string();
        };

        if results.is_empty() {
            return "No semantic matches found".to_string();
        }

        let mut formatted = Vec::new();
        for res in results {
            let file_path = res
                .pointer("/unit/file")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let content = res
                .pointer("/unit/content")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let score = res.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);

            if !file_path.is_empty() {
                // Ensure we don't return files outside the sandbox
                if sandbox_check(&self.working_dir, file_path).is_err() {
                    continue;
                }
            }

            formatted.push(format!(
                "--- MATCH (score: {score:.3}) ---
File: {file_path}
{content}
"
            ));
        }

        formatted.join(
            "
",
        )
    }
}

impl Tool for ColGrepTool {
    const NAME: &'static str = "colgrep";
    type Error = ToolError;
    type Args = ColGrepInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "colgrep".into(),
            description: "Semantic code search. Find code by meaning, intent, or concept instead of exact text matches. \
                          Returns highly relevant function or class snippets. Useful when you don't know the exact keywords."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The natural language query or concept to search for (e.g., 'database connection pooling')"
                    },
                    "regex_filter": {
                        "type": "string",
                        "description": "Optional text pattern to pre-filter results before semantic ranking"
                    },
                    "include_pattern": {
                        "type": "string",
                        "description": "Optional glob to restrict which files are searched (e.g. '*.rs')"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return (default 15)"
                    }
                },
                "required": ["query"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let max = args.max_results.unwrap_or(DEFAULT_MAX_RESULTS);

        let max_str = max.to_string();
        let mut cmd_args = vec!["--json", "-k", &max_str];

        if let Some(ref r) = args.regex_filter {
            cmd_args.push("-e");
            cmd_args.push(r);
        }

        if let Some(ref i) = args.include_pattern {
            cmd_args.push("--include");
            cmd_args.push(i);
        }

        cmd_args.push(&args.query);

        // Run colgrep. Output is JSON array of matches or error text
        let output = run_command_with_timeout(
            "colgrep",
            &cmd_args,
            &self.working_dir,
            DEFAULT_TIMEOUT_SECS,
        )
        .await?;

        Ok(self.parse_output(&output))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_definition() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = ColGrepTool::new(&repo_root);
        let def = tool.definition("".to_string()).await;

        assert_eq!(def.name, "colgrep");
        assert!(def.description.contains("Semantic code search"));
    }

    #[test]
    fn test_parse_output_valid_matches() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = ColGrepTool::new(&repo_root);

        // Mock a valid JSON response from colgrep
        let json_output = r#"[
            {
                "score": 0.95,
                "unit": {
                    "file": "Cargo.toml",
                    "content": "some text"
                }
            }
        ]"#;

        let parsed = tool.parse_output(json_output);
        assert!(parsed.contains("--- MATCH (score: 0.950) ---"));
        assert!(parsed.contains("File: Cargo.toml"));
        assert!(parsed.contains("some text"));
    }

    #[test]
    fn test_parse_output_no_matches() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = ColGrepTool::new(&repo_root);

        let json_output = "[]";
        let parsed = tool.parse_output(json_output);

        assert_eq!(parsed, "No semantic matches found");
    }

    #[test]
    fn test_parse_output_invalid_json() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = ColGrepTool::new(&repo_root);

        let text_output = "Error: colgrep not found";
        let parsed = tool.parse_output(text_output);

        // Plain text error should be returned as-is
        assert_eq!(parsed, "Error: colgrep not found");
    }

    #[test]
    fn test_parse_output_escapes_sandbox() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = ColGrepTool::new(&repo_root);

        // Mock a match trying to access a file outside the workspace
        let json_output = r#"[
            {
                "score": 0.88,
                "unit": {
                    "file": "../../../../../etc/passwd",
                    "content": "root:x:0:0"
                }
            }
        ]"#;

        let parsed = tool.parse_output(json_output);

        // The malicious file should be skipped, resulting in no formatted output
        // (Since the array had 1 element which was skipped, join("\n") is empty)
        assert_eq!(parsed, "");
    }
}
