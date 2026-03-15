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

        // Basic validation that output is valid JSON (if not, it might be an error message we should return)
        if !output.trim().starts_with('[') {
            return Ok(output);
        }

        let Ok(results) = serde_json::from_str::<Vec<serde_json::Value>>(&output) else {
            return Ok(output);
        };

        if results.is_empty() {
            return Ok("No semantic matches found".to_string());
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
                "--- MATCH (score: {score:.3}) ---\nFile: {file_path}\n{content}\n"
            ));
        }

        Ok(formatted.join("\n"))
    }
}
