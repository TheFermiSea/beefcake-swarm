//! Structural code search using ast-grep (sg).

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{run_command_with_timeout, ToolError};

const DEFAULT_TIMEOUT_SECS: u64 = 30;

#[derive(Deserialize)]
pub struct AstGrepInput {
    /// The AST pattern to search for (e.g. `fn $NAME($$$ARGS) { $$$BODY }`).
    pub pattern: String,
    /// Language to parse the code as (e.g., `rust`, `python`, `typescript`). Optional but recommended.
    pub language: Option<String>,
}

/// Search source code structurally using ast-grep, sandboxed to the worktree.
pub struct AstGrepTool {
    pub working_dir: PathBuf,
}

impl AstGrepTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for AstGrepTool {
    const NAME: &'static str = "ast_grep";
    type Error = ToolError;
    type Args = AstGrepInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "ast_grep".into(),
            description: "Search for code structurally using ast-grep (sg). Matches AST nodes rather than text. \
                          Use $VAR to match single syntax nodes, and $$$VARS to match multiple nodes/arguments."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The AST pattern to search for (e.g., 'fn $NAME($$$ARGS) { $$$BODY }' or 'unwrap()')"
                    },
                    "language": {
                        "type": "string",
                        "description": "The language of the code (e.g., 'rust', 'python', 'typescript')"
                    }
                },
                "required": ["pattern"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut cmd_args = vec!["run", "--pattern", &args.pattern];

        if let Some(ref l) = args.language {
            cmd_args.push("--language");
            cmd_args.push(l);
        }

        // Run sg. Output is matching code segments.
        let output =
            run_command_with_timeout("sg", &cmd_args, &self.working_dir, DEFAULT_TIMEOUT_SECS)
                .await?;

        if output.trim().is_empty() {
            Ok("No structural matches found".to_string())
        } else {
            Ok(output)
        }
    }
}
