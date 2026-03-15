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
    /// Mutually exclusive with `rule`.
    pub pattern: Option<String>,
    /// Language to parse the code as (e.g., `rust`, `python`, `typescript`). Optional but recommended.
    /// Only used with `pattern` mode.
    pub language: Option<String>,
    /// Path to a YAML rule file (relative to worktree root), e.g.
    /// `rules/ast-grep/rules/unwrap-in-production.yml`.
    /// Mutually exclusive with `pattern`.
    pub rule: Option<String>,
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
                          Two modes: (1) inline pattern via `pattern` field — use $VAR to match single nodes, \
                          $$$VARS for multiple; (2) named rule file via `rule` field — path to a YAML rule \
                          in rules/ast-grep/rules/ (e.g. 'rules/ast-grep/rules/unwrap-in-production.yml'). \
                          Available rules: unwrap-in-production, silent-error-discard, tool-impl-audit, \
                          missing-sandbox-check, blocking-in-async, hardcoded-endpoints, \
                          clone-on-string-literal, todo-fixme-comments, expect-missing-context."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Inline AST pattern to search for (e.g., '$EXPR.unwrap()' or 'fn $NAME($$$ARGS) -> $RET { $$$BODY }'). Use this for ad-hoc searches."
                    },
                    "language": {
                        "type": "string",
                        "description": "Language for inline pattern mode (e.g., 'rust', 'python', 'typescript')"
                    },
                    "rule": {
                        "type": "string",
                        "description": "Path to a YAML rule file relative to the worktree root (e.g., 'rules/ast-grep/rules/unwrap-in-production.yml'). Use for named, documented audit rules."
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let output = match (&args.rule, &args.pattern) {
            (Some(rule_path), _) => {
                // Rule file mode: `sg scan --rule <path> .`
                // sandbox_check not needed — rule path is in-repo, not user-supplied file target
                let cmd_args = vec!["scan", "--rule", rule_path.as_str(), "."];
                run_command_with_timeout("sg", &cmd_args, &self.working_dir, DEFAULT_TIMEOUT_SECS)
                    .await?
            }
            (None, Some(pattern)) => {
                // Inline pattern mode: `sg run --pattern <pat> [--language <lang>] .`
                let mut cmd_args = vec!["run", "--pattern", pattern.as_str()];
                if let Some(ref l) = args.language {
                    cmd_args.push("--language");
                    cmd_args.push(l.as_str());
                }
                // Search current directory (worktree root)
                cmd_args.push(".");
                run_command_with_timeout("sg", &cmd_args, &self.working_dir, DEFAULT_TIMEOUT_SECS)
                    .await?
            }
            (None, None) => {
                return Err(ToolError::Policy(
                    "ast_grep requires either `pattern` or `rule` — provide one".into(),
                ));
            }
        };

        if output.trim().is_empty() {
            Ok("No structural matches found".to_string())
        } else {
            Ok(output)
        }
    }
}
