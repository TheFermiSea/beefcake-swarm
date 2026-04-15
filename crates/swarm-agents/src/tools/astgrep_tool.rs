//! Structural code search using ast-grep (sg).

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

#[cfg(not(test))]
use super::run_command_with_timeout;
#[cfg(test)]
use tests::mock_run_command_with_timeout as run_command_with_timeout;

use super::ToolError;

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
                // Inline pattern mode: `sg run --pattern <pat> [--lang <lang>] .`
                let mut cmd_args = vec!["run", "--pattern", pattern.as_str()];
                if let Some(ref l) = args.language {
                    cmd_args.push("--lang");
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

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::collections::HashMap;
    use std::sync::Mutex;

    pub static MOCK_CALLS: Lazy<Mutex<HashMap<PathBuf, Vec<Vec<String>>>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));
    pub static MOCK_OUTPUT: Lazy<Mutex<HashMap<PathBuf, String>>> =
        Lazy::new(|| Mutex::new(HashMap::new()));

    pub(crate) async fn mock_run_command_with_timeout(
        program: &str,
        args: &[&str],
        working_dir: &Path,
        _timeout_secs: u64,
    ) -> Result<String, ToolError> {
        let mut call = vec![program.to_string()];
        call.extend(args.iter().map(|s| s.to_string()));
        MOCK_CALLS
            .lock()
            .unwrap()
            .entry(working_dir.to_path_buf())
            .or_default()
            .push(call);

        let output = MOCK_OUTPUT
            .lock()
            .unwrap()
            .get(working_dir)
            .cloned()
            .unwrap_or_default();

        Ok(output)
    }

    fn setup_mock(dir: &Path, output: &str) {
        MOCK_OUTPUT
            .lock()
            .unwrap()
            .insert(dir.to_path_buf(), output.to_string());
        MOCK_CALLS
            .lock()
            .unwrap()
            .insert(dir.to_path_buf(), Vec::new());
    }

    fn get_mock_calls(dir: &Path) -> Vec<Vec<String>> {
        MOCK_CALLS
            .lock()
            .unwrap()
            .get(dir)
            .cloned()
            .unwrap_or_default()
    }

    #[tokio::test]
    async fn test_astgrep_rule_mode() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        setup_mock(path, "match found in rule");

        let tool = AstGrepTool::new(path);
        let result = tool
            .call(AstGrepInput {
                rule: Some("rules/my-rule.yml".to_string()),
                pattern: None,
                language: None,
            })
            .await
            .unwrap();

        assert_eq!(result, "match found in rule");

        let calls = get_mock_calls(path);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0],
            vec!["sg", "scan", "--rule", "rules/my-rule.yml", "."]
        );
    }

    #[tokio::test]
    async fn test_astgrep_pattern_mode_no_language() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        setup_mock(path, "match found for pattern");

        let tool = AstGrepTool::new(path);
        let result = tool
            .call(AstGrepInput {
                rule: None,
                pattern: Some("fn test() {}".to_string()),
                language: None,
            })
            .await
            .unwrap();

        assert_eq!(result, "match found for pattern");

        let calls = get_mock_calls(path);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0],
            vec!["sg", "run", "--pattern", "fn test() {}", "."]
        );
    }

    #[tokio::test]
    async fn test_astgrep_pattern_mode_with_language() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        setup_mock(path, "match found for language pattern");

        let tool = AstGrepTool::new(path);
        let result = tool
            .call(AstGrepInput {
                rule: None,
                pattern: Some("def test(): pass".to_string()),
                language: Some("python".to_string()),
            })
            .await
            .unwrap();

        assert_eq!(result, "match found for language pattern");

        let calls = get_mock_calls(path);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0],
            vec![
                "sg",
                "run",
                "--pattern",
                "def test(): pass",
                "--lang",
                "python",
                "."
            ]
        );
    }

    #[tokio::test]
    async fn test_astgrep_empty_output_returns_no_matches() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();
        setup_mock(path, "   \n  "); // empty/whitespace output

        let tool = AstGrepTool::new(path);
        let result = tool
            .call(AstGrepInput {
                rule: Some("rules/test.yml".to_string()),
                pattern: None,
                language: None,
            })
            .await
            .unwrap();

        assert_eq!(result, "No structural matches found");
    }

    #[tokio::test]
    async fn test_astgrep_missing_both_rule_and_pattern() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path();

        let tool = AstGrepTool::new(path);
        let result = tool
            .call(AstGrepInput {
                rule: None,
                pattern: None,
                language: None,
            })
            .await;

        assert!(result.is_err());
        if let Err(ToolError::Policy(msg)) = result {
            assert!(msg.contains("requires either `pattern` or `rule`"));
        } else {
            panic!("Expected Policy error, got {:?}", result);
        }
    }
}
