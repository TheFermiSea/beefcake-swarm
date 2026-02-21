//! Rig tool for querying the NotebookLM knowledge base.
//!
//! Allows the Manager agent to query project knowledge on-demand
//! during task delegation. NOT sandboxed to worktree — knowledge
//! queries don't read/write worktree files.

use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use tracing::warn;

use super::ToolError;
use crate::notebook_bridge::KnowledgeBase;

#[derive(Deserialize)]
pub struct QueryNotebookArgs {
    /// The notebook role to query: "project_brain", "debugging_kb", "codebase", "security".
    pub role: String,
    /// The natural language question to ask.
    pub question: String,
}

/// Query the project knowledge base (NotebookLM) by role.
pub struct QueryNotebookTool {
    knowledge_base: Arc<dyn KnowledgeBase>,
}

impl QueryNotebookTool {
    pub fn new(knowledge_base: Arc<dyn KnowledgeBase>) -> Self {
        Self { knowledge_base }
    }
}

impl Tool for QueryNotebookTool {
    const NAME: &'static str = "query_notebook";
    type Error = ToolError;
    type Args = QueryNotebookArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "query_notebook".into(),
            description: "Query the project knowledge base (NotebookLM). \
                          Roles: \"project_brain\" (architecture decisions, context), \
                          \"debugging_kb\" (error patterns, known fixes), \
                          \"codebase\" (code structure, understanding), \
                          \"security\" (compliance rules, best practices). \
                          Use BEFORE delegating complex or unfamiliar tasks."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "role": {
                        "type": "string",
                        "enum": ["project_brain", "debugging_kb", "codebase", "security", "research"],
                        "description": "Which knowledge notebook to query"
                    },
                    "question": {
                        "type": "string",
                        "description": "Natural language question to ask the knowledge base"
                    }
                },
                "required": ["role", "question"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let response = match self.knowledge_base.query(&args.role, &args.question) {
            Ok(r) => r,
            Err(err) => {
                warn!(
                    role = %args.role,
                    error = %err,
                    "Knowledge base query failed; degrading gracefully"
                );
                return Ok(format!(
                    "Knowledge base query failed for role '{}' \u{2014} proceeding without KB context. Error: {}",
                    args.role, err
                ));
            }
        };

        if response.is_empty() {
            Ok(format!(
                "No knowledge available for role '{}'. The notebook may not be configured or seeded yet.",
                args.role
            ))
        } else {
            Ok(format!(
                "## Knowledge Base Response ({})\n\n{}",
                args.role, response
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Result;

    /// A mock `KnowledgeBase` that always returns an error.
    struct AlwaysFailKnowledgeBase;

    impl KnowledgeBase for AlwaysFailKnowledgeBase {
        fn query(&self, _role: &str, _question: &str) -> Result<String> {
            anyhow::bail!("simulated NotebookLM connection failure")
        }

        fn add_source_text(&self, _role: &str, _title: &str, _content: &str) -> Result<()> {
            Ok(())
        }

        fn add_source_file(&self, _role: &str, _file_path: &str) -> Result<()> {
            Ok(())
        }

        fn is_available(&self) -> bool {
            false
        }
    }

    /// When the knowledge base returns an error, `call()` must NOT propagate
    /// the error — it must return `Ok(...)` with a degraded message.
    #[tokio::test]
    async fn call_degrades_gracefully_on_kb_error() {
        let kb: Arc<dyn KnowledgeBase> = Arc::new(AlwaysFailKnowledgeBase);
        let tool = QueryNotebookTool::new(kb);

        let args = QueryNotebookArgs {
            role: "project_brain".to_string(),
            question: "What is the architecture?".to_string(),
        };

        let result = tool.call(args).await;

        // Must be Ok — errors must NOT propagate to the caller.
        assert!(result.is_ok(), "expected Ok but got: {:?}", result);

        let message = result.unwrap();
        assert!(
            message.contains("Knowledge base query failed"),
            "degraded message should contain 'Knowledge base query failed', got: {message:?}"
        );
        assert!(
            message.contains("project_brain"),
            "degraded message should echo the role, got: {message:?}"
        );
    }

    /// When the knowledge base returns an empty string, `call()` returns a
    /// "no knowledge available" message (not an error).
    #[tokio::test]
    async fn call_returns_no_knowledge_message_on_empty_response() {
        use crate::notebook_bridge::NoOpKnowledgeBase;

        let kb: Arc<dyn KnowledgeBase> = Arc::new(NoOpKnowledgeBase);
        let tool = QueryNotebookTool::new(kb);

        let args = QueryNotebookArgs {
            role: "debugging_kb".to_string(),
            question: "How to fix E0382?".to_string(),
        };

        let result = tool.call(args).await;
        assert!(result.is_ok());
        let message = result.unwrap();
        assert!(
            message.contains("No knowledge available"),
            "expected 'No knowledge available' message, got: {message:?}"
        );
    }
}
