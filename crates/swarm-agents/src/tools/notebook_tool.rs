//! Rig tool for querying the NotebookLM knowledge base.
//!
//! Allows the Manager agent to query project knowledge on-demand
//! during task delegation. NOT sandboxed to worktree — knowledge
//! queries don't read/write worktree files.

use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

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
        let response = self
            .knowledge_base
            .query(&args.role, &args.question)
            .map_err(|e| ToolError::Notebook(e.to_string()))?;

        if response.is_empty() {
            Ok(format!(
                "No knowledge available for role '{}'. The notebook may not be configured or seeded yet.",
                args.role
            ))
        } else {
            Ok(format!("## Knowledge Base Response ({})\n\n{}", args.role, response))
        }
    }
}
