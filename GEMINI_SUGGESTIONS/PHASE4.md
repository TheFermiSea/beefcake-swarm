Phase 4: SOTA Tool Integration (ast-grep & Graph-RAG)Target Files:crates/swarm-agents/src/tools/ast_grep_tool.rs (NEW)crates/swarm-agents/src/tools/graph_rag_tool.rs (NEW)crates/swarm-agents/src/tools/verifier_tool.rs (MODIFIED)crates/swarm-agents/src/agents/reviewer.rs (MODIFIED)To make the swarm truly "better than the sum of its parts" and highly autonomous, the Reviewer needs superhuman code comprehension. We achieve this by wrapping SOTA tools into Rig Tool implementations, allowing the models to actively query the AST and semantic graph during the debate loop.1. Structural Analysis: AstGrepToolStandard grep is insufficient for LLMs because they struggle with regex escaping and whitespace. ast-grep (sg) allows the agents to query the Abstract Syntax Tree directly.File: crates/swarm-agents/src/tools/ast_grep_tool.rsuse rig::tool::Tool;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{debug, error, instrument};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AstGrepError {
    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),
    #[error("Parse error: {0}")]
    ParseError(String),
}

#[derive(Deserialize, Serialize)]
pub struct AstGrepArgs {
    /// The structural pattern to search for (e.g., 'unwrap()', 'pub fn $NAME($$$ARGS) { $$$ }')
    pattern: String,
    /// The language to parse (e.g., 'rust', 'python')
    lang: String,
    /// Optional specific file or directory to target
    target_path: Option<String>,
}

#[derive(Deserialize, Serialize)]
pub struct AstGrepTool;

impl AstGrepTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for AstGrepTool {
    const NAME: &'static str = "ast_grep_search";
    type Error = AstGrepError;
    type Args = AstGrepArgs;
    type Output = String;

    #[instrument(skip(self), err)]
    async fn definition(&self, _prompt: String) -> serde_json::Value {
        serde_json::json!({
            "name": Self::NAME,
            "description": "Perform structural search on the AST. Use this to find specific function definitions, unsafe blocks, or anti-patterns without relying on regex.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The AST pattern to search for using `sg` syntax. Example: '$A.unwrap()'"
                    },
                    "lang": {
                        "type": "string",
                        "description": "Target language, typically 'rust'"
                    },
                    "target_path": {
                        "type": "string",
                        "description": "Optional specific file to search within"
                    }
                },
                "required": ["pattern", "lang"]
            }
        })
    }

    #[instrument(skip(self), err)]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        debug!("Executing ast-grep with pattern: {}", args.pattern);

        let mut cmd = Command::new("sg");
        cmd.arg("run")
           .arg("--pattern").arg(&args.pattern)
           .arg("--lang").arg(&args.lang);

        if let Some(path) = args.target_path {
            cmd.arg(&path);
        }

        let output = cmd.output().await.map_err(|e| {
            error!("Failed to spawn sg process: {}", e);
            AstGrepError::ExecutionFailed(e.to_string())
        })?;

        let result = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        if !output.status.success() {
            // ast-grep returns non-zero if no matches are found, handle gracefully
            if stderr.contains("No files matched") || result.is_empty() {
                return Ok("No structural matches found in the workspace.".to_string());
            }
            return Err(AstGrepError::ExecutionFailed(stderr));
        }

        // Truncate massively long outputs to protect context window
        if result.len() > 8000 {
            Ok(format!("{}...\n[TRUNCATED: Over 8000 bytes. Refine your structural search.]", &result[..8000]))
        } else {
            Ok(result)
        }
    }
}
2. Semantic Context: GraphRagTool (CocoIndex)Based on your indexing/index_flow_v2.py and COCOINDEX_GRAPH_RAG.md, you have a Graph RAG pipeline. We must expose this to the swarm so the Reviewer can ask: "What files depend on the function the Coder just changed?"File: crates/swarm-agents/src/tools/graph_rag_tool.rsuse rig::tool::Tool;
use serde::{Deserialize, Serialize};
use tokio::process::Command;
use tracing::{debug, error, instrument};

#[derive(Deserialize, Serialize)]
pub struct GraphRagArgs {
    /// Natural language or specific node query to search the code graph
    query: String,
    /// Type of query: 'impact_analysis', 'dependency_search', or 'general'
    query_type: String, 
}

#[derive(Deserialize, Serialize)]
pub struct GraphRagTool;

impl GraphRagTool {
    pub fn new() -> Self {
        Self
    }
}

impl Tool for GraphRagTool {
    const NAME: &'static str = "code_graph_rag";
    type Error = String;
    type Args = GraphRagArgs;
    type Output = String;

    #[instrument(skip(self))]
    async fn definition(&self, _prompt: String) -> serde_json::Value {
        serde_json::json!({
            "name": Self::NAME,
            "description": "Query the CocoIndex Graph RAG database to understand code dependencies, call chains, and impact analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The target to investigate, e.g., 'What functions call DebateOrchestrator::run_debate?'"
                    },
                    "query_type": {
                        "type": "string",
                        "enum": ["impact_analysis", "dependency_search", "general"]
                    }
                },
                "required": ["query", "query_type"]
            }
        })
    }

    #[instrument(skip(self), err)]
    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        debug!("Querying Graph RAG: {}", args.query);

        // Execute your local index_flow_v2.py script in query mode
        // Ensure the python environment has cocoindex installed as per your config
        let output = Command::new("python3")
            .arg("indexing/index_flow_v2.py")
            .arg("--query")
            .arg(&args.query)
            .arg("--mode")
            .arg(&args.query_type)
            .output()
            .await
            .map_err(|e| {
                error!("Failed to execute CocoIndex python bridge: {}", e);
                e.to_string()
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("Graph RAG Query failed: {}", stderr);
            return Err(stderr.into_owned());
        }

        Ok(String::from_utf8_lossy(&output.stdout).into_owned())
    }
}
3. Upgrading the Reviewer Agent for Autonomous CorrectionNow that we have these SOTA tools, we integrate them into the Reviewer so it can enforce strict rules during the DebateOrchestrator::run_debate loop (Phase 3).File: crates/swarm-agents/src/agents/reviewer.rs (Modifications)use rig::agent::{Agent, AgentBuilder};
use rig::providers::anthropic::{Client, CompletionModel};

// Import the new tools
use crate::tools::verifier_tool::VerifierTool;
use crate::tools::ast_grep_tool::AstGrepTool;
use crate::tools::graph_rag_tool::GraphRagTool;

pub fn build_reviewer_agent(client: &Client) -> Agent<CompletionModel> {
    let builder = client.agent("claude-3-5-sonnet-20241022")
        .preamble(
            "You are an elite, autonomous Security and Architecture Reviewer in a multi-agent debate loop. \
            The Coder has proposed a patch. Your job is to rigorously break it or approve it. \
            \
            YOUR TOOL ARSENAL: \n\
            1. verifier_tool: Runs cargo clippy/check to ensure it compiles. \n\
            2. ast_grep_search: Use this to scan the Coder's modified files for `unwrap()`, `panic!()`, or `println!()` which violate our strict rules. \n\
            3. code_graph_rag: Use this to check for broken dependencies. If the Coder changed a struct signature, use Graph RAG to find all downstream consumers and verify they were updated. \n\
            \
            PROCESS: \n\
            - You MUST run the verifier_tool first. \n\
            - If it compiles, you MUST use ast_grep_search to check for anti-patterns in the touched files. \n\
            - If the Coder changed a public API, you MUST use code_graph_rag to ensure impact analysis is complete. \n\
            \
            If everything passes, output 'CONSENSUS_REACHED'. Otherwise, output a detailed critique with the exact failed CLI outputs."
        )
        .tool(VerifierTool::new())
        .tool(AstGrepTool::new())
        .tool(GraphRagTool::new());

    builder.build().unwrap_or_else(|e| {
        tracing::error!("FATAL: Failed to build reviewer agent: {}", e);
        std::process::exit(1);
    })
}
How this supercharges Phase 3 (The Debate Loop):When the DebateOrchestrator hands the Coder's proposed patch to the Reviewer:The Coder blindly writes the file (using PatchTool).The Reviewer wakes up, reads its instructions, and autonomously invokes ast_grep_search looking for pattern: "$A.unwrap()".If ast_grep returns a hit, the Reviewer formulates the critique: "You used an unwrap on line 42. Our rules (rules/no-unwrap-in-prod.yml) forbid this. Refactor to return a Result."The Orchestrator passes this exact critique back to the Coder for the next loop iteration.This entirely eliminates the need for human oversight during standard CI/CD failures and enforces your rules/*.yml files programmatically at the AI level.
