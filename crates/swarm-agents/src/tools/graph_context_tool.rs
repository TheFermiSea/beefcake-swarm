//! Graph context tool — exposes SemanticCodeGraph to worker agents.
//!
//! Allows agents to query the dependency graph for callers, callees,
//! implementors, and dependencies of a target symbol.

use std::path::PathBuf;
use std::sync::Arc;

use coordination::context_packer::SemanticCodeGraph;
use coordination::reviewer_tools::graph_rag::QueryKind;
use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::ToolError;

#[derive(Deserialize)]
pub struct GraphContextInput {
    /// The symbol name or substring to query (e.g. "WorkPacket", "format_task_prompt").
    pub target: String,
    /// Number of hops to traverse (1-5, default 2).
    pub hops: Option<u32>,
    /// Query direction: "callers", "callees", "implementors", or "dependencies".
    pub kind: Option<String>,
}

/// Query the semantic code graph for dependency context around a symbol.
pub struct GraphContextTool {
    pub working_dir: PathBuf,
    pub graph: Arc<SemanticCodeGraph>,
}

impl GraphContextTool {
    pub fn new(working_dir: &std::path::Path, graph: Arc<SemanticCodeGraph>) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
            graph,
        }
    }
}

fn parse_query_kind(s: &str) -> QueryKind {
    match s.to_lowercase().as_str() {
        "callers" => QueryKind::Callers,
        "callees" => QueryKind::Callees,
        "implementors" => QueryKind::Implementors,
        "dependencies" => QueryKind::Dependencies,
        "dependents" => QueryKind::Dependents,
        _ => QueryKind::Callees,
    }
}

impl Tool for GraphContextTool {
    const NAME: &'static str = "graph_context";
    type Error = ToolError;
    type Args = GraphContextInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "graph_context".into(),
            description: "Query the semantic code dependency graph. Returns callers, callees, \
                          implementors, or dependencies of a target symbol within a configurable \
                          hop distance. Useful for understanding how code connects before making changes."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "Symbol name or substring to query (e.g. 'WorkPacket', 'format_task_prompt')"
                    },
                    "hops": {
                        "type": "integer",
                        "description": "Number of graph hops to traverse (1-5, default 2)"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["callers", "callees", "implementors", "dependencies"],
                        "description": "Query direction (default: callees)"
                    }
                },
                "required": ["target"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let hops = args.hops.unwrap_or(2).clamp(1, 5);
        let kind = args
            .kind
            .as_deref()
            .map(parse_query_kind)
            .unwrap_or(QueryKind::Callees);

        let result = self.graph.get_subgraph_context(&args.target, hops, kind);

        if let Some(ref err) = result.error {
            return Ok(format!("Graph query error: {}", err));
        }

        let mut output = String::new();
        output.push_str(&format!(
            "Graph context for '{}' ({} nodes, {} edges, depth={}, {}ms)\n\n",
            args.target,
            result.nodes.len(),
            result.edges.len(),
            result.depth_reached,
            result.execution_ms
        ));

        if !result.nodes.is_empty() {
            output.push_str("Nodes:\n");
            for node in &result.nodes {
                let indent = "  ".repeat(node.depth as usize);
                output.push_str(&format!(
                    "{}[{}] {} ({}:{})\n",
                    indent, node.symbol_kind, node.symbol, node.file, node.line
                ));
            }
        }

        if !result.edges.is_empty() {
            output.push_str("\nEdges:\n");
            for edge in &result.edges {
                output.push_str(&format!(
                    "  {} --{}-> {}\n",
                    edge.from, edge.relation, edge.to
                ));
            }
        }

        if result.truncated {
            output.push_str(&format!(
                "\n(truncated: showing {} of {} total results)\n",
                result.nodes.len(),
                result.total_results
            ));
        }

        Ok(output)
    }
}
