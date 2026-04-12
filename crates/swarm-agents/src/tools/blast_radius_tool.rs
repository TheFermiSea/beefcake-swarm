//! Blast-radius analysis tools powered by code-review-graph MCP sidecar.
//!
//! Provides three tools for dependency-aware code understanding:
//! - `blast_radius`: Given changed files, find all affected files via graph traversal
//! - `review_context`: Token-optimized summary of affected code for a set of changes
//! - `graph_query`: Direct callers/callees/dependencies lookup for a symbol
//!
//! These tools shell out to the `code-review-graph` CLI which maintains a persistent
//! SQLite dependency graph built from tree-sitter AST parsing (19 languages).
//! The graph survives across iterations and updates incrementally (<2s).
//!
//! Install: `pip install code-review-graph && code-review-graph build`
//! Reference: docs/research/sota-multi-agent-harness-2026-04.md

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{run_command_with_timeout, ToolError};

const CRG_TIMEOUT_SECS: u64 = 30;
const CRG_BIN: &str = "code-review-graph";

/// Parse a comma-separated file list, trimming whitespace and filtering empties.
fn parse_file_list(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// Run a code-review-graph subcommand with standard timeout and error handling.
async fn run_crg(
    subcommand: &str,
    flags: &[&str],
    positional: &[String],
    working_dir: &Path,
) -> Result<String, ToolError> {
    let mut cmd_args: Vec<&str> = vec![subcommand];
    cmd_args.extend_from_slice(flags);
    // `--` terminates option parsing so file paths starting with `-` aren't
    // misinterpreted as flags.
    cmd_args.push("--");
    let positional_refs: Vec<&str> = positional.iter().map(|s| s.as_str()).collect();
    cmd_args.extend_from_slice(&positional_refs);

    let output =
        run_command_with_timeout(CRG_BIN, &cmd_args, working_dir, CRG_TIMEOUT_SECS).await?;

    if output.is_empty() {
        return Ok("No dependency graph found. Run `code-review-graph build` first.".into());
    }
    // run_command_with_timeout returns Ok even on non-zero exit (with stderr in output).
    // Check for the common error prefix to surface failures clearly.
    if output.starts_with("Exit code:") {
        return Err(ToolError::External(format!(
            "code-review-graph {subcommand} failed: {output}"
        )));
    }
    Ok(output)
}

// ---------------------------------------------------------------------------
// Tool 1: Blast Radius
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct BlastRadiusInput {
    /// Comma-separated list of changed file paths (relative to repo root).
    pub changed_files: String,
    /// Maximum traversal depth (default 3).
    pub max_depth: Option<u32>,
    /// Maximum number of affected files to return (default 30).
    pub max_files: Option<usize>,
}

/// Find all files affected by a set of changes via dependency graph traversal.
pub struct BlastRadiusTool {
    pub working_dir: PathBuf,
}

impl BlastRadiusTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for BlastRadiusTool {
    const NAME: &'static str = "blast_radius";
    type Error = ToolError;
    type Args = BlastRadiusInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "blast_radius".into(),
            description: "Find all files affected by code changes. Given a list of changed files, \
                          traces dependencies (calls, imports, inheritance, tests) to identify every \
                          file that could be impacted. Use this BEFORE making edits to understand \
                          the full scope of changes needed."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "changed_files": {
                        "type": "string",
                        "description": "Comma-separated list of changed file paths relative to repo root"
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum traversal depth for dependency tracing (default 3)"
                    },
                    "max_files": {
                        "type": "integer",
                        "description": "Maximum number of affected files to return (default 30)"
                    }
                },
                "required": ["changed_files"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let depth = args.max_depth.unwrap_or(3).to_string();
        let max_files = args.max_files.unwrap_or(30).to_string();
        let files = parse_file_list(&args.changed_files);
        if files.is_empty() {
            return Ok("No files specified.".into());
        }
        run_crg(
            "impact",
            &["--depth", &depth, "--max-files", &max_files, "--json"],
            &files,
            &self.working_dir,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Tool 2: Review Context
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct ReviewContextInput {
    /// Comma-separated list of changed file paths.
    pub changed_files: String,
    /// Maximum token budget for the context summary (default 8000).
    pub max_tokens: Option<usize>,
}

/// Get a token-optimized summary of code affected by changes.
pub struct ReviewContextTool {
    pub working_dir: PathBuf,
}

impl ReviewContextTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for ReviewContextTool {
    const NAME: &'static str = "review_context";
    type Error = ToolError;
    type Args = ReviewContextInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "review_context".into(),
            description: "Get a token-optimized summary of code affected by changes. \
                          Returns function signatures, call relationships, and test coverage \
                          for the blast radius of changed files."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "changed_files": {
                        "type": "string",
                        "description": "Comma-separated list of changed file paths"
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum token budget for the summary (default 8000)"
                    }
                },
                "required": ["changed_files"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let max_tokens = args.max_tokens.unwrap_or(8000).to_string();
        let files = parse_file_list(&args.changed_files);
        if files.is_empty() {
            return Ok("No files specified.".into());
        }
        run_crg(
            "review",
            &["--max-tokens", &max_tokens, "--json"],
            &files,
            &self.working_dir,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Tool 3: Graph Query
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct GraphQueryInput {
    /// The symbol name or pattern to query.
    pub target: String,
    /// Query kind: "callers", "callees", "dependencies", "dependents", "tests".
    pub kind: Option<String>,
    /// Maximum traversal depth (default 2).
    pub depth: Option<u32>,
}

/// Query the code dependency graph for a specific symbol.
pub struct GraphQueryTool {
    pub working_dir: PathBuf,
}

impl GraphQueryTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for GraphQueryTool {
    const NAME: &'static str = "graph_query";
    type Error = ToolError;
    type Args = GraphQueryInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "graph_query".into(),
            description: "Query the code dependency graph for a symbol. Find callers, callees, \
                          dependencies, dependents, or tests for any function, struct, or trait."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The symbol name or pattern to query"
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["callers", "callees", "dependencies", "dependents", "tests"],
                        "description": "Query kind (default: 'callers')"
                    },
                    "depth": {
                        "type": "integer",
                        "description": "Maximum traversal depth (default 2)"
                    }
                },
                "required": ["target"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let kind = args.kind.as_deref().unwrap_or("callers");
        let depth = args.depth.unwrap_or(2).to_string();
        run_crg(
            "query",
            &["--kind", kind, "--depth", &depth, "--json"],
            &[args.target],
            &self.working_dir,
        )
        .await
    }
}

// ---------------------------------------------------------------------------
// Utility: check if code-review-graph is available
// ---------------------------------------------------------------------------

/// Check if code-review-graph CLI is installed and a graph exists for this repo.
pub async fn is_graph_available(working_dir: &Path) -> bool {
    if !working_dir.join(".code-review-graph").exists() {
        return false;
    }
    // run_command_with_timeout returns Ok even on non-zero exit.
    // Check that the output contains a version string (not an error message).
    match run_command_with_timeout(CRG_BIN, &["--version"], working_dir, 5).await {
        Ok(output) => !output.starts_with("Exit code:"),
        Err(_) => false,
    }
}
