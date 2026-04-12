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
///
/// Uses code-review-graph's SQLite-backed dependency graph to perform BFS
/// from changed files through call, import, inheritance, and test edges.
/// Returns the minimal set of files that could be impacted.
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
                        "description": "Comma-separated list of changed file paths relative to repo root (e.g. 'src/config.rs,src/driver.rs')"
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

        // code-review-graph CLI: `code-review-graph impact <files> --depth N --max-files N --json`
        let mut cmd_args = vec![
            "impact",
            "--depth",
            &depth,
            "--max-files",
            &max_files,
            "--json",
        ];

        // Split changed_files and add each as a positional arg
        let files: Vec<&str> = args.changed_files.split(',').map(|s| s.trim()).collect();
        for f in &files {
            cmd_args.push(f);
        }

        let output = run_command_with_timeout(
            "code-review-graph",
            &cmd_args,
            &self.working_dir,
            CRG_TIMEOUT_SECS,
        )
        .await?;

        if output.is_empty() {
            return Ok("No dependency graph found. Run `code-review-graph build` first.".into());
        }

        Ok(output)
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
///
/// Combines blast-radius analysis with smart context extraction to produce
/// a compact summary of affected functions, their signatures, and relationships.
/// Designed to fit within LLM context windows while preserving essential information.
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
                          for the blast radius of changed files. Use this to understand what \
                          your changes affect without reading entire files."
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

        let mut cmd_args = vec!["review", "--max-tokens", &max_tokens, "--json"];

        let files: Vec<&str> = args.changed_files.split(',').map(|s| s.trim()).collect();
        for f in &files {
            cmd_args.push(f);
        }

        let output = run_command_with_timeout(
            "code-review-graph",
            &cmd_args,
            &self.working_dir,
            CRG_TIMEOUT_SECS,
        )
        .await?;

        if output.is_empty() {
            return Ok("No dependency graph found. Run `code-review-graph build` first.".into());
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tool 3: Graph Query
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct GraphQueryInput {
    /// The symbol name or pattern to query (e.g. "SwarmConfig", "handle_implementing").
    pub target: String,
    /// Query kind: "callers", "callees", "dependencies", "dependents", "tests".
    pub kind: Option<String>,
    /// Maximum traversal depth (default 2).
    pub depth: Option<u32>,
}

/// Query the code dependency graph for a specific symbol.
///
/// Returns callers, callees, dependencies, or test coverage for a target symbol.
/// Uses the persistent SQLite graph built by code-review-graph.
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
                          dependencies, dependents, or tests for any function, struct, or trait. \
                          Use this to understand how a symbol is used across the codebase."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "target": {
                        "type": "string",
                        "description": "The symbol name or pattern to query (e.g. 'SwarmConfig', 'handle_implementing')"
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

        let cmd_args = vec![
            "query",
            "--kind",
            kind,
            "--depth",
            &depth,
            "--json",
            &args.target,
        ];

        let output = run_command_with_timeout(
            "code-review-graph",
            &cmd_args,
            &self.working_dir,
            CRG_TIMEOUT_SECS,
        )
        .await?;

        if output.is_empty() {
            return Ok("No dependency graph found. Run `code-review-graph build` first.".into());
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Utility: check if code-review-graph is available
// ---------------------------------------------------------------------------

/// Check if code-review-graph CLI is installed and a graph exists for this repo.
pub async fn is_graph_available(working_dir: &Path) -> bool {
    // Check if the .code-review-graph directory exists (graph has been built)
    let graph_dir = working_dir.join(".code-review-graph");
    if !graph_dir.exists() {
        return false;
    }

    // Verify the CLI is installed
    run_command_with_timeout("code-review-graph", &["--version"], working_dir, 5)
        .await
        .is_ok()
}
