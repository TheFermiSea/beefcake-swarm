//! Manager/orchestrator agent.
//!
//! When cloud is available: backed by Opus 4.6 / G3-Pro with local workers as tools.
//! Fallback: backed by Qwen3.5-Architect (local reasoning) with coders as tools.
//!
//! The Manager gets worker agents as tools (agent-as-tool pattern) plus
//! deterministic tools (verifier, read_file, list_files, query_notebook).

use std::path::Path;
use std::sync::Arc;

use rig::client::CompletionClient;
use rig::providers::openai;

use crate::notebook_bridge::KnowledgeBase;
use crate::prompts;
use crate::tools::bundles;

use super::coder::OaiAgent;

const DEFAULT_MANAGER_MAX_TURNS: usize = 20;

fn manager_max_turns() -> usize {
    std::env::var("SWARM_MANAGER_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_MANAGER_MAX_TURNS)
}

/// Bundled workers and tools for building a Manager agent.
///
/// Avoids passing 8+ individual arguments to the builder functions.
pub struct ManagerWorkers {
    pub rust_coder: OaiAgent,
    pub general_coder: OaiAgent,
    pub reviewer: OaiAgent,
    /// Planning specialist — produces structured repair plans (read-only tools).
    pub planner: OaiAgent,
    /// Implementation specialist — follows plans with targeted edits.
    pub fixer: OaiAgent,
    /// Qwen3.5-Architect reasoning worker (cloud manager only).
    pub reasoning_worker: Option<OaiAgent>,
    /// Optional knowledge base for the query_notebook tool.
    pub notebook_bridge: Option<Arc<dyn KnowledgeBase>>,
}

/// Build the cloud-backed Manager with reasoning_worker and coders as tools.
///
/// Cloud model (Opus 4.6 / G3-Pro) manages local workers:
/// - reasoning_worker (Qwen3.5-Architect): deep analysis, repair plans
/// - rust_coder (Qwen3.5-Implementer): fast Rust fixes
/// - general_coder (Qwen3.5-Implementer): multi-file scaffolding
/// - reviewer: blind code review
pub fn build_cloud_manager(
    client: &openai::CompletionsClient,
    model: &str,
    workers: ManagerWorkers,
    wt_path: &Path,
    verifier_packages: &[String],
) -> OaiAgent {
    let mut builder = client
        .agent(model)
        .name("manager")
        .description("Cloud-backed orchestrator that delegates to local HPC model workers")
        .preamble(prompts::CLOUD_MANAGER_PREAMBLE)
        .temperature(0.3)
        // Agent-as-Tool: specialists
        .tool(workers.planner)
        .tool(workers.fixer)
        // Agent-as-Tool: workers
        .tool(workers.rust_coder)
        .tool(workers.general_coder)
        .tool(workers.reviewer);

    // Reasoning worker only present in cloud manager
    if let Some(rw) = workers.reasoning_worker {
        builder = builder.tool(rw);
    }

    // Deterministic tools — proxy-prefixed for CLIAPIProxy compatibility.
    builder = builder.tools(bundles::manager_tools(wt_path, verifier_packages, true));

    // Knowledge base tool (optional — gracefully absent if not configured)
    let kb_tools = bundles::notebook_tool(workers.notebook_bridge, true);
    if !kb_tools.is_empty() {
        builder = builder.tools(kb_tools);
    }

    builder.default_max_turns(manager_max_turns()).build()
}

/// Build the local-only Manager (Qwen3.5-Architect fallback when cloud unavailable).
///
/// Workers are coders only (no reasoning_worker — Qwen3.5-Architect IS the manager).
pub fn build_local_manager(
    client: &openai::CompletionsClient,
    model: &str,
    workers: ManagerWorkers,
    wt_path: &Path,
    verifier_packages: &[String],
) -> OaiAgent {
    let mut builder = client
        .agent(model)
        .name("manager")
        .description("Orchestrator that decomposes tasks and delegates to specialized workers")
        .preamble(prompts::LOCAL_MANAGER_PREAMBLE)
        .temperature(0.3)
        // Agent-as-Tool: specialists
        .tool(workers.planner)
        .tool(workers.fixer)
        // Agent-as-Tool: workers
        .tool(workers.rust_coder)
        .tool(workers.general_coder)
        .tool(workers.reviewer)
        // Deterministic tools — no proxy prefix for local models
        .tools(bundles::manager_tools(wt_path, verifier_packages, false));

    // Knowledge base tool (optional)
    let kb_tools = bundles::notebook_tool(workers.notebook_bridge, false);
    if !kb_tools.is_empty() {
        builder = builder.tools(kb_tools);
    }

    builder.default_max_turns(manager_max_turns()).build()
}
