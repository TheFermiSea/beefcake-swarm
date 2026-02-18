//! Manager/orchestrator agent.
//!
//! When cloud is available: backed by Opus 4.6 / G3-Pro with local workers as tools.
//! Fallback: backed by OR1-Behemoth 72B (local reasoning) with coders as tools.
//!
//! The Manager gets worker agents as tools (agent-as-tool pattern) plus
//! deterministic tools (verifier, read_file, list_files, query_notebook).

use std::path::Path;
use std::sync::Arc;

use rig::client::CompletionClient;
use rig::providers::openai;

use crate::notebook_bridge::KnowledgeBase;
use crate::prompts;
use crate::tools::fs_tools::{ListFilesTool, ReadFileTool};
use crate::tools::notebook_tool::QueryNotebookTool;
use crate::tools::proxy_wrappers::{
    ProxyListFiles, ProxyQueryNotebook, ProxyReadFile, ProxyRunVerifier,
};
use crate::tools::verifier_tool::RunVerifierTool;

use super::coder::OaiAgent;

/// Bundled workers and tools for building a Manager agent.
///
/// Avoids passing 8+ individual arguments to the builder functions.
pub struct ManagerWorkers {
    pub rust_coder: OaiAgent,
    pub general_coder: OaiAgent,
    pub reviewer: OaiAgent,
    /// OR1-Behemoth reasoning worker (cloud manager only).
    pub reasoning_worker: Option<OaiAgent>,
    /// Optional knowledge base for the query_notebook tool.
    pub notebook_bridge: Option<Arc<dyn KnowledgeBase>>,
}

/// Build the cloud-backed Manager with reasoning_worker and coders as tools.
///
/// Cloud model (Opus 4.6 / G3-Pro) manages local workers:
/// - reasoning_worker (OR1-Behemoth): deep analysis, repair plans
/// - rust_coder (strand-14B): fast Rust fixes
/// - general_coder (Qwen3-Coder-Next): multi-file scaffolding
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
        // Agent-as-Tool: workers
        .tool(workers.rust_coder)
        .tool(workers.general_coder)
        .tool(workers.reviewer);

    // Reasoning worker only present in cloud manager
    if let Some(rw) = workers.reasoning_worker {
        builder = builder.tool(rw);
    }

    // Deterministic tools — proxy-prefixed for CLIAPIProxy compatibility.
    // The proxy prepends `proxy_` to tool names; pre-prefixing prevents mismatch.
    builder = builder
        .tool(ProxyRunVerifier(
            RunVerifierTool::new(wt_path).with_packages(verifier_packages.to_vec()),
        ))
        .tool(ProxyReadFile(ReadFileTool::new(wt_path)))
        .tool(ProxyListFiles(ListFilesTool::new(wt_path)));

    // Knowledge base tool (optional — gracefully absent if not configured)
    if let Some(kb) = workers.notebook_bridge {
        builder = builder.tool(ProxyQueryNotebook(QueryNotebookTool::new(kb)));
    }

    builder.default_max_turns(20).build()
}

/// Build the local-only Manager (OR1-Behemoth fallback when cloud unavailable).
///
/// Workers are coders only (no reasoning_worker — OR1 IS the manager).
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
        // Agent-as-Tool: workers
        .tool(workers.rust_coder)
        .tool(workers.general_coder)
        .tool(workers.reviewer)
        // Deterministic tools
        .tool(RunVerifierTool::new(wt_path).with_packages(verifier_packages.to_vec()))
        .tool(ReadFileTool::new(wt_path))
        .tool(ListFilesTool::new(wt_path));

    // Knowledge base tool (optional)
    if let Some(kb) = workers.notebook_bridge {
        builder = builder.tool(QueryNotebookTool::new(kb));
    }

    builder.default_max_turns(20).build()
}
