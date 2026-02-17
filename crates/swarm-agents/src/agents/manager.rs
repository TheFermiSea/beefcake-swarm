//! Manager/orchestrator agent.
//!
//! When cloud is available: backed by Opus 4.6 / G3-Pro with local workers as tools.
//! Fallback: backed by OR1-Behemoth 72B (local reasoning) with coders as tools.
//!
//! The Manager gets worker agents as tools (agent-as-tool pattern) plus
//! deterministic tools (verifier, read_file, list_files).

use std::path::Path;

use rig::client::CompletionClient;
use rig::providers::openai;

use crate::prompts;
use crate::tools::fs_tools::{ListFilesTool, ReadFileTool};
use crate::tools::verifier_tool::RunVerifierTool;

use super::coder::OaiAgent;

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
    reasoning_worker: OaiAgent,
    rust_coder: OaiAgent,
    general_coder: OaiAgent,
    reviewer: OaiAgent,
    wt_path: &Path,
) -> OaiAgent {
    client
        .agent(model)
        .name("manager")
        .description("Cloud-backed orchestrator that delegates to local HPC model workers")
        .preamble(prompts::CLOUD_MANAGER_PREAMBLE)
        .temperature(0.3)
        // Agent-as-Tool: workers
        .tool(reasoning_worker)
        .tool(rust_coder)
        .tool(general_coder)
        .tool(reviewer)
        // Deterministic tools
        .tool(RunVerifierTool::new(wt_path))
        .tool(ReadFileTool::new(wt_path))
        .tool(ListFilesTool::new(wt_path))
        .default_max_turns(20)
        .build()
}

/// Build the local-only Manager (OR1-Behemoth fallback when cloud unavailable).
///
/// Workers are coders only (no reasoning_worker — OR1 IS the manager).
pub fn build_local_manager(
    client: &openai::CompletionsClient,
    model: &str,
    rust_coder: OaiAgent,
    general_coder: OaiAgent,
    reviewer: OaiAgent,
    wt_path: &Path,
) -> OaiAgent {
    client
        .agent(model)
        .name("manager")
        .description("Orchestrator that decomposes tasks and delegates to specialized workers")
        .preamble(prompts::LOCAL_MANAGER_PREAMBLE)
        .temperature(0.3)
        // Agent-as-Tool: workers
        .tool(rust_coder)
        .tool(general_coder)
        .tool(reviewer)
        // Deterministic tools
        .tool(RunVerifierTool::new(wt_path))
        .tool(ReadFileTool::new(wt_path))
        .tool(ListFilesTool::new(wt_path))
        .default_max_turns(20)
        .build()
}
