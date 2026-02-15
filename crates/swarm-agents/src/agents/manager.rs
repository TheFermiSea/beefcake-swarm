//! Manager/orchestrator agent (OR1-Behemoth 72B).
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

/// Build the Manager agent with worker agents as tools.
///
/// Workers are passed by value — `Agent<M>` implements `Tool`,
/// so the Manager can call them via tool-calling.
pub fn build_manager(
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
        .preamble(prompts::MANAGER_PREAMBLE)
        .temperature(0.3)
        // Agent-as-Tool: workers
        .tool(rust_coder)
        .tool(general_coder)
        .tool(reviewer)
        // Deterministic tools
        .tool(RunVerifierTool::new(wt_path))
        .tool(ReadFileTool::new(wt_path))
        .tool(ListFilesTool::new(wt_path))
        .default_max_turns(25)
        .build()
}
