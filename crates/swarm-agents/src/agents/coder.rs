//! Coder agent builders: Rust specialist and general-purpose.

use std::path::Path;

use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::providers::openai;

use crate::prompts;
use crate::tools::exec_tool::RunCommandTool;
use crate::tools::fs_tools::{ListFilesTool, ReadFileTool, WriteFileTool};
use crate::tools::patch_tool::EditFileTool;

/// Type alias for agents built from OpenAI-compatible endpoints.
pub type OaiAgent = Agent<openai::completion::CompletionModel>;

/// Build the Rust specialist coder (strand-rust-coder-14B).
///
/// Tools: read_file, write_file, run_command.
/// Used for borrow checker, lifetime, trait bound, and type mismatch errors.
pub fn build_rust_coder(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
) -> OaiAgent {
    build_rust_coder_named(client, model, wt_path, "rust_coder")
}

/// Build the Rust specialist coder with a custom agent name.
///
/// Used to create proxy-prefixed variants for the cloud manager
/// (CLIAPIProxy prepends `proxy_` to tool names, so we pre-prefix).
pub fn build_rust_coder_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
) -> OaiAgent {
    client
        .agent(model)
        .name(name)
        .description("Rust specialist for borrow checker, lifetimes, trait bounds, type errors")
        .preamble(prompts::RUST_CODER_PREAMBLE)
        .temperature(0.2)
        .tool(ReadFileTool::new(wt_path))
        .tool(WriteFileTool::new(wt_path))
        .tool(EditFileTool::new(wt_path))
        .tool(RunCommandTool::new(wt_path))
        .default_max_turns(50)
        .build()
}

/// Build the reasoning worker (OR1-Behemoth 72B).
///
/// Tools: read_file, write_file, list_files, run_command.
/// Used by the cloud manager for deep analysis, repair plans, and complex fixes.
pub fn build_reasoning_worker(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
) -> OaiAgent {
    build_reasoning_worker_named(client, model, wt_path, "reasoning_worker")
}

/// Build the reasoning worker with a custom agent name.
pub fn build_reasoning_worker_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
) -> OaiAgent {
    client
        .agent(model)
        .name(name)
        .description("Deep reasoning specialist for complex Rust architecture and debugging")
        .preamble(prompts::REASONING_WORKER_PREAMBLE)
        .temperature(0.2)
        .tool(ReadFileTool::new(wt_path))
        .tool(WriteFileTool::new(wt_path))
        .tool(EditFileTool::new(wt_path))
        .tool(ListFilesTool::new(wt_path))
        .tool(RunCommandTool::new(wt_path))
        .default_max_turns(50)
        .build()
}

/// Build the general-purpose coder (Qwen3-Coder-Next).
///
/// Tools: read_file, write_file, list_files, run_command.
/// Used for multi-file changes, scaffolding, and cross-cutting refactors.
pub fn build_general_coder(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
) -> OaiAgent {
    build_general_coder_named(client, model, wt_path, "general_coder")
}

/// Build the general-purpose coder with a custom agent name.
pub fn build_general_coder_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
) -> OaiAgent {
    client
        .agent(model)
        .name(name)
        .description("General coding agent for multi-file scaffolding and cross-cutting changes")
        .preamble(prompts::GENERAL_CODER_PREAMBLE)
        .temperature(0.3)
        .tool(ReadFileTool::new(wt_path))
        .tool(WriteFileTool::new(wt_path))
        .tool(EditFileTool::new(wt_path))
        .tool(ListFilesTool::new(wt_path))
        .tool(RunCommandTool::new(wt_path))
        .default_max_turns(50)
        .build()
}
