//! Coder agent builders: Rust specialist and general-purpose.

use std::path::Path;

use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::providers::openai;

use crate::prompts;
use crate::tools::exec_tool::RunCommandTool;
use crate::tools::fs_tools::{ListFilesTool, ReadFileTool, WriteFileTool};
use crate::tools::patch_tool::EditFileTool;
use crate::tools::proxy_wrappers::{
    ProxyEditFile, ProxyListFiles, ProxyReadFile, ProxyRunCommand, ProxyWriteFile,
};

/// Type alias for agents built from OpenAI-compatible endpoints.
pub type OaiAgent = Agent<openai::completion::CompletionModel>;

const DEFAULT_WORKER_MAX_TURNS: usize = 15;
const DEFAULT_REASONING_MAX_TURNS: usize = 20;

fn worker_max_turns() -> usize {
    std::env::var("SWARM_WORKER_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_WORKER_MAX_TURNS)
}

fn reasoning_max_turns() -> usize {
    std::env::var("SWARM_REASONING_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_REASONING_MAX_TURNS)
}

/// Build the Rust specialist coder (HydraCoder).
///
/// Tools: read_file, write_file, run_command.
/// Used for borrow checker, lifetime, trait bound, and type mismatch errors.
pub fn build_rust_coder(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
) -> OaiAgent {
    build_rust_coder_named(client, model, wt_path, "rust_coder", false)
}

/// Build the Rust specialist coder with a custom agent name.
///
/// When `proxy_tools` is true, registers tools with `proxy_` prefix for
/// CLIAPIProxy compatibility (the proxy prepends `proxy_` to tool names in
/// responses, so tools must already have the prefix to match).
pub fn build_rust_coder_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
) -> OaiAgent {
    let builder = client
        .agent(model)
        .name(name)
        .description("Rust specialist for borrow checker, lifetimes, trait bounds, type errors")
        .preamble(prompts::RUST_CODER_PREAMBLE)
        .temperature(0.2);

    if proxy_tools {
        builder
            .tool(ProxyReadFile(ReadFileTool::new(wt_path)))
            .tool(ProxyWriteFile(WriteFileTool::new(wt_path)))
            .tool(ProxyEditFile(EditFileTool::new(wt_path)))
            .tool(ProxyRunCommand(RunCommandTool::new(wt_path)))
            .default_max_turns(worker_max_turns())
            .build()
    } else {
        builder
            .tool(ReadFileTool::new(wt_path))
            .tool(WriteFileTool::new(wt_path))
            .tool(EditFileTool::new(wt_path))
            .tool(RunCommandTool::new(wt_path))
            .default_max_turns(worker_max_turns())
            .build()
    }
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
    build_reasoning_worker_named(client, model, wt_path, "reasoning_worker", false)
}

/// Build the reasoning worker with a custom agent name.
pub fn build_reasoning_worker_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
) -> OaiAgent {
    let builder = client
        .agent(model)
        .name(name)
        .description("Deep reasoning specialist for complex Rust architecture and debugging")
        .preamble(prompts::REASONING_WORKER_PREAMBLE)
        .temperature(0.2);

    if proxy_tools {
        builder
            .tool(ProxyReadFile(ReadFileTool::new(wt_path)))
            .tool(ProxyWriteFile(WriteFileTool::new(wt_path)))
            .tool(ProxyEditFile(EditFileTool::new(wt_path)))
            .tool(ProxyListFiles(ListFilesTool::new(wt_path)))
            .tool(ProxyRunCommand(RunCommandTool::new(wt_path)))
            .default_max_turns(reasoning_max_turns())
            .build()
    } else {
        builder
            .tool(ReadFileTool::new(wt_path))
            .tool(WriteFileTool::new(wt_path))
            .tool(EditFileTool::new(wt_path))
            .tool(ListFilesTool::new(wt_path))
            .tool(RunCommandTool::new(wt_path))
            .default_max_turns(reasoning_max_turns())
            .build()
    }
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
    build_general_coder_named(client, model, wt_path, "general_coder", false)
}

/// Build the general-purpose coder with a custom agent name.
///
/// When `proxy_tools` is true, registers tools with `proxy_` prefix for
/// CLIAPIProxy compatibility.
pub fn build_general_coder_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
) -> OaiAgent {
    let builder = client
        .agent(model)
        .name(name)
        .description("General coding agent for multi-file scaffolding and cross-cutting changes")
        .preamble(prompts::GENERAL_CODER_PREAMBLE)
        .temperature(0.3);

    if proxy_tools {
        builder
            .tool(ProxyReadFile(ReadFileTool::new(wt_path)))
            .tool(ProxyWriteFile(WriteFileTool::new(wt_path)))
            .tool(ProxyEditFile(EditFileTool::new(wt_path)))
            .tool(ProxyListFiles(ListFilesTool::new(wt_path)))
            .tool(ProxyRunCommand(RunCommandTool::new(wt_path)))
            .default_max_turns(worker_max_turns())
            .build()
    } else {
        builder
            .tool(ReadFileTool::new(wt_path))
            .tool(WriteFileTool::new(wt_path))
            .tool(EditFileTool::new(wt_path))
            .tool(ListFilesTool::new(wt_path))
            .tool(RunCommandTool::new(wt_path))
            .default_max_turns(worker_max_turns())
            .build()
    }
}
