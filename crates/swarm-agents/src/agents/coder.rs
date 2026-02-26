//! Coder agent builders: Rust specialist and general-purpose.

use std::path::Path;

use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::providers::openai;

use crate::prompts;
use crate::tools::bundles::{self, WorkerRole};

/// Type alias for agents built from OpenAI-compatible endpoints.
pub type OaiAgent = Agent<openai::completion::CompletionModel>;

const DEFAULT_WORKER_MAX_TURNS: usize = 15;
const DEFAULT_REASONING_MAX_TURNS: usize = 20;

/// Default temperature for worker agents.
///
/// HydraCoder (30B MoE) drops from 100% to ~40% tool-call reliability when
/// temperature rises from 0.0 to 0.3 (empirically measured). Cloud models
/// (Opus 4.6) handle higher temperatures fine, so this is overridable.
const DEFAULT_WORKER_TEMPERATURE: f64 = 0.0;

pub fn worker_temperature() -> f64 {
    std::env::var("SWARM_WORKER_TEMPERATURE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|v| (0.0..=2.0).contains(v))
        .unwrap_or(DEFAULT_WORKER_TEMPERATURE)
}

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

/// Build the Rust specialist coder (Qwen3.5-Implementer).
///
/// Tools: read_file, write_file, edit_file, run_command (no list_files).
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
    client
        .agent(model)
        .name(name)
        .description("Rust specialist for borrow checker, lifetimes, trait bounds, type errors")
        .preamble(prompts::RUST_CODER_PREAMBLE)
        .temperature(worker_temperature())
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::RustSpecialist,
            proxy_tools,
        ))
        .default_max_turns(worker_max_turns())
        .build()
}

/// Build the reasoning worker (Qwen3.5-Architect).
///
/// Tools: read_file, write_file, edit_file, list_files, run_command.
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
    client
        .agent(model)
        .name(name)
        .description("Deep reasoning specialist for complex Rust architecture and debugging")
        .preamble(prompts::REASONING_WORKER_PREAMBLE)
        .temperature(worker_temperature())
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::General,
            proxy_tools,
        ))
        .default_max_turns(reasoning_max_turns())
        .build()
}

/// Build the general-purpose coder (Qwen3.5-Implementer).
///
/// Tools: read_file, write_file, edit_file, list_files, run_command.
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
    client
        .agent(model)
        .name(name)
        .description("General coding agent for multi-file scaffolding and cross-cutting changes")
        .preamble(prompts::GENERAL_CODER_PREAMBLE)
        .temperature(worker_temperature())
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::General,
            proxy_tools,
        ))
        .default_max_turns(worker_max_turns())
        .build()
}
