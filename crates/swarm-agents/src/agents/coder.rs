//! Coder agent builders: Rust specialist and general-purpose.

use std::path::Path;

use rig::agent::Agent;
use rig::client::CompletionClient;
use rig::completion::message::ToolChoice;
use rig::providers::openai;
use serde_json::Value;

use crate::prompts;
use crate::tools::bundles::{self, WorkerRole};

/// Type alias for agents built from OpenAI-compatible endpoints.
pub type OaiAgent = Agent<openai::completion::CompletionModel>;

const DEFAULT_WORKER_MAX_TURNS: usize = 10;
const DEFAULT_REASONING_MAX_TURNS: usize = 20;

/// Default temperature for worker agents.
///
/// 0.05: slight jitter above 0.0 to escape deterministic text-analysis loops.
/// At temp=0.0 the same bad prompt reliably produces the same text-only
/// response (33% failure rate observed in dogfood run 4). At 0.05 the model
/// picks tool calls ~90%+ of the time without losing reliability.
/// HydraCoder (30B MoE) drops from 100% to ~40% tool-call reliability when
/// temperature rises from 0.0 to 0.3 (empirically measured). Cloud models
/// (Opus 4.6) handle higher temperatures fine, so this is overridable.
const DEFAULT_WORKER_TEMPERATURE: f64 = 0.05;

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

/// Tool choice for worker agents.
///
/// Default: `Required` — forces every turn to produce a tool call. This
/// prevents HydraCoder (30B MoE) from dropping into text-analysis mode
/// (6-10K chars of prose instead of calling edit_file). With `Required`,
/// each turn is productive: read_file → edit_file → ... until max_turns.
///
/// Override with `SWARM_WORKER_TOOL_CHOICE=auto` to allow text-only responses.
fn worker_tool_choice() -> ToolChoice {
    match std::env::var("SWARM_WORKER_TOOL_CHOICE")
        .unwrap_or_default()
        .to_lowercase()
        .as_str()
    {
        "auto" => ToolChoice::Auto,
        "none" => ToolChoice::None,
        _ => ToolChoice::Required,
    }
}

/// Additional sampling parameters for local Qwen3.5 workers.
///
/// `repetition_penalty: 1.1` — reduces repetitive tool-call loops where the
/// model calls the same tool with identical arguments across turns.
/// `presence_penalty: 0.1` — mild flat penalty for tokens already generated,
/// encouraging more varied output without strongly suppressing needed repetition.
///
/// Both are supported by llama-server's `/v1/chat/completions` endpoint and
/// OpenAI-compatible APIs. Passed via `additional_params` which Rig flattens
/// into the request body.
/// Default max_tokens for local workers.
///
/// 16384 tokens is generous for tool-call responses but prevents unbounded
/// `<think>` block generation that consumed the entire 30-minute timeout
/// during the 2026-03-04 dogfood run (8949+ thinking tokens, 0 tool calls).
/// Combined with `/no_think` in prompts, this is a safety net — the `/no_think`
/// tag should prevent thinking entirely, but max_tokens caps output even if
/// the chat template ignores `/no_think` for some reason.
const DEFAULT_WORKER_MAX_TOKENS: u64 = 16384;

pub fn worker_sampling_params() -> Value {
    let max_tokens = std::env::var("SWARM_WORKER_MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_WORKER_MAX_TOKENS);
    serde_json::json!({
        "repetition_penalty": 1.1,
        "presence_penalty": 0.1,
        "max_tokens": max_tokens
    })
}

/// Merge two JSON objects. Keys in `extra` overwrite keys in `base`.
///
/// Used to combine sampling params with grammar params when both are needed
/// (e.g., planner uses GBNF grammar + anti-repetition penalties).
pub fn merge_params(base: Value, extra: Value) -> Value {
    match (base, extra) {
        (Value::Object(mut a), Value::Object(b)) => {
            for (k, v) in b {
                a.insert(k, v);
            }
            Value::Object(a)
        }
        (_, extra) => extra,
    }
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
        .tool_choice(worker_tool_choice())
        .additional_params(worker_sampling_params())
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
        .tool_choice(worker_tool_choice())
        .additional_params(worker_sampling_params())
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
        .tool_choice(worker_tool_choice())
        .additional_params(worker_sampling_params())
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::General,
            proxy_tools,
        ))
        .default_max_turns(worker_max_turns())
        .build()
}
