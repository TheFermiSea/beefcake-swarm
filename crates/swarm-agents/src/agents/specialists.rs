//! Specialist agent builders: planner, fixer, architect, and editor.
//!
//! These agents are registered as tools on the manager, enabling an explicit
//! delegation protocol:
//!
//!   Manager → planner (produces structured plan) → fixer (implements plan)
//!
//! This separates planning from execution, letting the manager verify plans
//! before committing to implementation.

use std::path::Path;

use rig::client::CompletionClient;
use rig::providers::openai;

use crate::prompts;
use crate::tools::bundles::{self, WorkerRole};

use super::coder::{worker_sampling_params, OaiAgent};

const DEFAULT_PLANNER_MAX_TURNS: usize = 15;
const DEFAULT_FIXER_MAX_TURNS: usize = 25;

fn planner_max_turns() -> usize {
    std::env::var("SWARM_PLANNER_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_PLANNER_MAX_TURNS)
}

fn fixer_max_turns() -> usize {
    std::env::var("SWARM_FIXER_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_FIXER_MAX_TURNS)
}

/// Build the planner specialist.
///
/// Read-only tools: read_file, list_files, run_command (for `cargo check`, `rg`).
/// Produces structured JSON repair/implementation plans — never writes code.
pub fn build_planner(client: &openai::CompletionsClient, model: &str, wt_path: &Path) -> OaiAgent {
    build_planner_named(client, model, wt_path, "planner", false)
}

/// Build the planner specialist with a custom agent name.
pub fn build_planner_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
) -> OaiAgent {
    let sampling = worker_sampling_params();

    client
        .agent(model)
        .name(name)
        .description(
            "Planning specialist. Analyzes errors and produces structured JSON repair plans.",
        )
        .preamble(&prompts::load_prompt(
            "planner",
            wt_path,
            prompts::PLANNER_PREAMBLE,
        ))
        .temperature(0.2)
        .additional_params(sampling)
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::Planner,
            proxy_tools,
            None,
        ))
        .default_max_turns(planner_max_turns())
        .build()
}

/// Build the fixer specialist.
///
/// Full editing tools: read_file, write_file, edit_file, list_files, run_command.
/// Takes a plan from the planner and implements it step by step.
pub fn build_fixer(client: &openai::CompletionsClient, model: &str, wt_path: &Path) -> OaiAgent {
    build_fixer_named(client, model, wt_path, "fixer", false)
}

/// Build the fixer specialist with a custom agent name.
pub fn build_fixer_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
) -> OaiAgent {
    build_fixer_for_language(client, model, wt_path, name, proxy_tools, None)
}

/// Build the fixer specialist with language-aware prompts.
pub fn build_fixer_for_language(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
    language: Option<&str>,
) -> OaiAgent {
    client
        .agent(model)
        .name(name)
        .description(
            "Implementation specialist. Follows structured plans to write targeted code fixes.",
        )
        .preamble(&prompts::build_worker_prompt_for_language(
            &prompts::load_prompt("fixer", wt_path, prompts::FIXER_PREAMBLE),
            language,
        ))
        .temperature(0.2)
        .additional_params(worker_sampling_params())
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::General,
            proxy_tools,
            None,
        ))
        .default_max_turns(fixer_max_turns())
        .build()
}

// ── Architect/Editor Pattern ─────────────────────────────────────────────────

const DEFAULT_ARCHITECT_MAX_TURNS: usize = 20;
const DEFAULT_EDITOR_MAX_TURNS: usize = 15;

fn architect_max_turns() -> usize {
    std::env::var("SWARM_ARCHITECT_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_ARCHITECT_MAX_TURNS)
}

fn editor_max_turns() -> usize {
    std::env::var("SWARM_EDITOR_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_EDITOR_MAX_TURNS)
}

/// Build the Architect specialist.
///
/// Read-only tools: read_file, list_files, run_command, search_code, colgrep, ast_grep.
/// Produces `ArchitectPlan` JSON with exact SEARCH/REPLACE edit blocks.
/// Never writes files — the Editor applies the plan.
pub fn build_architect(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
) -> OaiAgent {
    build_architect_named(client, model, wt_path, "architect", false)
}

/// Build the Architect specialist with a custom agent name.
pub fn build_architect_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
) -> OaiAgent {
    client
        .agent(model)
        .name(name)
        .description(
            "Architect specialist. Reads the codebase and produces an exact SEARCH/REPLACE \
             edit plan in JSON. The plan is then applied mechanically by the Editor agent. \
             Use for complex multi-file changes that require deep codebase understanding.",
        )
        .preamble(&prompts::load_prompt(
            "architect",
            wt_path,
            prompts::ARCHITECT_PREAMBLE,
        ))
        .temperature(0.2)
        .additional_params(worker_sampling_params())
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::Planner, // Read-only tools (same as planner)
            proxy_tools,
            None,
        ))
        .default_max_turns(architect_max_turns())
        .build()
}

/// Build the Editor specialist.
///
/// Full editing tools: read_file, edit_file, write_file.
/// Receives an ArchitectPlan and applies SEARCH/REPLACE blocks mechanically.
/// Does NOT reason about code — just follows the plan.
pub fn build_editor(client: &openai::CompletionsClient, model: &str, wt_path: &Path) -> OaiAgent {
    build_editor_named(client, model, wt_path, "editor", false)
}

/// Build the Editor specialist with a custom agent name.
pub fn build_editor_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
) -> OaiAgent {
    use super::coder::worker_temperature;
    use rig::completion::message::ToolChoice;

    client
        .agent(model)
        .name(name)
        .description(
            "Editor specialist. Applies exact SEARCH/REPLACE edits from an Architect's plan. \
             Give it a plan with file paths and exact code blocks to find/replace. \
             It applies each edit mechanically using edit_file.",
        )
        .preamble(&prompts::load_prompt(
            "editor",
            wt_path,
            prompts::EDITOR_PREAMBLE,
        ))
        .temperature(worker_temperature())
        .tool_choice(ToolChoice::Required) // Force tool calls (no prose-only responses)
        .additional_params(worker_sampling_params())
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::General, // Full edit tools
            proxy_tools,
            None,
        ))
        .default_max_turns(editor_max_turns())
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_max_turns_default() {
        // Don't set env var — should use default
        assert!(planner_max_turns() > 0);
    }

    #[test]
    fn test_fixer_max_turns_default() {
        assert!(fixer_max_turns() > 0);
    }
}
