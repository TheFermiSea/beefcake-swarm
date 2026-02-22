//! Specialist agent builders: planner and fixer.
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

use super::coder::OaiAgent;

const DEFAULT_PLANNER_MAX_TURNS: usize = 10;
const DEFAULT_FIXER_MAX_TURNS: usize = 15;

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
    let mut builder = client
        .agent(model)
        .name(name)
        .description(
            "Planning specialist. Analyzes errors and produces structured JSON repair plans.",
        )
        .preamble(prompts::PLANNER_PREAMBLE)
        .temperature(0.2)
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::Planner,
            proxy_tools,
        ))
        .default_max_turns(planner_max_turns());

    // Attach GBNF grammar for structured plan output when enabled.
    if let Some(params) =
        crate::grammars::params_if_enabled(crate::grammars::Grammar::PlannerOutput)
    {
        builder = builder.additional_params(params);
    }

    builder.build()
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
    client
        .agent(model)
        .name(name)
        .description(
            "Implementation specialist. Follows structured plans to write targeted code fixes.",
        )
        .preamble(prompts::FIXER_PREAMBLE)
        .temperature(0.2)
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::General,
            proxy_tools,
        ))
        .default_max_turns(fixer_max_turns())
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
