Phase 1: Episode Architecture (Memory Management)

Logical Justification (Slate)

Models suffer from "Context Rot" in the Dumb Zone. Streaming raw tool outputs (e.g., full cargo check outputs or sed diffs) into the Orchestrator's context window degrades its ability to plan. We must implement "Episodes." Threads execute in isolation, and when finished, they yield a compressed EpisodeSummary.

Agent Instructions

Step 1: Define Episode Structures

Open coordination/src/memory/types.rs (create if it doesn't exist, or add to store.rs). Define the EpisodeSummary.

// In coordination/src/memory/store.rs or types.rs
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeSummary {
pub thread_id: String,
pub task_objective: String,
pub tactical_actions_taken: Vec<String>,
pub outcome: ExecutionOutcome,
pub discovered_knowledge: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionOutcome {
Success(String),
Blocked(String), // Requires Kernel intervention
Failed(String),
}

Step 2: Implement the Summarizer Loop in Swarm Agents

Open crates/swarm-agents/src/modes/agentic.rs.
Currently, the local worker probably loops and returns a giant string of context. You need to modify the execution loop so that after the default_max_turns or tool loop completes, a final "Summarizer" Rig agent compresses the context.

// Target: crates/swarm-agents/src/modes/agentic.rs
use rig::agent::Agent;
use rig::completion::Prompt;
use crate::coordination::memory::store::EpisodeSummary;

/// Compresses a raw thread trace into an EpisodeSummary before returning to Kernel
pub async fn finalize_episode(
summarizer_agent: &Agent<impl rig::completion::CompletionModel>,
raw_trace: &str,
objective: &str,
) -> Result<EpisodeSummary, anyhow::Error> {
let prompt = format!(
"You are an OS Kernel Memory Manager. Compress the following raw process trace into a structured EpisodeSummary JSON.
Objective: {}
Raw Trace: {}",
objective, raw_trace
);

    // We expect the model to output JSON matching the EpisodeSummary schema.
    // Ensure the agent is configured with a JSON extractor or tool.
    let summary: EpisodeSummary = summarizer_agent.prompt(prompt).await?;
    Ok(summary)

}

Task for Coder: 1. Add EpisodeSummary to coordination/src/memory/store.rs. 2. Update the Worker execution loop in crates/swarm-agents/src/modes/runner.rs or agentic.rs to trap the output and pipe it through a summarizer model before yielding back to the orchestrator channel.
