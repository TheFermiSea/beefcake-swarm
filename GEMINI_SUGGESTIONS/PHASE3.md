Phase 3: Deep-Dive Implementation of Arbitration LoopTarget File: * coordination/src/ensemble/arbitration.rsThis file implements the "Puzld.ai" pattern, orchestrating the Coder and Reviewer over the shared SwarmMemory until consensus is reached.The State Machine Implementation```rustuse rig::agent::Agent;use rig::providers::anthropic::CompletionModel;use tracing::{error, info, instrument, warn};use crate::state::compaction::SwarmMemory;#[derive(Debug)]pub enum ArbitrationResult {Success,Deadlock(String),SystemError(String),}pub struct DebateOrchestrator {coder: Agent<CompletionModel>,reviewer: Agent<CompletionModel>,max_iterations: usize,}impl DebateOrchestrator {pub fn new(coder: Agent<CompletionModel>, reviewer: Agent<CompletionModel>, max_iterations: usize) -> Self {Self { coder, reviewer, max_iterations }}/// The core loop: Coder -> Reviewer -> Consensus Check
#[instrument(skip(self, memory))]
pub async fn run_debate(&self, initial_task: &str, memory: &mut SwarmMemory) -> ArbitrationResult {
    let mut iteration = 1;
    let mut coder_prompt = format!("TASK: {}\nImplement this using your tools.", initial_task);

    while iteration <= self.max_iterations {
        info!(iteration, "Starting debate cycle.");

        // -----------------------------------------
        // 1. CODER PHASE
        // -----------------------------------------
        info!("Coder is actively working...");
        
        // Rig handles tool calling internally during `.prompt()`
        let coder_response = match self.coder.prompt(&coder_prompt).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Coder LLM/Tool execution failed: {}", e);
                return ArbitrationResult::SystemError(e.to_string());
            }
        };

        // Push to shared memory
        if let Err(e) = memory.push_and_compact(rig::message::Message::User { 
            content: vec![rig::message::MessageContent::Text(coder_prompt.clone())] 
        }).await {
            return ArbitrationResult::SystemError(e.to_string());
        }
        if let Err(e) = memory.push_and_compact(rig::message::Message::Assistant { 
            content: vec![rig::message::MessageContent::Text(coder_response.clone())] 
        }).await {
            return ArbitrationResult::SystemError(e.to_string());
        }

        // -----------------------------------------
        // 2. REVIEWER PHASE
        // -----------------------------------------
        info!("Reviewer is auditing the workspace...");
        let reviewer_task = format!(
            "The Coder has completed the implementation. Review the workspace using your verifier_tool. \n\nCoder's comments:\n{}", 
            coder_response
        );

        let reviewer_response = match self.reviewer.prompt(&reviewer_task).await {
            Ok(resp) => resp,
            Err(e) => {
                error!("Reviewer LLM/Tool execution failed: {}", e);
                return ArbitrationResult::SystemError(e.to_string());
            }
        };

        // Push to shared memory
        if let Err(e) = memory.push_and_compact(rig::message::Message::User { 
            content: vec![rig::message::MessageContent::Text(reviewer_task)] 
        }).await {
            return ArbitrationResult::SystemError(e.to_string());
        }
        if let Err(e) = memory.push_and_compact(rig::message::Message::Assistant { 
            content: vec![rig::message::MessageContent::Text(reviewer_response.clone())] 
        }).await {
            return ArbitrationResult::SystemError(e.to_string());
        }

        // -----------------------------------------
        // 3. CONSENSUS EVALUATION
        // -----------------------------------------
        if reviewer_response.contains("CONSENSUS_REACHED") {
            info!("Consensus reached on iteration {}. Task complete.", iteration);
            return ArbitrationResult::Success;
        } else {
            warn!("Reviewer rejected the implementation. Formatting feedback for next iteration.");
            
            // Construct feedback loop for the coder
            coder_prompt = format!(
                "The Reviewer rejected your code and provided the following feedback:\n\n{}\n\nAddress these issues and apply a new patch.",
                reviewer_response
            );
        }

        iteration += 1;
    }

    let deadlock_msg = format!("Debate deadlock after {} iterations. Human intervention required.", self.max_iterations);
    error!("{}", deadlock_msg);
    ArbitrationResult::Deadlock(deadlock_msg)
}
}```Integration into crates/swarm-agents/src/orchestrator.rsTo launch the swarm:```rust// Inside orchestrator.rsuse crate::agents::{coder::build_coder_agent, reviewer::build_reviewer_agent};use coordination::ensemble::arbitration::{DebateOrchestrator, ArbitrationResult};use coordination::state::compaction::SwarmMemory;pub async fn run_task(task: &str) {let client = rig::providers::anthropic::Client::from_env();let coder = build_coder_agent(&client);
let reviewer = build_reviewer_agent(&client);

let summarizer = client.agent("claude-3-haiku-20240307").build().unwrap();
let mut memory = SwarmMemory::new(summarizer, 16_000); // 16k token limit

let orchestrator = DebateOrchestrator::new(coder, reviewer, 5); // Max 5 iterations

match orchestrator.run_debate(task, &mut memory).await {
    ArbitrationResult::Success => tracing::info!("Swarm successfully completed the task!"),
    ArbitrationResult::Deadlock(e) => tracing::error!("Swarm deadlocked: {}", e),
    ArbitrationResult::SystemError(e) => tracing::error!("System failure: {}", e),
}
}```
