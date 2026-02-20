Phase 5: Implementation Glue and Test HarnessesTarget Files:crates/swarm-agents/src/tools/mod.rs (MODIFIED)crates/swarm-agents/src/tools/verifier_tool.rs (MODIFIED)coordination/tests/debate_integration_test.rs (NEW)crates/swarm-agents/src/tools/graph_rag_tool.rs (ENVIRONMENT FIX)This document provides the necessary connective tissue for the coding agent to successfully compile, run, and test the SOTA swarm integration (Phases 1-4).1. Tool Module ExportsThe new SOTA tools defined in Phase 4 must be explicitly exported in the tools module.File: crates/swarm-agents/src/tools/mod.rsAction required: Append the following module declarations.pub mod exec_tool;
pub mod fs_tools;
pub mod patch_tool;
pub mod proxy_wrappers;
pub mod verifier_tool;
pub mod notebook_tool;

// --- NEW SOTA TOOLS ---
pub mod ast_grep_tool;
pub mod graph_rag_tool;
2. Ingesting Rules into the VerifierIn Phase 4, the Reviewer agent was instructed to verify compliance against rules/no-unwrap-in-prod.yml and rules/use-tracing-not-print.yml. LLMs cannot read these files by magic; the VerifierTool must actively inject them into its output or the system prompt must pre-load them.File: crates/swarm-agents/src/tools/verifier_tool.rsAction required: Update the tool to read the rules/ directory and append the YAML rule definitions to the verification output, allowing the Reviewer to cross-reference clippy output with the project's custom rules.// Inside verifier_tool.rs `call` function implementation:

async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
    // 1. Run Cargo Check/Clippy
    let clippy_output = Command::new("cargo")
        .args(["clippy", "--message-format=short"])
        .output()
        .await
        .map_err(|e| e.to_string())?;
        
    let mut final_report = format!("## CLI RESULT\n{}\n\n", String::from_utf8_lossy(&clippy_output.stdout));
    
    // 2. Load Project Rules
    // This gives the Reviewer the exact context of what "secure" means for this specific project.
    let rules_dir = std::path::Path::new("rules");
    if rules_dir.exists() && rules_dir.is_dir() {
        final_report.push_str("## ACTIVE PROJECT RULES (Enforce these!)\n");
        let mut entries = tokio::fs::read_dir(rules_dir).await.map_err(|e| e.to_string())?;
        
        while let Some(entry) = entries.next_entry().await.unwrap_or(None) {
            if entry.path().extension().map_or(false, |ext| ext == "yml") {
                if let Ok(content) = tokio::fs::read_to_string(entry.path()).await {
                    final_report.push_str(&format!("--- Rule: {:?} ---\n{}\n", entry.file_name(), content));
                }
            }
        }
    }

    Ok(final_report)
}
3. Graph RAG HPC/Environment Bridge FixYour architecture runs on HPC/Slurm (evident from infrastructure/hpc-watchdog.sh and the Slurm scripts). Executing python3 blindly in graph_rag_tool.rs will fail if the active Slurm job doesn't have the .venv activated where cocoindex is installed.File: crates/swarm-agents/src/tools/graph_rag_tool.rsAction required: The coding agent must implement an environment-aware command execution wrapper.// Replace the naive `Command::new("python3")` with a shell-wrapped execution
// inside graph_rag_tool.rs `call` method:

let output = Command::new("bash")
    .arg("-c")
    // Note: Assuming standard python venv structure. Adjust path to venv as necessary.
    .arg(format!(
        "source .venv/bin/activate && python3 indexing/index_flow_v2.py --query '{}' --mode {}",
        args.query.replace("'", "'\\''"), // Shell escape the query
        args.query_type
    ))
    .output()
    .await
    .map_err(|e| {
        error!("Failed to execute CocoIndex bash bridge: {}", e);
        e.to_string()
    })?;
4. TDD / Mock Debate Integration TestTo ensure the coding agent implements Phase 3 (DebateOrchestrator) without burning API credits, we must provide a mocked integration test.File: coordination/tests/debate_integration_test.rs (NEW)use coordination::ensemble::arbitration::{DebateOrchestrator, ArbitrationResult};
use coordination::state::compaction::SwarmMemory;
use rig::agent::AgentBuilder;
use rig::providers::anthropic::Client;
// Note: In a real test suite, you'd use a Mock provider or an extremely cheap model.

#[tokio::test]
#[ignore] // Run manually or with mock server to avoid API costs
async fn test_debate_loop_consensus() {
    let client = Client::from_env();
    
    // Create highly constrained dummy agents for testing the loop mechanics
    let coder = client.agent("claude-3-haiku-20240307")
        .preamble("You are testing a loop. Always output 'fn test() { println!(\"hello\"); }'")
        .build()
        .unwrap();

    let reviewer = client.agent("claude-3-haiku-20240307")
        .preamble("You are testing a loop. If the input contains 'println', output exactly 'CONSENSUS_REACHED'.")
        .build()
        .unwrap();

    let summarizer = client.agent("claude-3-haiku-20240307").build().unwrap();
    let mut memory = SwarmMemory::new(summarizer, 1000);

    let orchestrator = DebateOrchestrator::new(coder, reviewer, 3);
    
    let result = orchestrator.run_debate("Write a test function.", &mut memory).await;
    
    assert!(matches!(result, ArbitrationResult::Success));
    
    // Verify memory compaction heuristics
    assert!(!memory.history.is_empty());
}

