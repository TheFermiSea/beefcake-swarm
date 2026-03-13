Phase 3: Strategy/Tactics Segregation

Logical Justification (Slate)

AlphaZero split the value network (Strategy) from the policy network (Tactics). The Cloud Manager should act as the Value Network: judging the state of the codebase and routing threads. It should never execute a tactical command like sed or cargo build. The Local Workers act as the Policy Network: executing tactical moves to reach a local goal.

Agent Instructions

Step 1: Segregate Tool Bundles

Open coordination/src/tool_bundle.rs or crates/swarm-agents/src/tools/bundles.rs.
You must explicitly separate the tools injected into the Rig agent builder based on the agent's role.

// Target: crates/swarm-agents/src/tools/bundles.rs

/// Tools EXCLUSIVELY for the Cloud Manager (Kernel / Strategy)
pub fn kernel_strategy_tools() -> Vec<Box<dyn rig::tool::Tool>> {
vec![
        Box::new(DispatchThreadTool::new()), // Spawns a new Gastown worktree + Worker
        Box::new(ReviewEpisodesTool::new()), // Reads completed thread summaries
        Box::new(BdhStatusTool::new()),      // Check team status
    ]
}

/// Tools EXCLUSIVELY for Local Workers (Processes / Tactics)
pub fn process_tactical_tools() -> Vec<Box<dyn rig::tool::Tool>> {
vec![
        Box::new(AstGrepTool::new()),
        Box::new(CargoCheckTool::new()),
        Box::new(GitOpsTool::new()),
        Box::new(PatchTool::new()),
    ]
}

Step 2: Update Agent Builders

Open crates/swarm-agents/src/agents/cloud.rs. Ensure kernel_strategy_tools are the only tools attached to the Opus agent.

Open crates/swarm-agents/src/agents/coder.rs and integrator.rs. Ensure process_tactical_tools are the only tools attached here. Remove any high-level planning prompts from their system prompts; they should only focus on "Solve this specific subtask and report the Episode."

Task for Coder:

Audit crates/swarm-agents/src/tools/ and strictly categorize them.

Update the AgentBuilder initializations in cloud.rs, coder.rs, and manager.rs to enforce this strict separation of concerns.
