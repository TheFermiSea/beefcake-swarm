Phase 4: BeadHub as Inter-Process Communication (IPC)

Logical Justification (Slate)

In the LLM OS model, isolated processes (threads) need to communicate with the Kernel (Orchestrator) and other processes. beefcake-swarm already has bdh (BeadHub) for this. We must formally map BeadHub commands to OS IPC concepts.

bdh mail: A thread returning an Episode / yielding control.

bdh chat: A system interrupt (syscall) when a thread encounters a fatal error and needs the Kernel to unblock it.

Agent Instructions

Step 1: Formalize IPC in Bdh Bridge

Open crates/swarm-agents/src/bdh_bridge.rs.

// Target: crates/swarm-agents/src/bdh_bridge.rs

impl BdhBridge {
/// OS Concept: Thread Yield / Episode Return
/// Tactical worker sends its final EpisodeSummary to the Orchestrator alias via mail
pub async fn yield_episode(&self, target_alias: &str, summary: &EpisodeSummary) -> Result<()> {
let json_payload = serde_json::to_string(summary)?;

        // Execute: bdh :aweb mail send <orchestrator> <json_payload>
        self.exec_bdh_command(vec![
            ":aweb", "mail", "send", target_alias, &json_payload
        ]).await?;

        Ok(())
    }

    /// OS Concept: Syscall / Interrupt
    /// Tactical worker hits an unrecoverable state and halts, pinging the kernel
    pub async fn sys_interrupt(&self, target_alias: &str, error_ctx: &str) -> Result<()> {
        // Execute: bdh :aweb chat send-and-wait <orchestrator> <error_ctx> --start-conversation
        self.exec_bdh_command(vec![
            ":aweb", "chat", "send-and-wait", target_alias, error_ctx, "--start-conversation"
        ]).await?;

        Ok(())
    }

}

Step 2: Update the Kernel's Event Loop

In crates/swarm-agents/src/orchestrator.rs, the Cloud Manager should periodically run bdh :aweb mail list to pull in completed episodes, parse the JSON EpisodeSummary, and append it to its working memory before deciding what threads to dispatch next.

Task for Coder:

Implement the yield_episode and sys_interrupt wrappers in bdh_bridge.rs.

Update the tactical worker's landing the plane logic (in runner.rs or agentic.rs) to automatically invoke yield_episode before closing its Gastown worktree.
