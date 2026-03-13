Phase 2: Thread Weaving (Massive Parallelism)

Logical Justification (Slate)

Rigid ReAct loops process one node in a task graph at a time. "Thread weaving" means the Orchestrator dispatches multiple threads concurrently. Because of our gastown worktree isolation, we can safely run multiple workers on different files at the exact same time.

Agent Instructions

Step 1: Async Orchestrator Dispatch

Open crates/swarm-agents/src/orchestrator.rs. Locate the main loop that dispatches work to local agents. We need to replace procedural, awaiting loops with tokio::task::spawn and an MPSC channel for collecting EpisodeSummary returns.

// Target: crates/swarm-agents/src/orchestrator.rs
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use futures::future::join_all;

pub struct ThreadWeaver {
episode_tx: mpsc::Sender<EpisodeSummary>,
episode_rx: mpsc::Receiver<EpisodeSummary>,
active_threads: Vec<JoinHandle<()>>,
}

impl ThreadWeaver {
pub fn new() -> Self {
let (tx, rx) = mpsc::channel(32);
Self {
episode_tx: tx,
episode_rx: rx,
active_threads: vec![],
}
}

    pub async fn dispatch_thread(&mut self, workspace_id: String, task: String) {
        let tx = self.episode_tx.clone();

        let handle = tokio::spawn(async move {
            // 1. Setup Gastown worktree
            // 2. Spawn local worker (Scout/Integrator) via RPC
            // 3. Execute agentic loop
            // 4. finalize_episode() -> EpisodeSummary
            // 5. tx.send(episode_summary).await
        });

        self.active_threads.push(handle);
    }

    pub async fn weave_episodes(&mut self) -> Vec<EpisodeSummary> {
        let mut summaries = vec![];
        // Wait for all threads to yield their episodes
        while let Some(summary) = self.episode_rx.recv().await {
            summaries.push(summary);
            // In a real OS, the kernel might react immediately to a Blocked outcome here
        }
        summaries
    }

}

Task for Coder:

Refactor the Orchestrator struct in crates/swarm-agents/src/orchestrator.rs to implement the ThreadWeaver pattern.

Ensure the Cloud Manager (Opus) is exposed to a dispatch_thread tool that accepts a list of parallel tasks, rather than a single delegate_task tool.
