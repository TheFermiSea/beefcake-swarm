mod beads_bridge;
mod config;
mod implementer;
mod validator;

use anyhow::Result;
use config::SwarmConfig;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = SwarmConfig::default();
    info!(
        fast = %config.fast_endpoint.url,
        reasoning = %config.reasoning_endpoint.url,
        "Swarm orchestrator starting"
    );

    // Phase 1: Just verify beads connectivity
    let beads = beads_bridge::BeadsBridge::new();
    match beads.list_open() {
        Ok(issues) => {
            info!(count = issues.len(), "Found open beads issues");
            for issue in &issues {
                info!(id = %issue.id, title = %issue.title, "  issue");
            }
        }
        Err(e) => {
            tracing::warn!("Beads not available (expected if br not installed): {e}");
        }
    }

    // TODO: Phase 2 — implement the 2-agent loop:
    // 1. Pick highest-priority open issue from beads
    // 2. Create Gastown worktree
    // 3. Run Implementer agent (72B)
    // 4. Run deterministic Verifier (cargo fmt/clippy/test)
    // 5. Run Validator agent (14B, blind review)
    // 6. If pass: merge worktree, close issue
    // 7. If fail: update issue notes, loop back to step 3

    info!("Orchestrator ready. No tasks to process yet (2-agent loop not implemented).");

    Ok(())
}
