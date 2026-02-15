use anyhow::Result;
use tracing::{error, info, warn};

use swarm_agents::agents::AgentFactory;
use swarm_agents::beads_bridge::{BeadsBridge, IssueTracker};
use swarm_agents::config::{check_endpoint, SwarmConfig};
use swarm_agents::orchestrator;
use swarm_agents::prompts;
use swarm_agents::worktree_bridge::WorktreeBridge;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let config = SwarmConfig::default();
    info!(
        fast = %config.fast_endpoint.url,
        coder = %config.coder_endpoint.url,
        reasoning = %config.reasoning_endpoint.url,
        cloud = config.cloud_endpoint.is_some(),
        max_retries = config.max_retries,
        prompt_version = prompts::PROMPT_VERSION,
        "Swarm orchestrator starting"
    );

    // --- Health check endpoints ---
    let local_ok = check_endpoint(
        &config.fast_endpoint.url,
        Some(&config.fast_endpoint.api_key),
    )
    .await;
    let reasoning_ok = check_endpoint(
        &config.reasoning_endpoint.url,
        Some(&config.reasoning_endpoint.api_key),
    )
    .await;
    info!(local_ok, reasoning_ok, "Endpoint health check");

    if !local_ok && !reasoning_ok {
        if config.cloud_endpoint.is_some() {
            warn!("Local endpoints down — will attempt cloud-only mode");
        } else {
            error!("All endpoints unreachable and no cloud configured — exiting");
            anyhow::bail!("No inference endpoints available");
        }
    }

    // --- Build agent factory ---
    let factory = AgentFactory::new(&config)?;

    // --- Initialize beads bridge ---
    let beads = BeadsBridge::new();

    // Detect repo root
    let repo_root = std::env::current_dir()?;
    let worktree_bridge = WorktreeBridge::new(config.worktree_base.clone(), &repo_root)?;

    // --- Pick highest-priority open issue ---
    let issues = match beads.list_open() {
        Ok(issues) => issues,
        Err(e) => {
            warn!("Beads not available: {e}");
            info!("No issues to process. Orchestrator exiting.");
            return Ok(());
        }
    };

    if issues.is_empty() {
        info!("No open issues. Orchestrator exiting.");
        return Ok(());
    }

    // Sort by priority (lowest number = highest priority), pick first
    let mut sorted = issues;
    sorted.sort_by_key(|i| i.priority.unwrap_or(4));
    let issue = &sorted[0];

    info!(
        id = %issue.id,
        title = %issue.title,
        priority = ?issue.priority,
        "Picked issue to work on"
    );

    // --- Process the issue ---
    orchestrator::process_issue(&config, &factory, &worktree_bridge, issue, &beads).await?;

    Ok(())
}
