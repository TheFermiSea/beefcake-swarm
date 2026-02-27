use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{error, info, warn};

use swarm_agents::agents::AgentFactory;
use swarm_agents::beads_bridge::{BeadsBridge, BeadsIssue, IssueTracker, NoOpTracker};
use swarm_agents::config::{check_endpoint_with_model, SwarmConfig};
use swarm_agents::modes::SwarmMode;
use swarm_agents::notebook_bridge::{KnowledgeBase, NotebookBridge};
use swarm_agents::orchestrator;
use swarm_agents::prompts;
use swarm_agents::worktree_bridge::WorktreeBridge;

/// Autonomous coding swarm orchestrator.
///
/// Picks the highest-priority issue from beads (or uses --issue for beads-free mode),
/// creates an isolated worktree, and runs the implement → verify → review → escalate loop.
#[derive(Parser, Debug)]
#[command(name = "swarm-agents", version, about)]
struct CliArgs {
    /// Path to the target repository root.
    /// Defaults to the current working directory.
    #[arg(long)]
    repo_root: Option<PathBuf>,

    /// Scope verifier to specific cargo packages (repeatable).
    /// When omitted, targets the entire workspace.
    #[arg(long = "package", short = 'p')]
    packages: Vec<String>,

    /// Issue ID for beads-free mode. Use with --objective.
    /// Bypasses beads entirely — useful for external repos.
    #[arg(long)]
    issue: Option<String>,

    /// Issue description/objective (used with --issue).
    #[arg(long)]
    objective: Option<String>,

    /// Path to a JSON file defining the issue.
    /// Expected shape: {"id": "...", "title": "...", "status": "open", "priority": 2}
    #[arg(long)]
    issue_file: Option<PathBuf>,

    /// Cloud-only mode: skip local endpoint health checks, route all work through cloud.
    /// Requires SWARM_CLOUD_URL to be configured.
    #[arg(long)]
    cloud_only: bool,

    /// Orchestration mode for the new NS-2/3/4 mode runners.
    ///
    /// - `contextual`  — Iterative Drafting → Critiquing → Condensing FSM (NS-2)
    /// - `deepthink`   — JoinSet fan-out across parallel strategy branches (NS-3)
    /// - `agentic`     — LLM-driven unified-diff file editing loop (NS-4)
    ///
    /// When omitted the default implement→verify loop is used.
    #[arg(long, value_enum)]
    mode: Option<SwarmMode>,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = CliArgs::parse();

    let mut config = SwarmConfig::default();

    // Apply CLI overrides
    if !args.packages.is_empty() {
        config.verifier_packages = args.packages;
    }
    if args.cloud_only {
        config.cloud_only = true;
    }
    info!(
        fast = %config.fast_endpoint.url,
        coder = %config.coder_endpoint.url,
        reasoning = %config.reasoning_endpoint.url,
        cloud = config.cloud_endpoint.is_some(),
        max_retries = config.max_retries,
        prompt_version = prompts::PROMPT_VERSION,
        mode = ?args.mode,
        "Swarm orchestrator starting"
    );

    // --- Health check endpoints with model verification ---
    if config.cloud_only {
        info!("Cloud-only mode — skipping local endpoint health checks");
        if let Some(ref cloud_ep) = config.cloud_endpoint {
            let cloud_ok =
                check_endpoint_with_model(&cloud_ep.url, Some(&cloud_ep.api_key), None).await;
            if !cloud_ok {
                error!(
                    url = %cloud_ep.url,
                    "Cloud endpoint not reachable — aborting (cloud-only mode)"
                );
                anyhow::bail!(
                    "Cloud endpoint {} is not reachable. Check proxy status.",
                    cloud_ep.url
                );
            }
            info!(url = %cloud_ep.url, "Cloud endpoint health check passed");
        } else {
            error!("--cloud-only requires SWARM_CLOUD_URL to be configured");
            anyhow::bail!("Cloud-only mode requires cloud_endpoint");
        }
    } else {
        let local_ok = check_endpoint_with_model(
            &config.fast_endpoint.url,
            Some(&config.fast_endpoint.api_key),
            Some(&config.fast_endpoint.model),
        )
        .await;
        let reasoning_ok = check_endpoint_with_model(
            &config.reasoning_endpoint.url,
            Some(&config.reasoning_endpoint.api_key),
            Some(&config.reasoning_endpoint.model),
        )
        .await;
        info!(local_ok, reasoning_ok, "Endpoint health check");
        if !local_ok {
            warn!(
                url = %config.fast_endpoint.url,
                model = %config.fast_endpoint.model,
                "Fast endpoint not ready. Start inference: sbatch run-14b.slurm"
            );
        }
        if !reasoning_ok {
            warn!(
                url = %config.reasoning_endpoint.url,
                model = %config.reasoning_endpoint.model,
                "Reasoning endpoint not ready. Start inference: sbatch run-72b-distributed.slurm"
            );
        }

        if !local_ok && !reasoning_ok {
            if config.cloud_endpoint.is_some() {
                warn!("Local endpoints down — will attempt cloud-only mode");
            } else {
                error!("All endpoints unreachable and no cloud configured — exiting");
                anyhow::bail!("No inference endpoints available");
            }
        }
    }

    // --- Initialize NotebookLM knowledge base ---
    let knowledge_base: Option<Arc<dyn KnowledgeBase>> = if let Some(ref registry_path) =
        config.notebook_registry_path
    {
        match NotebookBridge::from_registry(registry_path) {
            Ok(bridge) if bridge.is_available() => {
                info!("NotebookLM knowledge base initialized");
                Some(Arc::new(bridge))
            }
            Ok(_) => {
                warn!("NotebookLM registry found but `nlm` CLI not available — running without knowledge base");
                None
            }
            Err(e) => {
                warn!(error = %e, "NotebookLM registry not loaded — running without knowledge base");
                None
            }
        }
    } else {
        None
    };

    // --- Build agent factory ---
    let mut factory = AgentFactory::new(&config)?;
    if let Some(ref kb) = knowledge_base {
        factory = factory.with_notebook_bridge(kb.clone());
    }

    // Detect repo root
    let repo_root = match &args.repo_root {
        Some(path) => path
            .canonicalize()
            .context("--repo-root path does not exist")?,
        None => std::env::current_dir()?,
    };
    let worktree_bridge = WorktreeBridge::new(config.worktree_base.clone(), &repo_root)?;

    // --- Clean up zombie branches from previous crashed runs ---
    match worktree_bridge.cleanup_stale() {
        Ok(cleaned) if !cleaned.is_empty() => {
            info!(count = cleaned.len(), branches = ?cleaned, "Cleaned up zombie swarm branches");
        }
        Ok(_) => {}
        Err(e) => {
            warn!(error = %e, "Failed to clean up stale branches");
        }
    }

    // --- Issue selection: 3 branches ---
    let kb_ref: Option<&dyn KnowledgeBase> = knowledge_base
        .as_ref()
        .map(|kb| kb.as_ref() as &dyn KnowledgeBase);

    if let Some(ref issue_id) = args.issue {
        // Branch 1: --issue provided → beads-free synthetic issue
        let title = args
            .objective
            .clone()
            .unwrap_or_else(|| format!("CLI issue: {issue_id}"));
        let issue = BeadsIssue {
            id: issue_id.clone(),
            title,
            status: "open".to_string(),
            priority: Some(1),
            issue_type: Some("task".to_string()),
            labels: vec![],
        };
        let tracker = NoOpTracker;
        info!(id = %issue.id, title = %issue.title, mode = ?args.mode, "Beads-free mode: processing CLI issue");
        tokio::select! {
            result = orchestrator::process_issue(&config, &factory, &worktree_bridge, &issue, &tracker, kb_ref) => {
                result?;
            }
            _ = shutdown_signal() => {
                warn!(id = %issue.id, "Shutdown signal received — cleaning up worktree");
                if let Err(e) = worktree_bridge.cleanup(&issue.id) {
                    error!(id = %issue.id, "Cleanup failed: {e}");
                }
                let _ = std::process::Command::new("bd")
                    .args(["update", &issue.id, "--status=open"])
                    .status();
                info!(id = %issue.id, "Graceful shutdown complete");
                return Ok(());
            }
        }
    } else if let Some(ref issue_path) = args.issue_file {
        // Branch 2: --issue-file provided → deserialize from JSON
        let contents = std::fs::read_to_string(issue_path).context(format!(
            "Failed to read issue file: {}",
            issue_path.display()
        ))?;
        let issue: BeadsIssue =
            serde_json::from_str(&contents).context("Failed to parse issue JSON")?;
        let tracker = NoOpTracker;
        info!(id = %issue.id, title = %issue.title, "Beads-free mode: processing issue from file");
        tokio::select! {
            result = orchestrator::process_issue(&config, &factory, &worktree_bridge, &issue, &tracker, kb_ref) => {
                result?;
            }
            _ = shutdown_signal() => {
                warn!(id = %issue.id, "Shutdown signal received — cleaning up worktree");
                if let Err(e) = worktree_bridge.cleanup(&issue.id) {
                    error!(id = %issue.id, "Cleanup failed: {e}");
                }
                // Branch 2 uses NoOpTracker, but we try a best-effort bd update in case it was a real bead
                let _ = std::process::Command::new("bd")
                    .args(["update", &issue.id, "--status=open"])
                    .status();
                info!(id = %issue.id, "Graceful shutdown complete");
                return Ok(());
            }
        }
    } else if let Ok(target_id) = std::env::var("SWARM_ISSUE") {
        // Branch 3: SWARM_ISSUE env var — fetch specific issue from beads
        let beads = BeadsBridge::new();
        let issue = match beads.show(&target_id) {
            Ok(i) => i,
            Err(e) => {
                error!(target_id = %target_id, error = %e, "SWARM_ISSUE not found");
                return Ok(());
            }
        };
        info!(id = %issue.id, title = %issue.title, "SWARM_ISSUE: targeting specific issue");
        tokio::select! {
            result = orchestrator::process_issue(&config, &factory, &worktree_bridge, &issue, &beads, kb_ref) => {
                result?;
            }
            _ = shutdown_signal() => {
                warn!(id = %issue.id, "Shutdown signal received — cleaning up worktree");
                if let Err(e) = worktree_bridge.cleanup(&issue.id) {
                    error!(id = %issue.id, "Cleanup failed: {e}");
                }
                if let Err(e) = beads.update_status(&issue.id, "open") {
                    error!(id = %issue.id, "Failed to reset issue status: {e}");
                }
                info!(id = %issue.id, "Graceful shutdown complete");
                return Ok(());
            }
        }
    } else {
        // Branch 4: Default — pick from beads
        let beads = BeadsBridge::new();
        let issues = match beads.list_ready() {
            Ok(issues) => issues,
            Err(e) => {
                warn!(error = %e, "Beads not available");
                info!("No issues to process. Orchestrator exiting.");
                return Ok(());
            }
        };

        if issues.is_empty() {
            info!("No ready issues. Orchestrator exiting.");
            return Ok(());
        }

        // Sort by priority (lowest = highest priority), then swarm_complexity (simpler first).
        // Dogfooding showed additive tasks succeed; modification tasks fail more often.
        let mut sorted = issues;
        sorted.sort_by_key(|i| (i.priority.unwrap_or(4), i.swarm_complexity_rank()));

        // Try to claim each issue in priority order (prevents race with other instances)
        let mut claimed_issue = None;
        for candidate in &sorted {
            match beads.try_claim(&candidate.id) {
                Ok(true) => {
                    info!(
                        id = %candidate.id,
                        title = %candidate.title,
                        priority = ?candidate.priority,
                        "Claimed issue to work on"
                    );
                    claimed_issue = Some(candidate);
                    break;
                }
                Ok(false) => {
                    info!(id = %candidate.id, "Issue already claimed, trying next");
                }
                Err(e) => {
                    warn!(id = %candidate.id, error = %e, "Failed to claim issue, trying next");
                }
            }
        }

        let issue = match claimed_issue {
            Some(i) => i,
            None => {
                info!("All ready issues already claimed. Orchestrator exiting.");
                return Ok(());
            }
        };

        tokio::select! {
            result = orchestrator::process_issue(&config, &factory, &worktree_bridge, issue, &beads, kb_ref) => {
                result?;
            }
            _ = shutdown_signal() => {
                warn!(id = %issue.id, "Shutdown signal received — cleaning up worktree");
                if let Err(e) = worktree_bridge.cleanup(&issue.id) {
                    error!(id = %issue.id, "Cleanup failed: {e}");
                }
                if let Err(e) = beads.update_status(&issue.id, "open") {
                    error!(id = %issue.id, "Failed to reset issue status: {e}");
                }
                info!(id = %issue.id, "Graceful shutdown complete");
                return Ok(());
            }
        }
    }

    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal;
    #[cfg(unix)]
    {
        let mut sigterm = signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler");
        tokio::select! {
            _ = signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = signal::ctrl_c().await;
    }
}
