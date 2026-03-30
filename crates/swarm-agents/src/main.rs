use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use tracing::{error, info, warn};

use swarm_agents::agents::AgentFactory;
use swarm_agents::beads_bridge::{BeadsBridge, BeadsIssue, IssueTracker, NoOpTracker};
use swarm_agents::config::{check_endpoint_with_model, SwarmConfig};
use swarm_agents::modes::SwarmMode;
use swarm_agents::notebook_bridge::{KnowledgeBase, NotebookBridge};
use swarm_agents::orchestrator::{self, CANCEL_MSG};
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

    /// Batch of issue IDs to process in parallel (repeatable).
    /// Up to SWARM_PARALLEL_ISSUES (default: 3) run simultaneously.
    /// Each issue gets its own worktree; nodes are selected in round-robin order.
    #[arg(long)]
    issues: Vec<String>,

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

    /// Subcommand for specialized operations.
    #[command(subcommand)]
    command: Option<SubCommand>,
}

#[derive(clap::Subcommand, Debug)]
enum SubCommand {
    /// Pick the next workable issue from beads, filtering through reformulation state.
    ///
    /// Outputs a JSON object with the selected issue, or exits with code 1 if none available.
    /// Used by dogfood-loop.sh to replace bash-side `bd ready` parsing.
    PickNext {
        /// Output as JSON (default: true).
        #[arg(long, default_value_t = true)]
        json: bool,
    },
    /// Run the Autopilot analysis pipeline: archive → insights → recommendations.
    ///
    /// Writes JSON artifacts to .swarm/autopilot/ and prints an operator summary.
    Autopilot {
        /// Number of recent mutation records to analyze (default: 50).
        #[arg(long, default_value_t = 50)]
        window: usize,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = CliArgs::parse();

    // --- Handle subcommands before full orchestrator initialization ---
    if let Some(SubCommand::PickNext { json: _ }) = &args.command {
        return handle_pick_next(args.repo_root.as_deref());
    }
    if let Some(SubCommand::Autopilot { window }) = &args.command {
        let repo_root = args
            .repo_root
            .clone()
            .unwrap_or_else(|| std::env::current_dir().expect("current dir"));
        let runner =
            swarm_agents::autopilot::AutopilotRunner::new(&repo_root).with_window_size(*window);
        let report = runner.run().await;
        println!("{}", report.operator_summary());
        return Ok(());
    }

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
        let coder_ok = check_endpoint_with_model(
            &config.coder_endpoint.url,
            Some(&config.coder_endpoint.api_key),
            Some(&config.coder_endpoint.model),
        )
        .await;
        let reasoning_ok = check_endpoint_with_model(
            &config.reasoning_endpoint.url,
            Some(&config.reasoning_endpoint.api_key),
            Some(&config.reasoning_endpoint.model),
        )
        .await;
        info!(local_ok, coder_ok, reasoning_ok, "Endpoint health check");
        if !local_ok {
            error!(
                url = %config.fast_endpoint.url,
                model = %config.fast_endpoint.model,
                "Fast/scout endpoint not ready (vasp-03). sbatch run-27b-256k.slurm"
            );
        }
        if !coder_ok {
            error!(
                url = %config.coder_endpoint.url,
                model = %config.coder_endpoint.model,
                "Coder/integrator endpoint not ready (vasp-01). sbatch run-122b-rpc.slurm"
            );
        }
        if !reasoning_ok {
            error!(
                url = %config.reasoning_endpoint.url,
                model = %config.reasoning_endpoint.model,
                "Reasoning/integrator endpoint not ready (vasp-02). sbatch --nodelist=vasp-02 run-122b-rpc.slurm"
            );
        }

        if !local_ok && !coder_ok && !reasoning_ok {
            if config.cloud_endpoint.is_some() {
                warn!("All local endpoints down — will attempt cloud-only mode");
                config.cloud_only = true;
            } else {
                error!("All endpoints unreachable and no cloud configured — exiting");
                anyhow::bail!("No inference endpoints available");
            }
        } else if !local_ok || !coder_ok || !reasoning_ok {
            // At least one local endpoint is down but not all — warn loudly.
            // Workers routed to failed endpoints will get 502 errors and waste iterations.
            warn!(
                local_ok,
                coder_ok,
                reasoning_ok,
                "Some local endpoints are down — workers may fail. Fix before running dogfood."
            );
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

    // --- Build agent factory with health-aware routing ---
    let cluster_health = swarm_agents::cluster_health::ClusterHealth::from_config(&config);
    let mut factory = AgentFactory::new(&config)?.with_health(cluster_health);
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
    let worktree_bridge = Arc::new(WorktreeBridge::new(
        config.worktree_base.clone(),
        &repo_root,
    )?);

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
        // Branch 1: --issue provided → try beads lookup, fall back to synthetic
        //
        // If bd/bdh is available, fetch the real title + description so the file
        // targeting pipeline can extract identifiers and file paths from the
        // description text. Without this, the objective is just "CLI issue: {id}"
        // and the model gets sent to wrong files.
        let issue = if args.objective.is_some() {
            // Explicit --objective overrides beads lookup
            BeadsIssue {
                id: issue_id.clone(),
                title: args.objective.clone().unwrap(),
                status: "open".to_string(),
                priority: Some(1),
                issue_type: Some("task".to_string()),
                labels: vec![],
                description: None,
            }
        } else {
            // Try fetching from beads/bdh for rich title + description
            match show_issue(issue_id, args.repo_root.as_deref()) {
                Ok(mut issue) => {
                    issue.status = "open".to_string(); // ensure processable
                    issue
                }
                Err(e) => {
                    warn!(id = %issue_id, error = %e, "Beads lookup failed — using synthetic issue");
                    BeadsIssue {
                        id: issue_id.clone(),
                        title: format!("CLI issue: {issue_id}"),
                        status: "open".to_string(),
                        priority: Some(1),
                        issue_type: Some("task".to_string()),
                        labels: vec![],
                        description: None,
                    }
                }
            }
        };
        // Use real BeadsBridge when --repo-root is provided (dogfood mode),
        // fall back to NoOpTracker for truly beads-free single-issue runs.
        let tracker: Box<dyn IssueTracker> = new_tracker(args.repo_root.as_deref());
        let tracker_mode = if args.repo_root.is_some() {
            "beads"
        } else {
            "beads-free"
        };
        info!(id = %issue.id, title = %issue.title, mode = ?args.mode, tracker = tracker_mode, "Processing CLI issue");
        tokio::select! {
            result = orchestrator::process_issue(&config, &factory, &worktree_bridge, &issue, tracker.as_ref(), kb_ref, Arc::new(AtomicBool::new(false))) => {
                let resolved = result?;
                if !resolved {
                    error!(id = %issue.id, "Issue NOT resolved — exiting with failure");
                    std::process::exit(1);
                }
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
            result = orchestrator::process_issue(&config, &factory, &worktree_bridge, &issue, &tracker, kb_ref, Arc::new(AtomicBool::new(false))) => {
                let resolved = result?;
                if !resolved {
                    error!(id = %issue.id, "Issue NOT resolved — exiting with failure");
                    std::process::exit(1);
                }
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
    } else if !args.issues.is_empty() {
        // Branch 1.5: --issues batch → OS-thread parallel dispatch
        //
        // `process_issue` is now Send-safe (Slate Phase 0b), but we keep the
        // OS-thread dispatch pattern: std::thread::spawn + Handle::block_on.
        // Migration to JoinSet::spawn is tracked as Phase 2 (Thread Weaving).
        //
        // Fetch real issue details from beads, then fan-out. Each thread gets a
        // fresh BeadsBridge for status tracking. The AgentFactory clone shares
        // the Arc<AtomicUsize> round-robin counter so each thread lands on a
        // different node.
        let mut batch: Vec<BeadsIssue> = Vec::new();
        // Cap at parallel_issues to avoid overwhelming the cluster (Issue 6 fix).
        for id in args.issues.iter().take(config.parallel_issues) {
            match show_issue(id, args.repo_root.as_deref()) {
                Ok(issue) => batch.push(issue),
                Err(e) => {
                    warn!(id = %id, error = %e, "Could not fetch issue details — skipping");
                }
            }
        }
        if batch.is_empty() {
            info!("No valid issues in --issues batch. Orchestrator exiting.");
            return Ok(());
        }

        dispatch_parallel_issues(
            batch,
            &config,
            &factory,
            &worktree_bridge,
            knowledge_base.clone(),
        )
        .await?;
    } else if let Ok(target_id) = std::env::var("SWARM_ISSUE") {
        // Branch 3: SWARM_ISSUE env var — fetch specific issue from beads/bdh
        let tracker = new_tracker(args.repo_root.as_deref());
        let issue = match show_issue(&target_id, args.repo_root.as_deref()) {
            Ok(i) => i,
            Err(e) => {
                error!(target_id = %target_id, error = %e, "SWARM_ISSUE not found");
                return Ok(());
            }
        };
        info!(id = %issue.id, title = %issue.title, "SWARM_ISSUE: targeting specific issue");
        tokio::select! {
            result = orchestrator::process_issue(&config, &factory, &worktree_bridge, &issue, &*tracker, kb_ref, Arc::new(AtomicBool::new(false))) => {
                let resolved = result?;
                if !resolved {
                    error!(id = %issue.id, "Issue NOT resolved — exiting with failure");
                    std::process::exit(1);
                }
            }
            _ = shutdown_signal() => {
                warn!(id = %issue.id, "Shutdown signal received — cleaning up worktree");
                if let Err(e) = worktree_bridge.cleanup(&issue.id) {
                    error!(id = %issue.id, "Cleanup failed: {e}");
                }
                if let Err(e) = tracker.update_status(&issue.id, "open") {
                    error!(id = %issue.id, "Failed to reset issue status: {e}");
                }
                info!(id = %issue.id, "Graceful shutdown complete");
                return Ok(());
            }
        }
    } else {
        // Branch 4: Default — claim up to parallel_issues from beads/bdh and fan-out
        let tracker = new_tracker(args.repo_root.as_deref());
        let issues = match tracker.list_ready() {
            Ok(issues) => issues,
            Err(e) => {
                warn!(error = %e, "Issue tracker not available");
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

        // Claim up to parallel_issues issues in priority order.
        let max_parallel = config.parallel_issues;
        let mut claimed: Vec<BeadsIssue> = Vec::new();
        for candidate in &sorted {
            if claimed.len() >= max_parallel {
                break;
            }
            match tracker.try_claim(&candidate.id) {
                Ok(true) => {
                    info!(
                        id = %candidate.id,
                        title = %candidate.title,
                        priority = ?candidate.priority,
                        "Claimed issue to work on"
                    );
                    claimed.push(candidate.clone());
                }
                Ok(false) => {
                    info!(id = %candidate.id, "Issue already claimed, trying next");
                }
                Err(e) => {
                    warn!(id = %candidate.id, error = %e, "Failed to claim issue, trying next");
                }
            }
        }

        if claimed.is_empty() {
            info!("All ready issues already claimed. Orchestrator exiting.");
            return Ok(());
        }

        if claimed.len() == 1 {
            // Single issue: keep the existing tokio::select! shutdown path for graceful cleanup.
            let issue = claimed.remove(0);
            tokio::select! {
                result = orchestrator::process_issue(&config, &factory, &worktree_bridge, &issue, &*tracker, kb_ref, Arc::new(AtomicBool::new(false))) => {
                    let resolved = result?;
                    if !resolved {
                        error!(id = %issue.id, "Issue NOT resolved — exiting with failure");
                        std::process::exit(1);
                    }
                }
                _ = shutdown_signal() => {
                    warn!(id = %issue.id, "Shutdown signal received — cleaning up worktree");
                    if let Err(e) = worktree_bridge.cleanup(&issue.id) {
                        error!(id = %issue.id, "Cleanup failed: {e}");
                    }
                    if let Err(e) = tracker.update_status(&issue.id, "open") {
                        error!(id = %issue.id, "Failed to reset issue status: {e}");
                    }
                    info!(id = %issue.id, "Graceful shutdown complete");
                    return Ok(());
                }
            }
        } else {
            // Multiple issues: fan-out via OS threads. See dispatch_parallel_issues for details.
            dispatch_parallel_issues(
                claimed,
                &config,
                &factory,
                &worktree_bridge,
                knowledge_base.clone(),
            )
            .await?;
        }
    }

    Ok(())
}

/// Fan out a pre-assembled batch of issues across OS threads with cooperative cancellation.
///
/// `process_issue` is now `Send`-safe (Slate Phase 0b), but this function still
/// uses OS threads via `std::thread::spawn` + `Handle::block_on`. Migration to
/// `JoinSet::spawn` is tracked as Slate Phase 2 (Thread Weaving).
///
/// A shared `Arc<AtomicBool>` cancel flag is set when SIGTERM/Ctrl-C arrives.
/// Each thread checks it at iteration boundaries inside `process_issue`, resets the
/// issue to `open`, and returns early. The fan-in then exits with code 0 (not 1) so
/// `dogfood-loop.sh` treats SIGTERM as a clean stop rather than a failure.
async fn dispatch_parallel_issues(
    batch: Vec<BeadsIssue>,
    config: &SwarmConfig,
    factory: &AgentFactory,
    worktree_bridge: &Arc<WorktreeBridge>,
    knowledge_base: Option<Arc<dyn KnowledgeBase>>,
) -> Result<()> {
    let cancel_flag = Arc::new(AtomicBool::new(false));
    let flag_for_signal = Arc::clone(&cancel_flag);
    tokio::spawn(async move {
        shutdown_signal().await;
        flag_for_signal.store(true, Ordering::Release);
        warn!("Shutdown signal received — cancellation flag set for parallel dispatch");
    });

    info!(
        count = batch.len(),
        "Dispatching issues in parallel via OS threads"
    );
    let rt_handle = tokio::runtime::Handle::current();
    let thread_handles: Vec<std::thread::JoinHandle<(String, anyhow::Result<bool>)>> = batch
        .into_iter()
        .map(|issue| {
            let factory_clone = factory.clone();
            let config_clone = config.clone();
            let wb_clone = Arc::clone(worktree_bridge);
            let beads_clone = new_tracker(Some(worktree_bridge.repo_root()));
            let kb_clone = knowledge_base.clone();
            let rt = rt_handle.clone();
            let cancel = Arc::clone(&cancel_flag);
            std::thread::spawn(move || {
                let id = issue.id.clone();
                let kb_ref = kb_clone
                    .as_ref()
                    .map(|kb| kb.as_ref() as &dyn KnowledgeBase);
                let result = rt.block_on(orchestrator::process_issue(
                    &config_clone,
                    &factory_clone,
                    &wb_clone,
                    &issue,
                    &*beads_clone,
                    kb_ref,
                    cancel,
                ));
                (id, result)
            })
        })
        .collect();

    // Join via spawn_blocking to avoid blocking the async runtime.
    let join_results = tokio::task::spawn_blocking(move || {
        thread_handles
            .into_iter()
            .map(|h| h.join())
            .collect::<Vec<_>>()
    })
    .await?;

    // Distinguish shutdown-cancelled results from genuine failures.
    // Cancelled issues are already reset to "open" inside process_issue before returning.
    let cancelled = cancel_flag.load(Ordering::Acquire);
    let mut any_failed = false;
    for result in join_results {
        match result {
            Ok((id, Ok(true))) => info!(id = %id, "Issue resolved"),
            Ok((id, Ok(false))) => {
                if cancelled {
                    warn!(id = %id, "Issue not resolved (cancelled by shutdown signal)");
                } else {
                    error!(id = %id, "Issue NOT resolved");
                    any_failed = true;
                }
            }
            Ok((id, Err(e))) => {
                if cancelled && e.to_string().contains(CANCEL_MSG) {
                    warn!(id = %id, "Issue cancelled by shutdown signal");
                } else {
                    error!(id = %id, error = %e, "Issue errored");
                    any_failed = true;
                }
            }
            Err(_) => {
                error!("Task panicked in parallel dispatch");
                any_failed = true;
            }
        }
    }
    if cancelled {
        warn!("Graceful parallel shutdown complete");
        return Ok(());
    }
    if any_failed {
        anyhow::bail!("One or more parallel issues failed");
    }
    Ok(())
}

/// Create the issue tracker (always uses native `bd` CLI).
///
/// When `repo_root` is provided (via `--repo-root`), the tracker runs `bd`
/// commands in that directory so it accesses the target repo's `.beads/`
/// database instead of the CWD's.
fn new_tracker(repo_root: Option<&Path>) -> Box<dyn IssueTracker> {
    match repo_root {
        Some(path) => Box::new(BeadsBridge::with_worktree(path)),
        None => Box::new(BeadsBridge::new()),
    }
}

/// Look up a single issue by ID via `bd show`.
fn show_issue(id: &str, repo_root: Option<&Path>) -> Result<BeadsIssue> {
    match repo_root {
        Some(path) => BeadsBridge::with_worktree(path).show(id),
        None => BeadsBridge::new().show(id),
    }
}

/// Handle the `pick-next` subcommand: select the next workable issue from beads,
/// filtering through reformulation state to skip exhausted issues.
///
/// Outputs a JSON object on stdout. Exits with code 1 if no workable issue found.
fn handle_pick_next(repo_root: Option<&Path>) -> Result<()> {
    let repo_root = match repo_root {
        Some(p) => p
            .canonicalize()
            .context("--repo-root path does not exist")?,
        None => std::env::current_dir()?,
    };

    let tracker = new_tracker(Some(&repo_root));
    let issues = match tracker.list_ready() {
        Ok(issues) => issues,
        Err(e) => {
            eprintln!("Failed to query issues: {e}");
            std::process::exit(1);
        }
    };

    if issues.is_empty() {
        eprintln!("No ready issues");
        std::process::exit(1);
    }

    // Sort by priority then complexity (same as main orchestrator)
    let mut sorted = issues;
    sorted.sort_by_key(|i| (i.priority.unwrap_or(4), i.swarm_complexity_rank()));

    // Filter through reformulation state
    let store = swarm_agents::reformulation::ReformulationStore::new(&repo_root);
    for candidate in &sorted {
        if let Some(reason) = swarm_agents::reformulation::should_skip_issue(&store, &candidate.id)
        {
            eprintln!("Skipping {}: {reason}", candidate.id);
            continue;
        }

        // Found a workable issue — output as JSON
        let json = serde_json::to_string_pretty(candidate).context("Failed to serialize issue")?;
        println!("{json}");
        return Ok(());
    }

    eprintln!("All ready issues are exhausted or need human review");
    std::process::exit(1);
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
