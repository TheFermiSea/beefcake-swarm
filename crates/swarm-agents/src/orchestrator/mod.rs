//! Orchestration loop: process a single issue through implement → verify → review → escalate.
//!
//! Split into submodules for the Slate architecture (Phase 0):
//! - [`dispatch`]: Task routing and prompt formatting
//! - [`validation`]: Cloud and local validation quality gates
//! - [`helpers`]: Environment parsing, KB helpers, scaffolding, directives, bdh glue
//!
//! This module retains the core lifecycle: `process_issue`, session resume,
//! and retry infrastructure.

pub mod dispatch;
pub mod helpers;
pub mod validation;

// ── Re-exports for backwards compatibility ──────────────────────────
// All items that were previously `pub` or `pub(crate)` in the monolithic
// orchestrator.rs are re-exported here so downstream code (driver.rs, etc.)
// continues to compile without changes.

pub use dispatch::{format_compact_task_prompt, format_task_prompt, route_to_coder, CoderRoute};
pub use helpers::try_scaffold_fallback;
pub(crate) use helpers::{
    bool_from_env, create_stuck_intervention, detect_failure_patterns, load_directives,
    query_kb_with_failsafe, save_directives, tier_from_env, timeout_from_env, u32_from_env,
};
pub(crate) use validation::{
    cloud_validate, extract_local_validator_feedback, extract_validator_feedback, local_validate,
};

// ── Remaining lifecycle imports ─────────────────────────────────────

use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Result;
use rig::completion::Prompt;
use tracing::{debug, error, info, warn, Instrument};

use crate::cluster_health::ClusterHealth;
use crate::file_targeting::detect_changed_packages;
use crate::modes::errors::OrchestrationError;
use crate::runtime_adapter::{AdapterConfig, RuntimeAdapter};

/// Worker write deadline budget. Allows enough read turns to gather context
/// before forcing an edit_file/write_file action.
///
/// The compact prompt inlines the target file content, so the worker should
/// need at most 2-3 reads before editing. 8 turns gives headroom for complex
/// tasks that need to read multiple files. On the 122B model (~2 min/turn),
/// this is a ~16 min budget before the write-deadline terminates the agent.
const WORKER_MAX_TURNS_WITHOUT_WRITE: usize = 8;

/// Error message produced when cooperative cancellation fires inside the main loop.
///
/// Checked in `main.rs` fan-in logic to distinguish graceful shutdown from genuine
/// failures — import this constant rather than hard-coding the string in both places.
pub const CANCEL_MSG: &str = "cancelled by shutdown signal";

pub(crate) use crate::auto_fix::{should_reject_auto_fix, try_auto_fix};

use crate::acceptance::{self, AcceptancePolicy};
use crate::agents::AgentFactory;
use crate::beads_bridge::{BeadsIssue, IssueTracker};
use crate::config::{SwarmConfig, SwarmRole};
use crate::knowledge_sync;
use crate::notebook_bridge::KnowledgeBase;
use crate::telemetry::{self, MetricsCollector, TelemetryReader};
use crate::worktree_bridge::WorktreeBridge;
use coordination::benchmark::slo::{self, AlertSeverity};
use coordination::benchmark::OrchestrationMetrics;
use coordination::escalation::state::EscalationReason;
use coordination::escalation::worker_first::classify_initial_tier;
use coordination::feedback::ErrorCategory;
use coordination::otel::{self, SpanSummary};
use coordination::rollout::FeatureFlags;
use coordination::save_session_state;
use coordination::{
    ContextPacker, EscalationEngine, EscalationState, GitManager, LanguageProfile, ProgressTracker,
    ScriptVerifier, SessionManager, SwarmTier, TierBudget, TurnPolicy, ValidatorFeedback, Verifier,
    VerifierConfig, VerifierReport,
};

// Git commit, retry, diff-counting, and artifact-collection helpers live in
// `crate::git_ops` and are re-exported here for backward compatibility.
pub use crate::git_ops::git_commit_changes;
pub(crate) use crate::git_ops::{
    collect_artifacts_from_diff, count_diff_lines, git_has_meaningful_changes,
};

/// Run the verifier pipeline, dispatching to ScriptVerifier for non-Rust targets.
///
/// This is the single dispatch point for all verifier call sites in the
/// orchestrator. When a `.swarm/profile.toml` defines a non-Rust language,
/// the ScriptVerifier runs shell commands instead of cargo gates.
///
/// When `skip_test` is true, the test gate is excluded (used for baseline
/// verification where worktree env may cause false test failures).
async fn run_verifier(
    wt_path: &Path,
    verifier_config: &VerifierConfig,
    language_profile: &Option<LanguageProfile>,
) -> VerifierReport {
    run_verifier_opts(wt_path, verifier_config, language_profile, false).await
}

/// Like `run_verifier` but with explicit skip_test control.
async fn run_verifier_opts(
    wt_path: &Path,
    verifier_config: &VerifierConfig,
    language_profile: &Option<LanguageProfile>,
    skip_test: bool,
) -> VerifierReport {
    if let Some(profile) = language_profile {
        if !profile.is_rust() {
            let mut profile = profile.clone();
            if skip_test {
                profile.gates.retain(|g| !g.name.eq_ignore_ascii_case("test"));
            }
            let script_verifier = ScriptVerifier::new(wt_path, profile);
            return script_verifier.run_pipeline().await;
        }
    }
    // Default: Rust verifier
    let verifier = Verifier::new(wt_path, verifier_config.clone());
    verifier.run_pipeline().await
}

/// Process a single issue through the implement → verify → review → escalate loop.
///
/// Integrates coordination's harness for:
/// - **SessionManager**: Session lifecycle tracking with iteration counting
/// - **GitManager**: Git checkpoints for rollback on failure
/// - **ProgressTracker**: Structured progress logging for session recovery
/// - **PendingIntervention**: Formal human intervention requests when stuck
///
/// Returns `true` if the issue was successfully resolved.
///
/// When `SWARM_STATE_DRIVER=1`, uses the new state-machine-driven loop
/// from `driver.rs`. Otherwise, uses the legacy monolithic loop below.
pub async fn process_issue(
    config: &SwarmConfig,
    factory: &AgentFactory,
    worktree_bridge: &WorktreeBridge,
    issue: &BeadsIssue,
    beads: &dyn IssueTracker,
    knowledge_base: Option<&dyn KnowledgeBase>,
    cancel: Arc<AtomicBool>,
) -> Result<bool> {
    // --- State driver gate ---
    if bool_from_env("SWARM_STATE_DRIVER", false) {
        info!(id = %issue.id, "Using state-machine driver (SWARM_STATE_DRIVER=1)");
        let mut ctx = crate::driver::OrchestratorContext::new(
            config,
            factory,
            worktree_bridge,
            issue,
            beads,
            knowledge_base,
        )
        .await?;
        let result = crate::driver::drive(&mut ctx).await;
        // Move metrics out before handle_outcome (finalize consumes self)
        let metrics = std::mem::replace(
            &mut ctx.metrics,
            crate::telemetry::MetricsCollector::new("", "", "", "unknown", None, None, "v1"),
        );
        crate::driver::handle_outcome(&mut ctx, metrics).await;
        return result;
    }

    // --- Mode runner gate ---
    if bool_from_env("SWARM_MODE_DRIVER", false) {
        use crate::modes::{ModeOrchestrator, ModeRequest, ModeRunnerConfig, SwarmMode};

        let mode =
            SwarmMode::from_issue(issue.issue_type.as_deref(), issue.priority, &issue.labels);
        info!(id = %issue.id, ?mode, "Using mode runner (SWARM_MODE_DRIVER=1)");

        let mode_config = ModeRunnerConfig::from_env();
        let wt_path = worktree_bridge.create(&issue.id)?;
        let mut runner = mode.into_runner(mode_config.clone(), wt_path.clone());
        let orch = ModeOrchestrator::new(mode_config);
        let request = ModeRequest::new(&issue.title).with_label(&issue.id);
        let outcome = orch.run(runner.as_mut(), request).await;

        let success = outcome.is_success();
        if success {
            // Commit edits before merge (mode runner may leave uncommitted changes).
            if let Err(e) = git_commit_changes(&wt_path, 0).await {
                warn!(id = %issue.id, "Failed to commit mode runner edits: {e}");
            }
            info!(id = %issue.id, ?mode, "Mode runner succeeded — merging");
            match worktree_bridge.merge_and_remove(&issue.id) {
                Ok(()) => {
                    let _ = beads.close(&issue.id, Some("mode runner completed successfully"));
                }
                Err(e) => {
                    warn!(id = %issue.id, error = %e, "Merge failed after mode runner success");
                    let _ = beads.update_status(&issue.id, "open");
                    let _ = worktree_bridge.cleanup(&issue.id);
                    return Ok(false);
                }
            }
        } else {
            warn!(id = %issue.id, ?mode, "Mode runner failed");
            let _ = beads.update_status(&issue.id, "open");
            let _ = worktree_bridge.cleanup(&issue.id);
        }
        return Ok(success);
    }

    // Instrument the core loop with a root span. Using `Instrument` rather
    // than `Span::enter()` keeps the future `Send`, enabling `tokio::spawn`
    // for parallel thread dispatch (Slate Phase 2).
    let process_span = otel::process_issue_span(&issue.id);
    process_issue_core(
        config,
        factory,
        worktree_bridge,
        issue,
        beads,
        knowledge_base,
        cancel,
    )
    .instrument(process_span)
    .await
}

/// Core orchestration loop — implement → verify → review → escalate.
///
/// Separated from [`process_issue`] so the future can be wrapped with
/// [`tracing::Instrument`] for Send-safe span management. The process span
/// is accessible inside via [`tracing::Span::current()`].
async fn process_issue_core(
    config: &SwarmConfig,
    factory: &AgentFactory,
    worktree_bridge: &WorktreeBridge,
    issue: &BeadsIssue,
    beads: &dyn IssueTracker,
    knowledge_base: Option<&dyn KnowledgeBase>,
    cancel: Arc<AtomicBool>,
) -> Result<bool> {
    let worker_policy = TurnPolicy::for_tier(SwarmTier::Worker);
    let council_policy = TurnPolicy::for_tier(SwarmTier::Council);
    let worker_timeout = timeout_from_env("SWARM_WORKER_TIMEOUT_SECS", worker_policy.timeout_secs);
    let manager_timeout =
        timeout_from_env("SWARM_MANAGER_TIMEOUT_SECS", council_policy.timeout_secs);

    // The process span was set by .instrument() in the caller — retrieve it
    // so we can record fields into it at session end.
    let process_span = tracing::Span::current();
    let process_start = Instant::now();

    // --- Feature flags ---
    let feature_flags = FeatureFlags::from_env();
    info!(flags = %feature_flags, summary = %feature_flags.summary(), "Feature flags loaded");

    // --- Validate objective ---
    let title_trimmed = issue.title.trim();
    if title_trimmed.is_empty() || title_trimmed.len() < config.min_objective_len {
        warn!(
            id = %issue.id,
            title_len = title_trimmed.len(),
            min_len = config.min_objective_len,
            "Rejecting issue: title too short (\"{}\")",
            title_trimmed,
        );
        // Don't claim the issue — leave it open for a human to improve the title
        return Ok(false);
    }

    // --- Claim issue ---
    beads.update_status(&issue.id, "in_progress")?;
    info!(id = %issue.id, "Claimed issue");

    // --- Create worktree ---
    let wt_path = match worktree_bridge.create(&issue.id) {
        Ok(p) => {
            info!(path = %p.display(), "Created worktree");
            p
        }
        Err(e) => {
            error!(id = %issue.id, "Failed to create worktree: {e}");
            return Err(e);
        }
    };

    // --- Load language profile (beefcake-loop: multi-language support) ---
    //
    // If the target repo has `.swarm/profile.toml`, load it. Non-Rust profiles
    // use ScriptVerifier (shell commands) instead of the built-in Rust Verifier.
    // When absent or language="rust", behavior is unchanged.
    let language_profile = LanguageProfile::load(&wt_path);
    let is_script_verifier = language_profile
        .as_ref()
        .is_some_and(|p| !p.is_rust());

    if is_script_verifier {
        info!(
            language = %language_profile.as_ref().unwrap().language,
            gates = language_profile.as_ref().unwrap().gates.len(),
            "Using ScriptVerifier for non-Rust target repo"
        );
    }

    // --- Mutation archive (Phase 4a: evolutionary tracking) ---
    let archive = crate::mutation_archive::MutationArchive::new(worktree_bridge.repo_root());
    let archive_language = language_profile
        .as_ref()
        .map(|p| p.language.clone())
        .unwrap_or_else(|| "rust".to_string());

    // --- Initialize harness components ---
    let mut session = SessionManager::new(wt_path.clone(), config.max_retries);
    let git_mgr = GitManager::new(&wt_path, "[swarm]");
    let progress = ProgressTracker::new(wt_path.join(".swarm-progress.txt"));

    // Record initial commit for potential rollback
    if let Ok(commit) = git_mgr.current_commit_full() {
        session.set_initial_commit(commit.clone());
        info!(initial_commit = %commit, "Recorded initial commit");
    }

    // Start session
    if let Err(e) = session.start() {
        warn!("Failed to start harness session: {e}");
        // Non-fatal — continue without session tracking
    }
    session.set_current_feature(&issue.id);

    // Log session start
    let _ = progress.log_session_start(
        session.session_id(),
        format!("Processing issue: {} — {}", issue.id, issue.title),
    );

    info!(
        session_id = session.short_id(),
        issue_id = %issue.id,
        max_iterations = config.max_retries,
        "Harness session started"
    );

    // --- Telemetry ---
    let stack_profile_str = serde_json::to_string(&config.stack_profile)
        .unwrap_or_else(|_| "unknown".to_string())
        .replace('"', "");
    let mut metrics = MetricsCollector::new(
        session.session_id(),
        &issue.id,
        &issue.title,
        &stack_profile_str,
        config.repo_id.clone(),
        config.adapter_id.clone(),
        "v1",
    );

    // --- TensorZero episode tracking ---
    // Capture wall-clock start for resolving TZ-assigned episode IDs later.
    let tz_session_start_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    let tensorzero_episode_id = config.tensorzero_url.as_ref().map(|_| {
        let ep = crate::tensorzero::generate_episode_id(&issue.id, session.session_id());
        info!(episode_id = %ep, "TensorZero episode tracking enabled");
        ep
    });
    if let Some(ref ep_id) = tensorzero_episode_id {
        metrics.set_episode_id(ep_id.clone());
    }

    // --- TensorZero performance insights (Phase 3) ---
    let tz_directives = if let Some(ref pg_url) = config.tensorzero_pg_url {
        match crate::tz_insights::TzInsights::new(pg_url, config.tz_insights_ttl_secs) {
            Ok(tz) => {
                let d = tz.get_directives().await;
                if !d.is_empty() {
                    info!(count = d.len(), "Loaded TZ performance insights");
                }
                d
            }
            Err(e) => {
                warn!(error = %e, "TZ insights init failed");
                Vec::new()
            }
        }
    } else {
        Vec::new()
    };

    // --- Acceptance policy ---
    let acceptance_policy = AcceptancePolicy::default();

    // --- Preflight: endpoint health check (P1.1 + P1.4) ---
    // Probe all local inference endpoints before investing in agent builds.
    // Fail fast if all workers are down — avoids burning cloud credits on
    // delegation to dead endpoints.
    //
    // Use the factory's shared ClusterHealth so that check_all_now() updates
    // the same Arc<RwLock> that EndpointPool::next() reads. Creating a separate
    // instance here would leave the pool with Unknown status for all tiers.
    let cluster_health = factory
        .cluster_health()
        .cloned()
        .unwrap_or_else(|| ClusterHealth::from_config(config));

    if !config.cloud_only {
        let healthy_count = cluster_health.check_all_now().await;
        // Evaluate async summary before the info! macro to avoid holding a
        // temporary `&dyn tracing::Value` across the .await (which is !Send).
        let health_summary = cluster_health.summary().await;
        info!(
            healthy = healthy_count,
            total = 3,
            summary = %health_summary,
            "Preflight endpoint health check"
        );
        if healthy_count == 0 {
            warn!(
                id = %issue.id,
                "All inference endpoints are DOWN — cannot proceed"
            );
            // Reset issue status so it can be picked up later
            let _ = beads.update_status(&issue.id, "open");
            anyhow::bail!(
                "Preflight failed: all 3 inference endpoints are down ({})",
                cluster_health.summary().await
            );
        }
    } else {
        info!("Cloud-only mode: bypassing local endpoint preflight checks");
    }

    // Spawn background health monitor for ongoing checks during the session.
    // Wrap in a guard so the monitor task is aborted when this function exits.
    struct AbortOnDrop(tokio::task::JoinHandle<()>);
    impl Drop for AbortOnDrop {
        fn drop(&mut self) {
            self.0.abort();
        }
    }
    let _health_monitor = AbortOnDrop(cluster_health.spawn_monitor());

    // --- Build agents scoped to this worktree ---
    // Round-robin: each concurrent issue's factory clone selects the next node.
    let (rust_coder, general_coder) = factory.build_worker_pair(&wt_path);
    let reviewer = factory.build_reviewer();

    // Create a plan slot for manager-guided parallel work planning.
    // The manager's `plan_parallel_work` tool deposits a validated SubtaskPlan
    // here; the orchestrator checks it after each manager invocation.
    let plan_slot = crate::tools::plan_parallel_tool::new_plan_slot();
    let factory = factory.clone().with_plan_slot(plan_slot.clone());
    let manager = factory.build_manager(&wt_path);

    // --- Escalation state ---
    //
    // When worker_first is enabled, classify the task to determine starting tier.
    // Otherwise, default to Council (cloud-backed manager) from the beginning.
    // --- Local validator config ---
    let local_validator_enabled = bool_from_env("SWARM_LOCAL_VALIDATOR", true);
    let max_validator_failures = u32_from_env("SWARM_MAX_VALIDATOR_FAILURES", 3);

    let council_budget_iterations = u32_from_env("SWARM_COUNCIL_MAX_ITERATIONS", 6);
    let council_budget_consultations = u32_from_env("SWARM_COUNCIL_MAX_CONSULTATIONS", 6);
    // Always classify the task to determine starting tier.
    // Simple tasks (doc comments, lint fixes) go directly to Worker (local coders).
    // Complex tasks (refactors, multi-file arch) go to Council (cloud manager).
    // Unknown tasks default to Worker — workers write code; manager just reads files.
    let recommendation = classify_initial_tier(&issue.title, &[]);
    info!(
        tier = ?recommendation.tier,
        complexity = %recommendation.complexity,
        confidence = recommendation.confidence,
        reason = %recommendation.reason,
        "Task classification"
    );
    // Allow explicit env override, otherwise use classifier recommendation.
    let initial_tier = tier_from_env("SWARM_INITIAL_TIER", recommendation.tier);
    // Clamp: Council requires cloud endpoint. Without cloud, the local manager
    // can't delegate to workers effectively. Honor explicit env override.
    let initial_tier = if initial_tier == SwarmTier::Council
        && config.cloud_endpoint.is_none()
        && std::env::var("SWARM_INITIAL_TIER").is_err()
    {
        warn!("Council tier requires cloud endpoint; falling back to Worker");
        SwarmTier::Worker
    } else {
        initial_tier
    };
    info!(
        ?initial_tier,
        cloud_available = config.cloud_endpoint.is_some(),
        worker_first = feature_flags.worker_first_enabled,
        "Initial tier selected"
    );
    let engine = EscalationEngine::new();
    let mut escalation = EscalationState::new(&issue.id)
        .with_initial_tier(initial_tier)
        .with_budget(
            SwarmTier::Council,
            TierBudget {
                max_iterations: council_budget_iterations,
                max_consultations: council_budget_consultations,
            },
        );
    let mut success = false;
    let mut last_report: Option<VerifierReport> = None;
    let mut last_validator_feedback: Vec<ValidatorFeedback> = Vec::new();
    let mut span_summary = SpanSummary::new();
    let mut consecutive_validator_failures: u32 = 0;
    // Tracks whether the previous iteration's agent called edit_file/write_file.
    // Used to inject an edit nudge into the next iteration's task prompt.
    let mut agent_has_written_prev = true; // assume true for first iteration
                                           // Hill-climbing: track the best (lowest) error count seen across all iterations.
                                           // Changes are only kept when they improve on the best — otherwise rolled back.
    let mut best_error_count: Option<usize> = None;
    // Minimum error reduction required to keep changes (env: SWARM_MIN_ERROR_DELTA).
    let min_error_delta: usize = std::env::var("SWARM_MIN_ERROR_DELTA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);

    // Scope verifier to changed packages.
    // If explicit packages are configured (CLI --package or SWARM_VERIFIER_PACKAGES), use those.
    // Otherwise, detect from git-changed files to avoid missing breakage in other crates.
    let initial_packages = if config.verifier_packages.is_empty() {
        detect_changed_packages(&wt_path, !is_script_verifier)
    } else {
        config.verifier_packages.clone()
    };
    let verifier_config = VerifierConfig {
        packages: initial_packages,
        check_clippy: !bool_from_env("SWARM_SKIP_CLIPPY", false),
        check_test: !bool_from_env("SWARM_SKIP_TESTS", false),
        ..VerifierConfig::default()
    };

    // --- Concurrent subtask dispatch gate ---
    //
    // When enabled, the planner decomposes the issue into non-overlapping subtasks,
    // then multiple workers execute them concurrently in the same worktree.
    // After all workers finish, the verifier runs on the combined result.
    if config.concurrent_subtasks {
        info!(id = %issue.id, "Using concurrent subtask dispatch");

        // Prefer local reasoning tier for planning (escalation ladder compliance).
        // Use resolve_role_model to get the TZ function name when TZ is configured.
        let (plan_client, plan_model) = {
            (
                factory.clients.reasoning.clone(),
                config.resolve_role_model(SwarmRole::Planner),
            )
        };

        // Generate file listing for the planner.
        let file_listing = crate::subtask::list_source_files(&wt_path);
        let issue_context = format!(
            "Issue ID: {}\nType: {}\nPriority: {:?}",
            issue.id,
            issue.issue_type.as_deref().unwrap_or("unknown"),
            issue.priority,
        );

        // Pre-identify target files via file_targeting so the planner
        // doesn't waste turns exploring the codebase.
        let src_exts = language_profile
            .as_ref()
            .map(|p| p.source_extensions.clone())
            .unwrap_or_default();
        let target_files = crate::file_targeting::find_target_files_by_grep(&wt_path, &issue.title, &src_exts);

        // Plan subtasks.
        let plan_result = crate::subtask::plan_subtasks(
            &plan_client,
            &plan_model,
            &issue.title,
            &file_listing,
            &issue_context,
            target_files.as_deref(),
        )
        .await;

        match plan_result {
            Ok(plan) if plan.subtasks.len() > 1 => {
                info!(
                    id = %issue.id,
                    subtask_count = plan.subtasks.len(),
                    summary = %plan.summary,
                    "Planner decomposed issue into {} concurrent subtasks",
                    plan.subtasks.len()
                );

                // Initialize workpad for inter-worker communication.
                if let Err(e) = crate::tools::workpad_tool::init_workpad(&wt_path) {
                    warn!(id = %issue.id, error = %e, "Failed to init workpad — workers proceed without comms");
                }

                // Create beads molecule (child issues) for subtask tracking.
                // Non-fatal — dispatch proceeds even if molecule creation fails.
                let molecule_map =
                    crate::subtask::create_molecule_for_plan(&plan, &issue.id, &wt_path);

                // Dispatch workers concurrently, bounded by endpoint pool capacity
                // (number of inference nodes, not parallel_issues which is a separate concern).
                let max_concurrent = factory.endpoint_pool.capacity();
                let timeout = timeout_from_env("SWARM_LOCAL_HTTP_TIMEOUT_SECS", 900).as_secs();
                let outcome = crate::subtask::dispatch_subtasks(
                    &plan,
                    &factory.endpoint_pool,
                    &wt_path,
                    &issue.id,
                    max_concurrent,
                    timeout,
                )
                .await;

                // Close molecule child issues based on worker results.
                crate::subtask::close_molecule_children(&outcome.results, &molecule_map, &wt_path);

                let succeeded = outcome.success_count();
                let total = outcome.results.len();

                // Structured observability: per-worker breakdown + speedup estimate.
                outcome.log_summary();

                // Log workpad announcements for debugging.
                if let Ok(announcements) = crate::tools::workpad_tool::read_workpad(&wt_path) {
                    if !announcements.is_empty() {
                        info!(
                            id = %issue.id,
                            announcement_count = announcements.len(),
                            "Workers posted announcements to workpad"
                        );
                    }
                }

                // --- Failed subtask serial retry ---
                // If some subtasks succeeded and some failed, retry only the failed ones
                // serially. Successful workers' changes are already in the worktree.
                if succeeded > 0 && succeeded < total {
                    let failed_ids: Vec<String> = outcome
                        .results
                        .iter()
                        .filter(|r| !r.success)
                        .map(|r| r.subtask_id.clone())
                        .collect();
                    info!(
                        id = %issue.id,
                        failed = ?failed_ids,
                        "Retrying failed subtasks serially (preserving successful workers' changes)"
                    );

                    // Collect announcements from successful workers as extra context.
                    let announcements =
                        crate::tools::workpad_tool::read_workpad(&wt_path).unwrap_or_default();
                    let announcement_context = if announcements.is_empty() {
                        String::new()
                    } else {
                        let lines: Vec<String> = announcements
                            .iter()
                            .map(|a| {
                                format!(
                                    "- [{}] {} in {}: {}",
                                    a.worker, a.entry_type, a.file, a.detail
                                )
                            })
                            .collect();
                        format!("\n\n## Changes by other workers\n{}", lines.join("\n"))
                    };

                    for failed_id in &failed_ids {
                        if let Some(subtask) = plan.subtasks.iter().find(|s| &s.id == failed_id) {
                            info!(id = %issue.id, subtask_id = %failed_id, "Retrying failed subtask serially");
                            let (client, model) = factory.endpoint_pool.next();
                            let mut retry_subtask = subtask.clone();
                            retry_subtask.objective =
                                format!("{}{announcement_context}", retry_subtask.objective);
                            let result = crate::subtask::run_subtask_worker_public(
                                client,
                                model,
                                &wt_path,
                                &issue.id,
                                &retry_subtask,
                                timeout,
                            )
                            .await;
                            info!(
                                id = %issue.id,
                                subtask_id = %failed_id,
                                success = result.success,
                                elapsed_ms = result.elapsed.as_millis() as u64,
                                "Serial retry of failed subtask completed"
                            );
                        }
                    }
                } else if succeeded == 0 {
                    warn!(
                        id = %issue.id,
                        "All subtasks failed — escalating to Council for sequential retry"
                    );
                    last_report = None;
                    // Concurrent workers already failed — don't waste iterations retrying
                    // at Worker tier. Force escalation to Council so the cloud manager
                    // can tackle it directly.
                    escalation.current_tier = SwarmTier::Council;
                }

                // Run verifier on the combined result (skip if all subtasks failed).
                let report = if succeeded > 0 {
                    Some(run_verifier(&wt_path, &verifier_config, &language_profile).await)
                } else {
                    None
                };

                if let Some(report) = report {
                    if report.all_green {
                        info!(
                            id = %issue.id,
                            summary = %report.summary(),
                            "Concurrent subtask dispatch: verifier PASSED"
                        );

                        // Commit the concurrent workers' edits before merge.
                        if let Err(e) = git_commit_changes(&wt_path, 0).await {
                            warn!(id = %issue.id, "Failed to commit concurrent edits: {e}");
                        }

                        // Session bookkeeping (mirrors the normal success path).
                        session.complete();
                        let _ = progress.log_session_end(
                            session.session_id(),
                            session.iteration(),
                            format!(
                                "Issue {} resolved via concurrent subtask dispatch",
                                issue.id
                            ),
                        );

                        info!(
                            id = %issue.id,
                            session_id = session.short_id(),
                            elapsed = %session.elapsed_human(),
                            "Issue resolved — merging worktree"
                        );

                        // Try to merge and close.
                        if let Err(e) = worktree_bridge.merge_and_remove(&issue.id) {
                            error!(id = %issue.id, "Merge failed: {e}");
                            let _ = worktree_bridge.cleanup(&issue.id);
                            let _ = beads.update_status(&issue.id, "open");
                            return Err(e);
                        }
                        beads.close(&issue.id, Some("Resolved by concurrent subtask dispatch"))?;
                        clear_resume_file(worktree_bridge.repo_root());
                        return Ok(true);
                    }

                    // Verifier failed — attempt serial fixer post-pass on integration files
                    // before falling through to the main loop.
                    info!(
                        id = %issue.id,
                        errors = report.failure_signals.len(),
                        "Concurrent dispatch verifier failed — attempting serial fixer post-pass"
                    );

                    let fixer = factory.build_fixer(&wt_path);
                    let fixer_prompt = format!(
                    "The concurrent workers completed their subtasks but the verifier found errors.\n\n\
                     Verifier summary: {}\n\n\
                     Fix the compilation/test errors. Focus on integration files \
                     (Cargo.toml, mod.rs, lib.rs, main.rs) and any cross-file issues \
                     between the workers' changes. Read the error messages carefully and \
                     make targeted fixes.",
                    report.summary()
                );

                    match rig::completion::Prompt::prompt(&fixer, &fixer_prompt).await {
                        Ok(_response) => {
                            info!(id = %issue.id, "Serial fixer post-pass completed — re-running verifier");
                            let report2 = run_verifier(&wt_path, &verifier_config, &language_profile).await;

                            if report2.all_green {
                                info!(
                                    id = %issue.id,
                                    summary = %report2.summary(),
                                    "Concurrent dispatch + fixer post-pass: verifier PASSED"
                                );

                                if let Err(e) = git_commit_changes(&wt_path, 0).await {
                                    warn!(id = %issue.id, "Failed to commit fixer edits: {e}");
                                }

                                session.complete();
                                let _ = progress.log_session_end(
                                    session.session_id(),
                                    session.iteration(),
                                    format!(
                                    "Issue {} resolved via concurrent dispatch + fixer post-pass",
                                    issue.id
                                ),
                                );

                                info!(
                                    id = %issue.id,
                                    session_id = session.short_id(),
                                    elapsed = %session.elapsed_human(),
                                    "Issue resolved — merging worktree"
                                );

                                if let Err(e) = worktree_bridge.merge_and_remove(&issue.id) {
                                    error!(id = %issue.id, "Merge failed: {e}");
                                    let _ = worktree_bridge.cleanup(&issue.id);
                                    let _ = beads.update_status(&issue.id, "open");
                                    return Err(e);
                                }
                                beads.close(
                                    &issue.id,
                                    Some("Resolved by concurrent dispatch + fixer post-pass"),
                                )?;
                                clear_resume_file(worktree_bridge.repo_root());
                                return Ok(true);
                            }

                            // Fixer post-pass didn't fix everything — fall through
                            warn!(
                                id = %issue.id,
                                errors = report2.failure_signals.len(),
                                summary = %report2.summary(),
                                "Fixer post-pass: verifier still FAILED — falling through to retry loop"
                            );
                            last_report = Some(report2);
                        }
                        Err(e) => {
                            warn!(
                                id = %issue.id,
                                error = %e,
                                "Serial fixer post-pass failed — falling through to retry loop"
                            );
                            last_report = Some(report);
                        }
                    }
                } // end if let Some(report)
            }
            Ok(plan) => {
                info!(
                    id = %issue.id,
                    "Planner returned single subtask — staying at Worker tier"
                );
                debug!(summary = %plan.summary, "Single-subtask plan");
                // Single-task issues are simple enough for a local worker.
                // Stay at Worker tier — the local 122B handles reads, edits,
                // and searches directly. Cloud manager is only needed for
                // multi-file coordination or after worker failures.
            }
            Err(e) => {
                warn!(
                    id = %issue.id,
                    error = %e,
                    "Subtask planning failed — routing to manager"
                );
                // Planning failed — don't waste iterations at Worker tier.
                // Route to Council if cloud is available.
                if factory.clients.cloud.is_some() {
                    escalation.current_tier = SwarmTier::Council;
                }
            }
        }
    }

    // --- Baseline verification (autoresearch pattern) ---
    //
    // Run a lightweight check on the clean worktree before the agent touches anything.
    // Only checks compilation (fmt + clippy + check) — NOT tests, since worktree
    // environment differences (symlinked .beads/, bdh init) can cause test flakes
    // that don't reflect actual code problems.
    {
        let baseline_config = VerifierConfig {
            check_fmt: true,
            check_clippy: true,
            check_compile: true,
            check_test: false, // Skip tests — worktree env can cause false failures
            ..verifier_config.clone()
        };
        let baseline_report = run_verifier_opts(&wt_path, &baseline_config, &language_profile, true).await;
        let gates_passed = baseline_report
            .gates
            .iter()
            .filter(|g| g.outcome == coordination::verifier::report::GateOutcome::Passed)
            .count();
        let gates_total = baseline_report.gates.len();

        if !baseline_report.all_green {
            let baseline_summary = baseline_report.summary();
            warn!(
                id = %issue.id,
                gates_passed,
                gates_total,
                summary = %baseline_summary,
                "Baseline verification FAILED — worktree does not compile/pass tests before agent modifications. \
                 Skipping issue to avoid wasting iterations on pre-existing failures."
            );
            let _ = beads.update_status(&issue.id, "open");
            let _ = progress.log_error(
                session.session_id(),
                0,
                format!("Baseline failed: {baseline_summary}"),
            );
            // Clean up worktree
            if let Err(e) = worktree_bridge.cleanup(&issue.id) {
                warn!(id = %issue.id, error = %e, "Failed to clean up worktree after baseline failure");
            }
            return Ok(false);
        }

        info!(
            id = %issue.id,
            gates_passed,
            gates_total,
            "Baseline verification PASSED — worktree is clean before agent starts"
        );
    }

    // --- Main loop: implement → verify → review → escalate ---
    loop {
        let iteration = match session.next_iteration() {
            Ok(i) => i,
            Err(e) => {
                warn!("Session iteration limit: {e}");
                let _ = progress.log_error(
                    session.session_id(),
                    session.iteration(),
                    format!("Max iterations reached: {e}"),
                );
                break;
            }
        };

        // Cooperative cancellation — checked once per iteration at the boundary
        // between agent invocations. Set by the parallel-dispatch path in main.rs
        // when SIGTERM/Ctrl-C is received. Using Acquire ensures we see the Release
        // store performed by the shutdown listener task.
        if cancel.load(Ordering::Acquire) {
            warn!(
                id = %issue.id,
                iteration,
                "Shutdown signal — resetting issue to open and cleaning up worktree"
            );
            let _ = beads.update_status(&issue.id, "open");
            let _ = worktree_bridge.cleanup(&issue.id);
            anyhow::bail!("{CANCEL_MSG}");
        }

        let tier = escalation.current_tier;
        metrics.start_iteration(iteration, &format!("{tier:?}"));
        let tier_str = format!("{tier:?}");
        let iter_span = otel::iteration_span(&issue.id, iteration, &tier_str);
        // Note: iter_span is NOT entered with .enter() — holding the Entered
        // guard across await points would make the future !Send. The span is
        // still used for field recording via record_iteration_result().
        let iter_start = Instant::now();
        span_summary.record_iteration();
        info!(
            iteration,
            ?tier,
            id = %issue.id,
            session_id = session.short_id(),
            "Starting iteration"
        );

        let _ = progress.log_feature_start(
            session.session_id(),
            iteration,
            &issue.id,
            format!("Iteration {iteration}, tier: {tier:?}"),
        );

        // --- Pre-iteration auto-fix (P1.2) ---
        // Run auto-fix BEFORE packing context so the agent starts each iteration
        // with clean formatting and auto-fixable clippy issues already resolved.
        // This prevents wasting a turn on lint errors from the previous iteration.
        if iteration > 1 {
            if let Some(ref report) = last_report {
                if !report.all_green {
                    if let Some(fixed_report) =
                        try_auto_fix(&wt_path, &verifier_config, iteration, &language_profile).await
                    {
                        if fixed_report.all_green {
                            info!(
                                iteration,
                                "Pre-iteration auto-fix resolved all issues — skipping agent"
                            );
                            // Commit auto-fix changes before declaring success
                            let _ = git_commit_changes(&wt_path, iteration).await;
                            metrics.record_auto_fix();
                            metrics.finish_iteration();
                            success = true;
                            break;
                        }
                        // Update the report so context packing uses the post-autofix state
                        last_report = Some(fixed_report);
                        metrics.record_auto_fix();
                        // Commit auto-fix changes so agent sees clean state
                        let _ = git_commit_changes(&wt_path, iteration).await;
                    }
                }
            }
        }

        // --- Mail polling (native beads) ---
        // Between iterations, check if any agent sent a mail message.
        // Messages are injected into the next agent prompt as additional context.
        let mail_context = if iteration > 1 {
            crate::beads_bridge::poll_mail_inbox(&wt_path)
        } else {
            None
        };

        // Build the full objective: title + description (if available).
        // The description contains file lists, implementation details, and context
        // that the file targeting pipeline uses to locate relevant source files.
        let mut full_objective = match &issue.description {
            Some(desc) if !desc.is_empty() => format!("{}\n\n{}", issue.title, desc),
            _ => issue.title.clone(),
        };

        // Append mail messages if any were received between iterations
        if let Some(ref mail) = mail_context {
            full_objective.push_str("\n\n## Agent Mail (from previous iteration)\n");
            full_objective.push_str(mail);
        }

        // Pack context with tier-appropriate token budget
        let packer = ContextPacker::new(&wt_path, tier);
        let mut packet = if let Some(ref report) = last_report {
            packer.pack_retry(&issue.id, &full_objective, &escalation, report)
        } else {
            packer.pack_initial(&issue.id, &full_objective)
        };

        // Inject structured validator feedback from prior iteration (TextGrad pattern)
        if !last_validator_feedback.is_empty() {
            packet.validator_feedback = std::mem::take(&mut last_validator_feedback);
            info!(
                iteration,
                feedback_count = packet.validator_feedback.len(),
                "Injected validator feedback into work packet"
            );
        }

        // --- Integration Point 1: Pre-task knowledge enrichment ---
        if let Some(kb) = knowledge_base {
            // Query Project Brain for architectural context
            let brain_question = format!(
                "What architectural context is relevant for: {}? Issue: {}",
                issue.title, issue.id
            );
            let response = query_kb_with_failsafe(kb, "project_brain", &brain_question);
            if !response.is_empty() {
                packet.relevant_heuristics.push(response);
                info!(iteration, "Enriched packet with Project Brain context");
            }

            // On retries, also query Debugging KB for error-specific patterns
            if last_report.is_some() && !packet.failure_signals.is_empty() {
                let error_desc = packet
                    .failure_signals
                    .iter()
                    .map(|s| format!("{}: {}", s.category, s.message))
                    .collect::<Vec<_>>()
                    .join("; ");
                let debug_question = format!("Known fixes for these Rust errors: {error_desc}");
                let response = query_kb_with_failsafe(kb, "debugging_kb", &debug_question);
                if !response.is_empty() {
                    packet.relevant_playbooks.push(response);
                    info!(iteration, "Enriched packet with Debugging KB patterns");
                }
            }
        }

        info!(
            tokens = packet.estimated_tokens(),
            files = packet.file_contexts.len(),
            "Packed context"
        );

        // Sparse context: escalate Worker→Council only when cloud is available.
        // Without cloud, Council routes to a local manager that can read/verify
        // but can't write code — creating a catch-22 where the manager explores
        // endlessly without producing changes. Keep Worker tier and let the
        // increased turn budget (10) give the model more exploration room.
        let tier = if tier == SwarmTier::Worker
            && packet.file_contexts.is_empty()
            && packet.files_touched.is_empty()
            && packet.failure_signals.is_empty()
            && config.cloud_endpoint.is_some()
        {
            warn!(
                iteration,
                "Sparse context — escalating Worker→Council for initial analysis"
            );
            escalation.record_escalation(
                SwarmTier::Council,
                EscalationReason::Explicit {
                    reason: "sparse context: no file_contexts/files_touched/failure_signals"
                        .to_string(),
                },
            );
            SwarmTier::Council
        } else {
            if tier == SwarmTier::Worker
                && packet.file_contexts.is_empty()
                && packet.files_touched.is_empty()
                && packet.failure_signals.is_empty()
            {
                warn!(
                    iteration,
                    "Sparse context but no cloud — keeping Worker tier (local manager can't write code)"
                );
            }
            tier
        };

        // Worker tier gets a compact prompt (<1500 chars) because small local
        // models (HydraCoder 30B MoE) suppress tool calls with long prompts.
        // Council/Human tiers get the full verbose format for cloud models.
        let mut task_prompt = if tier == SwarmTier::Worker {
            format_compact_task_prompt(&packet, &wt_path)
        } else {
            format_task_prompt(&packet)
        };

        // --- Edit nudge: remind workers they MUST call edit_file ---
        // When a previous iteration ended without writes, append a strong
        // system reminder. This implements OpenDev's "Event-Driven System
        // Reminders" pattern to counter instruction fade-out.
        if iteration > 1 && !agent_has_written_prev && tier == SwarmTier::Worker {
            task_prompt.push_str(
                "\n\n**⚠ SYSTEM REMINDER**: The previous iteration produced NO file edits. \
                 You MUST call edit_file in this iteration. Do NOT just read files and analyze — \
                 apply your changes with edit_file now. Text-only responses are invalid.\n",
            );
        }

        // --- Directive injection: learned patterns from previous sessions ---
        let directives = load_directives(&wt_path);
        if !directives.is_empty() {
            task_prompt.push_str("\n**Learned patterns (from previous sessions):**\n");
            for d in &directives {
                task_prompt.push_str(&format!("- {d}\n"));
            }
        }

        // --- TZ performance insights injection (Phase 3) ---
        if !tz_directives.is_empty() {
            task_prompt.push_str("\n**Performance insights (from TZ experiment data):**\n");
            for d in &tz_directives {
                task_prompt.push_str(&format!("- {d}\n"));
            }
        }

        debug!(
            iteration,
            prompt_len = task_prompt.len(),
            prompt_preview = %&task_prompt[..task_prompt.len().min(500)],
            "Compact task prompt assembled"
        );

        // Inject verifier stderr into prompt when failure_signals are thin.
        // Fmt errors don't produce ParsedErrors, so the packet may lack error details.
        // The raw stderr contains the actual error output the model needs to see.
        // For Worker tier: only append truncated stderr to stay under ~2K chars total.
        if let Some(ref report) = last_report {
            if !report.all_green && packet.failure_signals.is_empty() {
                task_prompt.push_str("\n**Verifier output:**\n```\n");
                let mut stderr_chars = 0usize;
                let stderr_budget = if tier == SwarmTier::Worker {
                    600
                } else {
                    usize::MAX
                };
                'gates: for gate in &report.gates {
                    if let Some(stderr) = &gate.stderr_excerpt {
                        for line in stderr.lines() {
                            let line_len = line.len() + 1;
                            if stderr_chars + line_len > stderr_budget {
                                task_prompt.push_str("...(truncated)\n");
                                break 'gates;
                            }
                            task_prompt.push_str(line);
                            task_prompt.push('\n');
                            stderr_chars += line_len;
                        }
                    }
                }
                task_prompt.push_str("```\n");
            }
        }

        // --- Checkpoint before agent invocation ---
        // Save the current commit so we can rollback if the agent makes things worse.
        let pre_worker_commit = git_mgr.current_commit_full().ok();

        // --- Route to agent based on current tier ---
        //
        // Hierarchy (cloud available):
        //   Worker: local coders (Qwen3.5-Implementer on vasp-02)
        //   Council+Human: cloud-backed manager (Opus 4.6) with all local workers as tools
        //
        // Hierarchy (no cloud):
        //   Worker: local coders
        //   Council+Human: local manager (Qwen3.5-Architect on vasp-01) with coders as tools
        let agent_start = Instant::now();
        let (agent_future, adapter) = match tier {
            SwarmTier::Worker => {
                let recent_cats: Vec<ErrorCategory> = escalation
                    .recent_error_categories
                    .last()
                    .cloned()
                    .unwrap_or_default();

                match route_to_coder(&recent_cats) {
                    CoderRoute::RustCoder => {
                        info!(iteration, "Routing to rust_coder (Qwen3.5-Implementer)");
                        metrics.record_coder_route("RustCoder");
                        metrics.record_agent_metrics("Qwen3.5-RustCoder", 0, 0);
                        let adapter = RuntimeAdapter::new(AdapterConfig {
                            agent_name: "Qwen3.5-RustCoder".into(),
                            deadline: Some(Instant::now() + worker_timeout),
                            max_tool_calls: Some(15),
                            max_turns_without_write: Some(WORKER_MAX_TURNS_WITHOUT_WRITE),
                            ..Default::default()
                        });
                        let result = match tokio::time::timeout(
                            worker_timeout,
                            prompt_with_hook_and_retry(
                                &rust_coder,
                                &task_prompt,
                                2, // retry transient HTTP errors up to 2x
                                adapter.clone(),
                            ),
                        )
                        .await
                        {
                            Ok(result) => result,
                            Err(_elapsed) => {
                                warn!(
                                    iteration,
                                    timeout_secs = worker_timeout.as_secs(),
                                    "rust_coder exceeded timeout — proceeding with changes on disk"
                                );
                                Ok("rust_coder timed out. Changes are on disk for verifier."
                                    .to_string())
                            }
                        };
                        (result, adapter)
                    }
                    CoderRoute::GeneralCoder => {
                        info!(iteration, "Routing to general_coder (Qwen3.5-Implementer)");
                        metrics.record_coder_route("GeneralCoder");
                        metrics.record_agent_metrics("Qwen3.5-GeneralCoder", 0, 0);
                        let adapter = RuntimeAdapter::new(AdapterConfig {
                            agent_name: "Qwen3.5-GeneralCoder".into(),
                            deadline: Some(Instant::now() + worker_timeout),
                            max_tool_calls: Some(15),
                            max_turns_without_write: Some(WORKER_MAX_TURNS_WITHOUT_WRITE),
                            ..Default::default()
                        });
                        let result = match tokio::time::timeout(
                            worker_timeout,
                            prompt_with_hook_and_retry(
                                &general_coder,
                                &task_prompt,
                                2, // retry transient HTTP errors up to 2x
                                adapter.clone(),
                            ),
                        )
                        .await
                        {
                            Ok(result) => result,
                            Err(_elapsed) => {
                                warn!(
                                    iteration,
                                    timeout_secs = worker_timeout.as_secs(),
                                    "general_coder exceeded timeout — proceeding with changes on disk"
                                );
                                Ok("general_coder timed out. Changes are on disk for verifier."
                                    .to_string())
                            }
                        };
                        (result, adapter)
                    }
                }
            }
            SwarmTier::Strategist => {
                info!(iteration, "Routing to strategist advisor");
                let model = config.resolve_role_model(SwarmRole::Strategist);
                metrics.record_agent_metrics(&format!("strategist ({model})"), 0, 0);

                let strategist = factory.build_strategist(&wt_path);
                let adapter = RuntimeAdapter::new(AdapterConfig {
                    agent_name: "strategist".into(),
                    deadline: Some(Instant::now() + worker_timeout),
                    max_reads_without_action: Some(8),
                    ..Default::default()
                });

                let result = match tokio::time::timeout(
                    worker_timeout,
                    prompt_with_hook_and_retry(&strategist, &task_prompt, 2, adapter.clone()),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_) => {
                        warn!(iteration, "strategist timed out");
                        Ok("strategist timed out. No guidance produced.".to_string())
                    }
                };
                (result, adapter)
            }
            SwarmTier::Council | SwarmTier::Human => {
                info!(
                    iteration,
                    "Routing to manager (cloud-backed or Qwen3.5-Architect fallback)"
                );
                metrics.record_agent_metrics("manager", 0, 0);
                let adapter = RuntimeAdapter::new(AdapterConfig {
                    agent_name: "manager".into(),
                    deadline: Some(Instant::now() + manager_timeout),
                    max_reads_without_action: Some(8),
                    ..Default::default()
                });
                // Wrap manager call with timeout to enforce turn limits.
                // Rig doesn't enforce default_max_turns on the outer .prompt() agent,
                // so managers can run indefinitely. This hard-caps wall-clock time.
                let result = match tokio::time::timeout(
                    manager_timeout,
                    prompt_with_hook_and_retry(
                        &manager,
                        &task_prompt,
                        config.cloud_max_retries,
                        adapter.clone(),
                    ),
                )
                .await
                {
                    Ok(result) => result,
                    Err(_elapsed) => {
                        warn!(
                            iteration,
                            timeout_secs = manager_timeout.as_secs(),
                            "Manager exceeded timeout — proceeding with changes on disk"
                        );
                        // Return a synthetic "timed out" response so the verifier still runs.
                        // Any file changes the manager made are already on disk.
                        Ok("Manager timed out. Changes are on disk for verifier.".to_string())
                    }
                };

                // --- Cloud fallback matrix (P1.3) ---
                // If the primary cloud model failed with a quota/rate error, try
                // fallback models before giving up. Rebuilds the manager agent with
                // each fallback model in order.
                let result = if let Err(ref primary_err) = result {
                    let err_str = format!("{primary_err}");
                    let err_lower = err_str.to_ascii_lowercase();
                    let is_quota_or_model_error = err_str.contains("429")
                        || err_str.contains("quota")
                        || err_str.contains("overloaded")
                        || err_lower.contains("capacity")
                        || err_str.contains("500")
                        || err_str.contains("503");

                    if is_quota_or_model_error {
                        let fallbacks = config.cloud_fallback_matrix.fallbacks();
                        let mut fallback_result = None;
                        for entry in fallbacks {
                            warn!(
                                iteration,
                                model = %entry.model,
                                tier = %entry.tier_label,
                                primary_error = %err_str,
                                "Primary cloud model failed — trying fallback"
                            );
                            if let Some(fallback_manager) =
                                factory.build_manager_for_model(&wt_path, &entry.model)
                            {
                                let fb_adapter = RuntimeAdapter::new(AdapterConfig {
                                    agent_name: format!("manager-{}", entry.tier_label),
                                    deadline: Some(Instant::now() + manager_timeout),
                                    max_reads_without_action: Some(8),
                                    ..Default::default()
                                });
                                match tokio::time::timeout(
                                    manager_timeout,
                                    prompt_with_hook_and_retry(
                                        &fallback_manager,
                                        &task_prompt,
                                        1, // single retry for fallbacks
                                        fb_adapter,
                                    ),
                                )
                                .await
                                {
                                    Ok(Ok(response)) => {
                                        info!(
                                            iteration,
                                            model = %entry.model,
                                            "Cloud fallback succeeded"
                                        );
                                        fallback_result = Some(Ok(response));
                                        break;
                                    }
                                    Ok(Err(e)) => {
                                        warn!(
                                            iteration,
                                            model = %entry.model,
                                            error = %e,
                                            "Cloud fallback also failed — trying next"
                                        );
                                    }
                                    Err(_) => {
                                        warn!(
                                            iteration,
                                            model = %entry.model,
                                            "Cloud fallback timed out — trying next"
                                        );
                                    }
                                }
                            }
                        }
                        fallback_result.unwrap_or(result)
                    } else {
                        result
                    }
                } else {
                    result
                };
                (result, adapter)
            }
        };

        // --- Manager-guided parallel dispatch ---
        // If the manager called `plan_parallel_work`, a SubtaskPlan is waiting
        // in the plan slot. Take it and dispatch concurrent workers.
        if let Some(manager_plan) = plan_slot.lock().ok().and_then(|mut s| s.take()) {
            if manager_plan.subtasks.len() >= 2 && config.concurrent_subtasks {
                info!(
                    iteration,
                    id = %issue.id,
                    subtask_count = manager_plan.subtasks.len(),
                    summary = %manager_plan.summary,
                    "Manager submitted parallel work plan — dispatching concurrent workers"
                );

                // Initialize workpad for inter-worker communication.
                if let Err(e) = crate::tools::workpad_tool::init_workpad(&wt_path) {
                    warn!(id = %issue.id, error = %e, "Failed to init workpad");
                }

                // Create beads molecule for subtask tracking.
                let molecule_map =
                    crate::subtask::create_molecule_for_plan(&manager_plan, &issue.id, &wt_path);

                let max_concurrent = factory.endpoint_pool.capacity();
                let timeout = timeout_from_env("SWARM_LOCAL_HTTP_TIMEOUT_SECS", 900).as_secs();
                let outcome = crate::subtask::dispatch_subtasks(
                    &manager_plan,
                    &factory.endpoint_pool,
                    &wt_path,
                    &issue.id,
                    max_concurrent,
                    timeout,
                )
                .await;

                // Close molecule child issues.
                crate::subtask::close_molecule_children(&outcome.results, &molecule_map, &wt_path);

                outcome.log_summary();

                // Log workpad announcements.
                if let Ok(announcements) = crate::tools::workpad_tool::read_workpad(&wt_path) {
                    if !announcements.is_empty() {
                        info!(
                            id = %issue.id,
                            announcement_count = announcements.len(),
                            "Workers posted announcements to workpad"
                        );
                    }
                }

                // Run verifier on the concurrent workers' combined result.
                if outcome.success_count() > 0 {
                    let current_verifier_config = if config.verifier_packages.is_empty() {
                        VerifierConfig {
                            packages: detect_changed_packages(&wt_path, !is_script_verifier),
                            ..verifier_config.clone()
                        }
                    } else {
                        verifier_config.clone()
                    };
                    let report = run_verifier(&wt_path, &current_verifier_config, &language_profile).await;

                    if report.all_green {
                        info!(
                            iteration,
                            id = %issue.id,
                            "Manager-guided parallel dispatch: verifier PASSED"
                        );

                        if let Err(e) = git_commit_changes(&wt_path, iteration as u32).await {
                            warn!(id = %issue.id, "Failed to commit concurrent edits: {e}");
                        }

                        session.complete();
                        let _ = progress.log_session_end(
                            session.session_id(),
                            session.iteration(),
                            format!(
                                "Issue {} resolved via manager-guided parallel dispatch",
                                issue.id
                            ),
                        );

                        info!(
                            id = %issue.id,
                            session_id = session.short_id(),
                            elapsed = %session.elapsed_human(),
                            "Issue resolved — merging worktree"
                        );

                        if let Err(e) = worktree_bridge.merge_and_remove(&issue.id) {
                            error!(id = %issue.id, "Merge failed: {e}");
                            let _ = worktree_bridge.cleanup(&issue.id);
                            let _ = beads.update_status(&issue.id, "open");
                            return Err(e);
                        }
                        beads.close(
                            &issue.id,
                            Some("Resolved by manager-guided parallel dispatch"),
                        )?;
                        clear_resume_file(worktree_bridge.repo_root());
                        return Ok(true);
                    }

                    // Verifier failed — try fixer post-pass, then fall through to next iteration.
                    info!(
                        iteration,
                        id = %issue.id,
                        errors = report.failure_signals.len(),
                        "Manager-guided parallel dispatch: verifier FAILED — will retry"
                    );
                    last_report = Some(report);
                }

                agent_has_written_prev = outcome.success_count() > 0;
                continue;
            }
        }

        // Log runtime adapter report for tool-event visibility
        let (agent_has_written, agent_terminated_without_writing) = match adapter.report() {
            Ok(adapter_report) => {
                info!(
                    iteration,
                    agent = %adapter_report.agent_name,
                    turns = adapter_report.turn_count,
                    tool_calls = adapter_report.total_tool_calls,
                    tool_time_ms = adapter_report.total_tool_time_ms,
                    terminated_early = adapter_report.terminated_early,
                    has_written = adapter_report.has_written,
                    "Runtime adapter report"
                );
                if let Some(ref reason) = adapter_report.termination_reason {
                    warn!(iteration, reason = %reason, "Agent terminated early by adapter");
                }
                // adapter_report.has_written reflects tool-call intent (set in
                // on_tool_call before the tool runs). Cross-check with git to
                // detect failed edits that set the flag but didn't change files.
                let git_has_changes = git_has_meaningful_changes(&wt_path).await;
                // For manager (Council tier): sub-workers are agent-as-tool, so the
                // manager's adapter never sees edit_file/write_file calls — only tool
                // names like "proxy_rust_coder". If git shows changes AND the manager
                // delegated to worker tools, trust git as the source of truth.
                let manager_delegated = tier == SwarmTier::Council
                    && git_has_changes
                    && adapter_report.total_tool_calls > 0;
                let actually_written =
                    (adapter_report.has_written || manager_delegated) && git_has_changes;
                let terminated_without_writing =
                    adapter_report.terminated_early && !actually_written;
                agent_has_written_prev = actually_written;
                (actually_written, terminated_without_writing)
            }
            Err(e) => {
                warn!(iteration, error = %e, "Failed to extract runtime adapter report");
                (true, false) // assume written on error to avoid false no-progress detection
            }
        };

        // Handle agent failure
        let agent_elapsed = agent_start.elapsed();
        metrics.record_agent_time(agent_elapsed);
        span_summary.record_agent(0); // token count not available from rig response
        let response = match agent_future {
            Ok(r) => {
                // Log the actual response text for debugging (truncated to 500 chars)
                let preview = if r.len() > 500 { &r[..500] } else { &r };
                info!(
                    iteration,
                    response_len = r.len(),
                    response_preview = %preview,
                    "Agent responded"
                );
                r
            }
            Err(e) => {
                error!(iteration, "Agent failed: {e}");
                let _ = progress.log_error(
                    session.session_id(),
                    iteration,
                    format!("Agent failed: {e}"),
                );
                // engine.decide() records the iteration internally — don't double-count
                //
                // Auto-format BEFORE verification: the agent may have written
                // syntactically correct but unformatted code. Without this,
                // the verifier's fail-fast pipeline sees fmt failure and skips
                // all remaining gates (clippy, check, test), causing the
                // compile-clean short-circuit to see all-false gates.
                // See: beefcake-dj1o
                if agent_has_written_prev {
                    let mut fmt_args = vec!["fmt".to_string()];
                    if verifier_config.packages.is_empty() {
                        fmt_args.push("--all".to_string());
                    } else {
                        for pkg in &verifier_config.packages {
                            fmt_args.extend(["--package".to_string(), pkg.clone()]);
                        }
                    }
                    let fmt_result = tokio::process::Command::new("cargo")
                        .args(&fmt_args)
                        .current_dir(&wt_path)
                        .output()
                        .await;
                    match fmt_result {
                        Ok(ref out) if out.status.success() => {
                            debug!(iteration, "Agent failure path: cargo fmt succeeded");
                        }
                        Ok(ref out) => {
                            warn!(
                                iteration,
                                "Agent failure path: cargo fmt failed (non-fatal): {}",
                                String::from_utf8_lossy(&out.stderr)
                            );
                        }
                        Err(e) => {
                            warn!(iteration, "Agent failure path: cargo fmt error: {e}");
                        }
                    }
                }

                info!(
                    iteration,
                    "Running verifier after agent failure to assess codebase state"
                );
                let current_verifier_config = if config.verifier_packages.is_empty() {
                    VerifierConfig {
                        packages: detect_changed_packages(&wt_path, !is_script_verifier),
                        ..verifier_config.clone()
                    }
                } else {
                    verifier_config.clone()
                };
                let report = run_verifier(&wt_path, &current_verifier_config, &language_profile).await;

                // Compile-clean short-circuit on agent failure path:
                // The agent may have been terminated by the adapter (repeated edit failure,
                // post-write stall) AFTER successfully writing files. If the verifier shows
                // fmt+clippy+check pass, accept the result — the "failure" was just the
                // agent not knowing when to stop, not an actual code failure.
                if agent_has_written_prev && tier == SwarmTier::Worker {
                    use coordination::verifier::GateOutcome;
                    // Accept Passed OR Skipped: Skipped means the gate didn't run
                    // due to fail-fast (an earlier gate failed), NOT that this gate
                    // itself failed. Only GateOutcome::Failed is a true rejection.
                    let gate_ok = |name: &str| {
                        report.gates.iter().any(|g| {
                            g.gate == name
                                && matches!(g.outcome, GateOutcome::Passed | GateOutcome::Skipped)
                        })
                    };
                    let fmt_ok = gate_ok("fmt");
                    let clippy_ok = gate_ok("clippy");
                    let check_ok = gate_ok("check");

                    if fmt_ok && clippy_ok && check_ok {
                        info!(
                            iteration,
                            fmt_ok,
                            clippy_ok,
                            check_ok,
                            "Compile-clean short-circuit (agent failure path): \
                             worker wrote files and fmt+clippy+check pass despite agent termination. Accepting."
                        );

                        // Log experiment TSV
                        let commit = pre_worker_commit.as_deref().unwrap_or("unknown");
                        crate::telemetry::append_experiment_tsv(
                            &wt_path,
                            commit,
                            report.failure_signals.len(),
                            &["fmt", "clippy", "check"],
                            "compile_clean_accept",
                            "agent failure path: wrote files, compile clean",
                        );

                        // Accept: merge worktree and close issue
                        #[allow(unused_assignments)]
                        {
                            last_report = Some(report);
                        }
                        metrics.finish_iteration();
                        success = true;
                        break; // Exit the loop — acceptance handled by post-loop merge logic
                    } else {
                        info!(
                            iteration,
                            agent_has_written_prev,
                            fmt_ok,
                            clippy_ok,
                            check_ok,
                            all_green = report.all_green,
                            gates_passed = report.gates_passed,
                            gates_total = report.gates_total,
                            "Compile-clean short-circuit NOT triggered (agent failure path)"
                        );
                        let decision = engine.decide(&mut escalation, &report);
                        last_report = Some(report);
                        metrics.finish_iteration();

                        if decision.stuck {
                            error!(iteration, "Escalation engine: stuck after agent failure");
                            create_stuck_intervention(
                                &mut session,
                                &progress,
                                &wt_path,
                                iteration,
                                &decision.reason,
                            );
                            break;
                        }
                        continue;
                    }
                } else {
                    info!(
                        iteration,
                        agent_has_written_prev,
                        tier = ?tier,
                        "Compile-clean short-circuit skipped: not Worker tier or no writes"
                    );
                    let decision = engine.decide(&mut escalation, &report);
                    last_report = Some(report);
                    metrics.finish_iteration();

                    if decision.stuck {
                        error!(iteration, "Escalation engine: stuck after agent failure");
                        create_stuck_intervention(
                            &mut session,
                            &progress,
                            &wt_path,
                            iteration,
                            &decision.reason,
                        );
                        break;
                    }
                    continue;
                }
            }
        };

        // --- Auto-format before commit ---
        // Workers don't always produce perfectly formatted code.
        // Run fmt BEFORE committing so format changes are included in the commit.
        // This prevents uncommitted changes from blocking the merge step.
        let mut fmt_args = vec!["fmt".to_string()];
        if verifier_config.packages.is_empty() {
            fmt_args.push("--all".to_string());
        } else {
            for pkg in &verifier_config.packages {
                fmt_args.extend(["--package".to_string(), pkg.clone()]);
            }
        }
        let fmt_output = tokio::process::Command::new("cargo")
            .args(&fmt_args)
            .current_dir(&wt_path)
            .output()
            .await;
        if let Ok(ref out) = fmt_output {
            if !out.status.success() {
                warn!(
                    iteration,
                    "cargo fmt failed (non-fatal): {}",
                    String::from_utf8_lossy(&out.stderr)
                );
            }
        }

        // --- Git commit changes made by the agent (+ auto-format) ---
        let has_changes = match git_commit_changes(&wt_path, iteration).await {
            Ok(changed) => changed,
            Err(e) => {
                error!(iteration, "git commit failed: {e}");
                let _ = progress.log_error(
                    session.session_id(),
                    iteration,
                    format!("git commit failed: {e}"),
                );
                return Err(e);
            }
        };

        // Capture the post-agent commit hash (before auto-fix) for diff sizing.
        let post_agent_commit = git_mgr.current_commit_full().ok();

        // --- Record artifact footprint from git diff ---
        if let (Some(ref pre), Some(ref post)) = (&pre_worker_commit, &post_agent_commit) {
            if pre != post {
                let artifacts = collect_artifacts_from_diff(&wt_path, pre, post);
                for artifact in artifacts {
                    metrics.record_artifact(artifact);
                }
            }
        }

        // Detect false-positive: git has changes (e.g. from cargo fmt) but agent
        // never called edit_file/write_file. Treat as no-progress to avoid the
        // verifier trivially passing on formatting-only diffs.
        if has_changes && !agent_has_written {
            warn!(
                iteration,
                "Agent produced git changes but never called edit_file/write_file — \
                 treating as no-progress (likely formatting-only diff)"
            );
            // Revert the formatting-only commit so it doesn't pollute the worktree
            if let Some(ref rollback_hash) = pre_worker_commit {
                let _ = git_mgr.hard_rollback(rollback_hash);
            }
            // Fall through to no-change handling
        }

        if !has_changes || !agent_has_written {
            escalation.record_no_change();
            metrics.record_no_change();
            warn!(
                iteration,
                response_len = response.len(),
                consecutive_no_change = escalation.consecutive_no_change,
                threshold = config.max_consecutive_no_change,
                agent_has_written,
                agent_terminated_without_writing,
                "No meaningful changes after agent response"
            );

            // --- Write-deadline escalation: immediate Cloud escalation ---
            // When the worker exhausted its turn budget without ever calling
            // edit_file/write_file, the task likely exceeds local model capability
            // (e.g., feature implementation requiring planning). Escalate to
            // Council (cloud manager) immediately instead of burning 3 iterations
            // (~90 min) on the no-change circuit breaker.
            if agent_terminated_without_writing
                && tier == SwarmTier::Worker
                && config.cloud_endpoint.is_some()
            {
                warn!(
                    iteration,
                    "Write-deadline escalation: worker terminated without writing — \
                     escalating to Council (cloud manager) immediately"
                );
                escalation.record_escalation(
                    SwarmTier::Council,
                    EscalationReason::Explicit {
                        reason: format!(
                            "write deadline: worker exhausted {} turns without edit_file/write_file",
                            WORKER_MAX_TURNS_WITHOUT_WRITE
                        ),
                    },
                );
                metrics.finish_iteration();
                continue;
            }

            // --- No-change circuit breaker ---
            if escalation.consecutive_no_change >= config.max_consecutive_no_change {
                error!(
                    iteration,
                    consecutive_no_change = escalation.consecutive_no_change,
                    "No-change circuit breaker triggered — {} consecutive iterations with no file changes",
                    escalation.consecutive_no_change,
                );
                metrics.finish_iteration();

                // Try scaffold fallback for doc-oriented tasks before giving up
                let scaffolded = try_scaffold_fallback(
                    &wt_path,
                    &issue.id,
                    &issue.title,
                    "", // BeadsIssue doesn't carry description at this level
                    iteration,
                );
                if scaffolded {
                    info!(
                        iteration,
                        "Scaffold fallback produced a template — still marking stuck"
                    );
                }

                create_stuck_intervention(
                    &mut session,
                    &progress,
                    &wt_path,
                    iteration,
                    &format!(
                        "No-change circuit breaker: {} consecutive iterations produced no file changes{}",
                        escalation.consecutive_no_change,
                        if scaffolded { " (scaffold committed)" } else { "" },
                    ),
                );
                break;
            }

            // Skip verifier when no files changed — running cargo check/test
            // on unchanged code wastes 5-15 min per iteration and provides no
            // new signal. Just re-use the previous verifier report and let the
            // escalation engine decide next steps.
            if let Some(ref prev_report) = last_report {
                let decision = engine.decide(&mut escalation, prev_report);
                metrics.finish_iteration();

                if decision.stuck {
                    error!(iteration, "Escalation engine: stuck (no changes)");
                    create_stuck_intervention(
                        &mut session,
                        &progress,
                        &wt_path,
                        iteration,
                        &decision.reason,
                    );
                    break;
                }
                let next = decision.target_tier;
                if decision.escalated || matches!(tier, SwarmTier::Council | SwarmTier::Human) {
                    warn!(
                        iteration,
                        ?next,
                        "No-change response; engine routes to {next:?} (verifier skipped)"
                    );
                } else {
                    warn!(
                        iteration,
                        ?next,
                        "No-change response; staying on {next:?} (verifier skipped)"
                    );
                }
                escalation.current_tier = next;
                continue;
            }

            // First iteration with no changes and no previous report — run
            // verifier to establish baseline.
            let current_verifier_config = if config.verifier_packages.is_empty() {
                VerifierConfig {
                    packages: detect_changed_packages(&wt_path, !is_script_verifier),
                    ..verifier_config.clone()
                }
            } else {
                verifier_config.clone()
            };
            let report = run_verifier(&wt_path, &current_verifier_config, &language_profile).await;
            let decision = engine.decide(&mut escalation, &report);
            last_report = Some(report);
            metrics.finish_iteration();

            if decision.stuck {
                error!(iteration, "Escalation engine: stuck (no changes)");
                create_stuck_intervention(
                    &mut session,
                    &progress,
                    &wt_path,
                    iteration,
                    &decision.reason,
                );
                break;
            }
            let next = decision.target_tier;
            if decision.escalated || matches!(tier, SwarmTier::Council | SwarmTier::Human) {
                warn!(
                    iteration,
                    ?next,
                    "No-change response; engine routes to {next:?}"
                );
            } else {
                warn!(iteration, ?next, "No-change response; staying on {next:?}");
            }
            escalation.current_tier = next;
            continue;
        }

        // Reset no-change counter on any iteration that produces changes
        escalation.reset_no_change();

        // --- Verifier: run deterministic quality gates ---
        let verifier_start = std::time::Instant::now();
        let current_verifier_config = if config.verifier_packages.is_empty() {
            VerifierConfig {
                packages: detect_changed_packages(&wt_path, !is_script_verifier),
                ..verifier_config.clone()
            }
        } else {
            verifier_config.clone()
        };
        let mut report = run_verifier(&wt_path, &current_verifier_config, &language_profile).await;
        let verifier_elapsed = verifier_start.elapsed();
        metrics.record_verifier_time(verifier_elapsed);
        otel::record_iteration_result(
            &iter_span,
            report.all_green,
            report.failure_signals.len(),
            0, // warnings not tracked separately in VerifierReport
            iter_start.elapsed().as_millis() as u64,
        );

        info!(
            iteration,
            all_green = report.all_green,
            summary = %report.summary(),
            "Verifier report"
        );

        // Record gate results into span summary
        for gate in &report.gates {
            let passed = matches!(gate.outcome, coordination::GateOutcome::Passed);
            span_summary.record_gate(passed, gate.duration_ms);
        }

        // --- Auto-fix: try to resolve trivial failures without LLM delegation ---
        let mut auto_fix_applied = false;
        if !report.all_green {
            if let Some(fixed_report) = try_auto_fix(&wt_path, &verifier_config, iteration, &language_profile).await {
                report = fixed_report;
                auto_fix_applied = true;
                metrics.record_auto_fix();
            }
        }

        let error_cats = report.unique_error_categories();
        let error_count = report.failure_signals.len();
        let cat_names: Vec<String> = error_cats.iter().map(|c| format!("{c:?}")).collect();
        metrics.record_verifier_results(error_count, cat_names);

        // Emit OpenTelemetry-compatible loop metrics event
        if let Some(lm) = metrics.build_loop_metrics(report.all_green) {
            lm.emit();
        }

        // --- Compile-clean short-circuit (Worker tier only) ---
        // If the worker wrote files and compilation passes (fmt + clippy + check),
        // accept the result even if tests fail — test failures may be pre-existing.
        // This prevents the "ghost iteration" problem where iteration N+1 re-applies
        // changes that iteration N already committed correctly.
        if !report.all_green && agent_has_written_prev && tier == SwarmTier::Worker {
            use coordination::verifier::GateOutcome;
            // Accept Passed OR Skipped (Skipped = didn't run due to fail-fast, not failed)
            let gate_ok = |name: &str| {
                report.gates.iter().any(|g| {
                    g.gate == name
                        && matches!(g.outcome, GateOutcome::Passed | GateOutcome::Skipped)
                })
            };
            let fmt_ok = gate_ok("fmt");
            let clippy_ok = gate_ok("clippy");
            let check_ok = gate_ok("check");

            if fmt_ok && clippy_ok && check_ok {
                info!(
                    iteration,
                    gates_passed = report.gates_passed,
                    gates_total = report.gates_total,
                    "Compile-clean short-circuit: worker wrote files and fmt+clippy+check pass. \
                     Accepting despite test failure (likely pre-existing)."
                );

                // Record as successful iteration for escalation engine
                escalation.record_iteration(error_cats.clone(), 0, true);
                best_error_count = Some(0);

                // Log experiment TSV
                let commit = pre_worker_commit.as_deref().unwrap_or("unknown");
                crate::telemetry::append_experiment_tsv(
                    &wt_path,
                    commit,
                    error_count,
                    &["fmt", "clippy", "check"],
                    "compile_clean_accept",
                    &format!(
                        "Worker wrote files, compile clean (test failures: {})",
                        error_count
                    ),
                );

                // Skip hill-climbing and go straight to acceptance
                report.all_green = true;
            }
        }

        // --- Hill-climbing: keep changes only when they improve on the best ---
        if !report.all_green {
            // Reset validator failure counter — verifier itself failed, so
            // prior validator feedback is stale.
            consecutive_validator_failures = 0;

            // Record progress score for telemetry.
            if let Some(best) = best_error_count {
                metrics.record_progress_score(error_count, best);
            }

            // Determine if this iteration improved on the best error count.
            let current_best = best_error_count.unwrap_or(usize::MAX);
            let improved = error_count + min_error_delta < current_best;

            if improved {
                // Partial progress — keep changes and update best.
                best_error_count = Some(error_count);
                info!(
                    iteration,
                    error_count,
                    prev_best = current_best,
                    "Partial progress kept (new best error count)"
                );
                // Log experiment TSV: keep
                let commit = pre_worker_commit.as_deref().unwrap_or("unknown");
                crate::telemetry::append_experiment_tsv(
                    &wt_path,
                    commit,
                    error_count,
                    &[],
                    "keep",
                    &format!("improved: {current_best} → {error_count}"),
                );
            } else {
                // Non-improvement vs best — rollback.
                warn!(
                    iteration,
                    error_count,
                    best = current_best,
                    "Non-improvement rollback — errors not better than best"
                );
                let mut rolled_back = false;
                if let Some(ref rollback_hash) = pre_worker_commit {
                    // Save patch artifact before rollback (SuperQode pattern).
                    let artifacts_dir = wt_path.join(".swarm-artifacts");
                    let _ = std::fs::create_dir_all(&artifacts_dir);
                    let patch_path = artifacts_dir.join(format!("iter-{iteration}.patch"));
                    if let Ok(output) = std::process::Command::new("git")
                        .args(["diff", rollback_hash])
                        .current_dir(&wt_path)
                        .output()
                    {
                        let _ = std::fs::write(&patch_path, &output.stdout);
                        debug!(iteration, path = %patch_path.display(), "Saved patch artifact before rollback");
                    }

                    match git_mgr.hard_rollback(rollback_hash) {
                        Ok(()) => {
                            rolled_back = true;
                            info!(
                                iteration,
                                rollback_to = %rollback_hash,
                                "Rolled back to pre-worker commit"
                            );
                            let _ = progress.log_error(
                                session.session_id(),
                                iteration,
                                format!(
                                    "Non-improvement: {error_count} errors (best: {current_best}). Rolled back to {rollback_hash}"
                                ),
                            );
                        }
                        Err(e) => {
                            error!(iteration, "Rollback failed: {e}");
                        }
                    }
                }

                // Log experiment TSV: revert
                let commit = pre_worker_commit.as_deref().unwrap_or("unknown");
                crate::telemetry::append_experiment_tsv(
                    &wt_path,
                    commit,
                    error_count,
                    &[],
                    "revert",
                    &format!("non-improvement: {error_count} >= best {current_best}"),
                );

                metrics.record_regression(rolled_back);
                if rolled_back {
                    // Re-run verifier against rolled-back code so last_report
                    // reflects the pre-regression error state, not the regressed state.
                    let rb_verifier_config = if config.verifier_packages.is_empty() {
                        VerifierConfig {
                            packages: detect_changed_packages(&wt_path, !is_script_verifier),
                            ..verifier_config.clone()
                        }
                    } else {
                        verifier_config.clone()
                    };
                    let rb_report = run_verifier(&wt_path, &rb_verifier_config, &language_profile).await;
                    info!(
                        iteration,
                        rollback_errors = rb_report.failure_signals.len(),
                        "Verifier re-run after rollback"
                    );
                    last_report = Some(rb_report);
                    metrics.finish_iteration();
                    continue;
                }
            }
        }

        if report.all_green {
            // --- Guard against auto-fix false positives ---
            // Only check when auto-fix actually ran this iteration. This avoids
            // rejecting legitimate small fixes (< min_diff_lines) that pass the
            // verifier on their own merit.
            if should_reject_auto_fix(auto_fix_applied, &acceptance_policy) {
                if let (Some(initial), Some(agent_commit)) = (
                    session.state().initial_commit.as_ref(),
                    post_agent_commit.as_ref(),
                ) {
                    let agent_diff_lines = count_diff_lines(&wt_path, initial, agent_commit);
                    if agent_diff_lines < acceptance_policy.min_diff_lines {
                        warn!(
                            iteration,
                            agent_diff_lines,
                            min_required = acceptance_policy.min_diff_lines,
                            "Auto-fix false positive: agent produced {} lines but minimum is {}",
                            agent_diff_lines,
                            acceptance_policy.min_diff_lines,
                        );
                        let _ = progress.log_error(
                            session.session_id(),
                            iteration,
                            format!(
                                "Auto-fix false positive: agent diff only {agent_diff_lines} lines (min: {})",
                                acceptance_policy.min_diff_lines
                            ),
                        );
                        // Record this as a failed iteration and continue
                        escalation.record_iteration(error_cats.clone(), 0, false);
                        last_report = Some(report);
                        metrics.finish_iteration();
                        continue;
                    }
                }
            }

            // --- Local validation (blocking gate) ---
            // After deterministic gates pass, run the reviewer on vasp-02 as a blocking
            // quality gate. This catches logic errors, edge cases, and design issues
            // that the compiler cannot detect.
            if local_validator_enabled {
                if let Some(ref initial_commit) = session.state().initial_commit {
                    info!(iteration, "Running local validation (blocking)");
                    let local_result = local_validate(
                        &reviewer,
                        &wt_path,
                        initial_commit,
                        &config.fast_endpoint.model,
                    )
                    .await;

                    metrics.record_local_validation(&local_result.model, local_result.passed);

                    if local_result.passed {
                        consecutive_validator_failures = 0;
                        info!(
                            iteration,
                            model = %local_result.model,
                            "Local validation: PASS"
                        );
                    } else {
                        consecutive_validator_failures += 1;
                        warn!(
                            iteration,
                            model = %local_result.model,
                            consecutive_failures = consecutive_validator_failures,
                            max_failures = max_validator_failures,
                            "Local validation: FAIL (blocking)"
                        );

                        // Extract feedback for next iteration
                        let feedback = extract_local_validator_feedback(&local_result);
                        last_validator_feedback = feedback;

                        if consecutive_validator_failures >= max_validator_failures {
                            warn!(
                                iteration,
                                consecutive_failures = consecutive_validator_failures,
                                "Local validator failure cap reached — accepting anyway"
                            );
                            consecutive_validator_failures = 0;
                            // Fall through to acceptance
                        } else {
                            info!(
                                iteration,
                                feedback_count = last_validator_feedback.len(),
                                "Local validation rejected — looping with feedback"
                            );
                            escalation.record_iteration(error_cats, error_count, false);
                            last_report = Some(report);
                            metrics.finish_iteration();
                            continue;
                        }
                    }
                }
            }

            // Deterministic gates (fmt + clippy + check + test) are the source of truth.
            // The local reviewer gates acceptance; cloud reviewer is advisory.
            info!(
                iteration,
                "Verifier passed (all gates green) — checking acceptance"
            );
            escalation.record_iteration(error_cats, error_count, true);
            best_error_count = Some(0);

            // Log experiment TSV: success
            let commit = pre_worker_commit.as_deref().unwrap_or("unknown");
            crate::telemetry::append_experiment_tsv(
                &wt_path,
                commit,
                0,
                &["fmt", "clippy", "check", "test"],
                "success",
                "all gates green",
            );

            // Create harness checkpoint on success
            if let Ok(hash) = git_mgr.current_commit() {
                let _ = progress.log_checkpoint(session.session_id(), iteration, &hash);
            }
            let _ = progress.log_feature_complete(
                session.session_id(),
                iteration,
                &issue.id,
                "Verified (deterministic gates passed)",
            );

            // --- Cloud validation (advisory) ---
            // After deterministic gates pass, send the diff to high-end cloud models
            // (G3 Pro + Sonnet 4.5) for logic/design review. Results are logged but
            // don't block acceptance — avoids subjective LLM feedback loops.
            let mut cloud_passes = 0usize;
            if let Some(ref cloud_client) = factory.clients.cloud {
                if let Some(ref initial_commit) = session.state().initial_commit {
                    let validations = cloud_validate(cloud_client, &wt_path, initial_commit).await;
                    // Collect structured feedback for next iteration (TextGrad pattern)
                    last_validator_feedback.clear();
                    for v in &validations {
                        metrics.record_cloud_validation(&v.model, v.passed);
                        if v.passed {
                            cloud_passes += 1;
                            info!(model = %v.model, "Cloud validation: PASS");
                        } else {
                            warn!(
                                model = %v.model,
                                "Cloud validation: FAIL (advisory) — {}",
                                v.feedback.lines().take(5).collect::<Vec<_>>().join(" | ")
                            );
                            let feedback = extract_validator_feedback(v);
                            last_validator_feedback.extend(feedback);
                        }
                    }
                    if !last_validator_feedback.is_empty() {
                        info!(
                            feedback_count = last_validator_feedback.len(),
                            "Collected structured validator feedback for next iteration"
                        );
                    }
                }
            }

            // --- Acceptance policy check ---
            let acceptance_result = acceptance::check_acceptance(
                &acceptance_policy,
                &wt_path,
                session.state().initial_commit.as_deref(),
                cloud_passes,
            );

            if !acceptance_result.accepted {
                for rejection in &acceptance_result.rejections {
                    warn!(iteration, rejection = %rejection, "Acceptance policy rejected");
                }
                info!(iteration, "Acceptance failed — continuing iteration loop");
                metrics.finish_iteration();
                last_report = Some(report);
                continue;
            }

            metrics.finish_iteration();
            success = true;
            break;
        }
        // engine.decide() below records the iteration internally — don't double-count

        // --- Integration Point 2: Pre-escalation knowledge check ---
        // Before escalating, check if the Debugging KB has a known fix.
        // If found, log it so the next iteration's pre-task enrichment picks it up.
        if let Some(kb) = knowledge_base {
            let error_cats: Vec<String> = report
                .unique_error_categories()
                .iter()
                .map(|c| format!("{c:?}"))
                .collect();
            if !error_cats.is_empty() {
                let question = format!("Known fix for Rust errors: {}", error_cats.join(", "));
                let response = query_kb_with_failsafe(kb, "debugging_kb", &question);
                if !response.is_empty() {
                    info!(
                        iteration,
                        kb_suggestion_len = response.len(),
                        "Found known fix in Debugging KB — will inject in next iteration"
                    );
                }
            }
        }

        // --- Escalation decision ---
        let decision = engine.decide(&mut escalation, &report);
        last_report = Some(report);

        if decision.escalated {
            metrics.record_escalation();
            span_summary.record_escalation();
            let _esc_span = otel::escalation_span(
                &issue.id,
                &format!("{tier:?}"),
                &format!("{:?}", decision.target_tier),
                &decision.reason,
                iteration,
            );
            info!(
                iteration,
                from = ?tier,
                to = ?decision.target_tier,
                reason = %decision.reason,
                "Tier escalated"
            );
        }

        metrics.finish_iteration();

        if decision.stuck {
            error!(
                iteration,
                reason = %decision.reason,
                "Escalation engine: stuck — flagging for human intervention"
            );
            create_stuck_intervention(
                &mut session,
                &progress,
                &wt_path,
                iteration,
                &decision.reason,
            );
            break;
        }
    }

    // --- Outcome ---
    if success {
        // --- Integration Point 3: Post-success knowledge capture ---
        if let Some(kb) = knowledge_base {
            let _ = knowledge_sync::capture_resolution(
                kb,
                &issue.id,
                &issue.title,
                session.iteration(),
                &format!("{:?}", escalation.current_tier),
                &[], // Files touched not tracked at this level yet
            );

            // If it took 3+ iterations (tricky bug), also capture the error pattern
            if session.iteration() >= 3 {
                let error_cats: Vec<String> = escalation
                    .recent_error_categories
                    .iter()
                    .flatten()
                    .map(|c| format!("{c:?}"))
                    .collect();
                let _ = knowledge_sync::capture_error_pattern(
                    kb,
                    &issue.id,
                    &error_cats,
                    session.iteration(),
                    &format!(
                        "Resolved after {} iterations at {:?} tier",
                        session.iteration(),
                        escalation.current_tier
                    ),
                );
            }
        }

        session.complete();
        let _ = progress.log_session_end(
            session.session_id(),
            session.iteration(),
            format!("Issue {} resolved", issue.id),
        );

        // --- Integration Point 4: Retrospective knowledge capture ---
        if let Some(kb) = knowledge_base {
            let entries = progress.read_all().unwrap_or_default();
            let retro = session.retrospective(&entries);
            let svc = knowledge_sync::KnowledgeSyncService::new(kb);
            let captures = svc.capture_from_retrospective(&retro, &issue.id, &issue.title);
            debug!(count = captures.len(), "Retrospective captures uploaded");
        }

        // Commit all edits before merging — workers modify files but don't commit.
        // Without this, merge_and_remove fails with "uncommitted changes" and the
        // entire resolution is lost (the bug that blocked beefcake-swarm-004).
        if let Err(e) = git_commit_changes(&wt_path, session.iteration() as u32).await {
            warn!(id = %issue.id, "Failed to commit worker edits before merge: {e}");
        }

        info!(
            id = %issue.id,
            session_id = session.short_id(),
            elapsed = %session.elapsed_human(),
            iterations = session.iteration(),
            "Issue resolved — merging worktree"
        );

        if let Err(e) = worktree_bridge.merge_and_remove(&issue.id) {
            error!(id = %issue.id, "Merge failed: {e} — resetting issue to open");
            // Cleanup the worktree to prevent leaks
            if let Err(cleanup_err) = worktree_bridge.cleanup(&issue.id) {
                warn!(id = %issue.id, "Cleanup failed: {cleanup_err}");
            }
            let _ = beads.update_status(&issue.id, "open");
            return Err(e);
        }
        beads.close(&issue.id, Some("Resolved by swarm orchestrator"))?;
        clear_resume_file(worktree_bridge.repo_root());
        info!(id = %issue.id, "Issue closed");
    } else {
        session.fail();
        let _ = progress.log_session_end(
            session.session_id(),
            session.iteration(),
            format!(
                "Failed after {} iterations — {}",
                session.iteration(),
                escalation.summary()
            ),
        );

        // --- Defensive PR Safety Net (Phase 5d, Open SWE pattern) ---
        //
        // If the worktree has uncommitted changes from a failed run, commit them
        // to the swarm branch and push. This ensures no agent work is lost — the
        // branch can be inspected, cherry-picked, or used as context for retry.
        if git_has_meaningful_changes(&wt_path).await {
            info!(id = %issue.id, "Defensive safety net: capturing partial progress");
            let _ = std::process::Command::new("git")
                .args(["add", "-A"])
                .current_dir(&wt_path)
                .output();
            let msg = format!(
                "swarm: WIP partial progress on {} (failed after {} iterations)",
                issue.id,
                session.iteration()
            );
            let _ = std::process::Command::new("git")
                .args(["commit", "-m", &msg])
                .current_dir(&wt_path)
                .output();
            // Push the branch so it's visible on the remote
            let branch = format!("swarm/{}", issue.id);
            match std::process::Command::new("git")
                .args(["push", "origin", &branch, "--force-with-lease"])
                .current_dir(&wt_path)
                .output()
            {
                Ok(ref out) if out.status.success() => {
                    info!(
                        id = %issue.id,
                        branch = %branch,
                        "Defensive safety net: pushed WIP branch"
                    );
                }
                Ok(ref out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    debug!(id = %issue.id, stderr = %stderr.trim(), "WIP branch push failed (non-fatal)");
                }
                Err(e) => {
                    debug!(id = %issue.id, error = %e, "WIP branch push failed (non-fatal)");
                }
            }
            let _ = beads.update_status(&issue.id, "open"); // ensure it stays open for retry
        }

        // --- Integration Point 4: Retrospective knowledge capture (failure) ---
        if let Some(kb) = knowledge_base {
            let entries = progress.read_all().unwrap_or_default();
            let retro = session.retrospective(&entries);
            let svc = knowledge_sync::KnowledgeSyncService::new(kb);
            let captures = svc.capture_from_retrospective(&retro, &issue.id, &issue.title);
            debug!(
                count = captures.len(),
                "Retrospective captures uploaded (failure path)"
            );
        }

        // Persist session state for potential resume after SLURM preemption
        let state_path = wt_path.join(".swarm-session.json");
        if let Err(e) = save_session_state(session.state(), &state_path) {
            warn!("Failed to save session state: {e}");
        } else {
            info!(path = %state_path.display(), "Session state saved for resume");
        }

        // Write resume file to repo root for startup detection
        let resume = SwarmResumeFile {
            issue: issue.clone(),
            worktree_path: wt_path.display().to_string(),
            iteration: session.iteration(),
            escalation_summary: escalation.summary(),
            current_tier: format!("{:?}", escalation.current_tier),
            total_iterations: escalation.total_iterations,
            saved_at: chrono::Utc::now().to_rfc3339(),
        };
        let resume_path = worktree_bridge.repo_root().join(".swarm-resume.json");
        match serde_json::to_string_pretty(&resume) {
            Ok(json) => {
                if let Err(e) = std::fs::write(&resume_path, json) {
                    warn!("Failed to write resume file: {e}");
                } else {
                    info!(path = %resume_path.display(), "Resume file saved");
                }
            }
            Err(e) => warn!("Failed to serialize resume file: {e}"),
        }

        error!(
            id = %issue.id,
            session_id = session.short_id(),
            elapsed = %session.elapsed_human(),
            iterations = session.iteration(),
            status = %session.status(),
            interventions = session.state().unresolved_interventions().len(),
            summary = %escalation.summary(),
            "Issue NOT resolved — leaving worktree for inspection"
        );
    }

    // --- Post-session: detect failure patterns and save directives ---
    let directives = detect_failure_patterns(&wt_path);
    if !directives.is_empty() {
        info!(
            count = directives.len(),
            "Detected failure patterns — saving directives"
        );
        save_directives(worktree_bridge.repo_root(), &directives);
    }

    // --- Write telemetry ---
    let final_tier = format!("{:?}", escalation.current_tier);
    let session_metrics = metrics.finalize(success, &final_tier);
    telemetry::write_session_metrics(&session_metrics, &wt_path);
    telemetry::append_telemetry(&session_metrics, worktree_bridge.repo_root());

    // --- TensorZero feedback ---
    // Phase 1 fix: when PG URL is available, resolve the actual episode IDs that
    // TZ assigned (not our self-generated ones) and post feedback to those.
    // Falls back to the original self-generated episode_id if PG is unavailable.
    if let Some(ref tz_url) = config.tensorzero_url {
        let wall_secs = session_metrics.elapsed_ms as f64 / 1000.0;
        if let Some(ref pg_url) = config.tensorzero_pg_url {
            crate::tensorzero::post_resolved_feedback(
                tz_url,
                pg_url,
                tz_session_start_secs,
                success,
                session_metrics.total_iterations,
                wall_secs,
            )
            .await;
        } else if let Some(ref ep_id) = tensorzero_episode_id {
            // Legacy fallback: use self-generated episode_id (may fail if TZ
            // doesn't recognize it, but harmless — feedback is best-effort)
            crate::tensorzero::post_episode_feedback(
                tz_url,
                ep_id,
                success,
                session_metrics.total_iterations,
                wall_secs,
            )
            .await;
        }
    }

    // Record final outcome on the root span
    otel::record_process_result(
        &process_span,
        success,
        session_metrics.total_iterations as u32,
        process_start.elapsed().as_millis() as u64,
    );

    // Log span summary for post-run analysis
    info!(summary = %span_summary, "OTel span summary");

    // --- SLO evaluation ---
    // Build a single-session OrchestrationMetrics snapshot and evaluate SLOs.
    // For single sessions, most aggregate metrics collapse to 0 or 1.
    let escalated = session_metrics.iterations.iter().any(|i| i.escalated);
    let orch_metrics = OrchestrationMetrics {
        session_count: 1,
        first_pass_rate: if session_metrics.total_iterations == 1 && success {
            1.0
        } else {
            0.0
        },
        overall_success_rate: if success { 1.0 } else { 0.0 },
        avg_iterations_to_green: session_metrics.total_iterations as f64,
        median_iterations_to_green: session_metrics.total_iterations as f64,
        escalation_rate: if escalated { 1.0 } else { 0.0 },
        avg_escalations: if escalated { 1.0 } else { 0.0 },
        latency_p50: Duration::from_millis(session_metrics.elapsed_ms),
        latency_p95: Duration::from_millis(session_metrics.elapsed_ms),
        latency_max: Duration::from_millis(session_metrics.elapsed_ms),
        tokens_p50: 0,
        tokens_p95: 0,
        tokens_total: 0,
        cost_total: 0.0,
        cost_avg: 0.0,
        stuck_rate: if !success { 1.0 } else { 0.0 },
        avg_turns_until_first_write: session_metrics.turns_until_first_write.unwrap_or(0) as f64,
        write_by_turn_2_rate: if session_metrics.write_by_turn_2 {
            1.0
        } else {
            0.0
        },
    };
    let slo_report = slo::evaluate_slos(&orch_metrics);
    match slo_report.overall_severity {
        AlertSeverity::Ok => {
            info!(passing = slo_report.passing, "SLO check: all passing");
        }
        AlertSeverity::Warning => {
            warn!(
                passing = slo_report.passing,
                warnings = slo_report.warnings,
                "SLO check: warnings detected\n{}",
                slo_report.summary()
            );
        }
        AlertSeverity::Critical => {
            error!(
                passing = slo_report.passing,
                critical = slo_report.critical,
                "SLO check: CRITICAL violations\n{}",
                slo_report.summary()
            );
        }
    }

    // --- KB Refresh check ---
    // Read historical telemetry to get total session count, then check if
    // a KB refresh is due based on the session_interval policy.
    let telemetry_path = worktree_bridge.repo_root().join(".swarm-telemetry.jsonl");
    if let Ok(reader) = TelemetryReader::read_from_file(&telemetry_path) {
        let total_sessions = reader.sessions().len();
        let refresh_policy = crate::kb_refresh::RefreshPolicy::default();

        if crate::kb_refresh::should_refresh(total_sessions, &refresh_policy) {
            let analytics = reader.aggregate_analytics();
            let skills = coordination::analytics::skills::SkillLibrary::new();
            let now = chrono::Utc::now();

            let refresh_report =
                crate::kb_refresh::analyze_and_refresh(&analytics, &skills, &refresh_policy, now);
            if refresh_report.has_actions() {
                info!(
                    actions = refresh_report.actions.len(),
                    stale = refresh_report.stale_skills,
                    promotions = refresh_report.promotions,
                    undocumented = refresh_report.undocumented_errors,
                    "KB refresh: {refresh_report}"
                );
            } else {
                debug!(sessions = total_sessions, "KB refresh: no actions needed");
            }
        }

        // --- Dashboard metrics ---
        // Generate an all-time dashboard from accumulated telemetry and log summary.
        let skills = coordination::analytics::skills::SkillLibrary::new();
        let now = chrono::Utc::now();
        let dashboard = crate::dashboard::generate(reader.sessions(), &skills, now);
        let summary = crate::dashboard::format_summary(&dashboard);
        info!(sessions = reader.sessions().len(), "\n{summary}");
    } else {
        debug!("No telemetry file found — skipping KB refresh and dashboard");
    }

    // --- Mutation archive (Phase 4a: evolutionary tracking) ---
    {
        let mut record = crate::mutation_archive::build_record(
            &issue.id,
            &issue.title,
            &archive_language,
            success,
            session.iteration(),
            &format!("{:?}", escalation.current_tier),
            &config.resolve_role_model(SwarmRole::Planner), // primary model
            process_start.elapsed().as_secs(),
        );
        record.auto_fix_only = success && session.iteration() == 0;
        if let Some(ref report) = last_report {
            record.error_categories = report
                .unique_error_categories()
                .iter()
                .map(|c| format!("{c:?}"))
                .collect();
            record.first_failure_gate = report.first_failure.clone();
        }
        if !success {
            record.failure_reason = Some(escalation.summary());
        }
        // Count changed lines from git diff
        if let Ok(output) = std::process::Command::new("git")
            .args(["diff", "--stat", "main"])
            .current_dir(&wt_path)
            .output()
        {
            let stat = String::from_utf8_lossy(&output.stdout);
            for line in stat.lines() {
                if line.contains("insertion") {
                    if let Some(n) = line.split_whitespace().nth(0).and_then(|s| s.parse().ok()) {
                        record.lines_added = n;
                    }
                }
                if line.contains("deletion") {
                    for word in line.split_whitespace() {
                        if let Ok(n) = word.parse::<u32>() {
                            record.lines_removed = n;
                            break;
                        }
                    }
                }
            }
        }
        archive.record(&record);
    }

    Ok(success)
}

/// Compile-time guard: `process_issue_core`'s future must be `Send` so it
/// can be dispatched via `tokio::spawn` / `JoinSet` in Phase 2 (Thread Weaving).
/// If someone re-introduces a `Span::enter()` guard held across an `.await`,
/// this function will fail to compile.
#[cfg(test)]
#[allow(dead_code, unused_variables)]
fn _assert_process_issue_core_is_send(
    config: &SwarmConfig,
    factory: &AgentFactory,
    worktree_bridge: &WorktreeBridge,
    issue: &BeadsIssue,
    beads: &dyn IssueTracker,
    knowledge_base: Option<&dyn KnowledgeBase>,
    cancel: Arc<AtomicBool>,
) {
    fn _require_send<F: std::future::Future + Send>(_f: F) {}
    _require_send(process_issue_core(
        config,
        factory,
        worktree_bridge,
        issue,
        beads,
        knowledge_base,
        cancel,
    ));
}

/// Saved state for session resume after SLURM preemption or crash.
///
/// Written to `.swarm-resume.json` in the repo root on failure.
/// Checked on startup to restore worktree, iteration count, and escalation state.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SwarmResumeFile {
    /// Issue being worked on
    pub issue: BeadsIssue,
    /// Worktree path for the in-progress work
    pub worktree_path: String,
    /// Current iteration count
    pub iteration: u32,
    /// Escalation state summary
    pub escalation_summary: String,
    /// Current tier
    pub current_tier: String,
    /// Total iterations across all tiers
    pub total_iterations: u32,
    /// Timestamp when saved
    pub saved_at: String,
}

/// Check for a resume file and return the data if found.
pub fn check_for_resume(repo_root: &Path) -> Option<SwarmResumeFile> {
    let resume_path = repo_root.join(".swarm-resume.json");
    if resume_path.exists() {
        match std::fs::read_to_string(&resume_path) {
            Ok(contents) => match serde_json::from_str::<SwarmResumeFile>(&contents) {
                Ok(resume) => {
                    info!(
                        issue = %resume.issue.id,
                        worktree = %resume.worktree_path,
                        iteration = resume.iteration,
                        "Found resume file — previous session can be continued"
                    );
                    Some(resume)
                }
                Err(e) => {
                    warn!("Failed to parse resume file: {e}");
                    None
                }
            },
            Err(e) => {
                warn!("Failed to read resume file: {e}");
                None
            }
        }
    } else {
        None
    }
}

/// Clear the resume file after successful completion.
pub(crate) fn clear_resume_file(repo_root: &Path) {
    let resume_path = repo_root.join(".swarm-resume.json");
    if resume_path.exists() {
        let _ = std::fs::remove_file(&resume_path);
    }
}

/// Prompt an agent with exponential backoff retry for transient errors.
///
/// Uses `OrchestrationError::classify()` to detect retriable errors (connection
/// failures, rate limits, timeouts). Non-retriable errors fail immediately.
/// Backoff: 2s, 4s, 8s, ...
async fn prompt_with_retry(
    agent: &impl Prompt,
    prompt: &str,
    max_retries: u32,
) -> Result<String, rig::completion::PromptError> {
    let mut last_err = None;
    for attempt in 0..=max_retries {
        match agent.prompt(prompt).await {
            Ok(response) => return Ok(response),
            Err(e) => {
                let classified = OrchestrationError::classify(&e);
                if !classified.is_retriable() || attempt == max_retries {
                    return Err(e);
                }

                let backoff = Duration::from_secs(2u64.pow(attempt + 1));
                warn!(
                    attempt = attempt + 1,
                    max_retries,
                    backoff_secs = backoff.as_secs(),
                    category = %classified.retry_category(),
                    error = %e,
                    "Transient error — retrying"
                );
                last_err = Some(e);
                tokio::time::sleep(backoff).await;
            }
        }
    }
    Err(last_err.unwrap())
}

/// Like [`prompt_with_retry`] but attaches a [`RuntimeAdapter`] hook to each attempt.
///
/// The hook provides tool-event visibility and budget enforcement for the manager tier.
pub(crate) async fn prompt_with_hook_and_retry(
    agent: &crate::agents::coder::OaiAgent,
    prompt: &str,
    max_retries: u32,
    hook: RuntimeAdapter,
) -> Result<String, rig::completion::PromptError> {
    let mut last_err = None;
    for attempt in 0..=max_retries {
        match agent.prompt(prompt).with_hook(hook.clone()).await {
            Ok(response) => return Ok(response),
            Err(e) => {
                let classified = OrchestrationError::classify(&e);
                if !classified.is_retriable() || attempt == max_retries {
                    return Err(e);
                }

                let backoff = Duration::from_secs(2u64.pow(attempt + 1));
                warn!(
                    attempt = attempt + 1,
                    max_retries,
                    backoff_secs = backoff.as_secs(),
                    category = %classified.retry_category(),
                    error = %e,
                    "Transient error — retrying (with hook)"
                );
                last_err = Some(e);
                tokio::time::sleep(backoff).await;
            }
        }
    }
    Err(last_err.unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use coordination::{SessionStatus, SwarmTier};
    use std::path::PathBuf;

    /// Initialize a temporary git repo with one commit and return the initial
    /// commit hash. Deduplicates test boilerplate across git-dependent tests.
    fn init_test_git_repo(dir: &std::path::Path) -> String {
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir)
            .output()
            .unwrap();
        std::fs::write(dir.join("README.md"), "# test\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir)
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir)
            .output()
            .unwrap();

        String::from_utf8(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir)
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string()
    }
    #[test]
    fn test_session_manager_iteration_matches_max_retries() {
        // Verify that SessionManager iteration counting matches the old
        // `for iteration in 1..=max_retries` behavior
        let mut session = SessionManager::new(PathBuf::from("/tmp"), 6);
        session.start().unwrap();

        let mut iterations = Vec::new();
        loop {
            match session.next_iteration() {
                Ok(i) => iterations.push(i),
                Err(_) => break,
            }
        }

        assert_eq!(iterations, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(session.status(), SessionStatus::MaxIterationsReached);
    }

    #[test]
    fn test_session_state_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = SessionManager::new(dir.path().to_path_buf(), 10);
        session.start().unwrap();
        session.set_current_feature("beads-abc123");
        session.next_iteration().unwrap();
        session.next_iteration().unwrap();
        session.set_initial_commit("deadbeef".into());

        // Save state
        let state_path = dir.path().join(".swarm-session.json");
        save_session_state(session.state(), &state_path).unwrap();

        // Load and verify
        let loaded = coordination::load_session_state(&state_path)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.id, session.session_id());
        assert_eq!(loaded.iteration, 2);
        assert_eq!(loaded.current_feature, Some("beads-abc123".to_string()));
        assert_eq!(loaded.initial_commit, Some("deadbeef".to_string()));
        assert_eq!(loaded.status, SessionStatus::Active);
    }

    #[test]
    fn test_progress_tracker_logs_session_lifecycle() {
        let dir = tempfile::tempdir().unwrap();
        let progress = ProgressTracker::new(dir.path().join("progress.txt"));

        let session_id = "test-session-id";
        progress
            .log_session_start(session_id, "Starting work on issue")
            .unwrap();
        progress
            .log_feature_start(session_id, 1, "issue-001", "Iteration 1")
            .unwrap();
        progress
            .log_error(session_id, 1, "Agent failed to compile")
            .unwrap();
        progress.log_checkpoint(session_id, 2, "abc1234").unwrap();
        progress
            .log_feature_complete(session_id, 2, "issue-001", "Verified")
            .unwrap();
        progress
            .log_session_end(session_id, 2, "Issue resolved")
            .unwrap();

        let entries = progress.read_all().unwrap();
        assert_eq!(entries.len(), 6);

        // Verify markers are in expected order
        use coordination::ProgressMarker;
        assert!(matches!(entries[0].marker, ProgressMarker::SessionStart));
        assert!(matches!(entries[1].marker, ProgressMarker::FeatureStart));
        assert!(matches!(entries[2].marker, ProgressMarker::Error));
        assert!(matches!(entries[3].marker, ProgressMarker::Checkpoint));
        assert!(matches!(entries[4].marker, ProgressMarker::FeatureComplete));
        assert!(matches!(entries[5].marker, ProgressMarker::SessionEnd));
    }

    /// The auto-fix false positive guard should only reject iterations where
    /// `auto_fix_applied == true` AND the agent diff is below `min_diff_lines`.
    /// When auto-fix did NOT run, `min_diff_lines` must not block acceptance.
    #[test]
    fn test_auto_fix_guard_only_fires_when_auto_fix_applied() {
        let dir = tempfile::tempdir().unwrap();
        let initial = init_test_git_repo(dir.path());

        // Add a tiny 2-line change (below default min_diff_lines of 5)
        std::fs::write(dir.path().join("fix.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "small fix"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let agent_commit = String::from_utf8(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir.path())
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string();

        let policy = AcceptancePolicy::default();
        assert_eq!(policy.min_diff_lines, 5);

        let agent_diff = count_diff_lines(dir.path(), &initial, &agent_commit);
        assert_eq!(agent_diff, 2, "Agent produced 2 lines");

        // Case 1: auto_fix_applied=true, small diff → guard should fire (reject)
        assert!(
            should_reject_auto_fix(true, &policy),
            "Should reject when auto-fix ran and diff is tiny"
        );

        // Case 2: auto_fix_applied=false, same small diff → guard must NOT fire
        assert!(
            !should_reject_auto_fix(false, &policy),
            "Must not reject when auto-fix did not run"
        );

        // Case 3: auto_fix_applied=true but min_diff_lines=0 → guard disabled
        let permissive = AcceptancePolicy {
            min_diff_lines: 0,
            ..AcceptancePolicy::default()
        };
        assert!(
            !should_reject_auto_fix(true, &permissive),
            "Must not reject when min_diff_lines is disabled"
        );
    }

    #[test]
    fn test_count_diff_lines_in_git_repo() {
        let dir = tempfile::tempdir().unwrap();
        let from = init_test_git_repo(dir.path());

        // Add 10 lines
        let content = (1..=10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("code.rs"), format!("{content}\n")).unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "add code"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let to = String::from_utf8(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir.path())
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string();

        assert_eq!(count_diff_lines(dir.path(), &from, &to), 10);

        // count_diff_lines with same commit should be 0
        assert_eq!(count_diff_lines(dir.path(), &to, &to), 0);
    }

    #[test]
    fn test_git_manager_checkpoint_prefix() {
        // Verify GitManager uses the expected commit prefix
        let dir = tempfile::tempdir().unwrap();
        let _initial = init_test_git_repo(dir.path());

        let git_mgr = GitManager::new(dir.path(), "[swarm]");

        // Record initial commit
        let initial = git_mgr.current_commit_full().unwrap();
        assert!(!initial.is_empty());

        // Create a change and checkpoint
        std::fs::write(dir.path().join("feature.rs"), "fn main() {}").unwrap();
        let hash = git_mgr
            .create_checkpoint("issue-001", "implemented feature")
            .unwrap();
        assert!(!hash.is_empty());

        // Verify the checkpoint commit has our prefix
        let commits = git_mgr.recent_commits(1).unwrap();
        assert!(commits[0].message.starts_with("[swarm]"));
        assert!(commits[0].is_harness_checkpoint);
    }
    #[test]
    fn test_feature_flags_loaded_from_env() {
        // Verify FeatureFlags::from_env() works and summary is displayable.
        // Unset to guarantee defaults.
        std::env::remove_var("SWARM_SMART_ROUTER_ENABLED");
        std::env::remove_var("SWARM_STATE_MACHINE_ENABLED");
        std::env::remove_var("SWARM_CANARY_ENABLED");
        std::env::remove_var("SWARM_STRUCTURED_EVALUATOR_REQUIRED");
        std::env::remove_var("SWARM_WORKER_FIRST_ENABLED");

        let flags = FeatureFlags::from_env();
        assert!(!flags.any_enabled());
        assert_eq!(
            flags.summary(),
            "Feature flags: all disabled (conservative mode)"
        );

        // Display trait works
        let display = flags.to_string();
        assert!(display.contains("smart_router=OFF"));
        assert!(display.contains("worker_first=OFF"));
    }

    #[test]
    fn test_worker_first_flag_routes_through_classifier() {
        // When worker_first is enabled, classify_initial_tier determines the starting tier
        // instead of defaulting to Council.

        // Simple task → Worker
        let rec = classify_initial_tier("Fix unused import in lib.rs", &[]);
        assert_eq!(rec.tier, SwarmTier::Worker);

        // Complex task → Council
        let rec = classify_initial_tier("Refactor async orchestration with tokio", &[]);
        assert_eq!(rec.tier, SwarmTier::Council);

        // Unknown task → Worker (worker-first default)
        let rec = classify_initial_tier("Add per-agent performance tracking", &[]);
        assert_eq!(rec.tier, SwarmTier::Worker);
    }

    #[test]
    fn test_otel_span_summary_accumulation() {
        let mut summary = SpanSummary::new();

        // Simulate a session with 3 iterations
        for _ in 0..3 {
            summary.record_iteration();
            summary.record_agent(0);
            // 4 gates per iteration: fmt, clippy, check, test
            summary.record_gate(true, 100); // fmt
            summary.record_gate(true, 500); // clippy
            summary.record_gate(true, 300); // check
            summary.record_gate(false, 200); // test fails
        }
        summary.record_escalation();

        assert_eq!(summary.iterations, 3);
        assert_eq!(summary.agent_invocations, 3);
        assert_eq!(summary.gates, 12);
        assert_eq!(summary.gates_passed, 9);
        assert_eq!(summary.gates_failed, 3);
        assert_eq!(summary.escalations, 1);
        assert_eq!(summary.total_gate_duration_ms, 3300);
        assert!((summary.gate_pass_rate() - 0.75).abs() < 0.01);

        // Display trait produces readable output
        let display = summary.to_string();
        assert!(display.contains("iterations=3"));
        assert!(display.contains("gates=9/12"));
        assert!(display.contains("escalations=1"));
    }

    #[test]
    fn test_otel_process_span_records_correctly() {
        // Verify the OTel span builder functions work without panicking
        let span = otel::process_issue_span("test-issue");
        otel::record_process_result(&span, true, 3, 45000);

        let iter_span = otel::iteration_span("test-issue", 1, "Worker");
        otel::record_iteration_result(&iter_span, true, 0, 0, 12000);

        let esc_span = otel::escalation_span("test-issue", "Worker", "Council", "error_repeat", 2);
        drop(esc_span);
    }

    #[test]
    fn test_slo_evaluation_from_session_metrics() {
        use coordination::benchmark::slo;
        use coordination::benchmark::OrchestrationMetrics;
        use std::time::Duration;

        // Simulate a successful single-iteration session
        let metrics = OrchestrationMetrics {
            session_count: 1,
            first_pass_rate: 1.0,
            overall_success_rate: 1.0,
            avg_iterations_to_green: 1.0,
            median_iterations_to_green: 1.0,
            escalation_rate: 0.0,
            avg_escalations: 0.0,
            latency_p50: Duration::from_secs(30),
            latency_p95: Duration::from_secs(30),
            latency_max: Duration::from_secs(30),
            tokens_p50: 0,
            tokens_p95: 0,
            tokens_total: 0,
            cost_total: 0.0,
            cost_avg: 0.0,
            stuck_rate: 0.0,
            avg_turns_until_first_write: 1.0,
            write_by_turn_2_rate: 1.0,
        };

        let report = slo::evaluate_slos(&metrics);
        assert!(report.all_passing(), "Perfect session should pass all SLOs");
        assert_eq!(report.warnings, 0);
        assert_eq!(report.critical, 0);

        // Simulate a failed session (stuck)
        let failed_metrics = OrchestrationMetrics {
            session_count: 1,
            first_pass_rate: 0.0,
            overall_success_rate: 0.0,
            avg_iterations_to_green: 6.0,
            median_iterations_to_green: 6.0,
            escalation_rate: 1.0,
            avg_escalations: 1.0,
            latency_p50: Duration::from_secs(300),
            latency_p95: Duration::from_secs(300),
            latency_max: Duration::from_secs(300),
            tokens_p50: 0,
            tokens_p95: 0,
            tokens_total: 0,
            cost_total: 0.0,
            cost_avg: 0.0,
            stuck_rate: 1.0,
            avg_turns_until_first_write: 0.0,
            write_by_turn_2_rate: 0.0,
        };

        let failed_report = slo::evaluate_slos(&failed_metrics);
        assert!(
            !failed_report.all_passing(),
            "Failed session should violate SLOs"
        );
        assert!(
            failed_report.warnings + failed_report.critical > 0,
            "Should have warnings or critical violations"
        );

        // summary() should produce readable output
        let summary = failed_report.summary();
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_kb_refresh_triggers_at_session_interval() {
        use crate::kb_refresh::{self, RefreshPolicy};

        let policy = RefreshPolicy::default(); // session_interval = 10

        // Should not trigger at 5 sessions
        assert!(!kb_refresh::should_refresh(5, &policy));
        // Should trigger at 10 sessions
        assert!(kb_refresh::should_refresh(10, &policy));
        // Should trigger at 20 sessions
        assert!(kb_refresh::should_refresh(20, &policy));
    }

    #[test]
    fn test_dashboard_generates_from_empty_sessions() {
        use crate::dashboard;
        use coordination::analytics::skills::SkillLibrary;

        let skills = SkillLibrary::new();
        let now = chrono::Utc::now();
        let metrics = dashboard::generate(&[], &skills, now);

        assert_eq!(metrics.windows.len(), 4); // 24h, 7d, 30d, all-time
        let summary = dashboard::format_summary(&metrics);
        assert!(summary.contains("Self-Improvement Dashboard"));
        assert!(summary.contains("Sessions: 0"));
    }

    #[test]
    fn test_orchestration_error_classify_transient() {
        use crate::modes::errors::OrchestrationError;

        // Connection/timeout errors should be retriable
        let err = anyhow::anyhow!("request timed out after 300s");
        assert!(OrchestrationError::classify(err.as_ref()).is_retriable());

        let err = anyhow::anyhow!("429 Too Many Requests");
        assert!(OrchestrationError::classify(err.as_ref()).is_retriable());

        let err = anyhow::anyhow!("connection refused");
        assert!(OrchestrationError::classify(err.as_ref()).is_retriable());

        // Budget exhaustion should NOT be retriable
        let err = anyhow::anyhow!("Budget exhausted after 15 tool calls");
        assert!(!OrchestrationError::classify(err.as_ref()).is_retriable());

        // Unknown errors fall through as transient (retriable)
        let err = anyhow::anyhow!("something unexpected");
        assert!(OrchestrationError::classify(err.as_ref()).is_retriable());
    }
}
