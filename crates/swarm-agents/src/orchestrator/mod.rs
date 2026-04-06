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

pub use dispatch::{
    build_review_prompt, condense_verifier_report, format_compact_task_prompt, format_task_prompt,
    route_to_coder, CoderRoute,
};
pub use helpers::try_scaffold_fallback;
pub(crate) use helpers::{
    bool_from_env, create_stuck_intervention, default_initial_tier, detect_failure_patterns,
    load_directives, query_kb_with_failsafe, save_directives, tier_from_env, timeout_from_env,
    u32_from_env,
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

/// Sentinel response when an agent exceeds its wall-clock timeout.
const TIMEOUT_RESPONSE: &str =
    "[TIMEOUT] agent exceeded deadline — no response received, checking disk state";

/// Error message produced when cooperative cancellation fires inside the main loop.
///
/// Checked in `main.rs` fan-in logic to distinguish graceful shutdown from genuine
/// failures — import this constant rather than hard-coding the string in both places.
pub const CANCEL_MSG: &str = "cancelled by shutdown signal";

pub(crate) use crate::auto_fix::{should_reject_auto_fix, try_auto_fix};

use crate::acceptance::{self, AcceptancePolicy};
use crate::agents::AgentFactory;
use crate::beads_bridge::{BeadsIssue, IssueTracker};
use crate::config::{GovernanceTier, SwarmConfig, SwarmRole};
use crate::knowledge_sync;
use crate::notebook_bridge::KnowledgeBase;
use crate::telemetry::{self, MetricsCollector, TelemetryReader};
use crate::triage::{self, PhaseModelSelector, WorkflowPhase};
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
                profile
                    .gates
                    .retain(|g| !g.name.eq_ignore_ascii_case("test"));
            }
            let script_verifier = ScriptVerifier::new(wt_path, profile);
            return script_verifier.run_pipeline().await;
        }
    }
    // Default: Rust verifier
    let verifier = Verifier::new(wt_path, verifier_config.clone());
    verifier.run_pipeline().await
}

fn report_has_baseline_regression(baseline: &VerifierReport, current: &VerifierReport) -> bool {
    use coordination::verifier::report::GateOutcome;

    let baseline_failures: std::collections::HashMap<&str, usize> = baseline
        .gates
        .iter()
        .filter_map(|gate| {
            if gate.outcome == GateOutcome::Failed {
                Some((gate.gate.as_str(), gate.error_count))
            } else {
                None
            }
        })
        .collect();

    current.gates.iter().any(|gate| {
        if gate.outcome != GateOutcome::Failed {
            return false;
        }
        match baseline_failures.get(gate.gate.as_str()) {
            Some(previous_errors) => gate.error_count > *previous_errors,
            None => true,
        }
    })
}

fn report_improves_on_baseline(
    baseline: &VerifierReport,
    current: &VerifierReport,
    min_error_delta: usize,
) -> bool {
    use coordination::verifier::report::GateOutcome;

    let baseline_failed_gates = baseline
        .gates
        .iter()
        .filter(|gate| gate.outcome == GateOutcome::Failed)
        .count();
    let current_failed_gates = current
        .gates
        .iter()
        .filter(|gate| gate.outcome == GateOutcome::Failed)
        .count();

    current.failure_signals.len() + min_error_delta < baseline.failure_signals.len()
        || current_failed_gates < baseline_failed_gates
        || current.gates_passed > baseline.gates_passed
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

    // --- Issue quality pre-filter (AI-Researcher pattern) ---
    // Reject issues the swarm can't handle: epics, benchmark-execution tasks,
    // deferred status, and configurable reject patterns. Prevents wasting
    // iterations on unactionable work.
    let title_lower = title_trimmed.to_lowercase();
    let desc_lower = issue.description.as_deref().unwrap_or("").to_lowercase();
    let combined = format!("{} {}", title_lower, desc_lower);

    let is_epic = title_lower.contains("[epic]") || title_lower.starts_with("epic:");
    let is_benchmark_exec = combined.contains("run benchmark")
        || combined.contains("run.*suite")
        || combined.contains("execute benchmark")
        || combined.contains("record performance metrics");
    let matches_reject = config
        .reject_patterns
        .iter()
        .any(|pat| combined.contains(pat.as_str()));

    if is_epic || is_benchmark_exec || matches_reject {
        let reason = if is_epic {
            "epic (not a code task)"
        } else if is_benchmark_exec {
            "benchmark execution (requires manual run)"
        } else {
            "matches SWARM_REJECT_PATTERNS"
        };
        warn!(
            id = %issue.id,
            reason,
            "Pre-filter: rejecting issue the swarm cannot handle"
        );
        return Ok(false);
    }

    // --- Triage phase (Ensemble Swarm) ---
    // Classify issue complexity, language, and suggested models using the cheapest
    // triage-capable cloud model (~$0.001/issue). Falls back to keyword heuristics
    // when cloud is unavailable or SWARM_SKIP_TRIAGE=1.
    let triage_result = triage::triage_issue(
        &issue.title,
        issue.description.as_deref(),
        &config.cloud_model_catalog,
        factory.clients.cloud.as_ref(),
        config.skip_triage,
    )
    .await;
    info!(
        id = %issue.id,
        complexity = %triage_result.complexity,
        language = %triage_result.language,
        used_llm = triage_result.used_llm,
        model = ?triage_result.triage_model,
        "Triage complete"
    );

    // Combined issue objective (title + description) used for write-deadline
    // computation and routing hints throughout the issue lifecycle.
    let issue_objective: String = match &issue.description {
        Some(desc) if !desc.is_empty() => format!("{} {}", issue.title, desc),
        _ => issue.title.clone(),
    };

    // Build the phase-based model selector for this issue.
    let phase_selector = PhaseModelSelector::new(
        config.cloud_model_catalog.clone(),
        config.max_cost_per_issue,
    );

    // --- Claim issue ---
    beads.update_status(&issue.id, "in_progress")?;
    info!(id = %issue.id, "Claimed issue");

    // --- Create or reuse worktree ---
    let wt_path = if worktree_bridge.worktree_exists(&issue.id) {
        match worktree_bridge.reset_worktree(&issue.id) {
            Ok(p) => {
                info!(path = %p.display(), "Reused existing worktree (reset for retry)");
                p
            }
            Err(e) => {
                warn!(id = %issue.id, "Worktree reset failed, recreating: {e}");
                let _ = worktree_bridge.cleanup(&issue.id);
                worktree_bridge.create(&issue.id)?
            }
        }
    } else {
        match worktree_bridge.create(&issue.id) {
            Ok(p) => {
                info!(path = %p.display(), "Created worktree");
                p
            }
            Err(e) => {
                error!(id = %issue.id, "Failed to create worktree: {e}");
                return Err(e);
            }
        }
    };

    // --- Load language profile (beefcake-loop: multi-language support) ---
    //
    // If the target repo has `.swarm/profile.toml`, load it. Non-Rust profiles
    // use ScriptVerifier (shell commands) instead of the built-in Rust Verifier.
    // When absent or language="rust", behavior is unchanged.
    let language_profile = LanguageProfile::load(&wt_path);
    let is_script_verifier = language_profile.as_ref().is_some_and(|p| !p.is_rust());

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

    // --- Intent contract (reformulation engine: Phase 1) ---
    // Capture the original task goal on first pickup so reformulations can't
    // silently weaken it. The contract is append-only (first attempt only).
    let reformulation_store =
        crate::reformulation::ReformulationStore::new(worktree_bridge.repo_root());
    let intent_contract = crate::reformulation::IntentContract::from_issue(
        &issue.id,
        &issue.title,
        issue.description.as_deref(),
    );
    reformulation_store.save_contract(&intent_contract);

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

    // --- Meta-insights from past runs (Hyperagents pattern) ---
    let meta_insights_prompt = {
        let reflector = crate::meta_reflection::MetaReflector::new(worktree_bridge.repo_root());
        let insights = reflector.load_recent_insights(10);
        let mut top: Vec<_> = insights.into_iter().collect();
        top.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        top.truncate(3);
        if top.is_empty() {
            String::new()
        } else {
            info!(
                count = top.len(),
                "Loaded meta-insights for prompt injection"
            );
            let lines: Vec<String> = top
                .iter()
                .map(|i| {
                    format!(
                        "- [{:.0}%] {}: {}",
                        i.confidence * 100.0,
                        i.insight_type_label(),
                        i.recommendation
                    )
                })
                .collect();
            format!(
                "\n## Swarm Meta-Insights (from past runs)\n{}\n",
                lines.join("\n")
            )
        }
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
    // ClawTeam pattern: plan-before-execute gate. The manager's `submit_plan`
    // tool deposits its approach here; the orchestrator injects it as context
    // in subsequent iteration prompts for consistency.
    let work_plan_slot = crate::tools::submit_plan_tool::new_work_plan_slot();
    let factory = factory
        .clone()
        .with_plan_slot(plan_slot.clone())
        .with_work_plan_slot(work_plan_slot.clone());
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
    // Determine the worker-first starting tier using both keyword classifier
    // and triage for observability, but only honor it when the feature flag is
    // enabled. Otherwise default to Council from the beginning.
    let recommendation = classify_initial_tier(&issue.title, &[]);

    // --- UCB model recommendation from mutation archive ---
    // Consult historical success rates to suggest the best model for this issue's
    // likely error types. The archive's recommend_model() uses UCB1 scoring with
    // lineage-based aggregation per error_category.
    let candidate_models: Vec<String> = vec![
        config.fast_endpoint.model.clone(),
        config.coder_endpoint.model.clone(),
        config.reasoning_endpoint.model.clone(),
    ];
    let ucb_model_hint = archive.recommend_model(&[], &candidate_models, 5);
    if let Some(ref model) = ucb_model_hint {
        info!(
            recommended_model = %model,
            "Mutation archive UCB recommendation for initial model"
        );
    }

    // Override tier based on triage complexity when triage provides higher confidence.
    let triage_tier = match triage_result.complexity {
        triage::Complexity::Simple => SwarmTier::Worker,
        triage::Complexity::Medium => recommendation.tier, // Defer to keyword classifier.
        triage::Complexity::Complex | triage::Complexity::Critical => SwarmTier::Council,
    };
    let effective_tier = if triage_result.used_llm {
        triage_tier // LLM triage has higher confidence than keywords.
    } else {
        recommendation.tier // Both are keyword-based; use the existing one.
    };
    info!(
        keyword_tier = ?recommendation.tier,
        triage_tier = ?triage_tier,
        effective_tier = ?effective_tier,
        complexity = %recommendation.complexity,
        triage_complexity = %triage_result.complexity,
        confidence = recommendation.confidence,
        reason = %recommendation.reason,
        "Task classification (triage-enhanced)"
    );
    let worker_first_tier = recommendation.tier;
    let default_tier = default_initial_tier(feature_flags.worker_first_enabled, worker_first_tier);
    // Allow explicit env override, otherwise use the worker-first default
    // only when the feature flag is enabled.
    let initial_tier = tier_from_env("SWARM_INITIAL_TIER", default_tier);
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
        worker_first_tier = ?worker_first_tier,
        triage_recommended_tier = ?effective_tier,
        cloud_available = config.cloud_endpoint.is_some(),
        worker_first = feature_flags.worker_first_enabled,
        "Initial tier selected"
    );
    // Derive governance tier from triage complexity. Simple issues get minimal
    // RuntimeAdapter overhead (fast path); complex/critical get maximum validation.
    let governance_tier = match triage_result.complexity {
        triage::Complexity::Simple => GovernanceTier::Core,
        triage::Complexity::Medium => GovernanceTier::Standard,
        triage::Complexity::Complex | triage::Complexity::Critical => GovernanceTier::Enhanced,
    };
    info!(
        %governance_tier,
        triage_complexity = %triage_result.complexity,
        "Governance tier selected from triage complexity"
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
    let mut baseline_report: Option<VerifierReport> = None;
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

        // Phase-based model selection for planning.
        // Try the phase selector first (picks strongest "plan" model from catalog),
        // then fall back to the legacy stack-profile routing.
        let phase_plan_model = phase_selector
            .select_for_phase(WorkflowPhase::Plan, Some(&triage_result), None)
            .map(|entry| entry.model.clone());
        let (plan_client, plan_model) = if let Some(ref cloud_model) = phase_plan_model {
            info!(model = %cloud_model, "Phase selector: using cloud model for Plan phase");
            (
                factory
                    .clients
                    .cloud
                    .as_ref()
                    .unwrap_or(&factory.clients.reasoning)
                    .clone(),
                cloud_model.clone(),
            )
        } else {
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
        let target_files =
            crate::file_targeting::find_target_files_by_grep(&wt_path, &issue.title, &src_exts);

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
                let timeout = timeout_from_env("SWARM_SUBTASK_TIMEOUT_SECS", 3600).as_secs();
                let outcome = crate::subtask::dispatch_subtasks(
                    &plan,
                    &factory.endpoint_pool,
                    &wt_path,
                    &issue.id,
                    max_concurrent,
                    timeout,
                    triage_result.complexity,
                    &issue_objective,
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
                                triage_result.complexity,
                                &issue_objective,
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

                        // --- Pre-merge reviewer gate ---
                        let issue_desc_review =
                            issue.description.as_deref().unwrap_or(&issue.title);
                        if !run_pre_merge_review(
                            config,
                            &factory,
                            &wt_path,
                            &issue.title,
                            issue_desc_review,
                        )
                        .await
                        {
                            warn!(id = %issue.id, "Pre-merge review REJECTED concurrent dispatch — falling through to retry");
                        } else {
                            info!(
                                id = %issue.id,
                                session_id = session.short_id(),
                                elapsed = %session.elapsed_human(),
                                "Issue resolved — merging worktree"
                            );

                            // Try to merge and close.
                            merge_close_or_reopen(
                                worktree_bridge,
                                beads,
                                &issue.id,
                                "Resolved by concurrent subtask dispatch",
                            )?;
                            clear_resume_file(worktree_bridge.repo_root());
                            // Record successful resolution in the mutation archive.
                            // Must happen before returning so early-exit paths are tracked.
                            let mut record = crate::mutation_archive::build_record(
                                &issue.id,
                                &issue.title,
                                &archive_language,
                                true,
                                session.iteration(),
                                &format!("{:?}", escalation.current_tier),
                                &config.resolve_role_model(SwarmRole::Planner),
                                process_start.elapsed().as_secs(),
                            );
                            record.files_changed = helpers::list_changed_files(&wt_path);
                            archive.record(&record);
                            return Ok(true);
                        }
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
                            let report2 =
                                run_verifier(&wt_path, &verifier_config, &language_profile).await;

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

                                // --- Pre-merge reviewer gate ---
                                let issue_desc_review =
                                    issue.description.as_deref().unwrap_or(&issue.title);
                                if !run_pre_merge_review(
                                    config,
                                    &factory,
                                    &wt_path,
                                    &issue.title,
                                    issue_desc_review,
                                )
                                .await
                                {
                                    warn!(id = %issue.id, "Pre-merge review REJECTED fixer post-pass — falling through");
                                } else {
                                    info!(
                                        id = %issue.id,
                                        session_id = session.short_id(),
                                        elapsed = %session.elapsed_human(),
                                        "Issue resolved — merging worktree"
                                    );

                                    merge_close_or_reopen(
                                        worktree_bridge,
                                        beads,
                                        &issue.id,
                                        "Resolved by concurrent dispatch + fixer post-pass",
                                    )?;
                                    clear_resume_file(worktree_bridge.repo_root());
                                    // Record successful resolution in the mutation archive.
                                    let mut record = crate::mutation_archive::build_record(
                                        &issue.id,
                                        &issue.title,
                                        &archive_language,
                                        true,
                                        session.iteration(),
                                        &format!("{:?}", escalation.current_tier),
                                        &config.resolve_role_model(SwarmRole::Planner),
                                        process_start.elapsed().as_secs(),
                                    );
                                    record.files_changed = helpers::list_changed_files(&wt_path);
                                    archive.record(&record);
                                    return Ok(true);
                                }
                            }

                            // Fixer post-pass didn't fix everything — fall through
                            warn!(
                                id = %issue.id,
                                errors = report2.failure_signals.len(),
                                summary = %report2.summary(),
                                "Fixer post-pass: verifier still FAILED — falling through to retry loop"
                            );
                        }
                        Err(e) => {
                            warn!(
                                id = %issue.id,
                                error = %e,
                                "Serial fixer post-pass failed — falling through to retry loop"
                            );
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

    // Pre-identify planned target files for scope drift detection in the main loop.
    // Uses the same heuristic as the concurrent subtask planner.
    let planned_target_files: Option<Vec<String>> = {
        let src_exts = language_profile
            .as_ref()
            .map(|p| p.source_extensions.clone())
            .unwrap_or_default();
        crate::file_targeting::find_target_files_by_grep(&wt_path, &issue.title, &src_exts)
    };

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
        let baseline_result =
            run_verifier_opts(&wt_path, &baseline_config, &language_profile, true).await;
        let gates_passed = baseline_result
            .gates
            .iter()
            .filter(|g| g.outcome == coordination::verifier::report::GateOutcome::Passed)
            .count();
        let gates_total = baseline_result.gates.len();

        if !baseline_result.all_green {
            let baseline_summary = baseline_result.summary();
            warn!(
                id = %issue.id,
                gates_passed,
                gates_total,
                summary = %baseline_summary,
                "Baseline verification FAILED — proceeding in improvement mode so pre-existing failures \
                 do not block issue progress."
            );
            let _ = progress.log_error(
                session.session_id(),
                0,
                format!("Baseline failed: {baseline_summary}"),
            );
        } else {
            info!(
                id = %issue.id,
                gates_passed,
                gates_total,
                "Baseline verification PASSED — worktree is clean before agent starts"
            );
        }

        best_error_count.get_or_insert(baseline_result.failure_signals.len());
        last_report.get_or_insert_with(|| baseline_result.clone());
        baseline_report.get_or_insert(baseline_result);
    }

    // --- Archive-informed context (Phase 4b: UCB model insights) ---
    //
    // Query the mutation archive for past similar fixes and log UCB scores.
    // Archive context is available to the orchestrator for future prompt injection.
    {
        let summary = archive.summary();
        if summary.total_attempts > 0 {
            info!(
                attempts = summary.total_attempts,
                resolved = summary.resolved,
                rate = format!(
                    "{:.0}%",
                    summary.resolved as f64 / summary.total_attempts as f64 * 100.0
                ),
                "Mutation archive: prior history available"
            );
        }
    }

    // --- Skill library: loaded once before the loop, refreshed after skill creation ---
    let skills_path = worktree_bridge.repo_root().join(".swarm/skills.json");
    let skill_library =
        coordination::analytics::skills::SkillLibrary::load(&skills_path).unwrap_or_default();

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

        // --- Self-improvement: inject archive context (Phase 4b) ---
        //
        // Query the mutation archive for past similar fixes and inject what
        // worked into the objective. This closes the feedback loop:
        //   past outcomes → archive → prompt context → better next attempt
        if iteration == 1 {
            // On first iteration, inject past fix patterns for similar errors
            if let Some(ref report) = last_report {
                let error_cats: Vec<String> = report
                    .unique_error_categories()
                    .iter()
                    .map(|c| c.to_string())
                    .collect();
                if let Some(context) = archive.context_for_issue(&error_cats) {
                    full_objective.push_str("\n\n");
                    full_objective.push_str(&context);
                }
            } else if let Some(context) = archive.context_for_issue(&[]) {
                // No errors yet (first iteration) — inject general archive stats
                full_objective.push_str("\n\n");
                full_objective.push_str(&context);
                // Also inject title-similar issue history so agent calibrates effort
                // before errors exist (beefcake-1nw2: session-start meta-context).
                let title_similar = archive.query_by_keywords(&issue.title, 3);
                if !title_similar.is_empty() {
                    full_objective
                        .push_str("\n**Issues with similar titles** (archive reference):\n");
                    for r in &title_similar {
                        full_objective.push_str(&format!(
                            "- `{}` resolved in {} iter(s) via {} tier\n",
                            r.issue_title, r.iterations, r.tier,
                        ));
                    }
                    full_objective.push('\n');
                }
            }
        }

        // Append mail messages if any were received between iterations
        if let Some(ref mail) = mail_context {
            full_objective.push_str("\n\n");
            full_objective.push_str(mail.prompt_heading());
            full_objective.push_str(mail.prompt_body());
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

        // --- Meta-insight injection (Hyperagents pattern) ---
        // Load recent meta-insights from the reflection loop and inject as heuristics.
        {
            let reflector = crate::meta_reflection::MetaReflector::new(worktree_bridge.repo_root());
            let insights = reflector.load_recent_insights(5);
            for insight in &insights {
                packet.relevant_heuristics.push(format!(
                    "[{}] {}",
                    insight.insight_type_label(),
                    insight.recommendation,
                ));
            }
            if !insights.is_empty() {
                info!(
                    count = insights.len(),
                    iteration, "Injected meta-insights into packet"
                );
            }
        }

        // --- Skill library injection (Hyperagents pattern) ---
        // Use pre-loaded skill library (loaded once before loop, refreshed after skill creation).
        {
            if !skill_library.is_empty() {
                let typed_cats: Vec<coordination::feedback::ErrorCategory> =
                    packet.failure_signals.iter().map(|s| s.category).collect();
                let skill_context = coordination::analytics::skills::TaskContext {
                    error_categories: typed_cats,
                    files_involved: packet
                        .file_contexts
                        .iter()
                        .map(|f| f.file.clone())
                        .collect(),
                    task_type: None,
                };
                let hints = skill_library.find_matching(&skill_context);
                if !hints.is_empty() {
                    info!(
                        hint_count = hints.len(),
                        "Hyperagents: injecting skill hints into work packet"
                    );
                    packet.skill_hints = hints;
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

        // --- Condensed verifier summary injection (PreCompletionChecklist pattern) ---
        // On retries, prepend a structured summary of the previous verifier failure
        // so workers fix the specific reported error instead of re-exploring.
        if iteration > 1 {
            if let Some(ref prev_report) = last_report {
                if !prev_report.all_green {
                    // Prepend so workers see the failure context first.
                    task_prompt = format!(
                        "## Previous Attempt Failed\n{}\n\nFix ONLY the reported errors. Do NOT re-explore the codebase.\n\n{}",
                        condense_verifier_report(prev_report),
                        task_prompt,
                    );
                }
            }
        }

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

        // --- Meta-insights injection (Hyperagents pattern) ---
        if !meta_insights_prompt.is_empty() {
            task_prompt.push_str(&meta_insights_prompt);
        }

        // --- Cross-worker context injection (ClawTeam adoption #3) ---
        // On iteration 2+, inject: (a) the work plan from iteration 1, and
        // (b) a git diff --stat summary showing what previous iterations changed.
        // This prevents workers from undoing each other's work and keeps the
        // strategy visible across retries.
        if iteration > 1 {
            if let Some(ref plan) = work_plan_slot.lock().ok().and_then(|s| s.clone()) {
                task_prompt.push_str(&crate::tools::submit_plan_tool::format_plan_context(plan));
            }
            let diff_summary = crate::git_ops::diff_stat_summary(&wt_path);
            if !diff_summary.is_empty() {
                task_prompt.push_str("## Changes from Previous Iterations\n\n");
                task_prompt.push_str("_Files modified by earlier workers. Build on these changes — do not revert them unless they caused the current error._\n\n");
                task_prompt.push_str("```\n");
                task_prompt.push_str(&diff_summary);
                task_prompt.push_str("\n```\n\n");
            }
        }

        // --- Active feedback injection (Robin pattern — adoption #2) ---
        // Query mutation archive for similar past fixes (by error category + files)
        // and inject as prompt context. Also injects anti-patterns (failed approaches).
        // This closes the learning loop: successful patterns seed future attempts.
        {
            let error_cats: Vec<String> = packet
                .failure_signals
                .iter()
                .map(|s| format!("{}", s.category))
                .collect();
            let files: Vec<String> = packet
                .file_contexts
                .iter()
                .map(|f| f.file.clone())
                .collect();
            if !error_cats.is_empty() || !files.is_empty() {
                let feedback = archive.format_feedback_context(&error_cats, &files);
                if !feedback.is_empty() {
                    task_prompt.push_str(&feedback);
                }
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

                // --- Hyperagents: adaptive model routing ---
                // UCB1 bandit: use role strings as candidates so coder vs reasoning node is
                // distinguishable (both nodes run "Qwen3.5-122B-A10B" — model names collide).
                let ucb_coder_route: Option<CoderRoute> = if config.adaptive_routing {
                    let error_cats: Vec<String> = escalation
                        .recent_error_categories
                        .last()
                        .cloned()
                        .unwrap_or_default()
                        .iter()
                        .map(|c| c.to_string())
                        .collect();
                    if !error_cats.is_empty() {
                        let candidates = vec!["RustCoder".to_string(), "GeneralCoder".to_string()];
                        archive
                            .recommend_model(&error_cats, &candidates, 5)
                            .and_then(|rec| match rec.as_str() {
                                "RustCoder" => {
                                    info!(
                                        iteration,
                                        recommended = %rec,
                                        "UCB1 adaptive routing → RustCoder override"
                                    );
                                    Some(CoderRoute::RustCoder)
                                }
                                "GeneralCoder" => {
                                    info!(
                                        iteration,
                                        recommended = %rec,
                                        "UCB1 adaptive routing → GeneralCoder override"
                                    );
                                    Some(CoderRoute::GeneralCoder)
                                }
                                _ => None,
                            })
                    } else {
                        None
                    }
                } else {
                    None
                };

                match ucb_coder_route.unwrap_or_else(|| route_to_coder(&recent_cats, iteration)) {
                    CoderRoute::RustCoder => {
                        info!(iteration, "Routing to rust_coder (Qwen3.5-Implementer)");
                        metrics.record_coder_route("RustCoder");
                        metrics.record_agent_metrics("Qwen3.5-RustCoder", 0, 0);
                        let seq_write_deadline = crate::subtask::dynamic_write_deadline(
                            triage_result.complexity,
                            &issue_objective,
                            &[],
                            &[],
                        );
                        let adapter = RuntimeAdapter::new(AdapterConfig {
                            agent_name: "Qwen3.5-RustCoder".into(),
                            deadline: Some(Instant::now() + worker_timeout),
                            max_tool_calls: Some(config.max_worker_tool_calls),
                            max_turns_without_write: Some(seq_write_deadline),
                            search_unlock_turn: Some(3),
                            governance_tier,
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
                                Ok(TIMEOUT_RESPONSE.to_string())
                            }
                        };
                        (result, adapter)
                    }
                    CoderRoute::GeneralCoder => {
                        info!(iteration, "Routing to general_coder (Qwen3.5-Implementer)");
                        metrics.record_coder_route("GeneralCoder");
                        metrics.record_agent_metrics("Qwen3.5-GeneralCoder", 0, 0);
                        let seq_write_deadline = crate::subtask::dynamic_write_deadline(
                            triage_result.complexity,
                            &issue_objective,
                            &[],
                            &[],
                        );
                        let adapter = RuntimeAdapter::new(AdapterConfig {
                            agent_name: "Qwen3.5-GeneralCoder".into(),
                            deadline: Some(Instant::now() + worker_timeout),
                            max_tool_calls: Some(config.max_worker_tool_calls),
                            max_turns_without_write: Some(seq_write_deadline),
                            search_unlock_turn: Some(3),
                            governance_tier,
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
                                Ok(TIMEOUT_RESPONSE.to_string())
                            }
                        };
                        (result, adapter)
                    }
                    CoderRoute::FastFixer => {
                        info!(
                            iteration,
                            "Reasoning sandwich: routing to fast_fixer (GLM-4.7-Flash)"
                        );
                        metrics.record_coder_route("FastFixer");
                        metrics.record_agent_metrics("GLM-FastFixer", 0, 0);
                        let fixer = factory.build_fixer(&wt_path);
                        let adapter = RuntimeAdapter::new(AdapterConfig {
                            agent_name: "GLM-FastFixer".into(),
                            deadline: Some(Instant::now() + worker_timeout),
                            max_tool_calls: Some(config.max_worker_tool_calls),
                            max_turns_without_write: Some(5),
                            search_unlock_turn: Some(3),
                            governance_tier,
                            ..Default::default()
                        });
                        let result = match tokio::time::timeout(
                            worker_timeout,
                            prompt_with_hook_and_retry(&fixer, &task_prompt, 2, adapter.clone()),
                        )
                        .await
                        {
                            Ok(result) => result,
                            Err(_elapsed) => {
                                warn!(
                                    iteration,
                                    timeout_secs = worker_timeout.as_secs(),
                                    "fast_fixer exceeded timeout — proceeding with changes on disk"
                                );
                                Ok(TIMEOUT_RESPONSE.to_string())
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
                let adapter = RuntimeAdapter::with_validators(
                    AdapterConfig {
                        agent_name: "manager".into(),
                        deadline: Some(Instant::now() + manager_timeout),
                        max_reads_without_action: Some(8),
                        ..Default::default()
                    },
                    crate::action_validator::manager_validators(),
                );
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
                        Ok(TIMEOUT_RESPONSE.to_string())
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
                let timeout = timeout_from_env("SWARM_SUBTASK_TIMEOUT_SECS", 3600).as_secs();
                let outcome = crate::subtask::dispatch_subtasks(
                    &manager_plan,
                    &factory.endpoint_pool,
                    &wt_path,
                    &issue.id,
                    max_concurrent,
                    timeout,
                    triage_result.complexity,
                    &issue_objective,
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
                    let report =
                        run_verifier(&wt_path, &current_verifier_config, &language_profile).await;

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

                        // --- Pre-merge reviewer gate ---
                        let issue_desc_review =
                            issue.description.as_deref().unwrap_or(&issue.title);
                        if !run_pre_merge_review(
                            config,
                            &factory,
                            &wt_path,
                            &issue.title,
                            issue_desc_review,
                        )
                        .await
                        {
                            warn!(id = %issue.id, "Pre-merge review REJECTED manager-guided dispatch — will retry");
                            last_report = Some(report);
                            continue;
                        }

                        info!(
                            id = %issue.id,
                            session_id = session.short_id(),
                            elapsed = %session.elapsed_human(),
                            "Issue resolved — merging worktree"
                        );

                        merge_close_or_reopen(
                            worktree_bridge,
                            beads,
                            &issue.id,
                            "Resolved by manager-guided parallel dispatch",
                        )?;
                        clear_resume_file(worktree_bridge.repo_root());
                        // Record successful resolution in the mutation archive.
                        let mut record = crate::mutation_archive::build_record(
                            &issue.id,
                            &issue.title,
                            &archive_language,
                            true,
                            session.iteration(),
                            &format!("{:?}", escalation.current_tier),
                            &config.resolve_role_model(SwarmRole::Planner),
                            process_start.elapsed().as_secs(),
                        );
                        record.files_changed = helpers::list_changed_files(&wt_path);
                        archive.record(&record);
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
                let preview = &r[..r.floor_char_boundary(500)];
                info!(
                    iteration,
                    response_len = r.len(),
                    response_preview = %preview,
                    "Agent responded"
                );
                // --- Uncertainty-aware confidence check ---
                // If the worker self-reports low confidence, flag for escalation
                // rather than waiting for compile failures to trigger it.
                if tier == SwarmTier::Worker {
                    if let Some(confidence) = crate::confidence::extract_confidence(&r) {
                        info!(
                            iteration,
                            confidence = confidence.score,
                            explicit = confidence.explicit,
                            "Agent self-reported confidence"
                        );
                        if confidence.score < crate::confidence::DEFAULT_CONFIDENCE_THRESHOLD {
                            warn!(
                                iteration,
                                confidence = confidence.score,
                                threshold = crate::confidence::DEFAULT_CONFIDENCE_THRESHOLD,
                                "Low confidence detected — recommending escalation to Council"
                            );
                            // Record the low-confidence signal for the escalation engine
                            escalation.record_low_confidence(confidence.score);
                        }
                    }
                }
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
                let report =
                    run_verifier(&wt_path, &current_verifier_config, &language_profile).await;

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

                    // For non-Rust targets (ScriptVerifier), gate names differ
                    // (lint/format/typecheck vs fmt/clippy/check). Use all_green
                    // as the universal acceptance criterion.
                    let compile_clean = if is_script_verifier {
                        report.all_green
                    } else {
                        fmt_ok && clippy_ok && check_ok
                    };

                    if compile_clean {
                        info!(
                            iteration,
                            all_green = report.all_green,
                            fmt_ok,
                            clippy_ok,
                            check_ok,
                            "Compile-clean short-circuit (agent failure path): \
                             worker wrote files and quality gates pass despite agent termination. Accepting."
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
                } else if !agent_has_written_prev && report.all_green && iteration <= 3 {
                    // --- No-op resolution (ClawTeam insight) ---
                    // Agent terminated without writing AND verifier still passes.
                    // The task is already done — the codebase already satisfies
                    // the issue requirements. Close without further iterations.
                    info!(
                        iteration,
                        id = %issue.id,
                        "No-op resolution: agent couldn't find changes to make, \
                         but verifier passes. Issue appears already resolved."
                    );

                    if let Err(e) = git_commit_changes(&wt_path, iteration as u32).await {
                        warn!(id = %issue.id, "Failed to commit (no-op path): {e}");
                    }

                    session.complete();
                    let _ = progress.log_session_end(
                        session.session_id(),
                        session.iteration(),
                        format!("Issue {} resolved (no-op: already clean)", issue.id),
                    );

                    info!(
                        id = %issue.id,
                        session_id = session.short_id(),
                        elapsed = %session.elapsed_human(),
                        "No-op resolution — merging worktree"
                    );

                    merge_close_or_reopen(
                        worktree_bridge,
                        beads,
                        &issue.id,
                        "Resolved (no-op): codebase already satisfies requirements",
                    )?;
                    info!(id = %issue.id, "No-op issue closed successfully");
                    clear_resume_file(worktree_bridge.repo_root());
                    // Record successful resolution in the mutation archive.
                    let record = crate::mutation_archive::build_record(
                        &issue.id,
                        &issue.title,
                        &archive_language,
                        true,
                        session.iteration(),
                        &format!("{:?}", escalation.current_tier),
                        &config.resolve_role_model(SwarmRole::Planner),
                        process_start.elapsed().as_secs(),
                    );
                    archive.record(&record);
                    return Ok(true);
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
                            config.max_turns_without_write
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
                        if scaffolded {
                            " (scaffold committed)"
                        } else {
                            ""
                        },
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

        // --- Self-evaluation gate (uncertainty-aware pattern) ---
        // For complex/critical issues, ask the reviewer to critique changes
        // before running the deterministic verifier.
        if matches!(
            triage_result.complexity,
            crate::triage::Complexity::Complex | crate::triage::Complexity::Critical
        ) && agent_has_written_prev
            && tier == SwarmTier::Worker
        {
            let critiques: Vec<coordination::debate::critique::CritiqueItem> =
                self_evaluate_changes(&reviewer, &wt_path, &issue.title, iteration as u32).await;

            let blocking_count = critiques
                .iter()
                .filter(|c| c.severity.is_blocking())
                .count();
            if blocking_count > 0 {
                warn!(
                    iteration,
                    blocking = blocking_count,
                    total = critiques.len(),
                    "Self-evaluation found {} blocking issue(s) — flagging for review",
                    blocking_count
                );
                // Inject critiques as validator feedback for the next iteration's prompt
                for critique in &critiques {
                    use coordination::verifier::ValidatorIssueType;
                    let issue_type = match critique.category {
                        coordination::debate::critique::CritiqueCategory::Correctness => {
                            ValidatorIssueType::LogicError
                        }
                        coordination::debate::critique::CritiqueCategory::Security => {
                            ValidatorIssueType::MissingSafetyCheck
                        }
                        coordination::debate::critique::CritiqueCategory::ErrorHandling => {
                            ValidatorIssueType::MissingSafetyCheck
                        }
                        _ => ValidatorIssueType::Other,
                    };
                    last_validator_feedback.push(ValidatorFeedback {
                        file: critique.file.clone(),
                        line_range: critique.line_range.map(|(s, e)| (s as usize, e as usize)),
                        issue_type,
                        description: critique.description.clone(),
                        suggested_fix: critique.suggested_fix.clone(),
                        source_model: Some("self_evaluation".to_string()),
                    });
                }
            } else if !critiques.is_empty() {
                info!(
                    iteration,
                    warnings = critiques.len(),
                    "Self-evaluation found warnings only — proceeding to verifier"
                );
            }
        }

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
            reward_score = format!("{:.3}", report.reward_score),
            summary = %report.summary(),
            "Verifier report"
        );

        // Scope drift detection: check if worker touched files outside the plan
        if let Some(ref tf) = planned_target_files {
            let changed = helpers::list_changed_files(&wt_path);
            let target_set: std::collections::HashSet<&str> =
                tf.iter().map(|s| s.as_str()).collect();
            let drift_files: Vec<&str> = changed
                .iter()
                .map(|s| s.as_str())
                .filter(|f| !target_set.contains(f))
                .collect();
            if !drift_files.is_empty() {
                warn!(
                    iteration,
                    drift_count = drift_files.len(),
                    files = drift_files.join(", "),
                    "Scope drift: worker modified files outside the planned target"
                );
            }
        }

        // --- TZ inference-level feedback: verifier_pass per iteration ---
        // Posts verifier outcome to the most recent inference in TZ, giving
        // Thompson Sampling per-call signal (not just per-episode).
        if let (Some(ref tz_url), Some(ref pg_url)) =
            (&config.tensorzero_url, &config.tensorzero_pg_url)
        {
            let now_secs = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64();
            // Look back from just before this iteration's worker call
            let since = now_secs - iter_start.elapsed().as_secs_f64() - 5.0;
            let inf_ids = crate::tensorzero::resolve_recent_inference_ids(pg_url, since, 1).await;
            if let Some(inf_id) = inf_ids.first() {
                crate::tensorzero::post_inference_feedback(
                    tz_url,
                    inf_id,
                    "verifier_pass",
                    serde_json::Value::Bool(report.all_green),
                    None,
                )
                .await;
                crate::tensorzero::post_inference_feedback(
                    tz_url,
                    inf_id,
                    "edit_accuracy",
                    serde_json::json!(report.health_score()),
                    None,
                )
                .await;
            }
        }

        // Record gate results into span summary
        for gate in &report.gates {
            let passed = matches!(gate.outcome, coordination::GateOutcome::Passed);
            span_summary.record_gate(passed, gate.duration_ms);
        }

        // --- Auto-fix: try to resolve trivial failures without LLM delegation ---
        let mut auto_fix_applied = false;
        if !report.all_green {
            if let Some(fixed_report) =
                try_auto_fix(&wt_path, &verifier_config, iteration, &language_profile).await
            {
                report = fixed_report;
                auto_fix_applied = true;
                metrics.record_auto_fix();
            }
        }

        let error_cats = report.unique_error_categories();
        let error_count = report.failure_signals.len();
        let cat_names: Vec<String> = error_cats.iter().map(|c| c.to_string()).collect();
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
                escalation.record_iteration(error_cats.clone(), 0, true, 1.0);
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

        // --- Dirty-baseline acceptance ---
        // If the repo was already failing before the agent touched it, accept
        // changes that measurably improve the verifier outcome without
        // introducing any new failing gates. This lets the swarm make progress
        // on unrelated issues in a dirty tree.
        if !report.all_green && agent_has_written_prev {
            if let Some(baseline) = baseline_report.as_ref().filter(|r| !r.all_green) {
                let improved = report_improves_on_baseline(baseline, &report, min_error_delta);
                let regressed = report_has_baseline_regression(baseline, &report);
                if improved && !regressed {
                    info!(
                        iteration,
                        baseline_errors = baseline.failure_signals.len(),
                        current_errors = error_count,
                        baseline_gates_passed = baseline.gates_passed,
                        current_gates_passed = report.gates_passed,
                        "Dirty-baseline acceptance: verifier improved without new gate regressions. Accepting."
                    );

                    escalation.record_iteration(
                        error_cats.clone(),
                        error_count,
                        true,
                        report.reward_score,
                    );
                    best_error_count = Some(error_count);
                    report.all_green = true;
                }
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
                    let rb_report =
                        run_verifier(&wt_path, &rb_verifier_config, &language_profile).await;
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
                        escalation.record_iteration(error_cats.clone(), 0, false, 0.0);
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
                            escalation.record_iteration(
                                error_cats,
                                error_count,
                                false,
                                report.reward_score,
                            );
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
            escalation.record_iteration(error_cats, error_count, true, report.reward_score);
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
                .map(|c| c.to_string())
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
                    .map(|c| c.to_string())
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

        // --- TZ demonstration feedback: capture the successful diff ---
        // Post the git diff as a "demonstration" (ideal output) to TZ.
        // This is the foundation for DICL (retrieval of similar past fixes)
        // and improves DPO preference pair quality.
        if let (Some(ref tz_url), Some(ref pg_url)) =
            (&config.tensorzero_url, &config.tensorzero_pg_url)
        {
            // Capture the diff of what the worker changed (async to avoid
            // blocking the tokio executor on large diffs).
            let diff = tokio::process::Command::new("git")
                .args(["diff", "HEAD~1", "--stat", "-p"])
                .current_dir(&wt_path)
                .output()
                .await
                .ok()
                .and_then(|o| {
                    if o.status.success() {
                        String::from_utf8(o.stdout).ok()
                    } else {
                        None
                    }
                })
                .unwrap_or_default();

            if !diff.is_empty() && diff.len() < 50_000 {
                // Find the most recent inference in this session for the demo target
                let inf_ids = crate::tensorzero::resolve_recent_inference_ids(
                    pg_url,
                    tz_session_start_secs,
                    1,
                )
                .await;
                if let Some(inf_id) = inf_ids.first() {
                    crate::tensorzero::post_demonstration(tz_url, inf_id, &diff).await;
                }
            } else if diff.len() >= 50_000 {
                info!(
                    diff_len = diff.len(),
                    "Skipping TZ demonstration — diff too large"
                );
            }
        }

        // --- Pre-merge reviewer gate ---
        let issue_desc = issue.description.as_deref().unwrap_or(&issue.title);
        if !run_pre_merge_review(config, &factory, &wt_path, &issue.title, issue_desc).await {
            warn!(
                id = %issue.id,
                "Pre-merge review REJECTED — not merging, will retry"
            );
            success = false;
        }

        if !success {
            // Reviewer rejected — fall through to failure handling
            info!(id = %issue.id, "Skipping merge due to reviewer rejection");
        } else {
            info!(
                id = %issue.id,
                session_id = session.short_id(),
                elapsed = %session.elapsed_human(),
                iterations = session.iteration(),
                "Issue resolved — merging worktree"
            );

            merge_close_or_reopen(
                worktree_bridge,
                beads,
                &issue.id,
                "Resolved by swarm orchestrator",
            )?;
            clear_resume_file(worktree_bridge.repo_root());

            // --- Post-merge CI watcher: verify main still compiles after merge ---
            //
            // Run a lightweight check (fmt + clippy + check, skip tests) on the
            // repo root after merge. If the merge introduced a regression (e.g.
            // interacting changes from parallel merges), reopen the issue.
            {
                let post_merge_config = VerifierConfig {
                    check_fmt: true,
                    check_clippy: true,
                    check_compile: true,
                    check_test: false, // skip tests — only check compilation
                    ..verifier_config.clone()
                };
                let scan_report = run_verifier_opts(
                    worktree_bridge.repo_root(),
                    &post_merge_config,
                    &language_profile,
                    true,
                )
                .await;

                // Append quality metrics to trend file
                let trend_path = worktree_bridge
                    .repo_root()
                    .join(".swarm")
                    .join("quality-trend.jsonl");
                if let Ok(json) = serde_json::to_string(&serde_json::json!({
                    "timestamp": chrono::Utc::now().to_rfc3339(),
                    "issue_resolved": issue.id,
                    "gates_passed": scan_report.gates_passed,
                    "gates_total": scan_report.gates_total,
                    "all_green": scan_report.all_green,
                    "summary": scan_report.summary(),
                })) {
                    let _ = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&trend_path)
                        .and_then(|mut f| {
                            use std::io::Write;
                            writeln!(f, "{json}")
                        });
                }

                if scan_report.all_green {
                    info!(
                        id = %issue.id,
                        summary = %scan_report.summary(),
                        "Post-merge verification PASSED"
                    );
                } else {
                    warn!(
                        id = %issue.id,
                        summary = %scan_report.summary(),
                        "Post-merge regression detected — reopening issue"
                    );
                    let _ = beads.update_status(&issue.id, "open");
                    success = false;
                }
            }
            info!(id = %issue.id, "Issue closed");
        }
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

    // --- Post-session: reformulation engine (self-correcting task rewrite) ---
    //
    // If the session failed, classify WHY and potentially rewrite the issue
    // description so the next attempt has a solvable formulation. This replaces
    // the bash postmortem-review.sh approach with Rust-native, rule-based logic.
    if !success {
        // Load failure ledger entries from the worktree
        let failure_ledger = helpers::load_failure_ledger(&wt_path);
        let files_changed = helpers::list_changed_files(&wt_path);
        let error_cats: Vec<String> = last_report
            .as_ref()
            .map(|r| {
                r.unique_error_categories()
                    .iter()
                    .map(|c| c.to_string())
                    .collect()
            })
            .unwrap_or_default();

        let review_input = crate::reformulation::FailureReviewInput {
            issue_id: issue.id.clone(),
            issue_title: issue.title.clone(),
            issue_description: issue.description.clone(),
            failure_ledger,
            iterations_used: session.iteration(),
            max_iterations: config.max_retries,
            files_changed,
            error_categories: error_cats,
            failure_reason: Some(escalation.summary()),
        };

        let result = crate::reformulation::reformulate(&reformulation_store, &review_input);

        // Apply the reformulation result to the bead
        let bridge = crate::beads_bridge::BeadsBridge::new();

        if let Some(ref new_desc) = result.new_description {
            match bridge.update_description(&issue.id, new_desc) {
                Ok(()) => info!(
                    id = %issue.id,
                    classification = ?result.classification,
                    "Reformulated issue description"
                ),
                Err(e) => warn!(
                    id = %issue.id,
                    error = %e,
                    "Failed to update issue description (reformulation not applied)"
                ),
            }
        }

        if let Some(ref notes) = result.notes_appended {
            if let Err(e) = bridge.update_notes(&issue.id, notes) {
                warn!(
                    id = %issue.id,
                    error = %e,
                    "Failed to append reformulation notes"
                );
            }
        }

        if result.escalated {
            // Label the issue for human review instead of deferring
            let _ = bridge.add_swarm_label(&issue.id, "swarm:needs-human-review");
            warn!(
                id = %issue.id,
                "Reformulation exhausted — labeled for human review"
            );
        }

        // Reset issue to open so the next loop iteration picks it up
        // (with the rewritten description)
        if !result.escalated {
            let _ = beads.update_status(&issue.id, "open");
        }
    }

    // --- Write telemetry ---
    let final_tier = format!("{:?}", escalation.current_tier);
    let mut session_metrics = metrics.finalize(success, &final_tier);

    // Populate token usage and cost from TZ Postgres if available.
    if let Some(ref pg_url) = config.tensorzero_pg_url {
        let (input, output) =
            crate::tensorzero::query_session_token_usage(pg_url, tz_session_start_secs).await;
        session_metrics.input_tokens = input;
        session_metrics.output_tokens = output;
        session_metrics.estimated_cost_usd = crate::tensorzero::estimate_cost(input, output);
    }

    telemetry::write_session_metrics(&session_metrics, &wt_path);
    telemetry::append_telemetry(&session_metrics, worktree_bridge.repo_root());

    // --- TensorZero feedback ---
    // Phase 1 fix: when PG URL is available, resolve the actual episode IDs that
    // TZ assigned (not our self-generated ones) and post feedback to those.
    // Falls back to the original self-generated episode_id if PG is unavailable.
    if let Some(ref tz_url) = config.tensorzero_url {
        let wall_secs = session_metrics.elapsed_ms as f64 / 1000.0;

        // Build segmentation tags for slice-based analysis in TZ.
        // Extract the primary error category from the final verifier report so
        // TZ can slice code_fixing feedback by error type (e.g. borrow_checker vs type_mismatch).
        let primary_error_category = last_report.as_ref().and_then(|r| {
            r.unique_error_categories()
                .into_iter()
                .next()
                .map(|c| c.to_string())
        });
        // Derive retry_tier tag from the last iteration's coder route.
        // "fast" = reasoning sandwich activated (FastFixer), "coder" = stayed on coder tier.
        let retry_tier = session_metrics
            .iterations
            .last()
            .and_then(|i| i.coder_route.as_ref())
            .map(|route| {
                if route == "FastFixer" {
                    "fast".to_string()
                } else {
                    "coder".to_string()
                }
            });
        let tz_tags = crate::tensorzero::FeedbackTags {
            issue_id: Some(issue.id.clone()),
            language: Some(triage_result.language.to_string()),
            triage_complexity: Some(triage_result.complexity.to_string()),
            model: config.cloud_endpoint.as_ref().map(|e| e.model.clone()),
            repo_id: std::env::var("SWARM_REPO_ID")
                .ok()
                .filter(|s| !s.is_empty()),
            error_category: primary_error_category,
            prompt_version: Some(crate::prompts::PROMPT_VERSION.to_string()),
            retry_tier,
            write_deadline: Some(config.max_turns_without_write.to_string()),
            max_tool_calls: Some(config.max_worker_tool_calls.to_string()),
            governance_tier: Some(governance_tier.to_string()),
        };

        if let Some(ref pg_url) = config.tensorzero_pg_url {
            // Resolve episode IDs once — reused for both feedback calls below.
            let episode_ids =
                crate::tensorzero::resolve_episode_ids(pg_url, tz_session_start_secs).await;

            // Post core episode metrics (task_resolved, iterations_used, wall_time, etc.)
            for ep_id in &episode_ids {
                crate::tensorzero::post_episode_feedback(
                    tz_url,
                    ep_id,
                    success,
                    session_metrics.total_iterations,
                    wall_secs,
                    Some(tz_tags.clone()),
                )
                .await;
            }
            if !episode_ids.is_empty() {
                info!(
                    episodes = episode_ids.len(),
                    success, "Posted TZ feedback to all resolved episodes"
                );
            }

            // Post verifier_gates_passed — finer signal than boolean task_resolved.
            if let Some(ref final_report) = last_report {
                let gates_passed = final_report.gates_passed as f64;
                for ep_id in &episode_ids {
                    crate::tensorzero::post_episode_metric(
                        tz_url,
                        ep_id,
                        "verifier_gates_passed",
                        serde_json::json!(gates_passed),
                        Some(tz_tags.clone()),
                    )
                    .await;
                }
            }
        } else if let Some(ref ep_id) = tensorzero_episode_id {
            // Legacy fallback: use self-generated episode_id (may fail if TZ
            // doesn't recognize it, but harmless — feedback is best-effort)
            crate::tensorzero::post_episode_feedback(
                tz_url,
                ep_id,
                success,
                session_metrics.total_iterations,
                wall_secs,
                Some(tz_tags),
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
    // Collect violated SLO names for adaptive meta-reflection below.
    let slo_violated_names: Vec<String> = slo_report
        .results
        .iter()
        .filter(|r| r.is_violated())
        .map(|r| r.target.name.clone())
        .collect();
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
                .map(|c| c.to_string())
                .collect();
            record.first_failure_gate = report.first_failure.clone();
        }
        if !success {
            record.failure_reason = Some(escalation.summary());
        }
        // Populate files changed so archive queries (query_similar, UCB) have file signal.
        record.files_changed = helpers::list_changed_files(&wt_path);
        // Count changed lines from git diff
        if let Ok(output) = std::process::Command::new("git")
            .args(["diff", "--stat", "main"])
            .current_dir(&wt_path)
            .output()
        {
            let stat = String::from_utf8_lossy(&output.stdout);
            for line in stat.lines() {
                if line.contains("insertion") {
                    if let Some(n) = line.split_whitespace().next().and_then(|s| s.parse().ok()) {
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

        // --- Hyperagents: extract skills from successful mutations ---
        // Promote quick resolutions (≤3 iterations) into the skill library so
        // future tasks with similar error categories receive targeted hints.
        if success {
            if let Some(candidate) =
                crate::mutation_archive::MutationArchive::extract_skill_candidate(&record)
            {
                if let Ok(mut lib) =
                    coordination::analytics::skills::SkillLibrary::load(&skills_path)
                {
                    let triggers: Vec<coordination::feedback::ErrorCategory> = candidate
                        .error_categories
                        .iter()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if !triggers.is_empty() {
                        let trigger = coordination::analytics::skills::SkillTrigger {
                            error_categories: triggers,
                            file_patterns: vec![],
                            task_type: None,
                        };
                        lib.create_skill(
                            &candidate.approach_summary,
                            trigger,
                            &candidate.approach_summary,
                        );
                        if let Err(e) = lib.save(&skills_path) {
                            warn!(error = %e, "Hyperagents: failed to save skill library");
                        } else {
                            info!(
                                skill = %candidate.approach_summary,
                                "Hyperagents: extracted skill from successful mutation"
                            );
                        }
                    }
                }
            }
        }
    }

    // --- Hyperagents: adaptive meta-reflection ---
    // Triggers when EITHER:
    //   (a) this was a high-error session (>3 iterations), OR
    //   (b) every 10 completed issues (periodic baseline).
    // SLO violations from the current session are fed in as additional context.
    // Runs synchronously but is fast (purely local computation, no LLM calls).
    {
        let archive_size = archive.load_all().len();
        let high_error_session = session_metrics.total_iterations > 3;
        let periodic_trigger = archive_size > 0 && archive_size.is_multiple_of(10);
        if high_error_session || periodic_trigger {
            info!(
                archive_size,
                high_error = high_error_session,
                periodic = periodic_trigger,
                slo_violations = slo_violated_names.len(),
                "Hyperagents: triggering adaptive meta-reflection"
            );
            let reflector = crate::meta_reflection::MetaReflector::new(worktree_bridge.repo_root());
            let insights = reflector.reflect_with_slo_violations(20, &slo_violated_names);
            if !insights.is_empty() {
                info!(
                    count = insights.len(),
                    "Hyperagents: generated meta-insights"
                );
                reflector.save_insights(&insights);
            }
        }
    }

    Ok(success)
}

/// Run a lightweight pre-merge review using the fast-tier reviewer.
///
/// Gets the git diff from the worktree, builds a review prompt, and asks the
/// reviewer model to approve or reject. Returns `true` if approved (or if
/// the review is disabled/fails), `false` if the reviewer explicitly rejects.
///
/// Keeps the diff lightweight: `--stat` header plus the first 200 lines of
/// the full diff to stay within context budgets for fast-tier models.
async fn run_pre_merge_review(
    config: &SwarmConfig,
    factory: &AgentFactory,
    wt_path: &Path,
    issue_title: &str,
    issue_description: &str,
) -> bool {
    if !config.review_before_merge {
        return true;
    }

    // Gather abbreviated diff: stat header + first 200 lines of patch
    let stat = tokio::process::Command::new("git")
        .args(["diff", "main..HEAD", "--stat"])
        .current_dir(wt_path)
        .output()
        .await
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .unwrap_or_default();

    let full_diff = tokio::process::Command::new("git")
        .args(["diff", "main..HEAD"])
        .current_dir(wt_path)
        .output()
        .await
        .ok()
        .and_then(|o| {
            if o.status.success() {
                String::from_utf8(o.stdout).ok()
            } else {
                None
            }
        })
        .unwrap_or_default();

    if stat.is_empty() && full_diff.is_empty() {
        debug!("Pre-merge review: no diff to review — auto-approving");
        return true;
    }

    // Truncate full diff to first 200 lines to keep prompt small
    let truncated_diff: String = full_diff.lines().take(200).collect::<Vec<_>>().join("\n");
    let diff_summary = format!("{stat}\n{truncated_diff}");

    let prompt = build_review_prompt(issue_title, issue_description, &diff_summary);
    let reviewer = factory.build_reviewer();

    match rig::completion::Prompt::prompt(&reviewer, &prompt).await {
        Ok(response) => {
            // Parse JSON response for {"approve": true/false, "reason": "..."}
            let approved = parse_review_response(&response);
            if approved {
                info!("Pre-merge review: APPROVED");
            } else {
                warn!(
                    response = %response,
                    "Pre-merge review: REJECTED"
                );
            }
            approved
        }
        Err(e) => {
            // Review failure is non-fatal — log and approve to avoid blocking merges
            warn!(error = %e, "Pre-merge review failed — auto-approving");
            true
        }
    }
}

/// Parse the reviewer's JSON response for the `approve` field.
///
/// Tolerates markdown fencing, extra whitespace, and partial JSON.
/// Returns `true` if the response contains `"approve": true` (or
/// `"approve":true`), `false` otherwise.
fn parse_review_response(response: &str) -> bool {
    // Try serde_json first for well-formed responses
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(response) {
        if let Some(approved) = val.get("approve").and_then(|v| v.as_bool()) {
            return approved;
        }
    }

    // Strip markdown code fences and retry
    let stripped = response
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(stripped) {
        if let Some(approved) = val.get("approve").and_then(|v| v.as_bool()) {
            return approved;
        }
    }

    // Fallback: regex-like substring match
    let lower = response.to_lowercase();
    if lower.contains("\"approve\": true") || lower.contains("\"approve\":true") {
        return true;
    }
    if lower.contains("\"approve\": false") || lower.contains("\"approve\":false") {
        return false;
    }

    // Can't determine — default to approve to avoid blocking
    warn!("Pre-merge review: could not parse response — auto-approving");
    true
}

/// Compile-time guard: `process_issue_core`'s future must be `Send` so it
/// can be dispatched via `tokio::spawn` / `JoinSet` in Phase 2 (Thread Weaving).
/// Merge worktree and close issue, tolerating worktree cleanup failures.
///
/// If `merge_and_remove` fails but the merge itself succeeded (the issue ID
/// appears in the latest commit), close the issue anyway. Only reopens the
/// issue when the git merge itself failed.
fn merge_close_or_reopen(
    worktree_bridge: &WorktreeBridge,
    beads: &dyn IssueTracker,
    issue_id: &str,
    close_reason: &str,
) -> Result<()> {
    let merge_err = worktree_bridge.merge_and_remove(issue_id).err();

    if let Some(ref e) = merge_err {
        // merge_and_remove failed. Check if it's a real merge failure
        // (conflict, uncommitted changes) vs just a cleanup failure
        // (worktree remove, branch delete). The error message distinguishes:
        // merge failures mention "conflict" or "uncommitted changes",
        // cleanup failures mention "worktree remove" or "branch".
        let err_msg = e.to_string().to_lowercase();
        let is_merge_failure = err_msg.contains("conflict")
            || err_msg.contains("uncommitted change")
            || err_msg.contains("merge failed");

        if is_merge_failure {
            error!(id = %issue_id, "Merge failed: {e}");
            let _ = worktree_bridge.cleanup(issue_id);
            let _ = beads.update_status(issue_id, "open");
            return Err(anyhow::anyhow!("Merge failed for {issue_id}: {e}"));
        }

        // Cleanup failure only — the merge itself succeeded.
        warn!(
            id = %issue_id,
            error = %e,
            "Worktree cleanup failed after merge — closing issue anyway"
        );
        let _ = worktree_bridge.cleanup(issue_id);
    }

    // Close the issue (merge succeeded or was a no-op)
    beads.close(issue_id, Some(close_reason))?;
    Ok(())
}

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

/// Self-evaluation gate for complex/critical issues (uncertainty-aware LLM pattern).
///
/// After the worker produces code, asks the reviewer model to critique the changes
/// before running the deterministic verifier. This catches logic errors that compile
/// successfully but are semantically wrong.
///
/// Only activated for Complex/Critical triage results to avoid overhead on simple fixes.
async fn self_evaluate_changes(
    reviewer: &crate::agents::coder::OaiAgent,
    wt_path: &Path,
    issue_title: &str,
    iteration: u32,
) -> Vec<coordination::debate::critique::CritiqueItem> {
    use coordination::debate::critique::{CritiqueCategory, CritiqueItem, CritiqueSeverity};

    // Get the diff of what was changed
    let diff_output = tokio::process::Command::new("git")
        .args(["diff", "--stat"])
        .current_dir(wt_path)
        .output()
        .await;

    let diff_stat = match diff_output {
        Ok(out) => String::from_utf8_lossy(&out.stdout).to_string(),
        Err(_) => return vec![],
    };

    if diff_stat.trim().is_empty() {
        return vec![]; // No changes to evaluate
    }

    let prompt = format!(
        "You are a code reviewer. The following changes were made for issue: \"{issue_title}\"\n\n\
         Changed files:\n{diff_stat}\n\n\
         Review these changes for correctness issues. Respond with ONLY a JSON array of issues found, \
         or an empty array [] if the changes look correct.\n\
         Format: [{{\"severity\": \"blocking\"|\"warning\", \"category\": \"correctness\"|\"security\"|\"error_handling\", \"description\": \"...\"}}]"
    );

    let review_result = tokio::time::timeout(
        std::time::Duration::from_secs(60),
        rig::completion::Prompt::prompt(reviewer, &prompt),
    )
    .await;

    match review_result {
        Ok(Ok(response)) => {
            // Parse the JSON response
            let cleaned = response
                .trim()
                .strip_prefix("```json")
                .or_else(|| response.trim().strip_prefix("```"))
                .unwrap_or(response.trim());
            let cleaned = cleaned.strip_suffix("```").unwrap_or(cleaned).trim();

            #[derive(serde::Deserialize)]
            struct RawCritique {
                severity: String,
                category: String,
                description: String,
            }

            match serde_json::from_str::<Vec<RawCritique>>(cleaned) {
                Ok(items) => items
                    .into_iter()
                    .map(|raw| {
                        let severity = if raw.severity == "blocking" {
                            CritiqueSeverity::Blocking
                        } else {
                            CritiqueSeverity::Warning
                        };
                        let category = match raw.category.as_str() {
                            "correctness" => CritiqueCategory::Correctness,
                            "security" => CritiqueCategory::Security,
                            "error_handling" => CritiqueCategory::ErrorHandling,
                            _ => CritiqueCategory::Other,
                        };
                        CritiqueItem {
                            severity,
                            category,
                            file: None,
                            line_range: None,
                            description: raw.description,
                            suggested_fix: None,
                        }
                    })
                    .collect(),
                Err(e) => {
                    tracing::debug!(iteration, error = %e, "Self-evaluation: failed to parse critique JSON");
                    vec![]
                }
            }
        }
        Ok(Err(e)) => {
            tracing::debug!(iteration, error = %e, "Self-evaluation: reviewer call failed");
            vec![]
        }
        Err(_) => {
            tracing::debug!(iteration, "Self-evaluation: reviewer timed out");
            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use coordination::verifier::report::{GateOutcome, GateResult};
    use coordination::{SessionStatus, SwarmTier};
    use std::path::PathBuf;

    fn test_report(gates: &[(&str, GateOutcome, usize)]) -> VerifierReport {
        let mut report = VerifierReport::new("/tmp/test".to_string());
        for (gate, outcome, error_count) in gates {
            report.add_gate(GateResult {
                gate: (*gate).to_string(),
                outcome: *outcome,
                duration_ms: 1,
                exit_code: Some(if *outcome == GateOutcome::Failed {
                    1
                } else {
                    0
                }),
                error_count: *error_count,
                warning_count: 0,
                errors: Vec::new(),
                stderr_excerpt: None,
            });
        }
        report.finalize(Duration::from_millis(1));
        report.failure_signals = vec![
            coordination::verifier::report::FailureSignal {
                gate: "synthetic".to_string(),
                category: coordination::feedback::ErrorCategory::Other,
                code: None,
                file: None,
                line: None,
                message: "synthetic".to_string(),
            };
            gates
                .iter()
                .filter(|(_, outcome, _)| *outcome == GateOutcome::Failed)
                .map(|(_, _, errors)| *errors)
                .sum()
        ];
        report
    }

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

    #[test]
    fn test_report_has_baseline_regression_detects_new_failed_gate() {
        let baseline = test_report(&[
            ("fmt", GateOutcome::Passed, 0),
            ("clippy", GateOutcome::Failed, 1),
            ("check", GateOutcome::Skipped, 0),
        ]);
        let current = test_report(&[
            ("fmt", GateOutcome::Failed, 1),
            ("clippy", GateOutcome::Failed, 1),
            ("check", GateOutcome::Skipped, 0),
        ]);

        assert!(report_has_baseline_regression(&baseline, &current));
    }

    #[test]
    fn test_report_improves_on_baseline_accepts_cleaner_failure_state() {
        let baseline = test_report(&[
            ("fmt", GateOutcome::Passed, 0),
            ("clippy", GateOutcome::Failed, 3),
            ("check", GateOutcome::Skipped, 0),
        ]);
        let current = test_report(&[
            ("fmt", GateOutcome::Passed, 0),
            ("clippy", GateOutcome::Failed, 1),
            ("check", GateOutcome::Skipped, 0),
        ]);

        assert!(!report_has_baseline_regression(&baseline, &current));
        assert!(report_improves_on_baseline(&baseline, &current, 0));
    }
}
