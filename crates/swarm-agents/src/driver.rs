//! State-machine-driven orchestrator loop.
//!
//! Replaces the monolithic `loop {}` in `orchestrator.rs` with typed state handlers.
//! Each handler returns a `StateTransition` telling the driver which state to enter next.
//!
//! Gated behind `SWARM_STATE_DRIVER=1` — the legacy path remains the default.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Instant;

/// Serializes concurrent Explorer calls to prevent GPU contention on the
/// reasoning endpoint (vasp-02 / Devstral).
///
/// With `--parallel N`, multiple issues route to ExplorerCoder simultaneously
/// and all land on the same vasp-02 endpoint. Running two large Explorer
/// inference sessions concurrently on a single V100S fills the KV cache and
/// causes llama.cpp to return empty completions. A semaphore of size 1 ensures
/// only one Explorer runs at a time; GeneralCoder phases remain fully
/// concurrent since the permit is dropped before Phase 2 begins.
///
/// `tokio::sync::Semaphore` works across separate thread-local runtimes
/// (one per `std::thread::spawn + block_on`) because it only uses the `Waker`
/// from the calling context, not a shared runtime scheduler.
static EXPLORER_GATE: OnceLock<Arc<tokio::sync::Semaphore>> = OnceLock::new();

use anyhow::{Context as _, Result};
use tracing::{debug, error, info, warn};

use crate::acceptance::{self, AcceptancePolicy};
use crate::agents::coder::OaiAgent;
use crate::agents::AgentFactory;
use crate::beads_bridge::{BeadsBridge, BeadsIssue, IssueTracker};
use crate::cluster_health::ClusterHealth;
use crate::config::{SwarmConfig, SwarmRole};
use crate::file_targeting::detect_changed_packages;
use crate::harness::HarnessPolicy;
use crate::knowledge_sync;
use crate::notebook_bridge::KnowledgeBase;
use crate::orchestrator::{
    self, bool_from_env, cloud_validate, collect_artifacts_from_diff, count_diff_lines,
    create_stuck_intervention, extract_local_validator_feedback, extract_validator_feedback,
    format_compact_task_prompt, format_task_prompt, git_commit_changes, land_issue_or_reopen,
    local_validate, needs_cot_planner, prompt_with_hook_and_retry, query_kb_with_failsafe,
    route_to_coder, should_reject_auto_fix, try_auto_fix, try_scaffold_fallback, CoderRoute,
    SwarmResumeFile,
};
use crate::reformulation::{IntentContract, ReformulationStore};
use crate::runtime_adapter::{AdapterConfig, RuntimeAdapter};
use crate::state_machine::{BudgetTracker, OrchestratorState, StateMachine};
use crate::telemetry::{self, MetricsCollector, TelemetryReader};
use crate::worktree_bridge::WorktreeBridge;
use coordination::benchmark::slo::{self, AlertSeverity};
use coordination::benchmark::OrchestrationMetrics;
use coordination::escalation::state::EscalationReason;
use coordination::escalation::worker_first::classify_initial_tier;
use coordination::feedback::ErrorCategory;
use coordination::otel::{self, SpanSummary};
use coordination::rollout::FeatureFlags;
use coordination::router::task_classifier::{DynamicRouter, ModelTier};
use coordination::save_session_state;
use coordination::TieredCorrectionLoop;
use coordination::{
    ContextPacker, EscalationDecision, EscalationEngine, EscalationState, GitManager,
    ProgressTracker, SessionManager, SwarmTier, TierBudget, TurnPolicy, ValidatorFeedback,
    Verifier, VerifierConfig, VerifierReport,
};

// ---------------------------------------------------------------------------
// StateTransition — the return type for every state handler
// ---------------------------------------------------------------------------

/// What a state handler wants the driver to do next.
pub enum StateTransition {
    /// Move to a new state with a reason string (for the audit log).
    Advance {
        to: OrchestratorState,
        reason: String,
    },
    /// The issue has been fully resolved — enter `Resolved` terminal state.
    Complete,
    /// An unrecoverable error — enter `Failed` terminal state.
    Fail { reason: String },
}

// ---------------------------------------------------------------------------
// OrchestratorContext — consolidates all implicit state from process_issue()
// ---------------------------------------------------------------------------

/// All the state that was previously scattered across ~15 local variables
/// in `process_issue()`.
pub struct OrchestratorContext<'a> {
    // ── Immutable config ──
    pub config: &'a SwarmConfig,
    pub issue: &'a BeadsIssue,
    pub wt_path: PathBuf,
    pub verifier_config: VerifierConfig,
    pub acceptance_policy: AcceptancePolicy,
    pub feature_flags: FeatureFlags,
    pub worker_timeout: std::time::Duration,
    pub manager_timeout: std::time::Duration,

    // ── Agents ──
    pub rust_coder: OaiAgent,
    pub general_coder: OaiAgent,
    pub reviewer: OaiAgent,
    pub manager: OaiAgent,

    // ── Harness ──
    pub session: SessionManager,
    pub git_mgr: GitManager,
    pub progress: ProgressTracker,

    // ── Mutable loop state ──
    pub state_machine: StateMachine,
    pub budget_tracker: BudgetTracker,
    pub escalation: EscalationState,
    pub metrics: MetricsCollector,
    pub span_summary: SpanSummary,
    pub success: bool,
    pub last_report: Option<VerifierReport>,
    pub last_validator_feedback: Vec<ValidatorFeedback>,
    pub consecutive_validator_failures: u32,

    // ── Per-iteration transient ──
    pub pre_worker_commit: Option<String>,
    pub auto_fix_applied: bool,
    /// CoT planner output from the last iteration (if CoTPlanner was routed to).
    /// Injected into the next iteration's worker prompt as planning context.
    pub last_plan_output: Option<String>,
    /// How many times the CoT planner has been invoked this session.
    /// Capped at 1 to prevent the planner from consuming all iteration budget.
    pub cot_planner_invocations: u32,

    // ── Coordination integrations (P2.2 + P2.3) ──
    pub router: DynamicRouter,
    pub correction_loop: TieredCorrectionLoop,
    pub last_harness_policy: Option<HarnessPolicy>,

    // ── Reformulation engine ──
    pub reformulation_store: ReformulationStore,
    pub intent_contract: IntentContract,

    // ── External deps ──
    pub factory: &'a AgentFactory,
    pub beads: &'a dyn IssueTracker,
    pub knowledge_base: Option<&'a dyn KnowledgeBase>,
    pub worktree_bridge: &'a WorktreeBridge,
    pub cluster_health: ClusterHealth,

    // ── Validator config ──
    pub local_validator_enabled: bool,
    pub max_validator_failures: u32,

    // ── OTel ──
    pub process_span: tracing::Span,
    pub process_start: Instant,

    // ── Event-sourced session log ──
    /// Opened lazily in handle_preparing_worktree() once the worktree exists.
    pub session_log: Option<crate::session::SessionLog>,

    // ── Lazy provisioning flag ──
    /// Whether the worktree has been created yet. Set to true in
    /// handle_preparing_worktree() after worktree_bridge.create() succeeds.
    pub worktree_provisioned: bool,
}

impl<'a> OrchestratorContext<'a> {
    /// Safely append an event to the session log (no-op if log not yet opened).
    pub fn log_event(&self, kind: crate::session::EventKind) {
        if let Some(ref log) = self.session_log {
            let _ = log.append(kind);
        }
    }

    /// Build the context from the same inputs as `process_issue()`.
    ///
    /// Performs all one-time initialization: feature flags, worktree creation,
    /// harness setup, agent builds, escalation state, health preflight.
    /// The caller should drive the state machine after this returns.
    pub async fn new(
        config: &'a SwarmConfig,
        factory: &'a AgentFactory,
        worktree_bridge: &'a WorktreeBridge,
        issue: &'a BeadsIssue,
        beads: &'a dyn IssueTracker,
        knowledge_base: Option<&'a dyn KnowledgeBase>,
    ) -> Result<OrchestratorContext<'a>> {
        let worker_policy = TurnPolicy::for_tier(SwarmTier::Worker);
        let council_policy = TurnPolicy::for_tier(SwarmTier::Council);
        let worker_timeout =
            orchestrator::timeout_from_env("SWARM_WORKER_TIMEOUT_SECS", worker_policy.timeout_secs);
        let manager_timeout = orchestrator::timeout_from_env(
            "SWARM_MANAGER_TIMEOUT_SECS",
            council_policy.timeout_secs,
        );

        let process_span = otel::process_issue_span(&issue.id);
        let process_start = Instant::now();

        let feature_flags = FeatureFlags::from_env();
        info!(flags = %feature_flags, summary = %feature_flags.summary(), "Feature flags loaded (state driver)");

        let acceptance_policy = AcceptancePolicy::default();
        let local_validator_enabled = bool_from_env("SWARM_LOCAL_VALIDATOR", true);
        let max_validator_failures = orchestrator::u32_from_env("SWARM_MAX_VALIDATOR_FAILURES", 3);

        let cluster_health = ClusterHealth::from_config(config);

        // Build the state machine and budget tracker
        let state_machine = StateMachine::new();
        let budget_tracker = BudgetTracker::with_defaults();

        // Determine initial tier.
        let council_budget_iterations =
            orchestrator::u32_from_env("SWARM_COUNCIL_MAX_ITERATIONS", 6);
        let council_budget_consultations =
            orchestrator::u32_from_env("SWARM_COUNCIL_MAX_CONSULTATIONS", 6);
        // When worker_first is enabled, classify the task. Otherwise default
        // to Council from the beginning.
        let recommendation = classify_initial_tier(&issue.title, &[]);
        info!(
            tier = ?recommendation.tier,
            complexity = %recommendation.complexity,
            confidence = recommendation.confidence,
            reason = %recommendation.reason,
            "Task classification (state driver)"
        );
        let worker_first_tier = recommendation.tier;
        let default_tier = orchestrator::default_initial_tier(
            feature_flags.worker_first_enabled,
            worker_first_tier,
        );
        let initial_tier = orchestrator::tier_from_env("SWARM_INITIAL_TIER", default_tier);
        let initial_tier = if initial_tier == SwarmTier::Council
            && config.cloud_endpoint.is_none()
            && std::env::var("SWARM_INITIAL_TIER").is_err()
        {
            warn!("Council tier requires cloud endpoint; falling back to Worker (state driver)");
            SwarmTier::Worker
        } else {
            initial_tier
        };
        info!(
            ?initial_tier,
            worker_first_tier = ?worker_first_tier,
            cloud_available = config.cloud_endpoint.is_some(),
            worker_first = feature_flags.worker_first_enabled,
            "Initial tier selected (state driver)"
        );

        let escalation = EscalationState::new(&issue.id)
            .with_initial_tier(initial_tier)
            .with_budget(
                SwarmTier::Council,
                TierBudget {
                    max_iterations: council_budget_iterations,
                    max_consultations: council_budget_consultations,
                },
            );

        // Lazy worktree provisioning (Managed Agents "sandbox as tool" pattern):
        // Compute the planned worktree path but DON'T create it yet.
        // The actual `git worktree add` happens in handle_preparing_worktree().
        // This is safe because tools store wt_path at construction but only
        // validate filesystem access when called (after worktree exists).
        let wt_path = worktree_bridge.worktree_path(&issue.id);
        info!(path = %wt_path.display(), "Planned worktree path (deferred creation)");

        // Session log is also deferred — opened in handle_preparing_worktree()
        // once the worktree directory exists on disk.
        let session_log: Option<crate::session::SessionLog> = None;

        // Intent contract (reformulation engine Phase 1) — capture original goal on first pickup.
        // The contract is append-only: subsequent retries of the same issue reuse the stored one.
        let reformulation_store = ReformulationStore::new(worktree_bridge.repo_root());
        let intent_contract =
            IntentContract::from_issue(&issue.id, &issue.title, issue.description.as_deref());
        reformulation_store.save_contract(&intent_contract);
        info!(
            issue = %issue.id,
            outcomes = intent_contract.required_outcomes.len(),
            digest = %intent_contract.intent_digest,
            "Intent contract captured (state driver)"
        );

        // Harness
        let mut session = SessionManager::new(wt_path.clone(), config.max_retries);
        let git_mgr = GitManager::new(&wt_path, "[swarm]");
        let progress = ProgressTracker::new(wt_path.join(".swarm-progress.txt"));

        if let Ok(commit) = git_mgr.current_commit_full() {
            session.set_initial_commit(commit.clone());
        }
        if let Err(e) = session.start() {
            warn!("Failed to start harness session (state driver): {e}");
        }
        session.set_current_feature(&issue.id);
        let _ = progress.log_session_start(
            session.session_id(),
            format!(
                "Processing issue (state driver): {} — {}",
                issue.id, issue.title
            ),
        );

        // Telemetry
        let stack_profile_str = serde_json::to_string(&config.stack_profile)
            .unwrap_or_else(|_| "unknown".to_string())
            .replace('"', "");
        let metrics = MetricsCollector::new(
            session.session_id(),
            &issue.id,
            &issue.title,
            &stack_profile_str,
            config.repo_id.clone(),
            config.adapter_id.clone(),
            "v1",
        );

        // Verifier config — package detection deferred to handle_preparing_worktree()
        // (needs the worktree to exist for git diff). Use explicit packages if configured,
        // otherwise start with empty (will be populated after worktree creation).
        let verifier_config = VerifierConfig {
            packages: config.verifier_packages.clone(),
            check_clippy: !bool_from_env("SWARM_SKIP_CLIPPY", false),
            check_test: !bool_from_env("SWARM_SKIP_TESTS", false),
            ..VerifierConfig::default()
        };

        // Agents — language is already set on factory by the caller (process_issue).
        // The factory's .lang() method returns the language for prompt adaptation.
        let rust_coder = factory.build_rust_coder(&wt_path);
        let general_coder = factory.build_general_coder(&wt_path);
        let reviewer = factory.build_reviewer();
        let manager = factory.build_manager(&wt_path);

        Ok(OrchestratorContext {
            config,
            issue,
            wt_path,
            verifier_config,
            acceptance_policy,
            feature_flags,
            worker_timeout,
            manager_timeout,
            rust_coder,
            general_coder,
            reviewer,
            manager,
            session,
            git_mgr,
            progress,
            state_machine,
            budget_tracker,
            escalation,
            metrics,
            span_summary: SpanSummary::new(),
            success: false,
            last_report: None,
            last_validator_feedback: Vec::new(),
            consecutive_validator_failures: 0,
            pre_worker_commit: None,
            router: DynamicRouter::new(),
            correction_loop: {
                let mut cl = TieredCorrectionLoop::new();
                cl.start();
                cl
            },
            last_harness_policy: None,
            auto_fix_applied: false,
            last_plan_output: None,
            cot_planner_invocations: 0,
            reformulation_store,
            intent_contract,
            factory,
            beads,
            knowledge_base,
            worktree_bridge,
            cluster_health,
            local_validator_enabled,
            max_validator_failures,
            process_span,
            process_start,
            session_log,
            worktree_provisioned: false,
        })
    }

    /// Resume an orchestrator context from a crashed session (the `wake()` pattern).
    ///
    /// Instead of creating a new worktree and starting from scratch, this:
    /// 1. Verifies the worktree still exists
    /// 2. Reopens the session log and replays events to recover state
    /// 3. Reconstructs the state machine at the recovered position
    /// 4. Rebuilds agents for the existing worktree
    ///
    /// Returns `None` if the session has already completed (nothing to resume).
    pub async fn wake(
        config: &'a SwarmConfig,
        factory: &'a AgentFactory,
        worktree_bridge: &'a WorktreeBridge,
        issue: &'a BeadsIssue,
        beads: &'a dyn IssueTracker,
        knowledge_base: Option<&'a dyn KnowledgeBase>,
        wt_path: PathBuf,
    ) -> Result<Option<OrchestratorContext<'a>>> {
        use crate::session;

        let process_span = otel::process_issue_span(&issue.id);
        let process_start = Instant::now();

        // Verify the worktree still exists.
        if !wt_path.exists() {
            anyhow::bail!(
                "worktree {} no longer exists — cannot resume",
                wt_path.display()
            );
        }

        // Reopen session log and replay events.
        let session_log_path = wt_path.join(session::SESSION_LOG_FILENAME);
        let session_log = session::SessionLog::open(&session_log_path)
            .with_context(|| format!("reopening session log at {}", session_log_path.display()))?;

        let events = session_log.load_all()?;
        let recovered = match session::recover_from_events(&events)? {
            Some(r) => r,
            None => {
                info!(issue = %issue.id, "Session already completed — nothing to resume");
                return Ok(None);
            }
        };

        info!(
            issue = %issue.id,
            state = %recovered.current_state,
            iteration = recovered.iteration,
            last_event = recovered.last_event_id,
            "Waking from crashed session"
        );

        // Emit a resume note to the session log.
        let _ = session_log.append(session::EventKind::Note {
            message: format!(
                "wake(): resumed from state {:?}, iteration {}, event {}",
                recovered.current_state, recovered.iteration, recovered.last_event_id
            ),
        });

        // Reconstruct state machine from recovered transitions.
        let mut state_machine = StateMachine::new();
        // Replay transitions to bring the state machine to the recovered state.
        // We use advance() which validates each transition is legal.
        for t in &recovered.transitions {
            if let Err(e) = state_machine.advance(t.to, t.reason.as_deref()) {
                warn!(
                    from = %t.from, to = %t.to,
                    error = %e,
                    "Illegal transition during replay — skipping"
                );
            }
        }
        state_machine.set_iteration(recovered.iteration);

        // Rebuild all the same infrastructure as new(), but skip worktree creation.
        let worker_policy = TurnPolicy::for_tier(SwarmTier::Worker);
        let council_policy = TurnPolicy::for_tier(SwarmTier::Council);
        let worker_timeout =
            orchestrator::timeout_from_env("SWARM_WORKER_TIMEOUT_SECS", worker_policy.timeout_secs);
        let manager_timeout = orchestrator::timeout_from_env(
            "SWARM_MANAGER_TIMEOUT_SECS",
            council_policy.timeout_secs,
        );

        let feature_flags = FeatureFlags::from_env();
        let acceptance_policy = AcceptancePolicy::default();
        let local_validator_enabled = bool_from_env("SWARM_LOCAL_VALIDATOR", true);
        let max_validator_failures = orchestrator::u32_from_env("SWARM_MAX_VALIDATOR_FAILURES", 3);
        let cluster_health = ClusterHealth::from_config(config);
        let budget_tracker = BudgetTracker::with_defaults();

        let recommendation = classify_initial_tier(&issue.title, &[]);
        let worker_first_tier = recommendation.tier;
        let default_tier = orchestrator::default_initial_tier(
            feature_flags.worker_first_enabled,
            worker_first_tier,
        );
        let initial_tier = orchestrator::tier_from_env("SWARM_INITIAL_TIER", default_tier);

        let council_budget_iterations =
            orchestrator::u32_from_env("SWARM_COUNCIL_MAX_ITERATIONS", 6);
        let council_budget_consultations =
            orchestrator::u32_from_env("SWARM_COUNCIL_MAX_CONSULTATIONS", 6);
        let escalation = EscalationState::new(&issue.id)
            .with_initial_tier(initial_tier)
            .with_budget(
                SwarmTier::Council,
                TierBudget {
                    max_iterations: council_budget_iterations,
                    max_consultations: council_budget_consultations,
                },
            );

        let reformulation_store = ReformulationStore::new(worktree_bridge.repo_root());
        let intent_contract =
            IntentContract::from_issue(&issue.id, &issue.title, issue.description.as_deref());

        let mut session = SessionManager::new(wt_path.clone(), config.max_retries);
        let git_mgr = GitManager::new(&wt_path, "[swarm]");
        let progress = ProgressTracker::new(wt_path.join(".swarm-progress.txt"));

        if let Ok(commit) = git_mgr.current_commit_full() {
            session.set_initial_commit(commit);
        }
        if let Err(e) = session.start() {
            warn!("Failed to start harness session on wake: {e}");
        }
        session.set_current_feature(&issue.id);

        let stack_profile_str = serde_json::to_string(&config.stack_profile)
            .unwrap_or_else(|_| "unknown".to_string())
            .replace('"', "");
        let metrics = MetricsCollector::new(
            session.session_id(),
            &issue.id,
            &issue.title,
            &stack_profile_str,
            config.repo_id.clone(),
            config.adapter_id.clone(),
            "v1",
        );

        let initial_packages = if config.verifier_packages.is_empty() {
            detect_changed_packages(&wt_path, true)
        } else {
            config.verifier_packages.clone()
        };
        let verifier_config = VerifierConfig {
            packages: initial_packages,
            check_clippy: !bool_from_env("SWARM_SKIP_CLIPPY", false),
            check_test: !bool_from_env("SWARM_SKIP_TESTS", false),
            ..VerifierConfig::default()
        };

        let rust_coder = factory.build_rust_coder(&wt_path);
        let general_coder = factory.build_general_coder(&wt_path);
        let reviewer = factory.build_reviewer();
        let manager = factory.build_manager(&wt_path);

        Ok(Some(OrchestratorContext {
            config,
            issue,
            wt_path,
            verifier_config,
            acceptance_policy,
            feature_flags,
            worker_timeout,
            manager_timeout,
            rust_coder,
            general_coder,
            reviewer,
            manager,
            session,
            git_mgr,
            progress,
            state_machine,
            budget_tracker,
            escalation,
            metrics,
            span_summary: SpanSummary::new(),
            success: false,
            last_report: None,
            last_validator_feedback: Vec::new(),
            consecutive_validator_failures: 0,
            pre_worker_commit: None,
            router: DynamicRouter::new(),
            correction_loop: {
                let mut cl = TieredCorrectionLoop::new();
                cl.start();
                cl
            },
            last_harness_policy: None,
            auto_fix_applied: false,
            last_plan_output: None,
            cot_planner_invocations: 0,
            reformulation_store,
            intent_contract,
            factory,
            beads,
            knowledge_base,
            worktree_bridge,
            cluster_health,
            local_validator_enabled,
            max_validator_failures,
            process_span,
            process_start,
            session_log: Some(session_log),
            worktree_provisioned: true, // wake() always has an existing worktree
        }))
    }
}

// ---------------------------------------------------------------------------
// State handlers — each returns Result<StateTransition>
// ---------------------------------------------------------------------------

/// SelectingIssue: validate objective length, claim issue in beads.
pub async fn handle_selecting_issue(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    let title_trimmed = ctx.issue.title.trim();
    if title_trimmed.is_empty() || title_trimmed.len() < ctx.config.min_objective_len {
        warn!(
            id = %ctx.issue.id,
            title_len = title_trimmed.len(),
            min_len = ctx.config.min_objective_len,
            "Rejecting issue: title too short (state driver)",
        );
        return Ok(StateTransition::Fail {
            reason: format!(
                "Issue title too short ({} chars, min {})",
                title_trimmed.len(),
                ctx.config.min_objective_len
            ),
        });
    }

    if let Err(e) = ctx.beads.update_status(&ctx.issue.id, "in_progress") {
        warn!(id = %ctx.issue.id, error = %e, "Failed to claim issue via beads (non-fatal, may be SLURM node without Dolt)");
    } else {
        info!(id = %ctx.issue.id, "Claimed issue (state driver)");
    }

    Ok(StateTransition::Advance {
        to: OrchestratorState::PreparingWorktree,
        reason: "issue validated and claimed".into(),
    })
}

/// PreparingWorktree: health preflight (worktree already created in `new()`).
pub async fn handle_preparing_worktree(
    ctx: &mut OrchestratorContext<'_>,
) -> Result<StateTransition> {
    // Preflight: probe all inference endpoints
    let healthy_count = ctx.cluster_health.check_all_now().await;
    info!(
        healthy = healthy_count,
        total = 3,
        summary = %ctx.cluster_health.summary().await,
        "Preflight endpoint health check (state driver)"
    );
    if healthy_count == 0 {
        warn!(id = %ctx.issue.id, "All inference endpoints DOWN (state driver)");
        let _ = ctx.beads.update_status(&ctx.issue.id, "open");
        return Ok(StateTransition::Fail {
            reason: format!(
                "Preflight failed: all 3 endpoints down ({})",
                ctx.cluster_health.summary().await
            ),
        });
    }

    // ── Lazy worktree provisioning ──
    // Create the worktree now (deferred from OrchestratorContext::new).
    // Failures become clean StateTransition::Fail instead of crashing new().
    if !ctx.worktree_provisioned {
        match ctx.worktree_bridge.create(&ctx.issue.id) {
            Ok(created_path) => {
                // Update wt_path to the actual path (should match planned path).
                ctx.wt_path = created_path;
                ctx.worktree_provisioned = true;
                info!(path = %ctx.wt_path.display(), "Worktree provisioned (lazy)");
            }
            Err(e) => {
                warn!(issue = %ctx.issue.id, error = %e, "Worktree creation failed");
                let _ = ctx.beads.update_status(&ctx.issue.id, "open");
                return Ok(StateTransition::Fail {
                    reason: format!("worktree creation failed: {e}"),
                });
            }
        }

        // Open session log now that the worktree directory exists.
        let session_log_path = ctx.wt_path.join(crate::session::SESSION_LOG_FILENAME);
        match crate::session::SessionLog::open(&session_log_path) {
            Ok(log) => {
                // Emit the deferred startup events.
                let base_commit = ctx.git_mgr.current_commit_full().ok();
                let _ = log.append(crate::session::EventKind::SessionStarted {
                    issue_id: ctx.issue.id.clone(),
                    objective: ctx.issue.title.clone(),
                    base_commit: base_commit.clone(),
                });
                let _ = log.append(crate::session::EventKind::WorktreeProvisioned {
                    path: ctx.wt_path.display().to_string(),
                    branch: format!("swarm/{}", ctx.issue.id),
                    commit: base_commit.unwrap_or_default(),
                });
                ctx.session_log = Some(log);
            }
            Err(e) => {
                warn!(error = %e, "Failed to open session log (non-fatal)");
            }
        }

        // Detect changed packages now that the worktree has files.
        if ctx.verifier_config.packages.is_empty() {
            ctx.verifier_config.packages = detect_changed_packages(&ctx.wt_path, true);
        }

        // Re-initialize git manager with the actual worktree path.
        ctx.git_mgr = GitManager::new(&ctx.wt_path, "[swarm]");
        if let Ok(commit) = ctx.git_mgr.current_commit_full() {
            ctx.session.set_initial_commit(commit);
        }
    }

    // Spawn background health monitor
    let _health_monitor = ctx.cluster_health.spawn_monitor();

    info!(
        session_id = ctx.session.short_id(),
        issue_id = %ctx.issue.id,
        max_iterations = ctx.config.max_retries,
        "Harness session started (state driver)"
    );

    Ok(StateTransition::Advance {
        to: OrchestratorState::Planning,
        reason: "worktree ready, endpoints healthy".into(),
    })
}

/// Planning: pre-iteration auto-fix, context packing, KB enrichment, prompt formatting.
///
/// This handler is called at the start of each iteration (the driver loops
/// Implementing → Verifying → Validating/Escalating → back to Planning).
pub async fn handle_planning(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    // Advance iteration counter
    let iteration = match ctx.session.next_iteration() {
        Ok(i) => i,
        Err(e) => {
            warn!("Session iteration limit (state driver): {e}");
            let _ = ctx.progress.log_error(
                ctx.session.session_id(),
                ctx.session.iteration(),
                format!("Max iterations reached: {e}"),
            );
            return Ok(StateTransition::Fail {
                reason: format!("Max iterations reached: {e}"),
            });
        }
    };

    let tier = ctx.escalation.current_tier;
    ctx.metrics.start_iteration(iteration, &format!("{tier:?}"));
    ctx.span_summary.record_iteration();
    ctx.state_machine.set_iteration(iteration);

    info!(
        iteration,
        ?tier,
        id = %ctx.issue.id,
        session_id = ctx.session.short_id(),
        "Starting iteration (state driver)"
    );

    let _ = ctx.progress.log_feature_start(
        ctx.session.session_id(),
        iteration,
        &ctx.issue.id,
        format!("Iteration {iteration}, tier: {tier:?} (state driver)"),
    );

    // Pre-iteration auto-fix (P1.2) — only on retry iterations
    ctx.auto_fix_applied = false;
    if iteration > 1 {
        if let Some(ref report) = ctx.last_report {
            if !report.all_green {
                if let Some(fixed_report) =
                    try_auto_fix(&ctx.wt_path, &ctx.verifier_config, iteration, &None).await
                {
                    if fixed_report.all_green {
                        info!(
                            iteration,
                            "Pre-iteration auto-fix resolved all issues (state driver)"
                        );
                        let _ = git_commit_changes(&ctx.wt_path, iteration).await;
                        ctx.metrics.record_auto_fix();
                        ctx.metrics.finish_iteration();
                        // success is set after merge confirms in drive()
                        return Ok(StateTransition::Complete);
                    }
                    ctx.last_report = Some(fixed_report);
                    ctx.metrics.record_auto_fix();
                    ctx.auto_fix_applied = true;
                    let _ = git_commit_changes(&ctx.wt_path, iteration).await;
                }
            }
        }
    }

    Ok(StateTransition::Advance {
        to: OrchestratorState::Implementing,
        reason: format!("iteration {iteration} planned, tier={tier:?}"),
    })
}

/// Implementing: checkpoint, route to agent, cloud fallback, commit changes.
pub async fn handle_implementing(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    let iteration = ctx.session.iteration();
    let tier = ctx.escalation.current_tier;

    // Pack context
    let packer = ContextPacker::new(&ctx.wt_path, tier);
    let mut packet = if let Some(ref report) = ctx.last_report {
        packer.pack_retry(&ctx.issue.id, &ctx.issue.title, &ctx.escalation, report)
    } else {
        packer.pack_initial(&ctx.issue.id, &ctx.issue.title)
    };

    // Inject validator feedback (TextGrad pattern)
    if !ctx.last_validator_feedback.is_empty() {
        packet.validator_feedback = std::mem::take(&mut ctx.last_validator_feedback);
        info!(
            iteration,
            feedback_count = packet.validator_feedback.len(),
            "Injected validator feedback (state driver)"
        );
    }

    // KB enrichment
    if let Some(kb) = ctx.knowledge_base {
        let brain_question = format!(
            "What architectural context is relevant for: {}? Issue: {}",
            ctx.issue.title, ctx.issue.id
        );
        let response = query_kb_with_failsafe(kb, "project_brain", &brain_question);
        if !response.is_empty() {
            packet.relevant_heuristics.push(response);
        }

        if ctx.last_report.is_some() && !packet.failure_signals.is_empty() {
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
            }
        }
    }

    // --- Part B: Mutation archive strategy seeding ---
    //
    // On the first iteration (no last_report), query the archive for similar past
    // issues and inject the strategies that worked as a heuristic hint.  Uses
    // map_elites::sample_diverse() so we suggest a variety of approaches rather
    // than always repeating the most-recently-successful pattern.
    if ctx.last_report.is_none() {
        let archive =
            crate::mutation_archive::MutationArchive::new(ctx.worktree_bridge.repo_root());
        let keyword_matches = archive.query_by_keywords(&ctx.issue.title, 10);
        if !keyword_matches.is_empty() {
            // Build a temporary QD archive from the keyword matches so we can
            // sample diverse strategies (rather than just the top match).
            let mut qd = crate::map_elites::QualityDiversityArchive::new();
            for r in &keyword_matches {
                let lines = (r.lines_added + r.lines_removed) as usize;
                let files = r.files_changed.len();
                let features =
                    crate::map_elites::FeatureExtractor::extract_from_metadata(lines, files);
                let score = if r.resolved {
                    1.0 / r.iterations.max(1) as f64
                } else {
                    0.0
                };
                let node = crate::map_elites::ExperimentNode {
                    id: r.issue_id.clone(),
                    description: r.issue_title.clone(),
                    score,
                    lines_changed: lines,
                    files_changed: files,
                    strategy: crate::map_elites::FeatureExtractor::infer_strategy(files, lines),
                };
                qd.insert(node, features);
            }

            // Sample up to 3 diverse experiments from different bins.
            let diverse = qd.sample_diverse(3);
            if !diverse.is_empty() {
                let mut hint = String::from(
                    "## Strategy Hints from Past Similar Issues\n\n\
                     The mutation archive found similar past issues. \
                     Consider these strategies (do not blindly copy — adapt as needed):\n\n",
                );
                for node in &diverse {
                    // Find the original record for model/tier info.
                    if let Some(rec) = keyword_matches.iter().find(|r| r.issue_id == node.id) {
                        hint.push_str(&format!(
                            "- **{}** (resolved={}, {} iter, model=`{}`, strategy={:?})\n",
                            rec.issue_title, rec.resolved, rec.iterations, rec.model, node.strategy,
                        ));
                    }
                }
                hint.push('\n');
                packet.relevant_heuristics.push(hint);
                info!(
                    matches = keyword_matches.len(),
                    diverse = diverse.len(),
                    "Mutation archive: seeded {} diverse strategy hints (state driver)",
                    diverse.len()
                );
            }
        }
    }

    info!(
        tokens = packet.estimated_tokens(),
        files = packet.file_contexts.len(),
        "Packed context (state driver)"
    );

    // Sparse context guard — escalate Worker→Council when cloud is available
    let tier = if tier == SwarmTier::Worker
        && packet.file_contexts.is_empty()
        && packet.files_touched.is_empty()
        && packet.failure_signals.is_empty()
        && ctx.config.cloud_endpoint.is_some()
    {
        warn!(
            iteration,
            "Sparse context — escalating Worker→Council (state driver)"
        );
        ctx.escalation.record_escalation(
            SwarmTier::Council,
            EscalationReason::Explicit {
                reason: "sparse context: no file_contexts/files_touched/failure_signals".into(),
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
                "Sparse context but no cloud — keeping Worker (state driver)"
            );
        }
        tier
    };

    // --- P2.2: DynamicRouter integration ---
    // Ask the router for a tier recommendation based on error patterns + performance
    // history. If the router recommends Council and we're currently Worker (and cloud is
    // available), upgrade the tier for this iteration.
    let tier = if tier == SwarmTier::Worker && ctx.last_report.is_some() {
        // Convert ErrorCategories to ParsedErrors for router input
        let parsed_errors: Vec<coordination::feedback::error_parser::ParsedError> = ctx
            .last_report
            .as_ref()
            .map(|r| {
                r.unique_error_categories()
                    .into_iter()
                    .map(|cat| coordination::feedback::error_parser::ParsedError {
                        category: cat,
                        code: None,
                        message: String::new(),
                        file: None,
                        line: None,
                        column: None,
                        suggestion: None,
                        rendered: String::new(),
                        labels: Vec::new(),
                    })
                    .collect()
            })
            .unwrap_or_default();

        if !parsed_errors.is_empty() {
            let selection = ctx.router.select_for_errors_dynamic(&parsed_errors);
            if selection.tier == ModelTier::Council && ctx.config.cloud_endpoint.is_some() {
                info!(
                    iteration,
                    reason = %selection.reason,
                    "DynamicRouter recommends Council (P2.2)"
                );
                ctx.escalation.record_escalation(
                    SwarmTier::Council,
                    EscalationReason::Explicit {
                        reason: format!("DynamicRouter: {}", selection.reason),
                    },
                );
                SwarmTier::Council
            } else {
                debug!(
                    iteration,
                    tier = %selection.tier,
                    reason = %selection.reason,
                    "DynamicRouter stays at Worker"
                );
                tier
            }
        } else {
            tier
        }
    } else {
        tier
    };

    // Derive harness policy for prompt-building (may be re-derived below if the deep
    // probe escalates the tier — see "Re-derive harness policy" block after deep probe).
    let harness_policy = HarnessPolicy::derive(&packet, tier);
    harness_policy.apply_to_packet(&mut packet);
    ctx.metrics.record_harness_policy(&harness_policy);
    if harness_policy.economics.prefer_compact_prompt {
        ctx.metrics.record_compact_prompt();
    }
    if harness_policy.judgment.external_review_required {
        ctx.metrics.record_strict_external_judgment();
    }
    ctx.last_harness_policy = Some(harness_policy.clone());

    // Build prompt
    let mut task_prompt = if harness_policy.economics.prefer_compact_prompt {
        format_compact_task_prompt(&packet, &ctx.wt_path)
    } else {
        format_task_prompt(&packet)
    };

    // Prune accumulated context after prune_after_iteration iterations.
    // The state driver path was missing this — unlike orchestrator/mod.rs which calls
    // telemetry::prune_task_prompt() explicitly, driver.rs built task_prompt but never
    // pruned it. By iteration 4+ the prompt hits 135K tokens (4× the 32K context of
    // local models), causing 502 "exceed_context_size_error" from every endpoint.
    task_prompt = crate::telemetry::prune_task_prompt(
        &task_prompt,
        iteration,
        ctx.config.prune_after_iteration,
        2, // keep last 2 iteration sections (matches orchestrator/mod.rs)
    );

    // Inject verifier stderr when failure_signals are thin
    if let Some(ref report) = ctx.last_report {
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

    // Inject CoT planner output from a previous iteration (if any).
    // The planner has no tools and can't write files, but its analysis
    // provides valuable context for the worker that follows.
    if let Some(plan) = ctx.last_plan_output.take() {
        task_prompt.push_str("\n## Plan from Previous Analysis\n\n");
        let plan_budget = plan.len().min(2000);
        let plan_slice = &plan[..plan.floor_char_boundary(plan_budget)];
        task_prompt.push_str(plan_slice);
        if plan.len() > plan_budget {
            task_prompt.push_str("\n...(plan truncated)\n");
        }
        task_prompt.push('\n');
        info!(
            iteration,
            plan_chars = plan_budget,
            "Injected CoT plan into worker prompt"
        );
    }

    // High-churn issues (stringer-generated: "High churn: <file> (modified N times in M days)")
    // get a targeted prompt injection. Without it, agents correctly identify the churn as a
    // process issue but conclude "nothing to change" and hit the no-change circuit breaker.
    // The hint makes the refactoring intent explicit so the coder actually edits the file.
    if ctx.issue.title.starts_with("High churn:") {
        task_prompt.push_str(
            "\n\n## High-Churn Refactoring Task\n\
             This file has unusually high git churn — it is modified very frequently, which \
             suggests it is doing too much or has unclear boundaries. Your task is to make \
             at least one concrete code improvement that will reduce future churn:\n\
             - Extract a well-named helper function for a repeated or complex block\n\
             - Move a cohesive group of functions into a sub-module or separate file\n\
             - Replace an ad-hoc inline pattern with a named constant or struct\n\
             - Add a doc comment that clarifies the invariant or contract\n\
             **You MUST make at least one file edit.** Analysis without an edit will be \
             rejected. Choose the smallest change that clearly improves structure.\n",
        );
        info!(
            iteration,
            issue_id = %ctx.issue.id,
            "Injected high-churn refactoring hint into task prompt"
        );
    }

    // Checkpoint before agent invocation
    ctx.pre_worker_commit = ctx.git_mgr.current_commit_full().ok();

    // --- Multi-candidate fan-out (ARXIV:2510.21513) ---
    //
    // When SWARM_CANDIDATE_COUNT > 1 and we're in Worker tier, generate N
    // candidates concurrently (model + strategy diversity) and pick the first
    // that produces file changes.  This captures ~95% of ensemble benefit at
    // a fraction of the sequential cost.
    //
    // Only activates for the Worker tier — Council/Strategist run once and
    // produce authoritative output, so candidate redundancy isn't beneficial.
    if harness_policy.economics.allow_candidate_fanout
        && ctx.config.candidate_count > 1
        && tier == SwarmTier::Worker
    {
        let base_commit = ctx
            .pre_worker_commit
            .clone()
            .unwrap_or_else(|| "HEAD".to_string());
        let timeout = orchestrator::timeout_from_env("SWARM_SUBTASK_TIMEOUT_SECS", 3600).as_secs();
        let complexity = ctx
            .last_report
            .as_ref()
            .map(|_| crate::triage::Complexity::Medium)
            .unwrap_or(crate::triage::Complexity::Simple);

        let cfg = crate::subtask::CandidateGenerationConfig {
            candidate_count: ctx.config.candidate_count,
            model_diversity: true,
            strategy_diversity: true,
        };

        info!(
            iteration,
            candidate_count = ctx.config.candidate_count,
            "Running multi-candidate fan-out (state driver)"
        );

        let candidates = crate::subtask::generate_candidates(
            &cfg,
            &ctx.factory.endpoint_pool,
            &ctx.wt_path,
            &ctx.issue.id,
            &task_prompt,
            &base_commit,
            timeout,
            complexity,
            &ctx.issue.title,
        )
        .await;

        // Pick the first candidate that produced changes; fall through to
        // the single-agent path if no candidate wrote anything.
        let winner = candidates.into_iter().find(|c| c.has_changes);
        if let Some(w) = winner {
            info!(
                iteration,
                winner = w.index,
                elapsed_ms = w.elapsed.as_millis() as u64,
                "Candidate {} selected — proceeding to verification",
                w.index
            );
            ctx.metrics.record_agent_time(w.elapsed);
            ctx.span_summary.record_agent(0);
            ctx.escalation.reset_no_change();

            // Auto-format before commit (same as single-agent path).
            let mut fmt_args = vec!["fmt".to_string()];
            if ctx.verifier_config.packages.is_empty() {
                fmt_args.push("--all".to_string());
            } else {
                for pkg in &ctx.verifier_config.packages {
                    fmt_args.extend(["--package".to_string(), pkg.clone()]);
                }
            }
            let _ = tokio::process::Command::new("cargo")
                .args(&fmt_args)
                .current_dir(&ctx.wt_path)
                .output()
                .await;

            let _ = git_commit_changes(&ctx.wt_path, iteration).await;
            return Ok(StateTransition::Advance {
                to: OrchestratorState::Verifying,
                reason: format!(
                    "candidate {} produced changes, ready for verification",
                    w.index
                ),
            });
        }

        // All candidates produced no changes — fall through to single-agent path.
        warn!(
            iteration,
            "All {} candidates produced no changes — falling back to single-agent",
            ctx.config.candidate_count
        );
    }

    // Pre-flight deep probe: verify the target model can actually generate
    // (not just /health OK with hung slots). Prevents routing to hung models.
    // Design: docs/research/self-improving-swarm-architecture.md Layer 2.
    let tier = if tier == SwarmTier::Worker {
        // Probe the coder endpoint (primary worker target)
        if !ctx.cluster_health.deep_probe_tier("coder").await {
            warn!(
                iteration,
                "Deep probe failed for coder tier — escalating Worker→Council (state driver)"
            );
            // Escalate to Council tier — the cloud manager can still function
            // even when local workers are hung, because it delegates via proxy
            // tools which have their own timeouts.
            if ctx.config.cloud_endpoint.is_some() {
                info!(
                    iteration,
                    "Deep probe escalation: using Council tier instead of Worker"
                );
                SwarmTier::Council
            } else {
                warn!(
                    iteration,
                    "Deep probe: no cloud endpoint for escalation, proceeding with Worker"
                );
                tier
            }
        } else {
            tier
        }
    } else {
        tier
    };

    // Re-derive harness policy now that tier is fully finalized (the deep-probe
    // above may have escalated Worker→Council after the initial derivation).
    // Only update tracked state here — the packet and prompt are already built
    // and re-applying to the packet would inject duplicate prompt sections.
    let harness_policy = HarnessPolicy::derive(&packet, tier);
    ctx.metrics.record_harness_policy(&harness_policy);
    if harness_policy.economics.prefer_compact_prompt {
        ctx.metrics.record_compact_prompt();
    }
    if harness_policy.judgment.external_review_required {
        ctx.metrics.record_strict_external_judgment();
    }
    ctx.last_harness_policy = Some(harness_policy.clone());

    // Route to agent
    let agent_start = Instant::now();
    let (agent_future, adapter) = match tier {
        SwarmTier::Worker => {
            let recent_cats: Vec<ErrorCategory> = ctx
                .escalation
                .recent_error_categories
                .last()
                .cloned()
                .unwrap_or_default();
            // Complexity-gated CoT planner: check if this task should use
            // Devstral-24B in pure-reasoning mode (no tools) before falling
            // through to the standard error-driven routing.
            let decomposition_required = ctx
                .reformulation_store
                .last_classification(&ctx.issue.id)
                .map(|c| {
                    matches!(
                        c,
                        crate::reformulation::FailureClassification::DecompositionRequired { .. }
                    )
                })
                .unwrap_or(false);
            let changed_file_count =
                crate::orchestrator::helpers::list_changed_files(&ctx.wt_path).len();
            let cot_route = needs_cot_planner(
                decomposition_required,
                changed_file_count,
                &ctx.issue.title,
                ctx.cot_planner_invocations,
            );

            match if cot_route {
                CoderRoute::CoTPlanner
            } else {
                route_to_coder(&recent_cats, iteration)
            } {
                CoderRoute::RustCoder => {
                    info!(iteration, "Routing to rust_coder (state driver)");
                    ctx.metrics.record_coder_route("RustCoder");
                    ctx.metrics.record_agent_metrics("Qwen3.5-RustCoder", 0, 0);
                    let adapter = RuntimeAdapter::new(AdapterConfig {
                        agent_name: "Qwen3.5-RustCoder".into(),
                        deadline: Some(Instant::now() + ctx.worker_timeout),
                        max_tool_calls: Some(30),
                        max_turns_without_write: Some(ctx.config.max_turns_without_write),
                        search_unlock_turn: Some(1),
                        ..Default::default()
                    });
                    let result = match tokio::time::timeout(
                        ctx.worker_timeout,
                        crate::orchestrator::prompt_with_hook_and_retry(
                            &ctx.rust_coder,
                            &task_prompt,
                            2,
                            adapter.clone(),
                        ),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            warn!(iteration, "rust_coder timed out (state driver)");
                            Ok("rust_coder timed out. Changes on disk.".into())
                        }
                    };
                    (result, adapter)
                }
                CoderRoute::GeneralCoder => {
                    info!(iteration, "Routing to general_coder (state driver)");
                    ctx.metrics.record_coder_route("GeneralCoder");
                    ctx.metrics
                        .record_agent_metrics("Qwen3.5-GeneralCoder", 0, 0);
                    let adapter = RuntimeAdapter::new(AdapterConfig {
                        agent_name: "Qwen3.5-GeneralCoder".into(),
                        deadline: Some(Instant::now() + ctx.worker_timeout),
                        max_tool_calls: Some(30),
                        max_turns_without_write: Some(ctx.config.max_turns_without_write),
                        search_unlock_turn: Some(1),
                        ..Default::default()
                    });
                    let result = match tokio::time::timeout(
                        ctx.worker_timeout,
                        crate::orchestrator::prompt_with_hook_and_retry(
                            &ctx.general_coder,
                            &task_prompt,
                            2,
                            adapter.clone(),
                        ),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            warn!(iteration, "general_coder timed out (state driver)");
                            Ok("general_coder timed out. Changes on disk.".into())
                        }
                    };
                    (result, adapter)
                }
                CoderRoute::FastFixer => {
                    info!(
                        iteration,
                        "Reasoning sandwich: routing to fast_fixer (state driver)"
                    );
                    ctx.metrics.record_coder_route("FastFixer");
                    ctx.metrics.record_agent_metrics("GLM-FastFixer", 0, 0);
                    let fixer = ctx.factory.build_fixer(&ctx.wt_path);
                    let adapter = RuntimeAdapter::new(AdapterConfig {
                        agent_name: "GLM-FastFixer".into(),
                        deadline: Some(Instant::now() + ctx.worker_timeout),
                        max_tool_calls: Some(30),
                        max_turns_without_write: Some(ctx.config.max_turns_without_write),
                        search_unlock_turn: Some(1),
                        ..Default::default()
                    });
                    let result = match tokio::time::timeout(
                        ctx.worker_timeout,
                        crate::orchestrator::prompt_with_hook_and_retry(
                            &fixer,
                            &task_prompt,
                            2,
                            adapter.clone(),
                        ),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            warn!(iteration, "fast_fixer timed out (state driver)");
                            Ok("fast_fixer timed out. Changes on disk.".into())
                        }
                    };
                    (result, adapter)
                }
                CoderRoute::CoTPlanner => {
                    ctx.cot_planner_invocations += 1;
                    info!(
                        iteration,
                        invocation = ctx.cot_planner_invocations,
                        "Routing to CoT planner (Devstral-24B, no tools — state driver)"
                    );
                    ctx.metrics.record_coder_route("CoTPlanner");
                    ctx.metrics
                        .record_agent_metrics("Devstral-CoTPlanner", 0, 0);
                    let cot_planner = ctx.factory.build_cot_planner(&ctx.wt_path);
                    let adapter = RuntimeAdapter::new(AdapterConfig {
                        agent_name: "Devstral-CoTPlanner".into(),
                        deadline: Some(Instant::now() + ctx.worker_timeout),
                        ..Default::default()
                    });
                    let result = match tokio::time::timeout(
                        ctx.worker_timeout,
                        crate::orchestrator::prompt_with_hook_and_retry(
                            &cot_planner,
                            &task_prompt,
                            2,
                            adapter.clone(),
                        ),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            warn!(iteration, "cot_planner timed out (state driver)");
                            Ok("cot_planner timed out. No plan produced.".into())
                        }
                    };
                    (result, adapter)
                }
                CoderRoute::ExplorerCoder => {
                    // Two-phase pipeline: Explorer → GeneralCoder.
                    // Phase 1: Explorer reads files and git history, returns specific
                    // coder instructions. No write deadline — exploration takes time.
                    // Phase 2: GeneralCoder receives the enriched prompt and writes
                    // on its first or second turn.
                    info!(
                        iteration,
                        "Routing to ExplorerCoder pipeline (state driver)"
                    );
                    ctx.metrics.record_coder_route("ExplorerCoder");

                    // ── Phase 1: Explorer ────────────────────────────────────────
                    // Pre-flight deep probe: verify the reasoning endpoint can
                    // actually generate before acquiring the semaphore gate.
                    // vasp-02 occasionally returns empty completions even without
                    // concurrent load (independent of the inter-issue contention
                    // that EXPLORER_GATE prevents). A 10-second 1-token probe is
                    // far cheaper than waiting for a 25-turn Explorer to time out.
                    //
                    // If the probe passes: acquire the global gate and run Explorer.
                    // If the probe fails: skip Explorer immediately, fall through
                    // to GeneralCoder — same as the existing timeout/error fallback,
                    // but without burning 25-40s on transient retries.
                    let exploration = if ctx.cluster_health.deep_probe_tier("reasoning").await {
                        // Acquire the global gate to prevent concurrent inference
                        // on the same reasoning endpoint. The permit is dropped
                        // explicitly before Phase 2 so GeneralCoder phases remain
                        // fully concurrent.
                        let explorer_permit = EXPLORER_GATE
                            .get_or_init(|| Arc::new(tokio::sync::Semaphore::new(1)))
                            .acquire()
                            .await
                            .expect("EXPLORER_GATE semaphore was closed");

                        ctx.metrics.record_agent_metrics("Devstral-Explorer", 0, 0);
                        let explorer = ctx.factory.build_explorer(&ctx.wt_path);
                        let explore_budget = ctx.worker_timeout / 2;
                        let explore_adapter = RuntimeAdapter::new(AdapterConfig {
                            agent_name: "Devstral-Explorer".into(),
                            deadline: Some(Instant::now() + explore_budget),
                            max_tool_calls: Some(25),
                            // No max_turns_without_write: explorer is read-only
                            ..Default::default()
                        });
                        let result = match tokio::time::timeout(
                            explore_budget,
                            crate::orchestrator::prompt_with_hook_and_retry(
                                &explorer,
                                &task_prompt,
                                1,
                                explore_adapter,
                            ),
                        )
                        .await
                        {
                            Ok(Ok(analysis)) if !analysis.trim().is_empty() => {
                                info!(
                                    iteration,
                                    analysis_len = analysis.len(),
                                    "Explorer produced analysis (state driver)"
                                );
                                Some(analysis)
                            }
                            Ok(Ok(_)) => {
                                warn!(
                                    iteration,
                                    "Explorer returned empty analysis — proceeding without"
                                );
                                None
                            }
                            Ok(Err(e)) => {
                                warn!(iteration, error = %e, "Explorer failed — proceeding without analysis");
                                None
                            }
                            Err(_) => {
                                warn!(
                                    iteration,
                                    "Explorer timed out — proceeding without analysis"
                                );
                                None
                            }
                        };

                        // Release the gate now — Phase 2 (GeneralCoder) runs on
                        // a different endpoint (vasp-01) and does not need exclusion.
                        drop(explorer_permit);
                        result
                    } else {
                        warn!(
                            iteration,
                            "Explorer pre-flight probe failed (reasoning endpoint unresponsive) \
                                 — skipping Explorer phase"
                        );
                        None
                    };

                    // ── Phase 2: GeneralCoder with enriched prompt ───────────────
                    ctx.metrics
                        .record_agent_metrics("Qwen3.5-GeneralCoder", 0, 0);
                    let enriched_prompt = match &exploration {
                        Some(analysis) => format!(
                            "{task_prompt}\n\n\
                             ## Explorer Analysis\n\
                             The Explorer has already read the relevant files. \
                             Follow the CODER INSTRUCTIONS below — do NOT re-read \
                             files already described. Write your first edit \
                             on your first or second tool call.\n\n\
                             {analysis}"
                        ),
                        None => task_prompt.clone(),
                    };
                    let adapter = RuntimeAdapter::new(AdapterConfig {
                        agent_name: "Qwen3.5-GeneralCoder".into(),
                        deadline: Some(Instant::now() + ctx.worker_timeout),
                        max_tool_calls: Some(30),
                        max_turns_without_write: Some(ctx.config.max_turns_without_write),
                        search_unlock_turn: Some(1),
                        ..Default::default()
                    });
                    let result = match tokio::time::timeout(
                        ctx.worker_timeout,
                        crate::orchestrator::prompt_with_hook_and_retry(
                            &ctx.general_coder,
                            &enriched_prompt,
                            2,
                            adapter.clone(),
                        ),
                    )
                    .await
                    {
                        Ok(result) => result,
                        Err(_) => {
                            warn!(
                                iteration,
                                "general_coder timed out in ExplorerCoder pipeline (state driver)"
                            );
                            Ok("general_coder timed out. Changes on disk.".into())
                        }
                    };
                    (result, adapter)
                }
            }
        }
        SwarmTier::Strategist => {
            info!(iteration, "Routing to strategist advisor");
            let model = ctx.config.resolve_role_model(SwarmRole::Strategist);
            ctx.metrics
                .record_agent_metrics(&format!("strategist ({model})"), 0, 0);

            let strategist = ctx.factory.build_strategist(&ctx.wt_path);
            let adapter = RuntimeAdapter::new(AdapterConfig {
                agent_name: "strategist".into(),
                deadline: Some(Instant::now() + ctx.worker_timeout),
                max_reads_without_action: Some(8),
                ..Default::default()
            });

            let result = match tokio::time::timeout(
                ctx.worker_timeout,
                crate::orchestrator::prompt_with_hook_and_retry(
                    &strategist,
                    &task_prompt,
                    2,
                    adapter.clone(),
                ),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => {
                    warn!(iteration, "strategist timed out");
                    Ok("strategist timed out. No guidance produced.".into())
                }
            };
            (result, adapter)
        }
        SwarmTier::Council | SwarmTier::Human => {
            info!(iteration, "Routing to manager");
            let role = if ctx.config.cloud_endpoint.is_some() {
                SwarmRole::Council
            } else {
                SwarmRole::LocalManagerFallback
            };
            let model = ctx.config.resolve_role_model(role);
            ctx.metrics
                .record_agent_metrics(&format!("manager ({model})"), 0, 0);
            let adapter = RuntimeAdapter::with_validators(
                AdapterConfig {
                    agent_name: "manager".into(),
                    deadline: Some(Instant::now() + ctx.manager_timeout),
                    max_reads_without_action: Some(8),
                    ..Default::default()
                },
                crate::action_validator::manager_validators(),
            );
            let result = match tokio::time::timeout(
                ctx.manager_timeout,
                prompt_with_hook_and_retry(
                    &ctx.manager,
                    &task_prompt,
                    ctx.config.cloud_max_retries,
                    adapter.clone(),
                ),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => {
                    warn!(iteration, "Manager timed out (state driver)");
                    Ok("Manager timed out. Changes on disk.".into())
                }
            };

            // Cloud fallback matrix (P1.3)
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
                    let fallbacks = ctx.config.cloud_fallback_matrix.fallbacks();
                    let mut fallback_result = None;
                    for entry in fallbacks {
                        warn!(
                            iteration,
                            model = %entry.model,
                            tier = %entry.tier_label,
                            "Trying cloud fallback (state driver)"
                        );
                        if let Some(fallback_manager) = ctx
                            .factory
                            .build_manager_for_model(&ctx.wt_path, &entry.model)
                        {
                            let fb_adapter = RuntimeAdapter::new(AdapterConfig {
                                agent_name: format!("manager-{}", entry.tier_label),
                                deadline: Some(Instant::now() + ctx.manager_timeout),
                                max_reads_without_action: Some(8),
                                ..Default::default()
                            });
                            match tokio::time::timeout(
                                ctx.manager_timeout,
                                prompt_with_hook_and_retry(
                                    &fallback_manager,
                                    &task_prompt,
                                    1,
                                    fb_adapter,
                                ),
                            )
                            .await
                            {
                                Ok(Ok(response)) if !response.trim().is_empty() => {
                                    info!(iteration, model = %entry.model, "Fallback succeeded (state driver)");
                                    fallback_result = Some(Ok(response));
                                    break;
                                }
                                Ok(Ok(_)) => {
                                    // Null/empty content — proxy returned HTTP 200 but no text.
                                    // Happens when a model's OAuth token is stale or the proxy
                                    // has a format compatibility issue. Skip to next fallback.
                                    warn!(iteration, model = %entry.model, "Fallback returned empty content, skipping (state driver)");
                                }
                                Ok(Err(e)) => {
                                    warn!(iteration, model = %entry.model, error = %e, "Fallback failed (state driver)");
                                }
                                Err(_) => {
                                    warn!(iteration, model = %entry.model, "Fallback timed out (state driver)");
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

    // Log runtime adapter report
    let agent_terminated_without_writing = match adapter.report() {
        Ok(adapter_report) => {
            info!(
                iteration,
                agent = %adapter_report.agent_name,
                turns = adapter_report.turn_count,
                tool_calls = adapter_report.total_tool_calls,
                terminated_early = adapter_report.terminated_early,
                has_written = adapter_report.has_written,
                "Runtime adapter report (state driver)"
            );
            if let Some(ref reason) = adapter_report.termination_reason {
                warn!(iteration, reason = %reason, "Agent terminated early (state driver)");
            }
            adapter_report.terminated_early && !adapter_report.has_written
        }
        Err(e) => {
            warn!(iteration, error = %e, "Failed to extract adapter report (state driver)");
            false
        }
    };

    // Handle agent failure
    let agent_elapsed = agent_start.elapsed();
    ctx.metrics.record_agent_time(agent_elapsed);
    ctx.span_summary.record_agent(0);
    let response = match agent_future {
        Ok(r) if !r.trim().is_empty() => {
            let preview = &r[..r.floor_char_boundary(500)];
            info!(iteration, response_len = r.len(), response_preview = %preview, "Agent responded (state driver)");
            r
        }
        Ok(_) => {
            // Primary manager returned empty/null content — treat as failure so
            // the fallback matrix can be tried on the next iteration.
            error!(
                iteration,
                "Agent returned empty response (state driver) — treating as failure"
            );
            let _ = ctx.progress.log_error(
                ctx.session.session_id(),
                iteration,
                "Manager returned empty response (proxy null-content bug)".to_string(),
            );
            return Err(anyhow::anyhow!("Manager returned empty response"));
        }
        Err(e) => {
            error!(iteration, "Agent failed (state driver): {e}");
            let _ = ctx.progress.log_error(
                ctx.session.session_id(),
                iteration,
                format!("Agent failed: {e}"),
            );
            // Run verifier to assess state after failure
            let engine = EscalationEngine::new();
            let current_vc = if ctx.config.verifier_packages.is_empty() {
                VerifierConfig {
                    packages: detect_changed_packages(&ctx.wt_path, true),
                    ..ctx.verifier_config.clone()
                }
            } else {
                ctx.verifier_config.clone()
            };
            let verifier = Verifier::new(&ctx.wt_path, current_vc);
            let report = verifier.run_pipeline().await;
            let decision = engine.decide(&mut ctx.escalation, &report);
            ctx.last_report = Some(report);
            ctx.metrics.finish_iteration();

            if decision.stuck {
                error!(iteration, "Stuck after agent failure (state driver)");
                create_stuck_intervention(
                    &mut ctx.session,
                    &ctx.progress,
                    &ctx.wt_path,
                    iteration,
                    &decision.reason,
                );
                return Ok(StateTransition::Fail {
                    reason: format!("Agent failed and stuck: {}", decision.reason),
                });
            }
            // Not stuck — loop back to Planning for next iteration
            return Ok(StateTransition::Advance {
                to: OrchestratorState::Planning,
                reason: format!("agent failed, escalation continues: {}", decision.reason),
            });
        }
    };

    // Auto-format before commit
    let mut fmt_args = vec!["fmt".to_string()];
    if ctx.verifier_config.packages.is_empty() {
        fmt_args.push("--all".to_string());
    } else {
        for pkg in &ctx.verifier_config.packages {
            fmt_args.extend(["--package".to_string(), pkg.clone()]);
        }
    }
    let _ = tokio::process::Command::new("cargo")
        .args(&fmt_args)
        .current_dir(&ctx.wt_path)
        .output()
        .await;

    // Commit agent changes
    let has_changes = match git_commit_changes(&ctx.wt_path, iteration).await {
        Ok(changed) => changed,
        Err(e) => {
            error!(iteration, "git commit failed (state driver): {e}");
            return Err(e);
        }
    };

    let post_agent_commit = ctx.git_mgr.current_commit_full().ok();

    // Record artifacts
    if let (Some(ref pre), Some(ref post)) = (&ctx.pre_worker_commit, &post_agent_commit) {
        if pre != post {
            let artifacts = collect_artifacts_from_diff(&ctx.wt_path, pre, post);
            for artifact in artifacts {
                ctx.metrics.record_artifact(artifact);
            }
        }
    }

    if !has_changes {
        // If the agent produced a substantial response without file changes,
        // treat it as a plan/analysis and inject it into the next iteration.
        // This enables the CoT planner (no tools) to contribute context
        // instead of being penalized by the no-change circuit breaker.
        //
        // Guard: only store as plan if this is the FIRST plan-only response.
        // If we already have a plan stored, the previous plan didn't help —
        // don't keep accumulating plans, let the no-change circuit breaker fire.
        let is_hallucinated = response.contains(".swarm-checkpoint")
            || response.contains(".swarm-session")
            || response.contains("tz_insights_join");
        let already_has_plan = ctx.last_plan_output.is_some();
        if response.len() > 100 && !already_has_plan && !is_hallucinated {
            info!(
                iteration,
                response_len = response.len(),
                "Storing agent response as plan context for next iteration"
            );
            ctx.last_plan_output = Some(response.clone());
            // Skip the no-change escalation — the plan IS the output.
            return Ok(StateTransition::Advance {
                to: OrchestratorState::Planning,
                reason: format!(
                    "agent produced plan/analysis ({} chars) — re-entering with plan context",
                    ctx.last_plan_output.as_ref().map(|p| p.len()).unwrap_or(0)
                ),
            });
        }
        if is_hallucinated {
            warn!(
                iteration,
                response_len = response.len(),
                "Discarding hallucinated plan output (references swarm internals)"
            );
        }

        ctx.escalation.record_no_change();
        ctx.metrics.record_no_change();
        warn!(
            iteration,
            response_len = response.len(),
            consecutive_no_change = ctx.escalation.consecutive_no_change,
            agent_terminated_without_writing,
            "No file changes (state driver)"
        );

        // Write-deadline escalation: immediate Cloud escalation
        if agent_terminated_without_writing
            && ctx.escalation.current_tier == SwarmTier::Worker
            && ctx.config.cloud_endpoint.is_some()
        {
            warn!(
                iteration,
                "Write-deadline escalation (state driver): worker terminated without writing — \
                 escalating to Council immediately"
            );
            ctx.escalation.record_escalation(
                SwarmTier::Council,
                EscalationReason::Explicit {
                    reason: "write deadline: worker exhausted turns without edit_file/write_file"
                        .to_string(),
                },
            );
            ctx.metrics.finish_iteration();
            return Ok(StateTransition::Advance {
                to: OrchestratorState::Planning,
                reason: "write-deadline escalation → Council".into(),
            });
        }

        // No-change circuit breaker
        if ctx.escalation.consecutive_no_change >= ctx.config.max_consecutive_no_change {
            error!(iteration, "No-change circuit breaker (state driver)");
            ctx.metrics.finish_iteration();

            let scaffolded =
                try_scaffold_fallback(&ctx.wt_path, &ctx.issue.id, &ctx.issue.title, "", iteration);

            create_stuck_intervention(
                &mut ctx.session,
                &ctx.progress,
                &ctx.wt_path,
                iteration,
                &format!(
                    "No-change circuit breaker: {} consecutive no-change iterations{}",
                    ctx.escalation.consecutive_no_change,
                    if scaffolded {
                        " (scaffold committed)"
                    } else {
                        ""
                    },
                ),
            );
            return Ok(StateTransition::Fail {
                reason: "No-change circuit breaker".into(),
            });
        }

        // Run verifier to get escalation decision
        let engine = EscalationEngine::new();
        let current_vc = if ctx.config.verifier_packages.is_empty() {
            VerifierConfig {
                packages: detect_changed_packages(&ctx.wt_path, true),
                ..ctx.verifier_config.clone()
            }
        } else {
            ctx.verifier_config.clone()
        };
        let verifier = Verifier::new(&ctx.wt_path, current_vc);
        let report = verifier.run_pipeline().await;
        let decision = engine.decide(&mut ctx.escalation, &report);
        ctx.last_report = Some(report);
        ctx.metrics.finish_iteration();

        if decision.stuck {
            create_stuck_intervention(
                &mut ctx.session,
                &ctx.progress,
                &ctx.wt_path,
                iteration,
                &decision.reason,
            );
            return Ok(StateTransition::Fail {
                reason: format!("Stuck (no changes): {}", decision.reason),
            });
        }
        ctx.escalation.current_tier = decision.target_tier;
        return Ok(StateTransition::Advance {
            to: OrchestratorState::Planning,
            reason: format!("no changes, tier → {:?}", decision.target_tier),
        });
    }

    // Reset no-change counter
    ctx.escalation.reset_no_change();

    Ok(StateTransition::Advance {
        to: OrchestratorState::Verifying,
        reason: "agent produced changes, ready for verification".into(),
    })
}

/// Build the VerifierConfig for the current run.
///
/// When `verifier_packages` is empty (the default), auto-detect from git changes.
/// Otherwise, use the pre-configured package list.
fn scoped_verifier_config(ctx: &OrchestratorContext<'_>) -> VerifierConfig {
    if ctx.config.verifier_packages.is_empty() {
        VerifierConfig {
            packages: detect_changed_packages(&ctx.wt_path, true),
            ..ctx.verifier_config.clone()
        }
    } else {
        ctx.verifier_config.clone()
    }
}

/// Handle regression detection and optional rollback.
///
/// If `error_count > prev_error_count`, attempts a hard rollback to `pre_worker_commit`
/// and re-verifies. Returns `Some(Planning)` if rollback was successful; `None` otherwise.
async fn maybe_rollback_on_regression(
    ctx: &mut OrchestratorContext<'_>,
    iteration: u32,
    error_count: usize,
    prev_error_count: Option<usize>,
) -> Option<StateTransition> {
    let prev_count = prev_error_count?;
    if error_count <= prev_count {
        return None;
    }
    warn!(
        iteration,
        prev_errors = prev_count,
        curr_errors = error_count,
        "Regression detected (state driver)"
    );
    let mut rolled_back = false;
    if let Some(ref rollback_hash) = ctx.pre_worker_commit.clone() {
        match ctx.git_mgr.hard_rollback(rollback_hash) {
            Ok(()) => {
                rolled_back = true;
                info!(iteration, rollback_to = %rollback_hash, "Rolled back (state driver)");
            }
            Err(e) => {
                error!(iteration, "Rollback failed (state driver): {e}");
            }
        }
    }
    ctx.metrics.record_regression(rolled_back);
    if rolled_back {
        let rb_verifier = Verifier::new(&ctx.wt_path, scoped_verifier_config(ctx));
        let rb_report = rb_verifier.run_pipeline().await;
        ctx.last_report = Some(rb_report);
        ctx.metrics.finish_iteration();
        return Some(StateTransition::Advance {
            to: OrchestratorState::Planning,
            reason: "regression rolled back, retrying".into(),
        });
    }
    None
}

/// Verifying: run deterministic quality gates, auto-fix, regression detection.
pub async fn handle_verifying(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    let iteration = ctx.session.iteration();

    let verifier_start = Instant::now();
    let verifier = Verifier::new(&ctx.wt_path, scoped_verifier_config(ctx));
    let mut report = verifier.run_pipeline().await;
    let verifier_elapsed = verifier_start.elapsed();
    ctx.metrics.record_verifier_time(verifier_elapsed);

    info!(
        iteration,
        all_green = report.all_green,
        summary = %report.summary(),
        "Verifier report (state driver)"
    );

    // Record gate results
    for gate in &report.gates {
        let passed = matches!(gate.outcome, coordination::GateOutcome::Passed);
        ctx.span_summary.record_gate(passed, gate.duration_ms);
    }

    // Auto-fix
    ctx.auto_fix_applied = false;
    if !report.all_green {
        if let Some(fixed_report) =
            try_auto_fix(&ctx.wt_path, &ctx.verifier_config, iteration, &None).await
        {
            report = fixed_report;
            ctx.auto_fix_applied = true;
            ctx.metrics.record_auto_fix();
        }
    }

    let error_cats = report.unique_error_categories();
    let error_count = report.failure_signals.len();
    let cat_names: Vec<String> = error_cats.iter().map(|c| format!("{c:?}")).collect();
    ctx.metrics.record_verifier_results(error_count, cat_names);

    // Post per-iteration inference-level feedback to TZ (verifier_pass, edit_accuracy).
    // Gives Thompson Sampling per-call signal, not just per-episode.
    if let (Some(ref tz_url), Some(ref pg_url)) =
        (&ctx.config.tensorzero_url, &ctx.config.tensorzero_pg_url)
    {
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64();
        let since = now_secs - verifier_elapsed.as_secs_f64() - 10.0;
        let inf_ids = crate::tensorzero::resolve_recent_inference_ids(pg_url, since, 1).await;
        if let Some(inf_id) = inf_ids.first() {
            let primary_err = error_cats.first().map(|c| c.to_string());
            let iter_tags = crate::tensorzero::FeedbackTags {
                issue_id: Some(ctx.issue.id.clone()),
                error_category: primary_err,
                prompt_version: Some(crate::prompts::PROMPT_VERSION.to_string()),
                language: ctx.factory.language.clone(),
                model: ctx.config.cloud_endpoint.as_ref().map(|e| e.model.clone()),
                ..Default::default()
            };
            crate::tensorzero::post_inference_feedback(
                tz_url,
                inf_id,
                "verifier_pass",
                serde_json::Value::Bool(report.all_green),
                Some(iter_tags.clone()),
            )
            .await;
            crate::tensorzero::post_inference_feedback(
                tz_url,
                inf_id,
                "edit_accuracy",
                serde_json::json!(report.health_score()),
                Some(iter_tags),
            )
            .await;
        }
    }

    if let Some(lm) = ctx.metrics.build_loop_metrics(report.all_green) {
        lm.emit();
    }

    // Regression detection
    let prev_error_count = ctx.last_report.as_ref().map(|r| r.failure_signals.len());
    if !report.all_green {
        ctx.consecutive_validator_failures = 0;
        if let Some(t) =
            maybe_rollback_on_regression(ctx, iteration, error_count, prev_error_count).await
        {
            return Ok(t);
        }
    }

    // --- P2.3: CorrectionLoop integration ---
    // Record this verification attempt in the correction loop for escalation tracking.
    let has_linker = report.failure_signals.iter().any(|s| {
        let msg = s.message.to_lowercase();
        msg.contains("linker") || msg.contains("link error") || msg.contains("ld returned")
    });
    ctx.correction_loop.record_attempt(error_count, has_linker);

    // --- P2.2: Record router outcome ---
    // Feed back whether the current tier succeeded so the DynamicRouter learns.
    let primary_cat = error_cats.first().map(|c| format!("{c:?}"));
    let router_selection = coordination::router::task_classifier::ModelSelection::new(
        if ctx.escalation.current_tier == SwarmTier::Worker {
            ModelTier::Worker
        } else {
            ModelTier::Council
        },
        "verifier result",
    );
    ctx.router
        .record_outcome(&router_selection, primary_cat.as_deref(), report.all_green);

    if report.all_green {
        ctx.last_report = Some(report);
        Ok(StateTransition::Advance {
            to: OrchestratorState::Validating,
            reason: "all gates green".into(),
        })
    } else {
        // --- P2.3: CorrectionLoop-triggered escalation ---
        // The TieredCorrectionLoop tracks consecutive failures and triggers
        // escalation independently of the EscalationEngine. If it fires,
        // go straight to Escalating without waiting for the engine.
        if ctx.correction_loop.should_escalate() {
            let new_tier = ctx.correction_loop.escalate();
            info!(
                iteration,
                new_tier = %new_tier,
                total_attempts = ctx.correction_loop.total_attempts(),
                "CorrectionLoop triggered escalation (P2.3)"
            );
            ctx.last_report = Some(report);
            return Ok(StateTransition::Advance {
                to: OrchestratorState::Escalating,
                reason: format!(
                    "correction loop escalation to {new_tier} ({error_count} errors remaining)"
                ),
            });
        }

        ctx.last_report = Some(report);
        // Normal escalation path
        Ok(StateTransition::Advance {
            to: OrchestratorState::Escalating,
            reason: format!("{error_count} errors remaining"),
        })
    }
}

/// Validating: auto-fix false-positive guard, local validation, cloud validation, acceptance.
pub async fn handle_validating(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    let iteration = ctx.session.iteration();
    let report = ctx
        .last_report
        .as_ref()
        .expect("report must exist in Validating");
    let error_cats = report.unique_error_categories();
    let error_count = report.failure_signals.len();

    // Auto-fix false-positive guard
    if should_reject_auto_fix(ctx.auto_fix_applied, &ctx.acceptance_policy) {
        if let (Some(initial), Some(agent_commit)) = (
            ctx.session.state().initial_commit.as_ref(),
            ctx.git_mgr.current_commit_full().ok().as_ref(),
        ) {
            let agent_diff_lines = count_diff_lines(&ctx.wt_path, initial, agent_commit);
            if agent_diff_lines < ctx.acceptance_policy.min_diff_lines {
                warn!(
                    iteration,
                    agent_diff_lines,
                    min_required = ctx.acceptance_policy.min_diff_lines,
                    "Auto-fix false positive (state driver)"
                );
                ctx.escalation
                    .record_iteration(error_cats.clone(), 0, false, 0.0);
                ctx.metrics.finish_iteration();
                return Ok(StateTransition::Advance {
                    to: OrchestratorState::Planning,
                    reason: "auto-fix false positive, retrying".into(),
                });
            }
        }
    }

    // Local validation (blocking gate)
    if ctx.local_validator_enabled {
        if let Some(ref initial_commit) = ctx.session.state().initial_commit {
            info!(iteration, "Running local validation (state driver)");
            let local_result = local_validate(
                &ctx.reviewer,
                &ctx.wt_path,
                initial_commit,
                &ctx.config.fast_endpoint.model,
            )
            .await;

            ctx.metrics
                .record_local_validation(&local_result.model, local_result.passed);

            if local_result.passed {
                ctx.consecutive_validator_failures = 0;
            } else {
                ctx.consecutive_validator_failures += 1;
                warn!(
                    iteration,
                    consecutive_failures = ctx.consecutive_validator_failures,
                    "Local validation FAIL (state driver)"
                );

                let feedback = extract_local_validator_feedback(&local_result);
                ctx.last_validator_feedback = feedback;

                if ctx.consecutive_validator_failures >= ctx.max_validator_failures {
                    warn!(
                        iteration,
                        "Validator failure cap reached — accepting (state driver)"
                    );
                    ctx.consecutive_validator_failures = 0;
                } else {
                    ctx.escalation
                        .record_iteration(error_cats, error_count, false, 0.0);
                    ctx.metrics.finish_iteration();
                    return Ok(StateTransition::Advance {
                        to: OrchestratorState::Planning,
                        reason: "local validation rejected".into(),
                    });
                }
            }
        }
    }

    // Acceptance
    info!(
        iteration,
        "Verifier passed — checking acceptance (state driver)"
    );
    ctx.escalation
        .record_iteration(error_cats, error_count, true, 1.0);

    if let Ok(hash) = ctx.git_mgr.current_commit() {
        let _ = ctx
            .progress
            .log_checkpoint(ctx.session.session_id(), iteration, &hash);
    }
    let _ = ctx.progress.log_feature_complete(
        ctx.session.session_id(),
        iteration,
        &ctx.issue.id,
        "Verified (state driver)",
    );

    // Cloud validation (advisory)
    let mut cloud_passes = 0usize;
    if let Some(ref cloud_client) = ctx.factory.clients.cloud {
        if let Some(ref initial_commit) = ctx.session.state().initial_commit {
            let validations = cloud_validate(cloud_client, &ctx.wt_path, initial_commit).await;
            ctx.last_validator_feedback.clear();
            for v in &validations {
                ctx.metrics.record_cloud_validation(&v.model, v.passed);
                if v.passed {
                    cloud_passes += 1;
                } else {
                    let feedback = extract_validator_feedback(v);
                    ctx.last_validator_feedback.extend(feedback);
                }
            }
        }
    }

    if let Some(policy) = ctx.last_harness_policy.as_ref() {
        if policy.judgment.external_review_required
            && ctx.factory.clients.cloud.is_some()
            && cloud_passes < policy.judgment.min_external_passes
        {
            warn!(
                iteration,
                required = policy.judgment.min_external_passes,
                observed = cloud_passes,
                "Harness judgment rejected resolution after external review"
            );
            ctx.metrics.finish_iteration();
            return Ok(StateTransition::Advance {
                to: OrchestratorState::Planning,
                reason: "external judgment rejected".into(),
            });
        }
    }

    // Acceptance policy check
    let acceptance_result = acceptance::check_acceptance_with_task(
        &ctx.acceptance_policy,
        &ctx.wt_path,
        ctx.session.state().initial_commit.as_deref(),
        cloud_passes,
        Some(acceptance::TaskMetadata::new(
            &ctx.issue.title,
            ctx.issue.description.as_deref(),
        )),
    );

    if !acceptance_result.accepted {
        for rejection in &acceptance_result.rejections {
            warn!(iteration, rejection = %rejection, "Acceptance rejected (state driver)");
        }
        ctx.metrics.finish_iteration();
        return Ok(StateTransition::Advance {
            to: OrchestratorState::Planning,
            reason: "acceptance policy rejected".into(),
        });
    }

    ctx.metrics.finish_iteration();
    // success is set after merge confirms in drive()
    Ok(StateTransition::Complete)
}

/// Escalating: pre-escalation KB check, engine decision, stuck detection.
///
/// Simplified high-level flow:
/// 1. State check: ensure verifier report is available
/// 2. Decision: delegate to orchestrator's EscalationEngine
/// 3. Execution: handle escalation event or stuck detection
pub async fn handle_escalating(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    let iteration = ctx.session.iteration();
    let report = ctx
        .last_report
        .take()
        .ok_or_else(|| anyhow::anyhow!("No verifier report available in Escalating state"))?;

    // Pre-escalation KB check
    handle_escalating_kb_check(ctx, &report, iteration);

    // Escalation decision
    let decision = handle_escalating_decision(ctx, &report);

    // Handle escalation event if tier changed
    handle_escalation_event(ctx, &decision, iteration);

    // Finish iteration metrics
    ctx.metrics.finish_iteration();

    // Handle stuck detection or return to Planning
    match decision.stuck {
        true => handle_escalating_stuck(ctx, &decision, iteration),
        false => {
            // Failure classification and intent guard — runs on every retry transition
            // so the swarm knows WHY it's retrying and can log drift early.
            handle_escalating_classify(ctx, &report, iteration);
            Ok(StateTransition::Advance {
                to: OrchestratorState::Planning,
                reason: format!("escalation → {:?}", decision.target_tier),
            })
        }
    }
}

/// Pre-escalation knowledge base check.
/// Queries the KB for known fixes based on error categories.
fn handle_escalating_kb_check(
    ctx: &OrchestratorContext<'_>,
    report: &VerifierReport,
    iteration: u32,
) {
    if let Some(kb) = ctx.knowledge_base {
        let error_cats: Vec<String> = report
            .unique_error_categories()
            .iter()
            .map(|c| format!("{c:?}"))
            .collect();
        if !error_cats.is_empty() {
            let question = format!("Known fix for Rust errors: {}", error_cats.join(", "));
            let response = query_kb_with_failsafe(kb, "debugging_kb", &question);
            if !response.is_empty() {
                info!(iteration, "Found KB suggestion (state driver)");
            }
        }
    }
}

/// Make the escalation decision using the EscalationEngine.
fn handle_escalating_decision(
    ctx: &mut OrchestratorContext<'_>,
    report: &VerifierReport,
) -> EscalationDecision {
    let engine = EscalationEngine::new();
    let decision = engine.decide(&mut ctx.escalation, report);
    ctx.last_report = Some(report.clone());
    decision
}

/// Handle the escalation event if the tier changed.
fn handle_escalation_event(
    ctx: &mut OrchestratorContext<'_>,
    decision: &EscalationDecision,
    iteration: u32,
) {
    if decision.escalated {
        ctx.metrics.record_escalation();
        ctx.span_summary.record_escalation();
        info!(
            iteration,
            from = ?ctx.escalation.current_tier,
            to = ?decision.target_tier,
            reason = %decision.reason,
            "Tier escalated (state driver)"
        );
    }
}

/// Classify the failure on a retry and check intent drift.
///
/// Called when escalation is NOT stuck (i.e., we're retrying). Classifies WHY
/// the iteration failed and warns if the reformulated task drifts from the
/// original intent contract. Recovery action routing:
///
/// - `DecompositionRequired` → logged; subtask fan-out handled by subtask.rs if triggered
/// - `ImplementationThrash`  → logged for the cloud manager to notice
/// - `ContextDeficit`        → logged; Planning auto-context enrichment helps next iter
/// - `ToolConstraintMismatch`→ logged; rewrite applied at session end by reformulate()
/// - `InfraTransient`        → logged; normal retry continues
/// - Others                  → logged; normal retry continues
fn handle_escalating_classify(
    ctx: &OrchestratorContext<'_>,
    report: &VerifierReport,
    iteration: u32,
) {
    use crate::orchestrator::helpers::{list_changed_files, load_failure_ledger};
    use crate::reformulation::{classify_failure, FailureReviewInput};

    let failure_ledger = load_failure_ledger(&ctx.wt_path);
    let files_changed = list_changed_files(&ctx.wt_path);
    let error_cats: Vec<String> = report
        .unique_error_categories()
        .iter()
        .map(|c| format!("{c:?}"))
        .collect();

    let review_input = FailureReviewInput {
        issue_id: ctx.issue.id.clone(),
        issue_title: ctx.issue.title.clone(),
        issue_description: ctx.issue.description.clone(),
        failure_ledger,
        iterations_used: iteration,
        max_iterations: ctx.config.max_retries,
        files_changed,
        error_categories: error_cats,
        failure_reason: Some(ctx.escalation.summary()),
    };

    let classification = classify_failure(&review_input);
    let fingerprint = classification.fingerprint();

    info!(
        iteration,
        issue = %ctx.issue.id,
        classification = ?classification,
        fingerprint = %fingerprint,
        "Failure classified on retry (state driver)"
    );

    // Intent guard: verify the current task description still matches the original intent.
    // Compute a fresh IntentContract from the current issue state and compare digests.
    // This detects drift early (before the reformulation engine rewrites at session end).
    let current = IntentContract::from_issue(
        &ctx.issue.id,
        &ctx.issue.title,
        ctx.issue.description.as_deref(),
    );
    if current.intent_digest != ctx.intent_contract.intent_digest {
        warn!(
            iteration,
            issue = %ctx.issue.id,
            original_digest = %ctx.intent_contract.intent_digest,
            current_digest = %current.intent_digest,
            "Intent drift detected: task description changed since initial pickup (state driver)"
        );
    }
}

/// Handle stuck detection and return a Fail transition.
fn handle_escalating_stuck(
    ctx: &mut OrchestratorContext<'_>,
    decision: &EscalationDecision,
    iteration: u32,
) -> Result<StateTransition> {
    error!(iteration, reason = %decision.reason, "Stuck (state driver)");
    create_stuck_intervention(
        &mut ctx.session,
        &ctx.progress,
        &ctx.wt_path,
        iteration,
        &decision.reason,
    );
    Ok(StateTransition::Fail {
        reason: format!("Stuck: {}", decision.reason),
    })
}

/// Merging: knowledge capture, merge worktree, close issue.
pub async fn handle_merging(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    // Knowledge capture
    if let Some(kb) = ctx.knowledge_base {
        let _ = knowledge_sync::capture_resolution(
            kb,
            &ctx.issue.id,
            &ctx.issue.title,
            ctx.session.iteration(),
            &format!("{:?}", ctx.escalation.current_tier),
            &[],
        );

        if ctx.session.iteration() >= 3 {
            let error_cats: Vec<String> = ctx
                .escalation
                .recent_error_categories
                .iter()
                .flatten()
                .map(|c| format!("{c:?}"))
                .collect();
            let _ = knowledge_sync::capture_error_pattern(
                kb,
                &ctx.issue.id,
                &error_cats,
                ctx.session.iteration(),
                &format!(
                    "Resolved after {} iterations at {:?} tier",
                    ctx.session.iteration(),
                    ctx.escalation.current_tier
                ),
            );
        }
    }

    info!(
        id = %ctx.issue.id,
        session_id = ctx.session.short_id(),
        elapsed = %ctx.session.elapsed_human(),
        iterations = ctx.session.iteration(),
        "Issue resolved — merging worktree (state driver)"
    );

    // --- Pre-merge verifier guarantee (RepoProver pattern) ---
    //
    // Run compile_only() verification IN THE WORKTREE before merging to main.
    // Only merge branches where all gates pass.
    if ctx.config.verify_before_merge {
        let pre_merge_config = VerifierConfig::compile_only();
        let pre_merge_report = Verifier::new(&ctx.wt_path, pre_merge_config)
            .run_pipeline()
            .await;
        if !pre_merge_report.all_green {
            warn!(
                id = %ctx.issue.id,
                summary = %pre_merge_report.summary(),
                "Pre-merge verifier failed in worktree (state driver) — not merging"
            );
            ctx.last_report = Some(pre_merge_report);
            ctx.success = false;
            return Ok(StateTransition::Advance {
                to: OrchestratorState::Planning,
                reason: "pre-merge verification failed — retrying".into(),
            });
        }
    }

    if let Err(e) = land_issue_or_reopen(
        ctx.worktree_bridge,
        ctx.beads,
        &ctx.issue.id,
        "Resolved by swarm orchestrator (state driver)",
    ) {
        error!(id = %ctx.issue.id, "Landing failed (state driver): {e}");
        return Err(e);
    }

    // Mark session complete only after authoritative landing succeeds
    ctx.session.complete();
    let _ = ctx.progress.log_session_end(
        ctx.session.session_id(),
        ctx.session.iteration(),
        format!("Issue {} resolved (state driver)", ctx.issue.id),
    );

    // Retrospective (post-merge)
    if let Some(kb) = ctx.knowledge_base {
        let entries = ctx.progress.read_all().unwrap_or_default();
        let retro = ctx.session.retrospective(&entries);
        let svc = knowledge_sync::KnowledgeSyncService::new(kb);
        let captures = svc.capture_from_retrospective(&retro, &ctx.issue.id, &ctx.issue.title);
        debug!(
            count = captures.len(),
            "Retrospective captures (state driver)"
        );
    }

    orchestrator::clear_resume_file(ctx.worktree_bridge.repo_root());
    info!(id = %ctx.issue.id, "Issue closed (state driver)");

    Ok(StateTransition::Advance {
        to: OrchestratorState::Resolved,
        reason: "merged and closed".into(),
    })
}

// ---------------------------------------------------------------------------
// drive() — the main state-machine loop
// ---------------------------------------------------------------------------

/// Drive the orchestrator through the state machine until a terminal state.
pub async fn drive(ctx: &mut OrchestratorContext<'_>) -> Result<bool> {
    loop {
        let current = ctx.state_machine.current();
        if current.is_terminal() {
            break;
        }

        // Budget enforcement
        if let Some(reason) = ctx.budget_tracker.check_budget(current) {
            warn!(state = %current, %reason, "Budget exhausted (state driver)");
            let reason_str = reason.to_string();
            ctx.state_machine.fail(&reason_str)?;
            ctx.log_event(crate::session::EventKind::StateTransition {
                from: current,
                to: OrchestratorState::Failed,
                iteration: ctx.state_machine.iteration(),
                reason: Some(format!("budget exhausted: {reason_str}")),
            });
            break;
        }

        let transition = match current {
            OrchestratorState::SelectingIssue => handle_selecting_issue(ctx).await?,
            OrchestratorState::PreparingWorktree => handle_preparing_worktree(ctx).await?,
            OrchestratorState::Planning => handle_planning(ctx).await?,
            OrchestratorState::Implementing => handle_implementing(ctx).await?,
            OrchestratorState::Verifying => handle_verifying(ctx).await?,
            OrchestratorState::Validating => handle_validating(ctx).await?,
            OrchestratorState::Escalating => handle_escalating(ctx).await?,
            OrchestratorState::Merging => handle_merging(ctx).await?,
            OrchestratorState::Resolved | OrchestratorState::Failed => break,
        };

        match transition {
            StateTransition::Advance { to, reason } => {
                let from = ctx.state_machine.current();
                ctx.state_machine.advance(to, Some(&reason))?;
                ctx.budget_tracker.on_state_entered(to);

                // Emit state transition event to session log.
                ctx.log_event(crate::session::EventKind::StateTransition {
                    from,
                    to,
                    iteration: ctx.state_machine.iteration(),
                    reason: Some(reason),
                });

                // Checkpoint at stable points (atomic: write-tmp → fsync → rename).
                let git_hash = ctx.git_mgr.current_commit_full().ok();
                if let Some(cp) = ctx
                    .state_machine
                    .checkpoint(&ctx.issue.id, git_hash.as_deref())
                {
                    let cp_path = ctx.wt_path.join(".swarm-checkpoint.json");
                    let tmp_path = ctx.wt_path.join(".swarm-checkpoint.json.tmp");
                    if let Ok(json) = serde_json::to_string_pretty(&cp) {
                        if let Ok(()) = std::fs::write(&tmp_path, &json) {
                            // fsync the temp file before rename to ensure data reaches
                            // stable storage (CodeRabbit review feedback on PR #132).
                            if let Ok(f) = std::fs::File::open(&tmp_path) {
                                let _ = f.sync_all();
                            }
                            let _ = std::fs::rename(&tmp_path, &cp_path);
                        }
                    }

                    ctx.log_event(crate::session::EventKind::CheckpointWritten {
                        checkpoint_id: cp.checkpoint_id,
                        state: cp.state,
                        iteration: cp.iteration,
                    });
                }
            }
            StateTransition::Complete => {
                let from = ctx.state_machine.current();
                ctx.state_machine
                    .advance(OrchestratorState::Merging, Some("all gates passed"))?;
                ctx.budget_tracker
                    .on_state_entered(OrchestratorState::Merging);

                ctx.log_event(crate::session::EventKind::StateTransition {
                    from,
                    to: OrchestratorState::Merging,
                    iteration: ctx.state_machine.iteration(),
                    reason: Some("all gates passed".into()),
                });

                // Execute merge
                let merge_result = handle_merging(ctx).await?;
                match merge_result {
                    StateTransition::Advance { to, reason } => {
                        ctx.state_machine.advance(to, Some(&reason))?;
                        ctx.success = true;
                        ctx.log_event(crate::session::EventKind::StateTransition {
                            from: OrchestratorState::Merging,
                            to,
                            iteration: ctx.state_machine.iteration(),
                            reason: Some(reason),
                        });
                    }
                    StateTransition::Fail { reason } => {
                        ctx.state_machine.fail(&reason)?;
                        ctx.log_event(crate::session::EventKind::StateTransition {
                            from: OrchestratorState::Merging,
                            to: OrchestratorState::Failed,
                            iteration: ctx.state_machine.iteration(),
                            reason: Some(reason),
                        });
                    }
                    StateTransition::Complete => {
                        ctx.state_machine
                            .advance(OrchestratorState::Resolved, Some("merged and resolved"))?;
                        ctx.success = true;
                        ctx.log_event(crate::session::EventKind::StateTransition {
                            from: OrchestratorState::Merging,
                            to: OrchestratorState::Resolved,
                            iteration: ctx.state_machine.iteration(),
                            reason: Some("merged and resolved".into()),
                        });
                    }
                }
                break;
            }
            StateTransition::Fail { reason } => {
                let from = ctx.state_machine.current();
                ctx.state_machine.fail(&reason)?;
                ctx.log_event(crate::session::EventKind::StateTransition {
                    from,
                    to: OrchestratorState::Failed,
                    iteration: ctx.state_machine.iteration(),
                    reason: Some(reason),
                });
                break;
            }
        }
    }

    // Emit terminal session event.
    let duration_ms = ctx.process_start.elapsed().as_millis() as u64;
    let merge_commit = if ctx.success {
        ctx.git_mgr.current_commit_full().ok()
    } else {
        None
    };
    let failure_reason = if !ctx.success {
        ctx.state_machine
            .transitions()
            .last()
            .and_then(|t| t.reason.clone())
    } else {
        None
    };
    ctx.log_event(crate::session::EventKind::SessionCompleted {
        resolved: ctx.success,
        total_iterations: ctx.state_machine.iteration(),
        duration_ms,
        merge_commit,
        failure_reason,
    });

    Ok(ctx.success)
}

// ---------------------------------------------------------------------------
// handle_outcome() — shared outcome handling for both legacy and driver paths
// ---------------------------------------------------------------------------

/// Handle the post-loop outcome: telemetry, SLO evaluation, KB refresh.
///
/// Extracted from the end of `process_issue()` so both legacy and state driver
/// paths can share it.
///
/// Takes `MetricsCollector` by value because `finalize()` consumes `self`.
/// The caller should `std::mem::replace` the collector out of the context
/// before calling this.
pub async fn handle_outcome(ctx: &mut OrchestratorContext<'_>, metrics: MetricsCollector) {
    if !ctx.success {
        ctx.session.fail();
        let _ = ctx.progress.log_session_end(
            ctx.session.session_id(),
            ctx.session.iteration(),
            format!(
                "Failed after {} iterations — {}",
                ctx.session.iteration(),
                ctx.escalation.summary()
            ),
        );

        // Retrospective on failure
        if let Some(kb) = ctx.knowledge_base {
            let entries = ctx.progress.read_all().unwrap_or_default();
            let retro = ctx.session.retrospective(&entries);
            let svc = knowledge_sync::KnowledgeSyncService::new(kb);
            let captures = svc.capture_from_retrospective(&retro, &ctx.issue.id, &ctx.issue.title);
            debug!(
                count = captures.len(),
                "Retrospective (failure, state driver)"
            );
        }

        // Reformulation engine — classify WHY the session failed and rewrite the bead
        // description so the next attempt has a solvable formulation. Mirrors the
        // equivalent logic in the legacy orchestrator/mod.rs path.
        {
            use crate::orchestrator::helpers::{list_changed_files, load_failure_ledger};
            use crate::reformulation::{reformulate, FailureReviewInput};

            let failure_ledger = load_failure_ledger(&ctx.wt_path);
            let files_changed = list_changed_files(&ctx.wt_path);
            let error_cats: Vec<String> = ctx
                .last_report
                .as_ref()
                .map(|r| {
                    r.unique_error_categories()
                        .iter()
                        .map(|c| c.to_string())
                        .collect()
                })
                .unwrap_or_default();

            let review_input = FailureReviewInput {
                issue_id: ctx.issue.id.clone(),
                issue_title: ctx.issue.title.clone(),
                issue_description: ctx.issue.description.clone(),
                failure_ledger,
                iterations_used: ctx.session.iteration(),
                max_iterations: ctx.config.max_retries,
                files_changed,
                error_categories: error_cats,
                failure_reason: Some(ctx.escalation.summary()),
            };

            let result = reformulate(&ctx.reformulation_store, &review_input);
            let bridge = BeadsBridge::new();

            if let Some(ref new_desc) = result.new_description {
                match bridge.update_description(&ctx.issue.id, new_desc) {
                    Ok(()) => info!(
                        id = %ctx.issue.id,
                        classification = ?result.classification,
                        "Reformulated issue description (state driver)"
                    ),
                    Err(e) => warn!(
                        id = %ctx.issue.id,
                        error = %e,
                        "Failed to update issue description (reformulation not applied, state driver)"
                    ),
                }
            }

            if let Some(ref notes) = result.notes_appended {
                if let Err(e) = bridge.update_notes(&ctx.issue.id, notes) {
                    warn!(
                        id = %ctx.issue.id,
                        error = %e,
                        "Failed to append reformulation notes (state driver)"
                    );
                }
            }

            if result.escalated {
                let _ = bridge.add_swarm_label(&ctx.issue.id, "swarm:needs-human-review");
                warn!(
                    id = %ctx.issue.id,
                    "Reformulation exhausted — labeled for human review (state driver)"
                );
            }

            // Reset issue to open so the next loop iteration picks it up with the
            // rewritten description (unless escalated to human review).
            if !result.escalated {
                let _ = ctx.beads.update_status(&ctx.issue.id, "open");
            }
        }

        // Save session state for resume
        let state_path = ctx.wt_path.join(".swarm-session.json");
        if let Err(e) = save_session_state(ctx.session.state(), &state_path) {
            warn!("Failed to save session state (state driver): {e}");
        }

        // Resume file
        let resume = SwarmResumeFile {
            issue: ctx.issue.clone(),
            worktree_path: ctx.wt_path.display().to_string(),
            iteration: ctx.session.iteration(),
            escalation_summary: ctx.escalation.summary(),
            current_tier: format!("{:?}", ctx.escalation.current_tier),
            total_iterations: ctx.escalation.total_iterations,
            saved_at: chrono::Utc::now().to_rfc3339(),
        };
        let resume_path = ctx.worktree_bridge.repo_root().join(".swarm-resume.json");
        if let Ok(json) = serde_json::to_string_pretty(&resume) {
            let _ = std::fs::write(&resume_path, json);
        }

        error!(
            id = %ctx.issue.id,
            session_id = ctx.session.short_id(),
            elapsed = %ctx.session.elapsed_human(),
            iterations = ctx.session.iteration(),
            summary = %ctx.escalation.summary(),
            "Issue NOT resolved (state driver)"
        );
    }

    // Telemetry
    let final_tier = format!("{:?}", ctx.escalation.current_tier);
    let mut session_metrics = metrics.finalize(ctx.success, &final_tier);
    telemetry::write_session_metrics(&session_metrics, &ctx.wt_path);
    if telemetry::write_proof_of_work(&session_metrics, &ctx.wt_path) {
        session_metrics.harness_trace.proof_of_work_emitted = true;
    }
    telemetry::append_telemetry(&session_metrics, ctx.worktree_bridge.repo_root());

    // TensorZero feedback — health-gated to prevent data poisoning.
    //
    // Only post feedback when the outcome reflects model quality, not infrastructure
    // failures. Infrastructure issues (0 iterations, <10s wall time on failure,
    // empty-response errors) would pollute Thompson Sampling with false negatives
    // that blame the model for crashes, hangs, or context overflow.
    //
    // Design: docs/research/self-improving-swarm-architecture.md Layer 1.
    if let Some(tz_url) = ctx.config.tensorzero_url.as_deref() {
        let wall_secs = ctx.process_start.elapsed().as_secs_f64();
        let iterations = ctx.session.iteration();

        // Health gate: determine if this outcome is meaningful model signal.
        // Infrastructure failures have characteristic signatures:
        // - 0 iterations completed (crashed before any LLM call)
        // - Very short wall time on failure (<30s means crash, not failed attempt)
        // - Success always gets posted (it's always real signal)
        let is_infra_failure = !ctx.success && (iterations == 0 || wall_secs < 30.0);

        if is_infra_failure {
            info!(
                id = %ctx.issue.id,
                iterations,
                wall_secs = format!("{wall_secs:.0}"),
                "Skipping TZ feedback: likely infrastructure failure (0 iters or <30s)"
            );
        } else {
            // Resolve actual TZ episode IDs from Postgres — TZ auto-generates
            // UUIDv7 episode IDs per inference, which differ from our session_id.
            // We must post feedback to TZ's own episode IDs for Thompson Sampling
            // to correlate feedback with the correct variant.
            // Build rich tags for TZ episode feedback (was sparse: only 3/16 fields).
            let primary_err = ctx.last_report.as_ref().and_then(|r| {
                r.unique_error_categories()
                    .into_iter()
                    .next()
                    .map(|c| c.to_string())
            });
            let tags = crate::tensorzero::FeedbackTags {
                issue_id: Some(ctx.issue.id.clone()),
                language: ctx.factory.language.clone(),
                model: ctx.config.cloud_endpoint.as_ref().map(|e| e.model.clone()),
                repo_id: ctx.config.repo_id.clone(),
                error_category: primary_err,
                prompt_version: Some(crate::prompts::PROMPT_VERSION.to_string()),
                retry_tier: Some(final_tier.clone()),
                write_deadline: Some(ctx.config.max_turns_without_write),
                max_tool_calls: Some(ctx.config.max_worker_tool_calls),
                ..Default::default()
            };

            // Resolve TZ episode IDs. If PG is unreachable, use a generated fallback.
            let tz_pg = match ctx.config.tensorzero_pg_url.as_deref() {
                Some(pg) => pg,
                None => {
                    warn!(id = %ctx.issue.id, "TZ PG URL not configured — skipping episode feedback");
                    // Jump past the feedback block
                    ""
                }
            };
            if !tz_pg.is_empty() {
                let session_start = ctx.process_start.elapsed().as_secs_f64();
                let start_timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs_f64()
                    - session_start;
                let episode_ids =
                    crate::tensorzero::resolve_episode_ids(tz_pg, start_timestamp).await;

                if episode_ids.is_empty() {
                    let fallback_id = crate::tensorzero::generate_episode_id(
                        &ctx.issue.id,
                        ctx.session.session_id(),
                    );
                    warn!(
                        id = %ctx.issue.id,
                        fallback_id = %fallback_id,
                        "No TZ episode IDs resolved — using generated UUIDv7 fallback"
                    );
                    crate::tensorzero::post_episode_feedback(
                        tz_url,
                        &fallback_id,
                        ctx.success,
                        iterations,
                        wall_secs,
                        Some(tags),
                    )
                    .await;
                } else {
                    info!(
                        count = episode_ids.len(),
                        success = ctx.success,
                        "TZ episode feedback: posting to {} resolved episodes",
                        episode_ids.len()
                    );
                    for ep_id in &episode_ids {
                        crate::tensorzero::post_episode_feedback(
                            tz_url,
                            ep_id,
                            ctx.success,
                            iterations,
                            wall_secs,
                            Some(tags.clone()),
                        )
                        .await;
                    }
                }
            }
        }
    }

    otel::record_process_result(
        &ctx.process_span,
        ctx.success,
        session_metrics.total_iterations,
        ctx.process_start.elapsed().as_millis() as u64,
    );

    info!(summary = %ctx.span_summary, "OTel span summary (state driver)");

    // SLO evaluation
    let escalated = session_metrics.iterations.iter().any(|i| i.escalated);
    let orch_metrics = OrchestrationMetrics {
        session_count: 1,
        first_pass_rate: if session_metrics.total_iterations == 1 && ctx.success {
            1.0
        } else {
            0.0
        },
        overall_success_rate: if ctx.success { 1.0 } else { 0.0 },
        avg_iterations_to_green: session_metrics.total_iterations as f64,
        median_iterations_to_green: session_metrics.total_iterations as f64,
        escalation_rate: if escalated { 1.0 } else { 0.0 },
        avg_escalations: if escalated { 1.0 } else { 0.0 },
        latency_p50: std::time::Duration::from_millis(session_metrics.elapsed_ms),
        latency_p95: std::time::Duration::from_millis(session_metrics.elapsed_ms),
        latency_max: std::time::Duration::from_millis(session_metrics.elapsed_ms),
        tokens_p50: 0,
        tokens_p95: 0,
        tokens_total: 0,
        cost_total: 0.0,
        cost_avg: 0.0,
        stuck_rate: if !ctx.success { 1.0 } else { 0.0 },
        avg_turns_until_first_write: session_metrics.turns_until_first_write.unwrap_or(0) as f64,
        write_by_turn_2_rate: if session_metrics.write_by_turn_2 {
            1.0
        } else {
            0.0
        },
    };
    let slo_report = slo::evaluate_slos(&orch_metrics);
    match slo_report.overall_severity {
        AlertSeverity::Ok => info!(
            passing = slo_report.passing,
            "SLO: all passing (state driver)"
        ),
        AlertSeverity::Warning => warn!(
            passing = slo_report.passing,
            "SLO: warnings (state driver)\n{}",
            slo_report.summary()
        ),
        AlertSeverity::Critical => error!(
            passing = slo_report.passing,
            "SLO: CRITICAL (state driver)\n{}",
            slo_report.summary()
        ),
    }

    // KB refresh
    let telemetry_path = ctx
        .worktree_bridge
        .repo_root()
        .join(".swarm-telemetry.jsonl");
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
                    "KB refresh (state driver): {refresh_report}"
                );
            }
        }

        let skills = coordination::analytics::skills::SkillLibrary::new();
        let now = chrono::Utc::now();
        let dashboard = crate::dashboard::generate(reader.sessions(), &skills, now);
        let summary = crate::dashboard::format_summary(&dashboard);
        info!(sessions = reader.sessions().len(), "\n{summary}");
    }

    // --- Mutation archive (Phase 4a: evolutionary tracking) ---
    //
    // Record outcome for SERA-14B critic training and strategy seeding.
    // Mirrors the legacy path in orchestrator/mod.rs:4056-4145.
    {
        let archive =
            crate::mutation_archive::MutationArchive::new(ctx.worktree_bridge.repo_root());
        let language = ctx
            .factory
            .language
            .as_deref()
            .unwrap_or("rust")
            .to_string();
        let final_tier = format!("{:?}", ctx.escalation.current_tier);
        let primary_model = ctx
            .config
            .resolve_role_model(crate::config::SwarmRole::Planner);

        let mut record = crate::mutation_archive::build_record(
            &ctx.issue.id,
            &ctx.issue.title,
            &language,
            ctx.success,
            ctx.session.iteration(),
            &final_tier,
            &primary_model,
            ctx.process_start.elapsed().as_secs(),
        );

        // Populate error categories and first-failure gate from last verifier report.
        record.auto_fix_only = ctx.success && ctx.session.iteration() == 0;
        if let Some(ref report) = ctx.last_report {
            record.error_categories = report
                .unique_error_categories()
                .iter()
                .map(|c| c.to_string())
                .collect();
            record.first_failure_gate = report.first_failure.clone();
        }
        if !ctx.success {
            record.failure_reason = Some(ctx.escalation.summary());
        }

        // Populate files changed and line counts.
        record.files_changed = crate::orchestrator::helpers::list_changed_files(&ctx.wt_path);
        if let Ok(output) = std::process::Command::new("git")
            .args(["diff", "--stat", "main"])
            .current_dir(&ctx.wt_path)
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

        // --- Part C: MAP-Elites diversity update ---
        //
        // Insert the outcome into the quality-diversity archive so the swarm
        // tracks which (complexity, strategy) bins have been explored and
        // avoids converging on a single approach.
        {
            let lines_changed = (record.lines_added + record.lines_removed) as usize;
            let files_changed = record.files_changed.len();
            let features = crate::map_elites::FeatureExtractor::extract_from_metadata(
                lines_changed,
                files_changed,
            );
            let score = if ctx.success {
                // Reward speed: fewer iterations → higher score.
                let iter = ctx.session.iteration().max(1) as f64;
                1.0 / iter
            } else {
                0.0
            };
            let node = crate::map_elites::ExperimentNode {
                id: ctx.issue.id.clone(),
                description: ctx.issue.title.clone(),
                score,
                lines_changed,
                files_changed,
                strategy: crate::map_elites::FeatureExtractor::infer_strategy(
                    files_changed,
                    lines_changed,
                ),
            };
            let mut qd_archive = crate::map_elites::QualityDiversityArchive::new();
            // Load existing archive records into the in-memory grid so coverage
            // reflects accumulated history, not just the current session.
            for past in archive.load_all() {
                let past_lines = (past.lines_added + past.lines_removed) as usize;
                let past_files = past.files_changed.len();
                let past_features = crate::map_elites::FeatureExtractor::extract_from_metadata(
                    past_lines, past_files,
                );
                let past_score = if past.resolved {
                    1.0 / past.iterations.max(1) as f64
                } else {
                    0.0
                };
                let past_node = crate::map_elites::ExperimentNode {
                    id: past.issue_id.clone(),
                    description: past.issue_title.clone(),
                    score: past_score,
                    lines_changed: past_lines,
                    files_changed: past_files,
                    strategy: crate::map_elites::FeatureExtractor::infer_strategy(
                        past_files, past_lines,
                    ),
                };
                qd_archive.insert(past_node, past_features);
            }
            let inserted = qd_archive.insert(node, features);
            info!(
                issue = %ctx.issue.id,
                complexity_bin = features.complexity_bin,
                strategy_bin = features.strategy_bin,
                score,
                inserted,
                coverage = qd_archive.coverage(),
                occupied_cells = qd_archive.len(),
                "MAP-Elites: updated quality-diversity archive (state driver)"
            );
        }

        // --- Hyperagents: extract skills from successful mutations ---
        if ctx.success {
            let skills_path = ctx.worktree_bridge.repo_root().join(".swarm/skills.json");
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
                            warn!(
                                error = %e,
                                "Hyperagents: failed to save skill library (state driver)"
                            );
                        } else {
                            info!(
                                skill = %candidate.approach_summary,
                                "Hyperagents: extracted skill from successful mutation (state driver)"
                            );
                        }
                    }
                }
            }
        }
    }

    // --- Self-assessment (Layer 3) ---
    //
    // Periodically query TZ for variant performance and detect anomalies.
    // Uses a static counter to trigger every ASSESSMENT_INTERVAL issues.
    // **NOT YET VALIDATED** — corrective actions are logged only.
    // Design: docs/research/self-improving-swarm-architecture.md Layer 3.
    {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COMPLETED_COUNT: AtomicUsize = AtomicUsize::new(0);
        let count = COMPLETED_COUNT.fetch_add(1, Ordering::Relaxed) + 1;

        if count.is_multiple_of(crate::self_assessment::ASSESSMENT_INTERVAL) {
            info!(
                completed = count,
                "Triggering periodic self-assessment (every {} issues)",
                crate::self_assessment::ASSESSMENT_INTERVAL
            );
            // Respect opt-out: if TZ PG URL is not configured, skip assessment.
            let tz_pg = ctx.config.tensorzero_pg_url.as_deref();
            if let Some(report) =
                crate::self_assessment::run_assessment(tz_pg, ctx.worktree_bridge.repo_root()).await
            {
                if report.degradation_detected {
                    error!(
                        window = format!("{:.1}%", report.window_success_rate),
                        baseline = format!("{:.1}%", report.baseline_success_rate),
                        "SELF-ASSESSMENT: degradation detected — check TZ variant performance"
                    );
                }
                if !report.broken_variants.is_empty() {
                    warn!(
                        variants = ?report.broken_variants,
                        "SELF-ASSESSMENT: broken variants detected (0% success)"
                    );
                }
                if report.traffic_concentrated {
                    if let Some((ref variant, pct)) = report.dominant_variant {
                        warn!(
                            variant = %variant,
                            pct = format!("{pct:.1}%"),
                            "SELF-ASSESSMENT: traffic overly concentrated on single variant"
                        );
                    }
                }
            }
        }
    }
}
