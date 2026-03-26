//! State-machine-driven orchestrator loop.
//!
//! Replaces the monolithic `loop {}` in `orchestrator.rs` with typed state handlers.
//! Each handler returns a `StateTransition` telling the driver which state to enter next.
//!
//! Gated behind `SWARM_STATE_DRIVER=1` — the legacy path remains the default.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use tracing::{debug, error, info, warn};

use crate::acceptance::{self, AcceptancePolicy};
use crate::agents::coder::OaiAgent;
use crate::agents::AgentFactory;
use crate::beads_bridge::{BeadsIssue, IssueTracker};
use crate::cluster_health::ClusterHealth;
use crate::config::{SwarmConfig, SwarmRole};
use crate::file_targeting::detect_changed_packages;
use crate::knowledge_sync;
use crate::notebook_bridge::KnowledgeBase;
use crate::orchestrator::{
    self, bool_from_env, cloud_validate, collect_artifacts_from_diff, count_diff_lines,
    create_stuck_intervention, extract_local_validator_feedback, extract_validator_feedback,
    format_compact_task_prompt, format_task_prompt, git_commit_changes, local_validate,
    prompt_with_hook_and_retry, query_kb_with_failsafe, route_to_coder, should_reject_auto_fix,
    try_auto_fix, try_scaffold_fallback, CoderRoute, SwarmResumeFile,
};
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
    ContextPacker, EscalationEngine, EscalationState, GitManager, ProgressTracker, SessionManager,
    SwarmTier, TierBudget, TurnPolicy, ValidatorFeedback, Verifier, VerifierConfig, VerifierReport,
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

    // ── Coordination integrations (P2.2 + P2.3) ──
    pub router: DynamicRouter,
    pub correction_loop: TieredCorrectionLoop,

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
}

impl<'a> OrchestratorContext<'a> {
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

        // We defer worktree creation, session init, agent builds, and health
        // checks to the state handlers (handle_selecting_issue / handle_preparing_worktree).
        // For now, create placeholder values that will be set during those states.
        //
        // Actually — the legacy path does all of this upfront. For a clean
        // refactor, we mirror that: do it all here, just like the legacy path.

        // Worktree — created eagerly so agent builders have a path
        let wt_path = worktree_bridge.create(&issue.id)?;
        info!(path = %wt_path.display(), "Created worktree (state driver)");

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

        // Verifier config
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

        // Agents
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
            auto_fix_applied: false,
            factory,
            beads,
            knowledge_base,
            worktree_bridge,
            cluster_health,
            local_validator_enabled,
            max_validator_failures,
            process_span,
            process_start,
        })
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

    ctx.beads.update_status(&ctx.issue.id, "in_progress")?;
    info!(id = %ctx.issue.id, "Claimed issue (state driver)");

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

    // Build prompt
    let mut task_prompt = if tier == SwarmTier::Worker {
        format_compact_task_prompt(&packet, &ctx.wt_path)
    } else {
        format_task_prompt(&packet)
    };

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

    // Checkpoint before agent invocation
    ctx.pre_worker_commit = ctx.git_mgr.current_commit_full().ok();

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

            match route_to_coder(&recent_cats) {
                CoderRoute::RustCoder => {
                    info!(iteration, "Routing to rust_coder (state driver)");
                    ctx.metrics.record_coder_route("RustCoder");
                    ctx.metrics.record_agent_metrics("Qwen3.5-RustCoder", 0, 0);
                    let adapter = RuntimeAdapter::new(AdapterConfig {
                        agent_name: "Qwen3.5-RustCoder".into(),
                        deadline: Some(Instant::now() + ctx.worker_timeout),
                        max_tool_calls: Some(30),
                        max_turns_without_write: Some(5),
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
                        max_turns_without_write: Some(5),
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
                                Ok(Ok(response)) => {
                                    info!(iteration, model = %entry.model, "Fallback succeeded (state driver)");
                                    fallback_result = Some(Ok(response));
                                    break;
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
        Ok(r) => {
            let preview = if r.len() > 500 { &r[..500] } else { &r };
            info!(iteration, response_len = r.len(), response_preview = %preview, "Agent responded (state driver)");
            r
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

/// Verifying: run deterministic quality gates, auto-fix, regression detection.
pub async fn handle_verifying(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    let iteration = ctx.session.iteration();

    let verifier_start = Instant::now();
    let current_vc = if ctx.config.verifier_packages.is_empty() {
        VerifierConfig {
            packages: detect_changed_packages(&ctx.wt_path, true),
            ..ctx.verifier_config.clone()
        }
    } else {
        ctx.verifier_config.clone()
    };
    let verifier = Verifier::new(&ctx.wt_path, current_vc);
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

    if let Some(lm) = ctx.metrics.build_loop_metrics(report.all_green) {
        lm.emit();
    }

    // Regression detection
    let prev_error_count = ctx.last_report.as_ref().map(|r| r.failure_signals.len());
    if !report.all_green {
        ctx.consecutive_validator_failures = 0;
        if let Some(prev_count) = prev_error_count {
            if error_count > prev_count {
                warn!(
                    iteration,
                    prev_errors = prev_count,
                    curr_errors = error_count,
                    "Regression detected (state driver)"
                );
                let mut rolled_back = false;
                if let Some(ref rollback_hash) = ctx.pre_worker_commit {
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
                    let rb_vc = if ctx.config.verifier_packages.is_empty() {
                        VerifierConfig {
                            packages: detect_changed_packages(&ctx.wt_path, true),
                            ..ctx.verifier_config.clone()
                        }
                    } else {
                        ctx.verifier_config.clone()
                    };
                    let rb_verifier = Verifier::new(&ctx.wt_path, rb_vc);
                    let rb_report = rb_verifier.run_pipeline().await;
                    ctx.last_report = Some(rb_report);
                    ctx.metrics.finish_iteration();
                    return Ok(StateTransition::Advance {
                        to: OrchestratorState::Planning,
                        reason: "regression rolled back, retrying".into(),
                    });
                }
            }
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

    // Acceptance policy check
    let acceptance_result = acceptance::check_acceptance(
        &ctx.acceptance_policy,
        &ctx.wt_path,
        ctx.session.state().initial_commit.as_deref(),
        cloud_passes,
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
pub async fn handle_escalating(ctx: &mut OrchestratorContext<'_>) -> Result<StateTransition> {
    let iteration = ctx.session.iteration();
    let report = ctx
        .last_report
        .take()
        .ok_or_else(|| anyhow::anyhow!("No verifier report available in Escalating state"))?;

    // Pre-escalation KB check
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

    // Escalation decision
    let engine = EscalationEngine::new();
    let decision = engine.decide(&mut ctx.escalation, &report);
    ctx.last_report = Some(report);

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

    ctx.metrics.finish_iteration();

    if decision.stuck {
        error!(iteration, reason = %decision.reason, "Stuck (state driver)");
        create_stuck_intervention(
            &mut ctx.session,
            &ctx.progress,
            &ctx.wt_path,
            iteration,
            &decision.reason,
        );
        return Ok(StateTransition::Fail {
            reason: format!("Stuck: {}", decision.reason),
        });
    }

    // Back to Planning for next iteration
    Ok(StateTransition::Advance {
        to: OrchestratorState::Planning,
        reason: format!("escalation → {:?}", decision.target_tier),
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

    if let Err(e) = ctx.worktree_bridge.merge_and_remove(&ctx.issue.id) {
        error!(id = %ctx.issue.id, "Merge failed (state driver): {e}");
        if let Err(cleanup_err) = ctx.worktree_bridge.cleanup(&ctx.issue.id) {
            warn!(id = %ctx.issue.id, "Cleanup failed: {cleanup_err}");
        }
        let _ = ctx.beads.update_status(&ctx.issue.id, "open");
        return Err(e);
    }

    // Mark session complete only after merge succeeds
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

    ctx.beads.close(
        &ctx.issue.id,
        Some("Resolved by swarm orchestrator (state driver)"),
    )?;
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
            ctx.state_machine.fail(&reason.to_string())?;
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
                ctx.state_machine.advance(to, Some(&reason))?;
                ctx.budget_tracker.on_state_entered(to);

                // Checkpoint at stable points
                let git_hash = ctx.git_mgr.current_commit_full().ok();
                if let Some(cp) = ctx
                    .state_machine
                    .checkpoint(&ctx.issue.id, git_hash.as_deref())
                {
                    let cp_path = ctx.wt_path.join(".swarm-checkpoint.json");
                    if let Ok(json) = serde_json::to_string_pretty(&cp) {
                        let _ = std::fs::write(&cp_path, json);
                    }
                }
            }
            StateTransition::Complete => {
                ctx.state_machine
                    .advance(OrchestratorState::Merging, Some("all gates passed"))?;
                ctx.budget_tracker
                    .on_state_entered(OrchestratorState::Merging);
                // Execute merge
                let merge_result = handle_merging(ctx).await?;
                match merge_result {
                    StateTransition::Advance { to, reason } => {
                        ctx.state_machine.advance(to, Some(&reason))?;
                        ctx.success = true;
                    }
                    StateTransition::Fail { reason } => {
                        ctx.state_machine.fail(&reason)?;
                    }
                    StateTransition::Complete => {
                        ctx.state_machine
                            .advance(OrchestratorState::Resolved, Some("merged and resolved"))?;
                        ctx.success = true;
                    }
                }
                break;
            }
            StateTransition::Fail { reason } => {
                ctx.state_machine.fail(&reason)?;
                break;
            }
        }
    }

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
    let session_metrics = metrics.finalize(ctx.success, &final_tier);
    telemetry::write_session_metrics(&session_metrics, &ctx.wt_path);
    telemetry::append_telemetry(&session_metrics, ctx.worktree_bridge.repo_root());

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
}
