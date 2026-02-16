//! Orchestration loop: process a single issue through implement → verify → review → escalate.
//!
//! Integrates coordination's harness for session tracking, git checkpoints,
//! progress logging, and human intervention requests.

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use rig::completion::Prompt;
use tracing::{error, info, warn};

/// Maximum wall-clock time for a single agent prompt call.
/// Prevents runaway managers that exceed their turn limits (rig doesn't enforce
/// `default_max_turns` on the outer agent in `.prompt()` calls).
const AGENT_TIMEOUT: Duration = Duration::from_secs(10 * 60); // 10 minutes

use crate::agents::AgentFactory;
use crate::beads_bridge::{BeadsIssue, IssueTracker};
use crate::config::SwarmConfig;
use crate::worktree_bridge::WorktreeBridge;
use coordination::feedback::ErrorCategory;
use coordination::save_session_state;
use coordination::{
    ContextPacker, EscalationEngine, EscalationState, GitManager, InterventionType,
    PendingIntervention, ProgressTracker, SessionManager, SwarmTier, TierBudget, Verifier,
    VerifierConfig, VerifierReport, WorkPacket,
};

/// Coder routing decision with confidence level.
#[derive(Debug, PartialEq, Eq)]
pub enum CoderRoute {
    /// strand-14B: deep Rust expertise, fast on single-file fixes
    RustCoder,
    /// Qwen3-Coder-Next: 256K context, multi-file scaffolding
    GeneralCoder,
}

/// Route to the appropriate coder based on error category distribution.
///
/// Uses a weighted scoring system rather than a simple `any()` check:
/// - Rust-specific categories (borrow checker, lifetimes, traits) score toward strand-14B
/// - Structural categories (imports, syntax, macros) score toward Qwen3-Coder-Next
/// - Mixed errors with majority Rust → strand-14B; majority structural → general
/// - No errors (first iteration) → general coder for scaffolding
pub fn route_to_coder(error_cats: &[ErrorCategory]) -> CoderRoute {
    if error_cats.is_empty() {
        // First iteration — use general coder for scaffolding/multi-file work
        return CoderRoute::GeneralCoder;
    }

    let mut rust_score: i32 = 0;
    let mut general_score: i32 = 0;

    for cat in error_cats {
        match cat {
            // Deep Rust expertise required — strand-14B excels here
            ErrorCategory::BorrowChecker => rust_score += 3,
            ErrorCategory::Lifetime => rust_score += 3,
            ErrorCategory::TraitBound => rust_score += 2,
            ErrorCategory::Async => rust_score += 2,
            ErrorCategory::TypeMismatch => rust_score += 1,

            // Structural/multi-file work — Qwen3-Coder-Next's 256K context helps
            ErrorCategory::ImportResolution => general_score += 3,
            ErrorCategory::Macro => general_score += 2,
            ErrorCategory::Syntax => general_score += 1,

            // Ambiguous — slight general bias (may need broader context)
            ErrorCategory::Other => general_score += 1,
        }
    }

    if rust_score > general_score {
        CoderRoute::RustCoder
    } else {
        CoderRoute::GeneralCoder
    }
}

/// Format a WorkPacket into a structured prompt for agent consumption.
pub fn format_task_prompt(packet: &WorkPacket) -> String {
    let mut prompt = String::new();

    prompt.push_str(&format!("# Task: {}\n\n", packet.objective));
    prompt.push_str(&format!(
        "**Issue:** {} | **Branch:** {} | **Iteration:** {} | **Tier:** {}\n\n",
        packet.bead_id, packet.branch, packet.iteration, packet.target_tier
    ));

    if !packet.constraints.is_empty() {
        prompt.push_str("## Constraints\n");
        for c in &packet.constraints {
            prompt.push_str(&format!("- [{:?}] {}\n", c.kind, c.description));
        }
        prompt.push('\n');
    }

    if !packet.failure_signals.is_empty() {
        prompt.push_str("## Current Errors to Fix\n");
        for signal in &packet.failure_signals {
            prompt.push_str(&format!(
                "- **{}** ({}): {}\n",
                signal.category,
                signal.code.as_deref().unwrap_or("?"),
                signal.message
            ));
            if let Some(file) = &signal.file {
                prompt.push_str(&format!("  File: {}:{}\n", file, signal.line.unwrap_or(0)));
            }
        }
        prompt.push('\n');
    }

    if !packet.previous_attempts.is_empty() {
        prompt.push_str("## Previous Attempts (avoid repeating these)\n");
        for attempt in &packet.previous_attempts {
            prompt.push_str(&format!("- {attempt}\n"));
        }
        prompt.push('\n');
    }

    if !packet.file_contexts.is_empty() {
        prompt.push_str("## Relevant Files\n");
        for ctx in &packet.file_contexts {
            prompt.push_str(&format!(
                "- `{}` (lines {}-{}) — {}\n",
                ctx.file, ctx.start_line, ctx.end_line, ctx.relevance
            ));
        }
        prompt.push('\n');
        prompt
            .push_str("_Use the `read_file` tool to read these files before making changes._\n\n");
    }

    if !packet.key_symbols.is_empty() {
        prompt.push_str("## Key Symbols\n");
        for sym in &packet.key_symbols {
            prompt.push_str(&format!("- `{}` ({}) in {}", sym.name, sym.kind, sym.file));
            if let Some(line) = sym.line {
                prompt.push_str(&format!(":{line}"));
            }
            prompt.push('\n');
        }
        prompt.push('\n');
    }

    prompt.push_str(&format!(
        "**Max patch size:** {} LOC\n",
        packet.max_patch_loc
    ));

    prompt
}

/// Stage and commit all changes in the worktree.
///
/// Returns `true` if there were changes to commit, `false` if clean.
/// Uses `git add .` (not `-A`) to respect `.gitignore` and avoid staging
/// agent-generated artifacts.
pub async fn git_commit_changes(wt_path: &Path, iteration: u32) -> Result<bool> {
    // Stage changes (respects .gitignore)
    let add = tokio::process::Command::new("git")
        .args(["add", "."])
        .current_dir(wt_path)
        .output()
        .await?;
    if !add.status.success() {
        let stderr = String::from_utf8_lossy(&add.stderr);
        // Log error but don't crash — work is still in the worktree files
        error!(iteration, "git add failed (likely permissions): {stderr}");
        return Ok(false);
    }

    // Check if there are staged changes
    let status = tokio::process::Command::new("git")
        .args(["diff", "--cached", "--quiet"])
        .current_dir(wt_path)
        .output()
        .await?;

    if status.status.success() {
        // Exit code 0 means no diff — nothing to commit
        return Ok(false);
    }

    // Commit
    let msg = format!("swarm: iteration {iteration} changes");
    let commit = tokio::process::Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(wt_path)
        .output()
        .await?;
    if !commit.status.success() {
        let stderr = String::from_utf8_lossy(&commit.stderr);
        anyhow::bail!("git commit failed: {stderr}");
    }

    Ok(true)
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
pub async fn process_issue(
    config: &SwarmConfig,
    factory: &AgentFactory,
    worktree_bridge: &WorktreeBridge,
    issue: &BeadsIssue,
    beads: &dyn IssueTracker,
) -> Result<bool> {
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

    // --- Build agents scoped to this worktree ---
    let rust_coder = factory.build_rust_coder(&wt_path);
    let general_coder = factory.build_general_coder(&wt_path);
    let manager = factory.build_manager(&wt_path);

    // --- Escalation state ---
    //
    // Start at Integrator tier (cloud-backed manager) from the beginning.
    // Cloud models (Opus 4.6) are the managers; local models are workers.
    // Integrator gets 6 iterations, Cloud overflow gets 4 more (same manager).
    let engine = EscalationEngine::new();
    let mut escalation = EscalationState::new(&issue.id)
        .with_initial_tier(SwarmTier::Integrator)
        .with_budget(
            SwarmTier::Integrator,
            TierBudget {
                max_iterations: 6,
                max_consultations: 6,
            },
        )
        .with_budget(
            SwarmTier::Cloud,
            TierBudget {
                max_iterations: 4,
                max_consultations: 4,
            },
        );
    let mut success = false;
    let mut last_report: Option<VerifierReport> = None;

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

        let tier = escalation.current_tier;
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

        // Pack context with tier-appropriate token budget
        let packer = ContextPacker::new(&wt_path, tier);
        let packet = if let Some(ref report) = last_report {
            packer.pack_retry(&issue.id, &issue.title, &escalation, report)
        } else {
            packer.pack_initial(&issue.id, &issue.title)
        };

        info!(
            tokens = packet.estimated_tokens(),
            files = packet.file_contexts.len(),
            "Packed context"
        );

        let task_prompt = format_task_prompt(&packet);

        // --- Route to agent based on current tier ---
        //
        // Hierarchy (cloud available):
        //   Implementer: local coders (strand-14B, Qwen3-Coder-Next)
        //   Integrator+Cloud: cloud-backed manager (Opus 4.6) with all local workers as tools
        //
        // Hierarchy (no cloud):
        //   Implementer: local coders
        //   Integrator+Cloud: local manager (OR1-Behemoth) with coders as tools
        let agent_future = match tier {
            SwarmTier::Implementer | SwarmTier::Adversary => {
                let recent_cats: Vec<ErrorCategory> = escalation
                    .recent_error_categories
                    .last()
                    .cloned()
                    .unwrap_or_default();

                match route_to_coder(&recent_cats) {
                    CoderRoute::RustCoder => {
                        info!(iteration, "Routing to rust_coder (strand-14B)");
                        rust_coder.prompt(&task_prompt).await
                    }
                    CoderRoute::GeneralCoder => {
                        info!(iteration, "Routing to general_coder (Qwen3-Coder-Next)");
                        general_coder.prompt(&task_prompt).await
                    }
                }
            }
            SwarmTier::Integrator | SwarmTier::Cloud => {
                info!(
                    iteration,
                    "Routing to manager (cloud-backed or OR1 fallback)"
                );
                // Wrap manager call with timeout to enforce turn limits.
                // Rig doesn't enforce default_max_turns on the outer .prompt() agent,
                // so managers can run indefinitely. This hard-caps wall-clock time.
                match tokio::time::timeout(AGENT_TIMEOUT, manager.prompt(&task_prompt)).await {
                    Ok(result) => result,
                    Err(_elapsed) => {
                        warn!(
                            iteration,
                            timeout_secs = AGENT_TIMEOUT.as_secs(),
                            "Manager exceeded timeout — proceeding with changes on disk"
                        );
                        // Return a synthetic "timed out" response so the verifier still runs.
                        // Any file changes the manager made are already on disk.
                        Ok("Manager timed out. Changes are on disk for verifier.".to_string())
                    }
                }
            }
        };

        // Handle agent failure
        let _response = match agent_future {
            Ok(r) => {
                info!(iteration, response_len = r.len(), "Agent responded");
                r
            }
            Err(e) => {
                error!(iteration, "Agent failed: {e}");
                let _ = progress.log_error(
                    session.session_id(),
                    iteration,
                    format!("Agent failed: {e}"),
                );
                escalation.record_iteration(vec![], 1, false);

                let verifier = Verifier::new(&wt_path, VerifierConfig::default());
                let report = verifier.run_pipeline().await;
                let decision = engine.decide(&mut escalation, &report);
                last_report = Some(report);

                if decision.stuck {
                    error!(iteration, "Escalation engine: stuck after agent failure");
                    create_stuck_intervention(&mut session, &progress, iteration, &decision.reason);
                    break;
                }
                continue;
            }
        };

        // --- Git commit changes made by the agent ---
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

        if !has_changes {
            warn!(iteration, "No file changes after agent response");
            escalation.record_iteration(vec![], 0, false);

            let verifier = Verifier::new(&wt_path, VerifierConfig::default());
            let report = verifier.run_pipeline().await;
            let decision = engine.decide(&mut escalation, &report);
            last_report = Some(report);

            if decision.stuck {
                error!(iteration, "Escalation engine: stuck (no changes)");
                create_stuck_intervention(&mut session, &progress, iteration, &decision.reason);
                break;
            }
            continue;
        }

        // --- Auto-format before verifier ---
        // Workers don't always produce perfectly formatted code.
        // Formatting is deterministic — auto-fix it rather than failing on it.
        let fmt_output = std::process::Command::new("cargo")
            .args(["fmt", "--all"])
            .current_dir(&wt_path)
            .output();
        if let Ok(ref out) = fmt_output {
            if !out.status.success() {
                warn!(iteration, "cargo fmt failed (non-fatal): {}", String::from_utf8_lossy(&out.stderr));
            }
        }

        // --- Verifier: run deterministic quality gates ---
        let verifier = Verifier::new(&wt_path, VerifierConfig::default());
        let report = verifier.run_pipeline().await;

        info!(
            iteration,
            all_green = report.all_green,
            summary = %report.summary(),
            "Verifier report"
        );

        let error_cats = report.unique_error_categories();
        let error_count = report.failure_signals.len();

        if report.all_green {
            // Deterministic gates (fmt + clippy + check + test) are the source of truth.
            // The reviewer is advisory — don't let subjective LLM feedback cause loops.
            info!(
                iteration,
                "Verifier passed (all gates green) — accepting result"
            );
            escalation.record_iteration(error_cats, error_count, true);

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

            success = true;
            break;
        } else {
            escalation.record_iteration(error_cats, error_count, false);
        }

        // --- Escalation decision ---
        let decision = engine.decide(&mut escalation, &report);
        last_report = Some(report);

        if decision.escalated {
            info!(
                iteration,
                from = ?tier,
                to = ?decision.target_tier,
                reason = %decision.reason,
                "Tier escalated"
            );
        }

        if decision.stuck {
            error!(
                iteration,
                reason = %decision.reason,
                "Escalation engine: stuck — flagging for human intervention"
            );
            create_stuck_intervention(&mut session, &progress, iteration, &decision.reason);
            break;
        }
    }

    // --- Outcome ---
    if success {
        session.complete();
        let _ = progress.log_session_end(
            session.session_id(),
            session.iteration(),
            format!("Issue {} resolved", issue.id),
        );

        info!(
            id = %issue.id,
            session_id = session.short_id(),
            elapsed = %session.elapsed_human(),
            iterations = session.iteration(),
            "Issue resolved — merging worktree"
        );

        if let Err(e) = worktree_bridge.merge_and_remove(&issue.id) {
            error!(id = %issue.id, "Merge failed: {e} — resetting issue to open");
            let _ = beads.update_status(&issue.id, "open");
            return Err(e);
        }
        beads.close(&issue.id, Some("Resolved by swarm orchestrator"))?;
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

        // Persist session state for potential resume after SLURM preemption
        let state_path = wt_path.join(".swarm-session.json");
        if let Err(e) = save_session_state(session.state(), &state_path) {
            warn!("Failed to save session state: {e}");
        } else {
            info!(path = %state_path.display(), "Session state saved for resume");
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

    Ok(success)
}

/// Create a human intervention request when the escalation engine reports stuck.
///
/// Records the intervention in the session state and logs it to the progress file.
fn create_stuck_intervention(
    session: &mut SessionManager,
    progress: &ProgressTracker,
    iteration: u32,
    reason: &str,
) {
    let intervention = PendingIntervention::new(
        InterventionType::ReviewRequired,
        format!(
            "Stuck after iteration {}: {}. Manual review needed.",
            iteration, reason
        ),
    )
    .with_feature(session.current_feature().unwrap_or("unknown"));

    session.state_mut().add_intervention(intervention);

    let _ = progress.log_error(
        session.session_id(),
        iteration,
        format!("Stuck — human intervention requested: {reason}"),
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use coordination::{SessionStatus, SwarmTier};
    use std::path::PathBuf;

    fn test_packet(objective: &str, branch: &str, iteration: u32) -> WorkPacket {
        WorkPacket {
            bead_id: "test-bead".into(),
            branch: branch.into(),
            checkpoint: "abc123".into(),
            objective: objective.into(),
            files_touched: vec![],
            key_symbols: vec![],
            file_contexts: vec![],
            verification_gates: vec![],
            failure_signals: vec![],
            constraints: vec![],
            iteration,
            target_tier: SwarmTier::Implementer,
            escalation_reason: None,
            error_history: vec![],
            previous_attempts: vec![],
            relevant_heuristics: vec![],
            relevant_playbooks: vec![],
            decisions: vec![],
            generated_at: Utc::now(),
            max_patch_loc: 200,
        }
    }

    #[test]
    fn test_route_empty_errors_to_general() {
        assert_eq!(route_to_coder(&[]), CoderRoute::GeneralCoder);
    }

    #[test]
    fn test_route_borrow_checker_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::BorrowChecker]),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_lifetime_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Lifetime]),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_trait_bound_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::TraitBound]),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_async_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Async]),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_import_to_general() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::ImportResolution]),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_route_macro_to_general() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Macro]),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_route_syntax_to_general() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Syntax]),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_route_other_to_general() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Other]),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_route_type_mismatch_alone_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::TypeMismatch]),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_mixed_rust_heavy() {
        // BorrowChecker(+3) + Import(+3) → tie → general wins (>= check)
        assert_eq!(
            route_to_coder(&[
                ErrorCategory::BorrowChecker,
                ErrorCategory::ImportResolution
            ]),
            CoderRoute::GeneralCoder
        );
        // BorrowChecker(+3) + Lifetime(+3) + Import(+3) → 6 vs 3 → rust
        assert_eq!(
            route_to_coder(&[
                ErrorCategory::BorrowChecker,
                ErrorCategory::Lifetime,
                ErrorCategory::ImportResolution
            ]),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_format_task_prompt_basic() {
        let packet = test_packet("Fix type error", "swarm/test-1", 1);
        let prompt = format_task_prompt(&packet);
        assert!(prompt.contains("Fix type error"));
        assert!(prompt.contains("swarm/test-1"));
        assert!(prompt.contains("200 LOC"));
    }

    // ========================================================================
    // Harness integration tests
    // ========================================================================

    #[test]
    fn test_create_stuck_intervention_adds_to_session() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = SessionManager::new(dir.path().to_path_buf(), 10);
        session.start().unwrap();
        session.set_current_feature("test-issue-001");

        let progress = ProgressTracker::new(dir.path().join("progress.txt"));

        create_stuck_intervention(&mut session, &progress, 3, "repeated errors");

        // Intervention should be recorded in session state
        let interventions = session.state().unresolved_interventions();
        assert_eq!(interventions.len(), 1);
        assert!(interventions[0].question.contains("iteration 3"));
        assert!(interventions[0].question.contains("repeated errors"));
        assert_eq!(
            interventions[0].feature_id.as_deref(),
            Some("test-issue-001")
        );

        // Progress file should have the error logged
        let entries = progress.read_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].summary.contains("human intervention"));
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

    #[test]
    fn test_git_manager_checkpoint_prefix() {
        // Verify GitManager uses the expected commit prefix
        let dir = tempfile::tempdir().unwrap();

        // Initialize git repo
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::fs::write(dir.path().join("README.md"), "# test").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

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
}
