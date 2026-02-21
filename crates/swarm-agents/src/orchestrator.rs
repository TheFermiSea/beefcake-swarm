//! Orchestration loop: process a single issue through implement → verify → review → escalate.
//!
//! Integrates coordination's harness for session tracking, git checkpoints,
//! progress logging, and human intervention requests.

use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use rig::completion::Prompt;
use rig::providers::openai;
use tracing::{error, info, warn};

/// Default timeout for each cloud validation call.
const DEFAULT_VALIDATION_TIMEOUT_SECS: u64 = 120; // 2 minutes

use crate::acceptance::{self, AcceptancePolicy};
use crate::agents::reviewer::{self, ReviewResult};
use crate::agents::AgentFactory;
use crate::beads_bridge::{BeadsIssue, IssueTracker};
use crate::config::SwarmConfig;
use crate::knowledge_sync;
use crate::notebook_bridge::KnowledgeBase;
use crate::telemetry::{self, MetricsCollector};
use crate::worktree_bridge::WorktreeBridge;
use coordination::feedback::ErrorCategory;
use coordination::save_session_state;
use coordination::{
    ContextPacker, EscalationEngine, EscalationState, GitManager, InterventionType,
    PendingIntervention, ProgressTracker, SessionManager, SwarmTier, TierBudget, TurnPolicy,
    Verifier, VerifierConfig, VerifierReport, WorkPacket,
};

/// Coder routing decision with confidence level.
#[derive(Debug, PartialEq, Eq)]
pub enum CoderRoute {
    /// strand-14B: deep Rust expertise, fast on single-file fixes
    RustCoder,
    /// Qwen3-Coder-Next: 256K context, multi-file scaffolding
    GeneralCoder,
}

/// Query the knowledge base with graceful degradation on failure.
///
/// Wraps `KnowledgeBase::query` with error handling so that any KB failure
/// (connection error, auth failure, or a hanging `nlm` CLI subprocess) returns
/// an empty string instead of propagating an error. This ensures KB
/// unavailability never blocks the orchestration loop.
fn query_kb_with_failsafe(kb: &dyn KnowledgeBase, role: &str, question: &str) -> String {
    match kb.query(role, question) {
        Ok(response) => response,
        Err(e) => {
            warn!(role, error = %e, "KB query failed — proceeding without context");
            String::new()
        }
    }
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

    // Knowledge layer fields (populated by NotebookBridge)
    if !packet.relevant_heuristics.is_empty() {
        prompt.push_str("## Knowledge Base Context\n");
        for h in &packet.relevant_heuristics {
            prompt.push_str(&format!("{h}\n"));
        }
        prompt.push('\n');
    }

    if !packet.relevant_playbooks.is_empty() {
        prompt.push_str("## Known Fix Patterns\n");
        for p in &packet.relevant_playbooks {
            prompt.push_str(&format!("{p}\n"));
        }
        prompt.push('\n');
    }

    if !packet.decisions.is_empty() {
        prompt.push_str("## Relevant Decisions\n");
        for d in &packet.decisions {
            prompt.push_str(&format!("- {d}\n"));
        }
        prompt.push('\n');
    }

    prompt.push_str(&format!(
        "**Max patch size:** {} LOC\n",
        packet.max_patch_loc
    ));

    prompt
}

/// Try to auto-fix trivial verifier failures without LLM delegation.
///
/// Runs `cargo clippy --fix` for MachineApplicable suggestions and `cargo fmt`
/// to resolve formatting issues. If fixes are applied, re-runs the verifier
/// and returns the new report if it's now green.
///
/// This is the "Janitor" layer: handle mechanical fixes before involving expensive models.
pub async fn try_auto_fix(
    wt_path: &Path,
    verifier_config: &VerifierConfig,
    iteration: u32,
) -> Option<VerifierReport> {
    // Build package args for scoped commands
    let mut pkg_args: Vec<&str> = Vec::new();
    for pkg in &verifier_config.packages {
        pkg_args.push("-p");
        pkg_args.push(pkg);
    }

    // Step 1: Try cargo clippy --fix for MachineApplicable suggestions
    let mut clippy_args = vec!["clippy", "--fix", "--allow-dirty", "--allow-staged"];
    clippy_args.extend_from_slice(&pkg_args);
    clippy_args.extend_from_slice(&["--", "-D", "warnings"]);

    let clippy_fix = tokio::process::Command::new("cargo")
        .args(&clippy_args)
        .current_dir(wt_path)
        .output()
        .await;

    let clippy_fixed = match clippy_fix {
        Ok(ref out) if out.status.success() => {
            info!(iteration, "auto-fix: cargo clippy --fix succeeded");
            true
        }
        Ok(ref out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            // clippy --fix often "fails" because it can't fix everything — that's OK.
            // It still applies what it can.
            warn!(
                iteration,
                "auto-fix: cargo clippy --fix partial: {}",
                &stderr[..stderr.len().min(200)]
            );
            true // Still worth re-checking
        }
        Err(e) => {
            warn!(iteration, "auto-fix: cargo clippy --fix failed to run: {e}");
            false
        }
    };

    // Step 2: Run cargo fmt to fix formatting
    let mut fmt_args = vec!["fmt"];
    for pkg in &verifier_config.packages {
        fmt_args.push("--package");
        fmt_args.push(pkg);
    }

    let fmt_fix = tokio::process::Command::new("cargo")
        .args(&fmt_args)
        .current_dir(wt_path)
        .output()
        .await;

    let fmt_fixed = match fmt_fix {
        Ok(ref out) if out.status.success() => {
            info!(iteration, "auto-fix: cargo fmt succeeded");
            true
        }
        Ok(ref out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            warn!(
                iteration,
                "auto-fix: cargo fmt failed (syntax error?): {}",
                &stderr[..stderr.len().min(200)]
            );
            false
        }
        Err(e) => {
            warn!(iteration, "auto-fix: cargo fmt failed to run: {e}");
            false
        }
    };

    if !clippy_fixed && !fmt_fixed {
        return None; // Nothing was attempted
    }

    // Check if there are actual changes to commit
    let status = tokio::process::Command::new("git")
        .args(["diff", "--quiet"])
        .current_dir(wt_path)
        .output()
        .await;

    let has_changes = matches!(status, Ok(ref out) if !out.status.success());
    if !has_changes {
        info!(iteration, "auto-fix: no changes produced");
        return None;
    }

    // Commit auto-fix changes
    let _ = tokio::process::Command::new("git")
        .args(["add", "."])
        .current_dir(wt_path)
        .output()
        .await;

    let msg = format!("swarm: auto-fix iteration {iteration} (clippy --fix + fmt)");
    let _ = tokio::process::Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(wt_path)
        .output()
        .await;

    info!(
        iteration,
        "auto-fix: committed changes, re-running verifier"
    );

    // Re-run the full verifier pipeline
    let verifier = Verifier::new(wt_path, verifier_config.clone());
    let report = verifier.run_pipeline().await;

    if report.all_green {
        info!(
            iteration,
            summary = %report.summary(),
            "auto-fix: verifier now passes! Skipping LLM delegation"
        );
        Some(report)
    } else {
        info!(
            iteration,
            summary = %report.summary(),
            "auto-fix: verifier still failing after auto-fix"
        );
        // Return the updated report so the next iteration uses it
        Some(report)
    }
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
        anyhow::bail!("git add failed (iteration {iteration}): {stderr}");
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

/// Result of a single cloud model validation.
struct CloudValidationResult {
    model: String,
    passed: bool,
    feedback: String,
}

fn timeout_from_env(var: &str, default_secs: u64) -> Duration {
    let secs = std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default_secs);
    Duration::from_secs(secs)
}

fn u32_from_env(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

fn bool_from_env(var: &str, default: bool) -> bool {
    std::env::var(var)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

/// Count lines changed between two commits in the worktree.
fn count_diff_lines(wt_path: &Path, from: &str, to: &str) -> usize {
    let output = std::process::Command::new("git")
        .args(["diff", "--numstat", from, to])
        .current_dir(wt_path)
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.lines().fold(0, |acc, line| {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 2 {
                    let added: usize = parts[0].parse().unwrap_or(0);
                    let removed: usize = parts[1].parse().unwrap_or(0);
                    acc + added + removed
                } else {
                    acc
                }
            })
        }
        _ => 0,
    }
}

/// Collect artifact records from the git diff between two commits.
///
/// Parses `git diff --numstat` to determine which files were added, modified,
/// or deleted. Files that existed before (`from`) and after (`to`) are
/// `Modified`; files only in `to` are `Created`; files only in `from` are
/// `Deleted`. The `size_delta` is approximated as `(added - removed)` lines
/// (a line-count proxy; byte-level deltas would require `--stat`).
fn collect_artifacts_from_diff(
    wt_path: &Path,
    from: &str,
    to: &str,
) -> Vec<telemetry::ArtifactRecord> {
    let output = std::process::Command::new("git")
        .args(["diff", "--numstat", from, to])
        .current_dir(wt_path)
        .output();

    let stdout = match output {
        Ok(ref out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
        _ => return Vec::new(),
    };

    stdout
        .lines()
        .filter_map(|line| {
            // numstat format: "<added>\t<removed>\t<path>"
            // Binary files show "-\t-\t<path>"
            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            if parts.len() < 3 {
                return None;
            }
            let added: i64 = parts[0].parse().unwrap_or(0);
            let removed: i64 = parts[1].parse().unwrap_or(0);
            let path = parts[2].trim().to_string();

            let action = if added > 0 && removed == 0 {
                telemetry::ArtifactAction::Created
            } else if added == 0 && removed > 0 {
                telemetry::ArtifactAction::Deleted
            } else {
                telemetry::ArtifactAction::Modified
            };

            Some(telemetry::ArtifactRecord {
                path,
                action,
                line_range: None,
                size_delta: Some(added - removed),
            })
        })
        .collect()
}

/// Returns `true` when the auto-fix false-positive guard should apply.
///
/// The guard fires only when auto-fix actually ran this iteration AND a minimum
/// agent diff size is configured. This prevents rejecting legitimate small fixes
/// that pass the verifier on their own merit (i.e. without auto-fix).
fn should_reject_auto_fix(auto_fix_applied: bool, policy: &AcceptancePolicy) -> bool {
    auto_fix_applied && policy.min_diff_lines > 0
}

fn tier_from_env(var: &str, default: SwarmTier) -> SwarmTier {
    match std::env::var(var)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("worker") => SwarmTier::Worker,
        Some("human") => SwarmTier::Human,
        Some("council") => SwarmTier::Council,
        _ => default,
    }
}

/// Run cloud validation on the worktree diff using external high-end models.
///
/// Sends the git diff (since initial commit) to each configured cloud validator
/// model for blind PASS/FAIL review. This is **advisory** — the orchestrator
/// logs results but doesn't block on FAIL to avoid subjective LLM feedback loops.
///
/// Validator models are configured via env vars:
/// - `SWARM_VALIDATOR_MODEL_1` (default: `gemini-3-pro-preview`)
/// - `SWARM_VALIDATOR_MODEL_2` (default: `claude-sonnet-4-5-20250929`)
async fn cloud_validate(
    cloud_client: &openai::CompletionsClient,
    wt_path: &Path,
    initial_commit: &str,
) -> Vec<CloudValidationResult> {
    // Get the full diff since the initial commit
    let diff = match std::process::Command::new("git")
        .args(["diff", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).to_string()
        }
        Ok(output) => {
            warn!(
                "git diff failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            return Vec::new();
        }
        Err(e) => {
            warn!("Failed to run git diff: {e}");
            return Vec::new();
        }
    };

    if diff.trim().is_empty() {
        info!("No diff to validate — skipping cloud validation");
        return Vec::new();
    }

    // Truncate very large diffs to avoid token limits
    let max_diff_chars = 32_000;
    let diff_for_review = if diff.len() > max_diff_chars {
        format!(
            "{}\n\n... [truncated — {} total chars, showing first {}]",
            &diff[..max_diff_chars],
            diff.len(),
            max_diff_chars,
        )
    } else {
        diff
    };

    let models = [
        std::env::var("SWARM_VALIDATOR_MODEL_1").unwrap_or_else(|_| "gemini-3-pro-preview".into()),
        std::env::var("SWARM_VALIDATOR_MODEL_2")
            .unwrap_or_else(|_| "claude-sonnet-4-5-20250929".into()),
    ];

    let review_prompt = format!(
        "You are reviewing a Rust code change from an autonomous coding agent. \
         The change has already passed all deterministic gates (cargo fmt, clippy, \
         cargo check, cargo test). Your job is to catch logic errors, edge cases, \
         and design issues that the compiler cannot detect.\n\n\
         Respond with STRICT JSON ONLY using schema: \
         {{\"verdict\":\"pass|fail|needs_escalation\",\"confidence\":0.0-1.0,\
         \"blocking_issues\":[...],\"suggested_next_action\":\"...\",\
         \"touched_files\":[...]}}.\n\n\
         ```diff\n{diff_for_review}\n```"
    );
    let validation_timeout = timeout_from_env(
        "SWARM_VALIDATION_TIMEOUT_SECS",
        DEFAULT_VALIDATION_TIMEOUT_SECS,
    );

    let mut results = Vec::new();
    for model in &models {
        info!(model, "Running cloud validation");
        let validator = reviewer::build_reviewer(cloud_client, model);
        match tokio::time::timeout(
            validation_timeout,
            prompt_with_retry(&validator, &review_prompt, 3),
        )
        .await
        {
            Ok(Ok(response)) => {
                let review = ReviewResult::parse(&response);
                if !review.schema_valid {
                    warn!(
                        model,
                        "Cloud validation response was invalid schema; treating as FAIL"
                    );
                }
                let status = if review.passed { "PASS" } else { "FAIL" };
                info!(model, status, "Cloud validation complete");
                results.push(CloudValidationResult {
                    model: model.clone(),
                    passed: review.passed,
                    feedback: review.feedback,
                });
            }
            Ok(Err(e)) => {
                warn!(model, "Cloud validation error: {e}");
            }
            Err(_) => {
                warn!(
                    model,
                    "Cloud validation timed out ({}s)",
                    validation_timeout.as_secs()
                );
            }
        }
    }

    results
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
    knowledge_base: Option<&dyn KnowledgeBase>,
) -> Result<bool> {
    let worker_policy = TurnPolicy::for_tier(SwarmTier::Worker);
    let council_policy = TurnPolicy::for_tier(SwarmTier::Council);
    let worker_timeout = timeout_from_env("SWARM_WORKER_TIMEOUT_SECS", worker_policy.timeout_secs);
    let manager_timeout =
        timeout_from_env("SWARM_MANAGER_TIMEOUT_SECS", council_policy.timeout_secs);

    // Root span for the entire issue processing session (OpenTelemetry-compatible)
    let process_span = tracing::info_span!(
        "swarm.process_issue",
        "issue.id" = %issue.id,
        "issue.title" = %issue.title,
        "session.id" = tracing::field::Empty,
        "success" = tracing::field::Empty,
        "total_iterations" = tracing::field::Empty,
    );
    let _process_guard = process_span.enter();

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
    process_span.record("session.id", session.session_id());

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
    let mut metrics = MetricsCollector::new(session.session_id(), &issue.id, &issue.title);

    // --- Acceptance policy ---
    let acceptance_policy = AcceptancePolicy::default();

    // --- Build agents scoped to this worktree ---
    let rust_coder = factory.build_rust_coder(&wt_path);
    let general_coder = factory.build_general_coder(&wt_path);
    let manager = factory.build_manager(&wt_path);

    // --- Escalation state ---
    //
    // Start at Council tier (cloud-backed manager) from the beginning.
    // Cloud models (Opus 4.6) are the managers; local models are workers.
    // Council gets 6 iterations before escalating to Human.
    let council_budget_iterations = u32_from_env("SWARM_COUNCIL_MAX_ITERATIONS", 6);
    let council_budget_consultations = u32_from_env("SWARM_COUNCIL_MAX_CONSULTATIONS", 6);
    let initial_tier = tier_from_env("SWARM_INITIAL_TIER", SwarmTier::Council);
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

    // Scope verifier to configured packages.
    // When empty, targets the entire workspace (useful for external repos).
    // When set (e.g., ["swarm-agents"]), limits clippy/check/test to those packages.
    let verifier_config = VerifierConfig {
        packages: config.verifier_packages.clone(),
        check_clippy: !bool_from_env("SWARM_SKIP_CLIPPY", false),
        check_test: !bool_from_env("SWARM_SKIP_TESTS", false),
        ..VerifierConfig::default()
    };

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
        metrics.start_iteration(iteration, &format!("{tier:?}"));
        let iter_span = tracing::info_span!(
            "swarm.iteration",
            iteration = iteration,
            "tier" = ?tier,
            "issue.id" = %issue.id,
            "all_green" = tracing::field::Empty,
            "error_count" = tracing::field::Empty,
            "agent_ms" = tracing::field::Empty,
            "verifier_ms" = tracing::field::Empty,
        );
        let _iter_guard = iter_span.enter();
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
        let mut packet = if let Some(ref report) = last_report {
            packer.pack_retry(&issue.id, &issue.title, &escalation, report)
        } else {
            packer.pack_initial(&issue.id, &issue.title)
        };

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

        let mut task_prompt = format_task_prompt(&packet);

        // Inject verifier stderr into prompt when failure_signals are thin.
        // Fmt errors don't produce ParsedErrors, so the packet may lack error details.
        // The raw stderr contains the actual error output the model needs to see.
        if let Some(ref report) = last_report {
            if !report.all_green && packet.failure_signals.is_empty() {
                task_prompt.push_str("\n## Verifier Output (raw)\n");
                for gate in &report.gates {
                    if let Some(stderr) = &gate.stderr_excerpt {
                        task_prompt.push_str(&format!(
                            "### {} gate ({})\n```\n{}\n```\n\n",
                            gate.gate, gate.outcome, stderr
                        ));
                    }
                }
            }
        }

        // --- Checkpoint before agent invocation ---
        // Save the current commit so we can rollback if the agent makes things worse.
        let pre_worker_commit = git_mgr.current_commit_full().ok();
        let prev_error_count = last_report.as_ref().map(|r| r.failure_signals.len());

        // --- Route to agent based on current tier ---
        //
        // Hierarchy (cloud available):
        //   Worker: local coders (strand-14B, Qwen3-Coder-Next)
        //   Council+Human: cloud-backed manager (Opus 4.6) with all local workers as tools
        //
        // Hierarchy (no cloud):
        //   Worker: local coders
        //   Council+Human: local manager (OR1-Behemoth) with coders as tools
        let agent_start = std::time::Instant::now();
        let agent_future = match tier {
            SwarmTier::Worker => {
                let recent_cats: Vec<ErrorCategory> = escalation
                    .recent_error_categories
                    .last()
                    .cloned()
                    .unwrap_or_default();

                match route_to_coder(&recent_cats) {
                    CoderRoute::RustCoder => {
                        info!(iteration, "Routing to rust_coder (strand-14B)");
                        metrics.record_coder_route("RustCoder");
                        metrics.record_agent_metrics("strand-14B", 0, 0);
                        match tokio::time::timeout(worker_timeout, rust_coder.prompt(&task_prompt))
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
                        }
                    }
                    CoderRoute::GeneralCoder => {
                        info!(iteration, "Routing to general_coder (Qwen3-Coder-Next)");
                        metrics.record_coder_route("GeneralCoder");
                        metrics.record_agent_metrics("Qwen3-Coder-Next", 0, 0);
                        match tokio::time::timeout(
                            worker_timeout,
                            general_coder.prompt(&task_prompt),
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
                        }
                    }
                }
            }
            SwarmTier::Council | SwarmTier::Human => {
                info!(
                    iteration,
                    "Routing to manager (cloud-backed or OR1 fallback)"
                );
                metrics.record_agent_metrics("manager", 0, 0);
                // Wrap manager call with timeout to enforce turn limits.
                // Rig doesn't enforce default_max_turns on the outer .prompt() agent,
                // so managers can run indefinitely. This hard-caps wall-clock time.
                match tokio::time::timeout(
                    manager_timeout,
                    prompt_with_retry(&manager, &task_prompt, config.cloud_max_retries),
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
                }
            }
        };

        // Handle agent failure
        let agent_elapsed = agent_start.elapsed();
        metrics.record_agent_time(agent_elapsed);
        iter_span.record("agent_ms", agent_elapsed.as_millis() as u64);
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
                let verifier = Verifier::new(&wt_path, verifier_config.clone());
                let report = verifier.run_pipeline().await;
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

        if !has_changes {
            escalation.record_no_change();
            metrics.record_no_change();
            warn!(
                iteration,
                response_len = response.len(),
                consecutive_no_change = escalation.consecutive_no_change,
                threshold = config.max_consecutive_no_change,
                "No file changes after agent response — manager may not have called workers"
            );

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

            // engine.decide() records the iteration internally — don't double-count
            let verifier = Verifier::new(&wt_path, verifier_config.clone());
            let report = verifier.run_pipeline().await;
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
            if matches!(tier, SwarmTier::Council | SwarmTier::Human) {
                warn!(
                    iteration,
                    "No-change council response; forcing next iteration through Worker tier"
                );
            } else {
                warn!(
                    iteration,
                    "No-change worker response; keeping next iteration on Worker tier"
                );
            }
            escalation.current_tier = SwarmTier::Worker;
            continue;
        }

        // Reset no-change counter on any iteration that produces changes
        escalation.reset_no_change();

        // --- Verifier: run deterministic quality gates ---
        let verifier_start = std::time::Instant::now();
        let verifier = Verifier::new(&wt_path, verifier_config.clone());
        let mut report = verifier.run_pipeline().await;
        let verifier_elapsed = verifier_start.elapsed();
        metrics.record_verifier_time(verifier_elapsed);
        iter_span.record("verifier_ms", verifier_elapsed.as_millis() as u64);
        iter_span.record("all_green", report.all_green);
        iter_span.record("error_count", report.failure_signals.len() as u64);

        info!(
            iteration,
            all_green = report.all_green,
            summary = %report.summary(),
            "Verifier report"
        );

        // --- Auto-fix: try to resolve trivial failures without LLM delegation ---
        let mut auto_fix_applied = false;
        if !report.all_green {
            if let Some(fixed_report) = try_auto_fix(&wt_path, &verifier_config, iteration).await {
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

        // --- Regression detection: rollback if errors increased ---
        if !report.all_green {
            if let Some(prev_count) = prev_error_count {
                if error_count > prev_count {
                    warn!(
                        iteration,
                        prev_errors = prev_count,
                        curr_errors = error_count,
                        "Regression detected — errors increased after agent changes"
                    );
                    let mut rolled_back = false;
                    if let Some(ref rollback_hash) = pre_worker_commit {
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
                                        "Regression: {prev_count} → {error_count} errors. Rolled back to {rollback_hash}"
                                    ),
                                );
                            }
                            Err(e) => {
                                error!(iteration, "Rollback failed: {e}");
                            }
                        }
                    }
                    metrics.record_regression(rolled_back);
                    if rolled_back {
                        // Worktree is now at the pre-worker state — the current `report`
                        // reflects the regressed (post-worker) state. Continue to the next
                        // iteration so the verifier re-runs against the rolled-back code.
                        metrics.finish_iteration();
                        continue;
                    }
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

            // Deterministic gates (fmt + clippy + check + test) are the source of truth.
            // The reviewer is advisory — don't let subjective LLM feedback cause loops.
            info!(
                iteration,
                "Verifier passed (all gates green) — checking acceptance"
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

            // --- Cloud validation (advisory) ---
            // After deterministic gates pass, send the diff to high-end cloud models
            // (G3 Pro + Sonnet 4.5) for logic/design review. Results are logged but
            // don't block acceptance — avoids subjective LLM feedback loops.
            let mut cloud_passes = 0usize;
            if let Some(ref cloud_client) = factory.clients.cloud {
                if let Some(ref initial_commit) = session.state().initial_commit {
                    let validations = cloud_validate(cloud_client, &wt_path, initial_commit).await;
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
                        }
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

    // --- Write telemetry ---
    let final_tier = format!("{:?}", escalation.current_tier);
    let session_metrics = metrics.finalize(success, &final_tier);
    telemetry::write_session_metrics(&session_metrics, &wt_path);
    telemetry::append_telemetry(&session_metrics, worktree_bridge.repo_root());

    // Record final outcome on the root span
    process_span.record("success", success);
    process_span.record("total_iterations", session_metrics.total_iterations as u64);

    Ok(success)
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
fn clear_resume_file(repo_root: &Path) {
    let resume_path = repo_root.join(".swarm-resume.json");
    if resume_path.exists() {
        let _ = std::fs::remove_file(&resume_path);
    }
}

/// Prompt an agent with exponential backoff retry for transient HTTP errors.
///
/// Retries on connection errors, 502, 503, 429 with backoff: 2s, 4s, 8s, ...
/// Non-transient errors fail immediately.
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
                let err_str = format!("{e}");
                let err_lower = err_str.to_ascii_lowercase();
                let is_transient = err_str.contains("502")
                    || err_str.contains("503")
                    || err_str.contains("429")
                    || err_lower.contains("connection")
                    || err_lower.contains("timed out")
                    || err_lower.contains("timeout")
                    // Proxy occasionally returns empty-but-200 payloads; retry recovers.
                    || err_lower.contains("no message or tool call (empty)")
                    || err_lower.contains("response contained no message or tool call")
                    // Proxy/model schema mismatches can be intermittent during model churn.
                    || err_lower.contains("jsonerror");

                if !is_transient || attempt == max_retries {
                    return Err(e);
                }

                let backoff = Duration::from_secs(2u64.pow(attempt + 1));
                warn!(
                    attempt = attempt + 1,
                    max_retries,
                    backoff_secs = backoff.as_secs(),
                    error = %err_str,
                    "Transient error — retrying"
                );
                last_err = Some(e);
                tokio::time::sleep(backoff).await;
            }
        }
    }
    Err(last_err.unwrap())
}

/// Detect whether an issue is doc-oriented based on title/description keywords.
fn is_doc_task(title: &str, description: &str) -> bool {
    let combined = format!("{} {}", title, description).to_ascii_lowercase();
    let doc_keywords = [
        ".md",
        "rfc",
        "doc",
        "architecture",
        "planning",
        "readme",
        "design doc",
    ];
    doc_keywords.iter().any(|kw| combined.contains(kw))
}

/// Generate a minimal markdown scaffold for a doc-oriented task.
///
/// When doc tasks hit the no-change circuit breaker, this creates a template
/// file so at least a skeleton exists for human completion. Returns `true`
/// if a scaffold was committed.
pub fn try_scaffold_fallback(
    wt_path: &Path,
    issue_id: &str,
    issue_title: &str,
    issue_description: &str,
    iteration: u32,
) -> bool {
    if !is_doc_task(issue_title, issue_description) {
        return false;
    }

    // Generate a safe filename from the issue title
    let safe_name: String = issue_title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .to_ascii_lowercase();
    let filename = format!("docs/{}.md", safe_name.trim_matches('-'));

    let scaffold = format!(
        "# {title}\n\n\
         > Auto-generated scaffold by swarm orchestrator.\n\
         > Issue: `{id}` | Generated at iteration {iter}\n\n\
         ## Overview\n\n\
         <!-- TODO: Describe the purpose and scope -->\n\n\
         ## Details\n\n\
         <!-- TODO: Fill in the content -->\n\n\
         ## Open Questions\n\n\
         <!-- TODO: List any unresolved questions -->\n",
        title = issue_title,
        id = issue_id,
        iter = iteration,
    );

    // Ensure docs/ directory exists
    let docs_dir = wt_path.join("docs");
    if let Err(e) = std::fs::create_dir_all(&docs_dir) {
        warn!("Failed to create docs dir for scaffold: {e}");
        return false;
    }

    let file_path = wt_path.join(&filename);
    if let Err(e) = std::fs::write(&file_path, &scaffold) {
        warn!("Failed to write scaffold file: {e}");
        return false;
    }

    // Stage and commit the scaffold
    let add = std::process::Command::new("git")
        .args(["add", &filename])
        .current_dir(wt_path)
        .output();
    if !matches!(add, Ok(ref out) if out.status.success()) {
        warn!("Failed to git add scaffold");
        return false;
    }

    let msg = format!("swarm: scaffold fallback for {issue_id} (iteration {iteration})");
    let commit = std::process::Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(wt_path)
        .output();
    if !matches!(commit, Ok(ref out) if out.status.success()) {
        warn!("Failed to commit scaffold");
        return false;
    }

    info!(
        issue_id,
        filename, "Scaffold fallback committed for doc task"
    );
    true
}

/// Create a human intervention request when the escalation engine reports stuck.
///
/// Surfaces the intervention through 3 mechanisms:
/// 1. Records in session state (in-memory)
/// 2. Writes `.swarm-interventions.json` in the worktree root
/// 3. POSTs to `SWARM_WEBHOOK_URL` if configured
fn create_stuck_intervention(
    session: &mut SessionManager,
    progress: &ProgressTracker,
    wt_path: &Path,
    iteration: u32,
    reason: &str,
) {
    let feature_id = session.current_feature().unwrap_or("unknown").to_string();

    let intervention = PendingIntervention::new(
        InterventionType::ReviewRequired,
        format!("Stuck after iteration {iteration}: {reason}. Manual review needed."),
    )
    .with_feature(&feature_id);

    session.state_mut().add_intervention(intervention);

    let _ = progress.log_error(
        session.session_id(),
        iteration,
        format!("Stuck — human intervention requested: {reason}"),
    );

    // --- Mechanism 2: Write intervention JSON to worktree ---
    let intervention_data = serde_json::json!({
        "session_id": session.session_id(),
        "feature_id": feature_id,
        "iteration": iteration,
        "reason": reason,
        "type": "review_required",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    let intervention_path = wt_path.join(".swarm-interventions.json");
    match std::fs::write(
        &intervention_path,
        serde_json::to_string_pretty(&intervention_data).unwrap_or_default(),
    ) {
        Ok(()) => info!(path = %intervention_path.display(), "Wrote intervention file"),
        Err(e) => warn!("Failed to write intervention file: {e}"),
    }

    // --- Mechanism 3: Webhook notification ---
    if let Ok(webhook_url) = std::env::var("SWARM_WEBHOOK_URL") {
        // Fire-and-forget — don't block the orchestrator on webhook delivery
        let payload = intervention_data.clone();
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            match client
                .post(&webhook_url)
                .json(&payload)
                .timeout(Duration::from_secs(10))
                .send()
                .await
            {
                Ok(resp) => info!(status = %resp.status(), "Webhook notification sent"),
                Err(e) => warn!("Webhook notification failed: {e}"),
            }
        });
    }
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
            target_tier: SwarmTier::Worker,
            escalation_reason: None,
            error_history: vec![],
            previous_attempts: vec![],
            relevant_heuristics: vec![],
            relevant_playbooks: vec![],
            decisions: vec![],
            generated_at: Utc::now(),
            max_patch_loc: 200,
            delegation_chain: vec![],
        }
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

        create_stuck_intervention(&mut session, &progress, dir.path(), 3, "repeated errors");

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

    /// Verify that `query_kb_with_failsafe` returns an empty string when the
    /// KB fails, rather than propagating the error.
    #[test]
    fn test_query_kb_with_failsafe_on_failure() {
        use crate::notebook_bridge::KnowledgeBase;
        use anyhow::Result;

        struct FailingKb;
        impl KnowledgeBase for FailingKb {
            fn query(&self, _role: &str, _question: &str) -> Result<String> {
                anyhow::bail!("simulated nlm connection failure")
            }
            fn add_source_text(&self, _role: &str, _title: &str, _content: &str) -> Result<()> {
                Ok(())
            }
            fn add_source_file(&self, _role: &str, _file_path: &str) -> Result<()> {
                Ok(())
            }
            fn is_available(&self) -> bool {
                false
            }
        }

        let kb = FailingKb;
        // Must not panic or propagate error — returns empty string
        let result = query_kb_with_failsafe(&kb, "project_brain", "What is the architecture?");
        assert_eq!(result, "", "failsafe must return empty string on KB error");
    }

    /// Verify that `query_kb_with_failsafe` returns the response when the KB succeeds.
    #[test]
    fn test_query_kb_with_failsafe_on_success() {
        use crate::notebook_bridge::KnowledgeBase;
        use anyhow::Result;

        struct SucceedingKb;
        impl KnowledgeBase for SucceedingKb {
            fn query(&self, _role: &str, _question: &str) -> Result<String> {
                Ok("The architecture uses a 4-tier escalation ladder.".to_string())
            }
            fn add_source_text(&self, _role: &str, _title: &str, _content: &str) -> Result<()> {
                Ok(())
            }
            fn add_source_file(&self, _role: &str, _file_path: &str) -> Result<()> {
                Ok(())
            }
            fn is_available(&self) -> bool {
                true
            }
        }

        let kb = SucceedingKb;
        let result = query_kb_with_failsafe(&kb, "project_brain", "What is the architecture?");
        assert_eq!(result, "The architecture uses a 4-tier escalation ladder.");
    }
}
