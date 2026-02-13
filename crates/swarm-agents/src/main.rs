mod agents;
mod beads_bridge;
mod config;
mod prompts;
mod tools;
mod worktree_bridge;

use std::path::Path;

use anyhow::Result;
use rig::completion::Prompt;
use tracing::{error, info, warn};

use agents::reviewer::ReviewResult;
use agents::AgentFactory;
use config::{check_endpoint, SwarmConfig};
use coordination::feedback::ErrorCategory;
use coordination::{
    ContextPacker, EscalationEngine, EscalationState, SwarmTier, Verifier, VerifierConfig,
    VerifierReport, WorkPacket,
};

/// Format a WorkPacket into a structured prompt for agent consumption.
fn format_task_prompt(packet: &WorkPacket, cloud_guidance: Option<&str>) -> String {
    let mut prompt = String::new();

    prompt.push_str(&format!("# Task: {}\n\n", packet.objective));
    prompt.push_str(&format!(
        "**Branch:** {} | **Iteration:** {} | **Tier:** {}\n\n",
        packet.branch, packet.iteration, packet.target_tier
    ));

    if let Some(guidance) = cloud_guidance {
        prompt.push_str("## Architectural Guidance (from cloud escalation)\n");
        prompt.push_str(guidance);
        prompt.push_str("\n\n");
    }

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
fn git_commit_changes(wt_path: &Path, iteration: u32) -> Result<bool> {
    // Stage all changes
    let add = std::process::Command::new("git")
        .args(["add", "-A"])
        .current_dir(wt_path)
        .output()?;
    if !add.status.success() {
        let stderr = String::from_utf8_lossy(&add.stderr);
        anyhow::bail!("git add failed: {stderr}");
    }

    // Check if there are staged changes
    let status = std::process::Command::new("git")
        .args(["diff", "--cached", "--quiet"])
        .current_dir(wt_path)
        .output()?;

    if status.status.success() {
        // Exit code 0 means no diff — nothing to commit
        return Ok(false);
    }

    // Commit
    let msg = format!("swarm: iteration {iteration} changes");
    let commit = std::process::Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(wt_path)
        .output()?;
    if !commit.status.success() {
        let stderr = String::from_utf8_lossy(&commit.stderr);
        anyhow::bail!("git commit failed: {stderr}");
    }

    Ok(true)
}

/// Get the git diff of the worktree vs its parent branch.
fn git_diff(worktree_path: &Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(["diff", "HEAD~1..HEAD"])
        .current_dir(worktree_path)
        .output()?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Determine if the Rust specialist coder should handle this task
/// based on error categories from the last verifier run.
///
/// Rust-specific errors (borrow checker, lifetimes, trait bounds) go to
/// strand-14B. Everything else goes to the general coder.
fn should_use_rust_coder(error_cats: &[ErrorCategory]) -> bool {
    if error_cats.is_empty() {
        // No errors yet (first iteration) — use general coder for scaffolding
        return false;
    }
    error_cats.iter().any(|cat| {
        matches!(
            cat,
            ErrorCategory::BorrowChecker
                | ErrorCategory::Lifetime
                | ErrorCategory::TraitBound
                | ErrorCategory::TypeMismatch
                | ErrorCategory::Async
        )
    })
}

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
        "Swarm orchestrator starting"
    );

    // --- Health check endpoints ---
    let local_ok = check_endpoint(&config.fast_endpoint.url).await;
    let reasoning_ok = check_endpoint(&config.reasoning_endpoint.url).await;
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
    let beads = beads_bridge::BeadsBridge::new();

    // Detect repo root
    let repo_root = std::env::current_dir()?;
    let worktree_bridge =
        worktree_bridge::WorktreeBridge::new(config.worktree_base.clone(), &repo_root)?;

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

    // --- Build agents scoped to this worktree ---
    // Direct-use agents (Implementer tier — skip Manager overhead)
    let rust_coder = factory.build_rust_coder(&wt_path);
    let general_coder = factory.build_general_coder(&wt_path);
    let reviewer = factory.build_reviewer();

    // Manager agent (Integrator tier — orchestrates via tool-calling)
    let manager = factory.build_manager(&wt_path);

    // Cloud agent (optional, for architectural guidance)
    let cloud_agent = factory.build_cloud_agent();

    // --- Escalation state ---
    let engine = EscalationEngine::new();
    let mut escalation = EscalationState::new(&issue.id);
    let mut success = false;
    let mut last_report: Option<VerifierReport> = None;
    let mut cloud_guidance: Option<String> = None;

    // --- Main loop: implement → verify → review → escalate ---
    for iteration in 1..=config.max_retries {
        let tier = escalation.current_tier;
        info!(iteration, ?tier, id = %issue.id, "Starting iteration");

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

        let task_prompt = format_task_prompt(&packet, cloud_guidance.as_deref());

        // --- Route to agent based on current tier ---
        let agent_response: Result<String, _> = match tier {
            SwarmTier::Implementer | SwarmTier::Adversary => {
                // Direct coder — no Manager overhead for targeted fixes
                let recent_cats: Vec<ErrorCategory> = escalation
                    .recent_error_categories
                    .last()
                    .cloned()
                    .unwrap_or_default();

                if should_use_rust_coder(&recent_cats) {
                    info!(iteration, "Routing to rust_coder (strand-14B)");
                    rust_coder.prompt(&task_prompt).await
                } else {
                    info!(iteration, "Routing to general_coder (Qwen3-Coder-Next)");
                    general_coder.prompt(&task_prompt).await
                }
            }
            SwarmTier::Integrator => {
                // Manager orchestrates — 72B delegates to workers via tools
                info!(iteration, "Routing to manager (OR1-Behemoth)");
                manager.prompt(&task_prompt).await
            }
            SwarmTier::Cloud => {
                // Cloud escalation — architectural guidance only
                if let Some(ref cloud) = cloud_agent {
                    info!(iteration, "Routing to cloud escalation");
                    match cloud.prompt(&task_prompt).await {
                        Ok(guidance) => {
                            info!(
                                iteration,
                                guidance_len = guidance.len(),
                                "Cloud guidance received"
                            );
                            cloud_guidance = Some(guidance);
                            // Don't commit/verify — feed guidance into next iteration
                            escalation.record_iteration(vec![], 0, false);
                            continue;
                        }
                        Err(e) => {
                            error!(iteration, "Cloud agent failed: {e}");
                            escalation.record_iteration(vec![], 0, false);
                            continue;
                        }
                    }
                } else {
                    error!("Cloud tier requested but no cloud agent configured");
                    error!("Flagging issue for human intervention");
                    break;
                }
            }
        };

        // Clear cloud guidance after it's been consumed
        cloud_guidance = None;

        // Handle agent failure
        let _response = match agent_response {
            Ok(r) => {
                info!(iteration, response_len = r.len(), "Agent responded");
                r
            }
            Err(e) => {
                error!(iteration, "Agent failed: {e}");
                escalation.record_iteration(vec![], 1, false);

                // Run verifier for escalation decision even on agent failure
                let verifier = Verifier::new(&wt_path, VerifierConfig::default());
                let report = verifier.run_pipeline().await;
                let decision = engine.decide(&mut escalation, &report);
                last_report = Some(report);

                if decision.stuck {
                    error!(iteration, "Escalation engine: stuck after agent failure");
                    break;
                }
                continue;
            }
        };

        // --- Git commit changes made by the agent ---
        let has_changes = match git_commit_changes(&wt_path, iteration) {
            Ok(changed) => changed,
            Err(e) => {
                warn!(iteration, "git commit error: {e}");
                false
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
                break;
            }
            continue;
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
            // --- Reviewer: blind review of the diff ---
            let diff = git_diff(&wt_path)?;
            if diff.is_empty() {
                warn!(
                    iteration,
                    "Empty diff despite git changes — verifier may be wrong"
                );
                escalation.record_iteration(error_cats, error_count, false);
                let decision = engine.decide(&mut escalation, &report);
                last_report = Some(report);
                if decision.stuck {
                    break;
                }
                continue;
            }

            info!(
                iteration,
                diff_len = diff.len(),
                "Sending diff to blind reviewer"
            );
            match reviewer.prompt(&diff).await {
                Ok(resp) => {
                    let result = ReviewResult::parse(&resp);
                    if result.passed {
                        info!(iteration, "Reviewer PASSED — issue resolved");
                        escalation.record_iteration(error_cats, error_count, true);
                        success = true;
                        break;
                    } else {
                        warn!(
                            iteration,
                            feedback = %result.feedback,
                            "Reviewer FAILED — looping"
                        );
                        escalation.record_iteration(error_cats, error_count, false);
                    }
                }
                Err(e) => {
                    warn!(iteration, "Reviewer unavailable: {e}");
                    // Verifier passed — accept without reviewer
                    info!(
                        iteration,
                        "Verifier passed, reviewer unreachable — accepting result"
                    );
                    escalation.record_iteration(error_cats, error_count, true);
                    success = true;
                    break;
                }
            }
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
            break;
        }
    }

    // --- Outcome ---
    if success {
        info!(id = %issue.id, "Issue resolved — merging worktree");
        if let Err(e) = worktree_bridge.merge_and_remove(&issue.id) {
            error!(id = %issue.id, "Merge failed: {e} — resetting issue to open");
            let _ = beads.update_status(&issue.id, "open");
            return Err(e);
        }
        beads.close(&issue.id, Some("Resolved by swarm orchestrator"))?;
        info!(id = %issue.id, "Issue closed");
    } else {
        error!(
            id = %issue.id,
            iterations = config.max_retries,
            summary = %escalation.summary(),
            "Issue NOT resolved after max retries — leaving worktree for inspection"
        );
    }

    Ok(())
}
