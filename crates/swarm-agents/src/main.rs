mod beads_bridge;
mod config;
mod implementer;
mod validator;
mod worktree_bridge;

use anyhow::Result;
use config::SwarmConfig;
use coordination::context_packer::ContextPacker;
use coordination::escalation::EscalationState;
use coordination::verifier::{Verifier, VerifierConfig};
use coordination::work_packet::WorkPacket;
use coordination::SwarmTier;
use tracing::{error, info, warn};

/// Format a WorkPacket into a structured prompt for the implementer agent.
fn format_work_packet(packet: &WorkPacket) -> String {
    let mut prompt = String::new();

    prompt.push_str(&format!("# Task: {}\n\n", packet.objective));
    prompt.push_str(&format!(
        "**Branch:** {} | **Iteration:** {} | **Tier:** {}\n\n",
        packet.branch, packet.iteration, packet.target_tier
    ));

    if !packet.constraints.is_empty() {
        prompt.push_str("## Constraints\n");
        for c in &packet.constraints {
            prompt.push_str(&format!("- [{:?}] {}\n", c.kind, c.description));
        }
        prompt.push('\n');
    }

    if !packet.failure_signals.is_empty() {
        prompt.push_str("## Previous Failures\n");
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
        prompt.push_str("## Previous Attempts\n");
        for attempt in &packet.previous_attempts {
            prompt.push_str(&format!("- {attempt}\n"));
        }
        prompt.push('\n');
    }

    if !packet.file_contexts.is_empty() {
        prompt.push_str("## Relevant Code\n");
        for ctx in &packet.file_contexts {
            prompt.push_str(&format!(
                "### {} (lines {}-{}) — {}\n```rust\n{}\n```\n\n",
                ctx.file, ctx.start_line, ctx.end_line, ctx.relevance, ctx.content
            ));
        }
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

/// MVP stub: "apply" implementer output. In the real loop this parses the LLM
/// response into file edits. For now it creates an empty commit so the verifier
/// has something to diff.
fn apply_implementer_changes(worktree_path: &std::path::Path, _response: &str) -> Result<()> {
    warn!("apply_implementer_changes is an MVP stub — creating empty commit");

    let output = std::process::Command::new("git")
        .args([
            "commit",
            "--allow-empty",
            "-m",
            "swarm: implementer stub commit",
        ])
        .current_dir(worktree_path)
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("Failed to create stub commit: {stderr}");
    }

    Ok(())
}

/// Get the git diff of the worktree vs its parent branch.
fn git_diff(worktree_path: &std::path::Path) -> Result<String> {
    let output = std::process::Command::new("git")
        .args(["diff", "HEAD~1..HEAD"])
        .current_dir(worktree_path)
        .output()?;

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
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
        reasoning = %config.reasoning_endpoint.url,
        max_retries = config.max_retries,
        "Swarm orchestrator starting"
    );

    // Initialize components
    let beads = beads_bridge::BeadsBridge::new();
    let implementer = match implementer::Implementer::new(&config) {
        Ok(i) => i,
        Err(e) => {
            error!("Failed to initialize implementer: {e}");
            return Err(e);
        }
    };
    let validator = match validator::Validator::new(&config) {
        Ok(v) => v,
        Err(e) => {
            error!("Failed to initialize validator: {e}");
            return Err(e);
        }
    };

    // Detect repo root (current working directory for now)
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

    // --- Run implementer → verifier → validator loop ---
    let mut escalation = EscalationState::new(&issue.id);
    let packer = ContextPacker::new(&wt_path, SwarmTier::Implementer);
    let mut success = false;

    for iteration in 1..=config.max_retries {
        info!(iteration, id = %issue.id, "Starting iteration");

        // Pack context
        let packet = if iteration == 1 {
            packer.pack_initial(&issue.id, &issue.title)
        } else {
            // Re-run verifier to get fresh report for retry context
            let verifier = Verifier::new(&wt_path, VerifierConfig::default());
            let report = verifier.run_pipeline().await;
            packer.pack_retry(&issue.id, &issue.title, &escalation, &report)
        };

        info!(
            tokens = packet.estimated_tokens(),
            files = packet.file_contexts.len(),
            "Packed context"
        );

        // Implementer: generate code
        let formatted_prompt = format_work_packet(&packet);
        let impl_response = match implementer.implement(&formatted_prompt).await {
            Ok(r) => r,
            Err(e) => {
                error!(iteration, "Implementer failed: {e}");
                escalation.record_iteration(vec![], 1, false);
                continue;
            }
        };

        info!(
            iteration,
            response_len = impl_response.len(),
            "Implementer responded"
        );

        // Apply changes (MVP stub)
        if let Err(e) = apply_implementer_changes(&wt_path, &impl_response) {
            error!(iteration, "Failed to apply changes: {e}");
            escalation.record_iteration(vec![], 1, false);
            continue;
        }

        // Verifier: run deterministic gates
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
            // Validator: blind review of the diff
            let diff = git_diff(&wt_path)?;
            if diff.is_empty() {
                info!(iteration, "No diff to validate (empty commit stub)");
                escalation.record_iteration(error_cats, error_count, true);
                success = true;
                break;
            }

            match validator.validate(&diff).await {
                Ok(result) if result.passed => {
                    info!(iteration, "Validator PASSED");
                    escalation.record_iteration(error_cats, error_count, true);
                    success = true;
                    break;
                }
                Ok(result) => {
                    warn!(
                        iteration,
                        feedback = %result.feedback,
                        "Validator FAILED — looping"
                    );
                    escalation.record_iteration(error_cats, error_count, false);
                }
                Err(e) => {
                    error!(iteration, "Validator error: {e}");
                    escalation.record_iteration(error_cats, error_count, false);
                }
            }
        } else {
            escalation.record_iteration(error_cats, error_count, false);
        }
    }

    // --- Outcome ---
    if success {
        info!(id = %issue.id, "Issue resolved — merging worktree");
        if let Err(e) = worktree_bridge.merge_and_remove(&issue.id) {
            error!(id = %issue.id, "Merge failed: {e}");
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
