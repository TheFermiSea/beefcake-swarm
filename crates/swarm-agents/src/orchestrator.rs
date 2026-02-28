//! Orchestration loop: process a single issue through implement → verify → review → escalate.
//!
//! Integrates coordination's harness for session tracking, git checkpoints,
//! progress logging, and human intervention requests.

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result};
use rig::completion::Prompt;
use rig::providers::openai;
use tracing::{debug, error, info, warn};

use crate::runtime_adapter::{AdapterConfig, RuntimeAdapter};

/// Default timeout for each cloud validation call.
const DEFAULT_VALIDATION_TIMEOUT_SECS: u64 = 120; // 2 minutes

use crate::acceptance::{self, AcceptancePolicy};
use crate::agents::reviewer::{self, ReviewResult};
use crate::agents::AgentFactory;
use crate::beads_bridge::{BeadsIssue, IssueTracker};
use crate::config::SwarmConfig;
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
    ContextPacker, EscalationEngine, EscalationState, GitManager, InterventionType,
    PendingIntervention, ProgressTracker, SessionManager, SwarmTier, TierBudget, TurnPolicy,
    ValidatorFeedback, ValidatorIssueType, Verifier, VerifierConfig, VerifierReport, WorkPacket,
};

/// Coder routing decision with confidence level.
#[derive(Debug, PartialEq, Eq)]
pub enum CoderRoute {
    /// Qwen3.5-397B with Rust specialist system prompt
    RustCoder,
    /// Qwen3.5-397B with general coder system prompt (multi-file scaffolding)
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
/// - Rust-specific categories (borrow checker, lifetimes, traits) score toward RustCoder system prompt
/// - Structural categories (imports, syntax, macros) score toward GeneralCoder system prompt
/// - Mixed errors with majority Rust → RustCoder; majority structural → GeneralCoder
/// - No errors (first iteration) → general coder for scaffolding
///
/// Both routes use Qwen3.5-397B on vasp-02 — differentiation is by system prompt only.
pub fn route_to_coder(error_cats: &[ErrorCategory]) -> CoderRoute {
    if error_cats.is_empty() {
        // First iteration — use general coder for scaffolding/multi-file work
        return CoderRoute::GeneralCoder;
    }

    let mut rust_score: i32 = 0;
    let mut general_score: i32 = 0;

    for cat in error_cats {
        match cat {
            // Deep Rust expertise required — Rust specialist prompt excels here
            ErrorCategory::BorrowChecker => rust_score += 3,
            ErrorCategory::Lifetime => rust_score += 3,
            ErrorCategory::TraitBound => rust_score += 2,
            ErrorCategory::Async => rust_score += 2,
            ErrorCategory::TypeMismatch => rust_score += 1,

            // Structural/multi-file work — general coder's 65K context helps
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

    // Scope constraints — explicitly tell the worker what it may modify
    if !packet.files_touched.is_empty() {
        prompt.push_str("## Scope Constraints\n");
        prompt.push_str("**IMPORTANT:** Only modify these files:\n");
        for f in &packet.files_touched {
            prompt.push_str(&format!("- `{f}`\n"));
        }
        prompt.push_str("\nDo NOT modify any other files. Do NOT reformat, refactor, or ");
        prompt.push_str("\"improve\" code outside the listed files. If you believe additional ");
        prompt
            .push_str("files need changes, note them in your response but do not modify them.\n\n");
    }

    // Validator feedback from prior iteration (TextGrad pattern)
    if !packet.validator_feedback.is_empty() {
        prompt.push_str("## Reviewer Feedback (from prior iteration)\n");
        prompt.push_str("A code reviewer identified these issues. **Address each one:**\n\n");
        for (i, fb) in packet.validator_feedback.iter().enumerate() {
            prompt.push_str(&format!(
                "{}. **[{}]** {}\n",
                i + 1,
                fb.issue_type,
                fb.description
            ));
            if let Some(file) = &fb.file {
                if let Some((start, end)) = fb.line_range {
                    prompt.push_str(&format!("   Location: `{file}` lines {start}-{end}\n"));
                } else {
                    prompt.push_str(&format!("   Location: `{file}`\n"));
                }
            }
            if let Some(fix) = &fb.suggested_fix {
                prompt.push_str(&format!("   Suggested fix: {fix}\n"));
            }
        }
        prompt.push('\n');
    }

    prompt.push_str(&format!(
        "**Max patch size:** {} LOC\n",
        packet.max_patch_loc
    ));

    prompt
}

/// Format a compact task prompt for small local models (HydraCoder, etc).
///
/// Long structured prompts suppress tool-call generation in small MoE models.
/// This format keeps the user message under ~1500 chars by including only:
/// - The objective
/// - Target file(s)
/// - Error summary (retries only, truncated)
/// - A directive to call read_file first
///
/// The full verbose format (`format_task_prompt`) is for cloud/council models.
pub fn format_compact_task_prompt(packet: &WorkPacket, wt_root: &Path) -> String {
    let mut prompt = String::with_capacity(1500);

    prompt.push_str(&format!("# Task: {}\n\n", packet.objective));

    // Target files — prefer explicit scope, then extract from objective text,
    // then fall back to first few file_contexts filenames.
    let target_files: Vec<&str> = if !packet.files_touched.is_empty() {
        packet.files_touched.iter().map(|s| s.as_str()).collect()
    } else {
        // Try to extract file paths from objective (e.g., "File: src/foo.rs")
        // Strip trailing punctuation (period, comma, etc.) before checking extension.
        let objective_files: Vec<&str> = packet
            .objective
            .split_whitespace()
            .filter(|w| {
                let trimmed = w.trim_end_matches(|c: char| {
                    c.is_ascii_punctuation() && c != '/' && c != '.' && c != '_' && c != '-'
                });
                trimmed.contains('/') && (trimmed.ends_with(".rs") || trimmed.ends_with(".toml"))
            })
            .map(|w| {
                w.trim_end_matches(|c: char| {
                    c.is_ascii_punctuation() && c != '/' && c != '.' && c != '_' && c != '-'
                })
            })
            .collect();
        if !objective_files.is_empty() {
            objective_files
        } else {
            packet
                .file_contexts
                .iter()
                .map(|fc| fc.file.as_str())
                .take(3)
                .collect()
        }
    };

    // Guard against empty target_files — fall back to common entry points
    // to prevent prompt starvation on architectural tasks.
    let target_files = if target_files.is_empty() {
        ["src/lib.rs", "src/main.rs"]
            .iter()
            .copied()
            .filter(|p| wt_root.join(p).exists())
            .chain(std::iter::once("Cargo.toml"))
            .take(3)
            .collect()
    } else {
        target_files
    };

    if !target_files.is_empty() {
        prompt.push_str("**Files to modify:**\n");
        for f in &target_files {
            prompt.push_str(&format!("- `{f}`\n"));
        }
        prompt.push('\n');
    }

    // On retries: include error summary (compact — category + message only)
    if !packet.failure_signals.is_empty() {
        prompt.push_str("**Errors to fix:**\n");
        let mut error_chars = 0usize;
        for signal in &packet.failure_signals {
            let line = format!("- {}: {}\n", signal.category, signal.message);
            error_chars += line.len();
            if error_chars > 800 {
                prompt.push_str("- (more errors truncated)\n");
                break;
            }
            prompt.push_str(&line);
        }
        prompt.push('\n');
    }

    // Validator feedback (compact)
    if !packet.validator_feedback.is_empty() {
        prompt.push_str("**Reviewer feedback:**\n");
        for fb in packet.validator_feedback.iter().take(3) {
            prompt.push_str(&format!("- [{}] {}\n", fb.issue_type, fb.description));
        }
        prompt.push('\n');
    }

    // Inline the target file content when there's exactly one target file.
    // This saves a read_file turn (critical with max_turns=5).
    if target_files.len() == 1 {
        let target_path = wt_root.join(target_files[0]);
        if let Ok(content) = std::fs::read_to_string(&target_path) {
            let truncated = if content.len() > 4000 {
                format!("{}...\n[truncated at 4000 chars]", &content[..4000])
            } else {
                content
            };
            prompt.push_str(&format!(
                "**Current content of `{}`:**\n```\n{}\n```\n\n",
                target_files[0], truncated
            ));
            prompt.push_str(
                "Apply your edits directly with edit_file — the file content is above.\n",
            );
        } else {
            prompt.push_str(&format!(
                "Start by calling read_file on `{}`, then apply your edits with edit_file.\n",
                target_files[0]
            ));
        }
    } else {
        prompt.push_str(
            "Start by calling read_file on the target file(s), then apply your edits with edit_file.\n",
        );
    }

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
    // Stage changes (respects .gitignore) — retry for transient index.lock errors
    let add = retry_git_command_async(&["add", "."], wt_path, 3).await?;
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

    // Commit — retry for transient index.lock errors
    let msg = format!("swarm: iteration {iteration} changes");
    let commit = retry_git_command_async(&["commit", "-m", &msg], wt_path, 3).await?;
    if !commit.status.success() {
        let stderr = String::from_utf8_lossy(&commit.stderr);
        anyhow::bail!("git commit failed: {stderr}");
    }

    Ok(true)
}

/// Backoff delays for transient git retries (milliseconds).
const ASYNC_RETRY_DELAYS_MS: &[u64] = &[100, 500, 2000];

/// Async version of retry_git_command for tokio contexts.
async fn retry_git_command_async(
    args: &[&str],
    working_dir: &Path,
    max_retries: u32,
) -> Result<std::process::Output> {
    for attempt in 0..=max_retries {
        let output = tokio::process::Command::new("git")
            .args(args)
            .current_dir(working_dir)
            .output()
            .await
            .context("Failed to execute git command")?;

        if output.status.success() {
            return Ok(output);
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        let is_transient = stderr.contains("index.lock") || stderr.contains("Unable to create");

        if attempt < max_retries && is_transient {
            let delay = ASYNC_RETRY_DELAYS_MS
                .get(attempt as usize)
                .copied()
                .unwrap_or(2000);
            warn!(
                attempt = attempt + 1,
                max_retries,
                delay_ms = delay,
                "Transient git failure, retrying: {}",
                stderr.trim()
            );
            tokio::time::sleep(Duration::from_millis(delay)).await;
            continue;
        }

        return Ok(output);
    }

    unreachable!()
}

/// Result of a single cloud model validation.
struct CloudValidationResult {
    model: String,
    passed: bool,
    feedback: String,
}

/// Convert a cloud validation result into structured validator feedback entries.
///
/// Parses the reviewer's JSON response to extract blocking_issues and
/// touched_files, converting prose feedback into actionable deltas (TextGrad pattern).
fn extract_validator_feedback(result: &CloudValidationResult) -> Vec<ValidatorFeedback> {
    if result.passed {
        return vec![];
    }

    let review = ReviewResult::parse(&result.feedback);

    if review.blocking_issues.is_empty() {
        // Unstructured feedback — wrap as a single entry
        return vec![ValidatorFeedback {
            file: None,
            line_range: None,
            issue_type: ValidatorIssueType::Other,
            description: result
                .feedback
                .lines()
                .take(5)
                .collect::<Vec<_>>()
                .join(" "),
            suggested_fix: None,
            source_model: Some(result.model.clone()),
        }];
    }

    review
        .blocking_issues
        .iter()
        .map(|issue| {
            // Try to classify the issue type from keywords
            let issue_type = classify_issue(issue);

            // Try to extract file reference from touched_files
            let file = review.touched_files.first().cloned();

            ValidatorFeedback {
                file,
                line_range: None,
                issue_type,
                description: issue.clone(),
                suggested_fix: if review.suggested_next_action.is_empty() {
                    None
                } else {
                    Some(review.suggested_next_action.clone())
                },
                source_model: Some(result.model.clone()),
            }
        })
        .collect()
}

/// Classify a blocking issue description into a `ValidatorIssueType`.
fn classify_issue(description: &str) -> ValidatorIssueType {
    let lower = description.to_lowercase();
    if lower.contains("logic") || lower.contains("incorrect") || lower.contains("wrong") {
        ValidatorIssueType::LogicError
    } else if lower.contains("safety")
        || lower.contains("error handling")
        || lower.contains("unwrap")
        || lower.contains("panic")
    {
        ValidatorIssueType::MissingSafetyCheck
    } else if lower.contains("edge case")
        || lower.contains("boundary")
        || lower.contains("overflow")
        || lower.contains("empty")
    {
        ValidatorIssueType::UnhandledEdgeCase
    } else if lower.contains("style") || lower.contains("naming") || lower.contains("format") {
        ValidatorIssueType::StyleViolation
    } else if lower.contains("behavior")
        || lower.contains("specification")
        || lower.contains("spec")
    {
        ValidatorIssueType::IncorrectBehavior
    } else {
        ValidatorIssueType::Other
    }
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
/// Build the reviewer prompt for a given diff.
///
/// Shared between cloud and local validation to prevent prompt drift.
fn build_reviewer_prompt(diff_for_review: &str) -> String {
    format!(
        "You are reviewing a Rust code change from an autonomous coding agent. \
         The change has already passed all deterministic gates (cargo fmt, clippy, \
         cargo check, cargo test). Your job is to catch logic errors, edge cases, \
         and design issues that the compiler cannot detect.\n\n\
         Respond with STRICT JSON ONLY using schema: \
         {{\"verdict\":\"pass|fail|needs_escalation\",\"confidence\":0.0-1.0,\
         \"blocking_issues\":[...],\"suggested_next_action\":\"...\",\
         \"touched_files\":[...]}}.\n\n\
         ```diff\n{diff_for_review}\n```"
    )
}

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

    let review_prompt = build_reviewer_prompt(&diff_for_review);
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

/// Result of a local validator review (blocking gate).
struct LocalValidationResult {
    model: String,
    passed: bool,
    #[allow(dead_code)] // kept for diagnostics/future logging
    schema_valid: bool,
    feedback: String,
    blocking_issues: Vec<String>,
    suggested_next_action: String,
    touched_files: Vec<String>,
}

/// Run local validation via the reviewer agent (vasp-02/HydraCoder).
///
/// Generates a diff, sends it to the reviewer, and parses the structured JSON response.
/// - **Fail-open** on infrastructure errors (diff failure, timeout, LLM error) — deterministic gates already passed.
/// - **Fail-closed** on invalid JSON schema — malformed reviewer output counts as failure.
async fn local_validate(
    reviewer: &crate::agents::coder::OaiAgent,
    wt_path: &Path,
    initial_commit: &str,
    model_name: &str,
) -> LocalValidationResult {
    let validation_timeout = timeout_from_env("SWARM_LOCAL_VALIDATION_TIMEOUT_SECS", 60);

    // Generate diff (async to avoid blocking the tokio runtime)
    let diff = match tokio::process::Command::new("git")
        .args(["diff", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .await
    {
        Ok(output) if output.status.success() => {
            String::from_utf8_lossy(&output.stdout).to_string()
        }
        Ok(output) => {
            warn!(
                "local_validate: git diff failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            // Fail-open: infra error
            return LocalValidationResult {
                model: model_name.to_string(),
                passed: true,
                schema_valid: true,
                feedback: "git diff failed — fail-open".to_string(),
                blocking_issues: vec![],
                suggested_next_action: String::new(),
                touched_files: vec![],
            };
        }
        Err(e) => {
            warn!("local_validate: Failed to run git diff: {e}");
            return LocalValidationResult {
                model: model_name.to_string(),
                passed: true,
                schema_valid: true,
                feedback: format!("git diff error — fail-open: {e}"),
                blocking_issues: vec![],
                suggested_next_action: String::new(),
                touched_files: vec![],
            };
        }
    };

    if diff.trim().is_empty() {
        info!("local_validate: No diff to validate — pass");
        return LocalValidationResult {
            model: model_name.to_string(),
            passed: true,
            schema_valid: true,
            feedback: "No diff".to_string(),
            blocking_issues: vec![],
            suggested_next_action: String::new(),
            touched_files: vec![],
        };
    }

    // Truncate large diffs (on a valid char boundary to avoid panics)
    let max_diff_bytes = 32_000;
    let diff_for_review = if diff.len() > max_diff_bytes {
        let boundary = diff.floor_char_boundary(max_diff_bytes);
        format!(
            "{}\n\n... [truncated — {} total bytes, showing first {}]",
            &diff[..boundary],
            diff.len(),
            boundary,
        )
    } else {
        diff
    };

    let review_prompt = build_reviewer_prompt(&diff_for_review);

    // Call reviewer with timeout and retry
    match tokio::time::timeout(
        validation_timeout,
        prompt_with_retry(reviewer, &review_prompt, 2),
    )
    .await
    {
        Ok(Ok(response)) => {
            let review = ReviewResult::parse(&response);
            if !review.schema_valid {
                // Fail-closed: bad JSON schema
                warn!(
                    model = model_name,
                    "Local validation: invalid schema — fail-closed"
                );
                return LocalValidationResult {
                    model: model_name.to_string(),
                    passed: false,
                    schema_valid: false,
                    feedback: response,
                    blocking_issues: review.blocking_issues,
                    suggested_next_action: review.suggested_next_action,
                    touched_files: review.touched_files,
                };
            }
            let status = if review.passed { "PASS" } else { "FAIL" };
            info!(model = model_name, status, "Local validation complete");
            LocalValidationResult {
                model: model_name.to_string(),
                passed: review.passed,
                schema_valid: true,
                feedback: response,
                blocking_issues: review.blocking_issues,
                suggested_next_action: review.suggested_next_action,
                touched_files: review.touched_files,
            }
        }
        Ok(Err(e)) => {
            // Fail-open: LLM error
            warn!(model = model_name, error = %e, "Local validation LLM error — fail-open");
            LocalValidationResult {
                model: model_name.to_string(),
                passed: true,
                schema_valid: true,
                feedback: format!("LLM error — fail-open: {e}"),
                blocking_issues: vec![],
                suggested_next_action: String::new(),
                touched_files: vec![],
            }
        }
        Err(_) => {
            // Fail-open: timeout
            warn!(
                model = model_name,
                timeout_secs = validation_timeout.as_secs(),
                "Local validation timed out — fail-open"
            );
            LocalValidationResult {
                model: model_name.to_string(),
                passed: true,
                schema_valid: true,
                feedback: "Timed out — fail-open".to_string(),
                blocking_issues: vec![],
                suggested_next_action: String::new(),
                touched_files: vec![],
            }
        }
    }
}

/// Convert a local validation result into structured validator feedback entries.
///
/// Similar to `extract_validator_feedback` but operates on `LocalValidationResult`.
fn extract_local_validator_feedback(result: &LocalValidationResult) -> Vec<ValidatorFeedback> {
    if result.passed {
        return vec![];
    }

    if result.blocking_issues.is_empty() {
        // Unstructured feedback — wrap as a single entry
        return vec![ValidatorFeedback {
            file: None,
            line_range: None,
            issue_type: ValidatorIssueType::Other,
            description: result
                .feedback
                .lines()
                .take(5)
                .collect::<Vec<_>>()
                .join(" "),
            suggested_fix: None,
            source_model: Some(result.model.clone()),
        }];
    }

    result
        .blocking_issues
        .iter()
        .map(|issue| {
            let issue_type = classify_issue(issue);
            let file = result.touched_files.first().cloned();

            ValidatorFeedback {
                file,
                line_range: None,
                issue_type,
                description: issue.clone(),
                suggested_fix: if result.suggested_next_action.is_empty() {
                    None
                } else {
                    Some(result.suggested_next_action.clone())
                },
                source_model: Some(result.model.clone()),
            }
        })
        .collect()
}

/// Detect which Cargo packages have been modified in the worktree.
///
/// Combines committed changes (git diff main..HEAD) and working-tree
/// changes (git status --porcelain) to produce a deduplicated list of
/// package names. Falls back to an empty Vec (= full workspace) on any error.
fn detect_changed_packages(wt_path: &Path) -> Vec<String> {
    let mut changed_files: std::collections::HashSet<std::path::PathBuf> = Default::default();

    // Committed changes since branching from main
    if let Ok(out) = std::process::Command::new("git")
        .args(["diff", "--name-only", "main"])
        .current_dir(wt_path)
        .output()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if !line.trim().is_empty() {
                changed_files.insert(wt_path.join(line.trim()));
            }
        }
    }

    // Uncommitted working-tree changes (staged + unstaged)
    if let Ok(out) = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(wt_path)
        .output()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            // porcelain format: "XY filename" — filename starts at column 3
            if line.len() > 3 {
                let path = line[3..].trim();
                if !path.is_empty() {
                    changed_files.insert(wt_path.join(path));
                }
            }
        }
    }

    let mut packages: std::collections::HashSet<String> = Default::default();
    for file_path in &changed_files {
        if let Some(pkg) = find_package_name(file_path) {
            packages.insert(pkg);
        }
    }

    let result: Vec<String> = packages.into_iter().collect();
    if result.is_empty() {
        tracing::debug!("detect_changed_packages: no changes detected, targeting full workspace");
    } else {
        tracing::debug!(packages = ?result, "detect_changed_packages: scoping verifier to changed packages");
    }
    result
}

/// Walk up from `file_path` to find the nearest `Cargo.toml` and return the package `name`.
fn find_package_name(file_path: &Path) -> Option<String> {
    let mut dir = file_path.parent()?;
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                let mut in_package = false;
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed == "[package]" {
                        in_package = true;
                    } else if trimmed.starts_with('[') {
                        in_package = false;
                    } else if in_package && trimmed.starts_with("name") {
                        if let Some(name) = trimmed.split('"').nth(1) {
                            return Some(name.to_string());
                        }
                    }
                }
            }
        }
        match dir.parent() {
            Some(parent) if parent != dir => dir = parent,
            _ => break,
        }
    }
    None
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
    let process_span = otel::process_issue_span(&issue.id);
    let _process_guard = process_span.enter();
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
    let mut metrics = MetricsCollector::new(session.session_id(), &issue.id, &issue.title);

    // --- Acceptance policy ---
    let acceptance_policy = AcceptancePolicy::default();

    // --- Build agents scoped to this worktree ---
    let rust_coder = factory.build_rust_coder(&wt_path);
    let general_coder = factory.build_general_coder(&wt_path);
    let reviewer = factory.build_reviewer();
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
    let initial_tier = if feature_flags.worker_first_enabled {
        let recommendation = classify_initial_tier(&issue.title, &[]);
        info!(
            tier = ?recommendation.tier,
            complexity = %recommendation.complexity,
            confidence = recommendation.confidence,
            reason = %recommendation.reason,
            "Worker-first classification"
        );
        recommendation.tier
    } else {
        // Without cloud, default to Worker — local models write code directly.
        // With cloud, default to Council — cloud models handle delegation.
        let default_tier = if config.cloud_endpoint.is_some() {
            SwarmTier::Council
        } else {
            SwarmTier::Worker
        };
        tier_from_env("SWARM_INITIAL_TIER", default_tier)
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

    // Scope verifier to changed packages.
    // If explicit packages are configured (CLI --package or SWARM_VERIFIER_PACKAGES), use those.
    // Otherwise, detect from git-changed files to avoid missing breakage in other crates.
    let initial_packages = if config.verifier_packages.is_empty() {
        detect_changed_packages(&wt_path)
    } else {
        config.verifier_packages.clone()
    };
    let verifier_config = VerifierConfig {
        packages: initial_packages,
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
        let tier_str = format!("{tier:?}");
        let iter_span = otel::iteration_span(&issue.id, iteration, &tier_str);
        let _iter_guard = iter_span.enter();
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

        // Pack context with tier-appropriate token budget
        let packer = ContextPacker::new(&wt_path, tier);
        let mut packet = if let Some(ref report) = last_report {
            packer.pack_retry(&issue.id, &issue.title, &escalation, report)
        } else {
            packer.pack_initial(&issue.id, &issue.title)
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

        // Sparse context: escalate Worker→Council to avoid prompt starvation.
        let tier = if tier == SwarmTier::Worker
            && packet.file_contexts.is_empty()
            && packet.files_touched.is_empty()
            && packet.failure_signals.is_empty()
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
        let prev_error_count = last_report.as_ref().map(|r| r.failure_signals.len());

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
                            ..Default::default()
                        });
                        let result = match tokio::time::timeout(
                            worker_timeout,
                            rust_coder.prompt(&task_prompt).with_hook(adapter.clone()),
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
                            ..Default::default()
                        });
                        let result = match tokio::time::timeout(
                            worker_timeout,
                            general_coder
                                .prompt(&task_prompt)
                                .with_hook(adapter.clone()),
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
            SwarmTier::Council | SwarmTier::Human => {
                info!(
                    iteration,
                    "Routing to manager (cloud-backed or Qwen3.5-Architect fallback)"
                );
                metrics.record_agent_metrics("manager", 0, 0);
                let adapter = RuntimeAdapter::new(AdapterConfig {
                    agent_name: "manager".into(),
                    deadline: Some(Instant::now() + manager_timeout),
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
                (result, adapter)
            }
        };

        // Log runtime adapter report for tool-event visibility
        match adapter.report() {
            Ok(adapter_report) => {
                info!(
                    iteration,
                    agent = %adapter_report.agent_name,
                    turns = adapter_report.turn_count,
                    tool_calls = adapter_report.total_tool_calls,
                    tool_time_ms = adapter_report.total_tool_time_ms,
                    terminated_early = adapter_report.terminated_early,
                    "Runtime adapter report"
                );
                if let Some(ref reason) = adapter_report.termination_reason {
                    warn!(iteration, reason = %reason, "Agent terminated early by adapter");
                }
            }
            Err(e) => {
                warn!(iteration, error = %e, "Failed to extract runtime adapter report");
            }
        }

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
                info!(
                    iteration,
                    "Running verifier after agent failure to assess codebase state"
                );
                let current_verifier_config = if config.verifier_packages.is_empty() {
                    VerifierConfig {
                        packages: detect_changed_packages(&wt_path),
                        ..verifier_config.clone()
                    }
                } else {
                    verifier_config.clone()
                };
                let verifier = Verifier::new(&wt_path, current_verifier_config);
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
            let current_verifier_config = if config.verifier_packages.is_empty() {
                VerifierConfig {
                    packages: detect_changed_packages(&wt_path),
                    ..verifier_config.clone()
                }
            } else {
                verifier_config.clone()
            };
            let verifier = Verifier::new(&wt_path, current_verifier_config);
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
                packages: detect_changed_packages(&wt_path),
                ..verifier_config.clone()
            }
        } else {
            verifier_config.clone()
        };
        let verifier = Verifier::new(&wt_path, current_verifier_config);
        let mut report = verifier.run_pipeline().await;
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
            // Reset validator failure counter — verifier itself failed, so
            // prior validator feedback is stale.
            consecutive_validator_failures = 0;
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
                        // Re-run verifier against rolled-back code so last_report
                        // reflects the pre-regression error state, not the regressed state.
                        let rb_verifier_config = if config.verifier_packages.is_empty() {
                            VerifierConfig {
                                packages: detect_changed_packages(&wt_path),
                                ..verifier_config.clone()
                            }
                        } else {
                            verifier_config.clone()
                        };
                        let rb_verifier = Verifier::new(&wt_path, rb_verifier_config);
                        let rb_report = rb_verifier.run_pipeline().await;
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

    // --- Write telemetry ---
    let final_tier = format!("{:?}", escalation.current_tier);
    let session_metrics = metrics.finalize(success, &final_tier);
    telemetry::write_session_metrics(&session_metrics, &wt_path);
    telemetry::append_telemetry(&session_metrics, worktree_bridge.repo_root());

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
                let is_transient = is_transient_error(&err_str, &err_lower);

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

/// Like [`prompt_with_retry`] but attaches a [`RuntimeAdapter`] hook to each attempt.
///
/// The hook provides tool-event visibility and budget enforcement for the manager tier.
async fn prompt_with_hook_and_retry(
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
                let err_str = format!("{e}");
                let err_lower = err_str.to_ascii_lowercase();
                let is_transient = is_transient_error(&err_str, &err_lower);

                if !is_transient || attempt == max_retries {
                    return Err(e);
                }

                let backoff = Duration::from_secs(2u64.pow(attempt + 1));
                warn!(
                    attempt = attempt + 1,
                    max_retries,
                    backoff_secs = backoff.as_secs(),
                    error = %err_str,
                    "Transient error — retrying (with hook)"
                );
                last_err = Some(e);
                tokio::time::sleep(backoff).await;
            }
        }
    }
    Err(last_err.unwrap())
}

/// Classify whether an LLM API error is transient (connection failures, rate limits,
/// proxy hiccups) and worth retrying, vs permanent (auth errors, schema mismatches).
fn is_transient_error(err_str: &str, err_lower: &str) -> bool {
    // HTTP status codes
    err_str.contains("502")
        || err_str.contains("503")
        || err_str.contains("429")
        // Connection-level failures (reqwest)
        || err_lower.contains("connection")
        || err_lower.contains("timed out")
        || err_lower.contains("timeout")
        || err_lower.contains("error sending request")
        || err_lower.contains("broken pipe")
        || err_lower.contains("reset by peer")
        // Proxy occasionally returns empty-but-200 payloads; retry recovers.
        || err_lower.contains("no message or tool call (empty)")
        || err_lower.contains("response contained no message or tool call")
        // Proxy/model schema mismatches can be intermittent during model churn.
        || err_lower.contains("jsonerror")
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
            iteration_deltas: vec![],
            delegation_chain: vec![],
            skill_hints: vec![],
            replay_hints: vec![],
            validator_feedback: vec![],
            change_contract: None,
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

    #[test]
    fn test_extract_validator_feedback_pass_returns_empty() {
        let result = CloudValidationResult {
            model: "test-model".into(),
            passed: true,
            feedback: r#"{"verdict":"pass","confidence":0.95,"blocking_issues":[],"suggested_next_action":"merge","touched_files":["src/lib.rs"]}"#.into(),
        };
        assert!(extract_validator_feedback(&result).is_empty());
    }

    #[test]
    fn test_extract_validator_feedback_fail_with_blocking_issues() {
        let result = CloudValidationResult {
            model: "gemini-3-pro".into(),
            passed: false,
            feedback: r#"{"verdict":"fail","confidence":0.7,"blocking_issues":["missing error handling for edge case","logic error in loop termination"],"suggested_next_action":"add bounds checking","touched_files":["src/main.rs"]}"#.into(),
        };
        let feedback = extract_validator_feedback(&result);
        assert_eq!(feedback.len(), 2);
        // "missing error handling" matches safety check before "edge case"
        assert_eq!(
            feedback[0].issue_type,
            ValidatorIssueType::MissingSafetyCheck
        );
        assert_eq!(feedback[1].issue_type, ValidatorIssueType::LogicError);
        assert_eq!(feedback[0].source_model.as_deref(), Some("gemini-3-pro"));
        assert!(feedback[0].suggested_fix.is_some());
    }

    #[test]
    fn test_extract_validator_feedback_malformed_falls_back() {
        let result = CloudValidationResult {
            model: "test-model".into(),
            passed: false,
            feedback: "FAIL\nThis code has issues".into(),
        };
        let feedback = extract_validator_feedback(&result);
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].issue_type, ValidatorIssueType::Other);
    }

    #[test]
    fn test_classify_issue_keywords() {
        assert_eq!(
            classify_issue("missing error handling for None case"),
            ValidatorIssueType::MissingSafetyCheck
        );
        assert_eq!(
            classify_issue("logic error in loop"),
            ValidatorIssueType::LogicError
        );
        assert_eq!(
            classify_issue("edge case when input is empty"),
            ValidatorIssueType::UnhandledEdgeCase
        );
        assert_eq!(
            classify_issue("naming convention violated"),
            ValidatorIssueType::StyleViolation
        );
        assert_eq!(
            classify_issue("behavior differs from specification"),
            ValidatorIssueType::IncorrectBehavior
        );
        assert_eq!(
            classify_issue("something else entirely"),
            ValidatorIssueType::Other
        );
    }

    #[test]
    fn test_format_task_prompt_with_validator_feedback() {
        use coordination::verifier::report::{ValidatorFeedback, ValidatorIssueType};
        let mut packet = test_packet("Fix the bug", "swarm/test", 2);
        packet.validator_feedback = vec![
            ValidatorFeedback {
                file: Some("src/main.rs".into()),
                line_range: Some((10, 20)),
                issue_type: ValidatorIssueType::LogicError,
                description: "Loop never terminates for empty input".into(),
                suggested_fix: Some("Add early return for empty vec".into()),
                source_model: Some("gemini-3-pro".into()),
            },
            ValidatorFeedback {
                file: None,
                line_range: None,
                issue_type: ValidatorIssueType::MissingSafetyCheck,
                description: "No bounds checking on index access".into(),
                suggested_fix: None,
                source_model: None,
            },
        ];
        let prompt = format_task_prompt(&packet);
        assert!(prompt.contains("Reviewer Feedback"));
        assert!(prompt.contains("Loop never terminates"));
        assert!(prompt.contains("src/main.rs"));
        assert!(prompt.contains("lines 10-20"));
        assert!(prompt.contains("Add early return"));
        assert!(prompt.contains("No bounds checking"));
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
    fn test_is_transient_error_classifies_correctly() {
        use super::is_transient_error;

        // Connection-level failures from reqwest
        let err = "Http client error: error sending request for url (http://example.com/v1/chat)";
        assert!(
            is_transient_error(err, &err.to_ascii_lowercase()),
            "reqwest SendError should be transient"
        );

        // Standard HTTP status codes
        assert!(is_transient_error("502 Bad Gateway", "502 bad gateway"));
        assert!(is_transient_error(
            "503 Service Unavailable",
            "503 service unavailable"
        ));
        assert!(is_transient_error(
            "429 Too Many Requests",
            "429 too many requests"
        ));

        // Connection and timeout variants
        assert!(is_transient_error(
            "connection refused",
            "connection refused"
        ));
        assert!(is_transient_error("request timed out", "request timed out"));
        assert!(is_transient_error("read timeout", "read timeout"));
        assert!(is_transient_error("broken pipe", "broken pipe"));
        assert!(is_transient_error("reset by peer", "reset by peer"));

        // Empty response from proxy
        assert!(is_transient_error(
            "no message or tool call (empty)",
            "no message or tool call (empty)"
        ));

        // Permanent errors should NOT be transient
        assert!(!is_transient_error("401 Unauthorized", "401 unauthorized"));
        assert!(!is_transient_error("invalid api key", "invalid api key"));
        assert!(!is_transient_error(
            "model not found: gpt-99",
            "model not found: gpt-99"
        ));
    }

    // ========================================================================
    // Local validator feedback extraction tests
    // ========================================================================

    #[test]
    fn test_extract_local_validator_feedback_pass_returns_empty() {
        let result = LocalValidationResult {
            model: "test-model".into(),
            passed: true,
            schema_valid: true,
            feedback: "looks good".into(),
            blocking_issues: vec![],
            suggested_next_action: String::new(),
            touched_files: vec![],
        };
        let feedback = extract_local_validator_feedback(&result);
        assert!(feedback.is_empty());
    }

    #[test]
    fn test_extract_local_validator_feedback_fail_with_issues() {
        let result = LocalValidationResult {
            model: "HydraCoder".into(),
            passed: false,
            schema_valid: true,
            feedback: "structured review".into(),
            blocking_issues: vec![
                "missing error handling in parse_config".into(),
                "logic error in boundary check".into(),
            ],
            suggested_next_action: "fix and re-run".into(),
            touched_files: vec!["src/config.rs".into()],
        };
        let feedback = extract_local_validator_feedback(&result);
        assert_eq!(feedback.len(), 2);
        assert_eq!(
            feedback[0].issue_type,
            ValidatorIssueType::MissingSafetyCheck
        );
        assert_eq!(feedback[1].issue_type, ValidatorIssueType::LogicError);
        assert_eq!(feedback[0].file.as_deref(), Some("src/config.rs"));
        assert_eq!(feedback[0].suggested_fix.as_deref(), Some("fix and re-run"));
        assert_eq!(feedback[0].source_model.as_deref(), Some("HydraCoder"));
    }

    #[test]
    fn test_extract_local_validator_feedback_malformed_wraps_as_single() {
        let result = LocalValidationResult {
            model: "test-model".into(),
            passed: false,
            schema_valid: false,
            feedback: "PASS\nlooks okay\nbut it was malformed".into(),
            blocking_issues: vec![],
            suggested_next_action: String::new(),
            touched_files: vec![],
        };
        let feedback = extract_local_validator_feedback(&result);
        assert_eq!(feedback.len(), 1);
        assert_eq!(feedback[0].issue_type, ValidatorIssueType::Other);
        assert!(feedback[0].description.contains("PASS"));
        assert!(feedback[0].description.contains("looks okay"));
    }
}
