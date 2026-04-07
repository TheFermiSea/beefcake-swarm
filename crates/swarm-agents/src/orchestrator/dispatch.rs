//! Task routing and prompt formatting for worker dispatch.
//!
//! Responsible for selecting the right worker (RustCoder vs GeneralCoder) and
//! building task prompts in both verbose (cloud) and compact (local) formats.

use std::path::Path;

use tracing::{debug, info};

use crate::file_targeting::find_target_files_by_grep;
use coordination::feedback::ErrorCategory;
#[cfg(test)]
use coordination::verifier::report::GateOutcome;
use coordination::verifier::report::VerifierReport;
use coordination::WorkPacket;

/// Coder routing decision with confidence level.
#[derive(Debug, PartialEq, Eq)]
pub enum CoderRoute {
    /// Qwen3.5-122B-A10B with Rust specialist system prompt
    RustCoder,
    /// Qwen3.5-122B-A10B with general coder system prompt (multi-file scaffolding)
    GeneralCoder,
    /// GLM-4.7-Flash fast tier for simple errors on retry iterations
    /// (reasoning sandwich: save coder budget for first attempts and complex errors)
    FastFixer,
}

/// Route to the appropriate coder based on error category distribution and iteration.
///
/// Uses a weighted scoring system rather than a simple `any()` check:
/// - Rust-specific categories (borrow checker, lifetimes, traits) score toward RustCoder system prompt
/// - Structural categories (imports, syntax, macros) score toward GeneralCoder system prompt
/// - Mixed errors with majority Rust → RustCoder; majority structural → GeneralCoder
/// - No errors (first iteration) → general coder for scaffolding
///
/// **Reasoning sandwich** (iteration > 1): On retry iterations, simple errors
/// (type mismatch, imports, syntax) route to the fast tier (GLM-4.7-Flash)
/// instead of burning coder budget. Complex errors (borrow checker, lifetimes,
/// trait bounds, async) still use the coder tier on retry.
pub fn route_to_coder(error_cats: &[ErrorCategory], iteration: u32) -> CoderRoute {
    if error_cats.is_empty() {
        // First iteration — use general coder for scaffolding/multi-file work
        return CoderRoute::GeneralCoder;
    }

    // Reasoning sandwich: on retries with only simple errors, use the fast tier.
    // Complex errors (borrow checker, lifetimes, trait bounds, async) require
    // the coder tier's deeper expertise even on retry.
    if iteration > 1 {
        let all_simple = error_cats.iter().all(is_simple_error);
        if all_simple {
            info!(
                iteration,
                errors = ?error_cats,
                "Reasoning sandwich: routing retry to fast tier (simple errors)"
            );
            return CoderRoute::FastFixer;
        }
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

/// Whether an error category is "simple" — fixable by the fast tier without
/// deep Rust expertise. These are typically mechanical fixes: wrong type,
/// missing import, syntax typo, unused variable, etc.
fn is_simple_error(cat: &ErrorCategory) -> bool {
    matches!(
        cat,
        ErrorCategory::TypeMismatch
            | ErrorCategory::ImportResolution
            | ErrorCategory::Syntax
            | ErrorCategory::Other
    )
}

/// Format a WorkPacket into a structured prompt for agent consumption.
pub fn format_task_prompt(packet: &WorkPacket) -> String {
    let mut prompt = String::new();
    let scoped_files: Vec<&String> = packet
        .files_touched
        .iter()
        .filter(|f| {
            let path = f.as_str();
            !(path == ".beads"
                || path.starts_with(".beads/")
                || path == ".git"
                || path.starts_with(".git/")
                || path == ".claude"
                || path.starts_with(".claude/")
                || path == ".dolt"
                || path.starts_with(".dolt/"))
        })
        .collect();

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
        prompt.push_str("## What We've Already Tried (do NOT repeat these)\n");
        prompt.push_str("These approaches were attempted and failed. Do not repeat them:\n");
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

    // Repository Map — whole-codebase structure showing public symbols by file.
    // Gives the agent a complete mental model without reading files one by one.
    if let Some(ref repo_map) = packet.repo_map {
        prompt.push_str("## Repository Map\n");
        prompt.push_str(
            "_Codebase structure ranked by relevance. Use this to locate code without reading files blindly._\n",
        );
        prompt.push_str(repo_map);
        prompt.push('\n');
    }

    if let Some(ref dep_graph) = packet.dependency_graph {
        prompt.push_str("## Dependency Graph\n");
        prompt.push_str(dep_graph);
        prompt.push('\n');
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
    if !scoped_files.is_empty() {
        prompt.push_str("## Scope Constraints\n");
        prompt.push_str("**IMPORTANT:** Only modify these files:\n");
        for f in scoped_files {
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

    // --- Skill hints from past successful resolutions (Hyperagents pattern) ---
    if !packet.skill_hints.is_empty() {
        prompt.push_str("\n## Recommended Approaches (from past successes)\n\n");
        for hint in &packet.skill_hints {
            prompt.push_str(&format!(
                "- [confidence: {:.0}%] {}: {}\n",
                hint.confidence * 100.0,
                hint.label,
                hint.approach,
            ));
        }
        prompt.push('\n');
    }

    prompt.push_str(&format!(
        "**Max patch size:** {} LOC\n\n",
        packet.max_patch_loc
    ));

    prompt.push_str(
        "**STOP RULE**: Once you have applied your changes with edit_file or write_file, \
         YOU ARE DONE. Do NOT call any more tools. Do NOT run cargo check, cargo test, \
         or any verification commands — the orchestrator runs verification automatically. \
         After your edit succeeds, immediately return a brief summary of what you changed.\n",
    );

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
    // then grep the worktree for objective identifiers,
    // then fall back to first few file_contexts filenames.
    // Filter out metadata paths (e.g., .beads/, .git/) that aren't code targets.
    let source_files_touched: Vec<String> = packet
        .files_touched
        .iter()
        .filter(|f| {
            !f.starts_with(".beads") && !f.starts_with(".git/") && !f.starts_with(".claude")
        })
        .cloned()
        .collect();
    let target_files: Vec<String> = if !source_files_touched.is_empty() {
        source_files_touched
    } else {
        // Try to extract file paths from objective (e.g., "File: src/foo.rs")
        // Accepts both full paths (crates/swarm-agents/src/foo.rs) and bare
        // filenames (runtime_adapter.rs). Bare filenames are resolved by
        // searching the worktree.
        let objective_files: Vec<String> = packet
            .objective
            .split(|c: char| c.is_whitespace() || c == ',')
            .map(|w| {
                w.trim_end_matches(|c: char| {
                    c.is_ascii_punctuation()
                        && c != '/'
                        && c != '.'
                        && c != '_'
                        && c != '-'
                        && c != '*'
                })
                .trim()
            })
            .filter(|w| !w.is_empty() && (w.ends_with(".rs") || w.ends_with(".toml")))
            .flat_map(|w| {
                if w.contains('/') {
                    // Full path — use as-is
                    vec![w.to_string()]
                } else {
                    // Bare filename — search worktree for matching files
                    match std::process::Command::new("find")
                        .args([
                            wt_root.to_str().unwrap_or("."),
                            "-name",
                            w,
                            "-path",
                            "*/src/*",
                        ])
                        .output()
                    {
                        Ok(output) if output.status.success() => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            stdout
                                .lines()
                                .take(3)
                                .filter_map(|line| {
                                    line.strip_prefix(wt_root.to_str().unwrap_or(""))
                                        .map(|p| p.trim_start_matches('/').to_string())
                                })
                                .collect()
                        }
                        _ => vec![],
                    }
                }
            })
            .collect();
        if !objective_files.is_empty() {
            objective_files
        } else {
            // Grep the worktree for CamelCase identifiers from the objective.
            // The context packer's file_contexts only covers ~18 files (token budget)
            // and likely misses the target file in large workspaces (402 .rs files).
            find_target_files_by_grep(wt_root, &packet.objective, &[]).unwrap_or_else(|| {
                // Final fallback: first 3 file_contexts
                packet
                    .file_contexts
                    .iter()
                    .map(|fc| fc.file.clone())
                    .take(3)
                    .collect()
            })
        }
    };

    debug!(
        raw_files_touched = ?packet.files_touched,
        target_files = ?target_files,
        "compact prompt: file targeting pipeline"
    );

    // Guard against empty target_files — fall back to common entry points.
    let target_files: Vec<String> = if target_files.is_empty() {
        ["src/lib.rs", "src/main.rs"]
            .iter()
            .filter(|p| wt_root.join(p).exists())
            .map(|p| p.to_string())
            .chain(std::iter::once("Cargo.toml".to_string()))
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

    // Search tool hints — workers have these but won't use them without prompting.
    prompt.push_str(
        "**Search tools**: Use `colgrep` for semantic search, `search_code` for exact patterns, \
         `ast_grep` for structural code patterns (e.g. `$EXPR.unwrap()`).\n\n",
    );

    // Explicit stop instruction — critical for local LLMs that don't self-terminate.
    prompt.push_str(
        "**STOP RULE**: Once you have applied your changes with edit_file or write_file, \
         YOU ARE DONE. Do NOT call any more tools. Do NOT run cargo check, cargo test, \
         or any verification commands — the orchestrator runs verification automatically. \
         After your edit succeeds, immediately return a brief summary of what you changed.\n\n",
    );

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

    // --- Skill hints (compact) ---
    if !packet.skill_hints.is_empty() {
        prompt.push_str("\n<skill_hints>\n");
        for hint in &packet.skill_hints {
            prompt.push_str(&format!(
                "- [{:.0}%] {}: {}\n",
                hint.confidence * 100.0,
                hint.label,
                hint.approach
            ));
        }
        prompt.push_str("</skill_hints>\n");
    }

    // Inline the first (most relevant) target file content to save read_file turns.
    // Critical with max_turns_without_write=8: agent must write within 8 turns.
    if !target_files.is_empty() {
        let target_path = wt_root.join(&target_files[0]);
        if let Ok(content) = std::fs::read_to_string(&target_path) {
            let truncated = if content.len() > 4000 {
                let cut = crate::str_util::safe_truncate(&content, 4000);
                format!("{cut}...\n[truncated at {} chars]", cut.len())
            } else {
                content
            };
            prompt.push_str(&format!(
                "**Current content of `{}`:**\n```\n{}\n```\n\n",
                target_files[0], truncated
            ));
            if target_files.len() == 1 {
                prompt.push_str(
                    "Apply your edits directly with edit_file — the file content is above.\n",
                );
            } else {
                prompt.push_str(
                    "Apply your edits with edit_file. Use read_file for the other target files if needed.\n",
                );
            }
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

/// Condense a `VerifierReport` into a short summary suitable for retry prompts.
///
/// Outputs gate statuses on one line, followed by up to 3 key errors (each
/// truncated to 150 chars). Keeps total output well under 500 chars so it
/// fits in compact worker prompts without suppressing tool-call generation.
pub fn condense_verifier_report(report: &VerifierReport) -> String {
    let mut lines = Vec::new();

    // Gate status line: "fmt PASS, clippy FAIL, check SKIP, test SKIP"
    let gates: Vec<String> = report
        .gates
        .iter()
        .map(|g| format!("{} {}", g.gate, g.outcome))
        .collect();
    lines.push(format!("Gates: {}", gates.join(", ")));

    // Top 3 failure signals (most actionable)
    let errors: Vec<&str> = report
        .failure_signals
        .iter()
        .take(3)
        .map(|s| s.message.as_str())
        .collect();
    if !errors.is_empty() {
        lines.push("Key errors:".to_string());
        for e in errors {
            let truncated = crate::str_util::safe_truncate(e, 150);
            lines.push(format!("  - {truncated}"));
        }
    }

    lines.join("\n")
}

/// Build a review prompt for the pre-merge quality check.
///
/// The reviewer sees the issue context and an abbreviated diff, then decides
/// whether the change should merge. Response is expected as JSON with
/// `{"approve": true/false, "reason": "..."}`.
pub fn build_review_prompt(
    issue_title: &str,
    issue_description: &str,
    diff_summary: &str,
) -> String {
    format!(
        "## Pre-Merge Review\n\n\
         You are reviewing a code change before it merges to main.\n\n\
         **Issue:** {issue_title}\n\
         **Description:** {issue_description}\n\n\
         **Diff:**\n```\n{diff_summary}\n```\n\n\
         Check:\n\
         1. Does the diff address the issue? (yes/no + brief reason)\n\
         2. Are there unrelated changes? (yes/no)\n\
         3. Any obvious bugs or problems? (yes/no + what)\n\n\
         Respond with JSON only: {{\"approve\": true/false, \"reason\": \"...\"}}"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use coordination::SwarmTier;

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
            repo_map: None,
            failed_approach_summary: None,
            dependency_graph: None,
        }
    }

    #[test]
    fn test_route_empty_errors_to_general() {
        assert_eq!(route_to_coder(&[], 1), CoderRoute::GeneralCoder);
    }

    #[test]
    fn test_route_borrow_checker_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::BorrowChecker], 1),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_lifetime_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Lifetime], 1),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_trait_bound_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::TraitBound], 1),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_async_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Async], 1),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_full_prompt_filters_metadata_scope_paths() {
        let mut packet = test_packet("Fix issue", "swarm/test", 1);
        packet.files_touched = vec![
            ".beads/backup/backup_state.json".into(),
            ".git/index".into(),
            "coordination/src/lib.rs".into(),
        ];

        let prompt = format_task_prompt(&packet);

        assert!(prompt.contains("`coordination/src/lib.rs`"));
        assert!(!prompt.contains(".beads/backup/backup_state.json"));
        assert!(!prompt.contains(".git/index"));
    }

    #[test]
    fn test_route_import_to_general() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::ImportResolution], 1),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_route_macro_to_general() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Macro], 1),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_route_syntax_to_general() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Syntax], 1),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_route_other_to_general() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::Other], 1),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_route_type_mismatch_alone_to_rust() {
        assert_eq!(
            route_to_coder(&[ErrorCategory::TypeMismatch], 1),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_route_mixed_rust_heavy() {
        // BorrowChecker(+3) + Import(+3) → tie → general wins (>= check)
        assert_eq!(
            route_to_coder(
                &[
                    ErrorCategory::BorrowChecker,
                    ErrorCategory::ImportResolution
                ],
                1
            ),
            CoderRoute::GeneralCoder
        );
        // BorrowChecker(+3) + Lifetime(+3) + Import(+3) → 6 vs 3 → rust
        assert_eq!(
            route_to_coder(
                &[
                    ErrorCategory::BorrowChecker,
                    ErrorCategory::Lifetime,
                    ErrorCategory::ImportResolution
                ],
                1
            ),
            CoderRoute::RustCoder
        );
    }

    // ── Reasoning sandwich tests ────────────────────────────────────────

    #[test]
    fn test_retry_simple_errors_route_to_fast_fixer() {
        // Type mismatch on retry → fast tier
        assert_eq!(
            route_to_coder(&[ErrorCategory::TypeMismatch], 2),
            CoderRoute::FastFixer
        );
        // Import errors on retry → fast tier
        assert_eq!(
            route_to_coder(&[ErrorCategory::ImportResolution], 3),
            CoderRoute::FastFixer
        );
        // Syntax errors on retry → fast tier
        assert_eq!(
            route_to_coder(&[ErrorCategory::Syntax], 2),
            CoderRoute::FastFixer
        );
        // Other errors on retry → fast tier
        assert_eq!(
            route_to_coder(&[ErrorCategory::Other], 4),
            CoderRoute::FastFixer
        );
    }

    #[test]
    fn test_retry_complex_errors_stay_on_coder() {
        // Borrow checker on retry → still coder (complex)
        assert_eq!(
            route_to_coder(&[ErrorCategory::BorrowChecker], 2),
            CoderRoute::RustCoder
        );
        // Lifetime on retry → still coder
        assert_eq!(
            route_to_coder(&[ErrorCategory::Lifetime], 3),
            CoderRoute::RustCoder
        );
        // Trait bound on retry → still coder
        assert_eq!(
            route_to_coder(&[ErrorCategory::TraitBound], 2),
            CoderRoute::RustCoder
        );
        // Async on retry → still coder
        assert_eq!(
            route_to_coder(&[ErrorCategory::Async], 2),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_retry_mixed_errors_with_complex_stay_on_coder() {
        // TypeMismatch + BorrowChecker on retry → not all simple → coder
        assert_eq!(
            route_to_coder(
                &[ErrorCategory::TypeMismatch, ErrorCategory::BorrowChecker],
                2
            ),
            CoderRoute::RustCoder
        );
    }

    #[test]
    fn test_first_iteration_simple_errors_use_coder_not_fast() {
        // TypeMismatch on iteration 1 → coder (not fast), even though simple
        assert_eq!(
            route_to_coder(&[ErrorCategory::TypeMismatch], 1),
            CoderRoute::RustCoder
        );
        // ImportResolution on iteration 1 → general coder (not fast)
        assert_eq!(
            route_to_coder(&[ErrorCategory::ImportResolution], 1),
            CoderRoute::GeneralCoder
        );
    }

    #[test]
    fn test_retry_multiple_simple_errors_route_to_fast() {
        // Multiple simple errors on retry → all simple → fast tier
        assert_eq!(
            route_to_coder(
                &[
                    ErrorCategory::TypeMismatch,
                    ErrorCategory::ImportResolution,
                    ErrorCategory::Syntax
                ],
                2
            ),
            CoderRoute::FastFixer
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
    fn test_condense_verifier_report_mixed_gates() {
        use coordination::feedback::error_parser::ParsedError;
        use coordination::verifier::report::GateResult;
        use std::time::Duration;

        let mut report = VerifierReport::new("/tmp/test".to_string());
        report.add_gate(GateResult {
            gate: "fmt".to_string(),
            outcome: GateOutcome::Passed,
            duration_ms: 50,
            exit_code: Some(0),
            error_count: 0,
            warning_count: 0,
            errors: vec![],
            stderr_excerpt: None,
        });
        report.add_gate(GateResult {
            gate: "clippy".to_string(),
            outcome: GateOutcome::Failed,
            duration_ms: 200,
            exit_code: Some(1),
            error_count: 1,
            warning_count: 0,
            errors: vec![ParsedError {
                category: ErrorCategory::Other,
                code: Some("clippy::needless_return".to_string()),
                message: "unneeded `return` statement".to_string(),
                file: Some("src/lib.rs".to_string()),
                line: Some(10),
                column: Some(5),
                suggestion: None,
                rendered: String::new(),
                labels: vec![],
            }],
            stderr_excerpt: None,
        });
        report.add_gate(GateResult {
            gate: "check".to_string(),
            outcome: GateOutcome::Skipped,
            duration_ms: 0,
            exit_code: None,
            error_count: 0,
            warning_count: 0,
            errors: vec![],
            stderr_excerpt: None,
        });
        report.finalize(Duration::from_millis(250));

        let summary = condense_verifier_report(&report);
        assert!(summary.contains("fmt PASS"));
        assert!(summary.contains("clippy FAIL"));
        assert!(summary.contains("check SKIP"));
        assert!(summary.contains("Key errors:"));
        assert!(summary.contains("unneeded `return` statement"));
    }

    #[test]
    fn test_condense_verifier_report_all_green() {
        use std::time::Duration;

        let mut report = VerifierReport::new("/tmp/test".to_string());
        for name in &["fmt", "clippy", "check", "test"] {
            report.add_gate(coordination::verifier::report::GateResult {
                gate: name.to_string(),
                outcome: GateOutcome::Passed,
                duration_ms: 100,
                exit_code: Some(0),
                error_count: 0,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: None,
            });
        }
        report.finalize(Duration::from_millis(400));

        let summary = condense_verifier_report(&report);
        assert!(summary.contains("fmt PASS"));
        assert!(summary.contains("test PASS"));
        assert!(!summary.contains("Key errors:"));
    }

    #[test]
    fn test_condense_verifier_report_truncates_long_errors() {
        use coordination::feedback::error_parser::ParsedError;
        use coordination::verifier::report::GateResult;
        use std::time::Duration;

        let long_msg = "a".repeat(200);
        let mut report = VerifierReport::new("/tmp/test".to_string());
        report.add_gate(GateResult {
            gate: "check".to_string(),
            outcome: GateOutcome::Failed,
            duration_ms: 100,
            exit_code: Some(1),
            error_count: 1,
            warning_count: 0,
            errors: vec![ParsedError {
                category: ErrorCategory::TypeMismatch,
                code: Some("E0308".to_string()),
                message: long_msg.clone(),
                file: Some("src/main.rs".to_string()),
                line: Some(1),
                column: Some(1),
                suggestion: None,
                rendered: String::new(),
                labels: vec![],
            }],
            stderr_excerpt: None,
        });
        report.finalize(Duration::from_millis(100));

        let summary = condense_verifier_report(&report);
        // Should truncate to 150 chars, not include full 200
        assert!(!summary.contains(&long_msg));
        assert!(summary.contains(&"a".repeat(150)));
    }
}
