//! Context Packer — Builds token-budgeted WorkPackets for agent tiers
//!
//! `pack_initial` creates a WorkPacket from scratch (no prior failures).
//! `pack_retry` delegates to the existing WorkPacketGenerator for error-enriched packets.

use crate::context_packer::file_walker::FileWalker;
use crate::escalation::state::{EscalationState, SwarmTier};
use crate::verifier::report::VerifierReport;
use crate::work_packet::generator::WorkPacketGenerator;
use crate::work_packet::types::{FileContext, WorkPacket};
use chrono::Utc;
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

/// Regex for extracting file paths from cargo stderr output (e.g., ` --> path/to/file.rs:123:45`).
static STDERR_FILE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-->\s*([^\s:]+\.rs):(\d+)").expect("STDERR_FILE_RE regex should compile")
});

/// Token budgets per tier (4 chars ≈ 1 token, matching `estimated_tokens()`)
fn max_context_tokens(tier: SwarmTier) -> usize {
    fn from_env(var: &str, default: usize) -> usize {
        std::env::var(var)
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0)
            .unwrap_or(default)
    }

    match tier {
        SwarmTier::Worker => from_env("SWARM_CONTEXT_TOKENS_WORKER", 8_000),
        SwarmTier::Council => from_env("SWARM_CONTEXT_TOKENS_COUNCIL", 32_000),
        SwarmTier::Human => from_env("SWARM_CONTEXT_TOKENS_HUMAN", 32_000),
    }
}

/// Builds token-budgeted context for agent tiers.
pub struct ContextPacker {
    working_dir: PathBuf,
    generator: WorkPacketGenerator,
    file_walker: FileWalker,
    tier: SwarmTier,
    max_context_tokens: usize,
}

impl ContextPacker {
    pub fn new(working_dir: impl AsRef<Path>, tier: SwarmTier) -> Self {
        let wd = working_dir.as_ref().to_path_buf();
        Self {
            generator: WorkPacketGenerator::new(&wd),
            file_walker: FileWalker::new(&wd),
            tier,
            max_context_tokens: max_context_tokens(tier),
            working_dir: wd,
        }
    }

    /// Initial pack: walks worktree, gathers file headers within token budget (no prior failures).
    pub fn pack_initial(&self, bead_id: &str, objective: &str) -> WorkPacket {
        let rust_files = self.file_walker.rust_files();

        // Make paths relative to working_dir for consistency with WorkPacketGenerator
        let relative_files: Vec<String> = rust_files
            .iter()
            .filter_map(|p| p.strip_prefix(&self.working_dir).ok())
            .map(|p| p.display().to_string())
            .collect();

        // Extract symbols from all .rs files (reuse generator's extract logic via generate)
        let state = EscalationState::new(bead_id);
        let mut packet = self
            .generator
            .generate(bead_id, objective, self.tier, &state, None);

        // The generator only gets symbols from git-changed files.
        // For initial pack, we want symbols from ALL .rs files, so build
        // file contexts manually from the full file list.
        packet.file_contexts = self.build_file_contexts(&relative_files);
        packet.objective = objective.to_string();
        packet.generated_at = Utc::now();

        // Trim to fit token budget
        self.trim_to_budget(&mut packet);
        packet
    }

    /// Retry pack: delegates to WorkPacketGenerator with error context,
    /// then overrides file contexts to include FULL content of changed/failing files.
    ///
    /// The generator's default `extract_error_contexts()` only includes ~20-line
    /// windows around error locations. For retries, the worker needs to see the
    /// full file content to fix cascading errors and understand the broader context.
    /// Without this, iteration 2+ gets ~300-700 tokens vs ~24K on iteration 1
    /// (the "retry context collapse" bug from job 1653).
    pub fn pack_retry(
        &self,
        bead_id: &str,
        objective: &str,
        escalation_state: &EscalationState,
        verifier_report: &VerifierReport,
    ) -> WorkPacket {
        let mut packet = self.generator.generate(
            bead_id,
            objective,
            escalation_state.current_tier,
            escalation_state,
            Some(verifier_report),
        );

        // Override file contexts with FULL content of error/changed files.
        // Priority: (1) files with errors, (2) files modified since initial commit.
        packet.file_contexts =
            self.build_retry_file_contexts(verifier_report, &packet.files_touched);

        self.trim_to_budget(&mut packet);
        packet
    }

    /// Build file contexts for retry iterations with FULL file content.
    ///
    /// Unlike `pack_initial` (which includes 30-line headers for many files),
    /// retry contexts include the COMPLETE content of the most relevant files
    /// so the worker can actually see and fix the code.
    fn build_retry_file_contexts(
        &self,
        report: &VerifierReport,
        files_touched: &[String],
    ) -> Vec<FileContext> {
        use std::collections::HashSet;

        let mut contexts = Vec::new();
        let mut files_seen = HashSet::new();
        let mut total_chars = 0usize;
        let char_budget = self.max_context_tokens * 4; // 4 chars per token

        // Canonical working_dir for sandbox validation
        let canon_wd = match self.working_dir.canonicalize() {
            Ok(p) => p,
            Err(_) => return contexts,
        };

        // 1. Files with errors — highest priority, include FULL content
        for signal in &report.failure_signals {
            if let Some(file) = &signal.file {
                if files_seen.contains(file) {
                    continue;
                }

                let full_path = self.working_dir.join(file);

                // Sandbox check: ensure resolved path stays within working_dir
                let canonical = match std::fs::canonicalize(&full_path) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                if !canonical.starts_with(&canon_wd) {
                    continue;
                }

                files_seen.insert(file.clone());

                let content = match std::fs::read_to_string(&canonical) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                // Pre-estimate: ~8 chars overhead per line (line num + " | " + newline)
                let estimated_chars =
                    content.len() + content.lines().count() * 8 + file.len() + 100;
                if total_chars + estimated_chars > char_budget {
                    continue; // skip oversized, try smaller files
                }

                let lines: Vec<&str> = content.lines().collect();
                let line_count = lines.len();
                let context_content: String = lines
                    .iter()
                    .enumerate()
                    .map(|(i, l)| format!("{:4} | {}", i + 1, l))
                    .collect::<Vec<_>>()
                    .join("\n");

                let ctx_chars = context_content.len() + file.len() + 100;
                total_chars += ctx_chars;

                let error_line = signal.line.unwrap_or(0);
                contexts.push(FileContext {
                    file: file.clone(),
                    start_line: 1,
                    end_line: line_count,
                    content: context_content,
                    relevance: format!(
                        "ERROR: {} at line {} (full file)",
                        signal.category, error_line
                    ),
                });
            }
        }

        // Also check gate stderr for file references not in failure_signals
        for gate in &report.gates {
            if let Some(stderr) = &gate.stderr_excerpt {
                // Extract file paths from stderr using --> path.rs:line pattern
                for cap in STDERR_FILE_RE.captures_iter(stderr) {
                    let file = cap[1].to_string();
                    let rel_file = file
                        .strip_prefix(&format!("{}/", self.working_dir.display()))
                        .unwrap_or(&file)
                        .to_string();

                    if files_seen.contains(&rel_file) {
                        continue;
                    }

                    let full_path = self.working_dir.join(&rel_file);

                    // Sandbox check: ensure resolved path stays within working_dir
                    let canonical = match std::fs::canonicalize(&full_path) {
                        Ok(p) => p,
                        Err(_) => continue,
                    };
                    if !canonical.starts_with(&canon_wd) {
                        continue;
                    }

                    files_seen.insert(rel_file.clone());

                    let content = match std::fs::read_to_string(&canonical) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };

                    // Pre-estimate before formatting
                    let estimated_chars =
                        content.len() + content.lines().count() * 8 + rel_file.len() + 100;
                    if total_chars + estimated_chars > char_budget {
                        continue; // skip oversized, try smaller files
                    }

                    let lines: Vec<&str> = content.lines().collect();
                    let line_count = lines.len();
                    let context_content: String = lines
                        .iter()
                        .enumerate()
                        .map(|(i, l)| format!("{:4} | {}", i + 1, l))
                        .collect::<Vec<_>>()
                        .join("\n");

                    let ctx_chars = context_content.len() + rel_file.len() + 100;
                    total_chars += ctx_chars;

                    let error_line: usize = cap[2].parse().unwrap_or(1);
                    contexts.push(FileContext {
                        file: rel_file,
                        start_line: 1,
                        end_line: line_count,
                        content: context_content,
                        relevance: format!(
                            "ERROR: {} gate at line {} (full file)",
                            gate.gate, error_line
                        ),
                    });
                }
            }
        }

        // 2. Changed files not already included — full content
        for file in files_touched {
            if !file.ends_with(".rs") || files_seen.contains(file) {
                continue;
            }

            let full_path = self.working_dir.join(file);
            let content = match std::fs::read_to_string(&full_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Pre-estimate before formatting
            let estimated_chars = content.len() + content.lines().count() * 8 + file.len() + 100;
            if total_chars + estimated_chars > char_budget {
                continue; // skip oversized, try smaller files
            }

            let lines: Vec<&str> = content.lines().collect();
            let line_count = lines.len();
            let context_content: String = lines
                .iter()
                .enumerate()
                .map(|(i, l)| format!("{:4} | {}", i + 1, l))
                .collect::<Vec<_>>()
                .join("\n");

            let ctx_chars = context_content.len() + file.len() + 100;
            total_chars += ctx_chars;

            contexts.push(FileContext {
                file: file.clone(),
                start_line: 1,
                end_line: line_count,
                content: context_content,
                relevance: "Modified file (full content for retry)".to_string(),
            });
        }

        contexts
    }

    /// Build file contexts from a list of relative paths (first 30 lines each).
    fn build_file_contexts(&self, files: &[String]) -> Vec<FileContext> {
        let mut contexts = Vec::new();
        let mut total_chars = 0usize;
        let char_budget = self.max_context_tokens * 4; // 4 chars per token

        for file in files {
            if !file.ends_with(".rs") {
                continue;
            }

            let full_path = self.working_dir.join(file);
            let content = match std::fs::read_to_string(&full_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            let lines: Vec<&str> = content.lines().collect();
            let end = lines.len().min(30);

            let context_content: String = lines[..end]
                .iter()
                .enumerate()
                .map(|(i, l)| format!("{:4} | {}", i + 1, l))
                .collect::<Vec<_>>()
                .join("\n");

            let ctx_chars = context_content.len() + file.len() + 50; // overhead
            if total_chars + ctx_chars > char_budget {
                break;
            }
            total_chars += ctx_chars;

            contexts.push(FileContext {
                file: file.clone(),
                start_line: 1,
                end_line: end,
                content: context_content,
                relevance: "Worktree file (header)".to_string(),
            });
        }

        contexts
    }

    /// Trim a WorkPacket to fit within the token budget.
    fn trim_to_budget(&self, packet: &mut WorkPacket) {
        // Drop file contexts from the end until we're under budget
        while packet.estimated_tokens() > self.max_context_tokens
            && !packet.file_contexts.is_empty()
        {
            packet.file_contexts.pop();
        }

        // If still over budget, truncate previous_attempts
        while packet.estimated_tokens() > self.max_context_tokens
            && !packet.previous_attempts.is_empty()
        {
            packet.previous_attempts.pop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::error_parser::ErrorCategory;
    use crate::verifier::report::{FailureSignal, GateOutcome, GateResult};
    use std::fs;

    #[test]
    fn test_pack_initial_creates_packet() {
        let dir = tempfile::tempdir().unwrap();
        // Create a minimal Rust file
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("lib.rs"), "pub struct Foo;\npub fn bar() {}\n").unwrap();

        // Initialize git repo so generator can query git
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

        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);
        let packet = packer.pack_initial("beads-test", "Test objective");

        assert_eq!(packet.bead_id, "beads-test");
        assert_eq!(packet.objective, "Test objective");
        // Should have found our lib.rs in the file contexts
        assert!(!packet.file_contexts.is_empty());
        assert!(packet.estimated_tokens() <= 8_000);
    }

    #[test]
    fn test_token_budgets() {
        assert_eq!(max_context_tokens(SwarmTier::Worker), 8_000);
        assert_eq!(max_context_tokens(SwarmTier::Council), 32_000);
        assert_eq!(max_context_tokens(SwarmTier::Human), 32_000);
    }

    #[test]
    fn test_trim_to_budget() {
        let dir = tempfile::tempdir().unwrap();
        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);

        let state = EscalationState::new("test");
        let mut packet =
            packer
                .generator
                .generate("test", "test obj", SwarmTier::Worker, &state, None);

        // Add a bunch of large file contexts to blow the budget
        for i in 0..100 {
            packet.file_contexts.push(FileContext {
                file: format!("src/big_{i}.rs"),
                start_line: 1,
                end_line: 100,
                content: "x".repeat(500),
                relevance: "test".to_string(),
            });
        }

        packer.trim_to_budget(&mut packet);
        assert!(packet.estimated_tokens() <= 8_000);
    }

    /// Helper: create a VerifierReport with failure signals pointing at specific files.
    fn make_report_with_failures(files: &[(&str, usize)], working_dir: &str) -> VerifierReport {
        let mut report = VerifierReport::new(working_dir.to_string());
        for (file, line) in files {
            report.failure_signals.push(FailureSignal {
                gate: "check".to_string(),
                category: ErrorCategory::TypeMismatch,
                code: Some("E0308".to_string()),
                file: Some(file.to_string()),
                line: Some(*line),
                message: "mismatched types".to_string(),
            });
        }
        report
    }

    #[test]
    fn test_retry_context_includes_full_error_file() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        let file_content = "fn main() {\n    let x: i32 = \"hello\";\n    println!(\"{x}\");\n}\n";
        fs::write(src.join("main.rs"), file_content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/main.rs", 2)], &dir.path().display().to_string());

        let contexts = packer.build_retry_file_contexts(&report, &[]);

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].file, "src/main.rs");
        assert_eq!(contexts[0].start_line, 1);
        assert_eq!(contexts[0].end_line, 4); // 4 lines
        assert!(contexts[0].content.contains("let x: i32"));
        assert!(contexts[0].relevance.contains("ERROR"));
    }

    #[test]
    fn test_retry_context_includes_touched_files() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("lib.rs"), "pub mod foo;\n").unwrap();
        fs::write(src.join("foo.rs"), "pub fn foo() {}\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report = VerifierReport::new(dir.path().display().to_string());
        let touched = vec!["src/lib.rs".to_string(), "src/foo.rs".to_string()];

        let contexts = packer.build_retry_file_contexts(&report, &touched);

        assert_eq!(contexts.len(), 2);
        let files: Vec<&str> = contexts.iter().map(|c| c.file.as_str()).collect();
        assert!(files.contains(&"src/lib.rs"));
        assert!(files.contains(&"src/foo.rs"));
    }

    #[test]
    fn test_retry_context_respects_budget() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        // Create files that exceed the Implementer budget (8K tokens = 32K chars)
        let big_content = "x".repeat(20_000); // ~20K chars each
        fs::write(src.join("a.rs"), &big_content).unwrap();
        fs::write(src.join("b.rs"), &big_content).unwrap();
        fs::write(src.join("c.rs"), &big_content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);
        let report = make_report_with_failures(
            &[("src/a.rs", 1), ("src/b.rs", 1), ("src/c.rs", 1)],
            &dir.path().display().to_string(),
        );

        let contexts = packer.build_retry_file_contexts(&report, &[]);

        // Budget is 32K chars — each file ~20K + formatting overhead.
        // Should include 1 file, skip the rest.
        assert!(
            contexts.len() < 3,
            "Should have trimmed to fit budget, got {}",
            contexts.len()
        );
        let total_chars: usize = contexts.iter().map(|c| c.content.len()).sum();
        assert!(
            total_chars <= 32_000 + 1000,
            "Total chars {total_chars} exceeds budget"
        );
    }

    #[test]
    fn test_retry_context_skips_oversized_includes_smaller() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        // One huge file that exceeds budget, one small file that fits
        let huge_content = "x\n".repeat(50_000); // ~100K chars
        let small_content = "fn small() {}\n";
        fs::write(src.join("huge.rs"), &huge_content).unwrap();
        fs::write(src.join("small.rs"), small_content).unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Worker);
        let report = make_report_with_failures(
            &[("src/huge.rs", 1), ("src/small.rs", 1)],
            &dir.path().display().to_string(),
        );

        let contexts = packer.build_retry_file_contexts(&report, &[]);

        // The huge file should be skipped (continue, not break), small file included
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].file, "src/small.rs");
    }

    #[test]
    fn test_retry_context_deduplicates_error_and_touched() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("main.rs"), "fn main() {}\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let report =
            make_report_with_failures(&[("src/main.rs", 1)], &dir.path().display().to_string());
        // Same file in both failure_signals and files_touched
        let touched = vec!["src/main.rs".to_string()];

        let contexts = packer.build_retry_file_contexts(&report, &touched);

        // Should appear only once
        assert_eq!(contexts.len(), 1);
    }

    #[test]
    fn test_retry_context_extracts_from_gate_stderr() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(src.join("lib.rs"), "pub fn broken() -> i32 { \"oops\" }\n").unwrap();

        let packer = ContextPacker::new(dir.path(), SwarmTier::Council);
        let mut report = VerifierReport::new(dir.path().display().to_string());
        report.gates.push(GateResult {
            gate: "check".to_string(),
            outcome: GateOutcome::Failed,
            duration_ms: 100,
            exit_code: Some(1),
            error_count: 1,
            warning_count: 0,
            errors: vec![],
            stderr_excerpt: Some(format!(
                "error[E0308]: mismatched types\n --> src/lib.rs:1:26\n"
            )),
        });

        let contexts = packer.build_retry_file_contexts(&report, &[]);

        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].file, "src/lib.rs");
        assert!(contexts[0].content.contains("broken"));
    }
}
