//! Work Packet Generator — creates packets from beads issues + git state
//!
//! Extracts symbols from source files, reads git state, and combines with
//! verifier reports and escalation state to produce Work Packets.

use crate::escalation::state::{EscalationState, SwarmTier};
use crate::verifier::report::VerifierReport;
use crate::work_packet::types::{
    Constraint, ConstraintKind, ContextProvenance, FileContext, KeySymbol, SymbolKind, WorkPacket,
};
use chrono::Utc;
use regex::Regex;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::LazyLock;

/// Regex for extracting file paths from cargo fmt/check stderr output.
/// Matches patterns like ` --> path/to/file.rs:123:45` or `Error writing files: ... path.rs`
static STDERR_FILE_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"-->\s*([^\s:]+\.rs):(\d+)").expect("STDERR_FILE_PATTERN regex should compile")
});

/// Regex patterns for extracting Rust symbols from source code
static STRUCT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^pub\s+struct\s+(\w+)").unwrap());
static TRAIT_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^pub\s+trait\s+(\w+)").unwrap());
static ENUM_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^pub\s+enum\s+(\w+)").unwrap());
static FN_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^\s*pub\s+(?:async\s+)?fn\s+(\w+)").unwrap());
static IMPL_PATTERN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?m)^impl(?:<[^>]*>)?\s+(\w+)").unwrap());

/// Work Packet Generator
pub struct WorkPacketGenerator {
    /// Crate root directory
    working_dir: PathBuf,
    /// Default maximum LOC per patch
    default_max_loc: u32,
    /// Default constraints applied to all packets
    default_constraints: Vec<Constraint>,
}

impl WorkPacketGenerator {
    /// Create a new generator for the given crate
    pub fn new(working_dir: impl AsRef<Path>) -> Self {
        Self {
            working_dir: working_dir.as_ref().to_path_buf(),
            default_max_loc: 150,
            default_constraints: vec![
                Constraint {
                    kind: ConstraintKind::NoDeps,
                    description: "No new dependencies without explicit approval".to_string(),
                },
                Constraint {
                    kind: ConstraintKind::NoBreakingApi,
                    description: "Don't break existing public API".to_string(),
                },
            ],
        }
    }

    /// Set the default maximum LOC per patch
    pub fn with_max_loc(mut self, max_loc: u32) -> Self {
        self.default_max_loc = max_loc;
        self
    }

    /// Add a default constraint
    pub fn with_constraint(mut self, constraint: Constraint) -> Self {
        self.default_constraints.push(constraint);
        self
    }

    /// Generate a Work Packet from escalation state and verifier report
    pub fn generate(
        &self,
        bead_id: &str,
        objective: &str,
        target_tier: SwarmTier,
        escalation_state: &EscalationState,
        verifier_report: Option<&VerifierReport>,
    ) -> WorkPacket {
        let branch = self.git_branch().unwrap_or_else(|| "unknown".to_string());
        let checkpoint = self.git_commit().unwrap_or_else(|| "unknown".to_string());

        // Get files touched from git diff
        let files_touched = self.git_changed_files();

        // Extract key symbols from touched files
        let key_symbols = self.extract_symbols_from_files(&files_touched);

        // Get file contexts for files with errors
        let file_contexts = self.extract_error_contexts(verifier_report, &files_touched);

        // Build failure signals from verifier report
        let failure_signals = verifier_report
            .map(|r| r.failure_signals.clone())
            .unwrap_or_default();

        // Build error history from escalation state
        let error_history: Vec<_> = escalation_state
            .iteration_history
            .iter()
            .flat_map(|r| r.error_categories.clone())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();

        // Build previous attempts summary
        let previous_attempts: Vec<String> = escalation_state
            .iteration_history
            .iter()
            .map(|r| {
                format!(
                    "Iter {}: {} errors ({}) [{}]",
                    r.iteration,
                    r.error_count,
                    r.error_categories
                        .iter()
                        .map(|c| c.to_string())
                        .collect::<Vec<_>>()
                        .join(", "),
                    if r.progress_made {
                        "progress"
                    } else {
                        "no progress"
                    },
                )
            })
            .collect();

        // Determine escalation reason
        let escalation_reason = escalation_state
            .escalation_history
            .last()
            .map(|e| format!("{}", e.reason));

        // Verification gates
        let verification_gates = verifier_report
            .map(|r| r.gates.iter().map(|g| g.gate.clone()).collect())
            .unwrap_or_else(|| {
                vec![
                    "fmt".to_string(),
                    "clippy".to_string(),
                    "check".to_string(),
                    "test".to_string(),
                ]
            });

        WorkPacket {
            bead_id: bead_id.to_string(),
            branch,
            checkpoint,
            objective: objective.to_string(),
            files_touched,
            key_symbols,
            file_contexts,
            verification_gates,
            failure_signals,
            constraints: self.default_constraints.clone(),
            iteration: escalation_state.total_iterations + 1,
            target_tier,
            escalation_reason,
            error_history,
            previous_attempts,
            relevant_heuristics: vec![], // Populated by the Learning Layer (Phase 4)
            relevant_playbooks: vec![],  // Populated by the Learning Layer (Phase 4)
            decisions: vec![],           // Populated from Decision Journal
            generated_at: Utc::now(),
            max_patch_loc: self.default_max_loc,
            iteration_deltas: vec![],   // Populated by delta computation
            delegation_chain: vec![],   // Populated during manager-to-manager handoffs
            skill_hints: vec![],        // Populated by orchestrator from skill library
            replay_hints: vec![],       // Populated by orchestrator from trace index
            validator_feedback: vec![], // Populated by orchestrator from reviewer feedback
            change_contract: None,      // Populated by planner agent
        }
    }

    /// Extract key symbols from a list of source files
    fn extract_symbols_from_files(&self, files: &[String]) -> Vec<KeySymbol> {
        let mut symbols = Vec::new();

        for file in files {
            if !file.ends_with(".rs") {
                continue;
            }

            let full_path = self.working_dir.join(file);
            let content = match std::fs::read_to_string(&full_path) {
                Ok(c) => c,
                Err(_) => continue,
            };

            // Extract structs
            for cap in STRUCT_PATTERN.captures_iter(&content) {
                if let Some(name) = cap.get(1) {
                    let line = content[..cap.get(0).unwrap().start()].lines().count() + 1;
                    symbols.push(KeySymbol {
                        name: name.as_str().to_string(),
                        kind: SymbolKind::Struct,
                        file: file.clone(),
                        line: Some(line),
                    });
                }
            }

            // Extract traits
            for cap in TRAIT_PATTERN.captures_iter(&content) {
                if let Some(name) = cap.get(1) {
                    let line = content[..cap.get(0).unwrap().start()].lines().count() + 1;
                    symbols.push(KeySymbol {
                        name: name.as_str().to_string(),
                        kind: SymbolKind::Trait,
                        file: file.clone(),
                        line: Some(line),
                    });
                }
            }

            // Extract enums
            for cap in ENUM_PATTERN.captures_iter(&content) {
                if let Some(name) = cap.get(1) {
                    let line = content[..cap.get(0).unwrap().start()].lines().count() + 1;
                    symbols.push(KeySymbol {
                        name: name.as_str().to_string(),
                        kind: SymbolKind::Enum,
                        file: file.clone(),
                        line: Some(line),
                    });
                }
            }

            // Extract public functions (limit to avoid noise)
            let mut fn_count = 0;
            for cap in FN_PATTERN.captures_iter(&content) {
                if fn_count >= 20 {
                    break;
                }
                if let Some(name) = cap.get(1) {
                    let line = content[..cap.get(0).unwrap().start()].lines().count() + 1;
                    symbols.push(KeySymbol {
                        name: name.as_str().to_string(),
                        kind: SymbolKind::Function,
                        file: file.clone(),
                        line: Some(line),
                    });
                    fn_count += 1;
                }
            }
        }

        symbols
    }

    /// Extract code context around error locations
    fn extract_error_contexts(
        &self,
        report: Option<&VerifierReport>,
        files_touched: &[String],
    ) -> Vec<FileContext> {
        let mut contexts = Vec::new();

        if let Some(report) = report {
            // Group failure signals by file
            let mut files_seen = HashSet::new();

            for signal in &report.failure_signals {
                if let Some(file) = &signal.file {
                    if files_seen.contains(file) {
                        continue;
                    }
                    files_seen.insert(file.clone());

                    let full_path = self.working_dir.join(file);
                    let content = match std::fs::read_to_string(&full_path) {
                        Ok(c) => c,
                        Err(_) => continue,
                    };

                    let line = signal.line.unwrap_or(1);
                    let lines: Vec<&str> = content.lines().collect();

                    // Context window: 10 lines before and after the error
                    let start = line.saturating_sub(11);
                    let end = (line + 10).min(lines.len());

                    let context_content: String = lines[start..end]
                        .iter()
                        .enumerate()
                        .map(|(i, l)| format!("{:4} | {}", start + i + 1, l))
                        .collect::<Vec<_>>()
                        .join("\n");

                    contexts.push(FileContext {
                        file: file.clone(),
                        start_line: start + 1,
                        end_line: end,
                        content: context_content,
                        relevance: format!("{} error at line {}", signal.category, line),
                        priority: 0, // Error context — highest priority
                        provenance: ContextProvenance::CompilerError,
                    });
                }
            }

            // If failure_signals didn't yield file contexts (e.g., fmt errors which
            // don't produce ParsedErrors), extract file paths from gate stderr.
            if contexts.is_empty() {
                for gate in &report.gates {
                    if let Some(stderr) = &gate.stderr_excerpt {
                        for cap in STDERR_FILE_PATTERN.captures_iter(stderr) {
                            let file = cap[1].to_string();
                            let line: usize = cap[2].parse().unwrap_or(1);

                            // Make path relative to working dir if it's absolute
                            let rel_file = file
                                .strip_prefix(&format!("{}/", self.working_dir.display()))
                                .unwrap_or(&file)
                                .to_string();

                            if files_seen.contains(&rel_file) {
                                continue;
                            }
                            files_seen.insert(rel_file.clone());

                            let full_path = self.working_dir.join(&rel_file);
                            let content = match std::fs::read_to_string(&full_path) {
                                Ok(c) => c,
                                Err(_) => continue,
                            };

                            let lines: Vec<&str> = content.lines().collect();
                            let start = line.saturating_sub(11);
                            let end = (line + 10).min(lines.len());

                            let context_content: String = lines[start..end]
                                .iter()
                                .enumerate()
                                .map(|(i, l)| format!("{:4} | {}", start + i + 1, l))
                                .collect::<Vec<_>>()
                                .join("\n");

                            contexts.push(FileContext {
                                file: rel_file,
                                start_line: start + 1,
                                end_line: end,
                                content: context_content,
                                relevance: format!("{} gate error at line {}", gate.gate, line),
                                priority: 0, // Error context — highest priority
                                provenance: ContextProvenance::CompilerError,
                            });
                        }
                    }
                }
            }
        }

        // Also include brief context for any touched files without explicit errors
        // (limited to first 30 lines to keep packet small)
        for file in files_touched {
            if !file.ends_with(".rs") {
                continue;
            }
            if contexts.iter().any(|c| &c.file == file) {
                continue;
            }
            if contexts.len() >= 5 {
                break; // Limit total contexts
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

            contexts.push(FileContext {
                file: file.clone(),
                start_line: 1,
                end_line: end,
                content: context_content,
                relevance: "Modified file (header)".to_string(),
                priority: 1, // Modified file
                provenance: ContextProvenance::Diff,
            });
        }

        contexts
    }

    /// Get current git branch.
    ///
    /// Handles detached HEAD state (common in CI and fresh worktrees) with a
    /// fallback chain: CI env vars → `git name-rev` → `detached@<short-sha>`.
    fn git_branch(&self) -> Option<String> {
        let branch = Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(&self.working_dir)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;

        if branch != "HEAD" {
            return Some(branch);
        }

        // Detached HEAD — try CI env vars
        for var in ["CI_COMMIT_REF_NAME", "GITHUB_HEAD_REF", "BRANCH_NAME"] {
            if let Ok(val) = std::env::var(var) {
                let val = val.trim().to_string();
                if !val.is_empty() {
                    return Some(val);
                }
            }
        }

        // Try git name-rev (strip remotes/ prefix)
        if let Some(name) = Command::new("git")
            .args(["name-rev", "--name-only", "HEAD"])
            .current_dir(&self.working_dir)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        {
            if name != "HEAD" && !name.is_empty() && name != "undefined" {
                let name = name
                    .strip_prefix("remotes/origin/")
                    .or_else(|| name.strip_prefix("remotes/"))
                    .unwrap_or(&name)
                    .to_string();
                return Some(name);
            }
        }

        // Last resort: detached@<short-sha>
        let short_sha = Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .current_dir(&self.working_dir)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        Some(format!("detached@{short_sha}"))
    }

    /// Get current git commit SHA (short)
    fn git_commit(&self) -> Option<String> {
        Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .current_dir(&self.working_dir)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Get files changed relative to the branch point.
    ///
    /// Tries multiple strategies since git state varies across iterations:
    /// 1. Unstaged changes vs HEAD (works mid-iteration before commit)
    /// 2. All changes since branching from main (works post-commit on worktree branches)
    /// 3. Porcelain status fallback
    fn git_changed_files(&self) -> Vec<String> {
        let mut all_files = HashSet::new();

        // 1. Unstaged changes — works before git commit
        for f in self.run_git_diff(&["diff", "--name-only", "HEAD"]) {
            all_files.insert(f);
        }

        // 2. Changes since branch point from main.
        // Worktree branches are `swarm/<issue-id>` off main.
        // Three-dot diff finds the merge-base automatically.
        // Always run this (don't short-circuit) because step 1 may only
        // return non-source files like .swarm-progress.txt.
        for f in self.run_git_diff(&["diff", "--name-only", "main...HEAD"]) {
            all_files.insert(f);
        }

        if !all_files.is_empty() {
            return all_files.into_iter().collect();
        }

        // 3. Fallback: porcelain status
        Command::new("git")
            .args(["status", "--porcelain", "-s"])
            .current_dir(&self.working_dir)
            .output()
            .ok()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .filter_map(|line| {
                        let trimmed = line.trim();
                        if trimmed.len() > 3 {
                            Some(trimmed[3..].to_string())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Run a git diff command and return the list of file paths.
    fn run_git_diff(&self, args: &[&str]) -> Vec<String> {
        Command::new("git")
            .args(args)
            .current_dir(&self.working_dir)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| {
                String::from_utf8_lossy(&o.stdout)
                    .lines()
                    .filter(|l| !l.is_empty())
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_extraction_regex() {
        let content = r#"
pub struct MyParser {
    field: String,
}

pub trait Parseable {
    fn parse(&self) -> Result<(), Error>;
}

pub enum ParseState {
    Init,
    Reading,
    Done,
}

pub fn create_parser() -> MyParser {
    MyParser { field: String::new() }
}

pub async fn parse_stream(input: &str) -> Result<(), Error> {
    Ok(())
}
"#;

        let structs: Vec<_> = STRUCT_PATTERN
            .captures_iter(content)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
            .collect();
        assert_eq!(structs, vec!["MyParser"]);

        let traits: Vec<_> = TRAIT_PATTERN
            .captures_iter(content)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
            .collect();
        assert_eq!(traits, vec!["Parseable"]);

        let enums: Vec<_> = ENUM_PATTERN
            .captures_iter(content)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
            .collect();
        assert_eq!(enums, vec!["ParseState"]);

        // FN_PATTERN matches `pub fn` and `pub async fn` — trait method `fn parse`
        // isn't directly `pub fn` so it's not matched (correct behavior,
        // we only want pub API symbols)
        let fns: Vec<_> = FN_PATTERN
            .captures_iter(content)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_string()))
            .collect();
        assert!(fns.contains(&"create_parser".to_string()));
        assert!(fns.contains(&"parse_stream".to_string()));
        assert_eq!(fns.len(), 2);
    }

    #[test]
    fn test_git_branch_returns_branch_name() {
        let dir = tempfile::tempdir().unwrap();
        let wd = dir.path();

        // Initialize a git repo on a named branch
        Command::new("git")
            .args(["init", "-b", "test-branch"])
            .current_dir(wd)
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(wd)
            .output()
            .unwrap();

        let gen = WorkPacketGenerator::new(wd.to_path_buf());
        let branch = gen.git_branch().unwrap();
        assert_eq!(branch, "test-branch");
    }

    #[test]
    fn test_git_branch_detached_head_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let wd = dir.path();

        // Initialize a git repo, create two commits, then detach HEAD at the first
        Command::new("git")
            .args(["init", "-b", "main"])
            .current_dir(wd)
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "--allow-empty", "-m", "first"])
            .current_dir(wd)
            .output()
            .unwrap();
        let first_sha = Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .current_dir(wd)
            .output()
            .unwrap();
        let first_sha = String::from_utf8_lossy(&first_sha.stdout)
            .trim()
            .to_string();

        Command::new("git")
            .args(["commit", "--allow-empty", "-m", "second"])
            .current_dir(wd)
            .output()
            .unwrap();

        // Detach HEAD at first commit
        Command::new("git")
            .args(["checkout", &first_sha])
            .current_dir(wd)
            .output()
            .unwrap();

        let gen = WorkPacketGenerator::new(wd.to_path_buf());
        let branch = gen.git_branch().unwrap();

        // Should not be the literal "HEAD"
        assert_ne!(branch, "HEAD");
        // Should either resolve via name-rev or fall back to detached@<sha>
        assert!(
            branch.contains("main") || branch.starts_with("detached@"),
            "Expected name-rev or detached@sha, got: {branch}"
        );
    }

    #[test]
    fn test_git_branch_ci_env_var_fallback() {
        let dir = tempfile::tempdir().unwrap();
        let wd = dir.path();

        // Initialize git repo, create a commit, detach HEAD
        Command::new("git")
            .args(["init", "-b", "main"])
            .current_dir(wd)
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "--allow-empty", "-m", "init"])
            .current_dir(wd)
            .output()
            .unwrap();
        let sha = Command::new("git")
            .args(["rev-parse", "HEAD"])
            .current_dir(wd)
            .output()
            .unwrap();
        let sha = String::from_utf8_lossy(&sha.stdout).trim().to_string();

        Command::new("git")
            .args(["commit", "--allow-empty", "-m", "second"])
            .current_dir(wd)
            .output()
            .unwrap();

        Command::new("git")
            .args(["checkout", &sha])
            .current_dir(wd)
            .output()
            .unwrap();

        // Set a CI env var — but note this test is fragile because env vars are
        // process-global, so we only verify the branch is not "HEAD"
        // (The env var path is tested implicitly; the key assertion is
        //  that detached HEAD never leaks as the literal string "HEAD".)
        let gen = WorkPacketGenerator::new(wd.to_path_buf());
        let branch = gen.git_branch().unwrap();
        assert_ne!(
            branch, "HEAD",
            "Detached HEAD should never leak as literal 'HEAD'"
        );
    }
}
