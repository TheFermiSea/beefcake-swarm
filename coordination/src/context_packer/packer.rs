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
use std::path::{Path, PathBuf};

/// Token budgets per tier (4 chars ≈ 1 token, matching `estimated_tokens()`)
fn max_context_tokens(tier: SwarmTier) -> usize {
    match tier {
        SwarmTier::Implementer => 8_000,
        SwarmTier::Integrator | SwarmTier::Adversary => 24_000,
        SwarmTier::Cloud => 32_000,
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

    /// Retry pack: delegates to WorkPacketGenerator with error context.
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

        self.trim_to_budget(&mut packet);
        packet
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

        let packer = ContextPacker::new(dir.path(), SwarmTier::Implementer);
        let packet = packer.pack_initial("beads-test", "Test objective");

        assert_eq!(packet.bead_id, "beads-test");
        assert_eq!(packet.objective, "Test objective");
        // Should have found our lib.rs in the file contexts
        assert!(!packet.file_contexts.is_empty());
        assert!(packet.estimated_tokens() <= 8_000);
    }

    #[test]
    fn test_token_budgets() {
        assert_eq!(max_context_tokens(SwarmTier::Implementer), 8_000);
        assert_eq!(max_context_tokens(SwarmTier::Integrator), 24_000);
        assert_eq!(max_context_tokens(SwarmTier::Cloud), 32_000);
    }

    #[test]
    fn test_trim_to_budget() {
        let dir = tempfile::tempdir().unwrap();
        let packer = ContextPacker::new(dir.path(), SwarmTier::Implementer);

        let state = EscalationState::new("test");
        let mut packet =
            packer
                .generator
                .generate("test", "test obj", SwarmTier::Implementer, &state, None);

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
}
