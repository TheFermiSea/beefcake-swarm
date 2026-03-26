//! Structured session handoff for context resets.
//!
//! When the context window fills (or context anxiety is detected), a clean-slate
//! reset with a *rich structured handoff artifact* beats in-place compaction for
//! preventing context anxiety and accumulated confusion — a core pattern from
//! Anthropic's harness design article.
//!
//! The handoff is written to two files in the worktree:
//! - `.swarm/session-handoff.md` — human-readable CHANGELOG-style summary
//! - `.swarm/session-handoff.json` — machine-readable `SessionHandoff` struct
//!
//! On context reset, the new agent's **first action** must be to read
//! `session-handoff.md` before doing any other work.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{info, warn};

use crate::work_protocol::FailedApproach;

/// Structured handoff document generated before a context reset.
///
/// Contains everything a fresh agent needs to continue the work without
/// re-reading the entire conversation history. Analogous to a shift handoff
/// in medicine: concise, structured, actionable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionHandoff {
    /// Beads issue ID being worked on.
    pub issue_id: String,
    /// When the handoff was generated.
    pub timestamp: DateTime<Utc>,
    /// Current implementation state — what has been built so far.
    pub current_state: String,
    /// Last verifier gate results summary.
    pub verifier_status: String,
    /// The single most important next action for the incoming agent.
    ///
    /// Should be specific and immediately actionable, e.g.:
    /// "Fix the lifetime error in `src/parser.rs:42` — the issue is that
    ///  `input: &str` needs to outlive the `Parser` struct."
    pub next_step: String,
    /// Approaches that were tried and failed — DO NOT repeat these.
    pub dead_ends: Vec<FailedApproach>,
    /// Unresolved design questions the incoming agent should address.
    pub open_questions: Vec<String>,
    /// Files that are most relevant to the current work.
    pub key_files: Vec<String>,
    /// Which context reset number this is (1-indexed).
    pub reset_number: u32,
}

impl SessionHandoff {
    pub fn new(issue_id: impl Into<String>) -> Self {
        Self {
            issue_id: issue_id.into(),
            timestamp: Utc::now(),
            current_state: String::new(),
            verifier_status: String::new(),
            next_step: String::new(),
            dead_ends: Vec::new(),
            open_questions: Vec::new(),
            key_files: Vec::new(),
            reset_number: 1,
        }
    }

    pub fn with_state(mut self, state: impl Into<String>) -> Self {
        self.current_state = state.into();
        self
    }

    pub fn with_verifier_status(mut self, status: impl Into<String>) -> Self {
        self.verifier_status = status.into();
        self
    }

    pub fn with_next_step(mut self, step: impl Into<String>) -> Self {
        self.next_step = step.into();
        self
    }

    pub fn with_dead_ends(mut self, dead_ends: Vec<FailedApproach>) -> Self {
        self.dead_ends = dead_ends;
        self
    }

    pub fn with_open_questions(mut self, questions: Vec<String>) -> Self {
        self.open_questions = questions;
        self
    }

    pub fn with_key_files(mut self, files: Vec<String>) -> Self {
        self.key_files = files;
        self
    }

    pub fn with_reset_number(mut self, n: u32) -> Self {
        self.reset_number = n;
        self
    }

    /// Render the handoff as a human-readable Markdown document.
    ///
    /// This is what the incoming agent reads first. It should be concise
    /// enough to fit in ~500 tokens while containing everything needed.
    pub fn to_markdown(&self) -> String {
        let mut md = String::with_capacity(1024);

        md.push_str(&format!(
            "# Session Handoff — {} (Reset {})\n\n",
            self.issue_id, self.reset_number
        ));
        md.push_str(&format!(
            "_Generated: {}_\n\n",
            self.timestamp.format("%Y-%m-%d %H:%M UTC")
        ));

        md.push_str("## Current State\n");
        md.push_str(&self.current_state);
        md.push_str("\n\n");

        md.push_str("## Verifier Status\n");
        md.push_str(&self.verifier_status);
        md.push_str("\n\n");

        md.push_str("## Your Next Step\n");
        md.push_str("> **Start here.** Do this before reading any other files.\n\n");
        md.push_str(&self.next_step);
        md.push_str("\n\n");

        if !self.dead_ends.is_empty() {
            md.push_str("## Dead Ends — Do NOT Repeat These\n");
            for de in &self.dead_ends {
                md.push_str(&format!(
                    "- **[iter {}]** {} — *Why it failed:* {}\n",
                    de.iteration, de.summary, de.error_output
                ));
            }
            md.push('\n');
        }

        if !self.key_files.is_empty() {
            md.push_str("## Key Files\n");
            for f in &self.key_files {
                md.push_str(&format!("- `{f}`\n"));
            }
            md.push('\n');
        }

        if !self.open_questions.is_empty() {
            md.push_str("## Open Questions\n");
            for q in &self.open_questions {
                md.push_str(&format!("- {q}\n"));
            }
            md.push('\n');
        }

        md
    }
}

/// Write a `SessionHandoff` to the worktree's `.swarm/` directory.
///
/// Writes both the human-readable Markdown and machine-readable JSON forms.
/// Creates `.swarm/` if it does not exist.
pub fn write_handoff(handoff: &SessionHandoff, wt_path: &Path) {
    let swarm_dir = wt_path.join(".swarm");
    if let Err(e) = std::fs::create_dir_all(&swarm_dir) {
        warn!("Failed to create .swarm dir for handoff: {e}");
        return;
    }

    // Write JSON (machine-readable)
    let json_path = swarm_dir.join("session-handoff.json");
    match serde_json::to_string_pretty(handoff) {
        Ok(json) => match std::fs::write(&json_path, json) {
            Ok(()) => info!(path = %json_path.display(), "Wrote session handoff JSON"),
            Err(e) => warn!("Failed to write handoff JSON: {e}"),
        },
        Err(e) => warn!("Failed to serialize handoff: {e}"),
    }

    // Write Markdown (human-readable / agent-readable)
    let md_path = swarm_dir.join("session-handoff.md");
    match std::fs::write(&md_path, handoff.to_markdown()) {
        Ok(()) => info!(path = %md_path.display(), "Wrote session handoff Markdown"),
        Err(e) => warn!("Failed to write handoff Markdown: {e}"),
    }
}

/// Read a previously written handoff, if one exists.
pub fn read_handoff(wt_path: &Path) -> Option<SessionHandoff> {
    let json_path = wt_path.join(".swarm").join("session-handoff.json");
    let content = std::fs::read_to_string(&json_path).ok()?;
    match serde_json::from_str::<SessionHandoff>(&content) {
        Ok(h) => Some(h),
        Err(e) => {
            warn!("Failed to deserialize session handoff: {e}");
            None
        }
    }
}

/// Build a context-reset prompt prefix that injects the handoff document.
///
/// This is prepended to the first user message after a context reset,
/// instructing the fresh agent to orient itself using the handoff.
pub fn build_reset_context_prefix(handoff: &SessionHandoff) -> String {
    format!(
        "[CONTEXT RESET — continuing from session handoff #{}]\n\
         Read the following handoff document carefully before doing anything else.\n\
         Do NOT ask about past context — everything you need is below.\n\n\
         {}\n\
         ---\n",
        handoff.reset_number,
        handoff.to_markdown()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_handoff_markdown_contains_next_step() {
        let handoff = SessionHandoff::new("issue-42")
            .with_state("Parser module skeleton created")
            .with_verifier_status("cargo check: 2 errors (lifetime mismatch)")
            .with_next_step("Fix lifetime in parser.rs:42 — add `'a` to `Parser<'a>` struct");

        let md = handoff.to_markdown();
        assert!(md.contains("Fix lifetime in parser.rs:42"));
        assert!(md.contains("## Your Next Step"));
        assert!(md.contains("issue-42"));
    }

    #[test]
    fn test_handoff_dead_ends_rendered() {
        let dead_end = FailedApproach::new(2, "Tried wrapping in Arc", "Caused Send bound failure");
        let handoff = SessionHandoff::new("issue-99").with_dead_ends(vec![dead_end]);
        let md = handoff.to_markdown();
        assert!(md.contains("Dead Ends"));
        assert!(md.contains("Tried wrapping in Arc"));
    }

    #[test]
    fn test_write_and_read_handoff() {
        let dir = tempfile::tempdir().unwrap();
        let handoff = SessionHandoff::new("iss-1")
            .with_state("Done with module setup")
            .with_next_step("Fix clippy warning")
            .with_reset_number(2);

        write_handoff(&handoff, dir.path());

        let loaded = read_handoff(dir.path()).expect("handoff should be readable");
        assert_eq!(loaded.issue_id, "iss-1");
        assert_eq!(loaded.reset_number, 2);
        assert_eq!(loaded.next_step, "Fix clippy warning");
    }

    #[test]
    fn test_build_reset_context_prefix() {
        let handoff = SessionHandoff::new("iss-7").with_reset_number(3);
        let prefix = build_reset_context_prefix(&handoff);
        assert!(prefix.contains("CONTEXT RESET"));
        assert!(prefix.contains("#3"));
    }
}
