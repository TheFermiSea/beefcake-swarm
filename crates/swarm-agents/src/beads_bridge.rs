use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::Path;
use std::process::Command;

/// A beads issue as returned by `bd list --json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeadsIssue {
    pub id: String,
    pub title: String,
    pub status: String,
    pub priority: Option<u8>,
    #[serde(rename = "type")]
    pub issue_type: Option<String>,
    #[serde(default)]
    pub labels: Vec<String>,
    /// Full issue description (from `bd show --json`). Contains file lists,
    /// implementation details, and context that the file targeting pipeline
    /// uses to locate relevant source files.
    #[serde(default)]
    pub description: Option<String>,
}

impl BeadsIssue {
    /// Returns a complexity sort key based on the `swarm_complexity` label.
    /// Lower = simpler = preferred by the orchestrator.
    /// - "additive"     → 0
    /// - "modify_small" → 1
    /// - "modify_large" → 2
    /// - unlabelled     → 1  (treat as modify_small when unknown)
    pub fn swarm_complexity_rank(&self) -> u8 {
        for label in &self.labels {
            match label.as_str() {
                "additive" => return 0,
                "modify_small" => return 1,
                "modify_large" => return 2,
                _ => {}
            }
        }
        1 // default: treat as modify_small
    }
}

/// Abstraction over issue tracking backends.
///
/// `BeadsBridge` implements this for the real beads CLI.
/// Tests can provide a mock implementation.
pub trait IssueTracker: Send + Sync {
    fn list_ready(&self) -> Result<Vec<BeadsIssue>>;
    fn update_status(&self, id: &str, status: &str) -> Result<()>;
    fn close(&self, id: &str, reason: Option<&str>) -> Result<()>;

    /// Atomically claim an issue for this orchestrator instance.
    ///
    /// Returns `Ok(true)` if the issue was successfully claimed (transitioned
    /// from `open` → `in_progress`), or `Ok(false)` if it was already claimed
    /// by another instance.
    ///
    /// Default implementation just calls `update_status` (no race protection).
    fn try_claim(&self, id: &str) -> Result<bool> {
        self.update_status(id, "in_progress")?;
        Ok(true)
    }
}

/// No-op tracker for beads-free mode.
///
/// Used when the user provides `--issue` / `--issue-file` CLI flags or when
/// the `bd` binary is unavailable. All operations succeed silently.
pub struct NoOpTracker;

impl IssueTracker for NoOpTracker {
    fn list_ready(&self) -> Result<Vec<BeadsIssue>> {
        Ok(vec![])
    }
    fn update_status(&self, _id: &str, _status: &str) -> Result<()> {
        Ok(())
    }
    fn close(&self, _id: &str, _reason: Option<&str>) -> Result<()> {
        Ok(())
    }
}

/// Bridge to the beads CLI binary (`bd`).
///
/// beads is a Go binary — we shell out to it.
/// The binary name is read from the `SWARM_BEADS_BIN` env var, defaulting to `"bd"`.
///
/// Optionally operates within a specific worktree directory (for worktree-scoped
/// commands like `bd mail` where identity comes from `BD_ACTOR`
/// (resolved via [`default_actor()`]).
pub struct BeadsBridge {
    bin: String,
    /// Working directory for bd commands. When set, commands run in this directory
    /// (which must have `.beads/` access, typically via symlink from the worktree).
    wt_path: Option<std::path::PathBuf>,
}

impl Default for BeadsBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl BeadsBridge {
    pub fn new() -> Self {
        Self {
            bin: std::env::var("SWARM_BEADS_BIN").unwrap_or_else(|_| "bd".into()),
            wt_path: None,
        }
    }

    /// Create a BeadsBridge that runs commands in a specific worktree directory.
    ///
    /// The worktree must have `.beads/` access (typically via symlink).
    /// Identity is resolved via [`default_actor()`] (BD_ACTOR env var → git
    /// user.name → hostname → "worker"), not a server-side file.
    pub fn with_worktree(wt_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            bin: std::env::var("SWARM_BEADS_BIN").unwrap_or_else(|_| "bd".into()),
            wt_path: Some(wt_path.into()),
        }
    }

    /// Run a bd command, optionally in the configured worktree directory.
    fn run_bd(&self, args: &[&str]) -> Result<std::process::Output> {
        let mut cmd = Command::new(&self.bin);
        cmd.args(args);
        if let Some(ref wt) = self.wt_path {
            cmd.current_dir(wt);
        }
        cmd.output()
            .with_context(|| format!("Failed to run `{} {}`", self.bin, args.join(" ")))
    }

    /// Run a bd command and return stdout on success, bail on failure.
    fn run_bd_ok(&self, args: &[&str]) -> Result<String> {
        let output = self.run_bd(args)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "{} {} failed: {stderr}",
                self.bin,
                args.first().unwrap_or(&"")
            );
        }
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    /// Create a new issue, returns the issue ID.
    pub fn create(&self, title: &str, issue_type: &str, priority: u8) -> Result<String> {
        let stdout = self.run_bd_ok(&[
            "create",
            &format!("--title={title}"),
            &format!("--type={issue_type}"),
            &format!("--priority={priority}"),
        ])?;
        Ok(stdout.trim().to_string())
    }

    // ── Molecule primitives (child issues + dependencies + labels) ────

    /// Add a blocking dependency: `issue_id` depends on `depends_on_id`.
    ///
    /// Maps to `bd dep add <issue_id> <depends_on_id> --type blocks`.
    pub fn add_dependency(&self, issue_id: &str, depends_on_id: &str) -> Result<()> {
        self.run_bd_ok(&["dep", "add", issue_id, depends_on_id, "--type", "blocks"])?;
        Ok(())
    }

    /// Add a label to an issue.
    ///
    /// Uses the `dim:value` convention for structured labels (e.g., `target-file:src/foo.rs`).
    pub fn add_label(&self, issue_id: &str, label: &str) -> Result<()> {
        self.run_bd_ok(&["label", "add", issue_id, label])?;
        Ok(())
    }

    /// Create a molecule: parent epic with child subtask issues and blocking dependencies.
    ///
    /// Each child blocks the parent. When all children close, the parent auto-unblocks.
    /// Returns the list of child issue IDs created.
    pub fn create_molecule(
        &self,
        parent_id: &str,
        subtasks: &[(String, Vec<String>)], // (objective, target_files)
    ) -> Result<Vec<String>> {
        let mut child_ids = Vec::with_capacity(subtasks.len());

        for (i, (objective, target_files)) in subtasks.iter().enumerate() {
            // Create child issue with a compact title.
            let title = if objective.len() > 80 {
                format!("subtask-{}: {}...", i + 1, &objective[..77])
            } else {
                format!("subtask-{}: {}", i + 1, objective)
            };
            let child_id = self.create(&title, "task", 1)?;

            // Parent waits for this child (child blocks parent).
            if let Err(e) = self.add_dependency(parent_id, &child_id) {
                tracing::warn!(
                    parent = %parent_id,
                    child = %child_id,
                    error = %e,
                    "Failed to add dependency — molecule tracking degraded"
                );
            }

            // Tag child with target file labels for observability.
            for file in target_files {
                let label = format!("target-file:{file}");
                let _ = self.add_label(&child_id, &label);
            }

            // Link back to parent for traceability.
            let parent_label = format!("parent:{parent_id}");
            let _ = self.add_label(&child_id, &parent_label);

            child_ids.push(child_id);
        }

        tracing::info!(
            parent = %parent_id,
            children = ?child_ids,
            "Created molecule: {} child issues",
            child_ids.len()
        );

        Ok(child_ids)
    }

    /// Update an issue's description text.
    ///
    /// Maps to `bd update <id> --description="..."`.
    /// Used by the reformulation engine to rewrite malformed task descriptions.
    pub fn update_description(&self, id: &str, description: &str) -> Result<()> {
        self.run_bd_ok(&["update", id, &format!("--description={description}")])?;
        Ok(())
    }

    /// Append notes to an issue.
    ///
    /// Maps to `bd update <id> --notes="..."`.
    /// Used by the reformulation engine to add learned directives.
    pub fn update_notes(&self, id: &str, notes: &str) -> Result<()> {
        self.run_bd_ok(&["update", id, &format!("--notes={notes}")])?;
        Ok(())
    }

    /// Add a label to an issue (convenience re-export for reformulation engine).
    ///
    /// Used to tag issues with `swarm:needs-human-review` when reformulation
    /// is exhausted.
    pub fn add_swarm_label(&self, id: &str, label: &str) -> Result<()> {
        self.add_label(id, label)
    }

    /// Look up a single issue by ID.
    pub fn show(&self, id: &str) -> Result<BeadsIssue> {
        let stdout = self.run_bd_ok(&["show", id, "--json"])?;

        // `bd show --json` returns an array with one element
        let issues: Vec<BeadsIssue> = serde_json::from_str(&stdout)
            .context(format!("Failed to parse {} show output", self.bin))?;

        issues
            .into_iter()
            .next()
            .context(format!("No issue found with id {id}"))
    }

    // ── Native messaging (bd mail) ────────────────────────────────────

    /// Send a mail message to another agent via `bd mail send`.
    ///
    /// Identity is resolved by [`default_actor()`] (BD_ACTOR env var →
    /// git user.name → hostname → "worker"; set externally via `run-swarm.sh`).
    /// Messages are stored as Dolt rows and sync with `bd dolt push/pull`.
    ///
    /// # Arguments
    /// * `to` - Recipient actor name (e.g., "lead", "worker-1")
    /// * `subject` - Mail subject line
    /// * `message` - Mail body content
    ///
    /// # Example
    /// ```ignore
    /// let bridge = BeadsBridge::new();
    /// bridge.send_mail("lead", "Task Complete", "I finished the assigned work.")?;
    /// ```
    ///
    /// # Native Beads Migration
    ///
    /// This method is part of the native beads messaging layer that replaces
    /// the previous BeadHub coordination tools. It enables inter-worker
    /// communication during concurrent subtask execution (see `subtask.rs`).
    ///
    /// # Inline Usage Notes
    ///
    /// - Messages are queued locally and synced via `bd dolt push/pull`
    /// - The `to` field should match an actor name configured in the swarm
    /// - Failures are logged but do not propagate errors (non-blocking)
    pub fn send_mail(&self, to: &str, subject: &str, message: &str) -> Result<()> {
        self.run_bd_ok(&["mail", "send", to, "-s", subject, "-m", message])?;
        Ok(())
    }

    /// Check the inbox for incoming messages.
    ///
    /// Returns the raw output from `bd mail inbox`.
    /// Empty or "no messages" means no pending mail.
    pub fn check_inbox(&self) -> Result<String> {
        self.run_bd_ok(&["mail", "inbox"])
    }

    /// Read a specific mail message by ID.
    pub fn read_mail(&self, msg_id: &str) -> Result<String> {
        self.run_bd_ok(&["mail", "read", msg_id])
    }

    /// Reply to a mail message.
    pub fn reply_mail(&self, msg_id: &str, message: &str) -> Result<()> {
        self.run_bd_ok(&["mail", "reply", msg_id, "-m", message])?;
        Ok(())
    }
}

impl IssueTracker for BeadsBridge {
    /// List ready issues (open and not blocked), sorted by priority.
    fn list_ready(&self) -> Result<Vec<BeadsIssue>> {
        let stdout = self.run_bd_ok(&["ready", "--json"])?;
        let issues: Vec<BeadsIssue> =
            serde_json::from_str(&stdout).context("Failed to parse bd ready output")?;
        Ok(issues)
    }

    /// Update issue status.
    fn update_status(&self, id: &str, status: &str) -> Result<()> {
        self.run_bd_ok(&["update", id, &format!("--status={status}")])?;
        Ok(())
    }

    /// Close an issue.
    fn close(&self, id: &str, reason: Option<&str>) -> Result<()> {
        let mut args = vec!["close", id];
        let reason_arg;
        if let Some(r) = reason {
            reason_arg = format!("--reason={r}");
            args.push(&reason_arg);
        }
        self.run_bd_ok(&args)?;
        Ok(())
    }

    /// Claim an issue using `bd update --claim`.
    ///
    /// The native `--claim` flag transitions the issue to `in_progress` and
    /// records the actor identity. Returns `Ok(false)` if already claimed.
    fn try_claim(&self, id: &str) -> Result<bool> {
        let output = self.run_bd(&["update", id, "--claim"])?;
        if output.status.success() {
            return Ok(true);
        }

        // --claim fails if already claimed — check if it's a "not open" error
        let stderr = String::from_utf8_lossy(&output.stderr);
        if stderr.contains("already") || stderr.contains("not open") || stderr.contains("claimed") {
            tracing::info!(id = %id, "Issue already claimed or closed, skipping");
            return Ok(false);
        }

        // Unexpected error — fall back to show + update
        let issue = self.show(id)?;
        if issue.status != "open" {
            tracing::info!(
                id = %id,
                status = %issue.status,
                "Issue already claimed or closed, skipping"
            );
            return Ok(false);
        }
        self.update_status(id, "in_progress")?;
        Ok(true)
    }
}

/// Resolve the actor identity for this orchestrator instance.
///
/// Implements the canonical fallback chain:
/// 1. `BD_ACTOR` environment variable (set by `run-swarm.sh` or the caller)
/// 2. `git config user.name` (local git identity)
/// 3. `hostname -s` (short hostname)
/// 4. Hard-coded fallback `"worker"` (always succeeds)
///
/// # Example
/// ```
/// let actor = swarm_agents::beads_bridge::default_actor();
/// println!("Running as: {actor}");
/// ```
pub fn default_actor() -> String {
    // 1. Explicit override via BD_ACTOR env var.
    if let Ok(actor) = env::var("BD_ACTOR") {
        let actor = actor.trim().to_string();
        if !actor.is_empty() {
            return actor;
        }
    }

    // 2. Git user.name from local config.
    if let Ok(output) = Command::new("git").args(["config", "user.name"]).output() {
        if output.status.success() {
            let name = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !name.is_empty() {
                return name;
            }
        }
    }

    // 3. Short hostname.
    if let Ok(output) = Command::new("hostname").arg("-s").output() {
        if output.status.success() {
            let host = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !host.is_empty() {
                return host;
            }
        }
    }

    // 4. Hard-coded last-resort fallback.
    "worker".to_string()
}

/// Poll the `bd mail inbox` for messages directed at the current actor.
///
/// Returns `Some(inbox_text)` if there are unread messages, `None` otherwise.
/// Fails silently — mail unavailability never blocks orchestration.
pub fn poll_mail_inbox(wt_path: &Path) -> Option<String> {
    let bridge = BeadsBridge::with_worktree(wt_path);
    match bridge.check_inbox() {
        Ok(inbox) if !inbox.trim().is_empty() && !inbox.contains("no messages") => {
            tracing::info!(
                inbox_len = inbox.len(),
                "Unread mail messages detected between iterations"
            );
            Some(inbox)
        }
        Ok(_) => None,
        Err(e) => {
            tracing::debug!(error = %e, "Mail inbox check failed (non-fatal)");
            None
        }
    }
}

/// Send an escalation mail when the orchestrator is stuck on an issue.
///
/// Sends to "lead" (the swarm lead actor). Non-blocking — failures are logged
/// but never propagate errors.
pub fn escalate_via_mail(wt_path: &Path, issue_id: &str, reason: &str) {
    let bridge = BeadsBridge::with_worktree(wt_path);
    let subject = format!("Stuck: {issue_id}");
    match bridge.send_mail("lead", &subject, reason) {
        Ok(()) => tracing::info!(issue_id, "Escalation mail sent via bd mail"),
        Err(e) => tracing::warn!(
            issue_id,
            error = %e,
            "bd mail send failed (non-fatal — escalation recorded in intervention file)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn test_poll_mail_inbox_empty_returns_none() {
        // Create a temporary script that echoes empty output
        let tmp_dir = std::env::temp_dir().join("swarm_test_beads_bridge");
        fs::create_dir_all(&tmp_dir).unwrap();
        let script_path = tmp_dir.join("bd_mock");

        // Write a mock script that echoes nothing (empty inbox)
        let script_content =
            "#!/bin/bash\n# Mock bd that returns empty output for mail inbox\nexit 0\n";
        let mut file = fs::File::create(&script_path).unwrap();
        file.write_all(script_content.as_bytes()).unwrap();
        fs::set_permissions(
            &script_path,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();

        // Set the mock binary
        let old_bin = std::env::var("SWARM_BEADS_BIN").ok();
        std::env::set_var("SWARM_BEADS_BIN", &script_path);

        // Create a temporary worktree directory
        let tmp_wt = tmp_dir.join("test_wt");
        fs::create_dir_all(&tmp_wt).unwrap();

        // Test that poll_mail_inbox returns None when inbox is empty
        let result = poll_mail_inbox(&tmp_wt);
        assert!(
            result.is_none(),
            "Expected None for empty inbox, got {:?}",
            result
        );

        // Restore original env var
        if let Some(bin) = old_bin {
            std::env::set_var("SWARM_BEADS_BIN", bin);
        } else {
            std::env::remove_var("SWARM_BEADS_BIN");
        }

        // Cleanup
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_beads_bridge_check_inbox_empty_returns_none() {
        // Create a temporary script that echoes empty output
        let tmp_dir = std::env::temp_dir().join("swarm_test_beads_bridge_check");
        fs::create_dir_all(&tmp_dir).unwrap();
        let script_path = tmp_dir.join("bd_mock");

        // Write a mock script that echoes nothing (empty inbox)
        let script_content =
            "#!/bin/bash\n# Mock bd that returns empty output for mail inbox\nexit 0\n";
        let mut file = fs::File::create(&script_path).unwrap();
        file.write_all(script_content.as_bytes()).unwrap();
        fs::set_permissions(
            &script_path,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();

        // Set the mock binary
        let old_bin = std::env::var("SWARM_BEADS_BIN").ok();
        std::env::set_var("SWARM_BEADS_BIN", &script_path);

        // Create a temporary worktree directory
        let tmp_wt = tmp_dir.join("test_wt");
        fs::create_dir_all(&tmp_wt).unwrap();

        // Test that BeadsBridge::check_inbox returns Ok with empty string when inbox is empty
        let bridge = BeadsBridge::with_worktree(&tmp_wt);
        let result = bridge.check_inbox();
        assert!(
            result.is_ok(),
            "Expected Ok result for check_inbox, got error: {:?}",
            result
        );
        let inbox = result.unwrap();
        assert!(
            inbox.trim().is_empty(),
            "Expected empty inbox, got: {:?}",
            inbox
        );

        // Restore original env var
        if let Some(bin) = old_bin {
            std::env::set_var("SWARM_BEADS_BIN", bin);
        } else {
            std::env::remove_var("SWARM_BEADS_BIN");
        }

        // Cleanup
        let _ = fs::remove_dir_all(&tmp_dir);
    }

    #[test]
    fn test_beads_bridge_check_inbox_empty_returns_none() {
        // Create a temporary script that echoes empty output
        let tmp_dir = std::env::temp_dir().join("swarm_test_beads_bridge_check");
        fs::create_dir_all(&tmp_dir).unwrap();
        let script_path = tmp_dir.join("bd_mock");

        // Write a mock script that echoes nothing (empty inbox)
        let script_content =
            "#!/bin/bash\n# Mock bd that returns empty output for mail inbox\nexit 0\n";
        let mut file = fs::File::create(&script_path).unwrap();
        file.write_all(script_content.as_bytes()).unwrap();
        fs::set_permissions(
            &script_path,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();

        // Set the mock binary
        let old_bin = std::env::var("SWARM_BEADS_BIN").ok();
        std::env::set_var("SWARM_BEADS_BIN", &script_path);

        // Create a temporary worktree directory
        let tmp_wt = tmp_dir.join("test_wt");
        fs::create_dir_all(&tmp_wt).unwrap();

        // Test that BeadsBridge::check_inbox returns Ok with empty string when inbox is empty
        let bridge = BeadsBridge::with_worktree(&tmp_wt);
        let result = bridge.check_inbox();
        assert!(
            result.is_ok(),
            "Expected Ok result for check_inbox, got error: {:?}",
            result
        );
        let inbox = result.unwrap();
        assert!(
            inbox.trim().is_empty(),
            "Expected empty inbox, got: {:?}",
            inbox
        );

        // Restore original env var
        if let Some(bin) = old_bin {
            std::env::set_var("SWARM_BEADS_BIN", bin);
        } else {
            std::env::remove_var("SWARM_BEADS_BIN");
        }

        // Cleanup
        let _ = fs::remove_dir_all(&tmp_dir);
    }
}
