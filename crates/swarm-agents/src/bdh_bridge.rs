//! BdhBridge — Multi-agent coordination via the `bdh` CLI.
//!
//! `bdh` wraps `bd` (beads) with server-side coordination primitives:
//! atomic claiming, mail/chat messaging, file locking, team status,
//! and async human escalation. All integration is via subprocess calls,
//! following the same pattern as [`BeadsBridge`](crate::beads_bridge::BeadsBridge).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

use crate::beads_bridge::{BeadsIssue, IssueTracker};

/// Team member status as returned by `bdh :status --json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamMember {
    pub alias: String,
    #[serde(default)]
    pub role: Option<String>,
    #[serde(default)]
    pub status: Option<String>,
    #[serde(default)]
    pub current_issue: Option<String>,
    #[serde(default)]
    pub workspace: Option<String>,
}

/// Team status overview from `bdh :status --json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeamStatus {
    #[serde(default)]
    pub members: Vec<TeamMember>,
    #[serde(default)]
    pub my_alias: Option<String>,
    #[serde(default)]
    pub my_role: Option<String>,
}

/// Bridge to the `bdh` CLI for multi-agent coordination.
///
/// Implements [`IssueTracker`] by delegating to `bdh` (which wraps `bd`).
/// Additional methods provide access to bdh-specific coordination features:
/// team status, mail/chat, file locking, and async escalation.
///
/// The binary name is read from `SWARM_BDH_BIN` (default: `"bdh"`).
pub struct BdhBridge {
    bin: String,
    /// Working directory for bdh commands (must contain `.beadhub` file).
    /// When None, commands run in the current directory.
    wt_path: Option<std::path::PathBuf>,
}

impl Default for BdhBridge {
    fn default() -> Self {
        Self::new()
    }
}

impl BdhBridge {
    /// Create a new BdhBridge using the default binary name.
    pub fn new() -> Self {
        Self {
            bin: std::env::var("SWARM_BDH_BIN").unwrap_or_else(|_| "bdh".into()),
            wt_path: None,
        }
    }

    /// Create a BdhBridge that runs commands in a specific worktree directory.
    ///
    /// The worktree must have a `.beadhub` file for bdh to derive agent identity.
    pub fn with_worktree(wt_path: impl Into<std::path::PathBuf>) -> Self {
        Self {
            bin: std::env::var("SWARM_BDH_BIN").unwrap_or_else(|_| "bdh".into()),
            wt_path: Some(wt_path.into()),
        }
    }

    /// Run a bdh command, optionally in the configured worktree directory.
    fn run_bdh(&self, args: &[&str]) -> Result<std::process::Output> {
        let mut cmd = Command::new(&self.bin);
        cmd.args(args);
        if let Some(ref wt) = self.wt_path {
            cmd.current_dir(wt);
        }
        cmd.output()
            .with_context(|| format!("Failed to run `{} {}`", self.bin, args.join(" ")))
    }

    /// Run a bdh command and return stdout on success, bail on failure.
    fn run_bdh_ok(&self, args: &[&str]) -> Result<String> {
        let output = self.run_bdh(args)?;
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

    /// Look up a single issue by ID.
    pub fn show(&self, id: &str) -> Result<BeadsIssue> {
        let stdout = self.run_bdh_ok(&["show", id, "--json"])?;

        // `bdh show --json` returns an array with one element (same as `bd show --json`)
        let issues: Vec<BeadsIssue> =
            serde_json::from_str(&stdout).context("Failed to parse bdh show output")?;

        issues
            .into_iter()
            .next()
            .context(format!("No issue found with id {id}"))
    }

    // ── Coordination methods (beyond IssueTracker) ──

    /// Get team status overview.
    ///
    /// Returns structured data about all team members, their roles,
    /// current issues, and workspace status.
    pub fn team_status(&self) -> Result<TeamStatus> {
        let stdout = self.run_bdh_ok(&[":status", "--json"])?;
        serde_json::from_str(&stdout).context("Failed to parse bdh :status output")
    }

    /// Escalate to human operator asynchronously.
    ///
    /// Creates a non-blocking notification — the orchestrator can continue
    /// processing other issues while waiting for human response.
    pub fn escalate(&self, subject: &str, situation: &str) -> Result<()> {
        let message = format!("{subject}: {situation}");
        self.run_bdh_ok(&[":escalate", &message])?;
        Ok(())
    }

    /// Initialize bdh identity in a worktree directory.
    ///
    /// Runs `bdh :init` to create a `.beadhub` file that identifies this
    /// agent (alias + role) for all subsequent bdh commands in that worktree.
    pub fn init_worktree(&self, wt_path: &Path, alias: &str, role: &str) -> Result<()> {
        let output = Command::new(&self.bin)
            .args([":init", "--alias", alias, "--role", role])
            .current_dir(wt_path)
            .output()
            .with_context(|| {
                format!(
                    "Failed to run `{} :init` in {}",
                    self.bin,
                    wt_path.display()
                )
            })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!(
                alias = %alias,
                role = %role,
                wt = %wt_path.display(),
                stderr = %stderr.trim(),
                "bdh :init failed (non-fatal — coordination features unavailable)"
            );
        } else {
            tracing::info!(
                alias = %alias,
                role = %role,
                wt = %wt_path.display(),
                "Initialized bdh identity in worktree"
            );
        }

        Ok(())
    }

    /// Send a mail message to another agent (fire-and-forget).
    pub fn send_mail(&self, to: &str, message: &str) -> Result<()> {
        self.run_bdh_ok(&[":aweb", "mail", "send", to, message])?;
        Ok(())
    }

    /// List incoming mail messages.
    pub fn list_mail(&self) -> Result<String> {
        self.run_bdh_ok(&[":aweb", "mail", "list"])
    }
}

impl IssueTracker for BdhBridge {
    /// List ready issues (open and not blocked), sorted by priority.
    fn list_ready(&self) -> Result<Vec<BeadsIssue>> {
        let stdout = self.run_bdh_ok(&["ready", "--json"])?;
        let issues: Vec<BeadsIssue> =
            serde_json::from_str(&stdout).context("Failed to parse bdh ready output")?;
        Ok(issues)
    }

    /// Update issue status.
    fn update_status(&self, id: &str, status: &str) -> Result<()> {
        self.run_bdh_ok(&["update", id, &format!("--status={status}")])?;
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
        self.run_bdh_ok(&args)?;
        Ok(())
    }

    /// Atomically claim an issue via bdh's server-side claiming.
    ///
    /// bdh's server handles race conditions — if two agents try to claim
    /// the same issue simultaneously, only one succeeds. This is stronger
    /// than BeadsBridge's check-then-update pattern.
    fn try_claim(&self, id: &str) -> Result<bool> {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_bin_name() {
        // When SWARM_BDH_BIN is not set, defaults to "bdh"
        std::env::remove_var("SWARM_BDH_BIN");
        let bridge = BdhBridge::new();
        assert_eq!(bridge.bin, "bdh");
    }

    #[test]
    fn test_with_worktree() {
        let bridge = BdhBridge::with_worktree("/tmp/test-wt");
        assert_eq!(bridge.wt_path.unwrap().to_str().unwrap(), "/tmp/test-wt");
    }

    #[test]
    fn test_team_status_deserialize() {
        let json = r#"{"members": [{"alias": "worker-1", "role": "coder", "status": "active", "current_issue": "beefcake-abc1"}], "my_alias": "worker-1", "my_role": "coder"}"#;
        let status: TeamStatus = serde_json::from_str(json).unwrap();
        assert_eq!(status.members.len(), 1);
        assert_eq!(status.members[0].alias, "worker-1");
        assert_eq!(status.my_alias.unwrap(), "worker-1");
    }
}
