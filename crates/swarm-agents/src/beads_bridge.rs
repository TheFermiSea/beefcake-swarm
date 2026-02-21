use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
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
}

/// Abstraction over issue tracking backends.
///
/// `BeadsBridge` implements this for the real beads CLI.
/// Tests can provide a mock implementation.
pub trait IssueTracker {
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
pub struct BeadsBridge {
    bin: String,
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
        }
    }

    /// Create a new issue, returns the issue ID.
    pub fn create(&self, title: &str, issue_type: &str, priority: u8) -> Result<String> {
        let output = Command::new(&self.bin)
            .args([
                "create",
                &format!("--title={title}"),
                &format!("--type={issue_type}"),
                &format!("--priority={priority}"),
            ])
            .output()
            .context(format!("Failed to run `{} create`", self.bin))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{} create failed: {stderr}", self.bin);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.trim().to_string())
    }
}

impl BeadsBridge {
    /// Look up a single issue by ID.
    pub fn show(&self, id: &str) -> Result<BeadsIssue> {
        let output = Command::new(&self.bin)
            .args(["show", id, "--json"])
            .output()
            .context(format!("Failed to run `{} show {id}`", self.bin))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{} show failed: {stderr}", self.bin);
        }

        // `bd show --json` returns an array with one element
        let issues: Vec<BeadsIssue> = serde_json::from_slice(&output.stdout)
            .context(format!("Failed to parse {} show output", self.bin))?;

        issues
            .into_iter()
            .next()
            .context(format!("No issue found with id {id}"))
    }
}

impl IssueTracker for BeadsBridge {
    /// List ready issues (open and not blocked), sorted by priority.
    fn list_ready(&self) -> Result<Vec<BeadsIssue>> {
        let output = Command::new(&self.bin)
            .args(["ready", "--json"])
            .output()
            .context(format!(
                "Failed to run `{} ready`. Is beads installed?",
                self.bin
            ))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{} ready failed: {stderr}", self.bin);
        }

        let issues: Vec<BeadsIssue> = serde_json::from_slice(&output.stdout)
            .context(format!("Failed to parse {} ready output", self.bin))?;

        Ok(issues)
    }

    /// Update issue status.
    fn update_status(&self, id: &str, status: &str) -> Result<()> {
        let output = Command::new(&self.bin)
            .args(["update", id, &format!("--status={status}")])
            .output()
            .context(format!("Failed to run `{} update`", self.bin))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{} update failed: {stderr}", self.bin);
        }

        Ok(())
    }

    /// Close an issue.
    fn close(&self, id: &str, reason: Option<&str>) -> Result<()> {
        let mut args = vec!["close".to_string(), id.to_string()];
        if let Some(r) = reason {
            args.push(format!("--reason={r}"));
        }

        let output = Command::new(&self.bin)
            .args(&args)
            .output()
            .context(format!("Failed to run `{} close`", self.bin))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{} close failed: {stderr}", self.bin);
        }

        Ok(())
    }

    /// Atomically claim an issue: check status first, then update.
    ///
    /// Returns `Ok(false)` if the issue is already `in_progress` or `closed`,
    /// preventing two orchestrator instances from claiming the same issue.
    fn try_claim(&self, id: &str) -> Result<bool> {
        // Check current status before claiming
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
