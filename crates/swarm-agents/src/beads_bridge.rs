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
    #[serde(default)]
    pub labels: Vec<String>,
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
pub trait IssueTracker {
    fn list_ready(&self) -> Result<Vec<BeadsIssue>>;
    fn update_status(&self, id: &str, status: &str) -> Result<()>;
    fn close(&self, id: &str, reason: Option<&str>) -> Result<()>;
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
}
