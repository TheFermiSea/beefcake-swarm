use anyhow::{Context, Result};
use serde::Deserialize;
use std::process::Command;

/// A beads issue as returned by `br list --json`.
#[derive(Debug, Clone, Deserialize)]
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
/// `BeadsBridge` implements this for the real `br` CLI.
/// Tests can provide a mock implementation.
pub trait IssueTracker {
    fn list_open(&self) -> Result<Vec<BeadsIssue>>;
    fn update_status(&self, id: &str, status: &str) -> Result<()>;
    fn close(&self, id: &str, reason: Option<&str>) -> Result<()>;
}

/// Bridge to the `br` (beads_rust) CLI binary.
///
/// beads_rust is a binary-only tool — no lib.rs — so we shell out.
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
            bin: "br".to_string(),
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
            .context("Failed to run `br create`")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("br create failed: {stderr}");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(stdout.trim().to_string())
    }
}

impl IssueTracker for BeadsBridge {
    /// List open issues, sorted by priority.
    fn list_open(&self) -> Result<Vec<BeadsIssue>> {
        let output = Command::new(&self.bin)
            .args(["list", "--status=open", "--json"])
            .output()
            .context("Failed to run `br list`. Is beads_rust installed?")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("br list failed: {stderr}");
        }

        let issues: Vec<BeadsIssue> =
            serde_json::from_slice(&output.stdout).context("Failed to parse br list output")?;

        Ok(issues)
    }

    /// Update issue status.
    fn update_status(&self, id: &str, status: &str) -> Result<()> {
        let output = Command::new(&self.bin)
            .args(["update", id, &format!("--status={status}")])
            .output()
            .context("Failed to run `br update`")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("br update failed: {stderr}");
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
            .context("Failed to run `br close`")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("br close failed: {stderr}");
        }

        Ok(())
    }
}
