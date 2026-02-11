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

/// Bridge to the `br` (beads_rust) CLI binary.
///
/// beads_rust is a binary-only tool — no lib.rs — so we shell out.
pub struct BeadsBridge {
    bin: String,
}

impl BeadsBridge {
    pub fn new() -> Self {
        Self {
            bin: "br".to_string(),
        }
    }

    /// List open issues, sorted by priority.
    pub fn list_open(&self) -> Result<Vec<BeadsIssue>> {
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

    /// Update issue status.
    pub fn update_status(&self, id: &str, status: &str) -> Result<()> {
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
    pub fn close(&self, id: &str, reason: Option<&str>) -> Result<()> {
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
