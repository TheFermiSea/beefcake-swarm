//! File locking for parallel issue dispatch via bdh.
//!
//! Prevents concurrent agents from editing the same file across parallel issues.
//! Uses `bdh :aweb lock/unlock` with TTL to prevent deadlocks from crashed agents.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default lock TTL in seconds (30 minutes).
///
/// Generous enough for a full iteration cycle (compile + agent + verify),
/// but bounded so crashed agents don't hold locks forever.
const DEFAULT_LOCK_TTL_SECS: u32 = 1800;

/// Information about an active file lock.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockInfo {
    pub path: String,
    #[serde(default)]
    pub holder: Option<String>,
    #[serde(default)]
    pub ttl_remaining: Option<u32>,
}

/// Manages file locks via bdh for cross-issue conflict prevention.
///
/// When `SWARM_USE_BDH=1`, the orchestrator acquires locks on target files
/// before dispatching work to agents. If another issue already holds a lock,
/// that file is skipped from the target list.
pub struct FileLockManager {
    bin: String,
    wt_path: PathBuf,
}

impl FileLockManager {
    /// Create a new manager for the given worktree.
    pub fn new(wt_path: &Path) -> Self {
        Self {
            bin: std::env::var("SWARM_BDH_BIN").unwrap_or_else(|_| "bdh".into()),
            wt_path: wt_path.to_path_buf(),
        }
    }

    /// Try to acquire a lock on a file path. Returns true if the lock was acquired.
    ///
    /// The lock has a TTL to prevent deadlocks from crashed agents. If the file
    /// is already locked by another agent, returns false (non-blocking).
    pub fn try_lock(&self, path: &str, ttl_secs: u32) -> Result<bool> {
        let ttl_arg = format!("--ttl={ttl_secs}");
        let output = Command::new(&self.bin)
            .args([":aweb", "lock", path, &ttl_arg])
            .current_dir(&self.wt_path)
            .output()
            .with_context(|| format!("Failed to run `{} :aweb lock {path}`", self.bin))?;

        if output.status.success() {
            tracing::debug!(path = %path, ttl = ttl_secs, "File lock acquired");
            Ok(true)
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            if stderr.contains("already locked") || stderr.contains("conflict") {
                tracing::info!(path = %path, stderr = %stderr.trim(), "File already locked by another agent");
                Ok(false)
            } else {
                tracing::warn!(path = %path, stderr = %stderr.trim(), "Lock command failed");
                Ok(false)
            }
        }
    }

    /// Try to lock a file with the default TTL.
    pub fn try_lock_default(&self, path: &str) -> Result<bool> {
        self.try_lock(path, DEFAULT_LOCK_TTL_SECS)
    }

    /// Release a lock on a file path.
    pub fn unlock(&self, path: &str) -> Result<()> {
        let output = Command::new(&self.bin)
            .args([":aweb", "unlock", path])
            .current_dir(&self.wt_path)
            .output()
            .with_context(|| format!("Failed to run `{} :aweb unlock {path}`", self.bin))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!(path = %path, stderr = %stderr.trim(), "Unlock failed (may have expired)");
        }
        Ok(())
    }

    /// Release all locks held by this agent.
    ///
    /// Called during worktree cleanup to ensure no stale locks remain.
    pub fn unlock_all_mine(&self) -> Result<()> {
        let output = Command::new(&self.bin)
            .args([":aweb", "unlock", "--all"])
            .current_dir(&self.wt_path)
            .output()
            .with_context(|| format!("Failed to run `{} :aweb unlock --all`", self.bin))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!(stderr = %stderr.trim(), "unlock --all failed");
        }
        Ok(())
    }

    /// List active locks visible to this agent.
    pub fn active_locks(&self) -> Result<Vec<LockInfo>> {
        let output = Command::new(&self.bin)
            .args([":aweb", "locks", "--json"])
            .current_dir(&self.wt_path)
            .output()
            .with_context(|| format!("Failed to run `{} :aweb locks`", self.bin))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            tracing::warn!(stderr = %stderr.trim(), "Failed to list locks");
            return Ok(vec![]);
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        Ok(serde_json::from_str(&stdout).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "Failed to parse locks JSON");
            vec![]
        }))
    }

    /// Filter a list of target files, removing any that are locked by other agents.
    ///
    /// Tries to lock each file. Files that can't be locked (held by others) are
    /// removed from the list and logged. Returns the subset that was successfully locked.
    pub fn lock_target_files(&self, files: &[String]) -> Vec<String> {
        let mut locked = Vec::with_capacity(files.len());
        for file in files {
            match self.try_lock_default(file) {
                Ok(true) => locked.push(file.clone()),
                Ok(false) => {
                    tracing::info!(
                        file = %file,
                        "Skipping file — locked by another agent"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        file = %file,
                        error = %e,
                        "Lock attempt failed — including file anyway"
                    );
                    locked.push(file.clone());
                }
            }
        }
        locked
    }

    /// Release locks on a list of files.
    pub fn unlock_files(&self, files: &[String]) {
        for file in files {
            if let Err(e) = self.unlock(file) {
                tracing::warn!(file = %file, error = %e, "Failed to unlock file");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lock_info_deserialize() {
        let json = r#"[{"path": "src/main.rs", "holder": "worker-abc", "ttl_remaining": 1200}]"#;
        let locks: Vec<LockInfo> = serde_json::from_str(json).unwrap();
        assert_eq!(locks.len(), 1);
        assert_eq!(locks[0].path, "src/main.rs");
        assert_eq!(locks[0].holder.as_deref(), Some("worker-abc"));
    }

    #[test]
    fn test_default_ttl() {
        assert_eq!(DEFAULT_LOCK_TTL_SECS, 1800);
    }
}
