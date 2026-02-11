//! Git state manager for checkpoints and rollback
//!
//! Handles git operations for the harness.

use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::types::GitCommitInfo;
use chrono::{DateTime, Utc};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default number of retry attempts for transient failures
const DEFAULT_MAX_RETRIES: u32 = 3;

/// Base delay between retries in milliseconds
const RETRY_BASE_DELAY_MS: u64 = 100;

/// Git state manager
pub struct GitManager {
    working_dir: PathBuf,
    commit_prefix: String,
    max_retries: u32,
}

impl GitManager {
    /// Create manager for working directory
    pub fn new(working_dir: impl AsRef<Path>, commit_prefix: impl Into<String>) -> Self {
        Self {
            working_dir: working_dir.as_ref().to_path_buf(),
            commit_prefix: commit_prefix.into(),
            max_retries: DEFAULT_MAX_RETRIES,
        }
    }

    /// Create manager with custom retry settings
    pub fn with_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Run git command and return output
    fn run_git(&self, args: &[&str]) -> HarnessResult<String> {
        let output = Command::new("git")
            .args(args)
            .current_dir(&self.working_dir)
            .output()
            .map_err(|e| HarnessError::git("execute", e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(HarnessError::git(args.join(" "), stderr.to_string()));
        }

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Run git command with automatic retry for transient failures
    ///
    /// Uses exponential backoff: 100ms, 200ms, 400ms, etc.
    fn run_git_with_retry(&self, args: &[&str]) -> HarnessResult<String> {
        let mut last_error = None;

        for attempt in 0..=self.max_retries {
            match self.run_git(args) {
                Ok(output) => return Ok(output),
                Err(e) => {
                    if e.is_retryable() && attempt < self.max_retries {
                        // Exponential backoff
                        let delay = RETRY_BASE_DELAY_MS * (1 << attempt);
                        std::thread::sleep(std::time::Duration::from_millis(delay));
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| HarnessError::git("retry", "Max retries exceeded")))
    }

    /// Get current branch name
    pub fn current_branch(&self) -> HarnessResult<String> {
        self.run_git(&["rev-parse", "--abbrev-ref", "HEAD"])
    }

    /// Get current commit hash (short)
    pub fn current_commit(&self) -> HarnessResult<String> {
        self.run_git(&["rev-parse", "--short", "HEAD"])
    }

    /// Get current commit hash (full)
    pub fn current_commit_full(&self) -> HarnessResult<String> {
        self.run_git(&["rev-parse", "HEAD"])
    }

    /// Check if working directory has uncommitted changes
    pub fn has_uncommitted_changes(&self) -> HarnessResult<bool> {
        let status = self.run_git(&["status", "--porcelain"])?;
        Ok(!status.is_empty())
    }

    /// Get recent commits
    pub fn recent_commits(&self, count: usize) -> HarnessResult<Vec<GitCommitInfo>> {
        let format = "--format=%h|%s|%aI";
        let output = self.run_git(&["log", format, &format!("-{}", count)])?;

        let commits: Vec<GitCommitInfo> = output
            .lines()
            .filter_map(|line| {
                let parts: Vec<&str> = line.splitn(3, '|').collect();
                if parts.len() >= 2 {
                    let hash = parts[0].to_string();
                    let message = parts[1].to_string();
                    let timestamp = parts
                        .get(2)
                        .and_then(|ts| DateTime::parse_from_rfc3339(ts).ok())
                        .map(|dt| dt.with_timezone(&Utc));
                    let is_harness_checkpoint = message.starts_with(&self.commit_prefix);

                    Some(GitCommitInfo {
                        hash,
                        message,
                        timestamp,
                        is_harness_checkpoint,
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(commits)
    }

    /// Get harness checkpoints only
    pub fn harness_checkpoints(&self, count: usize) -> HarnessResult<Vec<GitCommitInfo>> {
        let commits = self.recent_commits(count * 3)?; // Fetch extra to filter
        Ok(commits
            .into_iter()
            .filter(|c| c.is_harness_checkpoint)
            .take(count)
            .collect())
    }

    /// Create checkpoint commit with automatic retry for transient failures
    pub fn create_checkpoint(&self, feature: &str, description: &str) -> HarnessResult<String> {
        // Stage all changes (retry-safe)
        self.run_git_with_retry(&["add", "-A"])?;

        // Check if there's anything to commit
        if !self.has_uncommitted_changes()? {
            return Err(HarnessError::git("commit", "Nothing to commit"));
        }

        // Create commit (retry-safe)
        let message = format!("{} {}: {}", self.commit_prefix, feature, description);
        self.run_git_with_retry(&["commit", "-m", &message])?;

        // Return new commit hash
        self.current_commit()
    }

    /// Rollback to specific commit (soft reset) with retry
    pub fn rollback(&self, commit_hash: &str) -> HarnessResult<()> {
        // Verify commit exists
        self.run_git(&["cat-file", "-t", commit_hash])?;

        // Soft reset to preserve changes in working directory (retry-safe)
        self.run_git_with_retry(&["reset", "--soft", commit_hash])?;

        Ok(())
    }

    /// Hard rollback (discard changes) with retry
    pub fn hard_rollback(&self, commit_hash: &str) -> HarnessResult<()> {
        // Verify commit exists
        self.run_git(&["cat-file", "-t", commit_hash])?;

        // Hard reset (retry-safe)
        self.run_git_with_retry(&["reset", "--hard", commit_hash])?;

        Ok(())
    }

    /// Stash current changes with retry
    pub fn stash(&self, message: &str) -> HarnessResult<()> {
        self.run_git_with_retry(&["stash", "push", "-m", message])?;
        Ok(())
    }

    /// Pop stashed changes with retry
    pub fn stash_pop(&self) -> HarnessResult<()> {
        self.run_git_with_retry(&["stash", "pop"])?;
        Ok(())
    }

    /// Get diff stat since commit
    pub fn diff_stat(&self, since_commit: &str) -> HarnessResult<String> {
        self.run_git(&["diff", "--stat", since_commit])
    }

    /// Count commits since reference
    pub fn commits_since(&self, since_commit: &str) -> HarnessResult<usize> {
        let output = self.run_git(&["rev-list", "--count", &format!("{}..HEAD", since_commit)])?;
        output
            .parse()
            .map_err(|_| HarnessError::git("count", "Failed to parse commit count"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use tempfile::tempdir;

    fn setup_git_repo() -> (tempfile::TempDir, GitManager) {
        let dir = tempdir().unwrap();

        // Initialize git repo
        Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // Configure git user
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // Create initial commit
        std::fs::write(dir.path().join("README.md"), "# Test").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "Initial commit"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let manager = GitManager::new(dir.path(), "[harness]");
        (dir, manager)
    }

    #[test]
    fn test_current_branch() {
        let (_dir, manager) = setup_git_repo();
        let branch = manager.current_branch().unwrap();
        assert!(!branch.is_empty());
    }

    #[test]
    fn test_current_commit() {
        let (_dir, manager) = setup_git_repo();
        let commit = manager.current_commit().unwrap();
        assert!(!commit.is_empty());
        assert!(commit.len() >= 7); // Short hash
    }

    #[test]
    fn test_has_uncommitted_changes() {
        let (dir, manager) = setup_git_repo();

        // Should be clean initially
        assert!(!manager.has_uncommitted_changes().unwrap());

        // Create uncommitted change
        std::fs::write(dir.path().join("new_file.txt"), "content").unwrap();
        assert!(manager.has_uncommitted_changes().unwrap());
    }

    #[test]
    fn test_recent_commits() {
        let (_dir, manager) = setup_git_repo();
        let commits = manager.recent_commits(10).unwrap();
        assert!(!commits.is_empty());
        assert_eq!(commits[0].message, "Initial commit");
    }

    #[test]
    fn test_create_checkpoint() {
        let (dir, manager) = setup_git_repo();

        // Create change
        std::fs::write(dir.path().join("feature.txt"), "feature content").unwrap();

        // Create checkpoint
        let hash = manager
            .create_checkpoint("feature-1", "Implemented feature")
            .unwrap();
        assert!(!hash.is_empty());

        // Verify commit message
        let commits = manager.recent_commits(1).unwrap();
        assert!(commits[0].message.contains("[harness]"));
        assert!(commits[0].message.contains("feature-1"));
        assert!(commits[0].is_harness_checkpoint);
    }
}
