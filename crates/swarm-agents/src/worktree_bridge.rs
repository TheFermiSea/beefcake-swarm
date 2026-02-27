//! Worktree Bridge — Git worktree isolation for agent tasks
//!
//! Each agent task runs in an isolated git worktree to prevent conflicts.
//! Uses direct `git worktree` commands (Gastown is overkill for single-agent use).

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::Duration;

/// Info about an active worktree.
#[derive(Debug, Clone)]
pub struct WorktreeInfo {
    pub path: PathBuf,
    pub branch: String,
}

/// Backoff delays for transient git retries (milliseconds).
const RETRY_DELAYS_MS: &[u64] = &[100, 500, 2000];

/// Retry a git command with exponential backoff for transient failures.
///
/// Transient failures include `index.lock` errors (concurrent access) and
/// `Unable to create` errors. Non-transient failures are returned immediately.
pub fn retry_git_command(args: &[&str], working_dir: &Path, max_retries: u32) -> Result<Output> {
    for attempt in 0..=max_retries {
        let output = Command::new("git")
            .args(args)
            .current_dir(working_dir)
            .output()
            .context("Failed to execute git command")?;

        if output.status.success() {
            return Ok(output);
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        let is_transient = stderr.contains("index.lock") || stderr.contains("Unable to create");

        if attempt < max_retries && is_transient {
            let delay = RETRY_DELAYS_MS
                .get(attempt as usize)
                .copied()
                .unwrap_or(2000);
            tracing::warn!(
                attempt = attempt + 1,
                max_retries,
                delay_ms = delay,
                stderr = %stderr.trim(),
                "Transient git failure, retrying"
            );
            std::thread::sleep(Duration::from_millis(delay));
            continue;
        }

        return Ok(output);
    }

    unreachable!()
}

/// Manages git worktrees for swarm agent tasks.
pub struct WorktreeBridge {
    base_dir: PathBuf,
    repo_root: PathBuf,
}

impl WorktreeBridge {
    /// Create a new WorktreeBridge.
    ///
    /// `base_dir`: parent directory for worktrees. If None, auto-detects:
    ///   - `/cluster/shared/wt/` if NFS mount exists (cluster)
    ///   - `/tmp/beefcake-wt/` otherwise (local dev)
    pub fn new(base_dir: Option<PathBuf>, repo_root: impl AsRef<Path>) -> Result<Self> {
        let repo_root = repo_root.as_ref().to_path_buf();

        // Verify repo_root is a git repository
        let check = Command::new("git")
            .args(["rev-parse", "--git-dir"])
            .current_dir(&repo_root)
            .output()
            .context("Failed to check git repo")?;
        if !check.status.success() {
            bail!("Not a git repository: {}", repo_root.display());
        }

        let base_dir = base_dir.unwrap_or_else(|| {
            let cluster_path = PathBuf::from("/cluster/shared/wt");
            if cluster_path.exists() {
                cluster_path
            } else {
                PathBuf::from("/tmp/beefcake-wt")
            }
        });

        // Ensure base directory exists
        std::fs::create_dir_all(&base_dir).with_context(|| {
            format!("Failed to create worktree base dir: {}", base_dir.display())
        })?;

        Ok(Self {
            base_dir,
            repo_root,
        })
    }

    /// Get the repository root path.
    pub fn repo_root(&self) -> &Path {
        &self.repo_root
    }

    /// Sanitize an issue ID for safe use in paths and branch names.
    /// Allows only ASCII alphanumerics, `_`, and `-`. Strips leading dots.
    fn sanitize_id(issue_id: &str) -> String {
        let sanitized: String = issue_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        let trimmed = sanitized.trim_start_matches('.');
        if trimmed.is_empty() {
            "_".to_string()
        } else {
            trimmed.to_string()
        }
    }

    /// Compute the worktree path for a given issue ID.
    pub fn worktree_path(&self, issue_id: &str) -> PathBuf {
        self.base_dir.join(Self::sanitize_id(issue_id))
    }

    /// Create a new worktree for the given issue, branching from HEAD.
    ///
    /// Creates branch `swarm/<issue_id>` and places the worktree at `<base_dir>/<issue_id>`.
    #[tracing::instrument(skip(self), fields(issue_id = %issue_id))]
    pub fn create(&self, issue_id: &str) -> Result<PathBuf> {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        let branch = format!("swarm/{safe_id}");

        if wt_path.exists() {
            bail!(
                "Worktree already exists for {issue_id}: {}",
                wt_path.display()
            );
        }

        // Clean up stale branches from previous failed runs
        // 1. Prune worktree bookkeeping
        let _ = Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&self.repo_root)
            .output();

        // 2. Check if branch already exists
        let check_output = Command::new("git")
            .args(["branch", "--list", &branch])
            .current_dir(&self.repo_root)
            .output()
            .context("Failed to check for existing branch")?;

        let branch_list = String::from_utf8_lossy(&check_output.stdout);
        if !branch_list.trim().is_empty() {
            // 3. If branch exists, delete it
            tracing::warn!(branch = %branch, "Branch already exists, deleting");
            let del_output = Command::new("git")
                .args(["branch", "-D", &branch])
                .current_dir(&self.repo_root)
                .output()
                .context("Failed to delete existing branch")?;

            if !del_output.status.success() {
                let stderr = String::from_utf8_lossy(&del_output.stderr);
                bail!("Failed to delete existing branch {branch}: {stderr}");
            }
        }

        let wt_path_str = wt_path.display().to_string();
        let output = retry_git_command(
            &["worktree", "add", "-b", &branch, &wt_path_str],
            &self.repo_root,
            3,
        )
        .context("Failed to run git worktree add")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("git worktree add failed: {stderr}");
        }

        // Write orchestrator artifact exclusions to the worktree's local exclude file
        // instead of modifying the tracked .gitignore. This prevents `git add .` from
        // staging .gitignore changes into agent-authored commits.
        let git_dir = wt_path.join(".git");
        let exclude_dir = if git_dir.is_file() {
            // Worktrees have a .git file pointing to the real git dir
            let pointer = std::fs::read_to_string(&git_dir).unwrap_or_default();
            let real_dir = pointer
                .strip_prefix("gitdir: ")
                .unwrap_or("")
                .trim()
                .to_string();
            if real_dir.is_empty() {
                None
            } else {
                Some(PathBuf::from(real_dir).join("info"))
            }
        } else if git_dir.is_dir() {
            Some(git_dir.join("info"))
        } else {
            None
        };

        if let Some(info_dir) = exclude_dir {
            let _ = std::fs::create_dir_all(&info_dir);
            let exclude_file = info_dir.join("exclude");
            let needs_write = if exclude_file.exists() {
                let content = std::fs::read_to_string(&exclude_file).unwrap_or_default();
                !content.contains(".swarm-progress.txt")
            } else {
                true
            };
            if needs_write {
                use std::io::Write;
                if let Ok(mut f) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&exclude_file)
                {
                    let _ = f.write_all(
                        b"\n# Orchestrator artifacts\n.swarm-progress.txt\n.swarm-session.json\n",
                    );
                }
            }
        }

        Ok(wt_path)
    }

    /// Merge the worktree branch back into the current branch and clean up.
    ///
    /// 1. Checks for uncommitted changes in the worktree
    /// 2. Merges `swarm/<issue_id>` with --no-ff
    /// 3. Removes the worktree
    /// 4. Deletes the branch
    #[tracing::instrument(skip(self), fields(issue_id = %issue_id))]
    pub fn merge_and_remove(&self, issue_id: &str) -> Result<()> {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        let branch = format!("swarm/{safe_id}");

        // Clean up orchestrator-generated artifact files that aren't part of the source code.
        // These are created by the harness (ProgressTracker, SessionManager) during the
        // orchestration loop and must not block the merge.
        if wt_path.exists() {
            for artifact in &[".swarm-progress.txt", ".swarm-session.json"] {
                let artifact_path = wt_path.join(artifact);
                if artifact_path.exists() {
                    let _ = std::fs::remove_file(&artifact_path);
                }
            }

            // Remove untracked files only (not tracked modifications)
            let _ = Command::new("git")
                .args(["clean", "-fd"])
                .current_dir(&wt_path)
                .output();
        }

        // Check for uncommitted changes in the worktree
        if wt_path.exists() {
            let status = Command::new("git")
                .args(["status", "--porcelain"])
                .current_dir(&wt_path)
                .output()
                .context("Failed to check worktree status")?;

            let status_text = String::from_utf8_lossy(&status.stdout);
            if !status_text.trim().is_empty() {
                bail!("Worktree {issue_id} has uncommitted changes — commit or discard first");
            }
        }

        // Merge the branch into the main repo
        let merge_msg = format!("swarm: merge {issue_id}");
        let merge = retry_git_command(
            &["merge", "--no-ff", &branch, "-m", &merge_msg],
            &self.repo_root,
            3,
        )
        .context("Failed to merge worktree branch")?;

        if !merge.status.success() {
            let stderr = String::from_utf8_lossy(&merge.stderr);
            bail!("Merge failed for {issue_id} (possible conflict): {stderr}");
        }

        // Remove the worktree
        let remove = Command::new("git")
            .args(["worktree", "remove", &wt_path.display().to_string()])
            .current_dir(&self.repo_root)
            .output()
            .context("Failed to remove worktree")?;

        if !remove.status.success() {
            let stderr = String::from_utf8_lossy(&remove.stderr);
            tracing::warn!(stderr = %stderr.trim(), "git worktree remove warning");
        }

        // Delete the branch
        let del = Command::new("git")
            .args(["branch", "-d", &branch])
            .current_dir(&self.repo_root)
            .output()
            .context("Failed to delete branch")?;

        if !del.status.success() {
            let stderr = String::from_utf8_lossy(&del.stderr);
            tracing::warn!(stderr = %stderr.trim(), "git branch -d warning");
        }

        Ok(())
    }

    /// Cleanup a worktree for merge failure recovery.
    ///
    /// Best-effort cleanup that:
    /// 1. Aborts any in-progress merge
    /// 2. Force-removes the worktree (with fallback to remove_dir_all)
    /// 3. Prunes worktree bookkeeping
    /// 4. Force-deletes the branch
    ///
    /// Always returns `Ok(())` regardless of individual step failures.
    #[tracing::instrument(skip(self), fields(issue_id = %issue_id))]
    pub fn cleanup(&self, issue_id: &str) -> Result<()> {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        let branch = format!("swarm/{safe_id}");

        // Abort any in-progress merge (may have left repo in conflicted state)
        let _ = Command::new("git")
            .args(["merge", "--abort"])
            .current_dir(&self.repo_root)
            .output();

        // Force-remove the worktree
        if wt_path.exists() {
            let remove = Command::new("git")
                .args([
                    "worktree",
                    "remove",
                    "--force",
                    &wt_path.display().to_string(),
                ])
                .current_dir(&self.repo_root)
                .output();

            match remove {
                Ok(ref out) if !out.status.success() => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    tracing::warn!(stderr = %stderr.trim(), "cleanup: git worktree remove --force failed");
                    // Fallback: try removing the directory directly
                    if wt_path.exists() {
                        let _ = std::fs::remove_dir_all(&wt_path);
                    }
                }
                Err(e) => {
                    tracing::warn!(error = %e, "cleanup: failed to run git worktree remove");
                    if wt_path.exists() {
                        let _ = std::fs::remove_dir_all(&wt_path);
                    }
                }
                _ => {}
            }
        }

        // Prune worktree bookkeeping for removed directories
        let _ = Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&self.repo_root)
            .output();

        // Force-delete the branch (capital -D since it's unmerged)
        let del = Command::new("git")
            .args(["branch", "-D", &branch])
            .current_dir(&self.repo_root)
            .output();

        match del {
            Ok(ref out) if !out.status.success() => {
                let stderr = String::from_utf8_lossy(&out.stderr);
                tracing::warn!(stderr = %stderr.trim(), "cleanup: git branch -D failed");
            }
            Err(e) => {
                tracing::warn!(error = %e, "cleanup: failed to run git branch -D");
            }
            _ => {}
        }

        Ok(())
    }

    /// Clean up zombie `swarm/*` branches that have no associated worktree.
    ///
    /// These accumulate when the orchestrator crashes mid-run or worktrees are
    /// manually deleted without cleaning up branches. Runs on startup to prevent
    /// branch pollution.
    ///
    /// Steps:
    /// 1. `git worktree prune` — sync bookkeeping with filesystem
    /// 2. List all local branches matching `swarm/*`
    /// 3. For each, check if a live worktree references that branch
    /// 4. If not, force-delete the orphaned branch
    ///
    /// Returns the list of branches that were cleaned up.
    pub fn cleanup_stale(&self) -> Result<Vec<String>> {
        // 1. Prune worktree bookkeeping
        let _ = Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&self.repo_root)
            .output();

        // 2. Get all local branches matching swarm/*
        let branch_output = Command::new("git")
            .args(["branch", "--list", "swarm/*"])
            .current_dir(&self.repo_root)
            .output()
            .context("Failed to list swarm branches")?;

        let branch_text = String::from_utf8_lossy(&branch_output.stdout);
        let swarm_branches: Vec<String> = branch_text
            .lines()
            .map(|l| l.trim().trim_start_matches("* ").to_string())
            .filter(|b| !b.is_empty())
            .collect();

        if swarm_branches.is_empty() {
            return Ok(vec![]);
        }

        // 3. Get live worktree branches
        let live_worktrees = self.list().unwrap_or_default();
        let live_branches: std::collections::HashSet<&str> =
            live_worktrees.iter().map(|w| w.branch.as_str()).collect();

        // 4. Delete orphaned branches
        let mut cleaned = Vec::new();
        for branch in &swarm_branches {
            if live_branches.contains(branch.as_str()) {
                continue;
            }

            tracing::info!(branch = %branch, "Cleaning up zombie branch (no worktree)");
            let del = Command::new("git")
                .args(["branch", "-D", branch])
                .current_dir(&self.repo_root)
                .output();

            match del {
                Ok(ref out) if out.status.success() => {
                    cleaned.push(branch.clone());
                }
                Ok(ref out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    tracing::warn!(branch = %branch, stderr = %stderr.trim(), "Failed to delete zombie branch");
                }
                Err(e) => {
                    tracing::warn!(branch = %branch, error = %e, "Failed to run git branch -D");
                }
            }
        }

        Ok(cleaned)
    }

    /// List active worktrees.
    pub fn list(&self) -> Result<Vec<WorktreeInfo>> {
        let output = Command::new("git")
            .args(["worktree", "list", "--porcelain"])
            .current_dir(&self.repo_root)
            .output()
            .context("Failed to list worktrees")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("git worktree list failed: {stderr}");
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut infos = Vec::new();
        let mut current_path: Option<PathBuf> = None;
        let mut current_branch: Option<String> = None;

        for line in stdout.lines() {
            if let Some(path_str) = line.strip_prefix("worktree ") {
                // Flush previous entry
                if let (Some(path), Some(branch)) = (current_path.take(), current_branch.take()) {
                    infos.push(WorktreeInfo { path, branch });
                }
                current_path = Some(PathBuf::from(path_str));
            } else if let Some(branch_ref) = line.strip_prefix("branch refs/heads/") {
                current_branch = Some(branch_ref.to_string());
            }
        }

        // Flush last entry
        if let (Some(path), Some(branch)) = (current_path, current_branch) {
            infos.push(WorktreeInfo { path, branch });
        }

        Ok(infos)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_git_command_succeeds_first_try() {
        let dir = tempfile::tempdir().unwrap();
        Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // A simple git command should succeed on the first try
        let output = retry_git_command(&["status"], dir.path(), 3).unwrap();
        assert!(output.status.success());
    }

    #[test]
    fn test_retry_git_command_non_transient_failure_no_retry() {
        let dir = tempfile::tempdir().unwrap();
        // Not a git repo — should fail immediately (non-transient error)
        let output = retry_git_command(&["status"], dir.path(), 3).unwrap();
        assert!(!output.status.success());
    }

    #[test]
    fn test_sanitize_id() {
        assert_eq!(WorktreeBridge::sanitize_id("beads-abc"), "beads-abc");
        assert_eq!(WorktreeBridge::sanitize_id("../../../etc"), "_________etc");
        assert_eq!(WorktreeBridge::sanitize_id("ok/path"), "ok_path");
        assert_eq!(WorktreeBridge::sanitize_id("...dots"), "___dots");
        assert_eq!(WorktreeBridge::sanitize_id(""), "_");
    }

    #[test]
    fn test_sanitize_id_edge_cases() {
        // Unicode/emoji: each char is replaced with '_'
        assert_eq!(WorktreeBridge::sanitize_id("issue-🔥-hot"), "issue-_-hot");
        // Spaces
        assert_eq!(WorktreeBridge::sanitize_id("hello world"), "hello_world");
        // All special chars
        assert_eq!(WorktreeBridge::sanitize_id("@#$%^&*()"), "_________");
        // Only dashes
        assert_eq!(WorktreeBridge::sanitize_id("---"), "---");
        // Only underscores
        assert_eq!(WorktreeBridge::sanitize_id("___"), "___");
        // Newlines/tabs
        assert_eq!(WorktreeBridge::sanitize_id("a\nb\tc"), "a_b_c");
        // Single valid char
        assert_eq!(WorktreeBridge::sanitize_id("x"), "x");
        // Single invalid char
        assert_eq!(WorktreeBridge::sanitize_id("/"), "_");
        // Mixed case preserved
        assert_eq!(WorktreeBridge::sanitize_id("AbCdEf"), "AbCdEf");
        // Numbers only
        assert_eq!(WorktreeBridge::sanitize_id("12345"), "12345");
        // Leading special then valid
        assert_eq!(WorktreeBridge::sanitize_id("!!valid"), "__valid");
        // Trailing special
        assert_eq!(WorktreeBridge::sanitize_id("valid!!"), "valid__");
    }

    #[test]
    fn test_worktree_path() {
        let bridge = WorktreeBridge {
            base_dir: PathBuf::from("/tmp/test-wt"),
            repo_root: PathBuf::from("/tmp/repo"),
        };
        assert_eq!(
            bridge.worktree_path("beads-abc"),
            PathBuf::from("/tmp/test-wt/beads-abc")
        );
        // Path traversal attempt gets sanitized
        assert_eq!(
            bridge.worktree_path("../../etc/passwd"),
            PathBuf::from("/tmp/test-wt/______etc_passwd")
        );
    }

    #[test]
    fn test_create_and_list() {
        let repo_dir = tempfile::tempdir().unwrap();
        let wt_base = tempfile::tempdir().unwrap();

        // Set up a proper git repo with an initial commit
        Command::new("git")
            .args(["init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        std::fs::write(repo_dir.path().join("README.md"), "hello").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");

        // Create a worktree
        let wt_path = bridge.create("test-issue").expect("create worktree");
        assert!(wt_path.exists());

        // List should include our new worktree
        let list = bridge.list().expect("list worktrees");
        assert!(list.iter().any(|w| w.branch == "swarm/test-issue"));

        // Creating the same one again should fail
        assert!(bridge.create("test-issue").is_err());
    }

    #[test]
    fn test_cleanup_removes_worktree() {
        let repo_dir = tempfile::tempdir().unwrap();
        let wt_base = tempfile::tempdir().unwrap();

        // Set up a proper git repo with an initial commit
        Command::new("git")
            .args(["init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        std::fs::write(repo_dir.path().join("README.md"), "hello").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");

        // Create a worktree
        let wt_path = bridge.create("cleanup-test").expect("create worktree");
        assert!(wt_path.exists());

        // Cleanup should remove it
        bridge.cleanup("cleanup-test").expect("cleanup");
        assert!(!wt_path.exists());

        // Branch should be gone too
        let branches = Command::new("git")
            .args(["branch", "--list", "swarm/cleanup-test"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        let branch_list = String::from_utf8_lossy(&branches.stdout);
        assert!(branch_list.trim().is_empty());
    }

    #[test]
    fn test_cleanup_nonexistent_is_ok() {
        let repo_dir = tempfile::tempdir().unwrap();
        let wt_base = tempfile::tempdir().unwrap();

        Command::new("git")
            .args(["init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        std::fs::write(repo_dir.path().join("README.md"), "hello").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");

        // Cleanup of non-existent worktree should succeed (best-effort)
        bridge
            .cleanup("does-not-exist")
            .expect("cleanup non-existent");
    }

    #[test]
    fn test_create_cleans_stale_branch() {
        let repo_dir = tempfile::tempdir().unwrap();
        let wt_base = tempfile::tempdir().unwrap();

        // Set up a proper git repo with an initial commit
        Command::new("git")
            .args(["init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        std::fs::write(repo_dir.path().join("README.md"), "hello").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");

        // Simulate a stale branch from a previous failed run
        let _ = Command::new("git")
            .args(["branch", "swarm/test-stale"])
            .current_dir(repo_dir.path())
            .output();

        // Verify branch exists
        let check = Command::new("git")
            .args(["branch", "--list", "swarm/test-stale"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        let branch_list = String::from_utf8_lossy(&check.stdout);
        assert!(!branch_list.trim().is_empty());

        // Create should clean up the stale branch and succeed
        let wt_path = bridge
            .create("test-stale")
            .expect("create worktree with stale branch");
        assert!(wt_path.exists());

        // List should include our new worktree
        let list = bridge.list().expect("list worktrees");
        assert!(list.iter().any(|w| w.branch == "swarm/test-stale"));
    }

    #[test]
    fn test_merge_conflict_reports_error() {
        let repo_dir = tempfile::tempdir().unwrap();
        let wt_base = tempfile::tempdir().unwrap();

        // Set up git repo with initial commit
        Command::new("git")
            .args(["init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        std::fs::write(repo_dir.path().join("README.md"), "hello").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");

        // Create worktree
        let wt_path = bridge.create("conflict-test").expect("create worktree");

        // Make a change in the worktree and commit
        std::fs::write(wt_path.join("README.md"), "worktree change").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(&wt_path)
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "worktree change"])
            .current_dir(&wt_path)
            .output()
            .unwrap();

        // Make a conflicting change on the main branch
        std::fs::write(repo_dir.path().join("README.md"), "main change").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "main change"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        // Merge should fail due to conflict
        let result = bridge.merge_and_remove("conflict-test");
        assert!(result.is_err(), "merge_and_remove should fail on conflict");
        let err_msg = result.unwrap_err().to_string();
        assert!(
            err_msg.contains("Merge failed"),
            "error should mention merge failure, got: {err_msg}"
        );

        // Cleanup should succeed even after failed merge
        bridge
            .cleanup("conflict-test")
            .expect("cleanup after conflict");
    }

    #[test]
    fn test_cleanup_stale_removes_orphaned_branches() {
        let repo_dir = tempfile::tempdir().unwrap();
        let wt_base = tempfile::tempdir().unwrap();

        Command::new("git")
            .args(["init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        std::fs::write(repo_dir.path().join("README.md"), "hello").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");

        // Create orphaned swarm branches (no worktree)
        Command::new("git")
            .args(["branch", "swarm/zombie-1"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["branch", "swarm/zombie-2"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        // Create a live worktree (should NOT be cleaned up)
        let _wt_path = bridge.create("live-issue").expect("create worktree");

        // Run cleanup_stale
        let cleaned = bridge.cleanup_stale().expect("cleanup_stale");

        // Should have cleaned the two zombies
        assert_eq!(
            cleaned.len(),
            2,
            "expected 2 zombies cleaned, got: {cleaned:?}"
        );
        assert!(cleaned.contains(&"swarm/zombie-1".to_string()));
        assert!(cleaned.contains(&"swarm/zombie-2".to_string()));

        // Verify zombie branches are gone
        let branches = Command::new("git")
            .args(["branch", "--list", "swarm/*"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        let branch_list = String::from_utf8_lossy(&branches.stdout);
        assert!(
            !branch_list.contains("zombie"),
            "zombie branches should be gone, got: {branch_list}"
        );

        // Live worktree branch should still exist
        assert!(
            branch_list.contains("swarm/live-issue"),
            "live branch should still exist, got: {branch_list}"
        );
    }

    #[test]
    fn test_cleanup_stale_no_zombies() {
        let repo_dir = tempfile::tempdir().unwrap();
        let wt_base = tempfile::tempdir().unwrap();

        Command::new("git")
            .args(["init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        std::fs::write(repo_dir.path().join("README.md"), "hello").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");

        // No swarm branches at all
        let cleaned = bridge.cleanup_stale().expect("cleanup_stale");
        assert!(cleaned.is_empty(), "nothing to clean");
    }
}
