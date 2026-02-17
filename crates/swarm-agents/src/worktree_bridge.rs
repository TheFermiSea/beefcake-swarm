//! Worktree Bridge — Git worktree isolation for agent tasks
//!
//! Each agent task runs in an isolated git worktree to prevent conflicts.
//! Uses direct `git worktree` commands (Gastown is overkill for single-agent use).

use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Info about an active worktree.
#[derive(Debug, Clone)]
pub struct WorktreeInfo {
    pub path: PathBuf,
    pub branch: String,
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

        let output = Command::new("git")
            .args([
                "worktree",
                "add",
                "-b",
                &branch,
                &wt_path.display().to_string(),
            ])
            .current_dir(&self.repo_root)
            .output()
            .context("Failed to run git worktree add")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            bail!("git worktree add failed: {stderr}");
        }

        // Ensure orchestrator artifacts are gitignored in the worktree.
        // The auto-fix step uses `git add .` which would otherwise stage these.
        let gitignore = wt_path.join(".gitignore");
        let artifacts = ".swarm-progress.txt\n.swarm-session.json\n";
        let needs_append = if gitignore.exists() {
            let content = std::fs::read_to_string(&gitignore).unwrap_or_default();
            !content.contains(".swarm-progress.txt")
        } else {
            true
        };
        if needs_append {
            use std::io::Write;
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&gitignore)
            {
                let _ = f.write_all(b"\n# Orchestrator artifacts\n");
                let _ = f.write_all(artifacts.as_bytes());
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
    pub fn merge_and_remove(&self, issue_id: &str) -> Result<()> {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        let branch = format!("swarm/{safe_id}");

        // Clean up orchestrator-generated files that aren't part of the source code.
        // These are created by the harness (ProgressTracker, SessionManager) during the
        // orchestration loop and must not block the merge.
        if wt_path.exists() {
            for artifact in &[".swarm-progress.txt", ".swarm-session.json", ".gitignore"] {
                let artifact_path = wt_path.join(artifact);
                if artifact_path.exists() {
                    let _ = std::fs::remove_file(&artifact_path);
                }
            }

            // Discard any remaining untracked/modified non-source files
            let _ = Command::new("git")
                .args(["checkout", "--", "."])
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
        let merge = Command::new("git")
            .args([
                "merge",
                "--no-ff",
                &branch,
                "-m",
                &format!("swarm: merge {issue_id}"),
            ])
            .current_dir(&self.repo_root)
            .output()
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
            tracing::warn!("git worktree remove warning: {stderr}");
        }

        // Delete the branch
        let del = Command::new("git")
            .args(["branch", "-d", &branch])
            .current_dir(&self.repo_root)
            .output()
            .context("Failed to delete branch")?;

        if !del.status.success() {
            let stderr = String::from_utf8_lossy(&del.stderr);
            tracing::warn!("git branch -d warning: {stderr}");
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
                    tracing::warn!("cleanup: git worktree remove --force failed: {stderr}");
                    // Fallback: try removing the directory directly
                    if wt_path.exists() {
                        let _ = std::fs::remove_dir_all(&wt_path);
                    }
                }
                Err(e) => {
                    tracing::warn!("cleanup: failed to run git worktree remove: {e}");
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
                tracing::warn!("cleanup: git branch -D failed: {stderr}");
            }
            Err(e) => {
                tracing::warn!("cleanup: failed to run git branch -D: {e}");
            }
            _ => {}
        }

        Ok(())
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
    fn test_sanitize_id() {
        assert_eq!(WorktreeBridge::sanitize_id("beads-abc"), "beads-abc");
        assert_eq!(WorktreeBridge::sanitize_id("../../../etc"), "_________etc");
        assert_eq!(WorktreeBridge::sanitize_id("ok/path"), "ok_path");
        assert_eq!(WorktreeBridge::sanitize_id("...dots"), "___dots");
        assert_eq!(WorktreeBridge::sanitize_id(""), "_");
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
}
