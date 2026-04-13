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

/// Remove stale git lock files from a repository's `.git/` directory.
///
/// Lock files accumulate from crashed git processes, rapid retry cycles, or
/// concurrent dogfood runs. They block ALL git operations (branch -D, worktree
/// add, etc.) with "Another git process seems to be running" errors.
///
/// Called before each lock-sensitive git operation rather than once at the top
/// of `create()`, because intermediate operations (worktree remove, prune) can
/// create new lock files.
fn clean_git_locks(repo_root: &Path) {
    for lock_name in &["packed-refs.lock", "index.lock", "HEAD.lock"] {
        let lock_path = repo_root.join(".git").join(lock_name);
        if lock_path.exists() {
            tracing::warn!(lock = %lock_path.display(), "Removing stale git lock file");
            let _ = std::fs::remove_file(&lock_path);
        }
    }
}

fn tracked_changes(repo_root: &Path) -> Result<Vec<String>> {
    let output = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .current_dir(repo_root)
        .output()
        .context("Failed to inspect repository status")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("git status failed while inspecting repo_root: {stderr}");
    }

    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| line.get(3..))
        .filter(|path| {
            !path.starts_with(".beads")
                && !path.starts_with(".swarm-")
                && !path.starts_with(".beadhub")
        })
        .map(|path| path.to_string())
        .collect())
}

/// Manages git worktrees for swarm agent tasks.
pub struct WorktreeBridge {
    base_dir: PathBuf,
    repo_root: PathBuf,
}

impl WorktreeBridge {
    /// Create a new WorktreeBridge.
    ///
    /// `base_dir`: parent directory for worktrees. If None, auto-detects using
    /// this fallback chain:
    ///   1. Explicit `base_dir` parameter (this argument)
    ///   2. `SWARM_WORKTREE_DIR` env var
    ///   3. `/cluster/shared/wt/` if NFS mount exists (cluster)
    ///   4. `/tmp/swarm-wt/{repo-name}/` derived from `repo_root` dirname
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

        let base_dir = base_dir
            .or_else(|| std::env::var("SWARM_WORKTREE_DIR").ok().map(PathBuf::from))
            .unwrap_or_else(|| {
                let cluster_path = PathBuf::from("/cluster/shared/wt");
                if cluster_path.exists() {
                    cluster_path
                } else {
                    let repo_name = repo_root
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("default");
                    PathBuf::from(format!("/tmp/swarm-wt/{repo_name}"))
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

    /// Check whether a worktree already exists for the given issue.
    pub fn worktree_exists(&self, issue_id: &str) -> bool {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        if !wt_path.exists() {
            return false;
        }
        // Verify git also tracks it (not just a stale directory)
        self.list()
            .unwrap_or_default()
            .iter()
            .any(|w| w.branch == format!("swarm/{safe_id}"))
    }

    /// Reset an existing worktree to a clean state for retry reuse.
    ///
    /// Runs `git checkout HEAD -- .`, `git clean -fd`, `git reset --hard HEAD`
    /// to restore the worktree to the branch tip. Returns the path on success,
    /// or an error if the worktree doesn't exist or any git command fails.
    /// Caller should fall back to `cleanup()` + `create()` on error.
    pub fn reset_worktree(&self, issue_id: &str) -> Result<PathBuf> {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        if !self.worktree_exists(issue_id) {
            bail!(
                "Worktree for {issue_id} does not exist at {}",
                wt_path.display()
            );
        }
        tracing::info!(path = %wt_path.display(), "Resetting existing worktree for reuse");

        for (args, label) in [
            (vec!["checkout", "HEAD", "--", "."], "git checkout"),
            (vec!["clean", "-fd"], "git clean"),
            (vec!["reset", "--hard", "HEAD"], "git reset"),
        ] {
            let out = Command::new("git")
                .args(&args)
                .current_dir(&wt_path)
                .output()
                .with_context(|| format!("{label} failed in worktree"))?;
            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                bail!("{label} failed: {stderr}");
            }
        }
        Ok(wt_path)
    }

    /// Create a new worktree for the given issue, branching from HEAD.
    ///
    /// Creates branch `swarm/<issue_id>` and places the worktree at `<base_dir>/<issue_id>`.
    #[tracing::instrument(skip(self), fields(issue_id = %issue_id))]
    pub fn create(&self, issue_id: &str) -> Result<PathBuf> {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        let branch = format!("swarm/{safe_id}");

        // Clean locks before any git operations. Called multiple times because
        // each git command can leave new stale locks if it crashes.
        clean_git_locks(&self.repo_root);

        // Clean up stale worktree from a previous failed run.
        // Without this, the loop retries the same issue forever.
        if wt_path.exists() {
            tracing::warn!(
                issue_id = %issue_id,
                path = %wt_path.display(),
                "Stale worktree found — removing before retry"
            );
            let _ = Command::new("git")
                .args([
                    "worktree",
                    "remove",
                    "--force",
                    &wt_path.display().to_string(),
                ])
                .current_dir(&self.repo_root)
                .output();
            // If git worktree remove failed, force-delete the directory
            if wt_path.exists() {
                let _ = std::fs::remove_dir_all(&wt_path);
            }
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
            // 3. If branch exists, delete it — clean locks first since worktree
            //    remove/prune above may have left new stale lock files.
            clean_git_locks(&self.repo_root);
            tracing::warn!(branch = %branch, "Branch already exists, deleting");
            let del_output = retry_git_command(&["branch", "-D", &branch], &self.repo_root, 3)
                .context("Failed to delete existing branch")?;

            if !del_output.status.success() {
                let stderr = String::from_utf8_lossy(&del_output.stderr);
                bail!("Failed to delete existing branch {branch}: {stderr}");
            }
        }

        // Clean locks again before worktree add — branch -D above may have
        // left new lock files if it needed retries.
        clean_git_locks(&self.repo_root);

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
                        b"\n# Orchestrator artifacts\n.swarm-progress.txt\n.swarm-session.json\n.swarm-*\n.swarm-*/\n# Beads data (auto-modified by bd, not real code changes)\n# .beads (no slash) matches the symlink; .beads/ matches the directory fallback\n.beads\n.beads/\n",
                    );
                }
            }
        }

        // Mark tracked .beads/ files as skip-worktree so git ignores
        // modifications by bd commands. Prevents beads backup changes from
        // appearing in git diff/status/add — fixes false positive acceptance,
        // circuit breaker bypass, and agent confusion from beads noise.
        //
        // Best-effort: worktree creation succeeds even if this fails, but we
        // log warnings so failures are diagnosable.
        match Command::new("git")
            .args(["ls-files", ".beads/"])
            .current_dir(&wt_path)
            .output()
        {
            Ok(ls_output) if ls_output.status.success() => {
                let file_list = String::from_utf8_lossy(&ls_output.stdout).to_string();
                let files: Vec<&str> = file_list.lines().filter(|l| !l.is_empty()).collect();
                if !files.is_empty() {
                    let mut args: Vec<&str> = vec!["update-index", "--skip-worktree"];
                    args.extend(&files);
                    match retry_git_command(&args, &wt_path, 3) {
                        Ok(ref out) if !out.status.success() => {
                            let stderr = String::from_utf8_lossy(&out.stderr);
                            tracing::warn!(
                                stderr = %stderr.trim(),
                                "git update-index --skip-worktree failed; .beads/ files may appear in status"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "Failed to run git update-index --skip-worktree"
                            );
                        }
                        _ => {}
                    }
                }
            }
            Ok(ls_output) => {
                let stderr = String::from_utf8_lossy(&ls_output.stderr);
                tracing::warn!(
                    stderr = %stderr.trim(),
                    "git ls-files .beads/ failed; skipping skip-worktree setup"
                );
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to execute git ls-files for .beads/"
                );
            }
        }

        // Replace the worktree's .beads/ directory with a symlink to the main repo's
        // .beads/ so that `bd` commands from the worktree connect to the same Dolt
        // database server as the main repo.
        //
        // Why symlink instead of copy:
        //   - The beads Dolt server uses its startup CWD as the database root.
        //   - The server is started by the first `bd` invocation in the main repo.
        //   - Worktrees share the same server (port 3307 locked in .beads/dolt-server.lock).
        //   - If the worktree runs `bd` against its own (fresh) .beads/dolt/, it starts a
        //     NEW server process with a different CWD → "database not found" errors.
        //   - Symlinking ensures all worktree `bd` commands use the main repo's running
        //     server and authoritative issue database.
        //
        // Safety: skip-worktree bits (set above) prevent git from showing the "missing"
        // tracked files as deleted. The symlink replaces a directory git thinks it owns,
        // but git won't complain because skip-worktree hides those paths.
        let wt_beads = wt_path.join(".beads");
        let main_beads = self.repo_root.join(".beads");
        if main_beads.exists() && wt_beads.exists() && !wt_beads.is_symlink() {
            match std::fs::remove_dir_all(&wt_beads) {
                Ok(()) => {
                    #[cfg(unix)]
                    match std::os::unix::fs::symlink(&main_beads, &wt_beads) {
                        Ok(()) => {
                            tracing::info!(
                                src = %main_beads.display(),
                                dst = %wt_beads.display(),
                                "Symlinked .beads/ to main repo for shared Dolt access"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "Failed to symlink .beads/; bd commands may fail in worktree"
                            );
                        }
                    }
                    #[cfg(not(unix))]
                    tracing::warn!(
                        "Non-Unix platform: cannot symlink .beads/; bd commands may fail in worktree"
                    );
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        "Failed to remove worktree .beads/ for symlink; bd commands may fail"
                    );
                }
            }
        }

        Ok(wt_path)
    }

    /// Merge the worktree branch for an issue into the current repository branch and remove the worktree and its branch.
    ///
    /// Performs these actions in order:
    /// 1. Ensures the worktree has no uncommitted changes (errors if any are present).
    /// 2. Detects and repairs an accidentally committed `.beads` symlink/blob on the branch (if present) so the merge can proceed.
    /// 3. Merges `swarm/<sanitized-issue_id>` into the current branch with `--no-ff`.
    /// 4. Removes the worktree and deletes the `swarm/<sanitized-issue_id>` branch (warnings are logged on non-fatal failures).
    ///
    /// If the merge fails (for example due to conflicts) this function returns an error and does not remove the branch.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// # use anyhow::Result;
    /// # fn example(bridge: &WorktreeBridge) -> Result<()> {
    /// bridge.merge_and_remove("ISSUE-123")?;
    /// # Ok(())
    /// # }
    /// ```
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

            // Restore .beads/ from HEAD to undo the symlink replacement.
            // WorktreeBridge::create() replaces .beads/ with a symlink for bd
            // compatibility. Before merging, we need to restore the tracked
            // directory so git status doesn't report type-change noise.
            let beads_path = wt_path.join(".beads");
            if beads_path.is_symlink() {
                let _ = std::fs::remove_file(&beads_path);
            }
            let _ = Command::new("git")
                .args(["checkout", "HEAD", "--", ".beads"])
                .current_dir(&wt_path)
                .output();
        }

        // Check for uncommitted changes in the worktree (ignoring .beads/ noise)
        if wt_path.exists() {
            let status = Command::new("git")
                .args(["status", "--porcelain"])
                .current_dir(&wt_path)
                .output()
                .context("Failed to check worktree status")?;

            let status_text = String::from_utf8_lossy(&status.stdout);
            // Filter out operational artifacts — these are infrastructure noise,
            // not real code changes. Includes .beads/ (beads DB), .swarm-* (telemetry,
            // workpad, experiment ledger), and .beadhub (coordination config).
            let non_beads_changes: Vec<&str> = status_text
                .lines()
                .filter(|line| {
                    let path = line.get(3..).unwrap_or("");
                    !path.starts_with(".beads")
                        && !path.starts_with(".swarm-")
                        && !path.starts_with(".beadhub")
                })
                .collect();

            if !non_beads_changes.is_empty() {
                bail!("Worktree {issue_id} has uncommitted changes — commit or discard first");
            }
        }

        // Pre-merge: detect and strip .beads symlink from the branch if it was accidentally
        // committed. When `git add .` stages the worktree's .beads symlink as a mode-120000
        // blob, git merge would try to replace .beads/ (a directory) with a blob in the
        // main working tree. This fails with "Updating the following directories would lose
        // untracked files in them" because .beads/ contains active beads database files.
        //
        // Fix: if the branch tip has .beads as a blob (symlink), not a tree (directory),
        // restore the proper .beads/ directory structure from main HEAD and create a fixup
        // commit so the merge can proceed cleanly.
        if wt_path.exists() {
            let ls_beads = Command::new("git")
                .args(["ls-tree", &branch, "--", ".beads"])
                .current_dir(&self.repo_root)
                .output();

            // A symlink shows up as "120000 blob <sha>\t.beads"; a directory as "040000 tree ..."
            let beads_is_blob = ls_beads
                .map(|o| {
                    let out = String::from_utf8_lossy(&o.stdout);
                    out.contains(" blob ") && out.contains(".beads")
                })
                .unwrap_or(false);

            if beads_is_blob {
                tracing::warn!(
                    issue_id = %issue_id,
                    branch = %branch,
                    ".beads committed as symlink/blob — stripping before merge to prevent working-tree conflict"
                );

                // Get the tree SHA for .beads/ from the current main HEAD so we can restore it.
                let tree_sha = Command::new("git")
                    .args(["rev-parse", "HEAD:.beads"])
                    .current_dir(&self.repo_root)
                    .output()
                    .ok()
                    .filter(|o| o.status.success())
                    .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
                    .filter(|s| !s.is_empty());

                if let Some(tree_sha) = tree_sha {
                    // Remove the .beads blob from the worktree branch's index.
                    let rm = retry_git_command(
                        &[
                            "rm",
                            "--cached",
                            "-f",
                            "-q",
                            "--ignore-unmatch",
                            "--",
                            ".beads",
                        ],
                        &wt_path,
                        3,
                    )?;
                    if !rm.status.success() {
                        let stderr = String::from_utf8_lossy(&rm.stderr);
                        bail!("Failed to remove .beads blob from index before merge: {stderr}");
                    }

                    // Restore .beads/ directory entries from main HEAD (index-only, no working
                    // tree update — the working tree still has the symlink, but skip-worktree
                    // hides the discrepancy from git status/add/diff).
                    let read_tree = retry_git_command(
                        &["read-tree", "--prefix=.beads/", &tree_sha],
                        &wt_path,
                        3,
                    )?;
                    if !read_tree.status.success() {
                        let stderr = String::from_utf8_lossy(&read_tree.stderr);
                        bail!("Failed to restore .beads/ tree into index before merge: {stderr}");
                    }

                    // Re-apply skip-worktree bits on the restored .beads/ entries so they
                    // don't appear as local modifications.
                    let ls = retry_git_command(&["ls-files", "--", ".beads/"], &wt_path, 3)?;
                    if ls.status.success() {
                        let file_list = String::from_utf8_lossy(&ls.stdout).to_string();
                        let files: Vec<&str> =
                            file_list.lines().filter(|l| !l.is_empty()).collect();
                        if !files.is_empty() {
                            let mut args = vec!["update-index", "--skip-worktree"];
                            args.extend_from_slice(&files);
                            let sw = retry_git_command(&args, &wt_path, 3)?;
                            if !sw.status.success() {
                                let stderr = String::from_utf8_lossy(&sw.stderr);
                                bail!(
                                    "Failed to set skip-worktree on .beads/ entries before merge: {stderr}"
                                );
                            }
                        }
                    }

                    // Commit the fixup so the merge sees a clean .beads/ directory.
                    let fixup = retry_git_command(
                        &[
                            "commit",
                            "--allow-empty",
                            "-m",
                            "swarm: restore .beads directory (strip accidental symlink)",
                        ],
                        &wt_path,
                        3,
                    )?;
                    if !fixup.status.success() {
                        let stderr = String::from_utf8_lossy(&fixup.stderr);
                        bail!("Failed to commit .beads fixup before merge: {stderr}");
                    }

                    tracing::info!(issue_id = %issue_id, ".beads symlink stripped from branch; proceeding with merge");
                } else {
                    bail!(
                        "Cannot strip .beads symlink from branch '{branch}': \
                         HEAD:.beads tree SHA not found — merge aborted to avoid working-tree conflict"
                    );
                }
            }
        }

        // Auto-stash dirty tracked files (e.g. Cargo.lock dirtied by builds)
        // so they don't block the merge. Pop the stash after merging.
        let repo_root_changes = tracked_changes(&self.repo_root)?;
        let did_stash = if !repo_root_changes.is_empty() {
            let sample = repo_root_changes
                .iter()
                .take(5)
                .cloned()
                .collect::<Vec<_>>()
                .join(", ");
            tracing::info!(
                issue_id = %issue_id,
                dirty_files = %sample,
                "Auto-stashing dirty tracked files before merge"
            );
            let stash = Command::new("git")
                .args([
                    "stash",
                    "push",
                    "-m",
                    &format!("swarm-auto-stash-{issue_id}"),
                ])
                .current_dir(&self.repo_root)
                .output()
                .context("Failed to auto-stash before merge")?;
            if !stash.status.success() {
                let stderr = String::from_utf8_lossy(&stash.stderr);
                bail!(
                    "Auto-stash failed for dirty files ({sample}): {stderr}. \
                     Commit or stash those files manually before retrying."
                );
            }
            true
        } else {
            false
        };

        // Merge the branch into the main repo
        let merge_msg = format!("swarm: merge {issue_id}");
        let merge = retry_git_command(
            &["merge", "--no-ff", &branch, "-m", &merge_msg],
            &self.repo_root,
            3,
        )
        .context("Failed to merge worktree branch")?;

        // Restore stash regardless of merge outcome
        if did_stash {
            let pop = Command::new("git")
                .args(["stash", "pop"])
                .current_dir(&self.repo_root)
                .output();
            match pop {
                Ok(out) if out.status.success() => {
                    tracing::debug!(issue_id = %issue_id, "Auto-stash restored after merge");
                }
                Ok(out) => {
                    let stderr = String::from_utf8_lossy(&out.stderr);
                    tracing::warn!(
                        issue_id = %issue_id,
                        stderr = %stderr.trim(),
                        "Auto-stash pop had conflicts — stash preserved, check manually"
                    );
                }
                Err(e) => {
                    tracing::warn!(issue_id = %issue_id, error = %e, "Failed to pop auto-stash");
                }
            }
        }

        if !merge.status.success() {
            let stderr = String::from_utf8_lossy(&merge.stderr);
            bail!("Merge failed for {issue_id} (possible conflict): {stderr}");
        }

        // --- Post-merge verification ---
        // Quick `cargo check` on the repo root to catch merge-induced regressions.
        // If the merge broke compilation, revert it and reopen the issue.
        // Skip for non-Rust repos (no Cargo.toml at root).
        let has_cargo = self.repo_root.join("Cargo.toml").exists();
        let check = if !has_cargo {
            tracing::debug!(issue_id = %issue_id, "No Cargo.toml — skipping post-merge verification");
            Ok(std::process::Output {
                status: std::process::ExitStatus::default(),
                stdout: vec![],
                stderr: vec![],
            })
        } else {
            Command::new("cargo")
                .args(["check", "--workspace", "--quiet"])
                .current_dir(&self.repo_root)
                .env(
                    "CARGO_TARGET_DIR",
                    std::env::var("CARGO_TARGET_DIR")
                        .unwrap_or_else(|_| "/tmp/beefcake-shared-target".into()),
                )
                .output()
        };
        match check {
            Ok(output) if !output.status.success() => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                tracing::error!(
                    issue_id = %issue_id,
                    stderr = %stderr.chars().take(500).collect::<String>(),
                    "Post-merge cargo check FAILED — reverting merge"
                );
                // Revert the merge commit
                let _ = Command::new("git")
                    .args(["reset", "--hard", "HEAD~1"])
                    .current_dir(&self.repo_root)
                    .output();
                bail!(
                    "Post-merge verification failed for {issue_id}: cargo check errors. Merge reverted."
                );
            }
            Ok(_) => {
                tracing::debug!(issue_id = %issue_id, "Post-merge cargo check passed");
            }
            Err(e) => {
                // cargo check failed to run (not installed, etc.) — proceed anyway
                tracing::warn!(
                    issue_id = %issue_id,
                    error = %e,
                    "Post-merge cargo check could not run — proceeding without verification"
                );
            }
        }

        // --- Push to remote (authoritative) ---
        // Push merged changes to origin/main. If push fails, revert the local merge
        // and fail this landing so the issue is NOT closed with stranded local state.
        let push = retry_git_command(&["push", "origin", "main"], &self.repo_root, 3);
        match push {
            Ok(output) if output.status.success() => {
                tracing::info!(issue_id = %issue_id, "Pushed merge to origin/main");
            }
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                tracing::error!(
                    issue_id = %issue_id,
                    stderr = %stderr.trim(),
                    "git push failed — reverting local merge"
                );
                let revert = Command::new("git")
                    .args(["reset", "--hard", "HEAD~1"])
                    .current_dir(&self.repo_root)
                    .output()
                    .context("Failed to revert merge after push failure")?;
                if !revert.status.success() {
                    let revert_stderr = String::from_utf8_lossy(&revert.stderr);
                    bail!("Push failed for {issue_id} and merge revert failed: {revert_stderr}");
                }
                bail!(
                    "Push failed for {issue_id}: {}. Merge reverted.",
                    stderr.trim()
                );
            }
            Err(e) => {
                tracing::error!(
                    issue_id = %issue_id,
                    error = %e,
                    "git push command failed to execute — reverting local merge"
                );
                let revert = Command::new("git")
                    .args(["reset", "--hard", "HEAD~1"])
                    .current_dir(&self.repo_root)
                    .output()
                    .context("Failed to revert merge after push execution error")?;
                if !revert.status.success() {
                    let revert_stderr = String::from_utf8_lossy(&revert.stderr);
                    bail!(
                        "Push command failed for {issue_id} and merge revert failed: {revert_stderr}"
                    );
                }
                bail!("Push command failed for {issue_id}: {e}. Merge reverted.");
            }
        }

        // Remove the worktree (--force: untracked .swarm-* artifacts are expected)
        match Command::new("git")
            .args([
                "worktree",
                "remove",
                "--force",
                &wt_path.display().to_string(),
            ])
            .current_dir(&self.repo_root)
            .output()
        {
            Ok(remove) => {
                if !remove.status.success() {
                    let stderr = String::from_utf8_lossy(&remove.stderr);
                    tracing::warn!(stderr = %stderr.trim(), "git worktree remove --force warning");
                    // Fallback: force-delete the directory
                    if wt_path.exists() {
                        let _ = std::fs::remove_dir_all(&wt_path);
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to run git worktree remove --force");
                if wt_path.exists() {
                    let _ = std::fs::remove_dir_all(&wt_path);
                }
            }
        }

        // Delete the branch (-D: force delete even if not fully merged,
        // since the merge already succeeded above)
        match Command::new("git")
            .args(["branch", "-D", &branch])
            .current_dir(&self.repo_root)
            .output()
        {
            Ok(del) => {
                if !del.status.success() {
                    let stderr = String::from_utf8_lossy(&del.stderr);
                    tracing::warn!(stderr = %stderr.trim(), "git branch -D warning");
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Failed to run git branch -D");
            }
        }

        Ok(())
    }

    /// Safe post-completion cleanup that preserves unpushed work.
    ///
    /// Unlike `cleanup()` (which force-deletes everything), this method:
    /// 1. Checks for uncommitted changes — commits them if found
    /// 2. Checks for unpushed commits — logs a warning but does NOT delete the branch
    /// 3. Removes the git worktree tracking
    /// 4. Removes the filesystem directory (including `.beads/` remnants)
    /// 5. Only deletes the branch if it's been merged or pushed
    ///
    /// Safe to call after both success and failure paths.
    pub fn safe_cleanup(&self, issue_id: &str) -> Result<()> {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        let branch = format!("swarm/{safe_id}");

        if !wt_path.exists() {
            // Already cleaned — just prune bookkeeping
            let _ = Command::new("git")
                .args(["worktree", "prune"])
                .current_dir(&self.repo_root)
                .output();
            return Ok(());
        }

        // Step 1: Salvage uncommitted changes
        let status = Command::new("git")
            .args(["status", "--porcelain"])
            .current_dir(&wt_path)
            .output();
        let has_changes = status
            .as_ref()
            .map(|o| {
                String::from_utf8_lossy(&o.stdout).lines().any(|line| {
                    let path = line.get(3..).unwrap_or("");
                    !path.starts_with(".beads")
                        && !path.starts_with(".swarm-")
                        && !path.starts_with(".beadhub")
                })
            })
            .unwrap_or(false);
        if has_changes {
            tracing::info!(issue_id, "safe_cleanup: salvaging uncommitted changes");
            let _ = Command::new("git")
                .args(["add", "-A"])
                .current_dir(&wt_path)
                .output();
            let _ = Command::new("git")
                .args([
                    "commit",
                    "-m",
                    &format!("salvage: uncommitted work ({issue_id})"),
                ])
                .current_dir(&wt_path)
                .output();
        }

        // Step 2: Check for unpushed commits
        let log_output = Command::new("git")
            .args(["log", "--oneline", &format!("origin/main..{branch}")])
            .current_dir(&self.repo_root)
            .output();
        let unpushed_count = log_output
            .as_ref()
            .map(|o| String::from_utf8_lossy(&o.stdout).lines().count())
            .unwrap_or(0);
        if unpushed_count > 0 {
            tracing::warn!(
                issue_id,
                unpushed = unpushed_count,
                branch = %branch,
                "safe_cleanup: branch has unpushed commits — preserving branch"
            );
        }

        // Step 3: Remove the git worktree (but not the branch if it has unpushed work)
        let _ = Command::new("git")
            .args([
                "worktree",
                "remove",
                "--force",
                &wt_path.display().to_string(),
            ])
            .current_dir(&self.repo_root)
            .output();

        // Step 4: Remove filesystem remnants (especially .beads/ which git doesn't track)
        if wt_path.exists() {
            if let Err(e) = std::fs::remove_dir_all(&wt_path) {
                tracing::warn!(
                    issue_id,
                    path = %wt_path.display(),
                    error = %e,
                    "safe_cleanup: failed to remove worktree directory"
                );
            }
        }

        // Step 5: Prune git bookkeeping
        let _ = Command::new("git")
            .args(["worktree", "prune"])
            .current_dir(&self.repo_root)
            .output();

        // Step 6: Delete branch only if it has no unpushed commits
        if unpushed_count == 0 {
            let _ = Command::new("git")
                .args(["branch", "-D", &branch])
                .current_dir(&self.repo_root)
                .output();
        }

        tracing::info!(
            issue_id,
            preserved_branch = unpushed_count > 0,
            "safe_cleanup: worktree cleaned"
        );
        Ok(())
    }

    /// Cleanup a worktree for merge failure recovery.
    ///
    /// Best-effort cleanup that:
    /// 1. **Salvages uncommitted work** by committing to the branch before removal
    /// 2. Aborts any in-progress merge
    /// 3. Force-removes the worktree (with fallback to remove_dir_all)
    /// 4. Prunes worktree bookkeeping
    /// 5. Force-deletes the branch
    ///
    /// The salvage commit (step 1) ensures that even when cleanup is called in
    /// failure paths, agent work is preserved on the branch and can be recovered
    /// with `git log swarm/<issue-id>`. Without this, uncommitted changes are
    /// permanently lost when the worktree is removed.
    ///
    /// Always returns `Ok(())` regardless of individual step failures.
    #[tracing::instrument(skip(self), fields(issue_id = %issue_id))]
    pub fn cleanup(&self, issue_id: &str) -> Result<()> {
        let safe_id = Self::sanitize_id(issue_id);
        let wt_path = self.base_dir.join(&safe_id);
        let branch = format!("swarm/{safe_id}");

        // Salvage: commit any uncommitted changes before destroying the worktree.
        // This preserves agent work on the branch even when cleanup is called
        // from failure paths (merge conflict, shutdown signal, circuit breaker).
        if wt_path.exists() {
            let status = Command::new("git")
                .args(["status", "--porcelain"])
                .current_dir(&wt_path)
                .output();

            let has_changes = status
                .as_ref()
                .map(|o| {
                    String::from_utf8_lossy(&o.stdout).lines().any(|line| {
                        let path = line.get(3..).unwrap_or("");
                        !path.starts_with(".beads")
                            && !path.starts_with(".swarm-")
                            && !path.starts_with(".beadhub")
                    })
                })
                .unwrap_or(false);

            if has_changes {
                tracing::info!(issue_id, "Salvaging uncommitted changes before cleanup");
                let _ = Command::new("git")
                    .args(["add", "-A"])
                    .current_dir(&wt_path)
                    .output();
                let _ = Command::new("git")
                    .args([
                        "commit",
                        "-m",
                        &format!("salvage: uncommitted work before cleanup ({issue_id})"),
                    ])
                    .current_dir(&wt_path)
                    .output();
            }
        }

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
            .args(["branch", "--no-color", "--list", "swarm/*"])
            .current_dir(&self.repo_root)
            .output()
            .context("Failed to list swarm branches")?;

        let branch_text = String::from_utf8_lossy(&branch_output.stdout);
        let swarm_branches: Vec<String> = branch_text
            .lines()
            .map(|l| {
                // git branch prefixes: "  " (normal), "* " (current), "+ " (previous)
                l.trim()
                    .trim_start_matches("* ")
                    .trim_start_matches("+ ")
                    .to_string()
            })
            .filter(|b| !b.is_empty() && b.starts_with("swarm/"))
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

        // Creating the same one again should succeed (auto-cleans stale worktree)
        let wt_path2 = bridge
            .create("test-issue")
            .expect("re-create worktree after cleanup");
        assert!(wt_path2.exists());
    }

    #[test]
    fn test_create_sets_skip_worktree_on_beads() {
        let repo_dir = tempfile::tempdir().unwrap();
        let wt_base = tempfile::tempdir().unwrap();

        // Set up a git repo with an initial commit including a .beads/ file
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
        let beads_dir = repo_dir.path().join(".beads").join("backup");
        std::fs::create_dir_all(&beads_dir).unwrap();
        std::fs::write(beads_dir.join("backup_state.json"), "{}").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "init with beads"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");

        // Create worktree — should set skip-worktree on .beads/ files AND symlink .beads/
        let wt_path = bridge.create("beads-test").expect("create worktree");
        assert!(wt_path.exists());

        // .beads/ should be replaced with a symlink to the main repo's .beads/
        let wt_beads_path = wt_path.join(".beads");
        assert!(
            wt_beads_path.is_symlink(),
            ".beads/ should be a symlink to main repo's .beads/"
        );
        assert_eq!(
            std::fs::read_link(&wt_beads_path).unwrap(),
            repo_dir.path().join(".beads"),
            ".beads/ symlink should point to main repo"
        );

        // Mutate the .beads/ file in the worktree (simulates bd backup writes)
        let wt_beads_file = wt_path.join(".beads/backup/backup_state.json");
        assert!(
            wt_beads_file.exists(),
            ".beads/ file should exist in worktree"
        );
        std::fs::write(&wt_beads_file, r#"{"mutated": true}"#).unwrap();

        // git status may show "?? .beads" for the untracked symlink; that is expected.
        // What matters is that .beads/ tracked files (backup JSONL) are hidden by skip-worktree,
        // and that `git add . && git restore --staged .beads` leaves nothing staged for .beads.
        // (git_commit_changes in orchestrator.rs performs this restore after git add .)

        // git diff should also be clean for .beads/
        let diff = Command::new("git")
            .args(["diff", "--stat"])
            .current_dir(&wt_path)
            .output()
            .unwrap();
        let diff_text = String::from_utf8_lossy(&diff.stdout);
        assert!(
            !diff_text.contains(".beads/"),
            "skip-worktree should hide .beads/ from diff, got: {diff_text}"
        );
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
    fn test_merge_and_remove_auto_stashes_dirty_target_repo() {
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
        std::fs::write(repo_dir.path().join("config.toml"), "value = 1\n").unwrap();
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

        // `merge_and_remove` now requires authoritative push to origin/main.
        // Seed a local bare remote so the test exercises the full landing path.
        Command::new("git")
            .args(["branch", "-M", "main"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        let remote_dir = tempfile::tempdir().unwrap();
        Command::new("git")
            .args(["init", "--bare"])
            .current_dir(remote_dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args([
                "remote",
                "add",
                "origin",
                &remote_dir.path().display().to_string(),
            ])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        let push = Command::new("git")
            .args(["push", "-u", "origin", "main"])
            .current_dir(repo_dir.path())
            .output()
            .unwrap();
        assert!(
            push.status.success(),
            "failed to seed origin/main for test: {}",
            String::from_utf8_lossy(&push.stderr)
        );

        let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
            .expect("bridge creation");
        let wt_path = bridge.create("dirty-target").expect("create worktree");

        // Commit a change in the worktree
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

        // Dirty a different file in the main repo (like Cargo.lock from a build)
        std::fs::write(repo_dir.path().join("config.toml"), "value = 2\n").unwrap();

        // Merge should succeed — auto-stash handles the dirty file
        let result = bridge.merge_and_remove("dirty-target");
        assert!(
            result.is_ok(),
            "merge_and_remove should auto-stash dirty files, got: {:?}",
            result.unwrap_err()
        );

        // The dirty file should be restored after merge
        let config = std::fs::read_to_string(repo_dir.path().join("config.toml")).unwrap();
        assert_eq!(
            config, "value = 2\n",
            "auto-stash should restore the dirty file after merge"
        );

        // The worktree change should have landed
        let readme = std::fs::read_to_string(repo_dir.path().join("README.md")).unwrap();
        assert_eq!(
            readme, "worktree change",
            "worktree change should be merged"
        );
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
