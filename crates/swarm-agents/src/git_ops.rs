//! Git operations for the swarm orchestrator.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context as _, Result};
use tracing::warn;

use crate::telemetry;

/// Backoff delays for transient git retries (milliseconds).
pub(crate) const ASYNC_RETRY_DELAYS_MS: &[u64] = &[100, 500, 2000];

/// Stage and commit all changes in the given worktree, avoiding accidental commits of worktree symlinks.
///
/// Stages changes using `git add .` (so `.gitignore` is respected), then commits with the message
/// `swarm: iteration {iteration} changes`. The implementation retries staging/commit on transient
/// index.lock errors, performs a best-effort unstage of the special `.beads` entry and forcibly
/// removes any `.beads` index blob to ensure the worktree symlink is never committed.
///
/// # Parameters
///
/// - `wt_path`: path to the worktree where git commands will be executed.
/// - `iteration`: iteration number used to compose the commit message.
///
/// # Returns
///
/// `true` if a commit was created, `false` if there were no staged changes to commit.
///
/// # Examples
///
/// ```ignore
/// use std::path::Path;
///
/// # async fn example() -> anyhow::Result<()> {
/// let committed = git_commit_changes(Path::new("."), 1).await?;
/// println!("Committed changes: {}", committed);
/// # Ok(()) }
/// ```
pub async fn git_commit_changes(wt_path: &Path, iteration: u32) -> Result<bool> {
    // Stage changes (respects .gitignore) — retry for transient index.lock errors
    let add = retry_git_command_async(&["add", "."], wt_path, 3).await?;
    if !add.status.success() {
        let stderr = String::from_utf8_lossy(&add.stderr);
        anyhow::bail!("git add failed (iteration {iteration}): {stderr}");
    }

    // Best-effort: unstage .beads if it was accidentally staged via the worktree symlink.
    // WorktreeBridge::create() replaces .beads/ with a symlink to the main repo's .beads/
    // so that `bd` commands connect to the shared Dolt server. `git add .` in a worktree
    // would stage this symlink (as a mode-120000 blob) and simultaneously delete the
    // tracked .beads/backup/ paths from the index — falsely appearing as a code change.
    // `git restore --staged .beads` is a no-op when .beads is not staged.
    let _ = tokio::process::Command::new("git")
        .args(["restore", "--staged", ".beads"])
        .current_dir(wt_path)
        .output()
        .await;

    // Belt-and-suspenders: force-remove the .beads blob entry from the index.
    // `git restore --staged` restores tracked paths from HEAD, but when `git add .`
    // stages a directory→symlink type change it may not fully undo the mode-120000 entry.
    // `update-index --force-remove` removes only the exact path '.beads' (not '.beads/<files>'),
    // ensuring the symlink is never committed to the branch even if restore didn't fully work.
    // Exit code is 0 whether the path was present (removed) or absent (no-op), so a
    // non-zero exit always indicates a real git error — fail-closed to prevent a dirty commit.
    let force_remove =
        retry_git_command_async(&["update-index", "--force-remove", ".beads"], wt_path, 3).await?;
    if !force_remove.status.success() {
        let stderr = String::from_utf8_lossy(&force_remove.stderr);
        anyhow::bail!("git update-index --force-remove .beads failed: {stderr}");
    }

    // Check if there are staged changes
    let status = tokio::process::Command::new("git")
        .args(["diff", "--cached", "--quiet"])
        .current_dir(wt_path)
        .output()
        .await?;

    if status.status.success() {
        // Exit code 0 means no diff — nothing to commit
        return Ok(false);
    }

    // Commit — retry for transient index.lock errors
    let msg = format!("swarm: iteration {iteration} changes");
    let commit = retry_git_command_async(&["commit", "-m", &msg], wt_path, 3).await?;
    if !commit.status.success() {
        let stderr = String::from_utf8_lossy(&commit.stderr);
        anyhow::bail!("git commit failed: {stderr}");
    }

    Ok(true)
}

/// Async version of retry_git_command for tokio contexts.
pub(crate) async fn retry_git_command_async(
    args: &[&str],
    working_dir: &Path,
    max_retries: u32,
) -> Result<std::process::Output> {
    for attempt in 0..=max_retries {
        let output = tokio::process::Command::new("git")
            .args(args)
            .current_dir(working_dir)
            .output()
            .await
            .context("Failed to execute git command")?;

        if output.status.success() {
            return Ok(output);
        }

        let stderr = String::from_utf8_lossy(&output.stderr);
        let is_transient = stderr.contains("index.lock") || stderr.contains("Unable to create");

        if attempt < max_retries && is_transient {
            let delay = ASYNC_RETRY_DELAYS_MS
                .get(attempt as usize)
                .copied()
                .unwrap_or(2000);
            warn!(
                attempt = attempt + 1,
                max_retries,
                delay_ms = delay,
                "Transient git failure, retrying: {}",
                stderr.trim()
            );
            tokio::time::sleep(Duration::from_millis(delay)).await;
            continue;
        }

        return Ok(output);
    }

    unreachable!()
}

/// Count lines changed between two commits in the worktree.
pub(crate) fn count_diff_lines(wt_path: &Path, from: &str, to: &str) -> usize {
    let output = std::process::Command::new("git")
        .args(["diff", "--numstat", from, to])
        .current_dir(wt_path)
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            stdout.lines().fold(0, |acc, line| {
                let parts: Vec<&str> = line.split('\t').collect();
                if parts.len() >= 2 {
                    let added: usize = parts[0].parse().unwrap_or(0);
                    let removed: usize = parts[1].parse().unwrap_or(0);
                    acc + added + removed
                } else {
                    acc
                }
            })
        }
        _ => 0,
    }
}

/// Collect artifact records from the git diff between two commits.
///
/// Parses `git diff --numstat` to determine which files were added, modified,
/// or deleted. Files that existed before (`from`) and after (`to`) are
/// `Modified`; files only in `to` are `Created`; files only in `from` are
/// `Deleted`. The `size_delta` is approximated as `(added - removed)` lines
/// (a line-count proxy; byte-level deltas would require `--stat`).
pub(crate) fn collect_artifacts_from_diff(
    wt_path: &Path,
    from: &str,
    to: &str,
) -> Vec<telemetry::ArtifactRecord> {
    let output = std::process::Command::new("git")
        .args(["diff", "--numstat", from, to])
        .current_dir(wt_path)
        .output();

    let stdout = match output {
        Ok(ref out) if out.status.success() => String::from_utf8_lossy(&out.stdout).to_string(),
        _ => return Vec::new(),
    };

    stdout
        .lines()
        .filter_map(|line| {
            // numstat format: "<added>\t<removed>\t<path>"
            // Binary files show "-\t-\t<path>"
            let parts: Vec<&str> = line.splitn(3, '\t').collect();
            if parts.len() < 3 {
                return None;
            }
            let added: i64 = parts[0].parse().unwrap_or(0);
            let removed: i64 = parts[1].parse().unwrap_or(0);
            let path = parts[2].trim().to_string();

            let action = if added > 0 && removed == 0 {
                telemetry::ArtifactAction::Created
            } else if added == 0 && removed > 0 {
                telemetry::ArtifactAction::Deleted
            } else {
                telemetry::ArtifactAction::Modified
            };

            Some(telemetry::ArtifactRecord {
                path,
                action,
                line_range: None,
                size_delta: Some(added - removed),
            })
        })
        .collect()
}
