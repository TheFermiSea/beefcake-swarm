#![allow(dead_code)]
//! Git operations for the swarm orchestrator.

use std::path::Path;
use std::time::Duration;

use anyhow::{Context as _, Result};
use tracing::warn;

use crate::telemetry;

/// Backoff delays for transient git retries (milliseconds).
pub(crate) const ASYNC_RETRY_DELAYS_MS: &[u64] = &[100, 500, 2000];

/// Return true when the path is swarm/beads bookkeeping noise rather than code.
pub(crate) fn is_operational_artifact_path(path: &str) -> bool {
    let path = path.trim();
    path == ".beads" || path.starts_with(".beads/") || path.starts_with(".swarm-")
}

fn parse_status_path(line: &str) -> Option<&str> {
    let line = line.trim_end();
    if line.len() < 4 {
        return None;
    }

    let payload = line[3..].trim();
    if payload.is_empty() {
        return None;
    }

    payload
        .rsplit_once(" -> ")
        .map(|(_, new_path)| new_path.trim())
        .or(Some(payload))
}

/// Remove operational-artifact lines from `git status --short` output.
pub(crate) fn filter_meaningful_status(output: &str) -> String {
    output
        .lines()
        .filter(|line| {
            parse_status_path(line)
                .map(|path| !is_operational_artifact_path(path))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Remove operational-artifact lines from `git diff --stat` / `--name-only`.
pub(crate) fn filter_meaningful_diff_output(output: &str, name_only: bool) -> String {
    output
        .lines()
        .filter(|line| {
            let path = if name_only {
                Some(line.trim())
            } else {
                line.split_once('|').map(|(path, _)| path.trim())
            };

            path.map(|path| !path.is_empty() && !is_operational_artifact_path(path))
                .unwrap_or(false)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Check whether the worktree has any meaningful (non-artifact) pending changes.
pub(crate) async fn git_has_meaningful_changes(wt_path: &Path) -> bool {
    tokio::process::Command::new("git")
        .args(["status", "--short"])
        .current_dir(wt_path)
        .output()
        .await
        .map(|output| {
            output.status.success()
                && !filter_meaningful_status(&String::from_utf8_lossy(&output.stdout))
                    .trim()
                    .is_empty()
        })
        .unwrap_or(false)
}

async fn unstage_operational_artifacts(wt_path: &Path) -> Result<()> {
    let staged = tokio::process::Command::new("git")
        .args(["diff", "--cached", "--name-only", "-z"])
        .current_dir(wt_path)
        .output()
        .await
        .context("Failed to inspect staged paths")?;

    if !staged.status.success() {
        let stderr = String::from_utf8_lossy(&staged.stderr);
        anyhow::bail!("git diff --cached --name-only failed: {stderr}");
    }

    for raw_path in staged.stdout.split(|byte| *byte == b'\0') {
        if raw_path.is_empty() {
            continue;
        }

        let path = String::from_utf8_lossy(raw_path).trim().to_string();
        if !is_operational_artifact_path(&path) {
            continue;
        }

        if path == ".beads" {
            let _ = tokio::process::Command::new("git")
                .args(["restore", "--staged", ".beads"])
                .current_dir(wt_path)
                .output()
                .await;

            let force_remove =
                retry_git_command_async(&["update-index", "--force-remove", ".beads"], wt_path, 3)
                    .await?;
            if !force_remove.status.success() {
                let stderr = String::from_utf8_lossy(&force_remove.stderr);
                anyhow::bail!("git update-index --force-remove .beads failed: {stderr}");
            }
            continue;
        }

        let restore = tokio::process::Command::new("git")
            .args(["restore", "--staged", "--", &path])
            .current_dir(wt_path)
            .output()
            .await
            .with_context(|| format!("Failed to unstage operational artifact: {path}"))?;

        if !restore.status.success() {
            warn!(
                path = %path,
                stderr = %String::from_utf8_lossy(&restore.stderr).trim(),
                "Failed to unstage operational artifact"
            );
        }
    }

    Ok(())
}

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

    // Drop bookkeeping artifacts (shared .beads symlink, .swarm-* files) from the
    // index before we decide whether there's anything worth committing.
    unstage_operational_artifacts(wt_path).await?;

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

/// Get a compact summary of uncommitted changes for cross-worker context.
///
/// Returns a markdown-formatted list of changed files with +/- line counts.
/// Used by the ClawTeam-style context sharing: workers see what previous
/// iterations changed so they don't undo or duplicate work.
///
/// Returns empty string if no changes or git fails.
pub(crate) fn diff_stat_summary(wt_path: &Path) -> String {
    let output = std::process::Command::new("git")
        .args(["diff", "--stat", "--stat-width=60", "HEAD"])
        .current_dir(wt_path)
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            let trimmed = stdout.trim();
            if trimmed.is_empty() {
                return String::new();
            }
            // Truncate to ~10 most-changed files to stay within prompt budget
            let lines: Vec<&str> = trimmed.lines().collect();
            if lines.len() <= 11 {
                trimmed.to_string()
            } else {
                // Keep first 10 file lines + the summary line (last line)
                let mut result: Vec<&str> = lines[..10].to_vec();
                result.push("...");
                if let Some(last) = lines.last() {
                    result.push(last);
                }
                result.join("\n")
            }
        }
        _ => String::new(),
    }
}

/// Get the diff content between two commits in the worktree.
///
/// Used by the diff-based evolution mode (ASI-Evolve pattern): on retry
/// iterations the previous worker's diff is injected into the task prompt
/// so the next worker refines rather than rewrites from scratch.
///
/// Returns `None` if there are no changes, either ref is missing, or git fails.
pub(crate) fn diff_between(wt_path: &Path, from: &str, to: &str) -> Option<String> {
    let output = std::process::Command::new("git")
        .args(["diff", from, to])
        .current_dir(wt_path)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
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

#[cfg(test)]
mod tests {
    use super::{filter_meaningful_diff_output, filter_meaningful_status};

    #[test]
    fn operational_artifact_detection_covers_beads_and_swarm_files() {
        assert!(super::is_operational_artifact_path(".beads"));
        assert!(super::is_operational_artifact_path(
            ".beads/backup/issues.jsonl"
        ));
        assert!(super::is_operational_artifact_path(".swarm-metrics.json"));
        assert!(super::is_operational_artifact_path(
            ".swarm-artifacts/session-1/report.json"
        ));
        assert!(!super::is_operational_artifact_path(
            "crates/swarm-agents/src/orchestrator.rs"
        ));
    }

    #[test]
    fn status_filter_drops_operational_artifacts() {
        let output = "
?? .beads
?? .swarm-metrics.json
 M crates/swarm-agents/src/orchestrator.rs";
        assert_eq!(
            filter_meaningful_status(output),
            " M crates/swarm-agents/src/orchestrator.rs"
        );
    }

    #[test]
    fn diff_filter_drops_operational_artifacts() {
        let output = "\
 .beads | 1 -
 .swarm-metrics.json | 12 ++++++++++++
 crates/swarm-agents/src/orchestrator.rs | 2 +-
 3 files changed, 13 insertions(+), 2 deletions(-)";

        assert_eq!(
            filter_meaningful_diff_output(output, false),
            " crates/swarm-agents/src/orchestrator.rs | 2 +-"
        );
    }

    #[test]
    fn name_only_diff_filter_drops_operational_artifacts() {
        let output = "\
.beads
.swarm-metrics.json
src/lib.rs";

        assert_eq!(filter_meaningful_diff_output(output, true), "src/lib.rs");
    }
}
