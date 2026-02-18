//! Task-completeness acceptance policy.
//!
//! Configurable checks beyond `all_green` that the orchestrator evaluates
//! after the verifier passes. If acceptance fails, the loop continues
//! iterating instead of breaking.
//!
//! Default: max_diff_lines = 500, all others disabled.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tracing::{info, warn};

/// Acceptance policy configuration.
///
/// Each field is an optional gate. Gates that are `None` or set to their
/// default (permissive) values are skipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptancePolicy {
    /// Maximum diff size in lines. Patches exceeding this are rejected.
    /// Default: 500. Set to 0 to disable.
    pub max_diff_lines: usize,

    /// Minimum number of cloud validators that must PASS.
    /// Default: 0 (disabled — cloud validation is advisory).
    pub min_cloud_passes: usize,

    /// Require that at least one test file was modified.
    /// Useful for enforcing test-driven development.
    /// Default: false.
    pub require_test_changes: bool,

    /// Scope acceptance to specific crates. If non-empty, the diff must
    /// only touch files within these crate directories.
    /// Default: empty (no scope restriction).
    pub scope_to_crates: Vec<String>,
}

impl Default for AcceptancePolicy {
    fn default() -> Self {
        Self {
            max_diff_lines: 500,
            min_cloud_passes: 0,
            require_test_changes: false,
            scope_to_crates: Vec::new(),
        }
    }
}

/// Result of evaluating the acceptance policy.
#[derive(Debug)]
pub struct AcceptanceResult {
    pub accepted: bool,
    pub rejections: Vec<String>,
}

impl AcceptanceResult {
    fn pass() -> Self {
        Self {
            accepted: true,
            rejections: Vec::new(),
        }
    }

    fn with_rejections(rejections: Vec<String>) -> Self {
        Self {
            accepted: rejections.is_empty(),
            rejections,
        }
    }
}

/// Check the acceptance policy against the current worktree state.
///
/// `initial_commit` is the commit hash before any agent changes (for diff sizing).
/// `cloud_passes` is the number of cloud validators that returned PASS.
pub fn check_acceptance(
    policy: &AcceptancePolicy,
    wt_path: &Path,
    initial_commit: Option<&str>,
    cloud_passes: usize,
) -> AcceptanceResult {
    let mut rejections = Vec::new();

    // Gate 1: Diff size
    if policy.max_diff_lines > 0 {
        if let Some(commit) = initial_commit {
            match diff_line_count(wt_path, commit) {
                Ok(lines) => {
                    if lines > policy.max_diff_lines {
                        rejections.push(format!(
                            "Diff too large: {lines} lines (max: {})",
                            policy.max_diff_lines
                        ));
                    } else {
                        info!(
                            diff_lines = lines,
                            max = policy.max_diff_lines,
                            "Diff size OK"
                        );
                    }
                }
                Err(e) => {
                    warn!("Failed to count diff lines: {e} — skipping diff size check");
                }
            }
        }
    }

    // Gate 2: Cloud validator passes
    if policy.min_cloud_passes > 0 && cloud_passes < policy.min_cloud_passes {
        rejections.push(format!(
            "Insufficient cloud validations: {cloud_passes} PASS (need: {})",
            policy.min_cloud_passes
        ));
    }

    // Gate 3: Test changes required
    if policy.require_test_changes {
        if let Some(commit) = initial_commit {
            match has_test_changes(wt_path, commit) {
                Ok(true) => {
                    info!("Test file changes detected");
                }
                Ok(false) => {
                    rejections
                        .push("No test file changes detected (require_test_changes=true)".into());
                }
                Err(e) => {
                    warn!("Failed to check test changes: {e} — skipping test change check");
                }
            }
        }
    }

    // Gate 4: Scope restriction
    if !policy.scope_to_crates.is_empty() {
        if let Some(commit) = initial_commit {
            match check_scope(wt_path, commit, &policy.scope_to_crates) {
                Ok(out_of_scope) => {
                    if !out_of_scope.is_empty() {
                        rejections.push(format!(
                            "Changes outside allowed crates: {}",
                            out_of_scope.join(", ")
                        ));
                    }
                }
                Err(e) => {
                    warn!("Failed to check scope: {e} — skipping scope check");
                }
            }
        }
    }

    if rejections.is_empty() {
        AcceptanceResult::pass()
    } else {
        AcceptanceResult::with_rejections(rejections)
    }
}

/// Count the number of lines in the diff since `initial_commit`.
fn diff_line_count(wt_path: &Path, initial_commit: &str) -> Result<usize, String> {
    let output = std::process::Command::new("git")
        .args(["diff", "--stat", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| format!("Failed to run git diff --stat: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "git diff --stat failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Parse the summary line: " N files changed, X insertions(+), Y deletions(-)"
    let stdout = String::from_utf8_lossy(&output.stdout);
    let last_line = stdout.lines().last().unwrap_or("");

    let mut total = 0usize;
    for word in last_line.split_whitespace() {
        // The number before "insertions" or "deletions"
        if let Ok(n) = word.parse::<usize>() {
            total = n; // Will be overwritten — we want the last parseable numbers
        }
    }

    // More reliable: use --numstat and sum
    let numstat = std::process::Command::new("git")
        .args(["diff", "--numstat", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| format!("Failed to run git diff --numstat: {e}"))?;

    if numstat.status.success() {
        let numstat_out = String::from_utf8_lossy(&numstat.stdout);
        total = 0;
        for line in numstat_out.lines() {
            let parts: Vec<&str> = line.split('\t').collect();
            if parts.len() >= 2 {
                let added: usize = parts[0].parse().unwrap_or(0);
                let removed: usize = parts[1].parse().unwrap_or(0);
                total += added + removed;
            }
        }
    }

    Ok(total)
}

/// Check if any test files were modified since `initial_commit`.
fn has_test_changes(wt_path: &Path, initial_commit: &str) -> Result<bool, String> {
    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| format!("Failed to run git diff --name-only: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let has_tests = stdout.lines().any(|file| {
        file.contains("/tests/")
            || file.starts_with("tests/")
            || file.contains("/test_")
            || file.contains("_test.rs")
            || file.ends_with("tests.rs")
    });

    Ok(has_tests)
}

/// Check if all changed files are within the allowed crate directories.
///
/// Returns a list of files that are outside the allowed scope.
fn check_scope(
    wt_path: &Path,
    initial_commit: &str,
    allowed_crates: &[String],
) -> Result<Vec<String>, String> {
    // Empty allowed list means no restriction
    if allowed_crates.is_empty() {
        return Ok(Vec::new());
    }

    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| format!("Failed to run git diff --name-only: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let out_of_scope: Vec<String> = stdout
        .lines()
        .filter(|file| {
            !allowed_crates
                .iter()
                .any(|crate_dir| file.starts_with(crate_dir))
        })
        .map(String::from)
        .collect();

    Ok(out_of_scope)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_policy() {
        let policy = AcceptancePolicy::default();
        assert_eq!(policy.max_diff_lines, 500);
        assert_eq!(policy.min_cloud_passes, 0);
        assert!(!policy.require_test_changes);
        assert!(policy.scope_to_crates.is_empty());
    }

    #[test]
    fn test_acceptance_all_disabled() {
        let policy = AcceptancePolicy {
            max_diff_lines: 0,
            min_cloud_passes: 0,
            require_test_changes: false,
            scope_to_crates: Vec::new(),
        };

        // With all gates disabled, should always pass
        let result = check_acceptance(&policy, Path::new("/tmp"), None, 0);
        assert!(result.accepted);
        assert!(result.rejections.is_empty());
    }

    #[test]
    fn test_cloud_validation_gate() {
        let policy = AcceptancePolicy {
            min_cloud_passes: 2,
            ..AcceptancePolicy::default()
        };

        // Not enough cloud passes
        let result = check_acceptance(&policy, Path::new("/tmp"), None, 1);
        assert!(!result.accepted);
        assert_eq!(result.rejections.len(), 1);
        assert!(result.rejections[0].contains("Insufficient cloud validations"));

        // Enough cloud passes
        let result = check_acceptance(&policy, Path::new("/tmp"), None, 2);
        assert!(result.accepted);
    }

    #[test]
    fn test_diff_line_count_in_git_repo() {
        let dir = tempfile::tempdir().unwrap();

        // Initialize git repo
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // Initial commit
        std::fs::write(dir.path().join("README.md"), "# test\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let initial = String::from_utf8(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir.path())
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string();

        // Add a file with 10 lines
        let content = (1..=10)
            .map(|i| format!("line {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.path().join("code.rs"), content).unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "add code"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let lines = diff_line_count(dir.path(), &initial).unwrap();
        assert_eq!(lines, 10);
    }

    #[test]
    fn test_has_test_changes_detection() {
        let dir = tempfile::tempdir().unwrap();

        // Initialize git repo
        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        std::fs::write(dir.path().join("README.md"), "# test\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let initial = String::from_utf8(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir.path())
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string();

        // Add a non-test file — should return false
        std::fs::write(dir.path().join("lib.rs"), "fn main() {}\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "add lib"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        assert!(!has_test_changes(dir.path(), &initial).unwrap());

        // Now add a test file
        std::fs::create_dir_all(dir.path().join("tests")).unwrap();
        std::fs::write(
            dir.path().join("tests/integration.rs"),
            "#[test] fn t() {}\n",
        )
        .unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "add test"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        assert!(has_test_changes(dir.path(), &initial).unwrap());
    }

    #[test]
    fn test_scope_check() {
        let dir = tempfile::tempdir().unwrap();

        std::process::Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        std::fs::write(dir.path().join("README.md"), "# test\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "init"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let initial = String::from_utf8(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir.path())
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string();

        // Add files in allowed and disallowed paths
        std::fs::create_dir_all(dir.path().join("crates/swarm-agents/src")).unwrap();
        std::fs::write(dir.path().join("crates/swarm-agents/src/lib.rs"), "// ok\n").unwrap();
        std::fs::write(dir.path().join("Cargo.toml"), "# root change\n").unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "changes"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let out_of_scope =
            check_scope(dir.path(), &initial, &["crates/swarm-agents/".into()]).unwrap();
        assert_eq!(out_of_scope, vec!["Cargo.toml"]);

        // With no scope restriction, nothing is out of scope
        let out_of_scope = check_scope(dir.path(), &initial, &[]).unwrap();
        assert!(out_of_scope.is_empty());
    }
}
