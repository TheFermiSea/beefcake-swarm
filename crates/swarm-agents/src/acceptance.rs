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

    /// Minimum diff size in lines produced by the agent (excluding auto-fix).
    /// Only enforced when `try_auto_fix` actually ran and made the verifier
    /// pass — prevents accepting iterations where the agent wrote nothing
    /// meaningful but auto-fix resolved pre-existing warnings.
    /// Default: 5. Set to 0 to disable.
    pub min_diff_lines: usize,

    /// Fine-grained file scope: if non-empty, only these specific files may
    /// be modified. Prevents workers from touching code outside the task scope.
    /// Paths are relative to the worktree root.
    /// Default: empty (no file-level restriction — use `scope_to_crates` for coarser control).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_files: Vec<String>,
}

impl Default for AcceptancePolicy {
    fn default() -> Self {
        Self {
            max_diff_lines: 500,
            min_cloud_passes: 0,
            require_test_changes: false,
            scope_to_crates: Vec::new(),
            min_diff_lines: 5,
            allowed_files: Vec::new(),
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

    // Gate 5: File-level scope restriction
    if !policy.allowed_files.is_empty() {
        if let Some(commit) = initial_commit {
            match check_file_scope(wt_path, commit, &policy.allowed_files) {
                Ok(out_of_scope) => {
                    if !out_of_scope.is_empty() {
                        rejections.push(format!(
                            "Changes outside allowed files: {}",
                            out_of_scope.join(", ")
                        ));
                    }
                }
                Err(e) => {
                    warn!("Failed to check file scope: {e} — skipping file scope check");
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

/// Count the number of added + removed lines in the diff since `initial_commit`.
///
/// Uses `git diff --numstat` for reliable per-file counts.
fn diff_line_count(wt_path: &Path, initial_commit: &str) -> Result<usize, String> {
    let output = std::process::Command::new("git")
        .args(["diff", "--numstat", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| format!("Failed to run git diff --numstat: {e}"))?;

    if !output.status.success() {
        return Err(format!(
            "git diff --numstat failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut total = 0usize;
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 2 {
            let added: usize = parts[0].parse().unwrap_or(0);
            let removed: usize = parts[1].parse().unwrap_or(0);
            total += added + removed;
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

/// Check if all changed files are in the allowed files list.
///
/// Returns a list of files that are outside the allowed scope.
fn check_file_scope(
    wt_path: &Path,
    initial_commit: &str,
    allowed_files: &[String],
) -> Result<Vec<String>, String> {
    if allowed_files.is_empty() {
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
            !allowed_files
                .iter()
                .any(|allowed| *file == allowed.as_str())
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
        assert_eq!(policy.min_diff_lines, 5);
    }

    #[test]
    fn test_acceptance_all_disabled() {
        let policy = AcceptancePolicy {
            max_diff_lines: 0,
            min_cloud_passes: 0,
            require_test_changes: false,
            scope_to_crates: Vec::new(),
            min_diff_lines: 0,
            allowed_files: Vec::new(),
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

    #[test]
    fn test_file_scope_check() {
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

        // Modify allowed and disallowed files
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "// ok\n").unwrap();
        std::fs::write(dir.path().join("src/main.rs"), "// not allowed\n").unwrap();
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

        // Only src/lib.rs is allowed — src/main.rs should be flagged
        let out_of_scope = check_file_scope(dir.path(), &initial, &["src/lib.rs".into()]).unwrap();
        assert_eq!(out_of_scope, vec!["src/main.rs"]);

        // With both files allowed, nothing is out of scope
        let out_of_scope = check_file_scope(
            dir.path(),
            &initial,
            &["src/lib.rs".into(), "src/main.rs".into()],
        )
        .unwrap();
        assert!(out_of_scope.is_empty());

        // With empty allowed list, no restriction
        let out_of_scope = check_file_scope(dir.path(), &initial, &[]).unwrap();
        assert!(out_of_scope.is_empty());
    }

    #[test]
    fn test_scope_constraints_in_prompt() {
        use coordination::escalation::state::SwarmTier;
        use coordination::work_packet::types::WorkPacket;

        let packet = WorkPacket {
            bead_id: "test-123".into(),
            branch: "fix/scope".into(),
            checkpoint: "abc123".into(),
            objective: "Add scope constraints".into(),
            files_touched: vec!["src/acceptance.rs".into(), "src/orchestrator.rs".into()],
            key_symbols: vec![],
            file_contexts: vec![],
            verification_gates: vec![],
            failure_signals: vec![],
            constraints: vec![],
            iteration: 1,
            target_tier: SwarmTier::Worker,
            escalation_reason: None,
            error_history: vec![],
            previous_attempts: vec![],
            iteration_deltas: vec![],
            relevant_heuristics: vec![],
            relevant_playbooks: vec![],
            decisions: vec![],
            generated_at: chrono::Utc::now(),
            max_patch_loc: 150,
            delegation_chain: vec![],
            skill_hints: vec![],
            replay_hints: vec![],
        };

        let prompt = crate::orchestrator::format_task_prompt(&packet);
        assert!(
            prompt.contains("Scope Constraints"),
            "Prompt should include scope section"
        );
        assert!(
            prompt.contains("src/acceptance.rs"),
            "Prompt should list allowed files"
        );
        assert!(
            prompt.contains("Do NOT modify any other files"),
            "Prompt should warn about scope"
        );
    }
}
