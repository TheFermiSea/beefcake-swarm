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

use crate::git_ops::is_operational_artifact_path;

/// Errors that can occur during acceptance gate evaluation.
#[derive(Debug, thiserror::Error)]
pub enum AcceptanceError {
    #[error("git command failed: {0}")]
    Git(String),
    #[error("git output parse error: {0}")]
    Parse(String),
}

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

    /// Minimum diff size in lines produced by the agent.
    ///
    /// Guards against trivial one-line/comment-only landings that technically
    /// pass verification but do not materially resolve the issue.
    /// Default: 5. Set to 0 to disable.
    pub min_diff_lines: usize,

    /// Fine-grained file scope: if non-empty, only these specific files may
    /// be modified. Prevents workers from touching code outside the task scope.
    /// Paths are relative to the worktree root.
    /// Default: empty (no file-level restriction — use `scope_to_crates` for coarser control).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub allowed_files: Vec<String>,

    /// Reject diffs that contain only whitespace/formatting changes.
    ///
    /// Disabled by default because `cargo fmt` runs automatically before commit,
    /// and its reformatting can make substantive code changes look "whitespace only"
    /// to the character-level comparison. The 5/5 verifier gates (fmt, clippy,
    /// check, test) are the authoritative quality signal.
    /// Default: false (disabled).
    #[serde(default)]
    pub reject_whitespace_only: bool,
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
            reject_whitespace_only: false,
        }
    }
}

/// Result of evaluating the acceptance policy.
#[derive(Debug)]
pub struct AcceptanceResult {
    pub accepted: bool,
    pub rejections: Vec<String>,
}

/// Lightweight task metadata used by the hard issue-resolution guard.
#[derive(Debug, Clone, Copy)]
pub struct TaskMetadata<'a> {
    pub issue_title: &'a str,
    pub issue_description: Option<&'a str>,
}

impl<'a> TaskMetadata<'a> {
    pub fn new(issue_title: &'a str, issue_description: Option<&'a str>) -> Self {
        Self {
            issue_title,
            issue_description,
        }
    }
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
    check_acceptance_with_task(policy, wt_path, initial_commit, cloud_passes, None)
}

/// Check the acceptance policy against the current worktree state with issue metadata.
pub fn check_acceptance_with_task(
    policy: &AcceptancePolicy,
    wt_path: &Path,
    initial_commit: Option<&str>,
    cloud_passes: usize,
    task: Option<TaskMetadata<'_>>,
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

    // Gate 2: Minimum meaningful diff size
    if policy.min_diff_lines > 0 {
        if let Some(commit) = initial_commit {
            match diff_line_count(wt_path, commit) {
                Ok(lines) => {
                    if lines < policy.min_diff_lines {
                        rejections.push(format!(
                            "Diff too small: {lines} lines (min: {})",
                            policy.min_diff_lines
                        ));
                    }
                }
                Err(e) => {
                    warn!("Failed to count diff lines: {e} — skipping min diff size check");
                }
            }
        }
    }

    // Gate 3: Reject whitespace-only diffs (disabled by default — cargo fmt
    // makes substantive changes look whitespace-only to character comparison)
    if policy.reject_whitespace_only {
        if let Some(commit) = initial_commit {
            match has_non_whitespace_changes(wt_path, commit) {
                Ok(true) => {}
                Ok(false) => {
                    rejections.push("Diff contains only whitespace/formatting changes".into());
                }
                Err(e) => {
                    warn!(
                        "Failed to inspect whitespace-only diff: {e} — skipping whitespace check"
                    );
                }
            }
        }
    }

    // Gate 4: Cloud validator passes
    if policy.min_cloud_passes > 0 && cloud_passes < policy.min_cloud_passes {
        rejections.push(format!(
            "Insufficient cloud validations: {cloud_passes} PASS (need: {})",
            policy.min_cloud_passes
        ));
    }

    // Gate 5: Test changes required
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

    // Gate 6: Scope restriction
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

    // Gate 7: File-level scope restriction
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

    // Gate 8: Issue-target alignment.
    if let (Some(commit), Some(task)) = (initial_commit, task) {
        let changed_files = match changed_files_since(wt_path, commit) {
            Ok(files) => files,
            Err(e) => {
                warn!("Failed to list changed files for issue guard: {e} — skipping target-file check");
                Vec::new()
            }
        };

        let explicit_targets = task
            .issue_description
            .map(extract_explicit_file_targets)
            .unwrap_or_default();
        if !explicit_targets.is_empty()
            && !changed_files.iter().any(|changed| {
                explicit_targets
                    .iter()
                    .any(|target| path_matches_target(changed, target))
            })
        {
            rejections.push(format!(
                "Changes do not touch the issue target files: {}",
                explicit_targets.join(", ")
            ));
        }

        let target_symbols = extract_target_symbols(task.issue_title);
        if explicit_targets.is_empty() && !target_symbols.is_empty() {
            match diff_mentions_any_symbol(wt_path, commit, &target_symbols) {
                Ok(true) => {}
                Ok(false) => rejections.push(format!(
                    "Diff does not mention the target symbol(s): {}",
                    target_symbols.join(", ")
                )),
                Err(e) => {
                    warn!("Failed to inspect target symbols in diff: {e} — skipping symbol check");
                }
            }
        }

        // Gate 9: Evidence-based acceptance via Validation Command
        if let Some(desc) = task.issue_description {
            if let Some(cmd) = extract_validation_command(desc) {
                info!("Found issue-specific validation command: {}", cmd);
                match run_validation_command(wt_path, &cmd) {
                    Ok(true) => info!("Issue-specific validation command passed"),
                    Ok(false) => rejections
                        .push(format!("Issue-specific validation command failed: {}", cmd)),
                    Err(e) => rejections.push(format!(
                        "Failed to execute issue-specific validation command: {} (error: {})",
                        cmd, e
                    )),
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
/// Uses `git diff --numstat` for reliable per-file counts and excludes
/// operational-artifact files that should not count as real code changes.
fn diff_line_count(wt_path: &Path, initial_commit: &str) -> Result<usize, AcceptanceError> {
    let output = std::process::Command::new("git")
        .args(["diff", "--numstat", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| AcceptanceError::Git(format!("Failed to run git diff --numstat: {e}")))?;

    if !output.status.success() {
        return Err(AcceptanceError::Git(format!(
            "git diff --numstat failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut total = 0usize;
    for line in stdout.lines() {
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() >= 3 && is_operational_artifact_path(parts[2]) {
            continue;
        }
        if parts.len() >= 2 {
            let added: usize = parts[0].parse().unwrap_or(0);
            let removed: usize = parts[1].parse().unwrap_or(0);
            total += added + removed;
        }
    }

    Ok(total)
}

/// Check if any test files were modified since `initial_commit`.
fn has_test_changes(wt_path: &Path, initial_commit: &str) -> Result<bool, AcceptanceError> {
    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| AcceptanceError::Git(format!("Failed to run git diff --name-only: {e}")))?;

    if !output.status.success() {
        return Err(AcceptanceError::Git(format!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
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
) -> Result<Vec<String>, AcceptanceError> {
    // Empty allowed list means no restriction
    if allowed_crates.is_empty() {
        return Ok(Vec::new());
    }

    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| AcceptanceError::Git(format!("Failed to run git diff --name-only: {e}")))?;

    if !output.status.success() {
        return Err(AcceptanceError::Git(format!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
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
) -> Result<Vec<String>, AcceptanceError> {
    if allowed_files.is_empty() {
        return Ok(Vec::new());
    }

    let changed_files = changed_files_since(wt_path, initial_commit)?;
    let out_of_scope = changed_files
        .into_iter()
        .filter(|file| !allowed_files.iter().any(|allowed| file == allowed))
        .collect();

    Ok(out_of_scope)
}

fn changed_files_since(
    wt_path: &Path,
    initial_commit: &str,
) -> Result<Vec<String>, AcceptanceError> {
    let output = std::process::Command::new("git")
        .args(["diff", "--name-only", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| AcceptanceError::Git(format!("Failed to run git diff --name-only: {e}")))?;

    if !output.status.success() {
        return Err(AcceptanceError::Git(format!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|path| !path.is_empty() && !is_operational_artifact_path(path))
        .map(ToOwned::to_owned)
        .collect())
}

fn has_non_whitespace_changes(
    wt_path: &Path,
    initial_commit: &str,
) -> Result<bool, AcceptanceError> {
    for path in changed_files_since(wt_path, initial_commit)? {
        let before = file_contents_at_revision(wt_path, initial_commit, &path)?;
        let after = file_contents_at_revision(wt_path, "HEAD", &path)?;

        if normalize_without_whitespace(before.as_deref())
            != normalize_without_whitespace(after.as_deref())
        {
            return Ok(true);
        }
    }

    Ok(false)
}

fn file_contents_at_revision(
    wt_path: &Path,
    revision: &str,
    path: &str,
) -> Result<Option<String>, AcceptanceError> {
    let spec = format!("{revision}:{path}");
    let output = std::process::Command::new("git")
        .args(["show", &spec])
        .current_dir(wt_path)
        .output()
        .map_err(|e| AcceptanceError::Git(format!("Failed to run git show {spec}: {e}")))?;

    if output.status.success() {
        return Ok(Some(String::from_utf8_lossy(&output.stdout).into_owned()));
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    if stderr.contains("exists on disk, but not in")
        || stderr.contains("does not exist in")
        || stderr.contains("unknown revision or path")
    {
        return Ok(None);
    }

    Err(AcceptanceError::Git(format!(
        "git show {spec} failed: {stderr}"
    )))
}

fn normalize_without_whitespace(contents: Option<&str>) -> String {
    contents
        .unwrap_or_default()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect()
}

fn extract_explicit_file_targets(description: &str) -> Vec<String> {
    description
        .lines()
        .filter_map(|line| {
            let (_, rest) = line.split_once("Location:")?;
            sanitize_location_token(rest.split_whitespace().next()?)
        })
        .collect()
}

/// Extract an explicit validation command from the issue description.
/// Looks for a line starting with `Validation Command: `
fn extract_validation_command(description: &str) -> Option<String> {
    for line in description.lines() {
        let line = line.trim();
        if let Some(cmd) = line.strip_prefix("Validation Command:") {
            let clean_cmd = cmd.trim().trim_matches('`').trim();
            if !clean_cmd.is_empty() {
                return Some(clean_cmd.to_string());
            }
        }
    }
    None
}

/// Run a validation command in the worktree.
/// Uses shlex to parse the command safely instead of passing through sh -c,
/// which would allow command injection via issue descriptions.
fn run_validation_command(wt_path: &Path, cmd: &str) -> anyhow::Result<bool> {
    use std::process::Command;
    let parts = shlex::split(cmd)
        .ok_or_else(|| anyhow::anyhow!("invalid quoting in validation command"))?;
    if parts.is_empty() {
        return Err(anyhow::anyhow!("empty validation command"));
    }
    let status = Command::new(&parts[0])
        .args(&parts[1..])
        .current_dir(wt_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()?;
    Ok(status.success())
}

fn sanitize_location_token(token: &str) -> Option<String> {
    let token = token
        .trim_matches(|c: char| matches!(c, '`' | '"' | '\'' | ',' | '.' | '(' | ')' | '[' | ']'));
    if token.is_empty() {
        return None;
    }

    let stripped = strip_line_suffix(token);
    if stripped.contains('/') || stripped.ends_with(".rs") || stripped.ends_with(".toml") {
        Some(stripped.to_string())
    } else {
        None
    }
}

fn strip_line_suffix(token: &str) -> &str {
    let mut segments = token.rsplitn(3, ':');
    let last = segments.next();
    let second = segments.next();
    match (last, second) {
        (Some(last), Some(_second)) if last.chars().all(|c| c.is_ascii_digit() || c == '-') => {
            &token[..token.len() - last.len() - 1]
        }
        _ => token,
    }
}

fn path_matches_target(changed: &str, target: &str) -> bool {
    changed == target || changed.ends_with(&format!("/{target}"))
}

fn extract_target_symbols(title: &str) -> Vec<String> {
    let Some((prefix, rest)) = title.split_once(':') else {
        return Vec::new();
    };

    let prefix = prefix.to_ascii_lowercase();
    if !(prefix.contains("function")
        || prefix.contains("method")
        || prefix.contains("struct")
        || prefix.contains("enum")
        || prefix.contains("type"))
    {
        return Vec::new();
    }

    rest.split(|c: char| c.is_whitespace() || matches!(c, ',' | '(' | ')' | '[' | ']'))
        .find(|token| !token.is_empty())
        .map(|token| vec![token.trim_matches('`').to_string()])
        .unwrap_or_default()
}

fn diff_mentions_any_symbol(
    wt_path: &Path,
    initial_commit: &str,
    target_symbols: &[String],
) -> Result<bool, AcceptanceError> {
    let output = std::process::Command::new("git")
        .args(["diff", "--unified=0", initial_commit, "HEAD"])
        .current_dir(wt_path)
        .output()
        .map_err(|e| AcceptanceError::Git(format!("Failed to run git diff --unified=0: {e}")))?;

    if !output.status.success() {
        return Err(AcceptanceError::Git(format!(
            "git diff --unified=0 failed: {}",
            String::from_utf8_lossy(&output.stderr)
        )));
    }

    let normalized_symbols: Vec<String> = target_symbols
        .iter()
        .map(|symbol| symbol.to_ascii_lowercase())
        .collect();
    let diff = String::from_utf8_lossy(&output.stdout);

    Ok(diff.lines().any(|line| {
        if !(line.starts_with('+') || line.starts_with('-'))
            || line.starts_with("+++")
            || line.starts_with("---")
        {
            return false;
        }
        let normalized_line = line.to_ascii_lowercase();
        normalized_symbols
            .iter()
            .any(|symbol| normalized_line.contains(symbol))
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init_git_repo(dir: &tempfile::TempDir) {
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
    }

    fn commit_all(dir: &tempfile::TempDir, message: &str) {
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", message])
            .current_dir(dir.path())
            .output()
            .unwrap();
    }

    fn head_commit(dir: &tempfile::TempDir) -> String {
        String::from_utf8(
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .current_dir(dir.path())
                .output()
                .unwrap()
                .stdout,
        )
        .unwrap()
        .trim()
        .to_string()
    }

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
            reject_whitespace_only: false,
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
    fn test_rejects_trivial_diff_for_targeted_issue() {
        let dir = tempfile::tempdir().unwrap();
        init_git_repo(&dir);

        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "fn old_name() {}\n").unwrap();
        commit_all(&dir, "init");
        let initial = head_commit(&dir);

        std::fs::write(dir.path().join("src/lib.rs"), "fn new_name() {}\n").unwrap();
        commit_all(&dir, "rename");

        let result = check_acceptance_with_task(
            &AcceptancePolicy {
                min_diff_lines: 5,
                ..AcceptancePolicy::default()
            },
            dir.path(),
            Some(&initial),
            0,
            Some(TaskMetadata::new(
                "Unused function: old_name",
                Some("Location: src/lib.rs:1"),
            )),
        );

        assert!(!result.accepted);
        assert!(result
            .rejections
            .iter()
            .any(|r| r.contains("Diff too small")));
    }

    #[test]
    fn test_rejects_whitespace_only_changes_for_targeted_issue() {
        let dir = tempfile::tempdir().unwrap();
        init_git_repo(&dir);

        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "fn process_issue_core() {\n    do_work();\n}\n",
        )
        .unwrap();
        commit_all(&dir, "init");
        let initial = head_commit(&dir);

        std::fs::write(
            dir.path().join("src/lib.rs"),
            "fn process_issue_core() { do_work(); }\n",
        )
        .unwrap();
        commit_all(&dir, "format only");

        // With reject_whitespace_only disabled (default), whitespace-only diffs are accepted.
        let result_lenient = check_acceptance_with_task(
            &AcceptancePolicy::default(),
            dir.path(),
            Some(&initial),
            0,
            Some(TaskMetadata::new(
                "Complex function: process_issue_core",
                Some("Location: src/lib.rs:1"),
            )),
        );
        assert!(
            !result_lenient
                .rejections
                .iter()
                .any(|r| r.contains("whitespace/formatting")),
            "Whitespace gate should be off by default"
        );

        // With reject_whitespace_only enabled, whitespace-only diffs are rejected.
        let result_strict = check_acceptance_with_task(
            &AcceptancePolicy {
                reject_whitespace_only: true,
                max_diff_lines: 0,
                min_cloud_passes: 0,
                require_test_changes: false,
                scope_to_crates: Vec::new(),
                min_diff_lines: 0,
                allowed_files: Vec::new(),
            },
            dir.path(),
            Some(&initial),
            0,
            Some(TaskMetadata::new(
                "Complex function: process_issue_core",
                Some("Location: src/lib.rs:1"),
            )),
        );

        assert!(!result_strict.accepted);
        assert!(result_strict
            .rejections
            .iter()
            .any(|r| r.contains("whitespace/formatting")));
    }

    #[test]
    fn test_rejects_changes_outside_issue_target_file() {
        let dir = tempfile::tempdir().unwrap();
        init_git_repo(&dir);

        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "fn get_event_bus() {}\n").unwrap();
        std::fs::write(dir.path().join("src/other.rs"), "fn helper() {}\n").unwrap();
        commit_all(&dir, "init");
        let initial = head_commit(&dir);

        std::fs::write(
            dir.path().join("src/other.rs"),
            "fn helper() {\n    let renamed = 1;\n    assert_eq!(renamed, 1);\n}\n",
        )
        .unwrap();
        commit_all(&dir, "wrong file");

        let result = check_acceptance_with_task(
            &AcceptancePolicy::default(),
            dir.path(),
            Some(&initial),
            0,
            Some(TaskMetadata::new(
                "Unused function: get_event_bus",
                Some("Location: src/lib.rs:1"),
            )),
        );

        assert!(!result.accepted);
        assert!(result
            .rejections
            .iter()
            .any(|r| r.contains("issue target files")));
    }

    #[test]
    fn test_accepts_meaningful_targeted_change() {
        let dir = tempfile::tempdir().unwrap();
        init_git_repo(&dir);

        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(
            dir.path().join("src/lib.rs"),
            "fn get_event_bus() {\n    let x = 1;\n    let y = 2;\n    let z = x + y;\n    assert_eq!(z, 3);\n}\n",
        )
        .unwrap();
        commit_all(&dir, "init");
        let initial = head_commit(&dir);

        std::fs::write(dir.path().join("src/lib.rs"), "fn helper() {}\n").unwrap();
        commit_all(&dir, "remove target");

        let result = check_acceptance_with_task(
            &AcceptancePolicy {
                min_diff_lines: 5,
                ..AcceptancePolicy::default()
            },
            dir.path(),
            Some(&initial),
            0,
            Some(TaskMetadata::new(
                "Unused function: get_event_bus",
                Some("Location: src/lib.rs:1"),
            )),
        );

        assert!(result.accepted, "rejections: {:?}", result.rejections);
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
    fn test_diff_line_count_excludes_beads_noise() {
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

        // Add a real code change (5 lines) + .beads noise (1000 lines)
        std::fs::write(dir.path().join("code.rs"), "a\nb\nc\nd\ne\n").unwrap();
        std::fs::create_dir_all(dir.path().join(".beads/backup")).unwrap();
        let beads_content = "noise\n".repeat(1000);
        std::fs::write(dir.path().join(".beads/backup/issues.jsonl"), beads_content).unwrap();
        std::process::Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::process::Command::new("git")
            .args(["commit", "-m", "code + beads noise"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        // .beads lines should be excluded: only the 5 code lines count
        let lines = diff_line_count(dir.path(), &initial).unwrap();
        assert_eq!(lines, 5, ".beads noise should be excluded from diff count");
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
            validator_feedback: vec![],
            change_contract: None,
            repo_map: None,
            failed_approach_summary: None,
            dependency_graph: None,
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
