//! File targeting and package detection for the swarm orchestrator.

use std::path::Path;
use tracing::{debug, warn};

/// Build a grep `--include` pattern from source extensions.
///
/// For `[".py", ".pyi"]` → `"*.{py,pyi}"`.
/// For `[".rs"]` → `"*.rs"`.
/// For empty (no profile) → `"*.rs"` (backward compatible default).
fn include_pattern_from_extensions(extensions: &[String]) -> String {
    if extensions.is_empty() {
        return "*.rs".to_string();
    }
    let exts: Vec<&str> = extensions
        .iter()
        .map(|e| e.trim_start_matches('.'))
        .collect();
    if exts.len() == 1 {
        format!("*.{}", exts[0])
    } else {
        format!("*.{{{}}}", exts.join(","))
    }
}

/// Search the worktree for source files containing identifiers from the objective.
///
/// Returns `Some(files)` if grep finds matches, `None` otherwise.
/// Uses `source_extensions` from the language profile to filter by file type.
/// Defaults to `*.rs` when no extensions are provided (backward compatible).
/// Extract CamelCase and snake_case identifier tokens from an objective string.
///
/// CamelCase tokens (likely struct/type/trait names) take priority; snake_case
/// tokens (function/variable names common in Rust) are appended after.
fn extract_identifier_patterns(objective: &str) -> Vec<&str> {
    let camel = objective
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| w.len() >= 4 && w.chars().next().is_some_and(|c| c.is_uppercase()));
    let snake = objective
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|w| {
            w.contains('_')
                && w.len() >= 5
                && w.chars()
                    .all(|c| c.is_lowercase() || c == '_' || c.is_ascii_digit())
        });

    let mut patterns: Vec<&str> = camel.take(3).collect();
    for tok in snake.take(3) {
        if !patterns.contains(&tok) {
            patterns.push(tok);
        }
    }
    patterns
}

/// Score a candidate file by how many patterns it contains, with path boosts.
fn score_candidate(patterns: &[&str], path: &str, wt_root: &Path) -> usize {
    let content = std::fs::read_to_string(wt_root.join(path)).unwrap_or_default();
    let mut score = patterns.iter().filter(|p| content.contains(*p)).count();
    if path.contains("/tools/") {
        score += 2;
    } else if path.starts_with("patches/") {
        score = score.saturating_sub(2);
    } else if path.contains("/tests/")
        || path.contains("test_")
        || path.ends_with("_test.rs")
        || path.ends_with("_tests.rs")
    {
        score = score.saturating_sub(1);
    }
    score
}

pub(crate) fn find_target_files_by_grep(
    wt_root: &Path,
    objective: &str,
    source_extensions: &[String],
) -> Option<Vec<String>> {
    let include = include_pattern_from_extensions(source_extensions);
    let patterns = extract_identifier_patterns(objective);

    debug!(
        wt_root = %wt_root.display(),
        ?patterns,
        "find_target_files_by_grep: extracted patterns"
    );

    if patterns.is_empty() {
        debug!("find_target_files_by_grep: no searchable patterns found");
        return None;
    }

    let mut all_files: Vec<String> = Vec::new();
    let include_arg = format!("--include={include}");
    for pattern in patterns.iter().take(6) {
        match std::process::Command::new("grep")
            .args(["-rl", &include_arg, pattern])
            .current_dir(wt_root)
            .output()
        {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                debug!(
                    pattern,
                    success = output.status.success(),
                    exit_code = output.status.code(),
                    stdout_lines = stdout.lines().count(),
                    stderr = %stderr,
                    "find_target_files_by_grep: grep result"
                );
                if output.status.success() {
                    for line in stdout.lines().take(10) {
                        let path = line.trim().to_string();
                        if !all_files.contains(&path) {
                            all_files.push(path);
                        }
                    }
                }
            }
            Err(e) => {
                warn!(pattern, error = %e, "find_target_files_by_grep: grep command failed");
            }
        }
    }

    debug!(
        found = all_files.len(),
        files = ?all_files,
        "find_target_files_by_grep: grep results"
    );

    if all_files.is_empty() {
        return None;
    }

    // Score files by pattern matches and path-based boosts/penalties.
    let mut scored: Vec<(usize, String)> = all_files
        .into_iter()
        .map(|f| (score_candidate(&patterns, &f, wt_root), f))
        .collect();
    scored.sort_by(|a, b| b.0.cmp(&a.0));

    let result: Vec<String> = scored.into_iter().map(|(_, f)| f).take(3).collect();
    if result.is_empty() {
        None
    } else {
        Some(result)
    }
}

/// Detect which Cargo packages have been modified in the worktree.
///
/// Combines committed changes (git diff main..HEAD) and working-tree
/// changes (git status --porcelain) to produce a deduplicated list of
/// package names. Falls back to an empty Vec (= full workspace) on any error.
///
/// For non-Rust targets (when `is_rust` is false), returns empty immediately
/// since Cargo package scoping doesn't apply.
pub(crate) fn detect_changed_packages(wt_path: &Path, is_rust: bool) -> Vec<String> {
    if !is_rust {
        tracing::debug!("detect_changed_packages: non-Rust target, targeting full project");
        return Vec::new();
    }
    let mut changed_files: std::collections::HashSet<std::path::PathBuf> = Default::default();

    // Committed changes since branching from main
    if let Ok(out) = std::process::Command::new("git")
        .args(["diff", "--name-only", "main"])
        .current_dir(wt_path)
        .output()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if !line.trim().is_empty() {
                changed_files.insert(wt_path.join(line.trim()));
            }
        }
    }

    // Uncommitted working-tree changes (staged + unstaged)
    if let Ok(out) = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(wt_path)
        .output()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            // porcelain format: "XY filename" — filename starts at column 3
            if line.len() > 3 {
                let path = line[3..].trim();
                if !path.is_empty() {
                    changed_files.insert(wt_path.join(path));
                }
            }
        }
    }

    let mut packages: std::collections::HashSet<String> = Default::default();
    for file_path in &changed_files {
        if let Some(pkg) = find_package_name(file_path) {
            packages.insert(pkg);
        }
    }

    let result: Vec<String> = packages.into_iter().collect();
    if result.is_empty() {
        tracing::debug!("detect_changed_packages: no changes detected, targeting full workspace");
    } else {
        tracing::debug!(packages = ?result, "detect_changed_packages: scoping verifier to changed packages");
    }
    result
}

/// Walk up from `file_path` to find the nearest `Cargo.toml` and return the package `name`.
fn find_package_name(file_path: &Path) -> Option<String> {
    let mut dir = file_path.parent()?;
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                let mut in_package = false;
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed == "[package]" {
                        in_package = true;
                    } else if trimmed.starts_with('[') {
                        in_package = false;
                    } else if in_package && trimmed.starts_with("name") {
                        if let Some(name) = trimmed.split('"').nth(1) {
                            return Some(name.to_string());
                        }
                    }
                }
            }
        }
        match dir.parent() {
            Some(parent) if parent != dir => dir = parent,
            _ => break,
        }
    }
    None
}
