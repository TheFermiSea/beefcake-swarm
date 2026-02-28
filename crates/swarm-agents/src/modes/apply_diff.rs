//! NS-4.1–4.5: Agentic Mode unified-diff editing toolchain.
//!
//! Implements a Rig `Tool` that accepts unified-diff patches and applies them
//! to files within a sandboxed workspace directory.
//!
//! ## Why unified diffs instead of full-file rewrites?
//!
//! - Minimal blast radius: agents only touch the lines they intend to change.
//! - Self-documenting: the diff shows exactly what changed and why.
//! - Feedback loop: apply failures carry precise error messages the agent can
//!   self-correct from (hunk mismatch at line N, context mismatch, etc.).
//! - Works with large files: agents never need to reproduce the whole file.
//!
//! ## Security (NS-4.2)
//!
//! All paths are canonicalized and checked against `working_dir` before any
//! file I/O. Path traversal (e.g. `../../etc/passwd`), symlink escapes, and
//! writes to files outside the workspace are rejected with a typed error.
//!
//! ## Diff format
//!
//! Standard unified diff (output of `git diff` or `diff -u`):
//!
//! ```text
//! --- a/src/main.rs
//! +++ b/src/main.rs
//! @@ -10,6 +10,7 @@
//!  fn main() {
//! -    println!("hello");
//! +    println!("hello, world");
//!  }
//! ```
//!
//! Multiple hunks per file are supported.  Multiple files per patch are NOT
//! supported in a single call — call the tool once per file.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{debug, warn};

// ── Error ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum DiffError {
    #[error("path traversal rejected: {0}")]
    PathTraversal(String),
    #[error("file not found: {0}")]
    FileNotFound(String),
    #[error("patch parse error: {0}")]
    ParseError(String),
    #[error("hunk apply error at line {line}: {message}")]
    HunkApplyError { line: usize, message: String },
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

// ── Tool args / result ────────────────────────────────────────────────────────

/// Arguments for the `apply_diff` Rig tool.
#[derive(Debug, Deserialize)]
pub struct ApplyDiffArgs {
    /// Workspace-relative path to the file to patch (e.g. `src/main.rs`).
    pub path: String,
    /// Full unified diff in standard format (output of `git diff` or `diff -u`).
    pub diff: String,
}

/// Success result returned to the agent.
#[derive(Debug, Serialize, Deserialize)]
pub struct ApplyDiffResult {
    pub success: bool,
    pub message: String,
    pub hunks_applied: usize,
    pub lines_changed: i64,
}

// ── Tool ──────────────────────────────────────────────────────────────────────

/// Rig tool that applies a unified diff to a workspace file.
pub struct ApplyDiffTool {
    working_dir: PathBuf,
}

impl ApplyDiffTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }

    /// Validate that `rel_path` resolves inside `working_dir`.
    ///
    /// Returns the canonicalized absolute path if valid.
    pub fn sandbox_path(&self, rel_path: &str) -> Result<PathBuf, DiffError> {
        // Reject obvious traversal attempts before canonicalize.
        if rel_path.contains("..") {
            return Err(DiffError::PathTraversal(format!(
                "path '{}' contains '..' which is not allowed",
                rel_path
            )));
        }

        let candidate = self.working_dir.join(rel_path);

        // Canonicalize the working_dir.
        let canon_base = self.working_dir.canonicalize().map_err(DiffError::Io)?;

        // For the candidate, if the file doesn't exist yet we can only check
        // that its parent is within the workspace.
        let resolved = if candidate.exists() {
            candidate.canonicalize()?
        } else {
            let parent = candidate.parent().unwrap_or(&candidate);
            let canon_parent = parent.canonicalize().map_err(|e| DiffError::Io(e))?;
            if !canon_parent.starts_with(&canon_base) {
                return Err(DiffError::PathTraversal(format!(
                    "path '{}' escapes workspace '{}'",
                    rel_path,
                    canon_base.display()
                )));
            }
            candidate
        };

        if !resolved.starts_with(&canon_base) {
            return Err(DiffError::PathTraversal(format!(
                "path '{}' resolves outside workspace '{}'",
                rel_path,
                canon_base.display()
            )));
        }

        Ok(resolved)
    }
}

impl Tool for ApplyDiffTool {
    const NAME: &'static str = "apply_diff";

    type Error = DiffError;
    type Args = ApplyDiffArgs;
    type Output = ApplyDiffResult;

    async fn definition(&self, _: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Apply a unified diff (patch) to a file in the workspace. \
                Use this instead of rewriting entire files — only describe what changed. \
                The diff must be in standard unified format (git diff / diff -u output). \
                Returns success/failure with diagnostic details for self-correction."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative path to the file to patch (e.g. src/main.rs)"
                    },
                    "diff": {
                        "type": "string",
                        "description": "Full unified diff in standard format (output of git diff or diff -u)"
                    }
                },
                "required": ["path", "diff"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        debug!(path = %args.path, "applying diff");

        let abs_path = self.sandbox_path(&args.path)?;

        // Read current file content (may not exist for new files).
        let original = if abs_path.exists() {
            std::fs::read_to_string(&abs_path).map_err(DiffError::Io)?
        } else {
            String::new()
        };

        match apply_unified_diff(&original, &args.diff) {
            Ok(PatchResult {
                patched,
                hunks_applied,
                lines_changed,
            }) => {
                // Ensure parent directory exists for new files.
                if let Some(parent) = abs_path.parent() {
                    std::fs::create_dir_all(parent).map_err(DiffError::Io)?;
                }
                std::fs::write(&abs_path, &patched).map_err(DiffError::Io)?;
                Ok(ApplyDiffResult {
                    success: true,
                    message: format!("Applied {hunks_applied} hunk(s) to {}", args.path),
                    hunks_applied,
                    lines_changed,
                })
            }
            Err(e) => {
                warn!(path = %args.path, error = %e, "diff apply failed");
                Ok(ApplyDiffResult {
                    success: false,
                    message: format!("Diff apply failed: {e}"),
                    hunks_applied: 0,
                    lines_changed: 0,
                })
            }
        }
    }
}

// ── Unified diff parser / applier ─────────────────────────────────────────────

#[derive(Debug)]
pub(crate) struct PatchResult {
    patched: String,
    hunks_applied: usize,
    lines_changed: i64,
}

/// Parse and apply a unified diff to `original` text.
///
/// Supports standard `@@ -L,N +L,N @@` hunks.
/// Does NOT require the `--- a/` / `+++ b/` header lines.
pub(crate) fn apply_unified_diff(original: &str, diff: &str) -> Result<PatchResult, DiffError> {
    let hunks = parse_hunks(diff)?;

    if hunks.is_empty() {
        return Ok(PatchResult {
            patched: original.to_string(),
            hunks_applied: 0,
            lines_changed: 0,
        });
    }

    let mut result: Vec<String> = original.lines().map(|l| l.to_string()).collect();
    let mut offset: i64 = 0;
    let mut total_changed: i64 = 0;
    let mut hunks_applied = 0;

    for hunk in &hunks {
        let adjusted_start = (hunk.orig_start as i64 + offset - 1).max(0) as usize;

        let expected_orig: Vec<&str> = hunk
            .lines
            .iter()
            .filter(|(op, _)| *op == ' ' || *op == '-')
            .map(|(_, c)| c.as_str())
            .collect();

        for (i, &expected) in expected_orig.iter().enumerate() {
            let file_idx = adjusted_start + i;
            if file_idx >= result.len() {
                return Err(DiffError::HunkApplyError {
                    line: file_idx + 1,
                    message: format!(
                        "file has {} lines but hunk expects line {}",
                        result.len(),
                        file_idx + 1
                    ),
                });
            }
            if result[file_idx] != expected {
                return Err(DiffError::HunkApplyError {
                    line: file_idx + 1,
                    message: format!(
                        "context mismatch: expected {:?}, found {:?}",
                        expected, result[file_idx]
                    ),
                });
            }
        }

        let replacement: Vec<String> = hunk
            .lines
            .iter()
            .filter(|(op, _)| *op == ' ' || *op == '+')
            .map(|(_, c)| c.clone())
            .collect();

        let orig_span = expected_orig.len();
        let end = adjusted_start + orig_span;
        result.splice(adjusted_start..end, replacement.iter().cloned());

        let delta = replacement.len() as i64 - orig_span as i64;
        offset += delta;
        total_changed += delta.abs();
        hunks_applied += 1;
    }

    let mut patched = result.join("\n");
    if original.ends_with('\n') && !patched.ends_with('\n') {
        patched.push('\n');
    }

    Ok(PatchResult {
        patched,
        hunks_applied,
        lines_changed: total_changed,
    })
}

#[derive(Debug)]
struct Hunk {
    orig_start: usize,
    _orig_count: usize,
    _new_count: usize,
    /// Lines with their diff prefix: ' ' (context), '-' (remove), '+' (add).
    lines: Vec<(char, String)>,
}

fn parse_hunks(diff: &str) -> Result<Vec<Hunk>, DiffError> {
    let mut hunks: Vec<Hunk> = Vec::new();
    let mut current: Option<Hunk> = None;

    for (lineno, line) in diff.lines().enumerate() {
        if line.starts_with("@@") {
            // Flush previous hunk.
            if let Some(h) = current.take() {
                hunks.push(h);
            }
            let (orig_start, orig_count, new_count) = parse_hunk_header(line)
                .map_err(|e| DiffError::ParseError(format!("line {}: {}", lineno + 1, e)))?;
            current = Some(Hunk {
                orig_start,
                _orig_count: orig_count,
                _new_count: new_count,
                lines: Vec::new(),
            });
        } else if line.starts_with("---") || line.starts_with("+++") {
            // Skip file header lines.
            continue;
        } else if let Some(ref mut hunk) = current {
            // Diff content lines.
            if let Some(stripped) = line.strip_prefix('-') {
                hunk.lines.push(('-', stripped.to_string()));
            } else if let Some(stripped) = line.strip_prefix('+') {
                hunk.lines.push(('+', stripped.to_string()));
            } else {
                // Context line (starts with ' ' or is empty).
                let stripped = line.strip_prefix(' ').unwrap_or(line);
                hunk.lines.push((' ', stripped.to_string()));
            }
        }
        // Lines before the first @@ header (index, diff stats) are ignored.
    }

    if let Some(h) = current {
        hunks.push(h);
    }

    Ok(hunks)
}

fn parse_hunk_header(header: &str) -> Result<(usize, usize, usize), String> {
    // Format: @@ -L[,N] +L[,N] @@[ optional label]
    let inner = header
        .split("@@")
        .nth(1)
        .ok_or_else(|| "malformed hunk header".to_string())?
        .trim();

    let parts: Vec<&str> = inner.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(format!("expected at least 2 range specs, got: {header}"));
    }

    let orig = parse_range(parts[0].trim_start_matches('-'))?;
    let new = parse_range(parts[1].trim_start_matches('+'))?;

    Ok((orig.0, orig.1, new.1))
}

fn parse_range(s: &str) -> Result<(usize, usize), String> {
    if let Some((start, count)) = s.split_once(',') {
        let start = start
            .parse::<usize>()
            .map_err(|e| format!("bad line number '{start}': {e}"))?;
        let count = count
            .parse::<usize>()
            .map_err(|e| format!("bad count '{count}': {e}"))?;
        Ok((start, count))
    } else {
        let start = s
            .parse::<usize>()
            .map_err(|e| format!("bad line number '{s}': {e}"))?;
        Ok((start, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn apply(orig: &str, diff: &str) -> String {
        apply_unified_diff(orig, diff).unwrap().patched
    }

    #[test]
    fn apply_simple_replacement() {
        let orig = "fn main() {\n    println!(\"hello\");\n}\n";
        let diff = "@@ -1,3 +1,3 @@\n fn main() {\n-    println!(\"hello\");\n+    println!(\"hello, world\");\n }\n";
        let result = apply(orig, diff);
        assert!(result.contains("hello, world"));
        assert!(!result.contains("println!(\"hello\");"));
    }

    #[test]
    fn apply_addition_only() {
        let orig = "line1\nline2\n";
        let diff = "@@ -1,2 +1,3 @@\n line1\n+inserted\n line2\n";
        let result = apply(orig, diff);
        assert_eq!(result, "line1\ninserted\nline2\n");
    }

    #[test]
    fn apply_deletion_only() {
        let orig = "line1\ndelete_me\nline2\n";
        let diff = "@@ -1,3 +1,2 @@\n line1\n-delete_me\n line2\n";
        let result = apply(orig, diff);
        assert_eq!(result, "line1\nline2\n");
    }

    #[test]
    fn apply_empty_diff_noop() {
        let orig = "unchanged\n";
        let result = apply_unified_diff(orig, "").unwrap();
        assert_eq!(result.patched, orig);
        assert_eq!(result.hunks_applied, 0);
    }

    #[test]
    fn context_mismatch_returns_error() {
        let orig = "a\nb\nc\n";
        // The diff says line 1 is "x" but it's "a".
        let diff = "@@ -1,2 +1,2 @@\n x\n-b\n+B\n";
        let err = apply_unified_diff(orig, diff).unwrap_err();
        assert!(matches!(err, DiffError::HunkApplyError { .. }));
    }

    #[test]
    fn sandbox_path_rejects_traversal() {
        let dir = TempDir::new().unwrap();
        let tool = ApplyDiffTool::new(dir.path());
        let result = tool.sandbox_path("../../etc/passwd");
        assert!(matches!(result, Err(DiffError::PathTraversal(_))));
    }

    #[test]
    fn sandbox_path_accepts_valid_path() {
        let dir = TempDir::new().unwrap();
        let file = dir.path().join("src").join("main.rs");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "fn main() {}").unwrap();
        let tool = ApplyDiffTool::new(dir.path());
        let result = tool.sandbox_path("src/main.rs");
        assert!(result.is_ok());
    }

    #[test]
    fn parse_hunk_header_standard() {
        let (start, orig, new) = parse_hunk_header("@@ -10,6 +10,7 @@ fn foo() {").unwrap();
        assert_eq!(start, 10);
        assert_eq!(orig, 6);
        assert_eq!(new, 7);
    }

    #[test]
    fn parse_hunk_header_single_line() {
        // @@ -1 +1 @@ — count defaults to 1
        let (start, orig, new) = parse_hunk_header("@@ -1 +1 @@").unwrap();
        assert_eq!(start, 1);
        assert_eq!(orig, 1);
        assert_eq!(new, 1);
    }
}
