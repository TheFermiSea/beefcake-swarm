//! Search/replace edit tool for targeted file modifications.
//!
//! Replaces the need for full-file rewrites via `write_file`. The model
//! specifies the exact text to find and its replacement — no line numbers
//! or diff headers required. This is the same pattern used by Aider
//! (SEARCH/REPLACE blocks) and OpenHands (str_replace_editor).

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{sandbox_check, ToolError};
use crate::action_validator::blake3_short;

pub mod fabrication;
pub use fabrication::{is_data_file, looks_like_fabricated_data};

pub mod omission;
pub use omission::detect_omission_placeholder;

pub mod formatting;
use formatting::{
    find_all, memchr_newline, min_indent, normalize_whitespace, parse_anchor, reindent_to_match,
    strip_line_number_prefixes, strip_line_number_prefixes_selective, strip_truncation_markers,
    unescape_if_double_encoded,
};

/// Maximum number of replacements allowed in a single call to prevent
/// accidental mass-edits (e.g., replacing a common keyword everywhere).
const MAX_REPLACEMENTS: usize = 1;

#[derive(Deserialize)]
pub struct EditFileArgs {
    /// Relative path within the workspace.
    pub path: String,
    /// The exact text to find in the file. Must match exactly once.
    /// Optional when using anchor_start/anchor_end.
    #[serde(default)]
    pub old_content: Option<String>,
    /// The replacement text.
    pub new_content: String,
    /// Anchor start: "line:hash" from read_file hashline output (e.g. "42:a3").
    /// When both anchors are provided, replaces the line range instead of using str_replace.
    #[serde(default)]
    pub anchor_start: Option<String>,
    /// Anchor end: "line:hash" from read_file hashline output (e.g. "44:0e").
    #[serde(default)]
    pub anchor_end: Option<String>,
}

/// Edit a file by replacing a unique substring with new content.
///
/// The `old_content` must appear exactly once in the file. This prevents
/// ambiguous edits and ensures the model targets the right location.
/// If the exact match fails, a whitespace-normalized fuzzy match is attempted.
pub struct EditFileTool {
    pub working_dir: PathBuf,
    /// When set, only paths in this set may be edited. Used by subtask dispatch
    /// to enforce the non-overlap constraint at the tool layer.
    allowed_files: Option<std::collections::HashSet<String>>,
}

impl EditFileTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
            allowed_files: None,
        }
    }

    pub fn new_with_allowlist(
        working_dir: &Path,
        allowed: std::collections::HashSet<String>,
    ) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
            allowed_files: Some(allowed),
        }
    }

    /// Anchor-based edit: replace a line range identified by hashline anchors.
    fn anchor_edit(
        &self,
        rel_path: &str,
        full_path: &std::path::Path,
        content: &str,
        start_anchor: &str,
        end_anchor: &str,
        new_content: &str,
    ) -> Result<String, ToolError> {
        let lines: Vec<&str> = content.lines().collect();

        let (start_line, start_hash) = parse_anchor(start_anchor).ok_or_else(|| {
            ToolError::Validation(format!(
                "Invalid anchor_start format: '{start_anchor}'. Expected 'line:hash' (e.g. '42:a3')"
            ))
        })?;

        let (end_line, end_hash) = parse_anchor(end_anchor).ok_or_else(|| {
            ToolError::Validation(format!(
                "Invalid anchor_end format: '{end_anchor}'. Expected 'line:hash' (e.g. '44:0e')"
            ))
        })?;

        // Validate line numbers
        if start_line == 0 || end_line == 0 || start_line > lines.len() || end_line > lines.len() {
            return Err(ToolError::Validation(format!(
                "Anchor line numbers out of range: start={start_line}, end={end_line}, \
                 file has {} lines. Re-read the file to get fresh anchors.",
                lines.len()
            )));
        }

        if start_line > end_line {
            return Err(ToolError::Validation(format!(
                "anchor_start line ({start_line}) must be <= anchor_end line ({end_line})"
            )));
        }

        // Verify hashes match current content
        let actual_start_hash = blake3_short(lines[start_line - 1].trim());
        if actual_start_hash != start_hash {
            return Err(ToolError::Validation(format!(
                "Stale anchor_start: line {start_line} hash is now '{actual_start_hash}' \
                 (expected '{start_hash}'). Re-read the file to get fresh anchors."
            )));
        }

        let actual_end_hash = blake3_short(lines[end_line - 1].trim());
        if actual_end_hash != end_hash {
            return Err(ToolError::Validation(format!(
                "Stale anchor_end: line {end_line} hash is now '{actual_end_hash}' \
                 (expected '{end_hash}'). Re-read the file to get fresh anchors."
            )));
        }

        // Build new file: lines before range + new_content + lines after range
        let mut result = String::with_capacity(content.len());
        for line in &lines[..start_line - 1] {
            result.push_str(line);
            result.push('\n');
        }
        result.push_str(new_content);
        if !new_content.ends_with('\n') {
            result.push('\n');
        }
        for line in &lines[end_line..] {
            result.push_str(line);
            result.push('\n');
        }
        // Trim trailing newline if original didn't have one
        if !content.ends_with('\n') && result.ends_with('\n') {
            result.pop();
        }

        // No-op detection
        if result == content {
            return Err(ToolError::Validation(format!(
                "edit_file: no-op anchor edit — replacement is identical to lines {start_line}-{end_line} \
                 in {rel_path}. Verify your change is different."
            )));
        }

        std::fs::write(full_path, &result)?;

        let old_lines = end_line - start_line + 1;
        let new_lines = new_content.lines().count();
        let diff = new_lines as i64 - old_lines as i64;
        let sign = if diff >= 0 { "+" } else { "" };

        Ok(format!(
            "Edited {rel_path} (anchor mode): replaced lines {start_line}-{end_line} \
             ({old_lines} lines) with {new_lines} lines ({sign}{diff})"
        ))
    }
}

/// Quick lint check: run `cargo check` on the crate containing the edited file.
///
/// Returns `Some(diagnostics)` if new errors are detected, `None` if clean.
/// Uses `--message-format=short` for compact error output suitable for LLM feedback.
/// Times out after 30 seconds to avoid blocking the agent loop.
fn lint_check_fast(working_dir: &Path, file_path: &str) -> Option<String> {
    use std::process::Command;

    // Find the nearest Cargo.toml to determine the package
    let full_path = working_dir.join(file_path);
    let mut dir = full_path.parent()?;
    let mut package = None;
    loop {
        let cargo_toml = dir.join("Cargo.toml");
        if cargo_toml.exists() {
            if let Ok(content) = std::fs::read_to_string(&cargo_toml) {
                for line in content.lines() {
                    let trimmed = line.trim();
                    if trimmed.starts_with("name") {
                        if let Some(name) = trimmed.split('"').nth(1) {
                            package = Some(name.to_string());
                            break;
                        }
                    }
                }
            }
            break;
        }
        match dir.parent() {
            Some(parent) if parent != dir => dir = parent,
            _ => break,
        }
    }

    let mut cmd = Command::new("cargo");
    cmd.arg("check").arg("--message-format=short");
    if let Some(ref pkg) = package {
        cmd.arg("-p").arg(pkg);
    }
    cmd.current_dir(working_dir)
        .env("CARGO_TERM_COLOR", "never");

    let output = match cmd.output() {
        Ok(o) => o,
        Err(e) => {
            tracing::debug!(error = %e, "lint_check_fast: cargo check failed to run");
            return None; // Can't lint — don't block the edit
        }
    };

    if output.status.success() {
        return None; // Clean
    }

    let stderr = String::from_utf8_lossy(&output.stderr);
    // Filter to only error lines (skip warnings)
    let errors: Vec<&str> = stderr
        .lines()
        .filter(|l| l.contains("error[E") || l.contains("error:"))
        .take(5)
        .collect();

    if errors.is_empty() {
        return None; // Only warnings, no errors
    }

    Some(errors.join("\n"))
}

/// Try to find a unique match using whitespace-normalized comparison.
/// Returns `(start_byte, end_byte)` in the original string if exactly one match.
fn fuzzy_find_unique(content: &str, old_content: &str) -> Option<(usize, usize)> {
    let norm_needle = normalize_whitespace(old_content);
    if norm_needle.is_empty() {
        return None;
    }

    // Build a normalized version of the file content while tracking byte offsets
    // back to the original. We normalize line-by-line and try to find the needle
    // as a contiguous sequence of normalized lines.
    let needle_lines: Vec<&str> = norm_needle.lines().collect();
    if needle_lines.is_empty() {
        return None;
    }

    let content_lines: Vec<&str> = content.lines().collect();
    let norm_content_lines: Vec<String> = content_lines
        .iter()
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();

    let mut matches = Vec::new();

    for i in 0..content_lines.len().saturating_sub(needle_lines.len() - 1) {
        let window = &norm_content_lines[i..i + needle_lines.len()];
        if window.iter().zip(needle_lines.iter()).all(|(a, b)| a == b) {
            matches.push(i);
        }
    }

    if matches.len() != 1 {
        return None;
    }

    let match_start_line = matches[0];
    let match_end_line = match_start_line + needle_lines.len();

    // Convert line indices back to byte offsets in the original content.
    // We scan the raw bytes to correctly handle both LF and CRLF line endings.
    let mut byte_offset = 0;
    let mut start_byte = 0;
    let bytes = content.as_bytes();
    let mut line_idx = 0;

    while byte_offset <= bytes.len() {
        if line_idx == match_start_line {
            start_byte = byte_offset;
        }
        if line_idx == match_end_line {
            return Some((start_byte, byte_offset));
        }
        if byte_offset >= bytes.len() {
            break;
        }
        // Find next newline
        match memchr_newline(bytes, byte_offset) {
            Some(nl_pos) => {
                // Skip past the \n (str::lines() already handles \r\n by stripping \r,
                // so byte_offset after \n is correct for both LF and CRLF)
                byte_offset = nl_pos + 1;
                line_idx += 1;
            }
            None => {
                // Last line without trailing newline
                byte_offset = bytes.len();
                line_idx += 1;
            }
        }
    }
    // If match extends to end of file
    if match_end_line == content_lines.len() {
        return Some((start_byte, content.len()));
    }
    None
}

impl Tool for EditFileTool {
    const NAME: &'static str = "edit_file";
    type Error = ToolError;
    type Args = EditFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "edit_file".into(),
            description: "Edit a file by replacing a unique text block with new content. \
                          PREFERRED over write_file for modifying existing files. \
                          The old_content must appear exactly once in the file. \
                          Include enough surrounding context (3-5 lines) to ensure uniqueness."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative path to the file within the workspace"
                    },
                    "old_content": {
                        "type": "string",
                        "description": "REQUIRED. The exact text block to find and replace. Must be unique in the file. Include enough surrounding context (3-5 lines) to be unambiguous. Pass empty string only when using anchor_start/anchor_end."
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The replacement text. Can be empty to delete the old_content block."
                    },
                    "anchor_start": {
                        "type": "string",
                        "description": "Start anchor from read_file hashline output (e.g. '42:a3'). When both anchor_start and anchor_end are provided, replaces the line range instead of matching old_content."
                    },
                    "anchor_end": {
                        "type": "string",
                        "description": "End anchor from read_file hashline output (e.g. '44:0e'). Used with anchor_start."
                    }
                },
                "required": ["path", "old_content", "new_content"],
                "additionalProperties": false
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if let Some(ref allowed) = self.allowed_files {
            if !allowed.contains(&args.path) {
                return Err(ToolError::Policy(format!(
                    "edit_file: path '{}' is not in this worker's target_files allowlist",
                    args.path
                )));
            }
        }
        let full_path = sandbox_check(&self.working_dir, &args.path)?;

        // Read the current file
        let content = std::fs::read_to_string(&full_path).map_err(|e| {
            ToolError::Io(std::io::Error::new(
                e.kind(),
                format!("cannot read {}: {e}", args.path),
            ))
        })?;

        let new_content = unescape_if_double_encoded(&args.new_content);

        // --- Anchor-based edit path (preferred when anchors are provided) ---
        if let (Some(ref start_anchor), Some(ref end_anchor)) =
            (&args.anchor_start, &args.anchor_end)
        {
            return self.anchor_edit(
                &args.path,
                &full_path,
                &content,
                start_anchor,
                end_anchor,
                &new_content,
            );
        }

        // old_content is required for non-anchor str_replace path
        let old_content_raw = match args.old_content {
            Some(ref oc) if !oc.is_empty() => oc.clone(),
            Some(_) | None => {
                return Err(ToolError::Validation(
                    "edit_file: old_content is required. To INSERT new lines, include surrounding \
                     context lines in BOTH old_content and new_content. Example — to add Ok(()) \
                     before a closing brace:\n\
                     old_content: \"    assert_eq!(x, 1);\\n    }\\n}\"\n\
                     new_content: \"    assert_eq!(x, 1);\\n\\n        Ok(())\\n    }\\n}\"\n\
                     Alternatively, use anchor_start and anchor_end (hashline anchors from read_file)."
                    .to_string(),
                ));
            }
        };
        let old_content = unescape_if_double_encoded(&old_content_raw);
        // Strip truncation markers that models copy from read_file output.
        let old_content = strip_truncation_markers(&old_content);

        // Try exact match first (raw old_content as provided by the model)
        let mut occurrences = find_all(&content, &old_content);

        // Fallback: strip read_file line-number prefixes and header, then retry.
        // Models often copy `   42: fn main()` from read_file output into
        // old_content, but the file on disk has `fn main()` without the prefix.
        // We only strip as a fallback to avoid corrupting legitimate content
        // that starts with digits followed by `: `.
        let old_content = if occurrences.is_empty() {
            let stripped = strip_line_number_prefixes(&old_content);
            if stripped != old_content {
                tracing::info!(
                    path = %args.path,
                    "edit_file: exact match failed, retrying after stripping line-number prefixes"
                );
                occurrences = find_all(&content, &stripped);
                stripped
            } else {
                old_content
            }
        } else {
            old_content
        };

        // Tier 3: Per-line selective strip — strip hashline/line-number prefixes
        // from individual lines that have them, even if <80% of lines have prefixes.
        // This handles mixed content where some lines were copied from read_file
        // output and others were typed manually.
        let old_content = if occurrences.is_empty() {
            let selectively_stripped = strip_line_number_prefixes_selective(&old_content);
            if selectively_stripped != old_content {
                tracing::info!(
                    path = %args.path,
                    "edit_file: retrying after per-line selective prefix strip"
                );
                occurrences = find_all(&content, &selectively_stripped);
                selectively_stripped
            } else {
                old_content
            }
        } else {
            old_content
        };

        // `actual_replacement` is what was written at the match site.
        // For exact matches this equals new_content; for fuzzy matches it is
        // the reindented variant. Used for accurate line counts and preview.
        let (new_file_content, actual_replacement): (String, String) = match occurrences.len() {
            0 => {
                // Exact match failed — try fuzzy (whitespace-normalized) match
                tracing::warn!(
                    path = %args.path,
                    "edit_file: exact match failed, trying whitespace-normalized match"
                );
                match fuzzy_find_unique(&content, &old_content) {
                    Some((start, end)) => {
                        // Re-indent new_content to match the original region's
                        // indentation. Local models often strip indentation when
                        // generating edits from truncated file reads.
                        let original_region = &content[start..end];
                        let reindented = reindent_to_match(original_region, &new_content);
                        tracing::info!(
                            path = %args.path,
                            orig_indent = min_indent(original_region),
                            new_indent = min_indent(&new_content),
                            "edit_file: fuzzy match with reindentation"
                        );
                        let mut result =
                            String::with_capacity(content.len() - (end - start) + reindented.len());
                        result.push_str(&content[..start]);
                        result.push_str(&reindented);
                        result.push_str(&content[end..]);
                        (result, reindented)
                    }
                    None => {
                        let preview = if content.len() > 500 {
                            format!("{}...", &content[..content.floor_char_boundary(500)])
                        } else {
                            content.clone()
                        };
                        // Check if the file was likely truncated during read
                        let max_chars: usize = std::env::var("SWARM_READ_FILE_MAX_CHARS")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(6000);
                        let truncation_hint = if content.len() > max_chars {
                            "\n\nThis file was truncated during read. Use read_file with \
                             start_line/end_line to see exact content, then use \
                             anchor_start/anchor_end for reliable edits."
                        } else {
                            ""
                        };
                        return Err(ToolError::Validation(format!(
                            "old_content not found in {}. \
                             Read the file first to see current content. \
                             First 500 chars:\n{}{}",
                            args.path, preview, truncation_hint
                        )));
                    }
                }
            }
            1 => {
                // Exactly one match — perfect
                let start = occurrences[0];
                let end = start + old_content.len();
                let mut result =
                    String::with_capacity(content.len() - old_content.len() + new_content.len());
                result.push_str(&content[..start]);
                result.push_str(&new_content);
                result.push_str(&content[end..]);
                (result, new_content.clone())
            }
            n if n > MAX_REPLACEMENTS => {
                return Err(ToolError::Validation(format!(
                    "old_content matches {n} locations in {}. \
                     Include more surrounding context to make the match unique.",
                    args.path
                )));
            }
            _ => unreachable!(),
        };

        // Omission guard: reject edits where new_content contains placeholder patterns
        // like "// ... existing code ..." that indicate the LLM truncated the replacement.
        // These silently delete code when applied. Pattern from Gemini CLI.
        if let Some(placeholder) = detect_omission_placeholder(&new_content) {
            return Err(ToolError::Validation(format!(
                "edit_file: new_content contains an omission placeholder: `{placeholder}`. \
                 This would delete code. Provide the COMPLETE replacement text — do not use \
                 comments like '// ...' or '// rest of file' to represent omitted code."
            )));
        }

        // No-op detection: reject edits that don't actually change the file.
        // Common with local models whose old_content/new_content differ only in
        // whitespace — fuzzy match finds the location but the replacement is
        // byte-identical to the original region.
        if new_file_content == content {
            return Err(ToolError::Validation(format!(
                "edit_file: no-op edit — replacement is identical to current \
                 content in {}. Re-read the file and verify your change \
                 is different from the existing code.",
                args.path
            )));
        }

        // Blast-radius guard: warn if the edit shrinks the file by more than 50%
        if new_file_content.len() < content.len() / 2 {
            tracing::warn!(
                path = %args.path,
                original_len = content.len(),
                new_len = new_file_content.len(),
                "edit_file: edit would shrink file by >50% — applying anyway"
            );
        }

        // Data fabrication guard: reject writes to data/results files that look
        // like fabricated benchmark or experimental output. Workers cannot run
        // Python benchmarks, so numeric data in these files is hallucinated.
        if is_data_file(&args.path) && looks_like_fabricated_data(&actual_replacement) {
            return Err(ToolError::Policy(format!(
                "edit_file: BLOCKED — writing to data file '{}' with content that \
                 looks like fabricated benchmark/experimental results. Workers cannot \
                 run benchmarks directly. If this task requires running benchmarks, \
                 return BLOCKED and let the manager escalate.",
                args.path
            )));
        }

        std::fs::write(&full_path, &new_file_content)?;

        // Linter guardrail: run a quick syntax check after writing. If the edit
        // introduced new compilation errors, revert to the original content and
        // return diagnostics to the LLM. This prevents compounding errors across
        // iterations. SWE-agent evidence: +3% SWE-bench resolve rate.
        // Only runs for .rs files and only when SWARM_EDIT_LINT=1 (opt-in to avoid
        // slowing down test environments).
        if args.path.ends_with(".rs")
            && std::env::var("SWARM_EDIT_LINT")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
        {
            if let Some(diagnostics) = lint_check_fast(&self.working_dir, &args.path) {
                // Revert the edit
                std::fs::write(&full_path, &content)?;
                return Err(ToolError::Validation(format!(
                    "edit_file: reverted — edit introduced compilation errors in {}:\n{}\n\
                     Fix the errors and try the edit again.",
                    args.path, diagnostics
                )));
            }
        }

        let old_lines = old_content.lines().count();
        // Use actual_replacement (not new_content) for line counts — after a fuzzy
        // match the reindented text may have different line counts than the raw input.
        let new_lines = actual_replacement.lines().count();
        let diff = new_lines as i64 - old_lines as i64;
        let sign = if diff >= 0 { "+" } else { "" };

        // Include the actual written content in the response so the model
        // knows the current file state for subsequent edits (prevents
        // old_content drift when tool_choice=required forces multiple edits).
        let written_preview = if new_file_content.len() <= 2000 {
            new_file_content.clone()
        } else {
            // Find the replacement region using actual_replacement (not new_content,
            // which may differ after reindentation in fuzzy match path).
            let replacement_start = new_file_content
                .find(actual_replacement.lines().next().unwrap_or(""))
                .unwrap_or(0);
            let preview_start = new_file_content[..replacement_start]
                .rfind('\n')
                .map(|p| p + 1)
                .unwrap_or(0);
            let preview_end =
                (replacement_start + actual_replacement.len() + 200).min(new_file_content.len());
            format!("...{}", &new_file_content[preview_start..preview_end])
        };

        Ok(format!(
            "Edited {}: replaced {old_lines} lines with {new_lines} lines ({sign}{diff})\n\
             \n\
             Current file content:\n\
             ```\n\
             {written_preview}\n\
             ```",
            args.path
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::formatting::indent_width;
    use super::*;

    #[test]
    fn test_normalize_whitespace_collapses_spaces() {
        assert_eq!(normalize_whitespace("  hello   world  "), "hello world");
    }

    #[test]
    fn test_normalize_whitespace_preserves_lines() {
        let input = "  fn main() {\n    println!(\"hi\");\n  }";
        let expected = "fn main() {\nprintln!(\"hi\");\n}";
        assert_eq!(normalize_whitespace(input), expected);
    }

    #[test]
    fn test_normalize_whitespace_empty() {
        assert_eq!(normalize_whitespace(""), "");
    }

    #[test]
    fn test_find_all_multiple_matches() {
        let offsets = find_all("abcabcabc", "abc");
        assert_eq!(offsets, vec![0, 3, 6]);
    }

    #[test]
    fn test_find_all_no_match() {
        let offsets = find_all("hello world", "xyz");
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_find_all_single_match() {
        let offsets = find_all("hello world", "world");
        assert_eq!(offsets, vec![6]);
    }

    #[test]
    fn test_find_all_overlapping() {
        let offsets = find_all("aaa", "aa");
        assert_eq!(offsets, vec![0, 1]);
    }

    // --- fuzzy_find_unique ---

    #[test]
    fn test_fuzzy_find_unique_whitespace_difference() {
        let content = "fn main() {\n    println!(\"hi\");\n}\n";
        let needle = "fn main() {\nprintln!(\"hi\");\n}";
        let result = fuzzy_find_unique(content, needle);
        assert!(result.is_some());
        let (start, end) = result.unwrap();
        // Should cover the entire content (3 lines)
        assert_eq!(start, 0);
        assert!(end <= content.len());
    }

    #[test]
    fn test_fuzzy_find_unique_no_match() {
        let content = "fn main() {\n    println!(\"hi\");\n}\n";
        let needle = "fn foo() {\nbar();\n}";
        let result = fuzzy_find_unique(content, needle);
        assert!(result.is_none());
    }

    #[test]
    fn test_fuzzy_find_unique_empty_needle() {
        let result = fuzzy_find_unique("some content", "");
        assert!(result.is_none());
    }

    #[test]
    fn test_fuzzy_find_unique_ambiguous() {
        // Two identical blocks — should return None (not unique)
        let content = "fn a() {}\nfn b() {}\nfn a() {}\n";
        let needle = "fn a() {}";
        let result = fuzzy_find_unique(content, needle);
        assert!(result.is_none());
    }

    // --- indent_width ---

    #[test]
    fn test_indent_width_spaces() {
        assert_eq!(indent_width("    hello"), 4);
    }

    #[test]
    fn test_indent_width_tab() {
        assert_eq!(indent_width("\thello"), 4);
    }

    #[test]
    fn test_indent_width_none() {
        assert_eq!(indent_width("hello"), 0);
    }

    // --- min_indent ---

    #[test]
    fn test_min_indent_skips_blank_lines() {
        let text = "    fn foo() {\n\n        bar();\n    }";
        assert_eq!(min_indent(text), 4);
    }

    #[test]
    fn test_reindent_to_match_adds_indent() {
        let original = "    fn foo() {\n        bar();\n    }";
        let new_content = "fn foo() {\n    bar();\n    baz();\n}";
        let result = reindent_to_match(original, new_content);
        assert!(result.starts_with("    fn foo()"));
        assert!(result.contains("        bar();"));
        assert!(result.contains("        baz();"));
        assert!(result.contains("    }"));
    }

    #[test]
    fn test_reindent_to_match_same_indent_unchanged() {
        let original = "    fn foo() {}";
        let new_content = "    fn bar() {}";
        let result = reindent_to_match(original, new_content);
        assert_eq!(result, "    fn bar() {}");
    }

    #[test]
    fn test_reindent_to_match_removes_excess_indent() {
        let original = "fn main() {}";
        let new_content = "    fn main() {\n        hello();\n    }";
        let result = reindent_to_match(original, new_content);
        assert!(result.starts_with("fn main()"));
        assert!(result.contains("    hello();"));
    }

    #[test]
    fn test_strip_line_numbers_from_ranged_read() {
        let input = "   42: fn main() {\n   43:     println!(\"hi\");\n   44: }";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, "fn main() {\n    println!(\"hi\");\n}");
    }

    #[test]
    fn test_strip_line_numbers_single_digit() {
        let input = "1: use std::io;\n2: \n3: fn main() {}";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, "use std::io;\n\nfn main() {}");
    }

    #[test]
    fn test_no_strip_when_no_prefixes() {
        let input = "fn main() {\n    println!(\"hi\");\n}";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_no_strip_when_few_prefixes() {
        // Only 1 out of 3 lines has a prefix — below 80% threshold
        let input = "42: fn main() {\n    println!(\"hi\");\n}";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_strip_empty_input() {
        assert_eq!(strip_line_number_prefixes(""), "");
    }

    #[test]
    fn test_strip_read_file_header() {
        // read_file ranged output includes a [Lines X-Y of Z total] header
        let input = "[Lines 42-44 of 100 total]\n   42: fn main() {\n   43:     println!(\"hi\");\n   44: }";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, "fn main() {\n    println!(\"hi\");\n}");
    }

    #[test]
    fn test_no_strip_legitimate_numeric_content() {
        // Content that legitimately starts with "N: " patterns (e.g., an enum
        // list or numbered items) should NOT be stripped. Only 2 of 4 non-empty
        // lines match the prefix pattern — below the 80% threshold.
        let input = "Error codes:\n1: connection refused\n2: timeout\nSee docs for more.";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_strip_hashline_prefixes() {
        let input = "1:a3|fn main() {\n2:0e|    println!(\"hi\");\n3:ff|}";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, "fn main() {\n    println!(\"hi\");\n}");
    }

    #[test]
    fn test_unescape_double_encoded_with_escapes() {
        let input = r#""hello\nworld""#;
        let result = unescape_if_double_encoded(input);
        assert_eq!(result, "hello\nworld");
    }

    #[test]
    fn test_unescape_plain_quoted_no_escapes() {
        // No escape sequences — should return unchanged
        let input = r#""hello world""#;
        let result = unescape_if_double_encoded(input);
        assert_eq!(result, r#""hello world""#);
    }

    #[test]
    fn test_unescape_not_quoted() {
        let input = "hello world";
        let result = unescape_if_double_encoded(input);
        assert_eq!(result, "hello world");
    }

    // --- EditFileArgs deserialization ---

    #[test]
    fn test_edit_file_args_deserialize() {
        let json =
            r#"{"path": "src/lib.rs", "old_content": "fn old()", "new_content": "fn new()"}"#;
        let args: EditFileArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.path, "src/lib.rs");
        assert_eq!(args.old_content, Some("fn old()".to_string()));
        assert_eq!(args.new_content, "fn new()");
    }

    #[test]
    fn test_edit_file_args_missing_field_fails() {
        let json = r#"{"path": "src/lib.rs", "old_content": "fn old()"}"#;
        let result = serde_json::from_str::<EditFileArgs>(json);
        assert!(result.is_err());
    }

    #[test]
    fn test_memchr_newline_finds_first() {
        let bytes = b"hello\nworld";
        assert_eq!(memchr_newline(bytes, 0), Some(5));
    }

    #[test]
    fn test_memchr_newline_none() {
        let bytes = b"hello world";
        assert_eq!(memchr_newline(bytes, 0), None);
    }

    #[test]
    fn test_memchr_newline_from_offset() {
        let bytes = b"a\nb\nc";
        assert_eq!(memchr_newline(bytes, 2), Some(3));
    }

    #[test]
    fn test_parse_anchor() {
        assert_eq!(parse_anchor("42:a3"), Some((42, "a3".to_string())));
        assert_eq!(parse_anchor("1:ff"), Some((1, "ff".to_string())));
        assert_eq!(parse_anchor("invalid"), None);
        assert_eq!(parse_anchor(""), None);
    }

    // --- EditFileArgs with anchors ---

    #[test]
    fn test_edit_file_args_with_anchors() {
        let json = r#"{"path": "x.rs", "old_content": "", "new_content": "y", "anchor_start": "1:ab", "anchor_end": "3:cd"}"#;
        let args: EditFileArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.anchor_start, Some("1:ab".to_string()));
        assert_eq!(args.anchor_end, Some("3:cd".to_string()));
    }

    #[test]
    fn test_edit_file_args_without_anchors() {
        let json = r#"{"path": "x.rs", "old_content": "a", "new_content": "b"}"#;
        let args: EditFileArgs = serde_json::from_str(json).unwrap();
        assert!(args.anchor_start.is_none());
        assert!(args.anchor_end.is_none());
    }

    #[test]
    fn test_edit_file_args_old_content_optional() {
        let json = r#"{"path": "src/lib.rs", "new_content": "fn new()"}"#;
        let args: EditFileArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.path, "src/lib.rs");
        assert!(args.old_content.is_none());
        assert_eq!(args.new_content, "fn new()");
    }

    #[test]
    fn test_edit_file_args_anchor_only() {
        let json =
            r#"{"path": "x.rs", "new_content": "y", "anchor_start": "1:ab", "anchor_end": "3:cd"}"#;
        let args: EditFileArgs = serde_json::from_str(json).unwrap();
        assert!(args.old_content.is_none());
        assert_eq!(args.anchor_start, Some("1:ab".to_string()));
    }

    #[test]
    fn test_strip_truncation_marker() {
        let input = "fn main() {\n    hello();\n[...386 more lines truncated. Use start_line/end_line to read a specific range.]\n}";
        let result = strip_truncation_markers(input);
        assert_eq!(result, "fn main() {\n    hello();\n}");
    }

    #[test]
    fn test_strip_truncation_marker_none_present() {
        let input = "fn main() {\n    hello();\n}";
        let result = strip_truncation_markers(input);
        assert_eq!(result, input);
    }
}
