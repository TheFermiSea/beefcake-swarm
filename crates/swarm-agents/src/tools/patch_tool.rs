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

/// Maximum number of replacements allowed in a single call to prevent
/// accidental mass-edits (e.g., replacing a common keyword everywhere).
const MAX_REPLACEMENTS: usize = 1;

#[derive(Deserialize)]
pub struct EditFileArgs {
    /// Relative path within the workspace.
    pub path: String,
    /// The exact text to find in the file. Must match exactly once.
    pub old_content: String,
    /// The replacement text.
    pub new_content: String,
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
}

/// Normalize whitespace for fuzzy matching: collapse runs of whitespace
/// to single spaces, trim each line, but preserve line structure.
fn normalize_whitespace(s: &str) -> String {
    s.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Count the leading whitespace width of a line (tabs count as 4 spaces).
fn indent_width(line: &str) -> usize {
    line.chars()
        .take_while(|c| c.is_whitespace())
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum()
}

/// Find the minimum indentation width among non-empty lines.
fn min_indent(text: &str) -> usize {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(indent_width)
        .min()
        .unwrap_or(0)
}

/// Re-indent `new_content` so its base indentation matches `original_region`.
///
/// When a fuzzy (whitespace-normalized) match is used, the model's `new_content`
/// typically has stripped or wrong indentation. This function shifts all lines
/// so the minimum indent matches the original region.
fn reindent_to_match(original_region: &str, new_content: &str) -> String {
    let orig_min = min_indent(original_region);
    let new_min = min_indent(new_content);

    if orig_min == new_min {
        return new_content.to_string();
    }

    let mut result = String::with_capacity(new_content.len() + 128);
    for (i, line) in new_content.lines().enumerate() {
        if i > 0 {
            result.push('\n');
        }
        if line.trim().is_empty() {
            // Preserve blank lines as-is
            continue;
        }
        let current = indent_width(line);
        let adjusted = if orig_min > new_min {
            current + (orig_min - new_min)
        } else {
            current.saturating_sub(new_min - orig_min)
        };
        for _ in 0..adjusted {
            result.push(' ');
        }
        result.push_str(line.trim_start());
    }
    // Preserve trailing newline from original if present
    if new_content.ends_with('\n') {
        result.push('\n');
    }
    result
}

/// Find all occurrences of `needle` in `haystack`, returning their byte offsets.
fn find_all(haystack: &str, needle: &str) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        offsets.push(start + pos);
        start += pos + 1;
    }
    offsets
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

/// Find the position of the next newline byte (`\n`) starting from `offset`.
fn memchr_newline(bytes: &[u8], offset: usize) -> Option<usize> {
    bytes[offset..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| offset + p)
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
                        "description": "The exact text block to find and replace. Must be unique in the file. Include enough context lines to be unambiguous."
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The replacement text. Can be empty to delete the old_content block."
                    }
                },
                "required": ["path", "old_content", "new_content"]
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

        // Heuristic: unescape double-encoded content from local models
        let old_content = unescape_if_double_encoded(&args.old_content);
        let new_content = unescape_if_double_encoded(&args.new_content);

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
                        // Provide helpful error: show what's in the file near where
                        // the match might have been
                        let preview = if content.len() > 500 {
                            format!("{}...", &content[..500])
                        } else {
                            content.clone()
                        };
                        return Err(ToolError::Io(std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            format!(
                                "old_content not found in {}. \
                                 Read the file first to see current content. \
                                 First 500 chars:\n{}",
                                args.path, preview
                            ),
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
                return Err(ToolError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "old_content matches {n} locations in {}. \
                         Include more surrounding context to make the match unique.",
                        args.path
                    ),
                )));
            }
            _ => unreachable!(),
        };

        // No-op detection: reject edits that don't actually change the file.
        // Common with local models whose old_content/new_content differ only in
        // whitespace — fuzzy match finds the location but the replacement is
        // byte-identical to the original region.
        if new_file_content == content {
            return Err(ToolError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "edit_file: no-op edit — replacement is identical to current \
                     content in {}. Re-read the file and verify your change \
                     is different from the existing code.",
                    args.path
                ),
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

        std::fs::write(&full_path, &new_file_content)?;

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
    use super::*;

    // --- normalize_whitespace ---

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

    // --- find_all ---

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

    // --- reindent_to_match ---

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

    // --- strip_line_number_prefixes ---

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

    // --- unescape_if_double_encoded ---

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
        assert_eq!(args.old_content, "fn old()");
        assert_eq!(args.new_content, "fn new()");
    }

    #[test]
    fn test_edit_file_args_missing_field_fails() {
        let json = r#"{"path": "src/lib.rs", "old_content": "fn old()"}"#;
        let result = serde_json::from_str::<EditFileArgs>(json);
        assert!(result.is_err());
    }

    // --- memchr_newline ---

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
}

/// Strip line-number prefixes added by `read_file` ranged output.
///
/// When `read_file` is called with `start_line`/`end_line`, the output has
/// `{:>5}: ` prefixes (e.g., `   42: fn main() {`). If the model copies this
/// into `old_content`, the exact match fails because the file on disk doesn't
/// have these prefixes. This function detects and strips them.
///
/// Detection heuristic: if ≥80% of non-empty lines match `^\s*\d+: `, strip
/// the prefix from ALL lines (don't strip selectively to avoid partial
/// mismatches). Returns the original string if line-number prefixes aren't
/// detected.
fn strip_line_number_prefixes(s: &str) -> String {
    // First, strip the `[Lines X-Y of Z total]` header that read_file
    // includes with ranged reads.
    let s = if s.starts_with("[Lines ") {
        match s.find('\n') {
            Some(pos) => &s[pos + 1..],
            None => return String::new(),
        }
    } else {
        s
    };

    let lines: Vec<&str> = s.lines().collect();
    let non_empty: Vec<&&str> = lines.iter().filter(|l| !l.trim().is_empty()).collect();
    if non_empty.is_empty() {
        return s.to_string();
    }

    // Count lines matching the read_file line-number format: optional whitespace,
    // digits, colon, space (e.g., "   42: code here" or "1: code here")
    let prefix_count = non_empty
        .iter()
        .filter(|line| {
            let trimmed = line.trim_start();
            // Must start with a digit
            if !trimmed.starts_with(|c: char| c.is_ascii_digit()) {
                return false;
            }
            // Find the ": " separator after digits
            if let Some(colon_pos) = trimmed.find(": ") {
                trimmed[..colon_pos].chars().all(|c| c.is_ascii_digit())
            } else {
                false
            }
        })
        .count();

    // Only strip if ≥80% of non-empty lines have the prefix
    if prefix_count * 5 < non_empty.len() * 4 {
        return s.to_string();
    }

    // Strip the prefix: everything up to and including the first ": "
    lines
        .iter()
        .map(|line| {
            let trimmed = line.trim_start();
            if let Some(colon_pos) = trimmed.find(": ") {
                if trimmed[..colon_pos].chars().all(|c| c.is_ascii_digit()) {
                    return &trimmed[colon_pos + 2..];
                }
            }
            line
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Detect and unescape double-JSON-encoded content from local models.
/// Only unescapes when escape sequences (`\n`, `\t`, `\"`, `\\`) are present,
/// preventing legitimate quoted text like `"hello"` from being stripped.
fn unescape_if_double_encoded(s: &str) -> String {
    if s.starts_with('"') && s.ends_with('"') && s.len() > 2 {
        let inner = &s[1..s.len() - 1];
        let has_escapes = inner.contains("\\n")
            || inner.contains("\\t")
            || inner.contains("\\r")
            || inner.contains("\\\"")
            || inner.contains("\\\\")
            || inner.contains("\\u");
        if has_escapes {
            match serde_json::from_str::<String>(s) {
                Ok(unescaped) if unescaped != s => return unescaped,
                _ => {}
            }
        }
    }
    s.to_string()
}
