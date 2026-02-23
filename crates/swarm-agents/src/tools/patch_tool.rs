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
}

impl EditFileTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
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

        // Try exact match first
        let occurrences = find_all(&content, &old_content);

        let new_file_content = match occurrences.len() {
            0 => {
                // Exact match failed — try fuzzy (whitespace-normalized) match
                tracing::warn!(
                    path = %args.path,
                    "edit_file: exact match failed, trying whitespace-normalized match"
                );
                match fuzzy_find_unique(&content, &old_content) {
                    Some((start, end)) => {
                        let mut result = String::with_capacity(
                            content.len() - (end - start) + new_content.len(),
                        );
                        result.push_str(&content[..start]);
                        result.push_str(&new_content);
                        result.push_str(&content[end..]);
                        result
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
                result
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
        let new_lines = new_content.lines().count();
        let diff = new_lines as i64 - old_lines as i64;
        let sign = if diff >= 0 { "+" } else { "" };
        Ok(format!(
            "Edited {}: replaced {old_lines} lines with {new_lines} lines ({sign}{diff})",
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
