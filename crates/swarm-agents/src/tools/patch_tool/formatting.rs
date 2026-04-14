/// Normalize whitespace for fuzzy matching: collapse runs of whitespace
/// to single spaces, trim each line, but preserve line structure.
pub fn normalize_whitespace(s: &str) -> String {
    s.lines()
        .map(|line| line.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Count the leading whitespace width of a line (tabs count as 4 spaces).
pub fn indent_width(line: &str) -> usize {
    line.chars()
        .take_while(|c| c.is_whitespace())
        .map(|c| if c == '\t' { 4 } else { 1 })
        .sum()
}

/// Find the minimum indentation width among non-empty lines.
pub fn min_indent(text: &str) -> usize {
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
pub fn reindent_to_match(original_region: &str, new_content: &str) -> String {
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

/// Parse an anchor string like "42:a3" into (line_number, hash_string).
pub fn parse_anchor(s: &str) -> Option<(usize, String)> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    if parts.len() != 2 {
        return None;
    }
    let line_num: usize = parts[0].parse().ok()?;
    Some((line_num, parts[1].to_string()))
}

/// Find all occurrences of `needle` in `haystack`, returning their byte offsets.
pub fn find_all(haystack: &str, needle: &str) -> Vec<usize> {
    let mut offsets = Vec::new();
    let mut start = 0;
    while let Some(pos) = haystack[start..].find(needle) {
        offsets.push(start + pos);
        start += pos + 1;
    }
    offsets
}

/// Find the position of the next newline byte (`\n`) starting from `offset`.
pub fn memchr_newline(bytes: &[u8], offset: usize) -> Option<usize> {
    bytes[offset..]
        .iter()
        .position(|&b| b == b'\n')
        .map(|p| offset + p)
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
pub fn strip_line_number_prefixes(s: &str) -> String {
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
    // Also matches hashline format: "42:a3|code here"
    let prefix_count = non_empty
        .iter()
        .filter(|line| {
            let trimmed = line.trim_start();
            // Must start with a digit
            if !trimmed.starts_with(|c: char| c.is_ascii_digit()) {
                return false;
            }
            // Match old format: "42: code" OR new hashline format: "42:a3|code"
            if let Some(colon_pos) = trimmed.find(':') {
                let before_colon = &trimmed[..colon_pos];
                if !before_colon.chars().all(|c| c.is_ascii_digit()) {
                    return false;
                }
                let after_colon = &trimmed[colon_pos + 1..];
                // Old format: ": " (space after colon)
                if after_colon.starts_with(' ') {
                    return true;
                }
                // Hashline format: "hex|" (hex chars then pipe)
                if let Some(pipe_pos) = after_colon.find('|') {
                    return after_colon[..pipe_pos]
                        .chars()
                        .all(|c| c.is_ascii_hexdigit());
                }
                false
            } else {
                false
            }
        })
        .count();

    // Only strip if ≥80% of non-empty lines have the prefix
    if prefix_count * 5 < non_empty.len() * 4 {
        return s.to_string();
    }

    // Strip the prefix: everything up to and including the first ": " or hashline "N:hex|"
    lines
        .iter()
        .map(|line| {
            let trimmed = line.trim_start();
            if let Some(colon_pos) = trimmed.find(':') {
                let before_colon = &trimmed[..colon_pos];
                if before_colon.chars().all(|c| c.is_ascii_digit()) {
                    let after_colon = &trimmed[colon_pos + 1..];
                    // Old format: "42: code"
                    if let Some(stripped) = after_colon.strip_prefix(' ') {
                        return stripped;
                    }
                    // Hashline format: "42:a3|code"
                    if let Some(pipe_pos) = after_colon.find('|') {
                        if after_colon[..pipe_pos]
                            .chars()
                            .all(|c| c.is_ascii_hexdigit())
                        {
                            return &after_colon[pipe_pos + 1..];
                        }
                    }
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
pub fn unescape_if_double_encoded(s: &str) -> String {
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

/// Strip truncation markers that models copy from read_file output.
/// Removes lines like `[...386 more lines truncated...]` or `[...N lines truncated. DO NOT...]`.
pub fn strip_truncation_markers(s: &str) -> String {
    let filtered: Vec<&str> = s
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !(trimmed.starts_with("[...")
                && (trimmed.contains("truncated") || trimmed.contains("lines"))
                && trimmed.ends_with(']'))
        })
        .collect();
    if filtered.len() == s.lines().count() {
        // No markers removed — return original to avoid allocating
        return s.to_string();
    }
    filtered.join("\n")
}

/// Per-line selective prefix stripping: strip hashline/line-number prefixes
/// from individual lines that match the pattern, regardless of what percentage
/// of lines have the prefix. Unlike `strip_line_number_prefixes` which requires
/// ≥80% of lines to match, this strips each line independently.
pub fn strip_line_number_prefixes_selective(s: &str) -> String {
    let lines: Vec<&str> = s.lines().collect();
    let mut any_stripped = false;
    let result: Vec<&str> = lines
        .iter()
        .map(|line| {
            let trimmed = line.trim_start();
            if let Some(colon_pos) = trimmed.find(':') {
                let before_colon = &trimmed[..colon_pos];
                if before_colon.chars().all(|c| c.is_ascii_digit()) && !before_colon.is_empty() {
                    let after_colon = &trimmed[colon_pos + 1..];
                    // Old format: "42: code"
                    if let Some(stripped) = after_colon.strip_prefix(' ') {
                        any_stripped = true;
                        return stripped;
                    }
                    // Hashline format: "42:a3|code"
                    if let Some(pipe_pos) = after_colon.find('|') {
                        if after_colon[..pipe_pos]
                            .chars()
                            .all(|c| c.is_ascii_hexdigit())
                        {
                            any_stripped = true;
                            return &after_colon[pipe_pos + 1..];
                        }
                    }
                }
            }
            line
        })
        .collect();
    if !any_stripped {
        return s.to_string();
    }
    result.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- normalize_whitespace ---

    #[test]
    fn test_normalize_whitespace_collapses_spaces() {
        let input = "fn    main()   {  \n    println!(\"hi\");\n}";
        let expected = "fn main() {\nprintln!(\"hi\");\n}";
        assert_eq!(normalize_whitespace(input), expected);
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
        assert_eq!(normalize_whitespace("   \n  \n"), "\n");
    }

    // --- find_all ---

    #[test]
    fn test_find_all_multiple_matches() {
        let haystack = "foo bar foo baz foo";
        let matches = find_all(haystack, "foo");
        assert_eq!(matches, vec![0, 8, 16]);
    }

    #[test]
    fn test_find_all_no_match() {
        let haystack = "foo bar baz";
        let matches = find_all(haystack, "qux");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_find_all_single_match() {
        let haystack = "fn main() {\n    hello();\n}";
        let matches = find_all(haystack, "hello()");
        assert_eq!(matches, vec![16]);
    }

    #[test]
    fn test_find_all_overlapping() {
        let haystack = "aaaaa";
        let matches = find_all(haystack, "aa");
        assert_eq!(matches, vec![0, 1, 2, 3]);
    }

    // --- indent_width & min_indent ---

    #[test]
    fn test_indent_width_spaces() {
        assert_eq!(indent_width("    fn foo()"), 4);
        assert_eq!(indent_width("  fn foo()"), 2);
    }

    #[test]
    fn test_indent_width_tab() {
        assert_eq!(indent_width("\tfn foo()"), 4);
        assert_eq!(indent_width("\t\tfn foo()"), 8);
    }

    #[test]
    fn test_indent_width_none() {
        assert_eq!(indent_width("fn foo()"), 0);
        assert_eq!(indent_width(""), 0);
    }

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
        assert!(result.contains("        baz();"));
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
        // Only 1 out of 3 lines has a prefix (< 80%)
        let input = "42: fn main() {\n    println!(\"hi\");\n}";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_strip_empty_input() {
        assert_eq!(strip_line_number_prefixes(""), "");
        assert_eq!(strip_line_number_prefixes("   \n  \n"), "   \n  \n");
    }

    #[test]
    fn test_strip_read_file_header() {
        // read_file outputs a header on ranged reads
        let input = "[Lines 42-44 of 100 total]\n   42: fn main() {\n   43:     println!(\"hi\");\n   44: }";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, "fn main() {\n    println!(\"hi\");\n}");
    }

    #[test]
    fn test_no_strip_legitimate_numeric_content() {
        // e.g. a literal string or struct that happens to look like a prefix
        let input = "let x = \"\n1: apples\n2: oranges\n3: bananas\n\";";
        // 3 out of 5 lines have prefixes, which is 60%, so it shouldn't strip.
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, input);
    }

    #[test]
    fn test_strip_hashline_prefixes() {
        let input = "1:a3|fn main() {\n2:0e|    println!(\"hi\");\n3:ff|}";
        let result = strip_line_number_prefixes(input);
        assert_eq!(result, "fn main() {\n    println!(\"hi\");\n}");
    }

    // --- unescape_if_double_encoded ---

    #[test]
    fn test_unescape_double_encoded_with_escapes() {
        // "\"fn main() {\\n    hello();\\n}\""
        let input = r#""fn main() {\n    hello();\n}""#;
        let expected = "fn main() {\n    hello();\n}";
        assert_eq!(unescape_if_double_encoded(input), expected);
    }

    #[test]
    fn test_unescape_plain_quoted_no_escapes() {
        // A plain string literal without escape sequences should be left alone
        let input = r#""hello world""#;
        assert_eq!(unescape_if_double_encoded(input), input);

        let input = r#""#;
        assert_eq!(unescape_if_double_encoded(input), input);
    }

    #[test]
    fn test_unescape_not_quoted() {
        let input = "fn main() {\n    hello();\n}";
        assert_eq!(unescape_if_double_encoded(input), input);
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

    // --- parse_anchor ---

    #[test]
    fn test_parse_anchor() {
        assert_eq!(parse_anchor("42:a3"), Some((42, "a3".to_string())));
        assert_eq!(parse_anchor("1:0e"), Some((1, "0e".to_string())));
        assert_eq!(parse_anchor("invalid"), None);
        assert_eq!(parse_anchor("not_a_number:a3"), None);
        assert_eq!(parse_anchor("42"), None);
    }

    // --- strip_truncation_markers ---

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

    // --- strip_line_number_prefixes_selective ---

    #[test]
    fn test_selective_strip_mixed_content() {
        // Some lines have hashline prefix, others don't
        let input = "42:a3|fn main() {\n    println!(\"hi\");\n44:ff|}";
        let result = strip_line_number_prefixes_selective(input);
        assert_eq!(result, "fn main() {\n    println!(\"hi\");\n}");
    }

    #[test]
    fn test_selective_strip_no_prefixes() {
        let input = "fn main() {\n    println!(\"hi\");\n}";
        let result = strip_line_number_prefixes_selective(input);
        assert_eq!(result, input);
    }
}
