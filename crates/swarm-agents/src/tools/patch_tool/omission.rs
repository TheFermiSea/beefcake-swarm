const STANDALONE_ELLIPSIS: &[&str] = &["...", "…", ".. ."];
const ELLIPSIS_MARKS: &[&str] = &["...", "…"];
const DESCRIPTIVE_OMISSION_KEYWORDS: &[&str] = &[
    "existing",
    "rest of",
    "remaining",
    "unchanged",
    "omitted",
    "truncated",
    "snip",
];
const BRACKETED_OMISSION_KEYWORDS: &[&str] = &["rest", "remaining", "implementation", "unchanged"];
const PARENTHESIZED_OMISSION_KEYWORDS: &[&str] = &["remaining", "unchanged", "omitted"];
const BLOCK_OMISSION_KEYWORDS: &[&str] = &["existing", "rest of", "remaining", "omitted"];

fn contains_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| haystack.contains(n))
}

/// Classify a line-comment body (already lowercased, with `//` or `#` stripped).
fn is_omission_line_comment(body: &str) -> bool {
    if STANDALONE_ELLIPSIS.contains(&body) {
        return true;
    }
    if contains_any(body, ELLIPSIS_MARKS) && contains_any(body, DESCRIPTIVE_OMISSION_KEYWORDS) {
        return true;
    }
    if body.starts_with('[')
        && body.ends_with(']')
        && contains_any(body, BRACKETED_OMISSION_KEYWORDS)
    {
        return true;
    }
    if body.starts_with('(')
        && body.ends_with(')')
        && contains_any(body, PARENTHESIZED_OMISSION_KEYWORDS)
    {
        return true;
    }
    false
}

/// Classify a `/* ... */` block comment's inner text (not lowercased).
fn is_omission_block_comment(inner: &str) -> bool {
    if inner == "..." || inner == "…" {
        return true;
    }
    contains_any(&inner.to_ascii_lowercase(), BLOCK_OMISSION_KEYWORDS)
}

/// Detect placeholder/omission patterns in content that indicate the LLM
/// truncated code instead of providing the complete replacement.
///
/// Returns `Some(pattern)` with the first detected placeholder, or `None` if clean.
/// Patterns from Gemini CLI, Claude Code, and SWE-agent research:
/// - `// ... existing code ...`, `// ... rest of file ...`
/// - `// (remaining code unchanged)`, `// [rest of implementation]`
/// - `/* ... */`, `// ...`  (standalone ellipsis comments)
/// - `# ... existing ...` (Python-style)
pub fn detect_omission_placeholder(content: &str) -> Option<&str> {
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(body_raw) = trimmed
            .strip_prefix("//")
            .or_else(|| trimmed.strip_prefix('#'))
        {
            if is_omission_line_comment(&body_raw.trim().to_ascii_lowercase()) {
                return Some(trimmed);
            }
        }
        if let Some(inner) = trimmed
            .strip_prefix("/*")
            .and_then(|s| s.strip_suffix("*/"))
            .map(str::trim)
        {
            if is_omission_block_comment(inner) {
                return Some(trimmed);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- detect_omission_placeholder ---

    #[test]
    fn test_omission_standalone_ellipsis() {
        let content = "fn main() {\n    // ...\n}";
        assert_eq!(detect_omission_placeholder(content), Some("// ..."));
    }

    #[test]
    fn test_omission_existing_code() {
        let content = "fn main() {\n    // ... existing code ...\n}";
        assert_eq!(
            detect_omission_placeholder(content),
            Some("// ... existing code ...")
        );
    }

    #[test]
    fn test_omission_rest_of_file() {
        let content = "fn main() {\n    // ... rest of file ...\n}";
        assert_eq!(
            detect_omission_placeholder(content),
            Some("// ... rest of file ...")
        );
    }

    #[test]
    fn test_omission_remaining_unchanged() {
        let content = "fn main() {\n    // (remaining code unchanged)\n}";
        assert_eq!(
            detect_omission_placeholder(content),
            Some("// (remaining code unchanged)")
        );
    }

    #[test]
    fn test_omission_bracketed() {
        let content = "fn main() {\n    // [rest of implementation]\n}";
        assert_eq!(
            detect_omission_placeholder(content),
            Some("// [rest of implementation]")
        );
    }

    #[test]
    fn test_omission_block_comment() {
        let content = "fn main() {\n    /* ... */\n}";
        assert_eq!(detect_omission_placeholder(content), Some("/* ... */"));
    }

    #[test]
    fn test_omission_block_comment_existing() {
        let content = "fn main() {\n    /* existing code omitted */\n}";
        assert_eq!(
            detect_omission_placeholder(content),
            Some("/* existing code omitted */")
        );
    }

    #[test]
    fn test_omission_clean_code_passes() {
        // Legitimate ellipses in strings shouldn't trigger
        let content = "fn main() {\n    println!(\"hello\");\n    // This is a normal comment\n}";
        assert_eq!(detect_omission_placeholder(content), None);
    }

    #[test]
    fn test_omission_legitimate_ellipsis_in_string() {
        // We only check comment lines, so strings shouldn't trigger
        let content = "fn main() {\n    let s = \"Loading...\";\n}";
        assert_eq!(detect_omission_placeholder(content), None);
    }

    #[test]
    fn test_omission_python_style() {
        let content = "def foo():\n    # ... existing ...\n";
        assert_eq!(
            detect_omission_placeholder(content),
            Some("# ... existing ...")
        );
    }

    #[test]
    fn test_omission_unicode_ellipsis() {
        let content = "fn main() {\n    // \u{2026}\n}";
        assert_eq!(detect_omission_placeholder(content), Some("// \u{2026}"));
    }
}
