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
        // Match comment-style omission markers
        if let Some(comment_body) = trimmed
            .strip_prefix("//")
            .or_else(|| trimmed.strip_prefix('#'))
        {
            let body = comment_body.trim().to_ascii_lowercase();
            // Standalone ellipsis: "// ..." or "// …"
            if body == "..." || body == "…" || body == ".. ." {
                return Some(trimmed);
            }
            // Descriptive omission: "// ... existing code ..."
            if (body.contains("...") || body.contains('…'))
                && (body.contains("existing")
                    || body.contains("rest of")
                    || body.contains("remaining")
                    || body.contains("unchanged")
                    || body.contains("omitted")
                    || body.contains("truncated")
                    || body.contains("snip"))
            {
                return Some(trimmed);
            }
            // Bracketed omission: "// [rest of implementation]"
            if body.starts_with('[')
                && body.ends_with(']')
                && (body.contains("rest")
                    || body.contains("remaining")
                    || body.contains("implementation")
                    || body.contains("unchanged"))
            {
                return Some(trimmed);
            }
            // Parenthesized: "// (remaining code unchanged)"
            if body.starts_with('(')
                && body.ends_with(')')
                && (body.contains("remaining")
                    || body.contains("unchanged")
                    || body.contains("omitted"))
            {
                return Some(trimmed);
            }
        }
        // Block comment omission: "/* ... */"
        if trimmed.starts_with("/*") && trimmed.ends_with("*/") {
            let inner = trimmed
                .strip_prefix("/*")
                .and_then(|s| s.strip_suffix("*/"))
                .unwrap_or("")
                .trim();
            if inner == "..." || inner == "…" || inner.is_empty() {
                return Some(trimmed);
            }
            let lower = inner.to_ascii_lowercase();
            if lower.contains("existing")
                || lower.contains("rest of")
                || lower.contains("remaining")
                || lower.contains("omitted")
            {
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
