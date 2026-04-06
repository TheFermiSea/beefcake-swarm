//! Deterministic fix templates for common error patterns.
//!
//! Code-as-policy: for simple, repetitive fixes, apply a template directly
//! instead of invoking the LLM. Eliminates cost and latency for the most
//! common error categories.
//!
//! Source: AutoHarness paper (arxiv:2603.03329)

use coordination::feedback::ErrorCategory;
use regex::Regex;
use std::sync::LazyLock;

/// Regex to extract the variable name from `unused variable: \`foo\``
static UNUSED_VAR_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"unused variable: `([^`]+)`").expect("valid unused variable regex")
});

/// Regex to extract the item name from dead-code warnings.
/// Matches patterns like:
///   - "function `foo` is never used"
///   - "field `bar` is never read"
///   - "constant `BAZ` is never used"
///   - "method `qux` is never used"
static DEAD_CODE_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?:function|method|constant|struct|enum|variant|field|type alias|associated function) `([^`]+)` is never (?:used|read|constructed)")
        .expect("valid dead code regex")
});

/// Regex to extract the import path from unused-import warnings.
/// Matches: "unused import: `foo::Bar`"
static UNUSED_IMPORT_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"unused import: `([^`]+)`").expect("valid unused import regex"));

/// Result of attempting to apply a fix template.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateResult {
    /// Template matched and produced a fix. Contains the patched file content.
    Applied {
        file_path: String,
        new_content: String,
        description: String,
    },
    /// No template matched this error pattern.
    NoMatch,
}

/// Attempt to apply a deterministic fix template for the given error.
///
/// Inspects the raw error message to detect common warning patterns
/// (unused variables, dead code, unused imports) and applies a mechanical
/// fix without LLM invocation.
///
/// The `category` parameter provides the parsed `ErrorCategory` from the
/// verifier pipeline, but most warning-level fixes are classified as
/// `ErrorCategory::Other` since the error parser focuses on hard errors.
/// We therefore primarily match on the raw `error_message` text.
pub fn try_template_fix(
    _category: &ErrorCategory,
    error_message: &str,
    file_path: &str,
    file_content: &str,
    error_line: Option<usize>,
) -> TemplateResult {
    // Try each template in order of specificity
    if let result @ TemplateResult::Applied { .. } =
        fix_unused_variable(error_message, file_path, file_content, error_line)
    {
        return result;
    }

    if let result @ TemplateResult::Applied { .. } =
        fix_dead_code(error_message, file_path, file_content, error_line)
    {
        return result;
    }

    if let result @ TemplateResult::Applied { .. } =
        fix_unused_import(error_message, file_path, file_content, error_line)
    {
        return result;
    }

    TemplateResult::NoMatch
}

/// Fix unused variable warnings by prefixing the variable name with `_`.
///
/// Parses the variable name from the rustc message "unused variable: `foo`"
/// and renames it to `_foo` on the reported line.
fn fix_unused_variable(
    error_message: &str,
    file_path: &str,
    file_content: &str,
    error_line: Option<usize>,
) -> TemplateResult {
    let var_name = match UNUSED_VAR_RE.captures(error_message) {
        Some(caps) => caps[1].to_string(),
        None => return TemplateResult::NoMatch,
    };

    // Already prefixed with underscore — nothing to do
    if var_name.starts_with('_') {
        return TemplateResult::NoMatch;
    }

    let Some(line_num) = error_line else {
        return TemplateResult::NoMatch;
    };

    let lines: Vec<&str> = file_content.lines().collect();
    if line_num == 0 || line_num > lines.len() {
        return TemplateResult::NoMatch;
    }

    let target_line = lines[line_num - 1];
    let prefixed = format!("_{var_name}");

    // Replace the first occurrence of the variable name on this line.
    // We use a word-boundary-aware replacement to avoid partial matches
    // (e.g. "foobar" when fixing "foo").
    let pattern = format!(r"\b{}\b", regex::escape(&var_name));
    let re = match Regex::new(&pattern) {
        Ok(re) => re,
        Err(_) => return TemplateResult::NoMatch,
    };

    let new_line = re.replace(target_line, prefixed.as_str());
    if new_line == target_line {
        return TemplateResult::NoMatch;
    }

    let mut new_lines: Vec<String> = lines.iter().map(|l| (*l).to_string()).collect();
    new_lines[line_num - 1] = new_line.to_string();

    // Preserve trailing newline if original had one
    let mut new_content = new_lines.join("\n");
    if file_content.ends_with('\n') {
        new_content.push('\n');
    }

    TemplateResult::Applied {
        file_path: file_path.to_string(),
        new_content,
        description: format!("Prefix unused variable `{var_name}` with underscore → `_{var_name}`"),
    }
}

/// Fix dead-code warnings by adding `#[allow(dead_code)]` above the flagged item.
///
/// Parses the item name from rustc messages like "function `foo` is never used"
/// and inserts the allow attribute on the line before the item definition.
fn fix_dead_code(
    error_message: &str,
    file_path: &str,
    file_content: &str,
    error_line: Option<usize>,
) -> TemplateResult {
    let item_name = match DEAD_CODE_RE.captures(error_message) {
        Some(caps) => caps[1].to_string(),
        None => return TemplateResult::NoMatch,
    };

    let Some(line_num) = error_line else {
        return TemplateResult::NoMatch;
    };

    let lines: Vec<&str> = file_content.lines().collect();
    if line_num == 0 || line_num > lines.len() {
        return TemplateResult::NoMatch;
    }

    let target_line = lines[line_num - 1];

    // Check if there's already an #[allow(dead_code)] on the preceding line
    if line_num >= 2 && lines[line_num - 2].contains("#[allow(dead_code)]") {
        return TemplateResult::NoMatch;
    }

    // Detect indentation of the target line
    let indent: String = target_line
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();

    let allow_attr = format!("{indent}#[allow(dead_code)]");

    let mut new_lines: Vec<String> = Vec::with_capacity(lines.len() + 1);
    for (i, line) in lines.iter().enumerate() {
        if i == line_num - 1 {
            new_lines.push(allow_attr.clone());
        }
        new_lines.push((*line).to_string());
    }

    let mut new_content = new_lines.join("\n");
    if file_content.ends_with('\n') {
        new_content.push('\n');
    }

    TemplateResult::Applied {
        file_path: file_path.to_string(),
        new_content,
        description: format!("Add `#[allow(dead_code)]` above `{item_name}` at line {line_num}"),
    }
}

/// Fix unused-import warnings by removing the import line.
///
/// Parses the import path from rustc message "unused import: `foo::Bar`"
/// and removes the offending `use` line. For single-item use statements
/// this removes the entire line; for multi-item braced imports, falls back
/// to NoMatch (too complex for a deterministic template).
fn fix_unused_import(
    error_message: &str,
    file_path: &str,
    file_content: &str,
    error_line: Option<usize>,
) -> TemplateResult {
    let import_path = match UNUSED_IMPORT_RE.captures(error_message) {
        Some(caps) => caps[1].to_string(),
        None => return TemplateResult::NoMatch,
    };

    let Some(line_num) = error_line else {
        return TemplateResult::NoMatch;
    };

    let lines: Vec<&str> = file_content.lines().collect();
    if line_num == 0 || line_num > lines.len() {
        return TemplateResult::NoMatch;
    }

    let target_line = lines[line_num - 1].trim();

    // Only handle simple single-item use statements.
    // Multi-item braced imports (use foo::{A, B}) are too complex for templates.
    if target_line.contains('{') {
        return TemplateResult::NoMatch;
    }

    // Verify this line is actually a `use` statement
    if !target_line.starts_with("use ") && !target_line.starts_with("pub use ") {
        return TemplateResult::NoMatch;
    }

    let mut new_lines: Vec<String> = Vec::with_capacity(lines.len());
    for (i, line) in lines.iter().enumerate() {
        if i != line_num - 1 {
            new_lines.push((*line).to_string());
        }
    }

    let mut new_content = new_lines.join("\n");
    if file_content.ends_with('\n') {
        new_content.push('\n');
    }

    TemplateResult::Applied {
        file_path: file_path.to_string(),
        new_content,
        description: format!("Remove unused import `{import_path}` at line {line_num}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- fix_unused_variable tests ----

    #[test]
    fn test_unused_variable_basic() {
        let content = "fn main() {\n    let count = 42;\n    println!(\"hello\");\n}\n";
        let msg = "warning: unused variable: `count`\n  --> src/main.rs:2:9\n   |\n2  |     let count = 42;\n   |         ^^^^^ help: if this is intentional, prefix it with an underscore: `_count`";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, Some(2));

        match result {
            TemplateResult::Applied {
                new_content,
                description,
                ..
            } => {
                assert!(
                    new_content.contains("let _count = 42;"),
                    "Expected `_count`, got: {new_content}"
                );
                assert!(description.contains("_count"));
            }
            TemplateResult::NoMatch => panic!("Expected Applied, got NoMatch"),
        }
    }

    #[test]
    fn test_unused_variable_in_pattern() {
        let content =
            "fn process(items: Vec<i32>) {\n    for item in &items {\n        // todo\n    }\n}\n";
        let msg = "warning: unused variable: `item`\n  --> src/lib.rs:2:9";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/lib.rs", content, Some(2));

        match result {
            TemplateResult::Applied { new_content, .. } => {
                assert!(
                    new_content.contains("for _item in &items"),
                    "Expected `_item`, got: {new_content}"
                );
                // Make sure we didn't corrupt "items"
                assert!(
                    new_content.contains("&items"),
                    "Should not modify `items` (only `item`): {new_content}"
                );
            }
            TemplateResult::NoMatch => panic!("Expected Applied, got NoMatch"),
        }
    }

    #[test]
    fn test_unused_variable_already_prefixed() {
        let content = "fn main() {\n    let _x = 42;\n}\n";
        let msg = "warning: unused variable: `_x`";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, Some(2));

        assert_eq!(result, TemplateResult::NoMatch);
    }

    #[test]
    fn test_unused_variable_no_line_number() {
        let content = "fn main() {\n    let x = 42;\n}\n";
        let msg = "warning: unused variable: `x`";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, None);

        assert_eq!(result, TemplateResult::NoMatch);
    }

    // ---- fix_dead_code tests ----

    #[test]
    fn test_dead_code_function() {
        let content = "pub fn used() {}\n\nfn helper() {\n    // internal\n}\n";
        let msg = "warning: function `helper` is never used\n  --> src/lib.rs:3:4\n   |\n3  | fn helper() {\n   |    ^^^^^^";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/lib.rs", content, Some(3));

        match result {
            TemplateResult::Applied {
                new_content,
                description,
                ..
            } => {
                assert!(
                    new_content.contains("#[allow(dead_code)]\nfn helper()"),
                    "Expected allow attribute before fn, got: {new_content}"
                );
                assert!(description.contains("helper"));
                assert!(description.contains("line 3"));
            }
            TemplateResult::NoMatch => panic!("Expected Applied, got NoMatch"),
        }
    }

    #[test]
    fn test_dead_code_struct_field() {
        let content = "pub struct Config {\n    pub name: String,\n    port: u16,\n}\n";
        let msg = "warning: field `port` is never read\n  --> src/config.rs:3:5";

        let result = try_template_fix(
            &ErrorCategory::Other,
            msg,
            "src/config.rs",
            content,
            Some(3),
        );

        match result {
            TemplateResult::Applied { new_content, .. } => {
                assert!(
                    new_content.contains("    #[allow(dead_code)]\n    port: u16,"),
                    "Expected indented allow attribute, got: {new_content}"
                );
            }
            TemplateResult::NoMatch => panic!("Expected Applied, got NoMatch"),
        }
    }

    #[test]
    fn test_dead_code_already_allowed() {
        let content = "#[allow(dead_code)]\nfn helper() {}\n";
        let msg = "warning: function `helper` is never used\n  --> src/lib.rs:2:4";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/lib.rs", content, Some(2));

        assert_eq!(result, TemplateResult::NoMatch);
    }

    #[test]
    fn test_dead_code_method() {
        let content = "impl Foo {\n    fn internal(&self) -> bool {\n        true\n    }\n}\n";
        let msg = "warning: associated function `internal` is never used\n  --> src/foo.rs:2:8";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/foo.rs", content, Some(2));

        match result {
            TemplateResult::Applied { new_content, .. } => {
                assert!(
                    new_content.contains("    #[allow(dead_code)]\n    fn internal(&self)"),
                    "Expected indented allow attribute, got: {new_content}"
                );
            }
            TemplateResult::NoMatch => panic!("Expected Applied, got NoMatch"),
        }
    }

    // ---- fix_unused_import tests ----

    #[test]
    fn test_unused_import_simple() {
        let content = "use std::collections::HashMap;\nuse std::io;\n\nfn main() {\n    let m: HashMap<String, i32> = HashMap::new();\n}\n";
        let msg = "warning: unused import: `std::io`\n  --> src/main.rs:2:5";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, Some(2));

        match result {
            TemplateResult::Applied {
                new_content,
                description,
                ..
            } => {
                assert!(
                    !new_content.contains("use std::io;"),
                    "Import line should be removed: {new_content}"
                );
                assert!(
                    new_content.contains("use std::collections::HashMap;"),
                    "Other imports should remain: {new_content}"
                );
                assert!(description.contains("std::io"));
            }
            TemplateResult::NoMatch => panic!("Expected Applied, got NoMatch"),
        }
    }

    #[test]
    fn test_unused_import_braced_skipped() {
        let content = "use std::collections::{HashMap, BTreeMap};\n\nfn main() {}\n";
        let msg = "warning: unused import: `BTreeMap`\n  --> src/main.rs:1:32";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, Some(1));

        // Braced imports are too complex for templates
        assert_eq!(result, TemplateResult::NoMatch);
    }

    #[test]
    fn test_unused_import_not_a_use_line() {
        let content = "// use std::io;\nfn main() {}\n";
        let msg = "warning: unused import: `std::io`\n  --> src/main.rs:1:5";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, Some(1));

        // Commented-out line should not be removed
        assert_eq!(result, TemplateResult::NoMatch);
    }

    // ---- edge cases ----

    #[test]
    fn test_no_match_unrelated_error() {
        let content = "fn main() {\n    let x: i32 = \"hello\";\n}\n";
        let msg = "error[E0308]: mismatched types\n  --> src/main.rs:2:19";

        let result = try_template_fix(
            &ErrorCategory::TypeMismatch,
            msg,
            "src/main.rs",
            content,
            Some(2),
        );

        assert_eq!(result, TemplateResult::NoMatch);
    }

    #[test]
    fn test_line_out_of_bounds() {
        let content = "fn main() {}\n";
        let msg = "warning: unused variable: `x`";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, Some(99));

        assert_eq!(result, TemplateResult::NoMatch);
    }

    #[test]
    fn test_preserves_trailing_newline() {
        let content = "fn main() {\n    let count = 42;\n}\n";
        let msg = "warning: unused variable: `count`";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, Some(2));

        match result {
            TemplateResult::Applied { new_content, .. } => {
                assert!(
                    new_content.ends_with('\n'),
                    "Should preserve trailing newline"
                );
            }
            TemplateResult::NoMatch => panic!("Expected Applied"),
        }
    }

    #[test]
    fn test_no_trailing_newline_preserved() {
        let content = "fn main() {\n    let count = 42;\n}";
        let msg = "warning: unused variable: `count`";

        let result = try_template_fix(&ErrorCategory::Other, msg, "src/main.rs", content, Some(2));

        match result {
            TemplateResult::Applied { new_content, .. } => {
                assert!(
                    !new_content.ends_with('\n'),
                    "Should not add trailing newline when original lacks one"
                );
            }
            TemplateResult::NoMatch => panic!("Expected Applied"),
        }
    }
}
