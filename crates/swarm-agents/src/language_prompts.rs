//! Language-adaptive prompt sections for multi-language swarm support.
//!
//! When a `.swarm/profile.toml` is loaded with a non-Rust language, these
//! functions provide language-appropriate expertise blocks, environment
//! descriptions, and verifier instructions to replace the Rust-specific
//! defaults in `prompts.rs`.

/// Language-specific expertise block for the specialist coder prompt.
///
/// Replaces the "Rust Expertise" section in RUST_CODER_PREAMBLE.
pub fn expertise_block(language: &str) -> &'static str {
    match language.to_lowercase().as_str() {
        "python" => {
            "\
## Python Expertise
- Type errors: check function signatures, return types, and Optional/None handling.
- Import errors: verify module paths, check __init__.py files, resolve circular imports.
- Indentation: Python is whitespace-sensitive — preserve exact indentation levels.
- Testing: pytest conventions — test functions start with test_, fixtures via conftest.py.
- Common patterns: use isinstance() for type checks, handle None explicitly, prefer f-strings."
        }

        "typescript" | "javascript" => {
            "\
## TypeScript Expertise
- Type errors (TS2345, TS2322): trace type inference, check generic constraints.
- Module resolution: verify import paths, check tsconfig.json paths/aliases.
- Null safety: handle undefined/null with optional chaining (?.) and nullish coalescing (??).
- Async patterns: await placement, Promise handling, error propagation.
- Testing: Jest/Vitest conventions — describe/it/expect, mock patterns."
        }

        "go" | "golang" => {
            "\
## Go Expertise
- Interface compliance: verify struct implements required methods.
- Error handling: always check returned errors, use fmt.Errorf for wrapping.
- Goroutine safety: check for data races, use sync.Mutex or channels.
- Import management: unused imports are compile errors in Go.
- Testing: table-driven tests, t.Run for subtests, testify assertions."
        }

        _ => {
            "\
## General Coding Expertise
- Read error messages carefully and trace them to root causes.
- Prefer minimal, targeted fixes over broad refactoring.
- Respect existing code style and conventions."
        }
    }
}

/// Language-specific environment description.
///
/// Replaces "Do NOT run cargo check/test yourself" with the equivalent.
pub fn environment_block(language: &str) -> String {
    let verifier_note = match language.to_lowercase().as_str() {
        "python" => "lint (ruff), typecheck (mypy), and tests (pytest)",
        "typescript" | "javascript" => "lint (eslint), typecheck (tsc), and tests (vitest/jest)",
        "go" | "golang" => "vet (go vet), lint (golangci-lint), and tests (go test)",
        "rust" => "fmt (cargo fmt), clippy, check, and tests (cargo test)",
        _ => "quality gates defined in .swarm/profile.toml",
    };
    format!(
        "Isolated git worktree. The verifier runs {verifier_note} automatically after you return. \
         Do NOT run the verifier yourself. Do NOT commit."
    )
}

/// Language-specific file extensions for search tool hints.
pub fn search_hints(language: &str) -> &'static str {
    match language.to_lowercase().as_str() {
        "python" => "Use `colgrep --include=\"*.py\" \"query\"` to search Python files.",
        "typescript" | "javascript" => {
            "Use `colgrep --include=\"*.{ts,tsx}\" \"query\"` to search TypeScript files."
        }
        "go" | "golang" => "Use `colgrep --include=\"*.go\" \"query\"` to search Go files.",
        _ => "Use `colgrep \"query\"` to search source files.",
    }
}

/// Default source file extensions for a language.
pub fn source_extensions(language: &str) -> Vec<&'static str> {
    match language.to_lowercase().as_str() {
        "python" => vec![".py", ".pyi"],
        "typescript" => vec![".ts", ".tsx", ".js", ".jsx"],
        "javascript" => vec![".js", ".jsx", ".mjs"],
        "go" | "golang" => vec![".go"],
        "rust" => vec![".rs"],
        _ => vec![],
    }
}

/// Package manifest filenames for a language.
pub fn package_manifests(language: &str) -> Vec<&'static str> {
    match language.to_lowercase().as_str() {
        "python" => vec![
            "pyproject.toml",
            "setup.py",
            "setup.cfg",
            "requirements.txt",
        ],
        "typescript" | "javascript" => vec!["package.json", "tsconfig.json"],
        "go" | "golang" => vec!["go.mod", "go.sum"],
        "rust" => vec!["Cargo.toml", "Cargo.lock"],
        _ => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expertise_block_returns_language_specific() {
        assert!(expertise_block("python").contains("pytest"));
        assert!(expertise_block("typescript").contains("TS2345"));
        assert!(expertise_block("go").contains("Goroutine"));
        assert!(expertise_block("rust").contains("General")); // falls through to default
        assert!(expertise_block("unknown").contains("General"));
    }

    #[test]
    fn test_environment_block_mentions_tools() {
        assert!(environment_block("python").contains("ruff"));
        assert!(environment_block("typescript").contains("tsc"));
        assert!(environment_block("go").contains("go vet"));
    }

    #[test]
    fn test_source_extensions() {
        assert!(source_extensions("python").contains(&".py"));
        assert!(source_extensions("go").contains(&".go"));
        assert!(source_extensions("unknown").is_empty());
    }
}
