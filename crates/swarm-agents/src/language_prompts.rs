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

/// Language-specific examples for prompt template substitution.
///
/// These replace Rust-specific references in built-in prompt constants
/// (GENERAL_CODER_PREAMBLE, CLOUD_MANAGER_PREAMBLE, LOCAL_MANAGER_PREAMBLE)
/// when the target repo uses a different language.
pub struct LanguageExamples {
    /// Verification commands (e.g., "cargo fmt, clippy, check, test" vs "ruff, pytest")
    pub verification_commands: &'static str,
    /// Example file path (e.g., "src/example.rs" vs "src/example.py")
    pub example_file_path: &'static str,
    /// Crate/workspace scope note (empty for non-Rust; only Rust has workspace crates)
    pub crate_scope_note: &'static str,
    /// Error focus description (e.g., "Rust compilation errors" vs "Python errors")
    pub error_focus: &'static str,
    /// Verifier tool description (e.g., "cargo fmt → clippy → check → test")
    pub verifier_pipeline: &'static str,
    /// "Do NOT run X yourself" instruction
    pub do_not_run: &'static str,
}

impl LanguageExamples {
    /// Return language-appropriate examples for prompt templates.
    ///
    /// When `lang` is "rust" or unrecognized, returns Rust defaults so that
    /// existing behavior is preserved.
    pub fn for_language(lang: &str) -> Self {
        match lang.to_lowercase().as_str() {
            "python" => Self {
                verification_commands: "ruff check, ruff format, mypy, pytest",
                example_file_path: "src/example.py",
                crate_scope_note: "",
                error_focus: "Python errors (import, type, runtime)",
                verifier_pipeline: "ruff check → ruff format → mypy → pytest",
                do_not_run: "Do NOT run pytest/mypy yourself. Do NOT commit.",
            },
            "typescript" | "javascript" => Self {
                verification_commands: "eslint, tsc, vitest/jest",
                example_file_path: "src/example.ts",
                crate_scope_note: "",
                error_focus: "TypeScript/JavaScript errors (type, module, runtime)",
                verifier_pipeline: "eslint → tsc → vitest",
                do_not_run: "Do NOT run tsc/vitest yourself. Do NOT commit.",
            },
            "go" | "golang" => Self {
                verification_commands: "go vet, golangci-lint, go test",
                example_file_path: "cmd/example.go",
                crate_scope_note: "",
                error_focus: "Go errors (type, interface, concurrency)",
                verifier_pipeline: "go vet → golangci-lint → go test",
                do_not_run: "Do NOT run go test yourself. Do NOT commit.",
            },
            // Rust or unrecognized — preserve existing behavior
            _ => Self {
                verification_commands: "cargo fmt, clippy, check, test",
                example_file_path: "crates/swarm-agents/src/example.rs",
                crate_scope_note: "When fixes span multiple workspace crates \
                    (e.g. `coordination/` and `crates/`), delegate ONE CRATE AT A TIME:\n\
                    - Fix the provider crate first (where the type/trait is defined), \
                    verify, then fix consumers.\n\
                    - Each delegation: at most 5 files. For larger changes, split into \
                    sequential delegations.\n\
                    - Run the verifier between each crate's delegation.\n\
                    - Never ask a single worker to modify files in two different workspace crates.",
                error_focus: "Rust compilation errors",
                verifier_pipeline: "cargo fmt → clippy → check → test",
                do_not_run: "Do NOT run cargo check/test yourself. Do NOT commit.",
            },
        }
    }
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

    #[test]
    fn test_language_examples_python() {
        let ex = LanguageExamples::for_language("python");
        assert!(ex.verification_commands.contains("pytest"));
        assert!(ex.example_file_path.ends_with(".py"));
        assert!(ex.crate_scope_note.is_empty());
        assert!(ex.error_focus.contains("Python"));
    }

    #[test]
    fn test_language_examples_rust_default() {
        let ex = LanguageExamples::for_language("rust");
        assert!(ex.verification_commands.contains("cargo"));
        assert!(ex.example_file_path.ends_with(".rs"));
        assert!(!ex.crate_scope_note.is_empty());
        assert!(ex.error_focus.contains("Rust"));
    }

    #[test]
    fn test_language_examples_unknown_falls_to_rust() {
        let ex = LanguageExamples::for_language("unknown");
        assert!(ex.verification_commands.contains("cargo"));
    }

    #[test]
    fn test_language_examples_go() {
        let ex = LanguageExamples::for_language("go");
        assert!(ex.verification_commands.contains("go vet"));
        assert!(ex.example_file_path.ends_with(".go"));
        assert!(ex.crate_scope_note.is_empty());
    }

    #[test]
    fn test_language_examples_typescript() {
        let ex = LanguageExamples::for_language("typescript");
        assert!(ex.verification_commands.contains("eslint"));
        assert!(ex.example_file_path.ends_with(".ts"));
    }
}
