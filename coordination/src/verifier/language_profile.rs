//! Language Profile — target-repo configuration for multi-language support.
//!
//! Each target repo provides a `.swarm/profile.toml` that defines quality gates,
//! source extensions, integration files, and auto-fix commands. This decouples the
//! verifier from any specific language (Rust, Python, TypeScript, etc.).
//!
//! When `language = "rust"` (or no profile exists), the built-in Rust verifier is
//! used unchanged. For all other languages, the `ScriptVerifier` runs the shell
//! commands defined in `[[gates]]`.
//!
//! Design inspired by:
//! - SWE-agent (Princeton): shell-native polyglot design
//! - Open SWE (LangChain): AGENTS.md convention files

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};

/// A single quality gate defined as a shell command.
///
/// Gates run sequentially. If a `blocking` gate fails and the pipeline is in
/// fail-fast mode, subsequent gates are skipped.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateSpec {
    /// Human-readable gate name (e.g., "lint", "typecheck", "test")
    pub name: String,
    /// Command to execute (e.g., "ruff", "pytest", "cargo")
    pub command: String,
    /// Arguments to pass to the command
    #[serde(default)]
    pub args: Vec<String>,
    /// Whether this gate blocks the pipeline on failure (default: true)
    #[serde(default = "default_true")]
    pub blocking: bool,
    /// Per-gate timeout in seconds (default: 300)
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
}

fn default_true() -> bool {
    true
}

fn default_timeout() -> u64 {
    300
}

/// An auto-fix command that runs before LLM delegation (the "Janitor" layer).
///
/// These commands attempt to fix trivial issues (formatting, simple lint fixes)
/// without involving the LLM, saving inference time and cost.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoFixCommand {
    /// Command to execute (e.g., "black", "ruff")
    pub command: String,
    /// Arguments (e.g., ["check", "--fix", "cflibs/"])
    #[serde(default)]
    pub args: Vec<String>,
}

/// Language-specific project profile loaded from `.swarm/profile.toml`.
///
/// This is the central abstraction that enables multi-language support.
/// The orchestrator loads this at the start of each issue and threads it
/// through the verifier, auto-fix, file targeting, and prompt systems.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LanguageProfile {
    /// Programming language (e.g., "rust", "python", "typescript")
    pub language: String,

    /// File extensions for source files (e.g., [".py", ".pyi"])
    #[serde(default)]
    pub source_extensions: Vec<String>,

    /// Package manifest file (e.g., "pyproject.toml", "Cargo.toml", "package.json")
    #[serde(default)]
    pub package_manifest: String,

    /// Files that should appear in at most one subtask (e.g., ["pyproject.toml", "__init__.py"])
    #[serde(default)]
    pub integration_files: Vec<String>,

    /// Ordered list of quality gates to run
    #[serde(default)]
    pub gates: Vec<GateSpec>,

    /// Auto-fix commands to run before LLM delegation
    #[serde(default)]
    pub auto_fix: Vec<AutoFixCommand>,

    /// Whether to run all gates even if earlier ones fail (default: false = fail-fast)
    #[serde(default)]
    pub comprehensive: bool,

    /// Maximum stderr bytes to capture per gate (default: 4096)
    #[serde(default = "default_stderr_max")]
    pub stderr_max_bytes: usize,
}

fn default_stderr_max() -> usize {
    4096
}

impl LanguageProfile {
    /// Load a language profile from `{repo_root}/.swarm/profile.toml`.
    ///
    /// Returns `None` if the file doesn't exist (caller should fall back to
    /// built-in Rust verifier behavior).
    pub fn load(repo_root: &Path) -> Option<Self> {
        let profile_path = repo_root.join(".swarm").join("profile.toml");
        Self::load_from(&profile_path)
    }

    /// Load from a specific file path.
    pub fn load_from(path: &Path) -> Option<Self> {
        if !path.exists() {
            debug!(path = %path.display(), "No .swarm/profile.toml found — using built-in Rust verifier");
            return None;
        }

        match std::fs::read_to_string(path) {
            Ok(content) => match toml::from_str::<LanguageProfile>(&content) {
                Ok(profile) => {
                    info!(
                        language = %profile.language,
                        gates = profile.gates.len(),
                        auto_fix = profile.auto_fix.len(),
                        path = %path.display(),
                        "Loaded language profile"
                    );
                    Some(profile)
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        path = %path.display(),
                        "Failed to parse .swarm/profile.toml — falling back to Rust verifier"
                    );
                    None
                }
            },
            Err(e) => {
                warn!(
                    error = %e,
                    path = %path.display(),
                    "Failed to read .swarm/profile.toml — falling back to Rust verifier"
                );
                None
            }
        }
    }

    /// Whether this profile uses the built-in Rust verifier.
    pub fn is_rust(&self) -> bool {
        self.language.eq_ignore_ascii_case("rust")
    }

    /// Get the profile path for a given repo root.
    pub fn profile_path(repo_root: &Path) -> PathBuf {
        repo_root.join(".swarm").join("profile.toml")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_python_profile() {
        let toml_str = r#"
language = "python"
source_extensions = [".py", ".pyi"]
package_manifest = "pyproject.toml"
integration_files = ["pyproject.toml", "conftest.py", "__init__.py"]

[[gates]]
name = "lint"
command = "ruff"
args = ["check", "cflibs/", "tests/"]
blocking = true
timeout_secs = 120

[[gates]]
name = "format"
command = "black"
args = ["--check", "cflibs/"]

[[gates]]
name = "typecheck"
command = "mypy"
args = ["cflibs/"]
blocking = false
timeout_secs = 300

[[gates]]
name = "test"
command = "pytest"
args = ["tests/", "-x", "-q"]
timeout_secs = 600

[[auto_fix]]
command = "black"
args = ["cflibs/"]

[[auto_fix]]
command = "ruff"
args = ["check", "--fix", "cflibs/"]
"#;

        let profile: LanguageProfile = toml::from_str(toml_str).unwrap();
        assert_eq!(profile.language, "python");
        assert!(!profile.is_rust());
        assert_eq!(profile.source_extensions, vec![".py", ".pyi"]);
        assert_eq!(profile.gates.len(), 4);
        assert_eq!(profile.gates[0].name, "lint");
        assert_eq!(profile.gates[0].command, "ruff");
        assert!(profile.gates[0].blocking);
        assert_eq!(profile.gates[2].name, "typecheck");
        assert!(!profile.gates[2].blocking);
        assert_eq!(profile.auto_fix.len(), 2);
        assert_eq!(profile.package_manifest, "pyproject.toml");
    }

    #[test]
    fn test_deserialize_rust_profile() {
        let toml_str = r#"
language = "rust"
source_extensions = [".rs"]
package_manifest = "Cargo.toml"
integration_files = ["Cargo.toml", "Cargo.lock", "mod.rs", "lib.rs", "main.rs"]
"#;

        let profile: LanguageProfile = toml::from_str(toml_str).unwrap();
        assert!(profile.is_rust());
        assert!(profile.gates.is_empty());
    }

    #[test]
    fn test_load_nonexistent_returns_none() {
        let result = LanguageProfile::load(Path::new("/nonexistent/path"));
        assert!(result.is_none());
    }

    #[test]
    fn test_defaults() {
        let toml_str = r#"
language = "go"
"#;
        let profile: LanguageProfile = toml::from_str(toml_str).unwrap();
        assert!(profile.source_extensions.is_empty());
        assert!(profile.gates.is_empty());
        assert!(profile.auto_fix.is_empty());
        assert!(!profile.comprehensive);
        assert_eq!(profile.stderr_max_bytes, 4096);
    }

    #[test]
    fn test_gate_defaults() {
        let toml_str = r#"
language = "python"

[[gates]]
name = "test"
command = "pytest"
"#;
        let profile: LanguageProfile = toml::from_str(toml_str).unwrap();
        assert_eq!(profile.gates.len(), 1);
        assert!(profile.gates[0].blocking); // default true
        assert_eq!(profile.gates[0].timeout_secs, 300); // default
        assert!(profile.gates[0].args.is_empty()); // default empty
    }
}
