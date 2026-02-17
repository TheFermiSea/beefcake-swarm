//! Bridge to the NotebookLM CLI (`nlm`) for knowledge base queries.
//!
//! Follows the same pattern as `BeadsBridge`: wraps a binary-only CLI tool
//! via `std::process::Command` with graceful degradation when unavailable.

use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::{info, warn};

/// Number of retries on authentication errors.
const AUTH_RETRY_COUNT: u32 = 1;

/// Delay between auth retries (seconds) — gives CSRF refresh time to settle.
const AUTH_RETRY_DELAY_SECS: u64 = 2;

/// A single notebook entry in the registry.
#[derive(Debug, Clone, Deserialize)]
pub struct NotebookEntry {
    pub id: String,
    pub role: String,
    #[serde(default)]
    pub auto_query: bool,
    #[serde(default)]
    pub auto_update: bool,
}

/// Registry mapping roles to notebook IDs, parsed from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct NotebookRegistry {
    pub notebooks: HashMap<String, NotebookEntry>,
}

impl NotebookRegistry {
    /// Load the registry from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content =
            std::fs::read_to_string(path).context(format!("Failed to read {}", path.display()))?;
        let registry: NotebookRegistry =
            toml::from_str(&content).context("Failed to parse notebook registry TOML")?;
        Ok(registry)
    }

    /// Get the notebook ID for a given role, if it exists and has a non-empty ID.
    pub fn id_for_role(&self, role: &str) -> Option<&str> {
        self.notebooks
            .get(role)
            .map(|e| e.id.as_str())
            .filter(|id| !id.is_empty())
    }

    /// Get an entry by role.
    pub fn entry_for_role(&self, role: &str) -> Option<&NotebookEntry> {
        self.notebooks.get(role)
    }

    /// List all roles that have auto_query enabled and a non-empty ID.
    pub fn auto_query_roles(&self) -> Vec<&str> {
        self.notebooks
            .values()
            .filter(|e| e.auto_query && !e.id.is_empty())
            .map(|e| e.role.as_str())
            .collect()
    }
}

/// Abstraction over knowledge base backends.
///
/// `NotebookBridge` implements this for the real `nlm` CLI.
/// Tests can provide a mock implementation.
pub trait KnowledgeBase: Send + Sync {
    /// Query a notebook by role with a natural language question.
    /// Returns the response text, or empty string if unavailable.
    fn query(&self, role: &str, question: &str) -> Result<String>;

    /// Add a text source to a notebook by role.
    fn add_source_text(&self, role: &str, title: &str, content: &str) -> Result<()>;

    /// Add a file source to a notebook by role.
    fn add_source_file(&self, role: &str, file_path: &str) -> Result<()>;

    /// Check if the knowledge base CLI is available.
    fn is_available(&self) -> bool;
}

/// Bridge to the `nlm` CLI binary (NotebookLM CLI).
///
/// The binary name is read from `SWARM_NLM_BIN` env var, defaulting to `"nlm"`.
pub struct NotebookBridge {
    bin: String,
    registry: NotebookRegistry,
}

impl NotebookBridge {
    /// Create a bridge from a registry file path.
    pub fn from_registry(path: &Path) -> Result<Self> {
        let registry = NotebookRegistry::from_file(path)?;
        let bin = std::env::var("SWARM_NLM_BIN").unwrap_or_else(|_| "nlm".into());
        Ok(Self { bin, registry })
    }

    /// Create a bridge with an explicit registry and binary name.
    pub fn new(registry: NotebookRegistry, bin: String) -> Self {
        Self { bin, registry }
    }

    /// Get a reference to the underlying registry.
    pub fn registry(&self) -> &NotebookRegistry {
        &self.registry
    }

    /// Run an `nlm` command, returning stdout on success.
    ///
    /// On authentication errors (RPC Error 16 / "Authentication expired"),
    /// retries once after a short delay. The CLI's internal 3-layer recovery
    /// (CSRF refresh → disk reload → headless Chrome) runs on each attempt,
    /// so a retry gives the CSRF token refresh a second chance after the
    /// ~20-minute session timeout.
    fn run_command(&self, args: &[&str]) -> Result<String> {
        let cmd_label = format!("{} {}", self.bin, args.first().unwrap_or(&""));

        for attempt in 0..=AUTH_RETRY_COUNT {
            let output = Command::new(&self.bin)
                .args(args)
                .output()
                .context(format!("Failed to run `{cmd_label}`"))?;

            if output.status.success() {
                return Ok(String::from_utf8_lossy(&output.stdout).trim().to_string());
            }

            let stderr = String::from_utf8_lossy(&output.stderr);

            // Check if this is an auth error worth retrying
            let is_auth_error = stderr.contains("Authentication expired")
                || stderr.contains("RPC Error 16")
                || stderr.contains("AuthenticationError");

            if is_auth_error && attempt < AUTH_RETRY_COUNT {
                warn!(
                    cmd = %cmd_label,
                    attempt = attempt + 1,
                    "NotebookLM auth expired, retrying after {AUTH_RETRY_DELAY_SECS}s"
                );
                std::thread::sleep(std::time::Duration::from_secs(AUTH_RETRY_DELAY_SECS));
                continue;
            }

            anyhow::bail!("{cmd_label} failed: {stderr}");
        }

        unreachable!()
    }
}

impl KnowledgeBase for NotebookBridge {
    fn query(&self, role: &str, question: &str) -> Result<String> {
        let notebook_id = match self.registry.id_for_role(role) {
            Some(id) => id,
            None => {
                warn!(role, "No notebook ID configured for role — returning empty");
                return Ok(String::new());
            }
        };

        info!(role, "Querying NotebookLM");
        self.run_command(&["query", "notebook", notebook_id, question])
    }

    fn add_source_text(&self, role: &str, title: &str, content: &str) -> Result<()> {
        let notebook_id = match self.registry.id_for_role(role) {
            Some(id) => id,
            None => {
                warn!(role, "No notebook ID configured for role — skipping upload");
                return Ok(());
            }
        };

        self.run_command(&[
            "source",
            "add",
            notebook_id,
            "--text",
            content,
            "--title",
            title,
        ])
        .map(|_| ())
    }

    fn add_source_file(&self, role: &str, file_path: &str) -> Result<()> {
        let notebook_id = match self.registry.id_for_role(role) {
            Some(id) => id,
            None => {
                warn!(role, "No notebook ID configured for role — skipping upload");
                return Ok(());
            }
        };

        self.run_command(&["source", "add", notebook_id, "--file", file_path])
            .map(|_| ())
    }

    fn is_available(&self) -> bool {
        Command::new(&self.bin)
            .args(["--help"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

/// A no-op knowledge base for when NotebookLM is unavailable.
///
/// All queries return empty strings, all writes succeed silently.
pub struct NoOpKnowledgeBase;

impl KnowledgeBase for NoOpKnowledgeBase {
    fn query(&self, _role: &str, _question: &str) -> Result<String> {
        Ok(String::new())
    }

    fn add_source_text(&self, _role: &str, _title: &str, _content: &str) -> Result<()> {
        Ok(())
    }

    fn add_source_file(&self, _role: &str, _file_path: &str) -> Result<()> {
        Ok(())
    }

    fn is_available(&self) -> bool {
        false
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Mock knowledge base for testing.
    pub struct MockKnowledgeBase {
        pub responses: Mutex<HashMap<String, String>>,
        pub captured_queries: Mutex<Vec<(String, String)>>,
        pub captured_uploads: Mutex<Vec<(String, String, String)>>,
    }

    impl MockKnowledgeBase {
        pub fn new() -> Self {
            Self {
                responses: Mutex::new(HashMap::new()),
                captured_queries: Mutex::new(Vec::new()),
                captured_uploads: Mutex::new(Vec::new()),
            }
        }

        pub fn with_response(self, role: &str, response: &str) -> Self {
            self.responses
                .lock()
                .unwrap()
                .insert(role.to_string(), response.to_string());
            self
        }
    }

    impl KnowledgeBase for MockKnowledgeBase {
        fn query(&self, role: &str, question: &str) -> Result<String> {
            self.captured_queries
                .lock()
                .unwrap()
                .push((role.to_string(), question.to_string()));

            Ok(self
                .responses
                .lock()
                .unwrap()
                .get(role)
                .cloned()
                .unwrap_or_default())
        }

        fn add_source_text(&self, role: &str, title: &str, content: &str) -> Result<()> {
            self.captured_uploads.lock().unwrap().push((
                role.to_string(),
                title.to_string(),
                content.to_string(),
            ));
            Ok(())
        }

        fn add_source_file(&self, _role: &str, _file_path: &str) -> Result<()> {
            Ok(())
        }

        fn is_available(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_registry_parse() {
        let toml_str = r#"
[notebooks.project_brain]
id = "abc123"
role = "project_brain"
auto_query = true
auto_update = true

[notebooks.debugging_kb]
id = "def456"
role = "debugging_kb"
auto_query = true
auto_update = false

[notebooks.codebase]
id = ""
role = "codebase"
auto_query = false
auto_update = false
"#;
        let registry: NotebookRegistry = toml::from_str(toml_str).unwrap();

        assert_eq!(registry.id_for_role("project_brain"), Some("abc123"));
        assert_eq!(registry.id_for_role("debugging_kb"), Some("def456"));
        assert_eq!(registry.id_for_role("codebase"), None); // empty ID
        assert_eq!(registry.id_for_role("nonexistent"), None);
    }

    #[test]
    fn test_registry_auto_query_roles() {
        let toml_str = r#"
[notebooks.project_brain]
id = "abc123"
role = "project_brain"
auto_query = true
auto_update = true

[notebooks.debugging_kb]
id = "def456"
role = "debugging_kb"
auto_query = true
auto_update = false

[notebooks.codebase]
id = ""
role = "codebase"
auto_query = true
auto_update = false
"#;
        let registry: NotebookRegistry = toml::from_str(toml_str).unwrap();
        let auto_roles = registry.auto_query_roles();

        // codebase has auto_query=true but empty ID, so excluded
        assert_eq!(auto_roles.len(), 2);
        assert!(auto_roles.contains(&"project_brain"));
        assert!(auto_roles.contains(&"debugging_kb"));
    }

    #[test]
    fn test_mock_knowledge_base() {
        let mock = MockKnowledgeBase::new()
            .with_response("project_brain", "The escalation ladder has 4 tiers.");

        let result = mock
            .query("project_brain", "What is the escalation ladder?")
            .unwrap();
        assert_eq!(result, "The escalation ladder has 4 tiers.");

        let result = mock.query("debugging_kb", "How to fix E0382?").unwrap();
        assert_eq!(result, ""); // No response configured

        let queries = mock.captured_queries.lock().unwrap();
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0].0, "project_brain");
    }

    #[test]
    fn test_noop_knowledge_base() {
        let noop = NoOpKnowledgeBase;
        assert!(!noop.is_available());
        assert_eq!(noop.query("any", "question").unwrap(), "");
        assert!(noop.add_source_text("any", "title", "content").is_ok());
    }

    #[test]
    fn test_registry_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("registry.toml");
        std::fs::write(
            &path,
            r#"
[notebooks.brain]
id = "test-id"
role = "brain"
auto_query = true
auto_update = false
"#,
        )
        .unwrap();

        let registry = NotebookRegistry::from_file(&path).unwrap();
        assert_eq!(registry.id_for_role("brain"), Some("test-id"));
    }
}
