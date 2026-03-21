//! Rig tool wrapper around the deterministic verifier pipeline.
//!
//! This exposes `coordination::verifier::Verifier` as a rig `Tool` so that
//! the Manager agent can request quality-gate checks via tool calling.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::ToolError;

#[derive(Deserialize)]
pub struct RunVerifierArgs {
    /// Which gates to run: "quick" (fmt+check), "compile" (clippy+check), or "full" (all).
    pub mode: Option<String>,
}

/// Run the quality gate pipeline and return a structured report.
///
/// For Rust targets: runs cargo fmt, clippy, check, test.
/// For non-Rust targets: runs shell commands from `.swarm/profile.toml`.
pub struct RunVerifierTool {
    pub working_dir: PathBuf,
    /// Scope cargo commands to specific packages (empty = whole workspace).
    pub packages: Vec<String>,
    /// Language profile for non-Rust targets (None = use built-in Rust verifier).
    pub language_profile: Option<coordination::LanguageProfile>,
}

impl RunVerifierTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
            packages: Vec::new(),
            language_profile: None,
        }
    }

    pub fn with_packages(mut self, packages: Vec<String>) -> Self {
        self.packages = packages;
        self
    }

    pub fn with_language_profile(mut self, profile: Option<coordination::LanguageProfile>) -> Self {
        self.language_profile = profile;
        self
    }
}

impl Tool for RunVerifierTool {
    const NAME: &'static str = "run_verifier";
    type Error = ToolError;
    type Args = RunVerifierArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let desc = if let Some(ref profile) = self.language_profile {
            if !profile.is_rust() {
                let gate_names: Vec<&str> = profile.gates.iter().map(|g| g.name.as_str()).collect();
                format!(
                    "Run the {} quality gate pipeline: {}. Returns a structured pass/fail report.",
                    profile.language,
                    gate_names.join(", ")
                )
            } else {
                "Run the Rust quality gate pipeline: cargo fmt, clippy, check, test. \
                 Returns a structured pass/fail report with error categories."
                    .to_string()
            }
        } else {
            "Run the Rust quality gate pipeline: cargo fmt, clippy, check, test. \
             Returns a structured pass/fail report with error categories."
                .to_string()
        };
        ToolDefinition {
            name: "run_verifier".into(),
            description: desc.into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["quick", "compile", "full"],
                        "description": "Gate selection: quick (fmt+check), compile (clippy+check), full (all gates). Defaults to full."
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        use coordination::verifier::{ScriptVerifier, Verifier, VerifierConfig};

        // Dispatch to ScriptVerifier for non-Rust targets
        let report = if let Some(ref profile) = self.language_profile {
            if !profile.is_rust() {
                let sv = ScriptVerifier::new(&self.working_dir, profile.clone());
                sv.run_pipeline().await
            } else {
                let mut config = match args.mode.as_deref() {
                    Some("quick") => VerifierConfig::quick(),
                    Some("compile") => VerifierConfig::compile_only(),
                    _ => VerifierConfig::default(),
                };
                config.packages = self.packages.clone();
                Verifier::new(&self.working_dir, config).run_pipeline().await
            }
        } else {
            let mut config = match args.mode.as_deref() {
                Some("quick") => VerifierConfig::quick(),
                Some("compile") => VerifierConfig::compile_only(),
                _ => VerifierConfig::default(),
            };
            config.packages = self.packages.clone();
            Verifier::new(&self.working_dir, config).run_pipeline().await
        };

        // Return the summary as a string for the agent to reason about
        let mut output = String::new();
        output.push_str("## Verifier Report\n\n");
        output.push_str(&format!(
            "**Result:** {}\n",
            if report.all_green {
                "ALL GREEN"
            } else {
                "FAILED"
            }
        ));
        output.push_str(&format!(
            "**Gates:** {}/{} passed\n",
            report.gates_passed, report.gates_total
        ));
        output.push_str(&format!("**Duration:** {}ms\n\n", report.total_duration_ms));

        if !report.failure_signals.is_empty() {
            output.push_str("### Errors\n\n");
            for signal in &report.failure_signals {
                output.push_str(&format!(
                    "- **{}** ({}): {}\n",
                    signal.category,
                    signal.code.as_deref().unwrap_or("?"),
                    signal.message
                ));
                if let Some(file) = &signal.file {
                    output.push_str(&format!("  File: {}:{}\n", file, signal.line.unwrap_or(0)));
                }
            }
        }

        if !report.unique_error_categories().is_empty() {
            output.push_str("\n### Error Categories\n\n");
            for cat in report.unique_error_categories() {
                output.push_str(&format!("- {cat:?}\n"));
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_run_verifier_args_deserialize_full() {
        let json = r#"{"mode": "full"}"#;
        let args: RunVerifierArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.mode.as_deref(), Some("full"));
    }

    #[test]
    fn test_run_verifier_args_deserialize_quick() {
        let json = r#"{"mode": "quick"}"#;
        let args: RunVerifierArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.mode.as_deref(), Some("quick"));
    }

    #[test]
    fn test_run_verifier_args_deserialize_none() {
        let json = r#"{}"#;
        let args: RunVerifierArgs = serde_json::from_str(json).unwrap();
        assert!(args.mode.is_none());
    }

    #[test]
    fn test_run_verifier_tool_new() {
        let tool = RunVerifierTool::new(Path::new("/tmp/test"));
        assert_eq!(tool.working_dir, PathBuf::from("/tmp/test"));
        assert!(tool.packages.is_empty());
    }

    #[test]
    fn test_run_verifier_tool_with_packages() {
        let tool = RunVerifierTool::new(Path::new("/tmp/test"))
            .with_packages(vec!["swarm-agents".to_string(), "coordination".to_string()]);
        assert_eq!(tool.packages.len(), 2);
        assert_eq!(tool.packages[0], "swarm-agents");
        assert_eq!(tool.packages[1], "coordination");
    }

    #[test]
    fn test_run_verifier_args_invalid_mode_still_deserializes() {
        // Invalid mode value still deserializes — validation happens at call time
        let json = r#"{"mode": "invalid"}"#;
        let args: RunVerifierArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.mode.as_deref(), Some("invalid"));
    }
}
