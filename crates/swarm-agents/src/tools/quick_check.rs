//! Lightweight compilation check tool for incremental verification.
//!
//! Runs `cargo check` only — no formatting, linting, or tests.
//! Completes in ~5s vs ~30s for the full verifier pipeline.
//!
//! This gives the LLM fast feedback on whether code compiles without
//! waiting for the full quality gate pipeline. The full `run_verifier`
//! remains the authoritative quality gate; this is a quick pre-check.

use std::path::{Path, PathBuf};
use std::process::Command;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::debug;

/// Fast compilation-only check. Runs `cargo check` and returns
/// pass/fail with error messages.
pub struct QuickCheckTool {
    working_dir: PathBuf,
    packages: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct QuickCheckArgs {
    /// Optional: check only specific packages (comma-separated).
    /// If empty, checks the entire workspace.
    pub packages: Option<String>,
}

impl QuickCheckTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
            packages: Vec::new(),
        }
    }

    pub fn with_packages(mut self, packages: Vec<String>) -> Self {
        self.packages = packages;
        self
    }
}

impl Tool for QuickCheckTool {
    const NAME: &'static str = "quick_check";

    type Error = std::convert::Infallible;
    type Args = QuickCheckArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "quick_check".into(),
            description: "Fast compilation check (cargo check only — no fmt, clippy, or tests). \
                          Returns PASS or FAIL with error messages. \
                          Use this for quick feedback during editing; use run_verifier for the full quality gate."
                .into(),
            parameters: serde_json::to_value(schemars::schema_for!(QuickCheckArgs))
                .unwrap_or_default(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let start = std::time::Instant::now();

        // Determine packages to check.
        let packages: Vec<String> = args
            .packages
            .map(|p| p.split(',').map(|s| s.trim().to_string()).collect())
            .unwrap_or_else(|| self.packages.clone());

        let mut cmd = Command::new("cargo");
        cmd.arg("check")
            .arg("--message-format=short")
            .current_dir(&self.working_dir);

        if !packages.is_empty() {
            for pkg in &packages {
                cmd.args(["-p", pkg]);
            }
        } else {
            cmd.arg("--workspace");
        }

        debug!(
            dir = %self.working_dir.display(),
            packages = ?packages,
            "Running quick_check (cargo check)"
        );

        let output = cmd.output();
        let elapsed_ms = start.elapsed().as_millis();

        match output {
            Ok(out) => {
                let _stdout = String::from_utf8_lossy(&out.stdout);
                let stderr = String::from_utf8_lossy(&out.stderr);

                if out.status.success() {
                    Ok(format!(
                        "PASS — cargo check succeeded ({}ms)\n\
                         No compilation errors.",
                        elapsed_ms
                    ))
                } else {
                    // Extract error lines from stderr (cargo check writes errors there).
                    let errors: Vec<&str> = stderr
                        .lines()
                        .filter(|l| l.contains("error[") || l.contains("error:"))
                        .take(20)
                        .collect();

                    let error_count = errors.len();
                    let error_summary = if errors.is_empty() {
                        // Fallback: show last 10 lines of stderr
                        stderr
                            .lines()
                            .rev()
                            .take(10)
                            .collect::<Vec<_>>()
                            .into_iter()
                            .rev()
                            .collect::<Vec<_>>()
                            .join("\n")
                    } else {
                        errors.join("\n")
                    };

                    Ok(format!(
                        "FAIL — cargo check failed ({error_count} errors, {elapsed_ms}ms)\n\n\
                         {error_summary}"
                    ))
                }
            }
            Err(e) => Ok(format!(
                "ERROR — failed to run cargo check: {e}\n\
                 Working directory: {}",
                self.working_dir.display()
            )),
        }
    }
}
