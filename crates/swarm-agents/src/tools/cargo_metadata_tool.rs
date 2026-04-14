//! Tool for querying Cargo workspace metadata.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{run_command_with_timeout, ToolError};

const TIMEOUT_SECS: u64 = 60;

#[derive(Deserialize)]
pub struct CargoMetadataInput {}

/// Run `cargo metadata` and return a compact human-readable summary of packages.
pub struct CargoMetadataTool {
    pub working_dir: PathBuf,
}

impl CargoMetadataTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for CargoMetadataTool {
    const NAME: &'static str = "cargo_metadata";
    type Error = ToolError;
    type Args = CargoMetadataInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "cargo_metadata".into(),
            description: "Return a compact summary of the Cargo workspace: \
                          workspace root, package names, manifest paths, and targets."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let output = run_command_with_timeout(
            "cargo",
            &["metadata", "--format-version=1", "--no-deps"],
            &self.working_dir,
            TIMEOUT_SECS,
        )
        .await?;

        let meta: serde_json::Value = serde_json::from_str(&output).map_err(|e| {
            ToolError::Parse(format!(
                "failed to parse cargo metadata JSON: {e}. Output was: {output}"
            ))
        })?;

        let workspace_root = meta
            .get("workspace_root")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        let mut out = format!("workspace_root: {workspace_root}\npackages:\n");

        if let Some(packages) = meta.get("packages").and_then(|v| v.as_array()) {
            for pkg in packages {
                let name = pkg.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                let manifest = pkg
                    .get("manifest_path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");

                // Make manifest path relative to workspace_root when possible.
                let rel_manifest = manifest
                    .strip_prefix(workspace_root)
                    .map(|s| s.trim_start_matches('/'))
                    .unwrap_or(manifest);

                out.push_str(&format!("  {name} ({rel_manifest})\n"));

                if let Some(targets) = pkg.get("targets").and_then(|v| v.as_array()) {
                    for target in targets {
                        let tname = target.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                        let kinds = target
                            .get("kind")
                            .and_then(|v| v.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|k| k.as_str())
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            })
                            .unwrap_or_default();
                        out.push_str(&format!("    {kinds}: {tname}\n"));
                    }
                }
            }
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cargo_metadata_contains_workspace_root() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = CargoMetadataTool::new(&repo_root);
        let result = tool
            .call(CargoMetadataInput {})
            .await
            .expect("cargo_metadata should not error");
        assert!(
            result.contains("workspace_root"),
            "output must mention workspace_root, got: {result}"
        );
    }

    #[tokio::test]
    async fn test_cargo_metadata_failure_on_non_workspace() {
        let temp_dir = tempfile::tempdir().unwrap();
        let tool = CargoMetadataTool::new(temp_dir.path());
        let result = tool.call(CargoMetadataInput {}).await;

        assert!(result.is_err());
        if let Err(ToolError::Parse(msg)) = result {
            assert!(msg.contains("failed to parse cargo metadata JSON"));
            assert!(msg.contains("could not find `Cargo.toml`"));
        } else {
            panic!("Expected Parse error, got {:?}", result);
        }
    }

    #[tokio::test]
    async fn test_cargo_metadata_valid_workspace() {
        let temp_dir = tempfile::tempdir().unwrap();
        let dir_path = temp_dir.path();

        std::fs::write(
            dir_path.join("Cargo.toml"),
            r#"[package]
name = "dummy_pkg"
version = "0.1.0"
edition = "2021"
"#,
        )
        .unwrap();

        std::fs::create_dir(dir_path.join("src")).unwrap();
        std::fs::write(dir_path.join("src/main.rs"), "fn main() {}").unwrap();

        let tool = CargoMetadataTool::new(dir_path);
        let result = tool
            .call(CargoMetadataInput {})
            .await
            .expect("cargo_metadata should succeed");

        assert!(result.contains("workspace_root"));
        assert!(result.contains("packages:\n"));
        assert!(result.contains("dummy_pkg (Cargo.toml)"));
        assert!(result.contains("bin: dummy_pkg"));
    }

    #[tokio::test]
    async fn test_cargo_metadata_invalid_manifest() {
        let temp_dir = tempfile::tempdir().unwrap();
        std::fs::write(temp_dir.path().join("Cargo.toml"), "invalid toml syntax {").unwrap();

        let tool = CargoMetadataTool::new(temp_dir.path());
        let result = tool.call(CargoMetadataInput {}).await;
        assert!(result.is_err(), "should fail on invalid manifest");
    }
}
