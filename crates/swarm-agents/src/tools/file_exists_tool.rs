//! Tool for checking whether one or more paths exist inside the sandbox.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::{sandbox_check, ToolError};

#[derive(Deserialize)]
pub struct FileExistsInput {
    /// List of relative paths to check (relative to the workspace root).
    pub paths: Vec<String>,
}

/// Check whether files or directories exist within the worktree sandbox.
pub struct FileExistsTool {
    pub working_dir: PathBuf,
}

impl FileExistsTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for FileExistsTool {
    const NAME: &'static str = "file_exists";
    type Error = ToolError;
    type Args = FileExistsInput;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "file_exists".into(),
            description: "Check whether one or more paths exist in the workspace. \
                          Returns a JSON object mapping each path to a boolean."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Relative paths to check for existence"
                    }
                },
                "required": ["paths"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let mut map = serde_json::Map::new();

        for rel in &args.paths {
            // sandbox_check canonicalises the path; if the file doesn't exist it
            // falls back to canonicalising the parent — which is fine for existence
            // checks because we just call .exists() on the original join.
            sandbox_check(&self.working_dir, rel)?;
            let full = self.working_dir.join(rel);
            map.insert(rel.clone(), serde_json::Value::Bool(full.exists()));
        }

        Ok(serde_json::Value::Object(map).to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_file_exists() {
        let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let tool = FileExistsTool::new(&repo_root);
        let result = tool
            .call(FileExistsInput {
                paths: vec!["Cargo.toml".to_string(), "nonexistent.rs".to_string()],
            })
            .await
            .expect("file_exists should not error");

        let parsed: serde_json::Value =
            serde_json::from_str(&result).expect("output must be valid JSON");
        assert_eq!(
            parsed["Cargo.toml"].as_bool(),
            Some(true),
            "Cargo.toml must exist"
        );
        assert_eq!(
            parsed["nonexistent.rs"].as_bool(),
            Some(false),
            "nonexistent.rs must not exist"
        );
    }
}
