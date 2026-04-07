//! Direct plan application tool — applies ArchitectPlan edits without LLM inference.
//!
//! When the cloud manager receives an ArchitectPlan JSON from `proxy_architect`,
//! it can call `apply_plan` to apply the edits directly instead of routing through
//! `proxy_editor` (which runs a full 15-turn agentic loop on a local model).
//!
//! This reduces the edit phase from ~10 minutes (local model agentic) to ~0.1s
//! (deterministic string replacement). Falls back gracefully: if any edit fails,
//! returns detailed diagnostics so the manager can retry with `proxy_editor`.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::ToolError;
use crate::pipeline::{direct_apply_plan, ArchitectPlan};

#[derive(Deserialize)]
pub struct ApplyPlanArgs {
    /// The ArchitectPlan JSON string (as returned by proxy_architect).
    pub plan_json: String,
}

/// Deterministic tool that applies an ArchitectPlan's SEARCH/REPLACE edits
/// directly to files in the worktree — no LLM inference needed.
pub struct ApplyPlanTool {
    pub working_dir: PathBuf,
}

impl ApplyPlanTool {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl Tool for ApplyPlanTool {
    const NAME: &'static str = "apply_plan";
    type Error = ToolError;
    type Args = ApplyPlanArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "apply_plan".into(),
            description: "Apply an ArchitectPlan directly to the codebase. Takes the JSON plan \
                          from proxy_architect and applies each SEARCH/REPLACE edit deterministically. \
                          Much faster than proxy_editor (instant vs 10+ minutes). \
                          Use this FIRST; only fall back to proxy_editor if edits fail."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "plan_json": {
                        "type": "string",
                        "description": "The ArchitectPlan JSON (paste the full JSON output from proxy_architect)"
                    }
                },
                "required": ["plan_json"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Parse the plan JSON
        let plan: ArchitectPlan = serde_json::from_str(&args.plan_json).map_err(|e| {
            ToolError::Parse(format!(
                "Failed to parse ArchitectPlan JSON: {e}. \
                 Make sure you pass the complete JSON object from proxy_architect."
            ))
        })?;

        // Validate the plan
        if let Err(e) = plan.validate() {
            return Err(ToolError::Validation(format!("Invalid ArchitectPlan: {e}")));
        }

        // Apply edits directly
        let result = direct_apply_plan(&plan, &self.working_dir);

        if result.all_succeeded() {
            Ok(format!(
                "All {} edits applied successfully.\n{}",
                result.applied.len(),
                result
                    .applied
                    .iter()
                    .map(|a| format!("  ✓ {a}"))
                    .collect::<Vec<_>>()
                    .join("\n")
            ))
        } else if result.applied.is_empty() {
            Err(ToolError::Validation(format!(
                "All edits failed. Use proxy_editor as fallback.\n{}",
                result.summary()
            )))
        } else {
            // Partial success — report what worked and what didn't
            Ok(format!(
                "Partial success: {}/{} edits applied.\n{}\n\n\
                 Failed edits may need proxy_editor for fuzzy application.",
                result.applied.len(),
                result.applied.len() + result.failed.len(),
                result.summary()
            ))
        }
    }
}
