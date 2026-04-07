//! Manager tool for submitting a parallel work plan.
//!
//! When the cloud manager calls `plan_parallel_work`, it provides a `SubtaskPlan`
//! JSON that the orchestrator captures and uses to switch to concurrent dispatch.
//! This replaces the local planner for manager-guided decomposition.

use std::sync::{Arc, Mutex};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::ToolError;
use crate::subtask::{parse_subtask_plan, SubtaskPlan};

/// Shared slot where the manager deposits a validated plan for the orchestrator to pick up.
pub type PlanSlot = Arc<Mutex<Option<SubtaskPlan>>>;

/// Create a new empty plan slot.
pub fn new_plan_slot() -> PlanSlot {
    Arc::new(Mutex::new(None))
}

#[derive(Deserialize)]
pub struct PlanParallelWorkArgs {
    /// JSON string containing the SubtaskPlan. Must have at least 2 subtasks
    /// with non-overlapping target_files.
    pub plan_json: String,
}

/// Tool that allows the manager to submit a parallel work decomposition plan.
///
/// The manager calls this when it determines an issue can benefit from
/// concurrent execution. The plan is validated (non-overlap, min 2 subtasks)
/// and stored in a shared slot for the orchestrator to pick up.
pub struct PlanParallelWorkTool {
    plan_slot: PlanSlot,
}

impl PlanParallelWorkTool {
    pub fn new(plan_slot: PlanSlot) -> Self {
        Self { plan_slot }
    }
}

impl Tool for PlanParallelWorkTool {
    const NAME: &'static str = "plan_parallel_work";
    type Error = ToolError;
    type Args = PlanParallelWorkArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "plan_parallel_work".into(),
            description: "Submit a parallel work plan to dispatch multiple workers concurrently. \
                          Use when an issue involves changes to multiple independent files. \
                          The plan must have at least 2 subtasks with non-overlapping target_files. \
                          Integration files (Cargo.toml, mod.rs, lib.rs, main.rs) may only appear \
                          in one subtask. After submitting, the orchestrator dispatches workers \
                          and runs the verifier on the combined result."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["plan_json"],
                "properties": {
                    "plan_json": {
                        "type": "string",
                        "description": "JSON string with the subtask plan. Schema: {\"summary\": \"...\", \"subtasks\": [{\"id\": \"subtask-1\", \"objective\": \"...\", \"target_files\": [\"path/to/file.rs\"], \"context_files\": [\"path/to/read.rs\"], \"worker_type\": \"rust_coder|general_coder\"}]}"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Parse and validate the plan.
        let plan = parse_subtask_plan(&args.plan_json)
            .map_err(|e| ToolError::Policy(format!("Invalid parallel work plan: {e}")))?;

        // Require at least 2 subtasks for parallel execution.
        if plan.subtasks.len() < 2 {
            return Err(ToolError::Policy(
                "Parallel work plan must have at least 2 subtasks. \
                 For single-file changes, use workers directly."
                    .to_string(),
            ));
        }

        let subtask_count = plan.subtasks.len();
        let summary = plan.summary.clone();

        // Store the validated plan for the orchestrator.
        // A poisoned mutex means a prior panic occurred — this is a programming
        // bug per M-PANIC-ON-BUG, so we propagate the panic.
        let mut slot = self
            .plan_slot
            .lock()
            .expect("plan_slot mutex poisoned — prior panic");
        *slot = Some(plan);

        Ok(format!(
            "Parallel work plan accepted: {subtask_count} subtasks. Summary: {summary}. \
             The orchestrator will now dispatch workers concurrently."
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn valid_plan_accepted() {
        let slot = new_plan_slot();
        let tool = PlanParallelWorkTool::new(slot.clone());

        let plan_json = r#"{
            "summary": "Split parser and tests",
            "subtasks": [
                {"id": "subtask-1", "objective": "Fix parser", "target_files": ["src/parser.rs"]},
                {"id": "subtask-2", "objective": "Add tests", "target_files": ["tests/parser_test.rs"]}
            ]
        }"#;

        let result = tool
            .call(PlanParallelWorkArgs {
                plan_json: plan_json.to_string(),
            })
            .await
            .unwrap();

        assert!(result.contains("2 subtasks"));
        assert!(slot.lock().unwrap().is_some());
    }
}
