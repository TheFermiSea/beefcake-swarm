//! Manager tool for submitting a work plan before execution.
//!
//! Inspired by ClawTeam's plan approval workflow: before delegating to coders,
//! the manager must articulate its approach. The plan is captured in a shared
//! slot and injected into subsequent iteration prompts as context.
//!
//! This reduces wasted iterations from workers going off-track — the plan
//! forces the manager to think through the approach and gives workers a
//! consistent reference for what they should be doing.

use std::sync::{Arc, Mutex};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

use super::ToolError;

/// A captured work plan submitted by the manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkPlan {
    /// High-level approach description.
    pub approach: String,
    /// Files that will be modified.
    pub target_files: Vec<String>,
    /// Risk assessment.
    pub risk: String,
    /// Iteration on which the plan was submitted.
    pub submitted_at_iteration: u32,
}

/// Shared slot where the manager deposits a plan for the orchestrator.
pub type WorkPlanSlot = Arc<Mutex<Option<WorkPlan>>>;

/// Create a new empty work plan slot.
pub fn new_work_plan_slot() -> WorkPlanSlot {
    Arc::new(Mutex::new(None))
}

#[derive(Deserialize)]
pub struct SubmitPlanArgs {
    /// High-level description of the fix/feature approach (1-3 sentences).
    pub approach: String,
    /// List of file paths that will be modified.
    pub target_files: Vec<String>,
    /// Risk level: "low", "medium", or "high".
    pub risk: String,
}

/// Tool that allows the manager to submit a work plan before delegating to coders.
///
/// The plan is stored in a shared slot and carried forward as context in
/// subsequent iteration prompts. This ensures the manager's strategy is
/// visible to both the orchestrator (for debugging) and future iterations
/// (for consistency).
pub struct SubmitPlanTool {
    plan_slot: WorkPlanSlot,
    iteration: u32,
}

impl SubmitPlanTool {
    pub fn new(plan_slot: WorkPlanSlot, iteration: u32) -> Self {
        Self {
            plan_slot,
            iteration,
        }
    }
}

impl Tool for SubmitPlanTool {
    const NAME: &'static str = "submit_plan";
    type Error = ToolError;
    type Args = SubmitPlanArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "submit_plan".into(),
            description: "Submit your work plan before delegating to coders. \
                          Describe your approach, list target files, and assess risk. \
                          The plan is recorded and carried forward as context for consistency. \
                          Call this ONCE on your first iteration, before any coder delegation."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["approach", "target_files", "risk"],
                "properties": {
                    "approach": {
                        "type": "string",
                        "description": "1-3 sentence description of the fix/feature strategy"
                    },
                    "target_files": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "File paths that will be modified"
                    },
                    "risk": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Risk level: low (isolated change), medium (cross-module), high (public API / unsafe)"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.approach.trim().is_empty() {
            return Err(ToolError::Policy(
                "Plan approach cannot be empty".to_string(),
            ));
        }
        if args.target_files.is_empty() {
            return Err(ToolError::Policy(
                "Plan must list at least one target file".to_string(),
            ));
        }

        let plan = WorkPlan {
            approach: args.approach.clone(),
            target_files: args.target_files.clone(),
            risk: args.risk.clone(),
            submitted_at_iteration: self.iteration,
        };

        let file_count = plan.target_files.len();
        let risk = plan.risk.clone();

        // A poisoned mutex means a prior panic occurred — this is a programming
        // bug per M-PANIC-ON-BUG, so we propagate the panic.
        let mut slot = self
            .plan_slot
            .lock()
            .expect("plan_slot mutex poisoned — prior panic");
        *slot = Some(plan);

        tracing::info!(
            approach = %args.approach,
            files = file_count,
            risk = %risk,
            iteration = self.iteration,
            "Manager submitted work plan"
        );

        Ok(format!(
            "Plan recorded ({file_count} files, risk: {risk}). \
             You may now delegate to coders. The plan will be carried \
             forward as context for subsequent iterations."
        ))
    }
}

/// Format a captured work plan as a prompt section for injection into task prompts.
pub fn format_plan_context(plan: &WorkPlan) -> String {
    let mut s = String::from("## Approved Plan (from iteration ");
    s.push_str(&plan.submitted_at_iteration.to_string());
    s.push_str(")\n\n");
    s.push_str("**Approach:** ");
    s.push_str(&plan.approach);
    s.push_str("\n\n**Target files:**\n");
    for f in &plan.target_files {
        s.push_str(&format!("- `{f}`\n"));
    }
    s.push_str(&format!("\n**Risk:** {}\n\n", plan.risk));
    s.push_str("_Follow this plan. If you need to deviate, call submit_plan again with the revised approach._\n\n");
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn valid_plan_accepted() {
        let slot = new_work_plan_slot();
        let tool = SubmitPlanTool::new(slot.clone(), 1);

        let result = tool
            .call(SubmitPlanArgs {
                approach: "Fix the borrow checker error by introducing a clone".to_string(),
                target_files: vec!["src/parser.rs".to_string()],
                risk: "low".to_string(),
            })
            .await
            .unwrap();

        assert!(result.contains("1 files"));
        assert!(result.contains("risk: low"));
        let plan = slot.lock().unwrap();
        assert!(plan.is_some());
        assert_eq!(plan.as_ref().unwrap().submitted_at_iteration, 1);
    }

    #[tokio::test]
    async fn empty_approach_rejected() {
        let slot = new_work_plan_slot();
        let tool = SubmitPlanTool::new(slot.clone(), 1);

        let result = tool
            .call(SubmitPlanArgs {
                approach: "   ".to_string(),
                target_files: vec!["src/parser.rs".to_string()],
                risk: "low".to_string(),
            })
            .await;

        assert!(
            matches!(result, Err(ToolError::Policy(ref m)) if m.contains("approach cannot be empty"))
        );
    }

    #[tokio::test]
    async fn no_files_rejected() {
        let slot = new_work_plan_slot();
        let tool = SubmitPlanTool::new(slot.clone(), 1);

        let result = tool
            .call(SubmitPlanArgs {
                approach: "Fix the borrow checker error".to_string(),
                target_files: vec![],
                risk: "low".to_string(),
            })
            .await;

        assert!(
            matches!(result, Err(ToolError::Policy(ref m)) if m.contains("list at least one target file"))
        );
    }
}
