//! Shared types for subtask planning and execution.\n//!
//! This module exists to break the circular dependency between:
//! - `subtask.rs` (execution logic)
//! - `tools/plan_parallel_tool.rs` (validation and storage)
//! - `agents/` (usage)
//!
//! Only types and functions needed by tools should live here.

use serde::{Deserialize, Serialize};

/// Errors that can occur when parsing or validating a subtask plan.
#[derive(Debug, thiserror::Error)]
pub enum SubtaskPlanError {
    #[error("failed to parse SubtaskPlan JSON: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("{0}")]
    Validation(String),
}

/// A single subtask within a concurrent plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    /// Subtask identifier (e.g., "subtask-1").
    pub id: String,
    /// What the worker should do.
    pub objective: String,
    /// Files this worker is allowed to modify (non-overlapping with other subtasks).
    pub target_files: Vec<String>,
    /// Files the worker may read but not modify (shared context).
    #[serde(default)]
    pub context_files: Vec<String>,
    /// Worker type: "rust_coder" or "general_coder".
    #[serde(default = "default_worker_type")]
    pub worker_type: String,
}

fn default_worker_type() -> String {
    "general_coder".to_string()
}

/// Plan produced by the manager for concurrent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskPlan {
    /// High-level summary of the decomposition strategy.
    pub summary: String,
    /// Ordered list of subtasks (executed concurrently).
    pub subtasks: Vec<Subtask>,
}

/// Parse a SubtaskPlan from JSON.
///
/// Returns an error if the JSON is malformed or the plan is invalid.
pub fn parse_subtask_plan(json: &str) -> Result<SubtaskPlan, SubtaskPlanError> {
    let plan: SubtaskPlan = serde_json::from_str(json)?;

    // Validate that all subtasks have at least one target file
    for (i, subtask) in plan.subtasks.iter().enumerate() {
        if subtask.target_files.is_empty() {
            return Err(SubtaskPlanError::Validation(format!(
                "Subtask {} (id={}) must have at least one target file",
                i + 1,
                subtask.id
            )));
        }
    }

    Ok(plan)
}
