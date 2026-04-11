//! Unified worker delegation tool.
//!
//! Implements the Managed Agents `execute(name, input) → string` pattern:
//! a single tool that routes work to any available worker by role, rather
//! than exposing N individual worker tools on the manager.
//!
//! Benefits over per-worker tools:
//! - Adding new worker roles is a registration call, not a code change
//! - The manager's tool list stays small and stable
//! - Role-based routing can incorporate dynamic decisions (model health, tier budget)

use std::collections::HashMap;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::{info, warn};

use crate::context_firewall::{CondensedAgentTool, CondensedAgentToolArgs};

/// Concrete model type used by all workers (OpenAI-compatible).
type WorkerModel = rig::providers::openai::completion::CompletionModel;

/// A single tool that delegates work to any registered worker by role.
///
/// The manager calls `delegate_worker(role="coder", prompt="Fix the borrow error...")`
/// and this tool routes to the appropriate `CondensedAgentTool` internally.
pub struct DelegateWorkerTool {
    /// Available workers keyed by role name (e.g., "coder", "planner", "reviewer").
    workers: HashMap<String, CondensedAgentTool<WorkerModel>>,
    /// Ordered list of available role names (for the tool description).
    available_roles: Vec<String>,
}

#[derive(Deserialize, JsonSchema)]
pub struct DelegateWorkerArgs {
    /// The worker role to delegate to.
    /// Available roles depend on the swarm configuration.
    pub role: String,
    /// The task prompt to send to the worker. Should include:
    /// - What needs to be done (clear objective)
    /// - Relevant file paths and context
    /// - Error messages to fix (if applicable)
    pub prompt: String,
}

impl Default for DelegateWorkerTool {
    fn default() -> Self {
        Self::new()
    }
}

impl DelegateWorkerTool {
    /// Create a new delegate worker tool with no workers registered.
    pub fn new() -> Self {
        Self {
            workers: HashMap::new(),
            available_roles: Vec::new(),
        }
    }

    /// Register a worker for a given role.
    pub fn register(mut self, role: impl Into<String>, worker: CondensedAgentTool<WorkerModel>) -> Self {
        let role = role.into();
        if !self.available_roles.contains(&role) {
            self.available_roles.push(role.clone());
        }
        self.workers.insert(role, worker);
        self
    }

    /// Get the list of registered role names.
    pub fn roles(&self) -> &[String] {
        &self.available_roles
    }
}

impl Tool for DelegateWorkerTool {
    const NAME: &'static str = "delegate_worker";

    type Error = rig::completion::PromptError;
    type Args = DelegateWorkerArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let roles_list = self.available_roles.join(", ");
        ToolDefinition {
            name: "delegate_worker".into(),
            description: format!(
                "Delegate a coding task to a specialized worker agent. \
                 Available roles: [{}]. \
                 Each role maps to a different model optimized for that task type. \
                 The worker will read files, write code, and return a summary of changes made.",
                roles_list
            ),
            parameters: serde_json::to_value(schemars::schema_for!(DelegateWorkerArgs))
                .unwrap_or_default(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let role = args.role.to_lowercase();

        let worker = match self.workers.get(&role) {
            Some(w) => w,
            None => {
                let available = self.available_roles.join(", ");
                warn!(
                    requested = %role,
                    available = %available,
                    "Unknown worker role in delegate_worker"
                );
                return Ok(format!(
                    "Error: unknown worker role '{}'. Available roles: [{}]",
                    role, available
                ));
            }
        };

        info!(role = %role, prompt_len = args.prompt.len(), "Delegating to worker");

        worker
            .call(CondensedAgentToolArgs {
                prompt: args.prompt,
            })
            .await
    }
}
