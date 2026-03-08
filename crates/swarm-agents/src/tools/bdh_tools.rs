//! Rig `Tool` implementations for bdh coordination.
//!
//! Provides team-awareness tools for the manager agent:
//! - `TeamStatusTool` — check what other agents are working on
//! - `CheckMailTool` — read incoming mail messages
//! - `SendMailTool` — send async messages to other agents
//! - `CheckLocksTool` — see which files are locked

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;

use super::ToolError;

/// Timeout for bdh coordination commands (seconds).
const BDH_TIMEOUT_SECS: u64 = 30;

// ── TeamStatusTool ─────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct TeamStatusArgs {}

/// Show team status: who is working on what, their roles, and current issues.
pub struct TeamStatusTool {
    bin: String,
    wt_path: PathBuf,
}

impl TeamStatusTool {
    pub fn new(wt_path: &Path) -> Self {
        Self {
            bin: std::env::var("SWARM_BDH_BIN").unwrap_or_else(|_| "bdh".into()),
            wt_path: wt_path.to_path_buf(),
        }
    }
}

impl Tool for TeamStatusTool {
    const NAME: &'static str = "team_status";
    type Error = ToolError;
    type Args = TeamStatusArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "team_status".into(),
            description: "Check team status: see which agents are active, what issues they're \
                          working on, and their roles. Use this before delegating work to avoid \
                          conflicts with other agents."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let bin = self.bin.clone();
        let wt = self.wt_path.clone();
        super::run_command_with_timeout(&bin, &[":status"], &wt, BDH_TIMEOUT_SECS).await
    }
}

// ── CheckMailTool ──────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CheckMailArgs {}

/// Check for incoming mail messages from other agents.
pub struct CheckMailTool {
    bin: String,
    wt_path: PathBuf,
}

impl CheckMailTool {
    pub fn new(wt_path: &Path) -> Self {
        Self {
            bin: std::env::var("SWARM_BDH_BIN").unwrap_or_else(|_| "bdh".into()),
            wt_path: wt_path.to_path_buf(),
        }
    }
}

impl Tool for CheckMailTool {
    const NAME: &'static str = "check_mail";
    type Error = ToolError;
    type Args = CheckMailArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "check_mail".into(),
            description: "Check your inbox for messages from other agents. \
                          Messages may contain status updates, handoff notes, \
                          or questions about your work."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let bin = self.bin.clone();
        let wt = self.wt_path.clone();
        super::run_command_with_timeout(&bin, &[":aweb", "mail", "list"], &wt, BDH_TIMEOUT_SECS)
            .await
    }
}

// ── SendMailTool ───────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct SendMailArgs {
    /// The alias of the recipient agent.
    pub to: String,
    /// The message to send.
    pub message: String,
}

/// Send an async (fire-and-forget) mail message to another agent.
pub struct SendMailTool {
    bin: String,
    wt_path: PathBuf,
}

impl SendMailTool {
    pub fn new(wt_path: &Path) -> Self {
        Self {
            bin: std::env::var("SWARM_BDH_BIN").unwrap_or_else(|_| "bdh".into()),
            wt_path: wt_path.to_path_buf(),
        }
    }
}

impl Tool for SendMailTool {
    const NAME: &'static str = "send_mail";
    type Error = ToolError;
    type Args = SendMailArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "send_mail".into(),
            description: "Send a message to another agent. Use for status updates, \
                          handoffs, and non-blocking questions. The message is delivered \
                          asynchronously — you won't get an immediate reply."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "to": {
                        "type": "string",
                        "description": "Alias of the recipient agent (from team_status)"
                    },
                    "message": {
                        "type": "string",
                        "description": "The message to send"
                    }
                },
                "required": ["to", "message"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let bin = self.bin.clone();
        let wt = self.wt_path.clone();
        super::run_command_with_timeout(
            &bin,
            &[":aweb", "mail", "send", &args.to, &args.message],
            &wt,
            BDH_TIMEOUT_SECS,
        )
        .await
    }
}

// ── CheckLocksTool ─────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CheckLocksArgs {}

/// Show active file locks across all agents.
pub struct CheckLocksTool {
    bin: String,
    wt_path: PathBuf,
}

impl CheckLocksTool {
    pub fn new(wt_path: &Path) -> Self {
        Self {
            bin: std::env::var("SWARM_BDH_BIN").unwrap_or_else(|_| "bdh".into()),
            wt_path: wt_path.to_path_buf(),
        }
    }
}

impl Tool for CheckLocksTool {
    const NAME: &'static str = "check_locks";
    type Error = ToolError;
    type Args = CheckLocksArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "check_locks".into(),
            description: "Show active file locks held by all agents. \
                          Use before assigning work to check if target files \
                          are already being edited by another agent."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
                "required": []
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let bin = self.bin.clone();
        let wt = self.wt_path.clone();
        super::run_command_with_timeout(&bin, &[":aweb", "locks"], &wt, BDH_TIMEOUT_SECS).await
    }
}
