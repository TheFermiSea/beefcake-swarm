//! Context firewalls for worker-to-manager handoffs.
//!
//! Condenses raw agent-as-tool output before the cloud manager sees it.
//! Instead of passing verbose tool call logs, the manager receives a structured
//! summary: files modified, write count, turns used, and termination reason.
//!
//! Research: NLAH (arxiv:2603.25723) — context firewalls (sub-agent isolation)
//! are the "most impactful structural decision" for multi-agent harnesses.
//!
//! # Usage
//!
//! ```ignore
//! use swarm_agents::context_firewall::CondensedAgentTool;
//!
//! // Wrap a worker agent for use as a tool in the manager
//! let condensed_coder = CondensedAgentTool::new(rust_coder);
//! let manager = client.agent(model).tool(condensed_coder).build();
//! ```

use rig::agent::Agent;
use rig::completion::{CompletionModel, PromptError, ToolDefinition};
use rig::tool::Tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::runtime_adapter::AdapterReport;

/// Truncate a string at a char boundary. Safe for multi-byte UTF-8
/// (e.g., em dash, Unicode quotes in rustc output).
fn safe_truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        return s;
    }
    let mut end = max_len;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Snap a byte offset forward to the nearest char boundary.
fn snap_to_char_boundary(s: &str, byte_offset: usize) -> usize {
    let mut pos = byte_offset;
    while pos < s.len() && !s.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Condensed worker result for manager consumption.
/// Strips raw tool call logs to prevent context bloat.
///
/// Research: NLAH (arxiv:2603.25723) — context firewalls are
/// "most impactful structural decision."
#[derive(Debug, Clone)]
pub struct CondensedWorkerResult {
    pub files_modified: Vec<String>,
    pub successful_writes: u32,
    pub turns_used: usize,
    pub max_turns: usize,
    pub terminated_early: bool,
    pub termination_reason: Option<String>,
    pub has_written: bool,
    pub last_failed_edits: Vec<(String, String)>,
}

impl CondensedWorkerResult {
    pub fn from_adapter_report(report: &AdapterReport, max_turns: usize) -> Self {
        Self {
            files_modified: report.files_modified.clone(),
            successful_writes: report.successful_writes,
            turns_used: report.turn_count,
            max_turns,
            terminated_early: report.terminated_early,
            termination_reason: report.termination_reason.clone(),
            has_written: report.has_written,
            last_failed_edits: report.last_failed_edits.clone(),
        }
    }

    pub fn to_summary(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "Worker completed: {}/{} turns used",
            self.turns_used, self.max_turns
        ));

        if self.has_written {
            lines.push(format!(
                "Files modified: {}",
                self.files_modified.join(", ")
            ));
            lines.push(format!("Successful writes: {}", self.successful_writes));
        } else {
            lines.push("No files were modified.".to_string());
        }

        if self.terminated_early {
            if let Some(reason) = &self.termination_reason {
                lines.push(format!("Terminated early: {}", reason));
            }
        }

        if !self.last_failed_edits.is_empty() {
            lines.push("Last failed edits:".to_string());
            for (path, err) in &self.last_failed_edits {
                let err_short = safe_truncate(err, 100);
                lines.push(format!("  {} — {}", path, err_short));
            }
        }

        lines.join("\n")
    }
}

/// Maximum length of the worker response tail to include in the condensed output.
/// The last N characters of the raw response often contain the worker's final
/// summary or conclusion, which is useful context for the manager.
const RESPONSE_TAIL_BUDGET: usize = 500;

/// Condense a raw worker agent response into a compact summary.
///
/// Extracts the worker's final message (last N chars, bounded at a sentence
/// boundary) and prepends a structured header with key metrics.
///
/// This is the string-only path — used when no `AdapterReport` is available
/// (agent-as-tool sub-agents don't carry RuntimeAdapter hooks).
fn condense_agent_response(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return "Worker returned empty response.".to_string();
    }

    // If already compact, return as-is.
    if trimmed.len() <= RESPONSE_TAIL_BUDGET {
        return trimmed.to_string();
    }

    // Take the last RESPONSE_TAIL_BUDGET chars, snapped to a sentence boundary.
    let tail_start =
        snap_to_char_boundary(trimmed, trimmed.len().saturating_sub(RESPONSE_TAIL_BUDGET));
    // Find the nearest sentence boundary (`. ` or `\n`) after tail_start.
    let snap_pos = trimmed[tail_start..]
        .find(". ")
        .or_else(|| trimmed[tail_start..].find('\n'))
        .map(|offset| tail_start + offset + 1)
        .unwrap_or(tail_start);
    let snap_pos = snap_to_char_boundary(trimmed, snap_pos);

    let tail = trimmed[snap_pos..].trim();

    let total_len = trimmed.len();
    format!(
        "[Context firewall: condensed {total_len} chars to summary]\n\
         ...{tail}"
    )
}

// ── CondensedAgentTool ──────────────────────────────────────────────────

/// Tool arguments — identical to Rig's `AgentToolArgs` so the manager's
/// tool-calling interface is unchanged.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct CondensedAgentToolArgs {
    /// The prompt for the agent to call.
    pub prompt: String,
}

/// Wrapper that interposes a context firewall between a worker agent
/// (used as agent-as-tool) and the manager that invokes it.
///
/// The inner agent runs normally, but its raw string response is condensed
/// before being returned to the manager. This prevents tool call logs,
/// verbose read_file output, and other intermediate noise from bloating
/// the manager's context window.
pub struct CondensedAgentTool<M: CompletionModel> {
    inner: Agent<M>,
}

impl<M: CompletionModel> CondensedAgentTool<M> {
    pub fn new(agent: Agent<M>) -> Self {
        Self { inner: agent }
    }
}

impl<M: CompletionModel> Tool for CondensedAgentTool<M> {
    const NAME: &'static str = "condensed_agent_tool";

    type Error = PromptError;
    type Args = CondensedAgentToolArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let name = self
            .inner
            .name
            .clone()
            .unwrap_or_else(|| Self::NAME.to_string());
        let description = format!(
            "Prompt a sub-agent to do a task for you.\n\n\
             Agent name: {name}\n\
             Agent description: {desc}\n",
            name = name,
            desc = self.inner.description.clone().unwrap_or_default(),
        );
        ToolDefinition {
            name: self.name(),
            description,
            parameters: serde_json::to_value(schemars::schema_for!(CondensedAgentToolArgs))
                .unwrap_or_default(),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        use rig::completion::Prompt;

        let raw_response = self.inner.prompt(args.prompt).await?;
        Ok(condense_agent_response(&raw_response))
    }

    fn name(&self) -> String {
        self.inner
            .name
            .clone()
            .unwrap_or_else(|| Self::NAME.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_condense_short_response() {
        let short = "Fixed the borrow checker error in parser.rs";
        assert_eq!(condense_agent_response(short), short);
    }

    #[test]
    fn test_condense_empty_response() {
        assert_eq!(
            condense_agent_response(""),
            "Worker returned empty response."
        );
        assert_eq!(
            condense_agent_response("   "),
            "Worker returned empty response."
        );
    }

    #[test]
    fn test_condense_long_response() {
        // Build a response that's longer than RESPONSE_TAIL_BUDGET
        let filler = "Reading file src/parser.rs. ";
        let mut long = String::new();
        for _ in 0..50 {
            long.push_str(filler);
        }
        long.push_str("Final summary. I fixed the borrow checker error by adding a clone.");

        let condensed = condense_agent_response(&long);
        assert!(condensed.contains("[Context firewall: condensed"));
        assert!(condensed.contains("I fixed the borrow checker error"));
        assert!(condensed.len() < long.len());
    }

    #[test]
    fn test_condensed_worker_result_summary_with_writes() {
        let result = CondensedWorkerResult {
            files_modified: vec!["src/parser.rs".into(), "src/lib.rs".into()],
            successful_writes: 3,
            turns_used: 8,
            max_turns: 15,
            terminated_early: false,
            termination_reason: None,
            has_written: true,
            last_failed_edits: vec![],
        };
        let summary = result.to_summary();
        assert!(summary.contains("8/15 turns used"));
        assert!(summary.contains("src/parser.rs, src/lib.rs"));
        assert!(summary.contains("Successful writes: 3"));
        assert!(!summary.contains("Terminated early"));
    }

    #[test]
    fn test_condensed_worker_result_summary_no_writes() {
        let result = CondensedWorkerResult {
            files_modified: vec![],
            successful_writes: 0,
            turns_used: 15,
            max_turns: 15,
            terminated_early: true,
            termination_reason: Some("max turns reached without edit_file".into()),
            has_written: false,
            last_failed_edits: vec![],
        };
        let summary = result.to_summary();
        assert!(summary.contains("No files were modified"));
        assert!(summary.contains("Terminated early: max turns reached"));
    }

    #[test]
    fn test_condensed_worker_result_with_failed_edits() {
        let result = CondensedWorkerResult {
            files_modified: vec!["src/main.rs".into()],
            successful_writes: 1,
            turns_used: 10,
            max_turns: 15,
            terminated_early: true,
            termination_reason: Some("repeated edit failures".into()),
            has_written: true,
            last_failed_edits: vec![(
                "src/lib.rs".into(),
                "old_string not found in file: expected 'fn foo()' but file has 'fn bar()'".into(),
            )],
        };
        let summary = result.to_summary();
        assert!(summary.contains("Last failed edits:"));
        assert!(summary.contains("src/lib.rs"));
    }

    #[test]
    fn test_condensed_worker_result_from_adapter_report() {
        let report = AdapterReport {
            agent_name: "rust_coder".into(),
            tool_events: vec![],
            turn_count: 5,
            total_tool_calls: 12,
            total_tool_time_ms: 3000,
            wall_time_ms: 10000,
            terminated_early: false,
            termination_reason: None,
            has_written: true,
            files_read: vec!["src/main.rs".into()],
            files_modified: vec!["src/parser.rs".into()],
            successful_writes: 2,
            last_failed_edits: vec![],
            total_reads_before_write: 0,
        };
        let condensed = CondensedWorkerResult::from_adapter_report(&report, 15);
        assert_eq!(condensed.turns_used, 5);
        assert_eq!(condensed.max_turns, 15);
        assert!(condensed.has_written);
        assert_eq!(condensed.files_modified, vec!["src/parser.rs"]);
        assert_eq!(condensed.successful_writes, 2);
    }
}
