//! NS-4.6: Agentic Mode — LLM-driven file editing via unified diffs.
//!
//! An agent is given a task and a sandbox directory. It iteratively:
//! 1. Receives tool result feedback from the previous `apply_diff` call.
//! 2. Emits a new diff (or finishes with a plain-text summary).
//! 3. The orchestrator calls `step()` again until Done or budget exhausted.
//!
//! ## Termination
//!
//! - **Done**: agent sends a plain text response without a diff tool call.
//! - **MaxIterations**: `OrchestrationError::MaxIterations` is returned.
//! - **Cancelled**: early exit on context cancellation.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::time::Instant;

use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use rig::tool::Tool;
use tracing::{debug, info, warn};

use crate::modes::{
    apply_diff::{ApplyDiffArgs, ApplyDiffTool},
    errors::OrchestrationError,
    provider_config::ModeRunnerConfig,
    runner::{ModeContext, ModeRequest, ModeRunner, StepResult},
    types::{Artifact, ModeOutcome},
};

// ── AgenticState ──────────────────────────────────────────────────────────────

#[derive(Debug)]
enum AgenticState {
    /// Haven't started yet — prepare() will set this to Editing.
    Idle,
    /// Agent is actively editing.
    Editing {
        task_prompt: String,
        iteration: u32,
        started: Instant,
    },
    /// Agent finished normally.
    Done { summary: String, iterations: u32 },
}

// ── AgenticRunner ─────────────────────────────────────────────────────────────

/// Agentic Mode runner: drives an LLM agent that edits files via unified diffs.
pub struct AgenticRunner {
    config: ModeRunnerConfig,
    /// Sandbox directory for all file I/O.
    working_dir: PathBuf,
    /// Current phase.
    state: AgenticState,
    /// Rolling conversation history (user ↔ assistant turns).
    history: VecDeque<Message>,
}

impl AgenticRunner {
    pub fn new(config: ModeRunnerConfig, working_dir: PathBuf) -> Self {
        Self {
            config,
            working_dir,
            state: AgenticState::Idle,
            history: VecDeque::new(),
        }
    }

    fn preamble(&self) -> String {
        format!(
            r#"You are an expert Rust programmer making targeted edits to files in a code workspace.

## Working directory
All files are relative to: {}

## Editing rules
1. To edit a file, output a JSON code block containing "file_path" and "diff":
   ```json
   {{"file_path": "src/main.rs", "diff": "@@ -1,3 +1,3 @@\n context\n-old\n+new\n context"}}
   ```
2. One JSON block per turn (one file at a time).
3. Hunks must include at least 3 lines of context unless the file is very short.
4. When all edits are complete, respond with plain text only (no JSON/diff block).

## Error recovery
If you receive an error from a previous apply, analyse the mismatch, then emit
a corrected diff.
"#,
            self.working_dir.display()
        )
    }

    fn push_user(&mut self, content: String) {
        self.history.push_back(Message::user(content));
    }

    fn push_assistant(&mut self, content: String) {
        self.history.push_back(Message::assistant(content));
    }

    fn history_vec(&self) -> Vec<Message> {
        self.history.iter().cloned().collect()
    }

    async fn apply_diff_call(&self, input: ApplyDiffArgs) -> String {
        let tool = ApplyDiffTool::new(&self.working_dir);
        match tool.call(input).await {
            Ok(output) => format!(
                "Applied {} hunk(s), {} lines changed.",
                output.hunks_applied, output.lines_changed
            ),
            Err(e) => format!("ERROR: {e}"),
        }
    }
}

// ── ModeRunner impl ───────────────────────────────────────────────────────────

#[async_trait]
impl ModeRunner for AgenticRunner {
    fn name(&self) -> &'static str {
        "agentic"
    }

    async fn prepare(
        &mut self,
        _ctx: &ModeContext,
        request: &ModeRequest,
    ) -> Result<(), OrchestrationError> {
        self.config
            .validate()
            .map_err(OrchestrationError::Configuration)?;
        self.history.clear();
        self.state = AgenticState::Editing {
            task_prompt: request.task.clone(),
            iteration: 0,
            started: Instant::now(),
        };
        // Seed history with the task.
        self.push_user(format!(
            "Task: {}\n\nPlease make the necessary edits using the apply_diff JSON format.",
            request.task
        ));
        Ok(())
    }

    async fn step(&mut self, ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
        if ctx.is_cancelled() {
            return Ok(StepResult::Failed(OrchestrationError::Cancelled(
                "cancelled by caller".to_string(),
            )));
        }

        let (task_prompt, iteration, started) = match &self.state {
            AgenticState::Editing {
                task_prompt,
                iteration,
                started,
            } => (task_prompt.clone(), *iteration, *started),
            AgenticState::Done {
                summary,
                iterations,
                ..
            } => {
                return Ok(StepResult::Done(ModeOutcome::Success {
                    artifact: Artifact::new(summary.clone()),
                    iterations: *iterations,
                    total_tokens: None,
                }));
            }
            AgenticState::Idle => {
                return Ok(StepResult::Failed(OrchestrationError::Configuration(
                    "prepare() must be called before step()".to_string(),
                )));
            }
        };

        if iteration >= self.config.max_iterations {
            return Ok(StepResult::Failed(OrchestrationError::MaxIterations(
                self.config.max_iterations,
            )));
        }

        debug!(iteration, "agentic step");

        let client = self.config.local_client().map_err(|e| {
            OrchestrationError::Configuration(format!("failed to build local client: {e}"))
        })?;

        let preamble = self.preamble();
        let agent = client
            .agent(&self.config.models.generator)
            .preamble(&preamble)
            .temperature(self.config.generator_temperature)
            .build();

        let history = self.history_vec();
        let response = agent
            .chat(&task_prompt, history)
            .await
            .map_err(|e| OrchestrationError::InferenceFailure(e.to_string()))?;

        self.push_assistant(response.clone());

        // Check whether the agent emitted a diff tool call.
        if let Some(diff_json) = extract_apply_diff_json(&response) {
            match serde_json::from_str::<ApplyDiffArgs>(&diff_json) {
                Ok(input) => {
                    let result = self.apply_diff_call(input).await;
                    debug!(result = %result, "apply_diff result");
                    let feedback = format!("Tool result: {result}");
                    self.push_user(feedback);
                    self.state = AgenticState::Editing {
                        task_prompt: task_prompt.clone(),
                        iteration: iteration + 1,
                        started,
                    };
                    Ok(StepResult::Continue(()))
                }
                Err(e) => {
                    let msg =
                        format!("Tool parse error: {e}. Please re-emit a valid JSON diff block.");
                    warn!(%e, "failed to parse apply_diff JSON");
                    self.push_user(msg);
                    self.state = AgenticState::Editing {
                        task_prompt,
                        iteration: iteration + 1,
                        started,
                    };
                    Ok(StepResult::Continue(()))
                }
            }
        } else {
            // No diff — agent is done.
            let elapsed_ms = started.elapsed().as_millis() as u64;
            info!(iteration, elapsed_ms, "agentic mode done");
            self.state = AgenticState::Done {
                summary: response.clone(),
                iterations: iteration + 1,
            };
            Ok(StepResult::Done(ModeOutcome::Success {
                artifact: Artifact::new(response),
                iterations: iteration + 1,
                total_tokens: None,
            }))
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract a JSON block for `apply_diff` from the agent response.
fn extract_apply_diff_json(response: &str) -> Option<String> {
    // Try fenced ```json block first.
    if let Some(start) = response.find("```json") {
        let after = &response[start + 7..];
        if let Some(end) = after.find("```") {
            let json = after[..end].trim();
            if json.contains("\"file_path\"") || json.contains("\"path\"") {
                return Some(json.to_string());
            }
        }
    }
    // Fall back to bare JSON object.
    if let Some(start) = response.find('{') {
        if let Some(end) = response.rfind('}') {
            if end > start {
                let json = response[start..=end].trim();
                if json.contains("\"file_path\"") || json.contains("\"path\"") {
                    return Some(json.to_string());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_fenced_json() {
        let response = r#"
Here is my edit:
```json
{"file_path": "src/main.rs", "diff": "@@ -1,1 +1,1 @@\n-old\n+new"}
```
"#;
        let json = extract_apply_diff_json(response).unwrap();
        assert!(json.contains("file_path"));
    }

    #[test]
    fn extract_bare_json_path_field() {
        let response = r#"Some text {"path": "foo.rs", "diff": ""} more text"#;
        let json = extract_apply_diff_json(response).unwrap();
        assert!(json.contains("path"));
    }

    #[test]
    fn no_json_returns_none() {
        let response = "I am done, no more edits needed.";
        assert!(extract_apply_diff_json(response).is_none());
    }
}
