//! NS-2: Contextual Mode finite-state-machine orchestrator.
//!
//! Implements a long-running iterative refinement loop using a typed Rust FSM:
//!
//! ```text
//! Drafting → Critiquing → Condensing (optional) → Drafting …
//!                       ↓ APPROVED
//!                     Done
//! any state → Error (terminal)
//! ```
//!
//! ## Agent roles
//!
//! | Agent     | Role                                                  |
//! |-----------|-------------------------------------------------------|
//! | Generator | Produces code artifacts from task + critique feedback |
//! | Critique  | Evaluates artifact; returns APPROVED or actionable feedback |
//! | Compactor | Summarises chat history when context budget is tight   |
//!
//! ## Usage
//!
//! ```rust,no_run
//! use swarm_agents::modes::{ModeOrchestrator, ModeRequest, ModeRunnerConfig};
//! use swarm_agents::modes::contextual::ContextualRunner;
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = ModeRunnerConfig::default();
//!     let mut runner = ContextualRunner::new(config.clone());
//!     let orch = ModeOrchestrator::new(config);
//!     let outcome = orch.run(&mut runner, ModeRequest::new("Implement a ring buffer in Rust")).await;
//! }
//! ```

use std::collections::VecDeque;

use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::{Chat, Message};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::modes::{
    errors::OrchestrationError,
    provider_config::ModeRunnerConfig,
    runner::{ModeContext, ModeRequest, ModeRunner, StepResult},
    types::{Artifact, CompactionSummary, CritiqueVerdict, ModeOutcome},
};

// ── State ────────────────────────────────────────────────────────────────────

/// All states in the Contextual Mode FSM.
///
/// Variants carry the exact data required for that phase — no other state
/// is accessible, eliminating a class of orchestration bugs at compile time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContextualState {
    /// Generator agent is producing / refining an artifact.
    Drafting {
        task_prompt: String,
        iteration: u32,
    },
    /// Critique agent is evaluating the latest artifact.
    Critiquing {
        task_prompt: String,
        artifact: Artifact,
        iteration: u32,
    },
    /// Memory agent is compressing history before the next draft.
    Condensing {
        /// State to resume after compaction completes.
        resume_task_prompt: String,
        resume_iteration: u32,
        compression_reason: String,
    },
    /// Terminal success — final artifact is ready.
    Done {
        artifact: Artifact,
        total_iterations: u32,
    },
    /// Terminal failure — budget exceeded or unrecoverable error.
    Error {
        reason: String,
        last_iteration: u32,
        partial_artifact: Option<Artifact>,
    },
}

impl ContextualState {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done { .. } | Self::Error { .. })
    }
}

// ── ContextualRunner ─────────────────────────────────────────────────────────

/// Implements the Contextual Mode FSM as a `ModeRunner`.
pub struct ContextualRunner {
    config: ModeRunnerConfig,
    /// Current FSM state.
    state: ContextualState,
    /// Rolling conversation history (trimmed by Condensing step).
    history: VecDeque<Message>,
    /// Approximate token count for the current history.
    estimated_tokens: u64,
}

impl ContextualRunner {
    pub fn new(config: ModeRunnerConfig) -> Self {
        Self {
            config,
            state: ContextualState::Drafting {
                task_prompt: String::new(),
                iteration: 0,
            },
            history: VecDeque::new(),
            estimated_tokens: 0,
        }
    }

    /// Rough token estimation: ~1 token per 4 chars.
    fn estimate_tokens(s: &str) -> u64 {
        (s.len() as u64) / 4
    }

    fn total_estimated_tokens(&self) -> u64 {
        self.estimated_tokens
    }

    fn should_compact(&self) -> bool {
        self.total_estimated_tokens()
            >= self.config.compaction.trigger_threshold_tokens()
            && self.history.len() >= self.config.compaction.min_messages_before_compaction
    }

    fn push_history(&mut self, msg: Message) {
        let text = extract_message_text(&msg);
        let token_est = Self::estimate_tokens(&text);
        self.estimated_tokens += token_est;
        self.history.push_back(msg);
    }

    fn history_vec(&self) -> Vec<Message> {
        self.history.iter().cloned().collect()
    }

    /// Apply a compaction summary by replacing old history with the summary message.
    fn apply_compaction(&mut self, summary: &CompactionSummary) {
        // Keep the most recent 2 messages (latest exchange) intact.
        let keep = 2.min(self.history.len());
        let tail: Vec<Message> = self.history.iter().rev().take(keep).cloned().collect();
        self.history.clear();
        self.estimated_tokens = 0;

        // Insert summary as a system-style user message.
        let summary_msg = Message::user(format!(
            "[CONTEXT SUMMARY — {} tokens compacted]\n{}",
            summary.tokens_compacted, summary.summary
        ));
        self.push_history(summary_msg);

        // Re-add the tail (reversed back to chronological).
        for msg in tail.into_iter().rev() {
            self.push_history(msg);
        }
    }

    // ── Step implementations ──────────────────────────────────────────────

    async fn step_drafting(
        &mut self,
        task_prompt: String,
        iteration: u32,
    ) -> Result<ContextualState, OrchestrationError> {
        if iteration >= self.config.max_iterations {
            return Ok(ContextualState::Error {
                reason: format!("max iterations ({}) exceeded", self.config.max_iterations),
                last_iteration: iteration,
                partial_artifact: None,
            });
        }

        debug!(iteration, "drafting artifact");

        let client = self.config.local_client().map_err(|e| {
            OrchestrationError::Configuration(format!("failed to build local client: {e}"))
        })?;

        let agent = client
            .agent(&self.config.models.generator)
            .preamble(
                "You are an expert Rust software engineer. \
                Generate precise, idiomatic, and correct code based on the provided task. \
                Respond with code only — no prose unless specifically requested.",
            )
            .temperature(self.config.generator_temperature)
            .build();

        let history = self.history_vec();
        let response = agent
            .chat(&task_prompt, history)
            .await
            .map_err(|e| OrchestrationError::InferenceFailure(e.to_string()))?;

        self.push_history(Message::user(task_prompt.clone()));
        self.push_history(Message::assistant(response.clone()));

        let artifact = Artifact::new(response)
            .with_language("rust")
            .with_iteration(iteration);

        Ok(ContextualState::Critiquing {
            task_prompt,
            artifact,
            iteration,
        })
    }

    async fn step_critiquing(
        &mut self,
        task_prompt: String,
        artifact: Artifact,
        iteration: u32,
    ) -> Result<ContextualState, OrchestrationError> {
        debug!(iteration, "critiquing artifact");

        let client = self.config.local_client().map_err(|e| {
            OrchestrationError::Configuration(format!("failed to build local client: {e}"))
        })?;

        let agent = client
            .agent(&self.config.models.critique)
            .preamble(
                "You are a ruthless code reviewer. \
                Analyze the provided code for correctness, safety, and architecture. \
                If the code is acceptable, reply EXACTLY with the single word: APPROVED\n\
                Otherwise list specific, actionable fixes required. Be terse.",
            )
            .temperature(self.config.critic_temperature)
            .build();

        let critique_prompt = format!(
            "Review this code:\n\n```rust\n{}\n```",
            artifact.content
        );
        let history = self.history_vec();
        let critique_raw = agent
            .chat(&critique_prompt, history)
            .await
            .map_err(|e| OrchestrationError::InferenceFailure(e.to_string()))?;

        self.push_history(Message::user(critique_prompt));
        self.push_history(Message::assistant(critique_raw.clone()));

        let verdict = CritiqueVerdict::from_llm_response(&critique_raw);

        match verdict {
            CritiqueVerdict::Approved => {
                info!(iteration, "critique approved — artifact accepted");
                Ok(ContextualState::Done {
                    artifact,
                    total_iterations: iteration + 1,
                })
            }
            CritiqueVerdict::NeedsRevision { feedback, .. } => {
                let next_prompt = format!(
                    "The previous implementation did not pass review. \
                    Address ALL of the following feedback:\n\n{feedback}\n\n\
                    Original task:\n{task_prompt}"
                );
                // Check if we should compact before the next draft.
                if self.should_compact() {
                    warn!(
                        tokens = self.total_estimated_tokens(),
                        threshold = self.config.compaction.trigger_threshold_tokens(),
                        "context approaching limit — triggering compaction"
                    );
                    Ok(ContextualState::Condensing {
                        resume_task_prompt: next_prompt,
                        resume_iteration: iteration + 1,
                        compression_reason: "context window threshold reached".to_string(),
                    })
                } else {
                    Ok(ContextualState::Drafting {
                        task_prompt: next_prompt,
                        iteration: iteration + 1,
                    })
                }
            }
            CritiqueVerdict::Rejected { reason } => Ok(ContextualState::Error {
                reason: format!("critique rejected artifact: {reason}"),
                last_iteration: iteration,
                partial_artifact: Some(artifact),
            }),
        }
    }

    async fn step_condensing(
        &mut self,
        resume_task_prompt: String,
        resume_iteration: u32,
        compression_reason: String,
    ) -> Result<ContextualState, OrchestrationError> {
        info!(resume_iteration, reason = %compression_reason, "compacting context");

        let client = self.config.local_client().map_err(|e| {
            OrchestrationError::Configuration(format!("failed to build local client: {e}"))
        })?;

        let agent = client
            .agent(&self.config.models.compactor)
            .preamble(
                "You are a context compressor. Summarize the following technical dialogue \
                into dense bullet points. Preserve all constraints, data structures, past \
                errors, and key decisions. Discard pleasantries.",
            )
            .temperature(0.2)
            .build();

        let history_text: String = self
            .history
            .iter()
            .map(|m| {
                let text = extract_message_text(m);
                match m {
                    Message::User { .. } => format!("User: {text}"),
                    Message::Assistant { .. } => format!("Assistant: {text}"),
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let tokens_before = self.total_estimated_tokens();
        let messages_before = self.history.len();

        let summary_text = agent
            .chat(
                &format!("Summarize this conversation:\n\n{history_text}"),
                vec![],
            )
            .await
            .map_err(|e| OrchestrationError::InferenceFailure(e.to_string()))?;

        let summary = CompactionSummary {
            summary: summary_text.clone(),
            tokens_compacted: tokens_before,
            tokens_summary: Self::estimate_tokens(&summary_text),
            messages_compacted: messages_before,
        };

        self.apply_compaction(&summary);

        info!(
            ratio = format!("{:.2}", summary.compression_ratio()),
            "compaction complete"
        );

        Ok(ContextualState::Drafting {
            task_prompt: resume_task_prompt,
            iteration: resume_iteration,
        })
    }
}

#[async_trait]
impl ModeRunner for ContextualRunner {
    fn name(&self) -> &'static str {
        "contextual"
    }

    async fn prepare(
        &mut self,
        _ctx: &ModeContext,
        request: &ModeRequest,
    ) -> Result<(), OrchestrationError> {
        self.config.validate().map_err(OrchestrationError::Configuration)?;
        self.history.clear();
        self.estimated_tokens = 0;
        self.state = ContextualState::Drafting {
            task_prompt: request.task.clone(),
            iteration: 0,
        };
        // Seed with initial artifact if resuming.
        if let Some(initial) = &request.initial_artifact {
            self.push_history(Message::user(request.task.clone()));
            self.push_history(Message::assistant(initial.content.clone()));
        }
        Ok(())
    }

    async fn step(&mut self, ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
        if ctx.is_cancelled() {
            return Ok(StepResult::Failed(OrchestrationError::Cancelled(
                "cancelled by caller".to_string(),
            )));
        }

        let current = self.state.clone();
        let next = match current {
            ContextualState::Drafting { task_prompt, iteration } => {
                self.step_drafting(task_prompt, iteration).await?
            }
            ContextualState::Critiquing { task_prompt, artifact, iteration } => {
                self.step_critiquing(task_prompt, artifact, iteration).await?
            }
            ContextualState::Condensing {
                resume_task_prompt,
                resume_iteration,
                compression_reason,
            } => {
                self.step_condensing(resume_task_prompt, resume_iteration, compression_reason)
                    .await?
            }
            ContextualState::Done { artifact, total_iterations } => {
                return Ok(StepResult::Done(ModeOutcome::Success {
                    artifact,
                    iterations: total_iterations,
                    total_tokens: Some(self.estimated_tokens),
                }));
            }
            ContextualState::Error { reason: _, last_iteration, partial_artifact: _ } => {
                return Ok(StepResult::Failed(OrchestrationError::MaxIterations(
                    last_iteration,
                )));
            }
        };

        self.state = next;

        if self.state.is_terminal() {
            // Will be picked up on next step call.
            Ok(StepResult::Continue(()))
        } else {
            Ok(StepResult::Continue(()))
        }
    }
}

// ── Public helpers ────────────────────────────────────────────────────────────

/// Run a Contextual Mode task to completion with default configuration.
pub async fn run_contextual(task: impl Into<String>) -> ModeOutcome {
    let config = ModeRunnerConfig::default();
    let mut runner = ContextualRunner::new(config.clone());
    let orch = crate::modes::ModeOrchestrator::new(config);
    orch.run(&mut runner, ModeRequest::new(task)).await
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Extract plain text from a `Message` (user or assistant).
pub(crate) fn extract_message_text(msg: &Message) -> String {
    use rig::completion::AssistantContent;
    use rig::message::UserContent;
    match msg {
        Message::User { content } => content
            .iter()
            .filter_map(|c| {
                if let UserContent::Text(t) = c {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|c| {
                if let AssistantContent::Text(t) = c {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contextual_state_terminal() {
        let done = ContextualState::Done {
            artifact: Artifact::new("ok"),
            total_iterations: 1,
        };
        assert!(done.is_terminal());

        let drafting = ContextualState::Drafting {
            task_prompt: "task".into(),
            iteration: 0,
        };
        assert!(!drafting.is_terminal());
    }

    #[test]
    fn push_history_estimates_tokens() {
        let config = ModeRunnerConfig::default();
        let mut runner = ContextualRunner::new(config);
        runner.push_history(Message::user("hello world"));
        assert!(runner.estimated_tokens > 0);
    }

    #[test]
    fn should_compact_false_when_below_threshold() {
        let mut config = ModeRunnerConfig::default();
        config.compaction.context_window_tokens = 10_000;
        config.compaction.compaction_threshold = 0.75;
        let runner = ContextualRunner::new(config);
        // No messages, so no compaction needed.
        assert!(!runner.should_compact());
    }

    #[test]
    fn apply_compaction_reduces_history() {
        let config = ModeRunnerConfig::default();
        let mut runner = ContextualRunner::new(config);
        for i in 0..10 {
            runner.push_history(Message::user(format!("msg {i}")));
        }
        let before = runner.history.len();
        let summary = CompactionSummary {
            summary: "summary here".into(),
            tokens_compacted: runner.estimated_tokens,
            tokens_summary: 10,
            messages_compacted: before,
        };
        runner.apply_compaction(&summary);
        // Should have summary + 2 tail messages = 3
        assert!(runner.history.len() <= 3);
    }
}
