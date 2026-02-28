//! NS-1.2: Shared mode runner traits and orchestrator interfaces.
//!
//! The three new operating modes (Contextual, Deepthink, Agentic) each
//! implement `ModeRunner`.  The `ModeOrchestrator` drives a single mode run
//! while managing cancellation, telemetry hooks, and budget enforcement.
//!
//! ## Lifecycle
//!
//! ```text
//! ModeOrchestrator::run(runner, request)
//!   → runner.prepare(ctx)         — one-time setup; build agents, validate config
//!   → loop:
//!       runner.step(ctx, state)   — execute one iteration; return next state
//!       check budget / cancellation
//!   → runner.finish(ctx, outcome) — cleanup, flush telemetry
//! ```
//!
//! Implementors only need to provide `prepare` and `step`; `finish` has a
//! no-op default.

use std::sync::Arc;

use async_trait::async_trait;

use crate::modes::{
    errors::OrchestrationError,
    provider_config::ModeRunnerConfig,
    types::{Artifact, ModeOutcome},
};

// ── Request / Context ────────────────────────────────────────────────────────

/// Input request handed to a mode runner.
#[derive(Debug, Clone)]
pub struct ModeRequest {
    /// The task description / prompt the mode should act on.
    pub task: String,
    /// Optional initial artifact to refine (for resuming prior work).
    pub initial_artifact: Option<Artifact>,
    /// Human-readable label for telemetry / logging (e.g. issue ID).
    pub label: String,
}

impl ModeRequest {
    pub fn new(task: impl Into<String>) -> Self {
        Self {
            task: task.into(),
            initial_artifact: None,
            label: String::new(),
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn with_initial_artifact(mut self, artifact: Artifact) -> Self {
        self.initial_artifact = Some(artifact);
        self
    }
}

/// Shared execution context passed to every `step` call.
///
/// All fields behind `Arc` so `step` can be `async` and futures are `'static`.
#[derive(Clone)]
pub struct ModeContext {
    /// Resolved runtime configuration (models, endpoints, budget).
    pub config: Arc<ModeRunnerConfig>,
    /// Human-readable run label (issue ID, task name, etc.).
    pub label: Arc<str>,
    /// Cancellation token — runners must check this at the top of each step.
    pub cancel: Arc<tokio_util::sync::CancellationToken>,
}

impl ModeContext {
    pub fn new(config: ModeRunnerConfig, label: impl Into<Arc<str>>) -> Self {
        Self {
            config: Arc::new(config),
            label: label.into(),
            cancel: Arc::new(tokio_util::sync::CancellationToken::new()),
        }
    }

    /// Returns `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Cancel this run.
    pub fn cancel(&self) {
        self.cancel.cancel();
    }
}

// ── StepResult ───────────────────────────────────────────────────────────────

/// Return value from a single `ModeRunner::step` call.
#[derive(Debug)]
pub enum StepResult<S> {
    /// Continue to the next iteration with the given state.
    Continue(S),
    /// Mode has reached a terminal success state.
    Done(ModeOutcome),
    /// Mode has reached a terminal failure state (after exhausting retries).
    Failed(OrchestrationError),
}

// ── ModeRunner trait ─────────────────────────────────────────────────────────

/// Core trait that all three new modes implement.
///
/// `S` is the opaque state type the mode uses to thread information between
/// steps (e.g. `ContextualState`, `DeepthinkState`, `AgenticState`).
///
/// Implementors are not required to be `Clone`; the orchestrator holds a single
/// owned instance for the lifetime of the run.
#[async_trait]
pub trait ModeRunner: Send + Sync {
    /// Mode name for logging and telemetry (e.g. `"contextual"`, `"deepthink"`).
    fn name(&self) -> &'static str;

    /// One-time setup before the step loop begins.
    ///
    /// Build agent instances, validate configuration, initialize state.
    /// Returns the initial opaque state `S`.
    ///
    /// # Errors
    ///
    /// Returns `OrchestrationError::Configuration` if the config is invalid.
    async fn prepare(
        &mut self,
        ctx: &ModeContext,
        request: &ModeRequest,
    ) -> Result<(), OrchestrationError>;

    /// Execute one iteration of the mode's inner loop.
    ///
    /// The mode is responsible for advancing its own internal state.
    /// Returns `StepResult::Continue` to keep looping or a terminal variant.
    async fn step(&mut self, ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError>;

    /// Optional cleanup hook called after the run reaches any terminal state.
    async fn finish(&mut self, _ctx: &ModeContext, _outcome: &ModeOutcome) {}
}

// ── ModeOrchestrator ─────────────────────────────────────────────────────────

/// Drives a `ModeRunner` through its lifecycle and enforces budget constraints.
pub struct ModeOrchestrator {
    config: ModeRunnerConfig,
}

impl ModeOrchestrator {
    pub fn new(config: ModeRunnerConfig) -> Self {
        Self { config }
    }

    pub fn with_defaults() -> Self {
        Self::new(ModeRunnerConfig::default())
    }

    /// Execute a mode run to completion (or budget exhaustion / cancellation).
    pub async fn run(&self, runner: &mut dyn ModeRunner, request: ModeRequest) -> ModeOutcome {
        let label = request.label.clone();
        let ctx = ModeContext::new(self.config.clone(), label.as_str());

        tracing::info!(mode = runner.name(), label = %label, "mode run starting");

        // --- prepare ---
        if let Err(e) = runner.prepare(&ctx, &request).await {
            tracing::error!(mode = runner.name(), error = %e, "prepare failed");
            let outcome = ModeOutcome::Failure {
                reason: e.to_string(),
                iterations: 0,
                partial_artifact: None,
            };
            runner.finish(&ctx, &outcome).await;
            return outcome;
        }

        // --- step loop ---
        let mut iterations: u32 = 0;
        let outcome = loop {
            // Budget check
            if iterations >= self.config.max_iterations {
                tracing::warn!(mode = runner.name(), iterations, "max iterations reached");
                break ModeOutcome::Failure {
                    reason: format!("max iterations ({}) exceeded", self.config.max_iterations),
                    iterations,
                    partial_artifact: None,
                };
            }

            // Cancellation check
            if ctx.is_cancelled() {
                tracing::info!(mode = runner.name(), "run cancelled");
                break ModeOutcome::Failure {
                    reason: "cancelled".to_string(),
                    iterations,
                    partial_artifact: None,
                };
            }

            iterations += 1;

            match runner.step(&ctx).await {
                Ok(StepResult::Continue(())) => {
                    // keep looping
                }
                Ok(StepResult::Done(outcome)) => {
                    tracing::info!(mode = runner.name(), iterations, "mode run succeeded");
                    break outcome;
                }
                Ok(StepResult::Failed(e)) => {
                    tracing::error!(
                        mode = runner.name(),
                        iterations,
                        error = %e,
                        "mode run failed"
                    );
                    break ModeOutcome::Failure {
                        reason: e.to_string(),
                        iterations,
                        partial_artifact: None,
                    };
                }
                Err(e) => {
                    tracing::error!(
                        mode = runner.name(),
                        iterations,
                        error = %e,
                        "step returned error"
                    );
                    if !e.is_retriable() {
                        break ModeOutcome::Failure {
                            reason: e.to_string(),
                            iterations,
                            partial_artifact: None,
                        };
                    }
                    // retriable — log and continue (iteration counter already advanced)
                }
            }
        };

        runner.finish(&ctx, &outcome).await;
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modes::types::Artifact;

    struct AlwaysSucceedRunner {
        steps_until_done: u32,
        steps_taken: u32,
    }

    #[async_trait]
    impl ModeRunner for AlwaysSucceedRunner {
        fn name(&self) -> &'static str {
            "always_succeed"
        }

        async fn prepare(
            &mut self,
            _ctx: &ModeContext,
            _request: &ModeRequest,
        ) -> Result<(), OrchestrationError> {
            Ok(())
        }

        async fn step(&mut self, _ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
            self.steps_taken += 1;
            if self.steps_taken >= self.steps_until_done {
                Ok(StepResult::Done(ModeOutcome::Success {
                    artifact: Artifact::new("fn main() {}"),
                    iterations: self.steps_taken,
                    total_tokens: None,
                }))
            } else {
                Ok(StepResult::Continue(()))
            }
        }
    }

    struct AlwaysFailRunner;

    #[async_trait]
    impl ModeRunner for AlwaysFailRunner {
        fn name(&self) -> &'static str {
            "always_fail"
        }

        async fn prepare(
            &mut self,
            _ctx: &ModeContext,
            _request: &ModeRequest,
        ) -> Result<(), OrchestrationError> {
            Ok(())
        }

        async fn step(&mut self, _ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
            Ok(StepResult::Failed(OrchestrationError::MaxIterations(0)))
        }
    }

    #[tokio::test]
    async fn orchestrator_succeeds_after_n_steps() {
        let mut runner = AlwaysSucceedRunner {
            steps_until_done: 3,
            steps_taken: 0,
        };
        let orch = ModeOrchestrator::with_defaults();
        let outcome = orch.run(&mut runner, ModeRequest::new("test task")).await;
        assert!(outcome.is_success());
        assert_eq!(outcome.iterations(), 3);
    }

    #[tokio::test]
    async fn orchestrator_hits_budget_limit() {
        let mut cfg = ModeRunnerConfig::default();
        cfg.max_iterations = 2;

        let mut runner = AlwaysSucceedRunner {
            steps_until_done: 100,
            steps_taken: 0,
        };
        let orch = ModeOrchestrator::new(cfg);
        let outcome = orch.run(&mut runner, ModeRequest::new("test task")).await;
        assert!(!outcome.is_success());
    }

    #[tokio::test]
    async fn orchestrator_propagates_failure() {
        let mut runner = AlwaysFailRunner;
        let orch = ModeOrchestrator::with_defaults();
        let outcome = orch.run(&mut runner, ModeRequest::new("test task")).await;
        assert!(!outcome.is_success());
    }
}
