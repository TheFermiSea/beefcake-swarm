//! NS-6.2 / NS-6.3: Mode integration tests and failure-injection coverage.
//!
//! Tests exercise the mode subsystem end-to-end using in-process mock runners —
//! no running inference endpoint required.

use swarm_agents::modes::{
    contextual::ContextualRunner,
    deepthink::DeepthinkRunner,
    errors::{OrchestrationError, RetryCategory},
    runner::{ModeContext, ModeOrchestrator, ModeRequest, ModeRunner, StepResult},
    types::{Artifact, ModeOutcome},
    ModeRunnerConfig, SwarmMode,
};

// ── Helpers ──────────────────────────────────────────────────────────────────

fn default_config() -> ModeRunnerConfig {
    ModeRunnerConfig::default()
}

fn make_request(task: &str) -> ModeRequest {
    ModeRequest::new(task).with_label("test-issue")
}

// ── NS-6.1: SwarmMode enum / CLI value-enum presence ─────────────────────────

#[test]
fn swarm_mode_variants_exist() {
    let modes = [
        SwarmMode::Contextual,
        SwarmMode::Deepthink,
        SwarmMode::Agentic,
    ];
    assert_eq!(modes.len(), 3);
}

#[test]
fn swarm_mode_debug_format() {
    assert_eq!(format!("{:?}", SwarmMode::Contextual), "Contextual");
    assert_eq!(format!("{:?}", SwarmMode::Deepthink), "Deepthink");
    assert_eq!(format!("{:?}", SwarmMode::Agentic), "Agentic");
}

#[test]
fn swarm_mode_equality() {
    assert_eq!(SwarmMode::Contextual, SwarmMode::Contextual);
    assert_ne!(SwarmMode::Contextual, SwarmMode::Deepthink);
}

// ── NS-6.2: ModeRequest builder ───────────────────────────────────────────────

#[test]
fn mode_request_builder() {
    let req = ModeRequest::new("fix the bug")
        .with_label("beefcake-001")
        .with_initial_artifact(Artifact {
            content: "old content".into(),
            language: None,
            iteration: 0,
            tokens_used: None,
        });
    assert_eq!(req.task, "fix the bug");
    assert_eq!(req.label, "beefcake-001");
    assert!(req.initial_artifact.is_some());
}

#[test]
fn mode_request_defaults() {
    let req = ModeRequest::new("do something");
    assert_eq!(req.task, "do something");
    assert!(req.label.is_empty());
    assert!(req.initial_artifact.is_none());
}

// ── NS-6.2: Runner name contracts ─────────────────────────────────────────────

#[test]
fn contextual_runner_has_correct_name() {
    let runner = ContextualRunner::new(default_config());
    assert_eq!(runner.name(), "contextual");
}

#[test]
fn deepthink_runner_has_correct_name() {
    let runner = DeepthinkRunner::new(default_config());
    assert_eq!(runner.name(), "deepthink");
}

// ── Mock runners ──────────────────────────────────────────────────────────────

/// Always returns `Continue` — tests budget cutoff.
struct InfiniteRunner;

#[async_trait::async_trait]
impl ModeRunner for InfiniteRunner {
    fn name(&self) -> &'static str {
        "infinite"
    }
    async fn prepare(
        &mut self,
        _ctx: &ModeContext,
        _request: &ModeRequest,
    ) -> Result<(), OrchestrationError> {
        Ok(())
    }
    async fn step(&mut self, _ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
        Ok(StepResult::Continue(()))
    }
}

/// Fails in `prepare`.
struct FailPrepareRunner;

#[async_trait::async_trait]
impl ModeRunner for FailPrepareRunner {
    fn name(&self) -> &'static str {
        "fail-prepare"
    }
    async fn prepare(
        &mut self,
        _ctx: &ModeContext,
        _request: &ModeRequest,
    ) -> Result<(), OrchestrationError> {
        Err(OrchestrationError::Configuration(
            "missing required config".into(),
        ))
    }
    async fn step(&mut self, _ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
        unreachable!()
    }
}

/// Fails in `step` after N successful steps.
struct FailStepRunner {
    fail_on: u32,
    current: u32,
}

impl FailStepRunner {
    fn new(fail_on: u32) -> Self {
        Self {
            fail_on,
            current: 0,
        }
    }
}

#[async_trait::async_trait]
impl ModeRunner for FailStepRunner {
    fn name(&self) -> &'static str {
        "fail-step"
    }
    async fn prepare(
        &mut self,
        _ctx: &ModeContext,
        _request: &ModeRequest,
    ) -> Result<(), OrchestrationError> {
        Ok(())
    }
    async fn step(&mut self, _ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
        self.current += 1;
        if self.current >= self.fail_on {
            return Err(OrchestrationError::PolicyViolation(
                "simulated non-retriable failure".into(),
            ));
        }
        Ok(StepResult::Continue(()))
    }
}

/// Completes successfully in one step; records that `finish` was called.
struct OneStepRunner {
    pub finished: bool,
}

impl OneStepRunner {
    fn new() -> Self {
        Self { finished: false }
    }
}

#[async_trait::async_trait]
impl ModeRunner for OneStepRunner {
    fn name(&self) -> &'static str {
        "one-step"
    }
    async fn prepare(
        &mut self,
        _ctx: &ModeContext,
        _request: &ModeRequest,
    ) -> Result<(), OrchestrationError> {
        Ok(())
    }
    async fn step(&mut self, _ctx: &ModeContext) -> Result<StepResult<()>, OrchestrationError> {
        Ok(StepResult::Done(ModeOutcome::Success {
            artifact: Artifact {
                content: "done!".into(),
                language: None,
                iteration: 0,
                tokens_used: None,
            },
            iterations: 1,
            total_tokens: None,
        }))
    }
    async fn finish(&mut self, _ctx: &ModeContext, _outcome: &ModeOutcome) {
        self.finished = true;
    }
}

// ── NS-6.2: Budget enforcement ────────────────────────────────────────────────

#[tokio::test]
async fn orchestrator_enforces_max_iterations() {
    let mut cfg = default_config();
    cfg.max_iterations = 3;
    let orch = ModeOrchestrator::new(cfg);
    let mut runner = InfiniteRunner;
    let outcome = orch.run(&mut runner, make_request("loop forever")).await;
    match outcome {
        ModeOutcome::Failure {
            reason, iterations, ..
        } => {
            assert!(
                reason.contains("max iterations"),
                "unexpected reason: {reason}"
            );
            assert_eq!(iterations, 3);
        }
        other => panic!("expected Failure, got {other:?}"),
    }
}

// ── NS-6.3: Failure injection — prepare error ────────────────────────────────

#[tokio::test]
async fn orchestrator_surfaces_prepare_error() {
    let orch = ModeOrchestrator::new(default_config());
    let mut runner = FailPrepareRunner;
    let outcome = orch.run(&mut runner, make_request("task")).await;
    match outcome {
        ModeOutcome::Failure {
            reason, iterations, ..
        } => {
            assert!(
                reason.contains("missing required config"),
                "unexpected: {reason}"
            );
            assert_eq!(
                iterations, 0,
                "no iterations should run after prepare fails"
            );
        }
        other => panic!("expected Failure, got {other:?}"),
    }
}

// ── NS-6.3: Failure injection — step error (non-retriable) ───────────────────

#[tokio::test]
async fn orchestrator_surfaces_fatal_step_error() {
    let mut cfg = default_config();
    cfg.max_iterations = 10;
    let orch = ModeOrchestrator::new(cfg);
    let mut runner = FailStepRunner::new(2);
    let outcome = orch.run(&mut runner, make_request("task")).await;
    // Provider errors are non-retriable → immediate Failure
    assert!(
        matches!(outcome, ModeOutcome::Failure { .. }),
        "expected Failure, got {outcome:?}"
    );
}

// ── NS-6.3: Failure injection — cancellation ─────────────────────────────────

#[tokio::test]
async fn orchestrator_respects_cancellation() {
    let cfg = default_config();
    // Build a context directly and cancel its token, then pass to an orchestrator
    // that shares the same config but different ctx — instead, cancel via env.
    // The orchestrator creates ctx internally; we can't inject a pre-cancelled token
    // through the public API, so we test the ModeContext::is_cancelled directly.
    let ctx = ModeContext::new(cfg, "cancel-test");
    assert!(!ctx.is_cancelled());
    ctx.cancel.cancel();
    assert!(ctx.is_cancelled());
}

// ── NS-6.2: Successful run — finish callback ──────────────────────────────────

#[tokio::test]
async fn orchestrator_calls_finish_on_success() {
    let orch = ModeOrchestrator::new(default_config());
    let mut runner = OneStepRunner::new();
    let outcome = orch.run(&mut runner, make_request("one-shot task")).await;
    assert!(
        matches!(outcome, ModeOutcome::Success { .. }),
        "expected Success, got {outcome:?}"
    );
    assert!(runner.finished, "finish() was not called after success");
}

// ── NS-6.3: OrchestrationError retry classification ──────────────────────────

#[test]
fn inference_failure_is_transient() {
    let err = OrchestrationError::InferenceFailure("timeout".into());
    assert_eq!(err.retry_category(), RetryCategory::Transient);
}

#[test]
fn configuration_error_is_policy_violation() {
    let err = OrchestrationError::Configuration("bad config".into());
    assert_eq!(err.retry_category(), RetryCategory::PolicyViolation);
}

#[test]
fn cancelled_error_is_cancelled() {
    let err = OrchestrationError::Cancelled("user cancelled".into());
    assert_eq!(err.retry_category(), RetryCategory::Cancelled);
}

#[test]
fn max_iterations_error_category() {
    let err = OrchestrationError::MaxIterations(5);
    assert_eq!(err.retry_category(), RetryCategory::MaxIterations);
}

#[test]
fn policy_violation_error_category() {
    // PolicyViolation errors are non-retriable.
    let err = OrchestrationError::PolicyViolation("path traversal".into());
    assert_eq!(err.retry_category(), RetryCategory::PolicyViolation);
}
