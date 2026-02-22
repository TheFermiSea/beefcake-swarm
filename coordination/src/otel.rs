//! OpenTelemetry-Compatible Span Helpers
//!
//! Provides structured `tracing` span builders for decision-grade observability
//! across the swarm orchestration pipeline. All spans use dot-notation field
//! names compatible with OpenTelemetry semantic conventions.
//!
//! # Span Hierarchy
//!
//! ```text
//! swarm.process_issue          (root — one per beads issue)
//!   └─ swarm.iteration         (one per implement→verify cycle)
//!       ├─ swarm.agent         (LLM agent invocation)
//!       ├─ swarm.gate          (quality gate: fmt, clippy, check, test, ...)
//!       ├─ swarm.escalation    (tier change decision)
//!       └─ swarm.tool          (individual tool call within agent)
//! swarm.voting                 (ensemble voting round)
//! swarm.arbitration            (dispute resolution)
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use coordination::otel;
//!
//! let span = otel::gate_span("clippy", "beefcake-655", 3);
//! let guard = span.enter();
//! // ... run the gate ...
//! otel::record_gate_result(&span, GateOutcome::Passed, 0, 2, 1450);
//! drop(guard);
//! ```

use serde::{Deserialize, Serialize};
use tracing::Span;

// ── Span Name Constants ──────────────────────────────────────────────

/// Root span for processing a beads issue end-to-end.
pub const SPAN_PROCESS_ISSUE: &str = "swarm.process_issue";

/// One iteration of the implement → verify loop.
pub const SPAN_ITERATION: &str = "swarm.iteration";

/// LLM agent invocation (implementer, reviewer, manager).
pub const SPAN_AGENT: &str = "swarm.agent";

/// Quality gate execution (fmt, clippy, check, test, deny, doc, sg).
pub const SPAN_GATE: &str = "swarm.gate";

/// Tier escalation decision.
pub const SPAN_ESCALATION: &str = "swarm.escalation";

/// Individual tool call within an agent turn.
pub const SPAN_TOOL: &str = "swarm.tool";

/// Ensemble voting round.
pub const SPAN_VOTING: &str = "swarm.voting";

/// Dispute arbitration.
pub const SPAN_ARBITRATION: &str = "swarm.arbitration";

// ── Field Name Constants ─────────────────────────────────────────────
// Using OpenTelemetry-style dot notation for structured export.

pub const FIELD_ISSUE_ID: &str = "issue.id";
pub const FIELD_ITERATION: &str = "swarm.iteration.number";
pub const FIELD_TIER: &str = "swarm.tier";
pub const FIELD_MODEL: &str = "swarm.model";
pub const FIELD_GATE_NAME: &str = "swarm.gate.name";
pub const FIELD_GATE_OUTCOME: &str = "swarm.gate.outcome";
pub const FIELD_ERROR_COUNT: &str = "swarm.error_count";
pub const FIELD_WARNING_COUNT: &str = "swarm.warning_count";
pub const FIELD_DURATION_MS: &str = "swarm.duration_ms";
pub const FIELD_SUCCESS: &str = "swarm.success";
pub const FIELD_TOOL_NAME: &str = "swarm.tool.name";
pub const FIELD_AGENT_ROLE: &str = "swarm.agent.role";
pub const FIELD_FROM_TIER: &str = "swarm.escalation.from_tier";
pub const FIELD_TO_TIER: &str = "swarm.escalation.to_tier";
pub const FIELD_ESCALATION_REASON: &str = "swarm.escalation.reason";
pub const FIELD_VOTE_STRATEGY: &str = "swarm.vote.strategy";
pub const FIELD_VOTE_OUTCOME: &str = "swarm.vote.outcome";
pub const FIELD_MODEL_COUNT: &str = "swarm.vote.model_count";
pub const FIELD_TOKENS_USED: &str = "swarm.tokens_used";

// ── Agent Role ───────────────────────────────────────────────────────

/// Agent role classification for span tagging.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// Fast implementer (14B-class).
    Implementer,
    /// Reasoning model (72B-class).
    Architect,
    /// General code generation.
    Coder,
    /// Cloud manager (Opus/Sonnet-class).
    Manager,
    /// Validation reviewer (blind review).
    Reviewer,
    /// Planning agent.
    Planner,
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Implementer => write!(f, "implementer"),
            Self::Architect => write!(f, "architect"),
            Self::Coder => write!(f, "coder"),
            Self::Manager => write!(f, "manager"),
            Self::Reviewer => write!(f, "reviewer"),
            Self::Planner => write!(f, "planner"),
        }
    }
}

// ── Span Builders ────────────────────────────────────────────────────

/// Create a root span for processing a beads issue.
///
/// Fields filled at creation: `issue.id`.
/// Fields filled later via [`record_process_result`]: `swarm.success`, `swarm.iteration.number`.
pub fn process_issue_span(issue_id: &str) -> Span {
    tracing::info_span!(
        "swarm.process_issue",
        "issue.id" = %issue_id,
        "swarm.success" = tracing::field::Empty,
        "swarm.iteration.number" = tracing::field::Empty,
        "swarm.duration_ms" = tracing::field::Empty,
    )
}

/// Record the final result on a process_issue span.
pub fn record_process_result(span: &Span, success: bool, total_iterations: u32, duration_ms: u64) {
    span.record("swarm.success", success);
    span.record("swarm.iteration.number", total_iterations);
    span.record("swarm.duration_ms", duration_ms);
}

/// Create a span for one iteration of the implement → verify loop.
///
/// Fields filled at creation: `issue.id`, `swarm.iteration.number`, `swarm.tier`.
/// Fields filled later via [`record_iteration_result`]: `swarm.success`, error/warning counts.
pub fn iteration_span(issue_id: &str, iteration: u32, tier: &str) -> Span {
    tracing::info_span!(
        "swarm.iteration",
        "issue.id" = %issue_id,
        "swarm.iteration.number" = iteration,
        "swarm.tier" = %tier,
        "swarm.success" = tracing::field::Empty,
        "swarm.error_count" = tracing::field::Empty,
        "swarm.warning_count" = tracing::field::Empty,
        "swarm.duration_ms" = tracing::field::Empty,
    )
}

/// Record the result of an iteration.
pub fn record_iteration_result(
    span: &Span,
    all_green: bool,
    error_count: usize,
    warning_count: usize,
    duration_ms: u64,
) {
    span.record("swarm.success", all_green);
    span.record("swarm.error_count", error_count as u64);
    span.record("swarm.warning_count", warning_count as u64);
    span.record("swarm.duration_ms", duration_ms);
}

/// Create a span for an LLM agent invocation.
///
/// Fields filled at creation: `swarm.agent.role`, `swarm.model`, `swarm.tier`.
/// Fields filled later via [`record_agent_result`]: `swarm.success`, timing, tokens.
pub fn agent_span(role: AgentRole, model: &str, tier: &str) -> Span {
    tracing::info_span!(
        "swarm.agent",
        "swarm.agent.role" = %role,
        "swarm.model" = %model,
        "swarm.tier" = %tier,
        "swarm.success" = tracing::field::Empty,
        "swarm.duration_ms" = tracing::field::Empty,
        "swarm.tokens_used" = tracing::field::Empty,
    )
}

/// Record the result of an agent invocation.
pub fn record_agent_result(span: &Span, success: bool, duration_ms: u64, tokens_used: u64) {
    span.record("swarm.success", success);
    span.record("swarm.duration_ms", duration_ms);
    span.record("swarm.tokens_used", tokens_used);
}

/// Create a span for a quality gate execution.
///
/// Fields filled at creation: `swarm.gate.name`, `issue.id`, `swarm.iteration.number`.
/// Fields filled later via [`record_gate_result`]: outcome, errors, warnings, duration.
pub fn gate_span(gate_name: &str, issue_id: &str, iteration: u32) -> Span {
    tracing::info_span!(
        "swarm.gate",
        "swarm.gate.name" = %gate_name,
        "issue.id" = %issue_id,
        "swarm.iteration.number" = iteration,
        "swarm.gate.outcome" = tracing::field::Empty,
        "swarm.error_count" = tracing::field::Empty,
        "swarm.warning_count" = tracing::field::Empty,
        "swarm.duration_ms" = tracing::field::Empty,
    )
}

/// Record the result of a quality gate.
pub fn record_gate_result(
    span: &Span,
    outcome: &str,
    error_count: usize,
    warning_count: usize,
    duration_ms: u64,
) {
    span.record("swarm.gate.outcome", outcome);
    span.record("swarm.error_count", error_count as u64);
    span.record("swarm.warning_count", warning_count as u64);
    span.record("swarm.duration_ms", duration_ms);
}

/// Create a span for a tier escalation decision.
///
/// All fields filled at creation since escalation is a point-in-time decision.
pub fn escalation_span(
    issue_id: &str,
    from_tier: &str,
    to_tier: &str,
    reason: &str,
    iteration: u32,
) -> Span {
    tracing::info_span!(
        "swarm.escalation",
        "issue.id" = %issue_id,
        "swarm.escalation.from_tier" = %from_tier,
        "swarm.escalation.to_tier" = %to_tier,
        "swarm.escalation.reason" = %reason,
        "swarm.iteration.number" = iteration,
    )
}

/// Create a span for a tool call within an agent turn.
///
/// Fields filled at creation: `swarm.tool.name`, `swarm.agent.role`.
/// Fields filled later via [`record_tool_result`]: `swarm.success`, duration.
pub fn tool_span(tool_name: &str, agent_role: AgentRole) -> Span {
    tracing::info_span!(
        "swarm.tool",
        "swarm.tool.name" = %tool_name,
        "swarm.agent.role" = %agent_role,
        "swarm.success" = tracing::field::Empty,
        "swarm.duration_ms" = tracing::field::Empty,
    )
}

/// Record the result of a tool call.
pub fn record_tool_result(span: &Span, success: bool, duration_ms: u64) {
    span.record("swarm.success", success);
    span.record("swarm.duration_ms", duration_ms);
}

/// Create a span for an ensemble voting round.
///
/// Fields filled at creation: `swarm.vote.strategy`, `swarm.vote.model_count`.
/// Fields filled later via [`record_voting_result`]: outcome, duration.
pub fn voting_span(strategy: &str, model_count: usize) -> Span {
    tracing::info_span!(
        "swarm.voting",
        "swarm.vote.strategy" = %strategy,
        "swarm.vote.model_count" = model_count as u64,
        "swarm.vote.outcome" = tracing::field::Empty,
        "swarm.duration_ms" = tracing::field::Empty,
    )
}

/// Record the result of a voting round.
pub fn record_voting_result(span: &Span, outcome: &str, duration_ms: u64) {
    span.record("swarm.vote.outcome", outcome);
    span.record("swarm.duration_ms", duration_ms);
}

/// Create a span for dispute arbitration.
///
/// Fields filled at creation: `issue.id`, `swarm.vote.model_count`.
/// Fields filled later via [`record_arbitration_result`]: outcome, duration.
pub fn arbitration_span(issue_id: &str, model_count: usize) -> Span {
    tracing::info_span!(
        "swarm.arbitration",
        "issue.id" = %issue_id,
        "swarm.vote.model_count" = model_count as u64,
        "swarm.vote.outcome" = tracing::field::Empty,
        "swarm.duration_ms" = tracing::field::Empty,
    )
}

/// Record the result of an arbitration.
pub fn record_arbitration_result(span: &Span, outcome: &str, duration_ms: u64) {
    span.record("swarm.vote.outcome", outcome);
    span.record("swarm.duration_ms", duration_ms);
}

// ── Convenience: GateResult Integration ──────────────────────────────

/// Record a [`crate::verifier::report::GateResult`] into a gate span.
///
/// This bridges the existing verifier types into OTel spans without
/// requiring the verifier to depend on this module directly.
pub fn record_gate_from_result(
    span: &Span,
    gate: &str,
    outcome: &str,
    error_count: usize,
    warning_count: usize,
    duration_ms: u64,
) {
    let _ = gate; // gate name is already in the span from creation
    record_gate_result(span, outcome, error_count, warning_count, duration_ms);
}

// ── Batch Span Summary ───────────────────────────────────────────────

/// Summary of span activity for a single issue processing session.
///
/// Useful for telemetry aggregation and post-run analysis.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SpanSummary {
    /// Total number of iteration spans created.
    pub iterations: u32,
    /// Total number of gate spans created.
    pub gates: u32,
    /// Number of gates that passed.
    pub gates_passed: u32,
    /// Number of gates that failed.
    pub gates_failed: u32,
    /// Total number of agent spans created.
    pub agent_invocations: u32,
    /// Total number of tool spans created.
    pub tool_calls: u32,
    /// Number of escalation events.
    pub escalations: u32,
    /// Total tokens consumed across all agent spans.
    pub total_tokens: u64,
    /// Total duration across all gate spans (ms).
    pub total_gate_duration_ms: u64,
}

impl SpanSummary {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a gate result into the summary.
    pub fn record_gate(&mut self, passed: bool, duration_ms: u64) {
        self.gates += 1;
        if passed {
            self.gates_passed += 1;
        } else {
            self.gates_failed += 1;
        }
        self.total_gate_duration_ms += duration_ms;
    }

    /// Record an agent invocation into the summary.
    pub fn record_agent(&mut self, tokens: u64) {
        self.agent_invocations += 1;
        self.total_tokens += tokens;
    }

    /// Record a tool call into the summary.
    pub fn record_tool(&mut self) {
        self.tool_calls += 1;
    }

    /// Record an escalation event.
    pub fn record_escalation(&mut self) {
        self.escalations += 1;
    }

    /// Record the start of a new iteration.
    pub fn record_iteration(&mut self) {
        self.iterations += 1;
    }

    /// Gate pass rate as a fraction (0.0 to 1.0).
    pub fn gate_pass_rate(&self) -> f64 {
        if self.gates == 0 {
            return 0.0;
        }
        self.gates_passed as f64 / self.gates as f64
    }

    /// Average gate duration in milliseconds.
    pub fn avg_gate_duration_ms(&self) -> f64 {
        if self.gates == 0 {
            return 0.0;
        }
        self.total_gate_duration_ms as f64 / self.gates as f64
    }
}

impl std::fmt::Display for SpanSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "iterations={} gates={}/{} agents={} tools={} escalations={} tokens={}",
            self.iterations,
            self.gates_passed,
            self.gates,
            self.agent_invocations,
            self.tool_calls,
            self.escalations,
            self.total_tokens,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Once;

    static INIT: Once = Once::new();

    /// Initialize a test subscriber so spans are not disabled.
    fn init_test_subscriber() {
        INIT.call_once(|| {
            let _ = tracing_subscriber::fmt()
                .with_test_writer()
                .with_max_level(tracing::Level::TRACE)
                .try_init();
        });
    }

    #[test]
    fn test_process_issue_span_creates_valid_span() {
        init_test_subscriber();
        let span = process_issue_span("beefcake-655");
        assert!(!span.is_disabled());
        record_process_result(&span, true, 3, 45000);
    }

    #[test]
    fn test_iteration_span_creates_valid_span() {
        init_test_subscriber();
        let span = iteration_span("beefcake-655", 1, "Worker");
        assert!(!span.is_disabled());
        record_iteration_result(&span, true, 0, 2, 12000);
    }

    #[test]
    fn test_agent_span_creates_valid_span() {
        init_test_subscriber();
        let span = agent_span(AgentRole::Implementer, "strand-14b", "Worker");
        assert!(!span.is_disabled());
        record_agent_result(&span, true, 8500, 2048);
    }

    #[test]
    fn test_gate_span_creates_valid_span() {
        init_test_subscriber();
        let span = gate_span("clippy", "beefcake-655", 1);
        assert!(!span.is_disabled());
        record_gate_result(&span, "PASS", 0, 1, 3200);
    }

    #[test]
    fn test_escalation_span_creates_valid_span() {
        init_test_subscriber();
        let span = escalation_span("beefcake-655", "Worker", "Council", "error_repeat_2x", 3);
        assert!(!span.is_disabled());
    }

    #[test]
    fn test_tool_span_creates_valid_span() {
        init_test_subscriber();
        let span = tool_span("read_file", AgentRole::Implementer);
        assert!(!span.is_disabled());
        record_tool_result(&span, true, 50);
    }

    #[test]
    fn test_voting_span_creates_valid_span() {
        init_test_subscriber();
        let span = voting_span("majority", 3);
        assert!(!span.is_disabled());
        record_voting_result(&span, "consensus", 1200);
    }

    #[test]
    fn test_arbitration_span_creates_valid_span() {
        init_test_subscriber();
        let span = arbitration_span("beefcake-655", 3);
        assert!(!span.is_disabled());
        record_arbitration_result(&span, "accepted_majority", 5000);
    }

    #[test]
    fn test_span_summary_default() {
        let summary = SpanSummary::new();
        assert_eq!(summary.iterations, 0);
        assert_eq!(summary.gates, 0);
        assert_eq!(summary.gate_pass_rate(), 0.0);
        assert_eq!(summary.avg_gate_duration_ms(), 0.0);
    }

    #[test]
    fn test_span_summary_recording() {
        let mut summary = SpanSummary::new();
        summary.record_iteration();
        summary.record_gate(true, 3000);
        summary.record_gate(true, 2000);
        summary.record_gate(false, 1500);
        summary.record_agent(2048);
        summary.record_agent(1024);
        summary.record_tool();
        summary.record_tool();
        summary.record_tool();
        summary.record_escalation();

        assert_eq!(summary.iterations, 1);
        assert_eq!(summary.gates, 3);
        assert_eq!(summary.gates_passed, 2);
        assert_eq!(summary.gates_failed, 1);
        assert_eq!(summary.agent_invocations, 2);
        assert_eq!(summary.tool_calls, 3);
        assert_eq!(summary.escalations, 1);
        assert_eq!(summary.total_tokens, 3072);
        assert_eq!(summary.total_gate_duration_ms, 6500);
        assert!((summary.gate_pass_rate() - 0.6667).abs() < 0.01);
        assert!((summary.avg_gate_duration_ms() - 2166.67).abs() < 1.0);
    }

    #[test]
    fn test_span_summary_display() {
        let mut summary = SpanSummary::new();
        summary.record_iteration();
        summary.record_gate(true, 1000);
        summary.record_agent(512);
        summary.record_tool();
        let display = summary.to_string();
        assert!(display.contains("iterations=1"));
        assert!(display.contains("gates=1/1"));
        assert!(display.contains("agents=1"));
        assert!(display.contains("tools=1"));
        assert!(display.contains("tokens=512"));
    }

    #[test]
    fn test_span_summary_serialization() {
        let mut summary = SpanSummary::new();
        summary.record_iteration();
        summary.record_gate(true, 1000);
        summary.record_agent(512);

        let json = serde_json::to_string(&summary).unwrap();
        let restored: SpanSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.iterations, 1);
        assert_eq!(restored.gates, 1);
        assert_eq!(restored.gates_passed, 1);
        assert_eq!(restored.total_tokens, 512);
    }

    #[test]
    fn test_agent_role_display() {
        assert_eq!(AgentRole::Implementer.to_string(), "implementer");
        assert_eq!(AgentRole::Architect.to_string(), "architect");
        assert_eq!(AgentRole::Coder.to_string(), "coder");
        assert_eq!(AgentRole::Manager.to_string(), "manager");
        assert_eq!(AgentRole::Reviewer.to_string(), "reviewer");
        assert_eq!(AgentRole::Planner.to_string(), "planner");
    }

    #[test]
    fn test_agent_role_serialization() {
        let role = AgentRole::Manager;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"manager\"");
        let restored: AgentRole = serde_json::from_str(&json).unwrap();
        assert_eq!(restored, AgentRole::Manager);
    }

    #[test]
    fn test_span_constants_are_dotted() {
        // All span names use dot notation for OTel compatibility
        assert!(SPAN_PROCESS_ISSUE.contains('.'));
        assert!(SPAN_ITERATION.contains('.'));
        assert!(SPAN_AGENT.contains('.'));
        assert!(SPAN_GATE.contains('.'));
        assert!(SPAN_ESCALATION.contains('.'));
        assert!(SPAN_TOOL.contains('.'));
        assert!(SPAN_VOTING.contains('.'));
        assert!(SPAN_ARBITRATION.contains('.'));
    }

    #[test]
    fn test_field_constants_are_dotted() {
        // All field names use dot notation
        assert!(FIELD_ISSUE_ID.contains('.'));
        assert!(FIELD_ITERATION.contains('.'));
        assert!(FIELD_TIER.contains('.'));
        assert!(FIELD_GATE_NAME.contains('.'));
        assert!(FIELD_GATE_OUTCOME.contains('.'));
        assert!(FIELD_TOOL_NAME.contains('.'));
        assert!(FIELD_AGENT_ROLE.contains('.'));
        assert!(FIELD_ESCALATION_REASON.contains('.'));
        assert!(FIELD_VOTE_STRATEGY.contains('.'));
    }
}
