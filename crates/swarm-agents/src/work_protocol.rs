//! Structured inter-agent communication protocol for the swarm.
//!
//! Replaces the flat `String` return from Rig's agent-as-tool pattern with
//! typed contracts that enable:
//! - **WorkOrder**: what the worker MUST do (behavioral contract)
//! - **WorkResult**: what the worker DID (structured outcome)
//! - **WorkStatus**: tristate completion signal (Complete / Partial / Stuck / Failed)
//!
//! # Design Basis
//!
//! - SEMAP (APSEC 2025): behavioral contracts + structured messaging + lifecycle gates
//! - Agentic Lybic: FSM-based Controller/Manager/Workers/Evaluator with quality gating
//! - CodeCRDT: observation-driven coordination via shared state (git worktree = blackboard)
//! - MAST taxonomy: verification failures are the #1 cause of agent failures
//!
//! # Usage
//!
//! ```ignore
//! use swarm_agents::work_protocol::*;
//!
//! // Orchestrator creates a work order
//! let order = WorkOrder::new("order-1", "beads-abc", "Fix borrow checker error in parser.rs")
//!     .target_files(vec!["src/parser.rs".into()])
//!     .done_when(DoneCriteria::CompileClean)
//!     .max_turns(15);
//!
//! // After worker completes, build structured result from adapter report
//! let result = WorkResult::from_adapter_report("order-1", &report, WorkStatus::Complete, "Fixed it");
//!
//! // Attach verification
//! let result = result.with_verification(verification);
//! ```

use serde::{Deserialize, Serialize};

use crate::runtime_adapter::AdapterReport;

// ── WorkOrder ────────────────────────────────────────────────────────────────

/// Structured task contract sent to a worker.
///
/// Defines the behavioral contract: what the worker may do, what it must not
/// touch, and what "done" looks like. Analogous to `Subtask` but richer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkOrder {
    /// Unique identifier for this work order.
    pub id: String,
    /// Beads issue ID this work order belongs to.
    pub issue_id: String,
    /// Human-readable objective — what the worker should accomplish.
    pub objective: String,
    /// Files the worker MAY modify.
    pub target_files: Vec<String>,
    /// Files the worker MUST NOT modify (hard constraint).
    #[serde(default)]
    pub forbidden_files: Vec<String>,
    /// What "done" means — the verification contract.
    pub done_criteria: DoneCriteria,
    /// Maximum LLM turns before the worker must stop.
    pub max_turns: usize,
    /// Wall-clock timeout in seconds.
    pub timeout_secs: u64,
    /// Contextual information to help the worker.
    pub context: WorkContext,
    /// Which worker tier should execute this order.
    #[serde(default)]
    pub worker_tier: Option<WorkerTier>,
    /// Current iteration number (0-based). Enables retry-aware behavior.
    #[serde(default)]
    pub iteration: usize,
    /// Optional sprint contract agreed with the evaluator before implementation.
    ///
    /// When present, both the coder prompt and reviewer receive the contract
    /// as a shared "done criteria" reference, aligning expectations before
    /// any code is written.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sprint_contract: Option<SprintContract>,
}

impl WorkOrder {
    /// Create a new work order with minimal required fields.
    pub fn new(
        id: impl Into<String>,
        issue_id: impl Into<String>,
        objective: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            issue_id: issue_id.into(),
            objective: objective.into(),
            target_files: Vec::new(),
            forbidden_files: Vec::new(),
            done_criteria: DoneCriteria::CompileClean,
            max_turns: 15,
            timeout_secs: 900,
            context: WorkContext::default(),
            worker_tier: None,
            iteration: 0,
            sprint_contract: None,
        }
    }

    /// Set target files the worker may modify.
    pub fn target_files(mut self, files: Vec<String>) -> Self {
        self.target_files = files;
        self
    }

    /// Set forbidden files the worker must not touch.
    pub fn forbidden_files(mut self, files: Vec<String>) -> Self {
        self.forbidden_files = files;
        self
    }

    /// Set the done criteria (verification contract).
    pub fn done_when(mut self, criteria: DoneCriteria) -> Self {
        self.done_criteria = criteria;
        self
    }

    /// Set maximum LLM turns.
    pub fn max_turns(mut self, turns: usize) -> Self {
        self.max_turns = turns;
        self
    }

    /// Set wall-clock timeout.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = secs;
        self
    }

    /// Set contextual information.
    pub fn context(mut self, ctx: WorkContext) -> Self {
        self.context = ctx;
        self
    }

    /// Set the worker tier.
    pub fn worker_tier(mut self, tier: WorkerTier) -> Self {
        self.worker_tier = Some(tier);
        self
    }

    /// Set the iteration number.
    pub fn iteration(mut self, iter: usize) -> Self {
        self.iteration = iter;
        self
    }

    /// Attach a sprint contract to this work order.
    pub fn sprint_contract(mut self, contract: SprintContract) -> Self {
        self.sprint_contract = Some(contract);
        self
    }
}

/// A sprint contract negotiated before the first implementation iteration.
///
/// Before any code is written, the manager proposes what will be built and how
/// it will be verified. The evaluator (reviewer) validates the contract is
/// testable and specific. The agreed contract is then embedded in all
/// `WorkOrder` instances for the issue, aligning both the coder and reviewer on
/// shared expectations — a key pattern from Anthropic's harness design article.
///
/// The contract is stored at `.swarm/sprint-contracts.jsonl` in the worktree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SprintContract {
    /// Issue ID this contract applies to.
    pub issue_id: String,
    /// What will be built (scope statement).
    pub scope: String,
    /// Observable, verifiable conditions that must be true for the sprint to be
    /// considered done. Each entry should be independently verifiable.
    pub done_criteria: Vec<String>,
    /// Work explicitly excluded from this sprint to avoid scope creep.
    pub out_of_scope: Vec<String>,
    /// Specific `cargo test` expectations (e.g., "test_parser_handles_eof passes").
    pub test_expectations: Vec<String>,
    /// Whether the evaluator agent has agreed to the contract terms.
    pub agreed: bool,
    /// How many negotiation rounds were needed to reach agreement.
    pub negotiation_rounds: u8,
}

impl SprintContract {
    pub fn new(issue_id: impl Into<String>, scope: impl Into<String>) -> Self {
        Self {
            issue_id: issue_id.into(),
            scope: scope.into(),
            done_criteria: Vec::new(),
            out_of_scope: Vec::new(),
            test_expectations: Vec::new(),
            agreed: false,
            negotiation_rounds: 0,
        }
    }

    /// Add a done criterion.
    pub fn with_criterion(mut self, criterion: impl Into<String>) -> Self {
        self.done_criteria.push(criterion.into());
        self
    }

    /// Mark out-of-scope to prevent scope creep.
    pub fn exclude(mut self, item: impl Into<String>) -> Self {
        self.out_of_scope.push(item.into());
        self
    }

    /// Mark the contract as agreed.
    pub fn agree(mut self) -> Self {
        self.agreed = true;
        self
    }

    /// Render the contract as a prompt section injected into both
    /// coder and reviewer preambles.
    pub fn to_prompt_section(&self) -> String {
        let mut out = String::from("## Sprint Contract\n");
        out.push_str(&format!("**Scope:** {}\n\n", self.scope));

        if !self.done_criteria.is_empty() {
            out.push_str("**Done when:**\n");
            for c in &self.done_criteria {
                out.push_str(&format!("- {c}\n"));
            }
            out.push('\n');
        }

        if !self.out_of_scope.is_empty() {
            out.push_str("**Out of scope:**\n");
            for item in &self.out_of_scope {
                out.push_str(&format!("- {item}\n"));
            }
            out.push('\n');
        }

        if !self.test_expectations.is_empty() {
            out.push_str("**Test expectations:**\n");
            for t in &self.test_expectations {
                out.push_str(&format!("- `{t}`\n"));
            }
            out.push('\n');
        }

        out
    }
}

/// What "done" means — the verification contract.
///
/// The orchestrator checks this AFTER the worker completes but BEFORE
/// accepting the result. This is the behavioral contract from SEMAP.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DoneCriteria {
    /// All target_files modified, `cargo check` passes.
    CompileClean,
    /// `cargo test` passes for specified packages.
    TestsPass { packages: Vec<String> },
    /// Specific pattern no longer appears in target files.
    PatternRemoved { pattern: String },
    /// Custom verification command succeeds (exit code 0).
    Custom { command: String },
}

/// A structured record of an approach that was tried and failed within the
/// current session.
///
/// Populated by the orchestrator after each failed iteration and injected
/// into the next worker prompt as a "What We've Tried" section. This
/// prevents workers from re-attempting approaches that are already known
/// to fail — a key pattern from Anthropic's harness design article.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedApproach {
    /// Iteration number (1-indexed) when this approach was attempted.
    pub iteration: usize,
    /// Short description of what was attempted (1–2 sentences).
    pub summary: String,
    /// The concrete error or reviewer feedback explaining why it failed.
    pub error_output: String,
    /// Blake3 digest of the diff content; used to detect if the agent
    /// is about to repeat an identical approach without noticing.
    pub approach_digest: String,
}

impl FailedApproach {
    pub fn new(
        iteration: usize,
        summary: impl Into<String>,
        error_output: impl Into<String>,
    ) -> Self {
        let summary = summary.into();
        let error = error_output.into();
        let digest = blake3::hash(format!("{summary}{error}").as_bytes())
            .to_hex()
            .to_string();
        Self {
            iteration,
            summary,
            error_output: error,
            approach_digest: digest,
        }
    }
}

/// Contextual information to help the worker understand the task.
///
/// Derived from the orchestrator's accumulated state: previous errors,
/// file contents, constraints from the escalation engine.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WorkContext {
    /// Compilation/test error messages from previous iterations.
    #[serde(default)]
    pub error_history: Vec<String>,
    /// Relevant file snippets: (path, content_preview).
    #[serde(default)]
    pub file_snippets: Vec<FileSnippet>,
    /// Constraints from the escalation engine or manager.
    #[serde(default)]
    pub constraints: Vec<String>,
    /// Summary of previous attempts (what was tried and why it failed).
    #[serde(default)]
    pub previous_attempts: Vec<String>,
    /// Structured records of failed approaches within this session.
    ///
    /// These are injected into the worker prompt as a "What We've Tried"
    /// section. The digest field lets the orchestrator detect identical
    /// re-attempts before they are dispatched.
    #[serde(default)]
    pub failed_approaches: Vec<FailedApproach>,
    /// Reviewer/validator feedback from previous iterations (TextGrad).
    #[serde(default)]
    pub reviewer_feedback: Vec<String>,
}

/// A file snippet with path and content preview for the worker's context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSnippet {
    pub path: String,
    pub content: String,
    /// Key symbols (functions, structs, traits) in this file.
    #[serde(default)]
    pub key_symbols: Vec<String>,
}

/// Which worker tier should execute a work order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerTier {
    /// Qwen3.5-27B-Distilled — fast, VRAM-resident, 192K context.
    Fast,
    /// Qwen3.5-122B-A10B MoE on vasp-01 — code specialist.
    Coder,
    /// Qwen3.5-122B-A10B MoE on vasp-02 — deep reasoning.
    Reasoning,
}

impl std::fmt::Display for WorkerTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkerTier::Fast => write!(f, "fast"),
            WorkerTier::Coder => write!(f, "coder"),
            WorkerTier::Reasoning => write!(f, "reasoning"),
        }
    }
}

// ── WorkResult ───────────────────────────────────────────────────────────────

/// Structured result returned by a worker after completing (or failing) a WorkOrder.
///
/// This is the core output type that replaces Rig's flat `String` return.
/// It provides the manager/orchestrator with structured information about
/// what happened, enabling programmatic decision-making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkResult {
    /// WorkOrder ID this result corresponds to.
    pub order_id: String,
    /// Completion status — the tristate signal.
    pub status: WorkStatus,
    /// Files the worker modified (from RuntimeAdapter or git diff).
    pub files_modified: Vec<String>,
    /// Files the worker read (from RuntimeAdapter).
    pub files_read: Vec<String>,
    /// File manifest with blake3 hashes for programmatic verification.
    #[serde(default)]
    pub file_manifest: Vec<FileManifestEntry>,
    /// Total tool calls made by the worker.
    pub tool_calls: usize,
    /// LLM turns used.
    pub turns_used: usize,
    /// Wall-clock time in milliseconds.
    pub wall_time_ms: u64,
    /// Git diff summary (from `git diff --stat`).
    pub git_diff_summary: String,
    /// Verification result (if verifier ran after worker completed).
    pub verification: Option<VerificationResult>,
    /// Worker's free-text explanation of what it did.
    pub worker_message: String,
    /// Worker self-assessment confidence (0.0–1.0).
    /// Below 0.5 → auto-escalate to manager.
    /// Below 0.3 → flag for human review.
    pub confidence: f32,
    /// Escalation request if the worker needs help.
    pub escalation: Option<EscalationRequest>,
}

impl WorkResult {
    /// Build a WorkResult from a RuntimeAdapter report.
    ///
    /// This is the primary constructor — bridges the existing telemetry
    /// infrastructure (`AdapterReport`) to the new structured protocol.
    pub fn from_adapter_report(
        order_id: impl Into<String>,
        report: &AdapterReport,
        status: WorkStatus,
        worker_message: impl Into<String>,
    ) -> Self {
        Self {
            order_id: order_id.into(),
            status,
            files_modified: report.files_modified.clone(),
            files_read: report.files_read.clone(),
            file_manifest: Vec::new(),
            tool_calls: report.total_tool_calls,
            turns_used: report.turn_count,
            wall_time_ms: report.wall_time_ms,
            git_diff_summary: String::new(),
            verification: None,
            worker_message: worker_message.into(),
            confidence: 0.5, // default — refined by heuristics or verification
            escalation: None,
        }
    }

    /// Infer WorkStatus from an AdapterReport heuristically.
    ///
    /// Uses the adapter's termination signals to determine the most likely
    /// outcome without relying on the worker's self-report (which may be
    /// unreliable for local LLMs).
    pub fn infer_status(report: &AdapterReport) -> WorkStatus {
        if report.terminated_early {
            let reason = report
                .termination_reason
                .clone()
                .unwrap_or_else(|| "budget exhausted".into());
            if report.has_written {
                WorkStatus::Partial { reason }
            } else {
                WorkStatus::Stuck { reason }
            }
        } else if report.has_written {
            WorkStatus::Complete
        } else {
            // Agent finished normally but never wrote — likely confused or blocked.
            WorkStatus::Stuck {
                reason: "agent completed without writing any files".into(),
            }
        }
    }

    /// Compute a heuristic confidence score from adapter telemetry.
    ///
    /// This provides a baseline confidence before verification runs.
    /// After verification, use `with_verification()` which refines confidence
    /// based on gate pass rates.
    pub fn heuristic_confidence(report: &AdapterReport) -> f32 {
        let mut score: f32 = 0.0;

        // Did it write files? (+0.3)
        if report.has_written {
            score += 0.3;
        }

        // Didn't get terminated early? (+0.2)
        if !report.terminated_early {
            score += 0.2;
        }

        // Tool call efficiency: fewer calls per write = more focused (+0.1–0.2)
        if report.has_written && report.total_tool_calls > 0 {
            let writes = report.successful_writes.max(1) as f32;
            let ratio = writes / report.total_tool_calls as f32;
            // ratio of 0.1+ is efficient (1 write per 10 calls)
            score += (ratio * 2.0).min(0.2);
        }

        // No failed edits? (+0.1)
        if report.last_failed_edits.is_empty() {
            score += 0.1;
        }

        score.clamp(0.0, 1.0)
    }

    /// Attach verification results and refine confidence.
    pub fn with_verification(mut self, verification: VerificationResult) -> Self {
        // Refine confidence based on verification gates.
        self.confidence = verification.confidence_contribution();
        self.verification = Some(verification);
        self
    }

    /// Attach git diff summary.
    pub fn with_diff(mut self, diff: String) -> Self {
        self.git_diff_summary = diff;
        self
    }

    /// Attach file manifest with blake3 hashes.
    pub fn with_manifest(mut self, manifest: Vec<FileManifestEntry>) -> Self {
        self.file_manifest = manifest;
        self
    }

    /// Attach an escalation request.
    pub fn with_escalation(mut self, escalation: EscalationRequest) -> Self {
        self.escalation = Some(escalation);
        self
    }

    /// Override the confidence score.
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }

    /// Whether this result should trigger auto-escalation to the manager.
    pub fn needs_escalation(&self) -> bool {
        self.confidence < 0.5
            || self.escalation.is_some()
            || matches!(
                self.status,
                WorkStatus::Stuck { .. } | WorkStatus::OutOfScope { .. }
            )
    }

    /// Whether this result should be flagged for human review.
    pub fn needs_human_review(&self) -> bool {
        self.confidence < 0.3 || matches!(self.status, WorkStatus::Failed { .. })
    }

    /// Whether the worker made any meaningful progress.
    pub fn made_progress(&self) -> bool {
        !self.files_modified.is_empty()
            && matches!(
                self.status,
                WorkStatus::Complete | WorkStatus::Partial { .. }
            )
    }

    /// Summary suitable for logging.
    pub fn summary(&self) -> String {
        format!(
            "WorkResult[{}] status={} files_mod={} tools={} turns={} conf={:.2} {}",
            self.order_id,
            self.status,
            self.files_modified.len(),
            self.tool_calls,
            self.turns_used,
            self.confidence,
            if self.needs_escalation() {
                "[ESCALATE]"
            } else {
                ""
            },
        )
    }
}

// ── WorkStatus ───────────────────────────────────────────────────────────────

/// Tristate completion signal from a worker.
///
/// Research (MAST taxonomy) shows that binary success/fail is insufficient —
/// workers often make partial progress that should be preserved rather than
/// rolled back. The tristate enables the orchestrator to make nuanced decisions:
///
/// - **Complete**: worker believes task is done → run verification
/// - **Partial**: worker made progress but couldn't finish → keep changes, retry
/// - **Stuck**: worker is blocked, needs help → escalate to manager
/// - **OutOfScope**: task requires changes outside allowed files → reassign
/// - **Failed**: unrecoverable error → roll back, escalate
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkStatus {
    /// Worker believes the task is done. Run verification to confirm.
    Complete,
    /// Worker made progress but couldn't finish. Keep changes.
    /// Retryable — the reason explains what's left.
    Partial { reason: String },
    /// Worker is blocked and needs help. Escalate to manager.
    Stuck { reason: String },
    /// Task requires changes outside the worker's allowed files.
    OutOfScope { reason: String },
    /// Unrecoverable error. Roll back changes.
    Failed { error: String },
}

impl WorkStatus {
    /// Whether changes from this status should be kept (not rolled back).
    pub fn keep_changes(&self) -> bool {
        matches!(self, WorkStatus::Complete | WorkStatus::Partial { .. })
    }

    /// Whether this status is retryable.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            WorkStatus::Partial { .. } | WorkStatus::Stuck { .. } | WorkStatus::OutOfScope { .. }
        )
    }

    /// Whether this status represents a terminal failure.
    pub fn is_terminal(&self) -> bool {
        matches!(self, WorkStatus::Failed { .. })
    }

    /// Short label for logging and events.
    pub fn label(&self) -> &'static str {
        match self {
            WorkStatus::Complete => "complete",
            WorkStatus::Partial { .. } => "partial",
            WorkStatus::Stuck { .. } => "stuck",
            WorkStatus::OutOfScope { .. } => "out_of_scope",
            WorkStatus::Failed { .. } => "failed",
        }
    }
}

impl std::fmt::Display for WorkStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkStatus::Complete => write!(f, "complete"),
            WorkStatus::Partial { reason } => write!(f, "partial: {reason}"),
            WorkStatus::Stuck { reason } => write!(f, "stuck: {reason}"),
            WorkStatus::OutOfScope { reason } => write!(f, "out_of_scope: {reason}"),
            WorkStatus::Failed { error } => write!(f, "failed: {error}"),
        }
    }
}

// ── VerificationResult ───────────────────────────────────────────────────────

/// Simplified verification result for inter-agent communication.
///
/// Mirrors the coordination crate's `VerifierReport` but focused on what
/// the manager/orchestrator needs for decision-making, not the full pipeline
/// details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Whether ALL gates passed (clean build).
    pub all_green: bool,
    /// cargo fmt passed.
    pub fmt_pass: bool,
    /// cargo clippy passed.
    pub clippy_pass: bool,
    /// cargo check passed.
    pub check_pass: bool,
    /// cargo test passed (None if tests weren't run).
    pub test_pass: Option<bool>,
    /// Number of remaining errors.
    pub error_count: usize,
    /// Gates passed vs total.
    pub gates_passed: usize,
    pub gates_total: usize,
    /// Human-readable summary.
    pub summary: String,
}

impl VerificationResult {
    /// Compute a confidence contribution from verification gates.
    ///
    /// Maps gate pass rates to a 0.0–1.0 confidence score:
    /// - fmt only: 0.3
    /// - fmt + clippy: 0.5
    /// - fmt + clippy + check: 0.75
    /// - all gates: 1.0
    pub fn confidence_contribution(&self) -> f32 {
        let mut score = 0.0f32;
        if self.fmt_pass {
            score += 0.2;
        }
        if self.clippy_pass {
            score += 0.2;
        }
        if self.check_pass {
            score += 0.3;
        }
        if let Some(true) = self.test_pass {
            score += 0.3;
        }
        score.clamp(0.0, 1.0)
    }
}

// ── FileManifestEntry ────────────────────────────────────────────────────────

/// File manifest entry with blake3 hash for programmatic verification.
///
/// Enables the manager/orchestrator to verify that the worker's claimed
/// changes actually happened, without reading file contents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileManifestEntry {
    pub path: String,
    /// blake3 hash of the file content (short form, 2 hex chars).
    pub hash: String,
    /// What the worker did to this file.
    pub action: FileAction,
}

/// What happened to a file in the worktree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FileAction {
    Created,
    Modified,
    Deleted,
}

impl std::fmt::Display for FileAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FileAction::Created => write!(f, "created"),
            FileAction::Modified => write!(f, "modified"),
            FileAction::Deleted => write!(f, "deleted"),
        }
    }
}

// ── EscalationRequest ────────────────────────────────────────────────────────

/// Worker requesting help from the manager or orchestrator.
///
/// Provides structured context about WHY the worker is stuck, enabling
/// the manager to provide targeted guidance rather than blind re-delegation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationRequest {
    /// Why the worker is stuck (e.g., "borrow checker error I can't resolve").
    pub reason: String,
    /// What the worker suggests the manager do.
    pub suggested_action: String,
    /// Files that are blocking progress (may be outside the worker's scope).
    #[serde(default)]
    pub blocking_files: Vec<String>,
}

/// Convert a `coordination::verifier::VerifierReport` to a `VerificationResult`.
///
/// This bridges the coordination crate's detailed report to the simplified
/// protocol type used for inter-agent communication.
pub fn verification_from_report(
    report: &coordination::verifier::VerifierReport,
) -> VerificationResult {
    use coordination::verifier::GateOutcome;

    let fmt_pass = report
        .gates
        .iter()
        .any(|g| g.gate == "fmt" && matches!(g.outcome, GateOutcome::Passed));
    let clippy_pass = report
        .gates
        .iter()
        .any(|g| g.gate == "clippy" && matches!(g.outcome, GateOutcome::Passed));
    let check_pass = report
        .gates
        .iter()
        .any(|g| g.gate == "check" && matches!(g.outcome, GateOutcome::Passed));
    let test_pass = report
        .gates
        .iter()
        .find(|g| g.gate == "test")
        .map(|g| matches!(g.outcome, GateOutcome::Passed));

    VerificationResult {
        all_green: report.all_green,
        fmt_pass,
        clippy_pass,
        check_pass,
        test_pass,
        error_count: report.failure_signals.len(),
        gates_passed: report.gates_passed,
        gates_total: report.gates_total,
        summary: format!(
            "{}/{} gates passed{}",
            report.gates_passed,
            report.gates_total,
            if report.all_green { " ✓" } else { "" }
        ),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_status_properties() {
        assert!(WorkStatus::Complete.keep_changes());
        assert!(WorkStatus::Partial { reason: "x".into() }.keep_changes());
        assert!(!WorkStatus::Stuck { reason: "x".into() }.keep_changes());
        assert!(!WorkStatus::Failed { error: "x".into() }.keep_changes());

        assert!(!WorkStatus::Complete.is_retryable());
        assert!(WorkStatus::Partial { reason: "x".into() }.is_retryable());
        assert!(WorkStatus::Stuck { reason: "x".into() }.is_retryable());
        assert!(!WorkStatus::Failed { error: "x".into() }.is_retryable());

        assert!(!WorkStatus::Complete.is_terminal());
        assert!(WorkStatus::Failed { error: "x".into() }.is_terminal());
    }

    #[test]
    fn work_status_labels() {
        assert_eq!(WorkStatus::Complete.label(), "complete");
        assert_eq!(
            WorkStatus::Partial { reason: "x".into() }.label(),
            "partial"
        );
        assert_eq!(
            WorkStatus::OutOfScope { reason: "x".into() }.label(),
            "out_of_scope"
        );
    }

    #[test]
    fn infer_status_complete() {
        let report = AdapterReport {
            agent_name: "test".into(),
            tool_events: vec![],
            turn_count: 5,
            total_tool_calls: 10,
            total_tool_time_ms: 1000,
            wall_time_ms: 5000,
            terminated_early: false,
            termination_reason: None,
            has_written: true,
            files_read: vec![],
            files_modified: vec![],
            successful_writes: 2,
            last_failed_edits: vec![],
        };
        assert!(matches!(
            WorkResult::infer_status(&report),
            WorkStatus::Complete
        ));
    }

    #[test]
    fn infer_status_partial() {
        let report = AdapterReport {
            agent_name: "test".into(),
            tool_events: vec![],
            turn_count: 15,
            total_tool_calls: 30,
            total_tool_time_ms: 5000,
            wall_time_ms: 15000,
            terminated_early: true,
            termination_reason: Some("deadline exceeded".into()),
            has_written: true,
            files_read: vec![],
            files_modified: vec![],
            successful_writes: 1,
            last_failed_edits: vec![],
        };
        let status = WorkResult::infer_status(&report);
        assert!(matches!(status, WorkStatus::Partial { .. }));
        if let WorkStatus::Partial { reason } = status {
            assert!(reason.contains("deadline"));
        }
    }

    #[test]
    fn infer_status_stuck() {
        let report = AdapterReport {
            agent_name: "test".into(),
            tool_events: vec![],
            turn_count: 40,
            total_tool_calls: 80,
            total_tool_time_ms: 10000,
            wall_time_ms: 30000,
            terminated_early: true,
            termination_reason: Some("write deadline".into()),
            has_written: false,
            files_read: vec!["src/lib.rs".into()],
            files_modified: vec![],
            successful_writes: 0,
            last_failed_edits: vec![],
        };
        let status = WorkResult::infer_status(&report);
        assert!(matches!(status, WorkStatus::Stuck { .. }));
    }

    #[test]
    fn verification_confidence_mapping() {
        let v = VerificationResult {
            all_green: true,
            fmt_pass: true,
            clippy_pass: true,
            check_pass: true,
            test_pass: Some(true),
            error_count: 0,
            gates_passed: 4,
            gates_total: 4,
            summary: "4/4 ✓".into(),
        };
        let conf = v.confidence_contribution();
        assert!(
            (conf - 1.0).abs() < f32::EPSILON,
            "expected 1.0, got {conf}"
        );

        let v2 = VerificationResult {
            all_green: false,
            fmt_pass: true,
            clippy_pass: true,
            check_pass: false,
            test_pass: None,
            error_count: 3,
            gates_passed: 2,
            gates_total: 3,
            summary: "2/3".into(),
        };
        let conf2 = v2.confidence_contribution();
        assert!(
            (conf2 - 0.4).abs() < f32::EPSILON,
            "expected 0.4, got {conf2}"
        );
    }

    #[test]
    fn work_result_escalation_signals() {
        let result = WorkResult {
            order_id: "o1".into(),
            status: WorkStatus::Stuck {
                reason: "can't fix".into(),
            },
            files_modified: vec![],
            files_read: vec![],
            file_manifest: vec![],
            tool_calls: 0,
            turns_used: 0,
            wall_time_ms: 0,
            git_diff_summary: String::new(),
            verification: None,
            worker_message: String::new(),
            confidence: 0.2,
            escalation: Some(EscalationRequest {
                reason: "borrow checker".into(),
                suggested_action: "restructure lifetimes".into(),
                blocking_files: vec!["src/main.rs".into()],
            }),
        };

        assert!(result.needs_escalation());
        assert!(result.needs_human_review()); // confidence < 0.3
        assert!(!result.made_progress());
    }

    #[test]
    fn work_result_with_verification_refines_confidence() {
        let report = AdapterReport {
            agent_name: "test".into(),
            tool_events: vec![],
            turn_count: 5,
            total_tool_calls: 10,
            total_tool_time_ms: 1000,
            wall_time_ms: 5000,
            terminated_early: false,
            termination_reason: None,
            has_written: true,
            files_read: vec![],
            files_modified: vec!["src/a.rs".into()],
            successful_writes: 2,
            last_failed_edits: vec![],
        };

        let result = WorkResult::from_adapter_report("o1", &report, WorkStatus::Complete, "done")
            .with_verification(VerificationResult {
                all_green: true,
                fmt_pass: true,
                clippy_pass: true,
                check_pass: true,
                test_pass: Some(true),
                error_count: 0,
                gates_passed: 4,
                gates_total: 4,
                summary: "4/4 ✓".into(),
            });

        // Confidence should be 1.0 (all gates passed)
        assert!(
            (result.confidence - 1.0).abs() < f32::EPSILON,
            "expected 1.0, got {}",
            result.confidence
        );
        assert!(!result.needs_escalation());
    }

    #[test]
    fn work_result_summary() {
        let result = WorkResult {
            order_id: "o1".into(),
            status: WorkStatus::Complete,
            files_modified: vec!["a.rs".into(), "b.rs".into()],
            files_read: vec![],
            file_manifest: vec![],
            tool_calls: 8,
            turns_used: 4,
            wall_time_ms: 5000,
            git_diff_summary: String::new(),
            verification: None,
            worker_message: String::new(),
            confidence: 0.8,
            escalation: None,
        };

        let s = result.summary();
        assert!(s.contains("o1"));
        assert!(s.contains("complete"));
        assert!(s.contains("files_mod=2"));
        assert!(s.contains("tools=8"));
        assert!(s.contains("conf=0.80"));
    }
}
