//! Debate orchestrator — drives the coder→reviewer iterative loop.
//!
//! Ties together the state machine, consensus protocol, and guardrails
//! to run a complete debate cycle end-to-end.

use chrono::Utc;
use serde::{Deserialize, Serialize};

use super::consensus::{ConsensusCheck, ConsensusProtocol, Verdict};
use super::guardrails::{DeadlockOutcome, GuardrailConfig, GuardrailEngine};
use super::state::{DebatePhase, DebateSession, RoundRecord};

/// Configuration for the debate orchestrator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateConfig {
    /// Maximum rounds for the debate.
    pub max_rounds: u32,
    /// Guardrail configuration.
    pub guardrails: GuardrailConfig,
    /// Consensus protocol settings.
    pub consensus: ConsensusProtocol,
}

impl Default for DebateConfig {
    fn default() -> Self {
        Self {
            max_rounds: 5,
            guardrails: GuardrailConfig::default(),
            consensus: ConsensusProtocol::default(),
        }
    }
}

/// Result of a single coder turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoderOutput {
    /// The code or patch produced by the coder.
    pub code: String,
    /// Files modified.
    pub files_changed: Vec<String>,
    /// Coder's explanation of changes.
    pub explanation: String,
}

/// Result of a single reviewer turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewerOutput {
    /// Structured consensus check.
    pub check: ConsensusCheck,
    /// Human-readable summary of the review.
    pub summary: String,
}

/// Error from the debate orchestrator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DebateError {
    /// State transition failed.
    TransitionFailed(String),
    /// Debate was already completed.
    AlreadyComplete,
    /// Invalid operation for current phase.
    InvalidPhase { expected: String, actual: String },
}

impl std::fmt::Display for DebateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TransitionFailed(msg) => write!(f, "transition failed: {}", msg),
            Self::AlreadyComplete => write!(f, "debate already complete"),
            Self::InvalidPhase { expected, actual } => {
                write!(f, "expected phase {}, got {}", expected, actual)
            }
        }
    }
}

impl std::error::Error for DebateError {}

/// Outcome of a completed debate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateOutcome {
    /// Final phase the debate ended in.
    pub terminal_phase: DebatePhase,
    /// Total rounds executed.
    pub rounds_completed: u32,
    /// Final consensus check (if any).
    pub final_check: Option<ConsensusCheck>,
    /// Whether the debate reached consensus.
    pub consensus_reached: bool,
    /// Guardrail outcome that terminated the debate (if not consensus).
    pub termination_reason: Option<DeadlockOutcome>,
    /// The session snapshot at completion.
    pub session: DebateSession,
}

impl DebateOutcome {
    /// Whether the debate succeeded (resolved).
    pub fn is_success(&self) -> bool {
        self.terminal_phase == DebatePhase::Resolved
    }

    /// Whether the debate needs escalation.
    pub fn needs_escalation(&self) -> bool {
        matches!(
            self.terminal_phase,
            DebatePhase::Deadlocked | DebatePhase::Escalated
        )
    }

    /// Compact summary line.
    pub fn summary_line(&self) -> String {
        let status = if self.is_success() {
            "RESOLVED"
        } else if self.needs_escalation() {
            "ESCALATED"
        } else {
            "ABORTED"
        };
        format!(
            "[{}] {} rounds | issue={}",
            status, self.rounds_completed, self.session.issue_id
        )
    }
}

/// The debate orchestrator — drives the iterative coder→reviewer loop.
///
/// Usage:
/// 1. Create with `new()` or `with_config()`
/// 2. Call `start()` to begin the debate
/// 3. Call `submit_code()` with coder output
/// 4. Call `submit_review()` with reviewer output
/// 5. Check `next_action()` to determine what happens next
/// 6. Repeat 3-5 until `is_complete()` returns true
/// 7. Call `outcome()` to get the final result
pub struct DebateOrchestrator {
    session: DebateSession,
    config: DebateConfig,
    engine: GuardrailEngine,
    checks: Vec<ConsensusCheck>,
    start_time_ms: u64,
    elapsed_ms: u64,
}

/// What the orchestrator expects next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NextAction {
    /// Waiting for coder to produce code.
    AwaitCoder,
    /// Waiting for reviewer to evaluate code.
    AwaitReviewer,
    /// Debate is complete — call `outcome()`.
    Complete,
}

impl std::fmt::Display for NextAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AwaitCoder => write!(f, "await_coder"),
            Self::AwaitReviewer => write!(f, "await_reviewer"),
            Self::Complete => write!(f, "complete"),
        }
    }
}

impl DebateOrchestrator {
    /// Create a new orchestrator with default config.
    pub fn new(debate_id: &str, issue_id: &str, subject: &str) -> Self {
        Self::with_config(debate_id, issue_id, subject, DebateConfig::default())
    }

    /// Create a new orchestrator with custom config.
    pub fn with_config(
        debate_id: &str,
        issue_id: &str,
        subject: &str,
        config: DebateConfig,
    ) -> Self {
        let session = DebateSession::new(debate_id, issue_id, subject, config.max_rounds);
        let engine = GuardrailEngine::new(config.guardrails.clone());
        Self {
            session,
            config,
            engine,
            checks: Vec::new(),
            start_time_ms: 0,
            elapsed_ms: 0,
        }
    }

    /// Start the debate — transitions from Idle to CoderTurn.
    pub fn start(&mut self) -> Result<(), DebateError> {
        self.session
            .start()
            .map_err(|e| DebateError::TransitionFailed(e.to_string()))?;
        self.start_time_ms = Utc::now().timestamp_millis() as u64;
        Ok(())
    }

    /// What action is expected next.
    pub fn next_action(&self) -> NextAction {
        match self.session.phase {
            DebatePhase::CoderTurn => NextAction::AwaitCoder,
            DebatePhase::ReviewerTurn => NextAction::AwaitReviewer,
            _ if self.session.is_complete() => NextAction::Complete,
            DebatePhase::Idle => NextAction::AwaitCoder, // not started yet
            _ => NextAction::Complete,
        }
    }

    /// Submit coder output and transition to reviewer turn.
    pub fn submit_code(&mut self, output: CoderOutput) -> Result<(), DebateError> {
        if self.session.phase != DebatePhase::CoderTurn {
            return Err(DebateError::InvalidPhase {
                expected: "coder_turn".to_string(),
                actual: self.session.phase.to_string(),
            });
        }

        self.session
            .transition(DebatePhase::ReviewerTurn, &output.explanation)
            .map_err(|e| DebateError::TransitionFailed(e.to_string()))?;

        Ok(())
    }

    /// Submit reviewer output and determine next step.
    ///
    /// This is the key decision point: check consensus, run guardrails,
    /// and either continue, resolve, deadlock, or escalate.
    pub fn submit_review(&mut self, output: ReviewerOutput) -> Result<NextAction, DebateError> {
        if self.session.phase != DebatePhase::ReviewerTurn {
            return Err(DebateError::InvalidPhase {
                expected: "reviewer_turn".to_string(),
                actual: self.session.phase.to_string(),
            });
        }

        self.checks.push(output.check.clone());
        self.update_elapsed();

        // Record the round
        let round = RoundRecord {
            round: self.session.current_round,
            coder_output: String::new(), // stored externally
            reviewer_verdict: output.summary.clone(),
            approved: output.check.verdict == Verdict::Approve,
            issues: output.check.blocking_issues.clone(),
            duration_ms: 0,
            started_at: Utc::now(),
        };
        self.session.record_round(round);

        // 1. Check for consensus
        if output.check.is_consensus()
            && output.check.confidence >= self.config.consensus.min_confidence
        {
            self.session
                .transition(DebatePhase::Resolved, "consensus reached")
                .map_err(|e| DebateError::TransitionFailed(e.to_string()))?;
            return Ok(NextAction::Complete);
        }

        // 2. Check guardrails
        let guardrail_outcome = self
            .engine
            .evaluate(&self.session, &self.checks, self.elapsed_ms);

        if guardrail_outcome.should_stop() {
            let (phase, reason) = match &guardrail_outcome {
                DeadlockOutcome::MaxRoundsExceeded { .. }
                | DeadlockOutcome::StallDetected { .. } => {
                    (DebatePhase::Deadlocked, guardrail_outcome.to_string())
                }
                DeadlockOutcome::EscalationRequired { .. } => {
                    (DebatePhase::Escalated, guardrail_outcome.to_string())
                }
                DeadlockOutcome::TimeoutExceeded { .. } => {
                    (DebatePhase::Aborted, guardrail_outcome.to_string())
                }
                DeadlockOutcome::Continue => unreachable!(),
            };

            self.session
                .transition(phase, &reason)
                .map_err(|e| DebateError::TransitionFailed(e.to_string()))?;
            return Ok(NextAction::Complete);
        }

        // 3. Continue — transition back to coder turn
        self.session
            .transition(DebatePhase::CoderTurn, "reviewer requested changes")
            .map_err(|e| DebateError::TransitionFailed(e.to_string()))?;

        Ok(NextAction::AwaitCoder)
    }

    /// Abort the debate.
    pub fn abort(&mut self, reason: &str) -> Result<(), DebateError> {
        if self.session.is_complete() {
            return Err(DebateError::AlreadyComplete);
        }
        self.session
            .transition(DebatePhase::Aborted, reason)
            .map_err(|e| DebateError::TransitionFailed(e.to_string()))
    }

    /// Whether the debate has completed.
    pub fn is_complete(&self) -> bool {
        self.session.is_complete()
    }

    /// Get the debate outcome (only valid after completion).
    pub fn outcome(&self) -> Option<DebateOutcome> {
        if !self.session.is_complete() {
            return None;
        }

        let final_check = self.checks.last().cloned();
        let consensus_reached = self.session.phase == DebatePhase::Resolved;

        let termination_reason = if !consensus_reached {
            Some(
                self.engine
                    .evaluate(&self.session, &self.checks, self.elapsed_ms),
            )
        } else {
            None
        };

        Some(DebateOutcome {
            terminal_phase: self.session.phase,
            rounds_completed: self.session.current_round,
            final_check,
            consensus_reached,
            termination_reason,
            session: self.session.clone(),
        })
    }

    /// Get a reference to the current session.
    pub fn session(&self) -> &DebateSession {
        &self.session
    }

    /// Get the accumulated consensus checks.
    pub fn checks(&self) -> &[ConsensusCheck] {
        &self.checks
    }

    /// Current round number.
    pub fn current_round(&self) -> u32 {
        self.session.current_round
    }

    /// Update elapsed time tracking.
    pub fn set_elapsed_ms(&mut self, elapsed_ms: u64) {
        self.elapsed_ms = elapsed_ms;
    }

    fn update_elapsed(&mut self) {
        if self.start_time_ms > 0 {
            let now = Utc::now().timestamp_millis() as u64;
            self.elapsed_ms = now.saturating_sub(self.start_time_ms);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approve_output(confidence: f64) -> ReviewerOutput {
        ReviewerOutput {
            check: ConsensusCheck {
                verdict: Verdict::Approve,
                confidence,
                blocking_issues: vec![],
                suggestions: vec![],
                approach_aligned: true,
            },
            summary: "Approved".to_string(),
        }
    }

    fn changes_output(issues: Vec<String>) -> ReviewerOutput {
        ReviewerOutput {
            check: ConsensusCheck {
                verdict: Verdict::RequestChanges,
                confidence: 0.9,
                blocking_issues: issues,
                suggestions: vec![],
                approach_aligned: true,
            },
            summary: "Changes requested".to_string(),
        }
    }

    fn abstain_output() -> ReviewerOutput {
        ReviewerOutput {
            check: ConsensusCheck {
                verdict: Verdict::Abstain,
                confidence: 0.3,
                blocking_issues: vec![],
                suggestions: vec![],
                approach_aligned: false,
            },
            summary: "Cannot evaluate".to_string(),
        }
    }

    fn code_output() -> CoderOutput {
        CoderOutput {
            code: "fn main() {}".to_string(),
            files_changed: vec!["src/main.rs".to_string()],
            explanation: "Initial implementation".to_string(),
        }
    }

    #[test]
    fn test_single_round_approval() {
        let mut orch = DebateOrchestrator::new("d-001", "issue-1", "diff");
        orch.start().unwrap();

        assert_eq!(orch.next_action(), NextAction::AwaitCoder);

        orch.submit_code(code_output()).unwrap();
        assert_eq!(orch.next_action(), NextAction::AwaitReviewer);

        let action = orch.submit_review(approve_output(0.95)).unwrap();
        assert_eq!(action, NextAction::Complete);
        assert!(orch.is_complete());

        let outcome = orch.outcome().unwrap();
        assert!(outcome.is_success());
        assert!(outcome.consensus_reached);
        assert_eq!(outcome.rounds_completed, 1);
        assert!(outcome.summary_line().contains("RESOLVED"));
    }

    #[test]
    fn test_multi_round_then_approval() {
        let mut orch = DebateOrchestrator::new("d-002", "issue-2", "diff");
        orch.start().unwrap();

        // Round 1: changes requested
        orch.submit_code(code_output()).unwrap();
        let action = orch
            .submit_review(changes_output(vec!["missing error handling".to_string()]))
            .unwrap();
        assert_eq!(action, NextAction::AwaitCoder);
        assert_eq!(orch.current_round(), 2);

        // Round 2: approved
        orch.submit_code(code_output()).unwrap();
        let action = orch.submit_review(approve_output(0.9)).unwrap();
        assert_eq!(action, NextAction::Complete);

        let outcome = orch.outcome().unwrap();
        assert!(outcome.is_success());
        assert_eq!(outcome.rounds_completed, 2);
    }

    #[test]
    fn test_max_rounds_deadlock() {
        let config = DebateConfig {
            max_rounds: 2,
            guardrails: GuardrailConfig {
                max_rounds: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut orch = DebateOrchestrator::with_config("d-003", "issue-3", "diff", config);
        orch.start().unwrap();

        // Round 1: changes
        orch.submit_code(code_output()).unwrap();
        let action = orch
            .submit_review(changes_output(vec!["issue-1".to_string()]))
            .unwrap();
        assert_eq!(action, NextAction::AwaitCoder);

        // Round 2: still changes — max rounds hit
        orch.submit_code(code_output()).unwrap();
        let action = orch
            .submit_review(changes_output(vec!["issue-1".to_string()]))
            .unwrap();
        assert_eq!(action, NextAction::Complete);

        let outcome = orch.outcome().unwrap();
        assert!(!outcome.is_success());
        assert!(outcome.needs_escalation());
        assert_eq!(outcome.terminal_phase, DebatePhase::Deadlocked);
        assert!(outcome.summary_line().contains("ESCALATED"));
    }

    #[test]
    fn test_escalation_on_abstain() {
        let mut orch = DebateOrchestrator::new("d-004", "issue-4", "diff");
        orch.start().unwrap();

        orch.submit_code(code_output()).unwrap();
        // First a changes check, then an abstain
        orch.submit_review(changes_output(vec!["issue".to_string()]))
            .unwrap();
        orch.submit_code(code_output()).unwrap();
        let action = orch.submit_review(abstain_output()).unwrap();
        assert_eq!(action, NextAction::Complete);

        let outcome = orch.outcome().unwrap();
        assert_eq!(outcome.terminal_phase, DebatePhase::Escalated);
        assert!(outcome.needs_escalation());
    }

    #[test]
    fn test_abort() {
        let mut orch = DebateOrchestrator::new("d-005", "issue-5", "diff");
        orch.start().unwrap();
        orch.abort("user cancelled").unwrap();
        assert!(orch.is_complete());

        let outcome = orch.outcome().unwrap();
        assert_eq!(outcome.terminal_phase, DebatePhase::Aborted);
        assert!(!outcome.is_success());
        assert!(!outcome.needs_escalation());
        assert!(outcome.summary_line().contains("ABORTED"));
    }

    #[test]
    fn test_abort_after_complete_fails() {
        let mut orch = DebateOrchestrator::new("d-006", "issue-6", "diff");
        orch.start().unwrap();
        orch.submit_code(code_output()).unwrap();
        orch.submit_review(approve_output(0.95)).unwrap();

        let err = orch.abort("too late").unwrap_err();
        assert_eq!(err, DebateError::AlreadyComplete);
    }

    #[test]
    fn test_wrong_phase_submit_code() {
        let mut orch = DebateOrchestrator::new("d-007", "issue-7", "diff");
        orch.start().unwrap();
        orch.submit_code(code_output()).unwrap();

        // Now in ReviewerTurn — submitting code should fail
        let err = orch.submit_code(code_output()).unwrap_err();
        assert!(matches!(err, DebateError::InvalidPhase { .. }));
    }

    #[test]
    fn test_wrong_phase_submit_review() {
        let mut orch = DebateOrchestrator::new("d-008", "issue-8", "diff");
        orch.start().unwrap();

        // In CoderTurn — submitting review should fail
        let err = orch.submit_review(approve_output(0.9)).unwrap_err();
        assert!(matches!(err, DebateError::InvalidPhase { .. }));
    }

    #[test]
    fn test_outcome_before_complete_is_none() {
        let orch = DebateOrchestrator::new("d-009", "issue-9", "diff");
        assert!(orch.outcome().is_none());
    }

    #[test]
    fn test_low_confidence_approval_continues() {
        let mut orch = DebateOrchestrator::new("d-010", "issue-10", "diff");
        orch.start().unwrap();
        orch.submit_code(code_output()).unwrap();

        // Low confidence approval doesn't reach consensus
        let action = orch.submit_review(approve_output(0.4)).unwrap();
        assert_eq!(action, NextAction::AwaitCoder);
        assert!(!orch.is_complete());
    }

    #[test]
    fn test_debate_config_default() {
        let config = DebateConfig::default();
        assert_eq!(config.max_rounds, 5);
    }

    #[test]
    fn test_debate_error_display() {
        let err = DebateError::TransitionFailed("bad".to_string());
        assert!(err.to_string().contains("bad"));

        let err = DebateError::AlreadyComplete;
        assert!(err.to_string().contains("already complete"));

        let err = DebateError::InvalidPhase {
            expected: "coder_turn".to_string(),
            actual: "reviewer_turn".to_string(),
        };
        assert!(err.to_string().contains("coder_turn"));
    }

    #[test]
    fn test_next_action_display() {
        assert_eq!(NextAction::AwaitCoder.to_string(), "await_coder");
        assert_eq!(NextAction::AwaitReviewer.to_string(), "await_reviewer");
        assert_eq!(NextAction::Complete.to_string(), "complete");
    }

    #[test]
    fn test_checks_accessor() {
        let mut orch = DebateOrchestrator::new("d-011", "issue-11", "diff");
        orch.start().unwrap();
        assert!(orch.checks().is_empty());

        orch.submit_code(code_output()).unwrap();
        orch.submit_review(approve_output(0.95)).unwrap();
        assert_eq!(orch.checks().len(), 1);
    }
}
