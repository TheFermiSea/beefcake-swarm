//! Deadlock and iteration guardrails for debate sessions.

use serde::{Deserialize, Serialize};

use super::consensus::{ConsensusCheck, ConsensusOutcome, ConsensusProtocol};
use super::state::DebateSession;

/// Outcome when guardrails trigger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadlockOutcome {
    /// Continue debate — no guardrails triggered.
    Continue,
    /// Force deadlock — max rounds exceeded.
    MaxRoundsExceeded { rounds: u32 },
    /// Force deadlock — stall detected.
    StallDetected { stalled_rounds: u32 },
    /// Force escalation — reviewer cannot decide.
    EscalationRequired { reason: String },
    /// Force abort — timeout exceeded.
    TimeoutExceeded { elapsed_ms: u64, budget_ms: u64 },
}

impl DeadlockOutcome {
    /// Whether the debate should stop.
    pub fn should_stop(&self) -> bool {
        !matches!(self, Self::Continue)
    }
}

impl std::fmt::Display for DeadlockOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Continue => write!(f, "continue"),
            Self::MaxRoundsExceeded { rounds } => {
                write!(f, "max_rounds_exceeded ({})", rounds)
            }
            Self::StallDetected { stalled_rounds } => {
                write!(f, "stall_detected ({} rounds)", stalled_rounds)
            }
            Self::EscalationRequired { reason } => {
                write!(f, "escalation_required: {}", reason)
            }
            Self::TimeoutExceeded {
                elapsed_ms,
                budget_ms,
            } => {
                write!(f, "timeout_exceeded ({}ms / {}ms)", elapsed_ms, budget_ms)
            }
        }
    }
}

/// Configuration for debate guardrails.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardrailConfig {
    /// Maximum total debate time in milliseconds (0 = unlimited).
    pub timeout_ms: u64,
    /// Maximum rounds before forced deadlock.
    pub max_rounds: u32,
    /// Consensus protocol for stall detection.
    pub consensus: ConsensusProtocol,
}

impl Default for GuardrailConfig {
    fn default() -> Self {
        Self {
            timeout_ms: 0,
            max_rounds: 5,
            consensus: ConsensusProtocol::default(),
        }
    }
}

/// Engine that evaluates guardrails against debate state.
pub struct GuardrailEngine {
    config: GuardrailConfig,
}

impl GuardrailEngine {
    /// Create a new guardrail engine.
    pub fn new(config: GuardrailConfig) -> Self {
        Self { config }
    }

    /// Evaluate whether any guardrail has triggered.
    ///
    /// Call this before each new round to determine whether to continue.
    pub fn evaluate(
        &self,
        session: &DebateSession,
        checks: &[ConsensusCheck],
        elapsed_ms: u64,
    ) -> DeadlockOutcome {
        // 1. Timeout check
        if self.config.timeout_ms > 0 && elapsed_ms >= self.config.timeout_ms {
            return DeadlockOutcome::TimeoutExceeded {
                elapsed_ms,
                budget_ms: self.config.timeout_ms,
            };
        }

        // 2. Max rounds check
        if session.current_round >= self.config.max_rounds {
            return DeadlockOutcome::MaxRoundsExceeded {
                rounds: session.current_round,
            };
        }

        // 3. Consensus-based stall/escalation check
        let outcome = self.config.consensus.evaluate(checks);
        match outcome {
            ConsensusOutcome::Stalled => DeadlockOutcome::StallDetected {
                stalled_rounds: self.config.consensus.max_stalled_rounds,
            },
            ConsensusOutcome::NeedsEscalation => DeadlockOutcome::EscalationRequired {
                reason: "reviewer abstained".to_string(),
            },
            _ => DeadlockOutcome::Continue,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::consensus::Verdict;
    use super::*;

    fn changes_check(blocking: usize) -> ConsensusCheck {
        ConsensusCheck {
            verdict: Verdict::RequestChanges,
            confidence: 0.9,
            blocking_issues: (0..blocking).map(|i| format!("issue-{}", i)).collect(),
            suggestions: vec![],
            approach_aligned: true,
        }
    }

    fn abstain_check() -> ConsensusCheck {
        ConsensusCheck {
            verdict: Verdict::Abstain,
            confidence: 0.3,
            blocking_issues: vec![],
            suggestions: vec![],
            approach_aligned: false,
        }
    }

    #[test]
    fn test_continue_when_no_guardrails_triggered() {
        let engine = GuardrailEngine::new(GuardrailConfig::default());
        let session = DebateSession::new("d-001", "issue-1", "diff", 5);
        let checks = vec![changes_check(2)];

        let outcome = engine.evaluate(&session, &checks, 0);
        assert_eq!(outcome, DeadlockOutcome::Continue);
        assert!(!outcome.should_stop());
    }

    #[test]
    fn test_max_rounds_exceeded() {
        let engine = GuardrailEngine::new(GuardrailConfig {
            max_rounds: 3,
            ..Default::default()
        });
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 3);
        session.start().unwrap(); // round 1
        session
            .transition(super::super::state::DebatePhase::ReviewerTurn, "code")
            .unwrap();
        session
            .transition(super::super::state::DebatePhase::CoderTurn, "revise")
            .unwrap(); // round 2
        session
            .transition(super::super::state::DebatePhase::ReviewerTurn, "code")
            .unwrap();
        session
            .transition(super::super::state::DebatePhase::CoderTurn, "revise")
            .unwrap(); // round 3

        let outcome = engine.evaluate(&session, &[], 0);
        assert!(matches!(outcome, DeadlockOutcome::MaxRoundsExceeded { .. }));
        assert!(outcome.should_stop());
    }

    #[test]
    fn test_timeout_exceeded() {
        let engine = GuardrailEngine::new(GuardrailConfig {
            timeout_ms: 30_000,
            ..Default::default()
        });
        let session = DebateSession::new("d-001", "issue-1", "diff", 5);

        let outcome = engine.evaluate(&session, &[], 30_001);
        assert!(matches!(outcome, DeadlockOutcome::TimeoutExceeded { .. }));
    }

    #[test]
    fn test_stall_detected() {
        let engine = GuardrailEngine::new(GuardrailConfig {
            max_rounds: 10,
            consensus: ConsensusProtocol {
                max_stalled_rounds: 2,
                ..Default::default()
            },
            ..Default::default()
        });
        let session = DebateSession::new("d-001", "issue-1", "diff", 10);

        let checks = vec![
            changes_check(3),
            changes_check(3), // stalled (1)
            changes_check(4), // stalled (2)
        ];

        let outcome = engine.evaluate(&session, &checks, 0);
        assert!(matches!(outcome, DeadlockOutcome::StallDetected { .. }));
    }

    #[test]
    fn test_escalation_required() {
        let engine = GuardrailEngine::new(GuardrailConfig::default());
        let session = DebateSession::new("d-001", "issue-1", "diff", 5);

        let checks = vec![changes_check(2), abstain_check()];

        let outcome = engine.evaluate(&session, &checks, 0);
        assert!(matches!(
            outcome,
            DeadlockOutcome::EscalationRequired { .. }
        ));
    }

    #[test]
    fn test_timeout_takes_priority() {
        let engine = GuardrailEngine::new(GuardrailConfig {
            timeout_ms: 1000,
            max_rounds: 1,
            ..Default::default()
        });
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 3);
        session.start().unwrap(); // round 1

        // Both timeout and max_rounds exceeded — timeout first
        let outcome = engine.evaluate(&session, &[], 2000);
        assert!(matches!(outcome, DeadlockOutcome::TimeoutExceeded { .. }));
    }

    #[test]
    fn test_deadlock_outcome_display() {
        assert_eq!(DeadlockOutcome::Continue.to_string(), "continue");
        assert!(DeadlockOutcome::MaxRoundsExceeded { rounds: 5 }
            .to_string()
            .contains("5"));
        assert!(DeadlockOutcome::TimeoutExceeded {
            elapsed_ms: 5000,
            budget_ms: 3000
        }
        .to_string()
        .contains("5000ms"));
    }

    #[test]
    fn test_guardrail_config_json_roundtrip() {
        let config = GuardrailConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let parsed: GuardrailConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_rounds, 5);
        assert_eq!(parsed.timeout_ms, 0);
    }
}
