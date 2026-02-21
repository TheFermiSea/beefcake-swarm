//! Consensus protocol — structured verdict and agreement detection.

use serde::{Deserialize, Serialize};

/// Reviewer verdict in a debate round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// Code is acceptable as-is.
    Approve,
    /// Code needs specific changes.
    RequestChanges,
    /// Reviewer cannot evaluate — needs escalation.
    Abstain,
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Approve => write!(f, "approve"),
            Self::RequestChanges => write!(f, "request_changes"),
            Self::Abstain => write!(f, "abstain"),
        }
    }
}

/// Structured consensus check result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusCheck {
    /// The verdict from the reviewer.
    pub verdict: Verdict,
    /// Confidence in the verdict (0.0–1.0).
    pub confidence: f64,
    /// Blocking issues that prevent approval.
    pub blocking_issues: Vec<String>,
    /// Non-blocking suggestions.
    pub suggestions: Vec<String>,
    /// Whether the reviewer agrees with the coder's approach.
    pub approach_aligned: bool,
}

impl ConsensusCheck {
    /// Whether consensus was reached (approval with high confidence).
    pub fn is_consensus(&self) -> bool {
        self.verdict == Verdict::Approve
            && self.confidence >= 0.7
            && self.blocking_issues.is_empty()
    }

    /// Whether escalation is needed.
    pub fn needs_escalation(&self) -> bool {
        self.verdict == Verdict::Abstain
    }
}

/// Outcome of evaluating consensus across rounds.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConsensusOutcome {
    /// Full consensus — reviewer approved.
    Reached,
    /// Progress being made — issues decreasing.
    Progressing,
    /// Stalled — same issues recurring.
    Stalled,
    /// Cannot evaluate — needs external help.
    NeedsEscalation,
}

impl std::fmt::Display for ConsensusOutcome {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reached => write!(f, "reached"),
            Self::Progressing => write!(f, "progressing"),
            Self::Stalled => write!(f, "stalled"),
            Self::NeedsEscalation => write!(f, "needs_escalation"),
        }
    }
}

/// Protocol for evaluating consensus across debate rounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusProtocol {
    /// Minimum confidence for consensus.
    pub min_confidence: f64,
    /// Maximum consecutive stalled rounds before deadlock.
    pub max_stalled_rounds: u32,
}

impl ConsensusProtocol {
    /// Evaluate consensus from a sequence of checks.
    pub fn evaluate(&self, checks: &[ConsensusCheck]) -> ConsensusOutcome {
        if checks.is_empty() {
            return ConsensusOutcome::Progressing;
        }

        let latest = checks.last().unwrap();

        // Check for explicit escalation
        if latest.needs_escalation() {
            return ConsensusOutcome::NeedsEscalation;
        }

        // Check for consensus
        if latest.is_consensus() && latest.confidence >= self.min_confidence {
            return ConsensusOutcome::Reached;
        }

        // Check for stall: same blocking issues appearing repeatedly
        if checks.len() >= 2 {
            let stalled_count = self.count_stalled_rounds(checks);
            if stalled_count >= self.max_stalled_rounds {
                return ConsensusOutcome::Stalled;
            }
        }

        ConsensusOutcome::Progressing
    }

    /// Count consecutive rounds where blocking issues haven't decreased.
    fn count_stalled_rounds(&self, checks: &[ConsensusCheck]) -> u32 {
        let mut stalled = 0;
        for window in checks.windows(2) {
            let prev_count = window[0].blocking_issues.len();
            let curr_count = window[1].blocking_issues.len();
            if curr_count >= prev_count && prev_count > 0 {
                stalled += 1;
            } else {
                stalled = 0;
            }
        }
        stalled
    }
}

impl Default for ConsensusProtocol {
    fn default() -> Self {
        Self {
            min_confidence: 0.7,
            max_stalled_rounds: 2,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approved_check(confidence: f64) -> ConsensusCheck {
        ConsensusCheck {
            verdict: Verdict::Approve,
            confidence,
            blocking_issues: vec![],
            suggestions: vec![],
            approach_aligned: true,
        }
    }

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
    fn test_consensus_reached() {
        let protocol = ConsensusProtocol::default();
        let checks = vec![changes_check(2), changes_check(1), approved_check(0.95)];
        assert_eq!(protocol.evaluate(&checks), ConsensusOutcome::Reached);
    }

    #[test]
    fn test_consensus_low_confidence() {
        let protocol = ConsensusProtocol::default();
        let checks = vec![approved_check(0.5)]; // Below min_confidence
        assert_eq!(protocol.evaluate(&checks), ConsensusOutcome::Progressing);
    }

    #[test]
    fn test_stalled_detection() {
        let protocol = ConsensusProtocol {
            max_stalled_rounds: 2,
            ..Default::default()
        };
        let checks = vec![
            changes_check(3),
            changes_check(3), // Stalled (1)
            changes_check(4), // Stalled (2) — increased
        ];
        assert_eq!(protocol.evaluate(&checks), ConsensusOutcome::Stalled);
    }

    #[test]
    fn test_progressing() {
        let protocol = ConsensusProtocol::default();
        let checks = vec![changes_check(5), changes_check(3), changes_check(1)];
        assert_eq!(protocol.evaluate(&checks), ConsensusOutcome::Progressing);
    }

    #[test]
    fn test_escalation() {
        let protocol = ConsensusProtocol::default();
        let checks = vec![changes_check(2), abstain_check()];
        assert_eq!(
            protocol.evaluate(&checks),
            ConsensusOutcome::NeedsEscalation
        );
    }

    #[test]
    fn test_empty_checks() {
        let protocol = ConsensusProtocol::default();
        assert_eq!(protocol.evaluate(&[]), ConsensusOutcome::Progressing);
    }

    #[test]
    fn test_verdict_display() {
        assert_eq!(Verdict::Approve.to_string(), "approve");
        assert_eq!(Verdict::RequestChanges.to_string(), "request_changes");
        assert_eq!(Verdict::Abstain.to_string(), "abstain");
    }

    #[test]
    fn test_verdict_serde() {
        let json = serde_json::to_string(&Verdict::RequestChanges).unwrap();
        assert_eq!(json, "\"request_changes\"");
        let parsed: Verdict = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, Verdict::RequestChanges);
    }

    #[test]
    fn test_consensus_check_is_consensus() {
        let check = approved_check(0.95);
        assert!(check.is_consensus());

        let low_conf = approved_check(0.5);
        assert!(!low_conf.is_consensus());

        let with_issues = ConsensusCheck {
            verdict: Verdict::Approve,
            confidence: 0.9,
            blocking_issues: vec!["still broken".to_string()],
            suggestions: vec![],
            approach_aligned: true,
        };
        assert!(!with_issues.is_consensus());
    }

    #[test]
    fn test_consensus_outcome_display() {
        assert_eq!(ConsensusOutcome::Reached.to_string(), "reached");
        assert_eq!(ConsensusOutcome::Progressing.to_string(), "progressing");
        assert_eq!(ConsensusOutcome::Stalled.to_string(), "stalled");
        assert_eq!(
            ConsensusOutcome::NeedsEscalation.to_string(),
            "needs_escalation"
        );
    }

    #[test]
    fn test_protocol_json_roundtrip() {
        let protocol = ConsensusProtocol::default();
        let json = serde_json::to_string(&protocol).unwrap();
        let parsed: ConsensusProtocol = serde_json::from_str(&json).unwrap();
        assert!((parsed.min_confidence - 0.7).abs() < f64::EPSILON);
        assert_eq!(parsed.max_stalled_rounds, 2);
    }
}
