//! Debate state machine — phases, transitions, and session tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Phase of a debate session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DebatePhase {
    /// Session created but not started.
    Idle,
    /// Coder is producing/revising code.
    CoderTurn,
    /// Reviewer is evaluating the code.
    ReviewerTurn,
    /// Consensus reached — debate succeeded.
    Resolved,
    /// Max rounds exhausted without consensus.
    Deadlocked,
    /// Escalated to arbitration (ensemble or human).
    Escalated,
    /// Aborted by policy or intervention.
    Aborted,
}

impl DebatePhase {
    /// Whether this is a terminal phase.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Resolved | Self::Deadlocked | Self::Escalated | Self::Aborted
        )
    }

    /// Whether this phase allows transition to a new phase.
    pub fn can_transition(self) -> bool {
        !self.is_terminal()
    }

    /// Valid transitions from this phase.
    pub fn valid_transitions(self) -> &'static [DebatePhase] {
        match self {
            Self::Idle => &[Self::CoderTurn, Self::Aborted],
            Self::CoderTurn => &[Self::ReviewerTurn, Self::Aborted],
            Self::ReviewerTurn => &[
                Self::CoderTurn,
                Self::Resolved,
                Self::Deadlocked,
                Self::Escalated,
                Self::Aborted,
            ],
            Self::Resolved | Self::Deadlocked | Self::Escalated | Self::Aborted => &[],
        }
    }
}

impl std::fmt::Display for DebatePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Idle => write!(f, "idle"),
            Self::CoderTurn => write!(f, "coder_turn"),
            Self::ReviewerTurn => write!(f, "reviewer_turn"),
            Self::Resolved => write!(f, "resolved"),
            Self::Deadlocked => write!(f, "deadlocked"),
            Self::Escalated => write!(f, "escalated"),
            Self::Aborted => write!(f, "aborted"),
        }
    }
}

/// Role of a participant in the debate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticipantRole {
    /// Code producer.
    Coder,
    /// Code evaluator.
    Reviewer,
}

impl std::fmt::Display for ParticipantRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coder => write!(f, "coder"),
            Self::Reviewer => write!(f, "reviewer"),
        }
    }
}

/// Record of a single debate round (coder turn + reviewer turn).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundRecord {
    /// Round number (1-indexed).
    pub round: u32,
    /// Coder's output for this round.
    pub coder_output: String,
    /// Reviewer's verdict for this round.
    pub reviewer_verdict: String,
    /// Whether the reviewer approved the code.
    pub approved: bool,
    /// Specific issues raised by the reviewer.
    pub issues: Vec<String>,
    /// Round duration in milliseconds.
    pub duration_ms: u64,
    /// When this round started.
    pub started_at: DateTime<Utc>,
}

/// A phase transition record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateTransition {
    /// Previous phase.
    pub from: DebatePhase,
    /// New phase.
    pub to: DebatePhase,
    /// When the transition occurred.
    pub timestamp: DateTime<Utc>,
    /// Reason for the transition.
    pub reason: String,
}

/// Error for invalid state transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionError {
    pub from: DebatePhase,
    pub to: DebatePhase,
    pub reason: String,
}

impl std::fmt::Display for TransitionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "invalid transition {} → {}: {}",
            self.from, self.to, self.reason
        )
    }
}

impl std::error::Error for TransitionError {}

/// A debate session tracking state and history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateSession {
    /// Unique session identifier.
    pub id: String,
    /// Current phase.
    pub phase: DebatePhase,
    /// Current round number.
    pub current_round: u32,
    /// Maximum rounds allowed.
    pub max_rounds: u32,
    /// Round history.
    pub rounds: Vec<RoundRecord>,
    /// Transition history.
    pub transitions: Vec<DebateTransition>,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
    /// Issue/task being debated.
    pub issue_id: String,
    /// Diff or code being debated.
    pub subject: String,
}

impl DebateSession {
    /// Create a new debate session.
    pub fn new(id: &str, issue_id: &str, subject: &str, max_rounds: u32) -> Self {
        Self {
            id: id.to_string(),
            phase: DebatePhase::Idle,
            current_round: 0,
            max_rounds,
            rounds: Vec::new(),
            transitions: Vec::new(),
            created_at: Utc::now(),
            issue_id: issue_id.to_string(),
            subject: subject.to_string(),
        }
    }

    /// Transition to a new phase with a reason.
    pub fn transition(&mut self, to: DebatePhase, reason: &str) -> Result<(), TransitionError> {
        if !self.phase.valid_transitions().contains(&to) {
            return Err(TransitionError {
                from: self.phase,
                to,
                reason: format!(
                    "not a valid transition (allowed: {:?})",
                    self.phase.valid_transitions()
                ),
            });
        }

        self.transitions.push(DebateTransition {
            from: self.phase,
            to,
            timestamp: Utc::now(),
            reason: reason.to_string(),
        });
        self.phase = to;

        // Increment round when entering CoderTurn
        if to == DebatePhase::CoderTurn {
            self.current_round += 1;
        }

        Ok(())
    }

    /// Start the debate (Idle → CoderTurn).
    pub fn start(&mut self) -> Result<(), TransitionError> {
        self.transition(DebatePhase::CoderTurn, "debate started")
    }

    /// Record a complete round and transition.
    pub fn record_round(&mut self, record: RoundRecord) {
        self.rounds.push(record);
    }

    /// Whether the debate has ended.
    pub fn is_complete(&self) -> bool {
        self.phase.is_terminal()
    }

    /// Whether more rounds are available.
    pub fn has_rounds_remaining(&self) -> bool {
        self.current_round < self.max_rounds
    }

    /// Total elapsed rounds.
    pub fn elapsed_rounds(&self) -> u32 {
        self.current_round
    }

    /// Compact status line.
    pub fn status_line(&self) -> String {
        format!(
            "[{}] round {}/{} | {} rounds recorded | issue={}",
            self.phase,
            self.current_round,
            self.max_rounds,
            self.rounds.len(),
            self.issue_id
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_session() {
        let session = DebateSession::new("d-001", "beefcake-123", "diff content", 5);
        assert_eq!(session.phase, DebatePhase::Idle);
        assert_eq!(session.current_round, 0);
        assert_eq!(session.max_rounds, 5);
        assert!(!session.is_complete());
    }

    #[test]
    fn test_start_debate() {
        let mut session = DebateSession::new("d-001", "beefcake-123", "diff", 5);
        session.start().unwrap();
        assert_eq!(session.phase, DebatePhase::CoderTurn);
        assert_eq!(session.current_round, 1);
    }

    #[test]
    fn test_full_round_cycle() {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 3);
        session.start().unwrap();

        // Coder → Reviewer
        session
            .transition(DebatePhase::ReviewerTurn, "code submitted")
            .unwrap();
        assert_eq!(session.phase, DebatePhase::ReviewerTurn);

        // Reviewer → Coder (not approved, another round)
        session
            .transition(DebatePhase::CoderTurn, "needs revision")
            .unwrap();
        assert_eq!(session.current_round, 2);

        // Coder → Reviewer → Resolved
        session
            .transition(DebatePhase::ReviewerTurn, "revised code")
            .unwrap();
        session
            .transition(DebatePhase::Resolved, "reviewer approved")
            .unwrap();
        assert!(session.is_complete());
        assert_eq!(session.current_round, 2);
    }

    #[test]
    fn test_deadlock() {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 2);
        session.start().unwrap();
        session
            .transition(DebatePhase::ReviewerTurn, "code")
            .unwrap();
        session
            .transition(DebatePhase::Deadlocked, "max rounds")
            .unwrap();
        assert!(session.is_complete());
    }

    #[test]
    fn test_escalation() {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 2);
        session.start().unwrap();
        session
            .transition(DebatePhase::ReviewerTurn, "code")
            .unwrap();
        session
            .transition(DebatePhase::Escalated, "unresolvable")
            .unwrap();
        assert!(session.is_complete());
    }

    #[test]
    fn test_abort() {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 5);
        session.start().unwrap();
        session.transition(DebatePhase::Aborted, "timeout").unwrap();
        assert!(session.is_complete());
    }

    #[test]
    fn test_invalid_transition() {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 5);
        // Can't go from Idle to Resolved
        let err = session
            .transition(DebatePhase::Resolved, "skip")
            .unwrap_err();
        assert_eq!(err.from, DebatePhase::Idle);
        assert_eq!(err.to, DebatePhase::Resolved);
    }

    #[test]
    fn test_terminal_no_transitions() {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 5);
        session.start().unwrap();
        session
            .transition(DebatePhase::ReviewerTurn, "code")
            .unwrap();
        session
            .transition(DebatePhase::Resolved, "approved")
            .unwrap();

        let err = session
            .transition(DebatePhase::CoderTurn, "restart")
            .unwrap_err();
        assert_eq!(err.from, DebatePhase::Resolved);
    }

    #[test]
    fn test_has_rounds_remaining() {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 2);
        assert!(session.has_rounds_remaining());
        session.start().unwrap(); // round 1
        assert!(session.has_rounds_remaining());
        session
            .transition(DebatePhase::ReviewerTurn, "code")
            .unwrap();
        session
            .transition(DebatePhase::CoderTurn, "revise")
            .unwrap(); // round 2
        assert!(!session.has_rounds_remaining());
    }

    #[test]
    fn test_transition_history() {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 5);
        session.start().unwrap();
        session
            .transition(DebatePhase::ReviewerTurn, "submitted")
            .unwrap();
        session
            .transition(DebatePhase::Resolved, "approved")
            .unwrap();

        assert_eq!(session.transitions.len(), 3);
        assert_eq!(session.transitions[0].from, DebatePhase::Idle);
        assert_eq!(session.transitions[0].to, DebatePhase::CoderTurn);
        assert_eq!(session.transitions[2].to, DebatePhase::Resolved);
    }

    #[test]
    fn test_status_line() {
        let mut session = DebateSession::new("d-001", "beefcake-123", "diff", 5);
        session.start().unwrap();
        let line = session.status_line();
        assert!(line.contains("[coder_turn]"));
        assert!(line.contains("round 1/5"));
        assert!(line.contains("beefcake-123"));
    }

    #[test]
    fn test_phase_display() {
        assert_eq!(DebatePhase::Idle.to_string(), "idle");
        assert_eq!(DebatePhase::CoderTurn.to_string(), "coder_turn");
        assert_eq!(DebatePhase::ReviewerTurn.to_string(), "reviewer_turn");
        assert_eq!(DebatePhase::Resolved.to_string(), "resolved");
        assert_eq!(DebatePhase::Deadlocked.to_string(), "deadlocked");
        assert_eq!(DebatePhase::Escalated.to_string(), "escalated");
        assert_eq!(DebatePhase::Aborted.to_string(), "aborted");
    }

    #[test]
    fn test_participant_role_display() {
        assert_eq!(ParticipantRole::Coder.to_string(), "coder");
        assert_eq!(ParticipantRole::Reviewer.to_string(), "reviewer");
    }
}
