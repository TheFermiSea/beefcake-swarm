//! Debate persistence — checkpoint and resume for interrupted debates.
//!
//! Supports serialization of debate state to JSON for checkpointing,
//! and restoration with integrity validation to prevent semantic drift.

use serde::{Deserialize, Serialize};

use super::consensus::ConsensusCheck;
use super::critique::PatchCritique;
use super::state::DebateSession;

/// A complete debate checkpoint for serialization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebateCheckpoint {
    /// Schema version for forward compatibility.
    pub version: u32,
    /// The session state at checkpoint time.
    pub session: DebateSession,
    /// Accumulated consensus checks.
    pub checks: Vec<ConsensusCheck>,
    /// Accumulated critique history.
    pub critiques: Vec<PatchCritique>,
    /// Elapsed time in milliseconds at checkpoint.
    pub elapsed_ms: u64,
    /// Checkpoint reason.
    pub reason: String,
    /// Monotonic checkpoint sequence number.
    pub sequence: u32,
}

impl DebateCheckpoint {
    /// Current schema version.
    pub const CURRENT_VERSION: u32 = 1;

    /// Create a new checkpoint.
    pub fn new(
        session: &DebateSession,
        checks: &[ConsensusCheck],
        critiques: &[PatchCritique],
        elapsed_ms: u64,
        reason: &str,
        sequence: u32,
    ) -> Self {
        Self {
            version: Self::CURRENT_VERSION,
            session: session.clone(),
            checks: checks.to_vec(),
            critiques: critiques.to_vec(),
            elapsed_ms,
            reason: reason.to_string(),
            sequence,
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, PersistenceError> {
        serde_json::to_string_pretty(self).map_err(|e| PersistenceError::SerializeFailed {
            reason: e.to_string(),
        })
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, PersistenceError> {
        let checkpoint: Self =
            serde_json::from_str(json).map_err(|e| PersistenceError::DeserializeFailed {
                reason: e.to_string(),
            })?;

        if checkpoint.version > Self::CURRENT_VERSION {
            return Err(PersistenceError::VersionMismatch {
                expected: Self::CURRENT_VERSION,
                found: checkpoint.version,
            });
        }

        Ok(checkpoint)
    }
}

/// Error during persistence operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PersistenceError {
    /// Serialization failed.
    SerializeFailed { reason: String },
    /// Deserialization failed.
    DeserializeFailed { reason: String },
    /// Schema version mismatch.
    VersionMismatch { expected: u32, found: u32 },
    /// Integrity check failed on restore.
    IntegrityCheckFailed { reason: String },
    /// Checkpoint sequence is stale.
    StaleCheckpoint { expected: u32, found: u32 },
}

impl std::fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SerializeFailed { reason } => write!(f, "serialize failed: {}", reason),
            Self::DeserializeFailed { reason } => write!(f, "deserialize failed: {}", reason),
            Self::VersionMismatch { expected, found } => {
                write!(
                    f,
                    "version mismatch: expected {}, found {}",
                    expected, found
                )
            }
            Self::IntegrityCheckFailed { reason } => {
                write!(f, "integrity check failed: {}", reason)
            }
            Self::StaleCheckpoint { expected, found } => {
                write!(
                    f,
                    "stale checkpoint: expected seq {}, found {}",
                    expected, found
                )
            }
        }
    }
}

impl std::error::Error for PersistenceError {}

/// Integrity check result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityStatus {
    /// Checkpoint is valid and can be resumed.
    Valid,
    /// Checkpoint has minor issues but is recoverable.
    Recoverable { warnings: Vec<String> },
    /// Checkpoint is corrupted and cannot be used.
    Corrupted { errors: Vec<String> },
}

impl IntegrityStatus {
    /// Whether resume is safe.
    pub fn can_resume(&self) -> bool {
        matches!(self, Self::Valid | Self::Recoverable { .. })
    }
}

/// Validate a checkpoint's integrity before resuming.
pub fn validate_checkpoint(checkpoint: &DebateCheckpoint) -> IntegrityStatus {
    let mut errors: Vec<String> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    // Check version
    if checkpoint.version > DebateCheckpoint::CURRENT_VERSION {
        errors.push(format!(
            "version {} > current {}",
            checkpoint.version,
            DebateCheckpoint::CURRENT_VERSION
        ));
    }

    // Check round consistency
    let expected_checks = checkpoint.session.current_round.min(
        checkpoint
            .session
            .rounds
            .len()
            .try_into()
            .unwrap_or(u32::MAX),
    );
    if checkpoint.checks.len() as u32 > expected_checks + 1 {
        warnings.push(format!(
            "more checks ({}) than expected for round {}",
            checkpoint.checks.len(),
            checkpoint.session.current_round
        ));
    }

    // Check critique history alignment
    for critique in &checkpoint.critiques {
        if critique.round > checkpoint.session.current_round {
            warnings.push(format!(
                "critique for round {} exceeds current round {}",
                critique.round, checkpoint.session.current_round
            ));
        }
    }

    // Check session isn't in impossible state
    if checkpoint.session.current_round > checkpoint.session.max_rounds + 1 {
        errors.push(format!(
            "current_round {} exceeds max_rounds {} by more than 1",
            checkpoint.session.current_round, checkpoint.session.max_rounds
        ));
    }

    // Check transition history consistency
    if !checkpoint.session.transitions.is_empty() {
        let last_transition = checkpoint.session.transitions.last().unwrap();
        if last_transition.to != checkpoint.session.phase {
            errors.push(format!(
                "last transition target {:?} doesn't match current phase {:?}",
                last_transition.to, checkpoint.session.phase
            ));
        }
    }

    if !errors.is_empty() {
        IntegrityStatus::Corrupted { errors }
    } else if !warnings.is_empty() {
        IntegrityStatus::Recoverable { warnings }
    } else {
        IntegrityStatus::Valid
    }
}

/// Manages checkpointing for a debate session.
pub struct CheckpointManager {
    /// Current checkpoint sequence number.
    sequence: u32,
    /// Maximum checkpoints to retain.
    max_retained: usize,
    /// Stored checkpoints (most recent last).
    checkpoints: Vec<DebateCheckpoint>,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(max_retained: usize) -> Self {
        Self {
            sequence: 0,
            max_retained,
            checkpoints: Vec::new(),
        }
    }

    /// Create a checkpoint of the current debate state.
    pub fn checkpoint(
        &mut self,
        session: &DebateSession,
        checks: &[ConsensusCheck],
        critiques: &[PatchCritique],
        elapsed_ms: u64,
        reason: &str,
    ) -> DebateCheckpoint {
        self.sequence += 1;
        let cp = DebateCheckpoint::new(
            session,
            checks,
            critiques,
            elapsed_ms,
            reason,
            self.sequence,
        );

        self.checkpoints.push(cp.clone());

        // Evict old checkpoints
        while self.checkpoints.len() > self.max_retained {
            self.checkpoints.remove(0);
        }

        cp
    }

    /// Get the latest checkpoint.
    pub fn latest(&self) -> Option<&DebateCheckpoint> {
        self.checkpoints.last()
    }

    /// Get a checkpoint by sequence number.
    pub fn get_by_sequence(&self, seq: u32) -> Option<&DebateCheckpoint> {
        self.checkpoints.iter().find(|cp| cp.sequence == seq)
    }

    /// Number of stored checkpoints.
    pub fn count(&self) -> usize {
        self.checkpoints.len()
    }

    /// Current sequence number.
    pub fn current_sequence(&self) -> u32 {
        self.sequence
    }

    /// Restore from a JSON checkpoint, validating integrity.
    pub fn restore(json: &str) -> Result<(DebateCheckpoint, IntegrityStatus), PersistenceError> {
        let checkpoint = DebateCheckpoint::from_json(json)?;
        let status = validate_checkpoint(&checkpoint);

        if !status.can_resume() {
            if let IntegrityStatus::Corrupted { ref errors } = status {
                return Err(PersistenceError::IntegrityCheckFailed {
                    reason: errors.join("; "),
                });
            }
        }

        Ok((checkpoint, status))
    }
}

#[cfg(test)]
mod tests {
    use super::super::consensus::Verdict;
    use super::super::state::DebatePhase;
    use super::*;

    fn make_session() -> DebateSession {
        let mut session = DebateSession::new("d-001", "issue-1", "diff", 5);
        session.start().unwrap();
        session
    }

    fn make_check() -> ConsensusCheck {
        ConsensusCheck {
            verdict: Verdict::RequestChanges,
            confidence: 0.8,
            blocking_issues: vec!["issue-1".to_string()],
            suggestions: vec![],
            approach_aligned: true,
        }
    }

    fn make_critique() -> PatchCritique {
        PatchCritique::new(1, "Needs work")
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let session = make_session();
        let checks = vec![make_check()];
        let critiques = vec![make_critique()];

        let cp = DebateCheckpoint::new(&session, &checks, &critiques, 1000, "test", 1);
        let json = cp.to_json().unwrap();
        let restored = DebateCheckpoint::from_json(&json).unwrap();

        assert_eq!(restored.version, DebateCheckpoint::CURRENT_VERSION);
        assert_eq!(restored.session.id, "d-001");
        assert_eq!(restored.checks.len(), 1);
        assert_eq!(restored.critiques.len(), 1);
        assert_eq!(restored.elapsed_ms, 1000);
        assert_eq!(restored.sequence, 1);
    }

    #[test]
    fn test_version_mismatch() {
        let session = make_session();
        let cp = DebateCheckpoint::new(&session, &[], &[], 0, "test", 1);
        let mut json_val: serde_json::Value = serde_json::to_value(&cp).unwrap();
        json_val["version"] = serde_json::Value::Number(serde_json::Number::from(999));
        let json = serde_json::to_string(&json_val).unwrap();

        let err = DebateCheckpoint::from_json(&json).unwrap_err();
        assert!(matches!(err, PersistenceError::VersionMismatch { .. }));
    }

    #[test]
    fn test_validate_valid_checkpoint() {
        let session = make_session();
        let cp = DebateCheckpoint::new(&session, &[], &[], 0, "test", 1);
        let status = validate_checkpoint(&cp);
        assert_eq!(status, IntegrityStatus::Valid);
        assert!(status.can_resume());
    }

    #[test]
    fn test_validate_transition_mismatch() {
        let mut session = make_session();
        // Manually corrupt: phase doesn't match last transition
        session.phase = DebatePhase::Resolved;
        let cp = DebateCheckpoint::new(&session, &[], &[], 0, "test", 1);
        let status = validate_checkpoint(&cp);
        assert!(matches!(status, IntegrityStatus::Corrupted { .. }));
        assert!(!status.can_resume());
    }

    #[test]
    fn test_validate_recoverable() {
        let session = make_session();
        // More checks than rounds — just a warning
        let checks = vec![make_check(), make_check(), make_check()];
        let cp = DebateCheckpoint::new(&session, &checks, &[], 0, "test", 1);
        let status = validate_checkpoint(&cp);
        assert!(matches!(status, IntegrityStatus::Recoverable { .. }));
        assert!(status.can_resume());
    }

    #[test]
    fn test_checkpoint_manager_basic() {
        let mut mgr = CheckpointManager::new(3);
        let session = make_session();

        let cp1 = mgr.checkpoint(&session, &[], &[], 100, "cp1");
        assert_eq!(cp1.sequence, 1);
        assert_eq!(mgr.count(), 1);

        let cp2 = mgr.checkpoint(&session, &[], &[], 200, "cp2");
        assert_eq!(cp2.sequence, 2);
        assert_eq!(mgr.count(), 2);

        assert_eq!(mgr.latest().unwrap().sequence, 2);
        assert_eq!(mgr.get_by_sequence(1).unwrap().elapsed_ms, 100);
    }

    #[test]
    fn test_checkpoint_manager_eviction() {
        let mut mgr = CheckpointManager::new(2);
        let session = make_session();

        mgr.checkpoint(&session, &[], &[], 100, "cp1");
        mgr.checkpoint(&session, &[], &[], 200, "cp2");
        mgr.checkpoint(&session, &[], &[], 300, "cp3");

        assert_eq!(mgr.count(), 2);
        // cp1 should be evicted
        assert!(mgr.get_by_sequence(1).is_none());
        assert!(mgr.get_by_sequence(2).is_some());
        assert!(mgr.get_by_sequence(3).is_some());
    }

    #[test]
    fn test_checkpoint_manager_restore() {
        let session = make_session();
        let cp = DebateCheckpoint::new(&session, &[], &[], 0, "test", 1);
        let json = cp.to_json().unwrap();

        let (restored, status) = CheckpointManager::restore(&json).unwrap();
        assert_eq!(restored.session.id, "d-001");
        assert!(status.can_resume());
    }

    #[test]
    fn test_checkpoint_manager_restore_corrupted() {
        let mut session = make_session();
        session.phase = DebatePhase::Resolved; // corrupt
        let cp = DebateCheckpoint::new(&session, &[], &[], 0, "test", 1);
        let json = cp.to_json().unwrap();

        let err = CheckpointManager::restore(&json).unwrap_err();
        assert!(matches!(err, PersistenceError::IntegrityCheckFailed { .. }));
    }

    #[test]
    fn test_persistence_error_display() {
        let err = PersistenceError::SerializeFailed {
            reason: "bad".to_string(),
        };
        assert!(err.to_string().contains("serialize"));

        let err = PersistenceError::VersionMismatch {
            expected: 1,
            found: 2,
        };
        assert!(err.to_string().contains("version mismatch"));

        let err = PersistenceError::IntegrityCheckFailed {
            reason: "corrupt".to_string(),
        };
        assert!(err.to_string().contains("integrity"));

        let err = PersistenceError::StaleCheckpoint {
            expected: 5,
            found: 3,
        };
        assert!(err.to_string().contains("stale"));
    }

    #[test]
    fn test_bad_json_deserialize() {
        let err = DebateCheckpoint::from_json("not json").unwrap_err();
        assert!(matches!(err, PersistenceError::DeserializeFailed { .. }));
    }

    #[test]
    fn test_current_sequence() {
        let mut mgr = CheckpointManager::new(5);
        assert_eq!(mgr.current_sequence(), 0);
        let session = make_session();
        mgr.checkpoint(&session, &[], &[], 0, "cp");
        assert_eq!(mgr.current_sequence(), 1);
    }
}
