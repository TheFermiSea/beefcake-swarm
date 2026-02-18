//! Event types for ensemble coordination
//!
//! These events drive the pub/sub system and are persisted for replay.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::state::{ModelId, SessionId, TaskId, VotingStrategy};

/// Unique identifier for events
pub type EventId = String;

/// All ensemble coordination events
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EnsembleEvent {
    /// A new task was created
    TaskCreated {
        task_id: TaskId,
        session_id: SessionId,
        prompt_preview: String,
        require_consensus: bool,
        timestamp: DateTime<Utc>,
    },

    /// A task was assigned to a model
    TaskAssigned {
        task_id: TaskId,
        model_id: ModelId,
        timestamp: DateTime<Utc>,
    },

    /// A model was loaded into memory
    ModelLoaded {
        model_id: ModelId,
        load_time_ms: u64,
        timestamp: DateTime<Utc>,
    },

    /// A model was unloaded from memory
    ModelUnloaded {
        model_id: ModelId,
        reason: UnloadReason,
        timestamp: DateTime<Utc>,
    },

    /// A model submitted its result for a task
    ResultSubmitted {
        task_id: TaskId,
        model_id: ModelId,
        confidence: f32,
        tokens_used: u32,
        latency_ms: u64,
        timestamp: DateTime<Utc>,
    },

    /// Voting process started for a task
    VotingStarted {
        task_id: TaskId,
        strategy: VotingStrategy,
        participating_models: Vec<ModelId>,
        timestamp: DateTime<Utc>,
    },

    /// Consensus was reached on a task
    ConsensusReached {
        task_id: TaskId,
        winner: ModelId,
        vote_summary: VoteSummary,
        timestamp: DateTime<Utc>,
    },

    /// Arbitration was requested for a task
    ArbitrationRequested {
        task_id: TaskId,
        reason: ArbitrationReason,
        timestamp: DateTime<Utc>,
    },

    /// Arbitration completed
    ArbitrationCompleted {
        task_id: TaskId,
        decision: ModelId,
        rationale: String,
        timestamp: DateTime<Utc>,
    },

    /// Shared context was updated
    ContextUpdated {
        session_id: SessionId,
        version: u64,
        updated_by: ContextUpdater,
        summary_preview: String,
        timestamp: DateTime<Utc>,
    },

    /// A new session was created
    SessionCreated {
        session_id: SessionId,
        harness_session_id: Option<String>,
        timestamp: DateTime<Utc>,
    },

    /// A session ended
    SessionEnded {
        session_id: SessionId,
        reason: SessionEndReason,
        tasks_completed: u32,
        timestamp: DateTime<Utc>,
    },

    /// Task execution failed
    TaskFailed {
        task_id: TaskId,
        model_id: Option<ModelId>,
        error: String,
        timestamp: DateTime<Utc>,
    },

    /// A manager delegated work to another model
    ManagerDelegated {
        from: ModelId,
        to: ModelId,
        reason: String,
        timestamp: DateTime<Utc>,
    },

    /// A council of managers was convened for a decision
    CouncilConvened {
        members: Vec<ModelId>,
        task_id: String,
        timestamp: DateTime<Utc>,
    },

    /// The council reached a decision
    CouncilDecided {
        winner: ModelId,
        confidence: f32,
        timestamp: DateTime<Utc>,
    },
}

impl EnsembleEvent {
    /// Get the timestamp of this event
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            EnsembleEvent::TaskCreated { timestamp, .. } => *timestamp,
            EnsembleEvent::TaskAssigned { timestamp, .. } => *timestamp,
            EnsembleEvent::ModelLoaded { timestamp, .. } => *timestamp,
            EnsembleEvent::ModelUnloaded { timestamp, .. } => *timestamp,
            EnsembleEvent::ResultSubmitted { timestamp, .. } => *timestamp,
            EnsembleEvent::VotingStarted { timestamp, .. } => *timestamp,
            EnsembleEvent::ConsensusReached { timestamp, .. } => *timestamp,
            EnsembleEvent::ArbitrationRequested { timestamp, .. } => *timestamp,
            EnsembleEvent::ArbitrationCompleted { timestamp, .. } => *timestamp,
            EnsembleEvent::ContextUpdated { timestamp, .. } => *timestamp,
            EnsembleEvent::SessionCreated { timestamp, .. } => *timestamp,
            EnsembleEvent::SessionEnded { timestamp, .. } => *timestamp,
            EnsembleEvent::TaskFailed { timestamp, .. } => *timestamp,
            EnsembleEvent::ManagerDelegated { timestamp, .. } => *timestamp,
            EnsembleEvent::CouncilConvened { timestamp, .. } => *timestamp,
            EnsembleEvent::CouncilDecided { timestamp, .. } => *timestamp,
        }
    }

    /// Get the event type as a string
    pub fn event_type(&self) -> &'static str {
        match self {
            EnsembleEvent::TaskCreated { .. } => "task_created",
            EnsembleEvent::TaskAssigned { .. } => "task_assigned",
            EnsembleEvent::ModelLoaded { .. } => "model_loaded",
            EnsembleEvent::ModelUnloaded { .. } => "model_unloaded",
            EnsembleEvent::ResultSubmitted { .. } => "result_submitted",
            EnsembleEvent::VotingStarted { .. } => "voting_started",
            EnsembleEvent::ConsensusReached { .. } => "consensus_reached",
            EnsembleEvent::ArbitrationRequested { .. } => "arbitration_requested",
            EnsembleEvent::ArbitrationCompleted { .. } => "arbitration_completed",
            EnsembleEvent::ContextUpdated { .. } => "context_updated",
            EnsembleEvent::SessionCreated { .. } => "session_created",
            EnsembleEvent::SessionEnded { .. } => "session_ended",
            EnsembleEvent::TaskFailed { .. } => "task_failed",
            EnsembleEvent::ManagerDelegated { .. } => "manager_delegated",
            EnsembleEvent::CouncilConvened { .. } => "council_convened",
            EnsembleEvent::CouncilDecided { .. } => "council_decided",
        }
    }

    /// Get the session ID if this event is session-scoped
    pub fn session_id(&self) -> Option<&str> {
        match self {
            EnsembleEvent::TaskCreated { session_id, .. } => Some(session_id),
            EnsembleEvent::ContextUpdated { session_id, .. } => Some(session_id),
            EnsembleEvent::SessionCreated { session_id, .. } => Some(session_id),
            EnsembleEvent::SessionEnded { session_id, .. } => Some(session_id),
            _ => None,
        }
    }

    /// Get the task ID if this event is task-scoped
    pub fn task_id(&self) -> Option<&str> {
        match self {
            EnsembleEvent::TaskCreated { task_id, .. } => Some(task_id),
            EnsembleEvent::TaskAssigned { task_id, .. } => Some(task_id),
            EnsembleEvent::ResultSubmitted { task_id, .. } => Some(task_id),
            EnsembleEvent::VotingStarted { task_id, .. } => Some(task_id),
            EnsembleEvent::ConsensusReached { task_id, .. } => Some(task_id),
            EnsembleEvent::ArbitrationRequested { task_id, .. } => Some(task_id),
            EnsembleEvent::ArbitrationCompleted { task_id, .. } => Some(task_id),
            EnsembleEvent::TaskFailed { task_id, .. } => Some(task_id),
            EnsembleEvent::CouncilConvened { task_id, .. } => Some(task_id),
            _ => None,
        }
    }

    /// Create a new unique event ID
    pub fn new_id() -> EventId {
        uuid::Uuid::new_v4().to_string()
    }
}

/// Reason for unloading a model
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnloadReason {
    /// Model swap to load different model
    ModelSwap,
    /// Memory pressure
    MemoryPressure,
    /// Session ended
    SessionEnd,
    /// Explicit unload request
    Manual,
    /// Error during execution
    Error,
}

impl std::fmt::Display for UnloadReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnloadReason::ModelSwap => write!(f, "model_swap"),
            UnloadReason::MemoryPressure => write!(f, "memory_pressure"),
            UnloadReason::SessionEnd => write!(f, "session_end"),
            UnloadReason::Manual => write!(f, "manual"),
            UnloadReason::Error => write!(f, "error"),
        }
    }
}

/// Summary of voting results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteSummary {
    /// Total votes cast
    pub total_votes: u32,
    /// Votes for each model
    pub vote_counts: Vec<(ModelId, u32)>,
    /// Winning margin (winner votes - runner up votes)
    pub margin: u32,
    /// Average confidence of votes
    pub avg_confidence: f32,
}

impl VoteSummary {
    /// Create a new vote summary
    pub fn new(vote_counts: Vec<(ModelId, u32)>) -> Self {
        let total_votes: u32 = vote_counts.iter().map(|(_, c)| *c).sum();
        let sorted: Vec<_> = {
            let mut v = vote_counts.clone();
            v.sort_by(|a, b| b.1.cmp(&a.1));
            v
        };
        let margin = if sorted.len() >= 2 {
            sorted[0].1.saturating_sub(sorted[1].1)
        } else if !sorted.is_empty() {
            sorted[0].1
        } else {
            0
        };

        Self {
            total_votes,
            vote_counts,
            margin,
            avg_confidence: 0.5, // Default
        }
    }

    /// Set average confidence
    pub fn with_avg_confidence(mut self, confidence: f32) -> Self {
        self.avg_confidence = confidence;
        self
    }
}

/// Reason for requesting arbitration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ArbitrationReason {
    /// Tie vote between models
    TieVote { tied_models: Vec<ModelId> },
    /// Low confidence from all models
    LowConfidence { max_confidence: f32 },
    /// Conflicting responses
    ConflictingResponses { description: String },
    /// Explicit arbitration request
    ExplicitRequest { requester: String },
}

impl std::fmt::Display for ArbitrationReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArbitrationReason::TieVote { tied_models } => {
                write!(f, "tie_vote({} models)", tied_models.len())
            }
            ArbitrationReason::LowConfidence { max_confidence } => {
                write!(f, "low_confidence(max={:.2})", max_confidence)
            }
            ArbitrationReason::ConflictingResponses { .. } => {
                write!(f, "conflicting_responses")
            }
            ArbitrationReason::ExplicitRequest { requester } => {
                write!(f, "explicit_request({})", requester)
            }
        }
    }
}

/// What updated the shared context
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContextUpdater {
    /// Updated by a model after execution
    Model(ModelId),
    /// Updated by Claude overseer
    Overseer,
    /// Updated by the system (e.g., merging results)
    System,
    /// Updated manually
    Manual,
}

impl std::fmt::Display for ContextUpdater {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextUpdater::Model(id) => write!(f, "model:{}", id),
            ContextUpdater::Overseer => write!(f, "overseer"),
            ContextUpdater::System => write!(f, "system"),
            ContextUpdater::Manual => write!(f, "manual"),
        }
    }
}

/// Reason for session ending
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionEndReason {
    /// All tasks completed successfully
    Completed,
    /// Session was cancelled
    Cancelled,
    /// Session timed out
    Timeout,
    /// Error during processing
    Error(String),
    /// Harness session ended
    HarnessEnded,
}

impl std::fmt::Display for SessionEndReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionEndReason::Completed => write!(f, "completed"),
            SessionEndReason::Cancelled => write!(f, "cancelled"),
            SessionEndReason::Timeout => write!(f, "timeout"),
            SessionEndReason::Error(e) => write!(f, "error: {}", e),
            SessionEndReason::HarnessEnded => write!(f, "harness_ended"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_event_serialization() {
        let event = EnsembleEvent::TaskCreated {
            task_id: "task-1".to_string(),
            session_id: "session-1".to_string(),
            prompt_preview: "Test prompt...".to_string(),
            require_consensus: true,
            timestamp: Utc::now(),
        };

        let json = serde_json::to_string(&event).unwrap();
        let parsed: EnsembleEvent = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.event_type(), "task_created");
    }

    #[test]
    fn test_event_accessors() {
        let event = EnsembleEvent::ResultSubmitted {
            task_id: "task-1".to_string(),
            model_id: ModelId::Opus45,
            confidence: 0.9,
            tokens_used: 100,
            latency_ms: 500,
            timestamp: Utc::now(),
        };

        assert_eq!(event.task_id(), Some("task-1"));
        assert_eq!(event.session_id(), None);
        assert_eq!(event.event_type(), "result_submitted");
    }

    #[test]
    fn test_vote_summary() {
        let summary = VoteSummary::new(vec![(ModelId::Opus45, 2), (ModelId::HydraCoder, 1)]);

        assert_eq!(summary.total_votes, 3);
        assert_eq!(summary.margin, 1);
    }
}
