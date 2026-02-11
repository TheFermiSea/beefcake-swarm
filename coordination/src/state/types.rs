//! Core types for ensemble state persistence
//!
//! These types are stored in RocksDB and represent the persistent state
//! of multi-agent ensemble coordination.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for ensemble sessions
pub type SessionId = String;

/// Unique identifier for tasks within an ensemble
pub type TaskId = String;

/// Model identifier for participating LLMs
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelId {
    /// OR1-Behemoth 73B - Deep reasoning and architecture
    Behemoth,
    /// Strand-Rust-Coder 14B - Fast idiomatic code
    StrandCoder,
    /// HydraCoder 31B MoE - Specialized Rust generation
    HydraCoder,
}

impl ModelId {
    /// Get the voting weight for this model
    ///
    /// Higher weights indicate more trusted/capable models for the task domain.
    pub fn weight(&self) -> f32 {
        match self {
            ModelId::Behemoth => 1.0,    // Highest (reasoning)
            ModelId::HydraCoder => 0.85, // Specialized Rust
            ModelId::StrandCoder => 0.7, // Fast generalist
        }
    }

    /// Get the model name as used in API requests
    pub fn api_name(&self) -> &'static str {
        match self {
            ModelId::Behemoth => "OR1-Behemoth.Q8_0",
            ModelId::StrandCoder => "Strand-Rust-Coder-14B-v1-Q8_0",
            ModelId::HydraCoder => "HydraCoder.Q6_K",
        }
    }

    /// Get all model IDs for ensemble execution
    pub fn all() -> &'static [ModelId] {
        &[ModelId::StrandCoder, ModelId::HydraCoder, ModelId::Behemoth]
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelId::Behemoth => write!(f, "behemoth"),
            ModelId::StrandCoder => write!(f, "strand_coder"),
            ModelId::HydraCoder => write!(f, "hydra_coder"),
        }
    }
}

/// Status of an ensemble task
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    /// Task is queued waiting for processing
    Pending,
    /// Task is being processed by models
    InProgress,
    /// All models have responded, awaiting voting
    AwaitingVote,
    /// Voting is in progress
    Voting,
    /// Task requires arbitration (tie or low confidence)
    AwaitingArbitration,
    /// Task completed successfully
    Completed,
    /// Task failed
    Failed,
}

/// An ensemble session tracking multiple tasks across model swaps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleSession {
    /// Unique session identifier
    pub id: SessionId,

    /// Session creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last activity timestamp
    pub updated_at: DateTime<Utc>,

    /// Currently loaded model (if any)
    pub active_model: Option<ModelId>,

    /// IDs of pending tasks
    pub pending_tasks: Vec<TaskId>,

    /// IDs of completed tasks
    pub completed_tasks: Vec<TaskId>,

    /// Context version for coherence tracking
    pub context_version: u64,

    /// Optional link to harness session ID
    pub harness_session_id: Option<String>,

    /// Whether this session is active
    pub active: bool,
}

impl EnsembleSession {
    /// Create a new ensemble session
    pub fn new() -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            created_at: now,
            updated_at: now,
            active_model: None,
            pending_tasks: Vec::new(),
            completed_tasks: Vec::new(),
            context_version: 0,
            harness_session_id: None,
            active: true,
        }
    }

    /// Link to an existing harness session
    pub fn with_harness(mut self, harness_id: String) -> Self {
        self.harness_session_id = Some(harness_id);
        self
    }

    /// Touch the session to update last activity
    pub fn touch(&mut self) {
        self.updated_at = Utc::now();
    }

    /// Add a pending task
    pub fn add_task(&mut self, task_id: TaskId) {
        self.pending_tasks.push(task_id);
        self.touch();
    }

    /// Move a task from pending to completed
    pub fn complete_task(&mut self, task_id: &TaskId) {
        self.pending_tasks.retain(|id| id != task_id);
        self.completed_tasks.push(task_id.clone());
        self.touch();
    }
}

impl Default for EnsembleSession {
    fn default() -> Self {
        Self::new()
    }
}

/// A task submitted for ensemble processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleTask {
    /// Unique task identifier
    pub id: TaskId,

    /// Session this task belongs to
    pub session_id: SessionId,

    /// Task creation timestamp
    pub created_at: DateTime<Utc>,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,

    /// The prompt/question for models
    pub prompt: String,

    /// Optional code context
    pub code_context: Option<String>,

    /// Current task status
    pub status: TaskStatus,

    /// Models assigned to this task
    pub assigned_models: Vec<ModelId>,

    /// Models that have submitted results
    pub completed_models: Vec<ModelId>,

    /// Whether consensus is required (all models must respond)
    pub require_consensus: bool,

    /// Maximum tokens for responses
    pub max_tokens: u32,

    /// Final selected response (after voting/arbitration)
    pub final_response: Option<String>,

    /// ID of winning model (after voting)
    pub winning_model: Option<ModelId>,
}

impl EnsembleTask {
    /// Create a new task for ensemble processing
    pub fn new(session_id: SessionId, prompt: String, require_consensus: bool) -> Self {
        let now = Utc::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id,
            created_at: now,
            updated_at: now,
            prompt,
            code_context: None,
            status: TaskStatus::Pending,
            assigned_models: if require_consensus {
                ModelId::all().to_vec()
            } else {
                vec![ModelId::Behemoth] // Default to most capable
            },
            completed_models: Vec::new(),
            require_consensus,
            max_tokens: 2048,
            final_response: None,
            winning_model: None,
        }
    }

    /// Set code context
    pub fn with_code(mut self, code: String) -> Self {
        self.code_context = Some(code);
        self
    }

    /// Set max tokens
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Check if all assigned models have completed
    pub fn all_models_complete(&self) -> bool {
        self.assigned_models
            .iter()
            .all(|m| self.completed_models.contains(m))
    }

    /// Mark a model as having completed
    pub fn mark_model_complete(&mut self, model: ModelId) {
        if !self.completed_models.contains(&model) {
            self.completed_models.push(model);
        }
        self.updated_at = Utc::now();

        // Update status if all models done
        if self.all_models_complete() {
            self.status = TaskStatus::AwaitingVote;
        }
    }
}

/// Result from a single model's execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResult {
    /// Task this result is for
    pub task_id: TaskId,

    /// Model that produced this result
    pub model_id: ModelId,

    /// Timestamp when result was received
    pub timestamp: DateTime<Utc>,

    /// The response content
    pub response: String,

    /// Reasoning content (if model supports it)
    pub reasoning: Option<String>,

    /// Self-reported confidence (0.0-1.0)
    pub confidence: f32,

    /// Token count for the response
    pub tokens_used: u32,

    /// Response latency in milliseconds
    pub latency_ms: u64,

    /// Whether this result was selected as winner
    pub selected: bool,
}

impl ModelResult {
    /// Create a new model result
    pub fn new(
        task_id: TaskId,
        model_id: ModelId,
        response: String,
        tokens_used: u32,
        latency_ms: u64,
    ) -> Self {
        Self {
            task_id,
            model_id,
            timestamp: Utc::now(),
            response,
            reasoning: None,
            confidence: 0.5, // Default neutral confidence
            tokens_used,
            latency_ms,
            selected: false,
        }
    }

    /// Set reasoning content
    pub fn with_reasoning(mut self, reasoning: String) -> Self {
        self.reasoning = Some(reasoning);
        self
    }

    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
}

/// Voting strategy for consensus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VotingStrategy {
    /// Simple majority (>50%)
    Majority,
    /// Weighted by model confidence and model weight
    Weighted,
    /// Unanimous agreement required
    Unanimous,
}

/// Record of a voting decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoteRecord {
    /// Task this vote is for
    pub task_id: TaskId,

    /// Voting strategy used
    pub strategy: VotingStrategy,

    /// Vote timestamp
    pub timestamp: DateTime<Utc>,

    /// Individual votes: model -> (selected_model, weight)
    pub votes: HashMap<ModelId, (ModelId, f32)>,

    /// Winning model
    pub winner: Option<ModelId>,

    /// Whether arbitration was needed
    pub arbitrated: bool,

    /// Arbitration reason (if applicable)
    pub arbitration_reason: Option<String>,

    /// Notes from voting process
    pub notes: Option<String>,
}

impl VoteRecord {
    /// Create a new vote record
    pub fn new(task_id: TaskId, strategy: VotingStrategy) -> Self {
        Self {
            task_id,
            strategy,
            timestamp: Utc::now(),
            votes: HashMap::new(),
            winner: None,
            arbitrated: false,
            arbitration_reason: None,
            notes: None,
        }
    }

    /// Record a vote
    pub fn add_vote(&mut self, voter: ModelId, selected: ModelId, weight: f32) {
        self.votes.insert(voter, (selected, weight));
    }

    /// Set winner
    pub fn set_winner(&mut self, winner: ModelId) {
        self.winner = Some(winner);
    }

    /// Mark as arbitrated
    pub fn mark_arbitrated(&mut self, reason: String) {
        self.arbitrated = true;
        self.arbitration_reason = Some(reason);
    }
}

/// Shared context maintained across model swaps
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedContext {
    /// Session this context belongs to
    pub session_id: SessionId,

    /// Context version (monotonically increasing)
    pub version: u64,

    /// Last update timestamp
    pub updated_at: DateTime<Utc>,

    /// Summary of conversation/work so far
    pub summary: String,

    /// Key decisions made
    pub key_decisions: Vec<String>,

    /// Important file references
    pub file_references: Vec<String>,

    /// Domain-specific context
    pub domain_context: HashMap<String, String>,
}

impl SharedContext {
    /// Create a new empty context
    pub fn new(session_id: SessionId) -> Self {
        Self {
            session_id,
            version: 0,
            updated_at: Utc::now(),
            summary: String::new(),
            key_decisions: Vec::new(),
            file_references: Vec::new(),
            domain_context: HashMap::new(),
        }
    }

    /// Update the summary and increment version
    pub fn update_summary(&mut self, summary: String) {
        self.summary = summary;
        self.version += 1;
        self.updated_at = Utc::now();
    }

    /// Add a key decision
    pub fn add_decision(&mut self, decision: String) {
        self.key_decisions.push(decision);
        self.version += 1;
        self.updated_at = Utc::now();
    }

    /// Add file reference
    pub fn add_file_reference(&mut self, file: String) {
        if !self.file_references.contains(&file) {
            self.file_references.push(file);
            self.version += 1;
            self.updated_at = Utc::now();
        }
    }

    /// Set domain context
    pub fn set_domain(&mut self, key: String, value: String) {
        self.domain_context.insert(key, value);
        self.version += 1;
        self.updated_at = Utc::now();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_weights() {
        assert!(ModelId::Behemoth.weight() > ModelId::HydraCoder.weight());
        assert!(ModelId::HydraCoder.weight() > ModelId::StrandCoder.weight());
    }

    #[test]
    fn test_session_lifecycle() {
        let mut session = EnsembleSession::new();
        assert!(session.active);
        assert!(session.pending_tasks.is_empty());

        session.add_task("task-1".to_string());
        assert_eq!(session.pending_tasks.len(), 1);

        session.complete_task(&"task-1".to_string());
        assert!(session.pending_tasks.is_empty());
        assert_eq!(session.completed_tasks.len(), 1);
    }

    #[test]
    fn test_task_consensus() {
        let mut task = EnsembleTask::new(
            "session-1".to_string(),
            "Test prompt".to_string(),
            true, // require consensus
        );

        assert_eq!(task.assigned_models.len(), 3);
        assert_eq!(task.status, TaskStatus::Pending);

        task.mark_model_complete(ModelId::StrandCoder);
        assert_eq!(task.status, TaskStatus::Pending);

        task.mark_model_complete(ModelId::HydraCoder);
        assert_eq!(task.status, TaskStatus::Pending);

        task.mark_model_complete(ModelId::Behemoth);
        assert_eq!(task.status, TaskStatus::AwaitingVote);
    }

    #[test]
    fn test_context_versioning() {
        let mut ctx = SharedContext::new("session-1".to_string());
        assert_eq!(ctx.version, 0);

        ctx.update_summary("Initial summary".to_string());
        assert_eq!(ctx.version, 1);

        ctx.add_decision("Use RocksDB".to_string());
        assert_eq!(ctx.version, 2);
    }
}
