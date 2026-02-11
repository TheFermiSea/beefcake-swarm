//! Arbitration manager for Claude overseer integration
//!
//! Handles cases where voting fails to produce a clear winner,
//! allowing Claude (the overseer) to make final decisions.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::events::{ArbitrationReason, EnsembleEvent, SharedEventBus};
use crate::state::{ModelId, ModelResult, SharedStateStore, TaskId, TaskStatus};

/// Error type for arbitration operations
#[derive(Debug, thiserror::Error)]
pub enum ArbitrationError {
    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("No results available for arbitration")]
    NoResults,

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Task not awaiting arbitration")]
    NotAwaitingArbitration,
}

/// Result type for arbitration operations
pub type ArbitrationResult<T> = Result<T, ArbitrationError>;

/// Request for Claude to arbitrate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrationRequest {
    /// Task requiring arbitration
    pub task_id: TaskId,
    /// Reason for arbitration
    pub reason: ArbitrationReason,
    /// All model responses for comparison
    pub responses: Vec<ModelResponseSummary>,
    /// The original prompt
    pub original_prompt: String,
    /// Context from previous work
    pub context: Option<String>,
}

/// Summary of a model's response for arbitration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelResponseSummary {
    /// Model that produced this response
    pub model_id: ModelId,
    /// The response content
    pub response: String,
    /// Reasoning (if available)
    pub reasoning: Option<String>,
    /// Self-reported confidence
    pub confidence: f32,
    /// Response latency
    pub latency_ms: u64,
}

impl From<&ModelResult> for ModelResponseSummary {
    fn from(result: &ModelResult) -> Self {
        Self {
            model_id: result.model_id,
            response: result.response.clone(),
            reasoning: result.reasoning.clone(),
            confidence: result.confidence,
            latency_ms: result.latency_ms,
        }
    }
}

/// Decision from Claude after arbitration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrationDecision {
    /// Selected winning model
    pub winner: ModelId,
    /// Rationale for the decision
    pub rationale: String,
    /// Optional modified response (Claude can synthesize)
    pub modified_response: Option<String>,
    /// Additional notes
    pub notes: Option<String>,
}

/// Arbitration manager for overseer integration
pub struct ArbitrationManager {
    store: SharedStateStore,
    event_bus: SharedEventBus,
}

impl ArbitrationManager {
    /// Create a new arbitration manager
    pub fn new(store: SharedStateStore, event_bus: SharedEventBus) -> Self {
        Self { store, event_bus }
    }

    /// Request arbitration for a task
    pub fn request_arbitration(
        &self,
        task_id: &TaskId,
        reason: ArbitrationReason,
    ) -> ArbitrationResult<ArbitrationRequest> {
        // Get the task
        let task = self
            .store
            .get_task(task_id)
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?
            .ok_or_else(|| ArbitrationError::TaskNotFound(task_id.clone()))?;

        // Get all results for this task
        let results = self
            .store
            .get_task_results(task_id)
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?;

        if results.is_empty() {
            return Err(ArbitrationError::NoResults);
        }

        // Get context if available
        let context = self
            .store
            .get_context(&task.session_id)
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?
            .map(|c| c.summary);

        // Build responses summary
        let responses: Vec<ModelResponseSummary> = results.iter().map(|r| r.into()).collect();

        // Update task status
        let mut task = task;
        task.status = TaskStatus::AwaitingArbitration;
        self.store
            .put_task(&task)
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?;

        // Publish event
        let _ = self.event_bus.publish(EnsembleEvent::ArbitrationRequested {
            task_id: task_id.clone(),
            reason: reason.clone(),
            timestamp: Utc::now(),
        });

        info!(task_id, reason = %reason, "Arbitration requested");

        Ok(ArbitrationRequest {
            task_id: task_id.clone(),
            reason,
            responses,
            original_prompt: task.prompt,
            context,
        })
    }

    /// Apply an arbitration decision from Claude
    pub fn apply_decision(
        &self,
        task_id: &TaskId,
        decision: ArbitrationDecision,
    ) -> ArbitrationResult<()> {
        // Get the task
        let mut task = self
            .store
            .get_task(task_id)
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?
            .ok_or_else(|| ArbitrationError::TaskNotFound(task_id.clone()))?;

        if task.status != TaskStatus::AwaitingArbitration {
            warn!(
                task_id,
                status = ?task.status,
                "Task not awaiting arbitration"
            );
            // Allow anyway for flexibility
        }

        // Get the winning result
        let winner_result = self
            .store
            .get_result(task_id, &decision.winner)
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?;

        // Update task with decision
        task.status = TaskStatus::Completed;
        task.winning_model = Some(decision.winner);
        task.final_response = decision
            .modified_response
            .or_else(|| winner_result.map(|r| r.response));

        self.store
            .put_task(&task)
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?;

        // Mark the winning result as selected
        if let Some(mut result) = self
            .store
            .get_result(task_id, &decision.winner)
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?
        {
            result.selected = true;
            self.store
                .put_result(&result)
                .map_err(|e| ArbitrationError::StoreError(e.to_string()))?;
        }

        // Publish event
        let _ = self.event_bus.publish(EnsembleEvent::ArbitrationCompleted {
            task_id: task_id.clone(),
            decision: decision.winner,
            rationale: decision.rationale.clone(),
            timestamp: Utc::now(),
        });

        info!(
            task_id,
            winner = %decision.winner,
            rationale = %decision.rationale,
            "Arbitration decision applied"
        );

        Ok(())
    }

    /// Generate an arbitration prompt for Claude
    pub fn generate_arbitration_prompt(&self, request: &ArbitrationRequest) -> String {
        let mut prompt = String::new();

        prompt.push_str("# Arbitration Required\n\n");
        prompt.push_str(&format!("**Task ID:** {}\n\n", request.task_id));

        // Reason
        prompt.push_str("## Reason for Arbitration\n\n");
        match &request.reason {
            ArbitrationReason::TieVote { tied_models } => {
                prompt.push_str(&format!(
                    "Voting resulted in a tie between: {}\n\n",
                    tied_models
                        .iter()
                        .map(|m| m.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            ArbitrationReason::LowConfidence { max_confidence } => {
                prompt.push_str(&format!(
                    "All models reported low confidence (max: {:.2})\n\n",
                    max_confidence
                ));
            }
            ArbitrationReason::ConflictingResponses { description } => {
                prompt.push_str(&format!(
                    "Models produced conflicting responses: {}\n\n",
                    description
                ));
            }
            ArbitrationReason::ExplicitRequest { requester } => {
                prompt.push_str(&format!("Explicitly requested by: {}\n\n", requester));
            }
        }

        // Context
        if let Some(ref context) = request.context {
            prompt.push_str("## Previous Context\n\n");
            prompt.push_str(context);
            prompt.push_str("\n\n");
        }

        // Original prompt
        prompt.push_str("## Original Prompt\n\n");
        prompt.push_str(&request.original_prompt);
        prompt.push_str("\n\n");

        // Model responses
        prompt.push_str("## Model Responses\n\n");
        for (i, resp) in request.responses.iter().enumerate() {
            prompt.push_str(&format!(
                "### Response {} - {} (confidence: {:.2})\n\n",
                i + 1,
                resp.model_id,
                resp.confidence
            ));

            if let Some(ref reasoning) = resp.reasoning {
                prompt.push_str("**Reasoning:**\n");
                prompt.push_str(reasoning);
                prompt.push_str("\n\n");
            }

            prompt.push_str("**Response:**\n");
            prompt.push_str(&resp.response);
            prompt.push_str("\n\n");
        }

        // Instructions
        prompt.push_str("## Your Decision\n\n");
        prompt.push_str("Please select the best response and provide your rationale. Consider:\n");
        prompt.push_str("1. Technical correctness and Rust best practices\n");
        prompt.push_str("2. Code quality and maintainability\n");
        prompt.push_str("3. Alignment with the original request\n");
        prompt.push_str("4. Safety and error handling\n\n");
        prompt.push_str("You may also provide a modified/synthesized response if none are fully satisfactory.\n");

        prompt
    }

    /// Get pending arbitration requests
    pub fn get_pending_arbitrations(&self) -> ArbitrationResult<Vec<TaskId>> {
        // This would need a proper index in production
        // For now, scan through sessions and tasks
        let sessions = self
            .store
            .list_sessions()
            .map_err(|e| ArbitrationError::StoreError(e.to_string()))?;

        let mut pending = Vec::new();
        for session in sessions {
            for task_id in &session.pending_tasks {
                if let Some(task) = self
                    .store
                    .get_task(task_id)
                    .map_err(|e| ArbitrationError::StoreError(e.to_string()))?
                {
                    if task.status == TaskStatus::AwaitingArbitration {
                        pending.push(task_id.clone());
                    }
                }
            }
        }

        debug!(count = pending.len(), "Found pending arbitrations");
        Ok(pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventBus;
    use crate::state::{EnsembleSession, EnsembleTask, StateStore};
    use tempfile::tempdir;

    fn test_setup() -> (ArbitrationManager, SharedStateStore, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let store = StateStore::open(dir.path().join("test.db"))
            .unwrap()
            .shared();
        let bus = EventBus::new().shared();
        (ArbitrationManager::new(store.clone(), bus), store, dir)
    }

    #[test]
    fn test_request_arbitration() {
        let (manager, store, _dir) = test_setup();

        // Create session and task
        let session = EnsembleSession::new();
        store.put_session(&session).unwrap();

        let task = EnsembleTask::new(session.id.clone(), "Test prompt".to_string(), true);
        store.put_task(&task).unwrap();

        // Add results
        let result1 = ModelResult::new(
            task.id.clone(),
            ModelId::StrandCoder,
            "Response 1".to_string(),
            100,
            200,
        )
        .with_confidence(0.8);

        let result2 = ModelResult::new(
            task.id.clone(),
            ModelId::Behemoth,
            "Response 2".to_string(),
            150,
            500,
        )
        .with_confidence(0.8);

        store.put_result(&result1).unwrap();
        store.put_result(&result2).unwrap();

        // Request arbitration
        let request = manager
            .request_arbitration(
                &task.id,
                ArbitrationReason::TieVote {
                    tied_models: vec![ModelId::StrandCoder, ModelId::Behemoth],
                },
            )
            .unwrap();

        assert_eq!(request.task_id, task.id);
        assert_eq!(request.responses.len(), 2);
    }

    #[test]
    fn test_apply_decision() {
        let (manager, store, _dir) = test_setup();

        // Create session and task
        let session = EnsembleSession::new();
        store.put_session(&session).unwrap();

        let mut task = EnsembleTask::new(session.id.clone(), "Test prompt".to_string(), true);
        task.status = TaskStatus::AwaitingArbitration;
        store.put_task(&task).unwrap();

        // Add result
        let result = ModelResult::new(
            task.id.clone(),
            ModelId::Behemoth,
            "Final response".to_string(),
            100,
            500,
        );
        store.put_result(&result).unwrap();

        // Apply decision
        let decision = ArbitrationDecision {
            winner: ModelId::Behemoth,
            rationale: "Best technical approach".to_string(),
            modified_response: None,
            notes: None,
        };

        manager.apply_decision(&task.id, decision).unwrap();

        // Verify
        let updated_task = store.get_task(&task.id).unwrap().unwrap();
        assert_eq!(updated_task.status, TaskStatus::Completed);
        assert_eq!(updated_task.winning_model, Some(ModelId::Behemoth));
    }

    #[test]
    fn test_generate_arbitration_prompt() {
        let (manager, _, _dir) = test_setup();

        let request = ArbitrationRequest {
            task_id: "task-1".to_string(),
            reason: ArbitrationReason::TieVote {
                tied_models: vec![ModelId::StrandCoder, ModelId::HydraCoder],
            },
            responses: vec![
                ModelResponseSummary {
                    model_id: ModelId::StrandCoder,
                    response: "Response A".to_string(),
                    reasoning: Some("Reasoning A".to_string()),
                    confidence: 0.75,
                    latency_ms: 200,
                },
                ModelResponseSummary {
                    model_id: ModelId::HydraCoder,
                    response: "Response B".to_string(),
                    reasoning: None,
                    confidence: 0.75,
                    latency_ms: 300,
                },
            ],
            original_prompt: "Implement error handling".to_string(),
            context: Some("Working on ensemble module".to_string()),
        };

        let prompt = manager.generate_arbitration_prompt(&request);

        assert!(prompt.contains("Arbitration Required"));
        assert!(prompt.contains("tie"));
        assert!(prompt.contains("Response A"));
        assert!(prompt.contains("Response B"));
        assert!(prompt.contains("Previous Context"));
    }
}
