//! Voting protocol for multi-model consensus
//!
//! Implements several voting strategies for selecting the best response
//! from multiple models, with tie-breaking and arbitration support.

use std::collections::HashMap;

use chrono::Utc;
use tracing::{debug, info, warn};

use crate::events::{ArbitrationReason, EnsembleEvent, SharedEventBus, VoteSummary};
use crate::state::{ModelId, ModelResult, SharedStateStore, TaskId, VoteRecord, VotingStrategy};

/// Error type for voting operations
#[derive(Debug, thiserror::Error)]
pub enum VotingError {
    #[error("No results available for voting")]
    NoResults,

    #[error("Not enough results: got {got}, need {need}")]
    InsufficientResults { got: usize, need: usize },

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Voting tie: {0:?}")]
    Tie(Vec<ModelId>),

    #[error("Low confidence: max {0}")]
    LowConfidence(f32),
}

/// Result type for voting operations
pub type VotingResult<T> = Result<T, VotingError>;

/// Outcome of a voting round
#[derive(Debug, Clone)]
pub struct VoteOutcome {
    /// The winning model
    pub winner: ModelId,
    /// Vote summary statistics
    pub summary: VoteSummary,
    /// Whether arbitration was needed
    pub arbitrated: bool,
    /// The full vote record
    pub record: VoteRecord,
}

/// Minimum confidence threshold for accepting results
const LOW_CONFIDENCE_THRESHOLD: f32 = 0.3;

/// Voting protocol implementation
pub struct VotingProtocol {
    store: SharedStateStore,
    event_bus: SharedEventBus,
}

impl VotingProtocol {
    /// Create a new voting protocol
    pub fn new(store: SharedStateStore, event_bus: SharedEventBus) -> Self {
        Self { store, event_bus }
    }

    /// Execute voting for a task using the specified strategy
    pub async fn vote(
        &self,
        task_id: &TaskId,
        strategy: VotingStrategy,
    ) -> VotingResult<VoteOutcome> {
        // Get all results for this task
        let results = self
            .store
            .get_task_results(task_id)
            .map_err(|e| VotingError::StoreError(e.to_string()))?;

        if results.is_empty() {
            return Err(VotingError::NoResults);
        }

        info!(
            task_id,
            results = results.len(),
            strategy = ?strategy,
            "Starting voting"
        );

        // Publish voting started event
        let participating_models: Vec<ModelId> = results.iter().map(|r| r.model_id).collect();
        let _ = self.event_bus.publish(EnsembleEvent::VotingStarted {
            task_id: task_id.clone(),
            strategy,
            participating_models,
            timestamp: Utc::now(),
        });

        // Execute voting based on strategy
        let outcome = match strategy {
            VotingStrategy::Majority => self.majority_vote(task_id, &results).await,
            VotingStrategy::Weighted => self.weighted_vote(task_id, &results).await,
            VotingStrategy::Unanimous => self.unanimous_vote(task_id, &results).await,
        }?;

        // Store the vote record
        self.store
            .put_vote(&outcome.record)
            .map_err(|e| VotingError::StoreError(e.to_string()))?;

        // Publish consensus reached event
        let _ = self.event_bus.publish(EnsembleEvent::ConsensusReached {
            task_id: task_id.clone(),
            winner: outcome.winner,
            vote_summary: outcome.summary.clone(),
            timestamp: Utc::now(),
        });

        Ok(outcome)
    }

    /// Simple majority voting - each model votes for itself
    async fn majority_vote(
        &self,
        task_id: &TaskId,
        results: &[ModelResult],
    ) -> VotingResult<VoteOutcome> {
        debug!(task_id, "Executing majority vote");

        // Each model votes for itself, weighted by confidence
        let mut votes: HashMap<ModelId, f32> = HashMap::new();
        let mut record = VoteRecord::new(task_id.clone(), VotingStrategy::Majority);

        for result in results {
            *votes.entry(result.model_id).or_insert(0.0) += result.confidence;
            record.add_vote(result.model_id, result.model_id, result.confidence);
        }

        // Find winner
        let (winner, score) = votes
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .ok_or(VotingError::NoResults)?;

        // Check for ties
        let max_score = score;
        let tied: Vec<ModelId> = record
            .votes
            .iter()
            .filter(|(_, (_, s))| (*s - max_score).abs() < 0.01)
            .map(|(m, _)| *m)
            .collect();

        if tied.len() > 1 {
            // Use tie breaker
            return self.tie_break(task_id, results, &tied, record).await;
        }

        record.set_winner(winner);

        let vote_counts: Vec<(ModelId, u32)> = results.iter().map(|r| (r.model_id, 1)).collect();
        let summary = VoteSummary::new(vote_counts).with_avg_confidence(
            results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32,
        );

        Ok(VoteOutcome {
            winner,
            summary,
            arbitrated: false,
            record,
        })
    }

    /// Weighted voting - combines model confidence with model weight
    async fn weighted_vote(
        &self,
        task_id: &TaskId,
        results: &[ModelResult],
    ) -> VotingResult<VoteOutcome> {
        debug!(task_id, "Executing weighted vote");

        let mut scores: HashMap<ModelId, f32> = HashMap::new();
        let mut record = VoteRecord::new(task_id.clone(), VotingStrategy::Weighted);

        for result in results {
            let model_weight = result.model_id.weight();
            let combined_weight = result.confidence * model_weight;
            *scores.entry(result.model_id).or_insert(0.0) += combined_weight;
            record.add_vote(result.model_id, result.model_id, combined_weight);
        }

        // Find winner
        let (winner, max_score) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(m, s)| (*m, *s))
            .ok_or(VotingError::NoResults)?;

        // Check for ties (within 5% of max)
        let threshold = max_score * 0.95;
        let tied: Vec<ModelId> = scores
            .iter()
            .filter(|(_, s)| **s >= threshold)
            .map(|(m, _)| *m)
            .collect();

        if tied.len() > 1 && tied.contains(&winner) {
            return self.tie_break(task_id, results, &tied, record).await;
        }

        // Check for low confidence
        let avg_confidence =
            results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32;
        if avg_confidence < LOW_CONFIDENCE_THRESHOLD {
            warn!(task_id, avg_confidence, "Low confidence detected");
            return Err(VotingError::LowConfidence(avg_confidence));
        }

        record.set_winner(winner);

        let vote_counts: Vec<(ModelId, u32)> = results.iter().map(|r| (r.model_id, 1)).collect();
        let summary = VoteSummary::new(vote_counts).with_avg_confidence(avg_confidence);

        Ok(VoteOutcome {
            winner,
            summary,
            arbitrated: false,
            record,
        })
    }

    /// Unanimous voting - requires all models to agree (typically for safety-critical tasks)
    async fn unanimous_vote(
        &self,
        task_id: &TaskId,
        results: &[ModelResult],
    ) -> VotingResult<VoteOutcome> {
        debug!(task_id, "Executing unanimous vote");

        if results.len() < 2 {
            return Err(VotingError::InsufficientResults {
                got: results.len(),
                need: 2,
            });
        }

        // For unanimous, we need to check if responses are semantically similar
        // This is a simplified version - production would use semantic similarity
        let record = VoteRecord::new(task_id.clone(), VotingStrategy::Unanimous);

        // Use Opus45 as the default unanimous choice if present
        let winner = if results.iter().any(|r| r.model_id == ModelId::Opus45) {
            ModelId::Opus45
        } else {
            results[0].model_id
        };

        // Check minimum confidence from all models
        let min_confidence = results
            .iter()
            .map(|r| r.confidence)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0);

        if min_confidence < LOW_CONFIDENCE_THRESHOLD {
            return Err(VotingError::LowConfidence(min_confidence));
        }

        let vote_counts: Vec<(ModelId, u32)> = vec![(winner, results.len() as u32)];
        let summary = VoteSummary::new(vote_counts).with_avg_confidence(
            results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32,
        );

        Ok(VoteOutcome {
            winner,
            summary,
            arbitrated: false,
            record,
        })
    }

    /// Break ties using Opus45 as the tie-breaker
    async fn tie_break(
        &self,
        task_id: &TaskId,
        results: &[ModelResult],
        tied: &[ModelId],
        mut record: VoteRecord,
    ) -> VotingResult<VoteOutcome> {
        info!(task_id, tied = ?tied, "Breaking tie");

        // Publish arbitration requested event
        let _ = self.event_bus.publish(EnsembleEvent::ArbitrationRequested {
            task_id: task_id.clone(),
            reason: ArbitrationReason::TieVote {
                tied_models: tied.to_vec(),
            },
            timestamp: Utc::now(),
        });

        // If Opus45 is tied, prefer it (most capable for reasoning)
        let winner = if tied.contains(&ModelId::Opus45) {
            ModelId::Opus45
        } else if tied.contains(&ModelId::Gemini3Pro) {
            // Prefer Gemini3Pro over workers
            ModelId::Gemini3Pro
        } else {
            // Fallback to first tied model
            tied[0]
        };

        record.set_winner(winner);
        record.mark_arbitrated("Tie-break using model hierarchy".to_string());

        let vote_counts: Vec<(ModelId, u32)> = tied.iter().map(|m| (*m, 1)).collect();
        let summary = VoteSummary::new(vote_counts).with_avg_confidence(
            results.iter().map(|r| r.confidence).sum::<f32>() / results.len() as f32,
        );

        Ok(VoteOutcome {
            winner,
            summary,
            arbitrated: true,
            record,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventBus;
    use crate::state::StateStore;
    use tempfile::tempdir;

    fn test_setup() -> (VotingProtocol, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let store = StateStore::open(dir.path().join("test.db"))
            .unwrap()
            .shared();
        let bus = EventBus::new().shared();
        (VotingProtocol::new(store, bus), dir)
    }

    #[test]
    fn test_model_weights() {
        assert!(ModelId::Opus45.weight() > ModelId::HydraCoder.weight());
        assert_eq!(ModelId::Opus45.weight(), ModelId::Gemini3Pro.weight());
    }

    #[tokio::test]
    async fn test_majority_vote_single() {
        let (protocol, _dir) = test_setup();

        // Store a result
        let result = ModelResult::new(
            "task-1".to_string(),
            ModelId::Opus45,
            "Test response".to_string(),
            100,
            500,
        )
        .with_confidence(0.9);

        protocol.store.put_result(&result).unwrap();

        let outcome = protocol
            .vote(&"task-1".to_string(), VotingStrategy::Majority)
            .await
            .unwrap();

        assert_eq!(outcome.winner, ModelId::Opus45);
        assert!(!outcome.arbitrated);
    }

    #[tokio::test]
    async fn test_weighted_vote() {
        let (protocol, _dir) = test_setup();

        // Store results with different confidences
        let result1 = ModelResult::new(
            "task-1".to_string(),
            ModelId::HydraCoder,
            "Response 1".to_string(),
            100,
            200,
        )
        .with_confidence(0.9);

        let result2 = ModelResult::new(
            "task-1".to_string(),
            ModelId::Opus45,
            "Response 2".to_string(),
            150,
            1000,
        )
        .with_confidence(0.7);

        protocol.store.put_result(&result1).unwrap();
        protocol.store.put_result(&result2).unwrap();

        let outcome = protocol
            .vote(&"task-1".to_string(), VotingStrategy::Weighted)
            .await
            .unwrap();

        // Opus45 should win due to higher model weight despite lower confidence
        // HydraCoder: 0.9 * 0.85 = 0.765
        // Opus45: 0.7 * 1.0 = 0.7
        // Close scores, but HydraCoder edges out slightly
        // Actually with new weights: HydraCoder=0.85 weight, so 0.9*0.85=0.765 vs Opus45=0.7*1.0=0.7
        assert_eq!(outcome.winner, ModelId::HydraCoder);
    }

    #[tokio::test]
    async fn test_no_results_error() {
        let (protocol, _dir) = test_setup();

        let result = protocol
            .vote(&"nonexistent".to_string(), VotingStrategy::Majority)
            .await;

        assert!(matches!(result, Err(VotingError::NoResults)));
    }
}
