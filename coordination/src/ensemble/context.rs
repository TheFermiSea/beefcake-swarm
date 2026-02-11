//! Context manager for sharing state across model swaps
//!
//! Maintains coherent context that survives model loading/unloading,
//! enabling effective multi-model collaboration.

use chrono::Utc;
use tracing::{debug, info};

use crate::events::{ContextUpdater, EnsembleEvent, SharedEventBus};
use crate::state::{ModelId, SessionId, SharedContext, SharedStateStore};

/// Error type for context operations
#[derive(Debug, thiserror::Error)]
pub enum ContextError {
    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Context version mismatch: expected {expected}, got {actual}")]
    VersionMismatch { expected: u64, actual: u64 },
}

/// Result type for context operations
pub type ContextResult<T> = Result<T, ContextError>;

/// Context manager for cross-model state sharing
pub struct ContextManager {
    store: SharedStateStore,
    event_bus: SharedEventBus,
}

impl ContextManager {
    /// Create a new context manager
    pub fn new(store: SharedStateStore, event_bus: SharedEventBus) -> Self {
        Self { store, event_bus }
    }

    /// Get or create context for a session
    pub fn get_or_create(&self, session_id: &SessionId) -> ContextResult<SharedContext> {
        self.store
            .get_or_create_context(session_id)
            .map_err(|e| ContextError::StoreError(e.to_string()))
    }

    /// Get context for a session
    pub fn get(&self, session_id: &SessionId) -> ContextResult<Option<SharedContext>> {
        self.store
            .get_context(session_id)
            .map_err(|e| ContextError::StoreError(e.to_string()))
    }

    /// Update context summary
    pub fn update_summary(
        &self,
        session_id: &SessionId,
        summary: String,
        updater: ContextUpdater,
    ) -> ContextResult<SharedContext> {
        let mut ctx = self.get_or_create(session_id)?;
        ctx.update_summary(summary.clone());
        self.store
            .put_context(&ctx)
            .map_err(|e| ContextError::StoreError(e.to_string()))?;

        // Publish update event
        let preview = if summary.len() > 100 {
            format!("{}...", &summary[..100])
        } else {
            summary
        };

        let _ = self.event_bus.publish(EnsembleEvent::ContextUpdated {
            session_id: session_id.clone(),
            version: ctx.version,
            updated_by: updater,
            summary_preview: preview,
            timestamp: Utc::now(),
        });

        info!(session_id, version = ctx.version, "Context summary updated");
        Ok(ctx)
    }

    /// Add a key decision to context
    pub fn add_decision(
        &self,
        session_id: &SessionId,
        decision: String,
        updater: ContextUpdater,
    ) -> ContextResult<SharedContext> {
        let mut ctx = self.get_or_create(session_id)?;
        ctx.add_decision(decision.clone());
        self.store
            .put_context(&ctx)
            .map_err(|e| ContextError::StoreError(e.to_string()))?;

        let _ = self.event_bus.publish(EnsembleEvent::ContextUpdated {
            session_id: session_id.clone(),
            version: ctx.version,
            updated_by: updater,
            summary_preview: format!("Decision: {}", decision),
            timestamp: Utc::now(),
        });

        debug!(session_id, decision, "Added decision to context");
        Ok(ctx)
    }

    /// Add a file reference to context
    pub fn add_file_reference(
        &self,
        session_id: &SessionId,
        file: String,
    ) -> ContextResult<SharedContext> {
        let mut ctx = self.get_or_create(session_id)?;
        ctx.add_file_reference(file.clone());
        self.store
            .put_context(&ctx)
            .map_err(|e| ContextError::StoreError(e.to_string()))?;

        debug!(session_id, file, "Added file reference to context");
        Ok(ctx)
    }

    /// Set domain-specific context
    pub fn set_domain(
        &self,
        session_id: &SessionId,
        key: String,
        value: String,
    ) -> ContextResult<SharedContext> {
        let mut ctx = self.get_or_create(session_id)?;
        ctx.set_domain(key.clone(), value.clone());
        self.store
            .put_context(&ctx)
            .map_err(|e| ContextError::StoreError(e.to_string()))?;

        debug!(session_id, key, "Set domain context");
        Ok(ctx)
    }

    /// Generate a context prompt for model injection
    ///
    /// This creates a formatted prompt section that can be prepended to
    /// model requests to provide context from previous model executions.
    pub fn generate_context_prompt(&self, session_id: &SessionId) -> ContextResult<String> {
        let ctx = match self.get(session_id)? {
            Some(c) => c,
            None => return Ok(String::new()),
        };

        let mut prompt = String::new();

        if !ctx.summary.is_empty() {
            prompt.push_str("## Previous Context\n\n");
            prompt.push_str(&ctx.summary);
            prompt.push_str("\n\n");
        }

        if !ctx.key_decisions.is_empty() {
            prompt.push_str("## Key Decisions Made\n\n");
            for decision in &ctx.key_decisions {
                prompt.push_str(&format!("- {}\n", decision));
            }
            prompt.push('\n');
        }

        if !ctx.file_references.is_empty() {
            prompt.push_str("## Relevant Files\n\n");
            for file in &ctx.file_references {
                prompt.push_str(&format!("- `{}`\n", file));
            }
            prompt.push('\n');
        }

        if !ctx.domain_context.is_empty() {
            prompt.push_str("## Domain Context\n\n");
            for (key, value) in &ctx.domain_context {
                prompt.push_str(&format!("**{}**: {}\n", key, value));
            }
            prompt.push('\n');
        }

        Ok(prompt)
    }

    /// Merge context from a model's execution
    ///
    /// This extracts relevant information from a model's response and
    /// updates the shared context accordingly.
    pub fn merge_from_response(
        &self,
        session_id: &SessionId,
        model_id: ModelId,
        response: &str,
    ) -> ContextResult<SharedContext> {
        // Simple heuristic extraction - production would use more sophisticated parsing
        let mut ctx = self.get_or_create(session_id)?;
        let mut updated = false;

        // Extract file references (simple pattern matching)
        for word in response.split_whitespace() {
            if (word.ends_with(".rs") || word.ends_with(".toml") || word.ends_with(".md"))
                && !ctx.file_references.contains(&word.to_string())
            {
                ctx.add_file_reference(word.to_string());
                updated = true;
            }
        }

        // Look for decision indicators
        let decision_patterns = [
            "decided to",
            "chose to",
            "will use",
            "should use",
            "recommending",
        ];
        for pattern in &decision_patterns {
            if let Some(pos) = response.to_lowercase().find(pattern) {
                // Extract the sentence containing the decision
                let start = response[..pos].rfind('.').map(|p| p + 1).unwrap_or(0);
                let end = response[pos..]
                    .find('.')
                    .map(|p| pos + p + 1)
                    .unwrap_or(response.len());
                let decision = response[start..end].trim().to_string();
                if !decision.is_empty() && !ctx.key_decisions.contains(&decision) {
                    ctx.add_decision(decision);
                    updated = true;
                    break; // Only extract one decision per response
                }
            }
        }

        if updated {
            self.store
                .put_context(&ctx)
                .map_err(|e| ContextError::StoreError(e.to_string()))?;

            let _ = self.event_bus.publish(EnsembleEvent::ContextUpdated {
                session_id: session_id.clone(),
                version: ctx.version,
                updated_by: ContextUpdater::Model(model_id),
                summary_preview: "Merged from response".to_string(),
                timestamp: Utc::now(),
            });

            debug!(
                session_id,
                model = %model_id,
                "Merged context from response"
            );
        }

        Ok(ctx)
    }

    /// Get context version for optimistic concurrency
    pub fn get_version(&self, session_id: &SessionId) -> ContextResult<u64> {
        match self.get(session_id)? {
            Some(ctx) => Ok(ctx.version),
            None => Ok(0),
        }
    }

    /// Update context with version check (optimistic concurrency)
    pub fn update_with_version(
        &self,
        session_id: &SessionId,
        expected_version: u64,
        summary: String,
        updater: ContextUpdater,
    ) -> ContextResult<SharedContext> {
        let ctx = self.get_or_create(session_id)?;

        if ctx.version != expected_version {
            return Err(ContextError::VersionMismatch {
                expected: expected_version,
                actual: ctx.version,
            });
        }

        self.update_summary(session_id, summary, updater)
    }
}

/// Context snapshot for serialization
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ContextSnapshot {
    pub session_id: SessionId,
    pub version: u64,
    pub summary: String,
    pub key_decisions: Vec<String>,
    pub file_references: Vec<String>,
    pub domain_context: std::collections::HashMap<String, String>,
}

impl From<SharedContext> for ContextSnapshot {
    fn from(ctx: SharedContext) -> Self {
        Self {
            session_id: ctx.session_id,
            version: ctx.version,
            summary: ctx.summary,
            key_decisions: ctx.key_decisions,
            file_references: ctx.file_references,
            domain_context: ctx.domain_context,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventBus;
    use crate::state::StateStore;
    use tempfile::tempdir;

    fn test_setup() -> (ContextManager, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let store = StateStore::open(dir.path().join("test.db"))
            .unwrap()
            .shared();
        let bus = EventBus::new().shared();
        (ContextManager::new(store, bus), dir)
    }

    #[test]
    fn test_get_or_create() {
        let (manager, _dir) = test_setup();

        let ctx1 = manager.get_or_create(&"session-1".to_string()).unwrap();
        assert_eq!(ctx1.version, 0);

        let ctx2 = manager.get_or_create(&"session-1".to_string()).unwrap();
        assert_eq!(ctx2.session_id, ctx1.session_id);
    }

    #[test]
    fn test_update_summary() {
        let (manager, _dir) = test_setup();

        let ctx = manager
            .update_summary(
                &"session-1".to_string(),
                "Test summary".to_string(),
                ContextUpdater::System,
            )
            .unwrap();

        assert_eq!(ctx.summary, "Test summary");
        assert_eq!(ctx.version, 1);
    }

    #[test]
    fn test_add_decision() {
        let (manager, _dir) = test_setup();

        let ctx = manager
            .add_decision(
                &"session-1".to_string(),
                "Use RocksDB for persistence".to_string(),
                ContextUpdater::Overseer,
            )
            .unwrap();

        assert!(ctx
            .key_decisions
            .contains(&"Use RocksDB for persistence".to_string()));
    }

    #[test]
    fn test_generate_context_prompt() {
        let (manager, _dir) = test_setup();
        let session_id = "session-1".to_string();

        // Add some context
        manager
            .update_summary(
                &session_id,
                "Working on ensemble coordination".to_string(),
                ContextUpdater::System,
            )
            .unwrap();
        manager
            .add_decision(
                &session_id,
                "Use weighted voting".to_string(),
                ContextUpdater::Overseer,
            )
            .unwrap();
        manager
            .add_file_reference(&session_id, "src/ensemble/voting.rs".to_string())
            .unwrap();

        let prompt = manager.generate_context_prompt(&session_id).unwrap();

        assert!(prompt.contains("Previous Context"));
        assert!(prompt.contains("Working on ensemble coordination"));
        assert!(prompt.contains("Key Decisions Made"));
        assert!(prompt.contains("Use weighted voting"));
        assert!(prompt.contains("Relevant Files"));
        assert!(prompt.contains("src/ensemble/voting.rs"));
    }

    #[test]
    fn test_merge_from_response() {
        let (manager, _dir) = test_setup();
        let session_id = "session-1".to_string();

        let response =
            "I've analyzed the code in src/lib.rs and decided to use the existing pattern. \
                       We should use async/await for the implementation.";

        let ctx = manager
            .merge_from_response(&session_id, ModelId::Behemoth, response)
            .unwrap();

        assert!(ctx.file_references.contains(&"src/lib.rs".to_string()));
        assert!(!ctx.key_decisions.is_empty());
    }

    #[test]
    fn test_version_check() {
        let (manager, _dir) = test_setup();
        let session_id = "session-1".to_string();

        // Create initial context
        manager
            .update_summary(&session_id, "Initial".to_string(), ContextUpdater::System)
            .unwrap();

        // Try to update with wrong version
        let result = manager.update_with_version(
            &session_id,
            0, // Wrong version (should be 1)
            "Updated".to_string(),
            ContextUpdater::System,
        );

        assert!(matches!(
            result,
            Err(ContextError::VersionMismatch {
                expected: 0,
                actual: 1
            })
        ));
    }
}
