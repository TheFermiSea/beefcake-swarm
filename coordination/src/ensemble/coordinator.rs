//! Ensemble coordinator - central orchestrator for multi-model coordination
//!
//! The coordinator manages the lifecycle of ensemble sessions and tasks,
//! orchestrating model execution, voting, and context management.

use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::events::{EnsembleEvent, SessionEndReason, SharedEventBus, UnloadReason};
use crate::harness::SharedHarnessState;
use crate::state::{
    EnsembleSession, EnsembleTask, ModelId, ModelResult, SharedStateStore, TaskId, TaskStatus,
    VotingStrategy,
};

use super::arbitration::{ArbitrationDecision, ArbitrationManager, ArbitrationRequest};
use super::context::ContextManager;
use super::voting::{VoteOutcome, VotingProtocol};

/// Error type for coordinator operations
#[derive(Debug, thiserror::Error)]
pub enum CoordinatorError {
    #[error("No active session")]
    NoActiveSession,

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("Task not found: {0}")]
    TaskNotFound(String),

    #[error("Model execution failed: {0}")]
    ModelExecutionFailed(String),

    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Voting error: {0}")]
    VotingError(String),

    #[error("HTTP error: {0}")]
    HttpError(String),
}

/// Result type for coordinator operations
pub type CoordinatorResult<T> = Result<T, CoordinatorError>;

/// Shared reference to EnsembleCoordinator
pub type SharedEnsembleCoordinator = Arc<EnsembleCoordinator>;

/// Configuration for the ensemble coordinator
#[derive(Debug, Clone)]
pub struct EnsembleConfig {
    /// LLM router URL
    pub router_url: String,
    /// Default max tokens for responses
    pub default_max_tokens: u32,
    /// Default voting strategy
    pub default_voting_strategy: VotingStrategy,
    /// Temperature for model queries
    pub temperature: f32,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            router_url: std::env::var("ROUTER_URL")
                .unwrap_or_else(|_| "http://10.0.0.31:8000/v1/chat/completions".to_string()),
            default_max_tokens: 2048,
            default_voting_strategy: VotingStrategy::Weighted,
            temperature: 0.3,
        }
    }
}

/// Central orchestrator for multi-model ensemble coordination
pub struct EnsembleCoordinator {
    store: SharedStateStore,
    event_bus: SharedEventBus,
    harness: Option<SharedHarnessState>,
    http: reqwest::Client,
    config: EnsembleConfig,
    voting: VotingProtocol,
    context: ContextManager,
    arbitration: ArbitrationManager,
    current_model: RwLock<Option<ModelId>>,
}

impl EnsembleCoordinator {
    /// Create a new ensemble coordinator
    pub fn new(
        store: SharedStateStore,
        event_bus: SharedEventBus,
        config: EnsembleConfig,
    ) -> CoordinatorResult<Self> {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .map_err(|e| CoordinatorError::HttpError(e.to_string()))?;

        let voting = VotingProtocol::new(store.clone(), event_bus.clone());
        let context = ContextManager::new(store.clone(), event_bus.clone());
        let arbitration = ArbitrationManager::new(store.clone(), event_bus.clone());

        Ok(Self {
            store,
            event_bus,
            harness: None,
            http,
            config,
            voting,
            context,
            arbitration,
            current_model: RwLock::new(None),
        })
    }

    /// Create a shared reference to this coordinator
    pub fn shared(self) -> SharedEnsembleCoordinator {
        Arc::new(self)
    }

    /// Set the harness state for integration
    pub fn with_harness(mut self, harness: SharedHarnessState) -> Self {
        self.harness = Some(harness);
        self
    }

    // =========================================================================
    // Session Management
    // =========================================================================

    /// Start a new ensemble session
    pub async fn start_session(
        &self,
        harness_session_id: Option<String>,
    ) -> CoordinatorResult<EnsembleSession> {
        let mut session = EnsembleSession::new();

        if let Some(ref harness_id) = harness_session_id {
            session = session.with_harness(harness_id.clone());
        }

        self.store
            .put_session(&session)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

        // Initialize context for this session
        self.context
            .get_or_create(&session.id)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

        // Publish event
        let _ = self.event_bus.publish(EnsembleEvent::SessionCreated {
            session_id: session.id.clone(),
            harness_session_id,
            timestamp: Utc::now(),
        });

        info!(session_id = %session.id, "Ensemble session started");

        Ok(session)
    }

    /// Get the current active session
    pub fn get_active_session(&self) -> CoordinatorResult<EnsembleSession> {
        self.store
            .get_active_session()
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?
            .ok_or(CoordinatorError::NoActiveSession)
    }

    /// Get a session by ID
    pub fn get_session(&self, session_id: &str) -> CoordinatorResult<EnsembleSession> {
        self.store
            .get_session(session_id)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?
            .ok_or_else(|| CoordinatorError::SessionNotFound(session_id.to_string()))
    }

    /// End the current session
    pub async fn end_session(
        &self,
        session_id: &str,
        reason: SessionEndReason,
    ) -> CoordinatorResult<()> {
        let mut session = self.get_session(session_id)?;
        session.active = false;

        self.store
            .put_session(&session)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

        // Unload current model
        self.unload_current_model(UnloadReason::SessionEnd).await;

        // Publish event
        let _ = self.event_bus.publish(EnsembleEvent::SessionEnded {
            session_id: session_id.to_string(),
            reason,
            tasks_completed: session.completed_tasks.len() as u32,
            timestamp: Utc::now(),
        });

        info!(session_id, "Ensemble session ended");
        Ok(())
    }

    // =========================================================================
    // Task Management
    // =========================================================================

    /// Submit a task for ensemble processing
    pub async fn submit_task(
        &self,
        prompt: String,
        code_context: Option<String>,
        require_consensus: bool,
    ) -> CoordinatorResult<EnsembleTask> {
        let session = self.get_active_session()?;

        let mut task = EnsembleTask::new(session.id.clone(), prompt.clone(), require_consensus);

        if let Some(code) = code_context {
            task = task.with_code(code);
        }

        task = task.with_max_tokens(self.config.default_max_tokens);

        self.store
            .put_task(&task)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

        // Update session with pending task
        let mut session = session;
        session.add_task(task.id.clone());
        self.store
            .put_session(&session)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

        // Publish event
        let prompt_preview = if prompt.len() > 100 {
            format!("{}...", &prompt[..100])
        } else {
            prompt
        };

        let _ = self.event_bus.publish(EnsembleEvent::TaskCreated {
            task_id: task.id.clone(),
            session_id: session.id.clone(),
            prompt_preview,
            require_consensus,
            timestamp: Utc::now(),
        });

        info!(task_id = %task.id, require_consensus, "Task submitted");

        Ok(task)
    }

    /// Execute a task by running it through assigned models
    pub async fn execute_task(&self, task_id: &TaskId) -> CoordinatorResult<EnsembleTask> {
        let mut task = self
            .store
            .get_task(task_id)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?
            .ok_or_else(|| CoordinatorError::TaskNotFound(task_id.clone()))?;

        if task.status != TaskStatus::Pending {
            warn!(task_id, status = ?task.status, "Task already in progress");
        }

        task.status = TaskStatus::InProgress;
        self.store
            .put_task(&task)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

        // Get context prompt
        let context_prompt = self
            .context
            .generate_context_prompt(&task.session_id)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

        // Execute each assigned model sequentially
        for model_id in task.assigned_models.clone() {
            // Publish assignment event
            let _ = self.event_bus.publish(EnsembleEvent::TaskAssigned {
                task_id: task_id.clone(),
                model_id,
                timestamp: Utc::now(),
            });

            // Execute model
            match self.execute_model(&task, model_id, &context_prompt).await {
                Ok(result) => {
                    // Store result
                    self.store
                        .put_result(&result)
                        .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

                    // Mark model complete
                    task.mark_model_complete(model_id);
                    self.store
                        .put_task(&task)
                        .map_err(|e| CoordinatorError::StoreError(e.to_string()))?;

                    // Merge context from response
                    let _ = self.context.merge_from_response(
                        &task.session_id,
                        model_id,
                        &result.response,
                    );

                    // Publish result event
                    let _ = self.event_bus.publish(EnsembleEvent::ResultSubmitted {
                        task_id: task_id.clone(),
                        model_id,
                        confidence: result.confidence,
                        tokens_used: result.tokens_used,
                        latency_ms: result.latency_ms,
                        timestamp: Utc::now(),
                    });
                }
                Err(e) => {
                    error!(task_id, model = %model_id, "Model execution failed: {}", e);

                    let _ = self.event_bus.publish(EnsembleEvent::TaskFailed {
                        task_id: task_id.clone(),
                        model_id: Some(model_id),
                        error: e.to_string(),
                        timestamp: Utc::now(),
                    });

                    // Continue with other models
                }
            }
        }

        // Refresh task
        let task = self
            .store
            .get_task(task_id)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))?
            .ok_or_else(|| CoordinatorError::TaskNotFound(task_id.clone()))?;

        Ok(task)
    }

    /// Execute a single model for a task
    async fn execute_model(
        &self,
        task: &EnsembleTask,
        model_id: ModelId,
        context_prompt: &str,
    ) -> CoordinatorResult<ModelResult> {
        // Load the model
        let load_start = Instant::now();
        self.load_model(model_id).await?;
        let load_time_ms = load_start.elapsed().as_millis() as u64;

        let _ = self.event_bus.publish(EnsembleEvent::ModelLoaded {
            model_id,
            load_time_ms,
            timestamp: Utc::now(),
        });

        // Build the prompt
        let system_prompt = self.get_system_prompt(model_id);
        let user_content = self.build_user_prompt(task, context_prompt);

        // Execute query
        let exec_start = Instant::now();
        let (response, reasoning) = self
            .query_model(model_id, &system_prompt, &user_content, task.max_tokens)
            .await?;
        let latency_ms = exec_start.elapsed().as_millis() as u64;

        // Estimate token count (rough approximation)
        let tokens_used = (response.len() / 4) as u32;

        // Extract confidence (simplified - production would parse from response)
        let confidence = self.extract_confidence(&response, model_id);

        let mut result =
            ModelResult::new(task.id.clone(), model_id, response, tokens_used, latency_ms)
                .with_confidence(confidence);

        if let Some(r) = reasoning {
            result = result.with_reasoning(r);
        }

        debug!(
            task_id = %task.id,
            model = %model_id,
            tokens = tokens_used,
            latency_ms,
            confidence,
            "Model execution complete"
        );

        Ok(result)
    }

    /// Load a model (switch if different from current)
    async fn load_model(&self, model_id: ModelId) -> CoordinatorResult<()> {
        let mut current = self.current_model.write().await;

        if *current == Some(model_id) {
            debug!(model = %model_id, "Model already loaded");
            return Ok(());
        }

        // Unload current model if different
        if current.is_some() {
            let old_model = current.take().unwrap();
            let _ = self.event_bus.publish(EnsembleEvent::ModelUnloaded {
                model_id: old_model,
                reason: UnloadReason::ModelSwap,
                timestamp: Utc::now(),
            });
        }

        *current = Some(model_id);
        info!(model = %model_id, "Model loaded");
        Ok(())
    }

    /// Unload the current model
    async fn unload_current_model(&self, reason: UnloadReason) {
        let mut current = self.current_model.write().await;
        if let Some(model_id) = current.take() {
            let _ = self.event_bus.publish(EnsembleEvent::ModelUnloaded {
                model_id,
                reason,
                timestamp: Utc::now(),
            });
            info!(model = %model_id, reason = %reason, "Model unloaded");
        }
    }

    /// Query a model via the LLM router
    async fn query_model(
        &self,
        model_id: ModelId,
        system_prompt: &str,
        user_content: &str,
        max_tokens: u32,
    ) -> CoordinatorResult<(String, Option<String>)> {
        #[derive(Serialize)]
        struct ChatMessage {
            role: String,
            content: String,
        }

        #[derive(Serialize)]
        struct ChatRequest {
            model: String,
            messages: Vec<ChatMessage>,
            max_tokens: u32,
            temperature: f32,
        }

        #[derive(Deserialize)]
        struct ChatResponse {
            choices: Vec<Choice>,
        }

        #[derive(Deserialize)]
        struct Choice {
            message: ResponseMessage,
        }

        #[derive(Deserialize)]
        struct ResponseMessage {
            content: Option<String>,
            reasoning_content: Option<String>,
        }

        let request = ChatRequest {
            model: model_id.api_name().to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: system_prompt.to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: user_content.to_string(),
                },
            ],
            max_tokens,
            temperature: self.config.temperature,
        };

        let response = self
            .http
            .post(&self.config.router_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| CoordinatorError::HttpError(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CoordinatorError::ModelExecutionFailed(format!(
                "HTTP {}: {}",
                status, body
            )));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| CoordinatorError::ModelExecutionFailed(e.to_string()))?;

        let choice = chat_response
            .choices
            .first()
            .ok_or_else(|| CoordinatorError::ModelExecutionFailed("No response".to_string()))?;

        let content = choice.message.content.clone().unwrap_or_default();
        let reasoning = choice.message.reasoning_content.clone();

        Ok((content, reasoning))
    }

    /// Get system prompt for a model
    fn get_system_prompt(&self, model_id: ModelId) -> String {
        match model_id {
            ModelId::Behemoth => {
                r#"You are an expert Rust architect with deep knowledge of:
- Ownership, borrowing, and lifetime patterns
- Async/await and Tokio ecosystem
- Error handling with thiserror/anyhow
- Type-state patterns and compile-time guarantees
- Zero-cost abstractions and performance optimization
- Memory safety without garbage collection

Provide detailed, well-reasoned analysis. Show your reasoning process.
At the end of your response, indicate your confidence level (0.0-1.0) in brackets like [confidence: 0.85]."#
                    .to_string()
            }
            ModelId::StrandCoder => {
                r#"You are an expert Rust coder specialized in writing idiomatic, safe, and efficient Rust code.
Focus on:
- Idiomatic Rust patterns and conventions
- Proper error handling (Result, Option, ?)
- Ownership and borrowing correctness
- Clippy-clean code
- Clear, readable implementations

Provide clean, compilable code with minimal explanation unless asked.
At the end of your response, indicate your confidence level (0.0-1.0) in brackets like [confidence: 0.85]."#
                    .to_string()
            }
            ModelId::HydraCoder => {
                r#"You are HydraCoder, a specialized Rust code generation model.
Your expertise includes:
- Complex lifetime and borrowing patterns
- Async/await with Tokio, futures, and streams
- Trait implementations and generic programming
- Macro development (declarative and procedural)
- Error handling patterns (thiserror, anyhow, custom errors)
- Popular ecosystem crates (serde, actix, axum, clap, tokio)

Generate idiomatic, zero-cost-abstraction Rust code. Prioritize compile-time safety and performance.
At the end of your response, indicate your confidence level (0.0-1.0) in brackets like [confidence: 0.85]."#
                    .to_string()
            }
        }
    }

    /// Build user prompt with context
    fn build_user_prompt(&self, task: &EnsembleTask, context_prompt: &str) -> String {
        let mut prompt = String::new();

        if !context_prompt.is_empty() {
            prompt.push_str(context_prompt);
            prompt.push_str("\n---\n\n");
        }

        prompt.push_str(&task.prompt);

        if let Some(ref code) = task.code_context {
            prompt.push_str("\n\n```rust\n");
            prompt.push_str(code);
            prompt.push_str("\n```");
        }

        prompt
    }

    /// Extract confidence from response
    fn extract_confidence(&self, response: &str, model_id: ModelId) -> f32 {
        // Look for [confidence: X.XX] pattern
        if let Some(start) = response.find("[confidence:") {
            if let Some(end) = response[start..].find(']') {
                let conf_str = &response[start + 12..start + end];
                if let Ok(conf) = conf_str.trim().parse::<f32>() {
                    return conf.clamp(0.0, 1.0);
                }
            }
        }

        // Default confidence based on model
        match model_id {
            ModelId::Behemoth => 0.7,
            ModelId::HydraCoder => 0.6,
            ModelId::StrandCoder => 0.5,
        }
    }

    // =========================================================================
    // Voting and Arbitration
    // =========================================================================

    /// Run voting on a task
    pub async fn vote_on_task(
        &self,
        task_id: &TaskId,
        strategy: Option<VotingStrategy>,
    ) -> CoordinatorResult<VoteOutcome> {
        let strategy = strategy.unwrap_or(self.config.default_voting_strategy);

        self.voting
            .vote(task_id, strategy)
            .await
            .map_err(|e| CoordinatorError::VotingError(e.to_string()))
    }

    /// Request arbitration for a task
    pub fn request_arbitration(
        &self,
        task_id: &TaskId,
        reason: crate::events::ArbitrationReason,
    ) -> CoordinatorResult<ArbitrationRequest> {
        self.arbitration
            .request_arbitration(task_id, reason)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))
    }

    /// Apply an arbitration decision
    pub fn apply_arbitration(
        &self,
        task_id: &TaskId,
        decision: ArbitrationDecision,
    ) -> CoordinatorResult<()> {
        self.arbitration
            .apply_decision(task_id, decision)
            .map_err(|e| CoordinatorError::StoreError(e.to_string()))
    }

    // =========================================================================
    // Context Access
    // =========================================================================

    /// Get the context manager
    pub fn context(&self) -> &ContextManager {
        &self.context
    }

    /// Get the store
    pub fn store(&self) -> &SharedStateStore {
        &self.store
    }
}

/// Status summary for MCP tool responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleStatus {
    pub session_id: String,
    pub session_active: bool,
    pub pending_tasks: usize,
    pub completed_tasks: usize,
    pub current_model: Option<String>,
    pub context_version: u64,
}

impl EnsembleStatus {
    pub fn from_session(session: &EnsembleSession, current_model: Option<ModelId>) -> Self {
        Self {
            session_id: session.id.clone(),
            session_active: session.active,
            pending_tasks: session.pending_tasks.len(),
            completed_tasks: session.completed_tasks.len(),
            current_model: current_model.map(|m| m.to_string()),
            context_version: session.context_version,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::EventBus;
    use crate::state::StateStore;
    use tempfile::tempdir;

    fn test_setup() -> (EnsembleCoordinator, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let store = StateStore::open(dir.path().join("test.db"))
            .unwrap()
            .shared();
        let bus = EventBus::new().shared();
        let config = EnsembleConfig::default();
        let coordinator = EnsembleCoordinator::new(store, bus, config).unwrap();
        (coordinator, dir)
    }

    #[tokio::test]
    async fn test_start_session() {
        let (coordinator, _dir) = test_setup();

        let session = coordinator.start_session(None).await.unwrap();
        assert!(session.active);
        assert!(session.pending_tasks.is_empty());
    }

    #[tokio::test]
    async fn test_submit_task() {
        let (coordinator, _dir) = test_setup();

        let session = coordinator.start_session(None).await.unwrap();
        let task = coordinator
            .submit_task("Implement error handling".to_string(), None, true)
            .await
            .unwrap();

        assert_eq!(task.session_id, session.id);
        assert!(task.require_consensus);
        assert_eq!(task.assigned_models.len(), 3); // All models for consensus
    }

    #[tokio::test]
    async fn test_end_session() {
        let (coordinator, _dir) = test_setup();

        let session = coordinator.start_session(None).await.unwrap();
        coordinator
            .end_session(&session.id, SessionEndReason::Completed)
            .await
            .unwrap();

        let ended = coordinator.get_session(&session.id).unwrap();
        assert!(!ended.active);
    }

    #[test]
    fn test_extract_confidence() {
        let (coordinator, _dir) = test_setup();

        let response_with_conf = "Here is the code...\n[confidence: 0.85]";
        let conf = coordinator.extract_confidence(response_with_conf, ModelId::Behemoth);
        assert!((conf - 0.85).abs() < 0.01);

        let response_without_conf = "Here is the code...";
        let conf = coordinator.extract_confidence(response_without_conf, ModelId::Behemoth);
        assert!((conf - 0.7).abs() < 0.01); // Default for Behemoth
    }

    #[test]
    fn test_build_user_prompt() {
        let (coordinator, _dir) = test_setup();

        let task = EnsembleTask::new("session-1".to_string(), "Fix this code".to_string(), false)
            .with_code("fn main() {}".to_string());

        let context = "## Previous Context\n\nWorking on error handling\n\n";
        let prompt = coordinator.build_user_prompt(&task, context);

        assert!(prompt.contains("Previous Context"));
        assert!(prompt.contains("Fix this code"));
        assert!(prompt.contains("fn main()"));
    }
}
