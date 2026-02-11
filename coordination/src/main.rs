//! MCP Server for Rust Cluster LLMs
//!
//! Provides three tools for delegating to local Rust-expert models:
//! - `ask_rust_architect`: OR1-Behemoth 73B for deep analysis, architecture, reasoning
//! - `ask_rust_coder`: Strand-Rust-Coder 14B for idiomatic fixes, refactoring, completion
//! - `ask_hydra_coder`: HydraCoder 31B MoE for specialized Rust code generation
//!
//! Also includes an agent harness module implementing Anthropic's patterns for
//! effective long-running agents.
//!
//! # Usage
//!
//! ```bash
//! # Standard MCP mode
//! rust-cluster-mcp
//!
//! # With harness tools enabled
//! rust-cluster-mcp --harness
//!
//! # Custom configuration
//! HARNESS_MAX_ITERATIONS=30 HARNESS_FEATURES_PATH=./features.json rust-cluster-mcp --harness
//! ```

// Suppress false positive dead_code warnings from #[tool_router] macro and serde deserialization
#![allow(dead_code)]

pub mod benchmark;
pub mod benchmark_tools;
pub mod ensemble;
pub mod events;
pub mod feedback;
pub mod harness;
pub mod router;
pub mod state;

use anyhow::Result;
use clap::Parser;
use rmcp::{
    handler::server::wrapper::Parameters,
    model::{ServerCapabilities, ServerInfo},
    schemars, tool, tool_router, ServerHandler, ServiceExt,
};
use serde::{Deserialize, Serialize};
use tokio::io::{stdin, stdout};

/// Command-line arguments
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Enable agent harness mode with session tracking, feature registry, and git checkpoints
    #[arg(long, default_value_t = false)]
    harness: bool,

    /// Maximum iterations for harness sessions (overrides HARNESS_MAX_ITERATIONS)
    #[arg(long)]
    max_iterations: Option<u32>,

    /// Path to features.json registry (overrides HARNESS_FEATURES_PATH)
    #[arg(long)]
    features_path: Option<std::path::PathBuf>,

    /// Path to progress file (overrides HARNESS_PROGRESS_PATH)
    #[arg(long)]
    progress_path: Option<std::path::PathBuf>,

    /// Require clean git state before starting harness
    #[arg(long, default_value_t = false)]
    require_clean_git: bool,

    /// Enable ensemble mode for multi-model coordination
    #[arg(long, default_value_t = false)]
    ensemble: bool,

    /// Path to RocksDB state directory for ensemble persistence
    #[arg(long)]
    state_path: Option<std::path::PathBuf>,
}

/// Configuration for LLM router endpoint
#[derive(Clone)]
struct LlmConfig {
    router_url: String,
    architect_model: String,
    coder_model: String,
    hydra_model: String,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            router_url: std::env::var("ROUTER_URL")
                .unwrap_or_else(|_| "http://10.0.0.31:8000/v1/chat/completions".to_string()),
            architect_model: std::env::var("ARCHITECT_MODEL")
                .unwrap_or_else(|_| "OR1-Behemoth.Q8_0".to_string()),
            coder_model: std::env::var("CODER_MODEL")
                .unwrap_or_else(|_| "Strand-Rust-Coder-14B-v1-Q8_0".to_string()),
            hydra_model: std::env::var("HYDRA_MODEL")
                .unwrap_or_else(|_| "HydraCoder.Q6_K".to_string()),
        }
    }
}

/// Request parameters for architect tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct ArchitectRequest {
    #[schemars(
        description = "Detailed question about Rust architecture, ownership patterns, async design, error handling strategies, or system design"
    )]
    prompt: String,
    #[schemars(description = "Optional: Existing code to analyze or improve")]
    code_context: Option<String>,
    #[schemars(description = "Maximum response length (default: 2048)")]
    max_tokens: Option<u32>,
}

/// Request parameters for coder tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct CoderRequest {
    #[schemars(description = "Specific Rust coding task: fix, refactor, complete, or implement")]
    prompt: String,
    #[schemars(description = "The Rust code to work with")]
    code: Option<String>,
    #[schemars(description = "Maximum response length (default: 1024)")]
    max_tokens: Option<u32>,
}

/// Request parameters for HydraCoder tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct HydraRequest {
    #[schemars(
        description = "Rust coding task requiring deep Rust expertise: complex implementations, async patterns, lifetime management, trait implementations"
    )]
    prompt: String,
    #[schemars(description = "Optional: Existing Rust code to work with or improve")]
    code: Option<String>,
    #[schemars(description = "Maximum response length (default: 2048)")]
    max_tokens: Option<u32>,
}

// ============================================================================
// Harness MCP Request Types (use rmcp's schemars)
// ============================================================================

/// MCP request for harness_start tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessStartRequest {
    #[schemars(description = "Maximum iterations before session auto-stops")]
    max_iterations: Option<u32>,
    #[schemars(description = "Require no uncommitted changes before starting")]
    require_clean_git: Option<bool>,
    #[schemars(description = "Resume interrupted session if found")]
    auto_resume: Option<bool>,
}

/// MCP request for harness_status tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessStatusRequest {
    #[schemars(description = "Include full feature list in response")]
    include_features: Option<bool>,
    #[schemars(description = "Include recent progress log entries")]
    include_progress: Option<bool>,
    #[schemars(description = "Maximum features to include in response (default: 20)")]
    max_features: Option<u32>,
    #[schemars(description = "Maximum progress entries to include (default: 10)")]
    max_progress_entries: Option<u32>,
}

/// MCP request for harness_iterate tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessIterateRequest {
    #[schemars(description = "Brief summary of what was accomplished")]
    summary: String,
    #[schemars(description = "Feature ID currently being worked on")]
    feature_id: Option<String>,
}

/// MCP request for harness_complete_feature tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessCompleteFeatureRequest {
    #[schemars(description = "ID of the feature to mark as complete")]
    feature_id: String,
    #[schemars(description = "Brief summary of how the feature was implemented")]
    summary: String,
    #[schemars(description = "Create git checkpoint after marking complete")]
    checkpoint: Option<bool>,
}

/// MCP request for harness_checkpoint tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessCheckpointRequest {
    #[schemars(description = "Description of what this checkpoint captures")]
    description: String,
    #[schemars(description = "Feature ID this checkpoint is for")]
    feature_id: Option<String>,
}

/// MCP request for harness_rollback tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessRollbackRequest {
    #[schemars(description = "Git commit hash to rollback to")]
    commit_hash: String,
    #[schemars(description = "Hard rollback discards changes, soft preserves them")]
    hard: Option<bool>,
}

/// MCP request for harness_compact_progress tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessCompactProgressRequest {
    #[schemars(description = "Keep this many recent entries without summarization (default: 10)")]
    keep_recent: Option<u32>,
    #[schemars(
        description = "Set to true to perform compaction, false to preview (default: false)"
    )]
    execute: Option<bool>,
}

/// MCP request for harness_work_on_feature tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessWorkOnFeatureRequest {
    #[schemars(description = "ID of the feature to start working on")]
    feature_id: String,
    #[schemars(description = "Brief summary of the work being done")]
    summary: String,
}

/// MCP request for harness_complete_and_next tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessCompleteAndNextRequest {
    #[schemars(description = "ID of the feature to mark as complete")]
    feature_id: String,
    #[schemars(description = "Brief summary of how the feature was implemented")]
    summary: String,
    #[schemars(description = "Create git checkpoint after completion (default: true)")]
    checkpoint: Option<bool>,
}

/// MCP request for harness_quick_status tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessQuickStatusRequest {}

/// MCP request for harness_acknowledge tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessAcknowledgeRequest {
    #[schemars(description = "IDs of checklist items that were reviewed (optional)")]
    reviewed_items: Option<Vec<String>>,
}

/// MCP request for harness_request_intervention tool (Phase 5)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessRequestInterventionRequest {
    #[schemars(
        description = "Type of intervention: review_required, approval_needed, decision_point, clarification_needed"
    )]
    intervention_type: String,
    #[schemars(description = "Question or description for the human")]
    question: String,
    #[schemars(description = "Associated feature ID (optional)")]
    feature_id: Option<String>,
    #[schemars(description = "Options for decision points (optional)")]
    options: Option<Vec<String>>,
}

/// MCP request for harness_resolve_intervention tool (Phase 5)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessResolveInterventionRequest {
    #[schemars(description = "Intervention ID to resolve")]
    intervention_id: String,
    #[schemars(description = "Resolution or decision made")]
    resolution: String,
}

/// MCP request for harness_delegate tool (Phase 6)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessDelegateRequest {
    #[schemars(description = "Feature ID for the delegated work")]
    feature_id: String,
    #[schemars(description = "Detailed description of the task for the sub-agent")]
    task_description: String,
    #[schemars(description = "Maximum iterations for the sub-agent (default: 10)")]
    max_iterations: Option<u32>,
}

/// MCP request for harness_sub_session_status tool (Phase 6)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessSubSessionStatusRequest {
    #[schemars(description = "Sub-session ID to check status of")]
    sub_session_id: String,
}

/// MCP request for harness_claim_sub_session_result tool (Phase 6)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessClaimSubSessionResultRequest {
    #[schemars(description = "Sub-session ID to claim results from")]
    sub_session_id: String,
    #[schemars(description = "Optional summary to use instead of sub-session's own summary")]
    summary: Option<String>,
}

/// MCP request for harness_complete_sub_session tool (Phase 6)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessCompleteSubSessionRequest {
    #[schemars(description = "Sub-session ID to complete")]
    sub_session_id: String,
    #[schemars(description = "Summary of work completed")]
    summary: String,
}

/// MCP request for harness_fail_sub_session tool (Phase 6)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpHarnessFailSubSessionRequest {
    #[schemars(description = "Sub-session ID that failed")]
    sub_session_id: String,
    #[schemars(description = "Reason for failure")]
    reason: String,
}

// ============================================================================
// Ensemble MCP Request Types
// ============================================================================

/// MCP request for ensemble_start tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpEnsembleStartRequest {
    #[schemars(description = "Optional harness session ID to link with")]
    harness_session_id: Option<String>,
}

/// MCP request for ensemble_submit tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpEnsembleSubmitRequest {
    #[schemars(description = "The prompt/task for the ensemble to process")]
    prompt: String,
    #[schemars(description = "Optional code context to analyze or improve")]
    code_context: Option<String>,
    #[schemars(description = "Whether consensus from all models is required (default: true)")]
    require_consensus: Option<bool>,
    #[schemars(description = "Execute immediately after submitting (default: true)")]
    execute: Option<bool>,
}

/// MCP request for ensemble_status tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpEnsembleStatusRequest {
    #[schemars(description = "Optional session ID (uses active session if not provided)")]
    session_id: Option<String>,
    #[schemars(description = "Include task details in response")]
    include_tasks: Option<bool>,
}

/// MCP request for ensemble_vote tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpEnsembleVoteRequest {
    #[schemars(description = "Task ID to vote on")]
    task_id: String,
    #[schemars(
        description = "Voting strategy: 'majority', 'weighted', or 'unanimous' (default: weighted)"
    )]
    strategy: Option<String>,
}

/// MCP request for ensemble_arbitrate tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpEnsembleArbitrateRequest {
    #[schemars(description = "Task ID to arbitrate")]
    task_id: String,
    #[schemars(
        description = "Reason for arbitration: 'tie', 'low_confidence', 'conflict', or 'explicit'"
    )]
    reason: Option<String>,
}

/// MCP request for ensemble_apply_decision tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpEnsembleApplyDecisionRequest {
    #[schemars(description = "Task ID to apply decision to")]
    task_id: String,
    #[schemars(description = "Winning model: 'behemoth', 'strand_coder', or 'hydra_coder'")]
    winner: String,
    #[schemars(description = "Rationale for the decision")]
    rationale: String,
    #[schemars(description = "Optional modified/synthesized response")]
    modified_response: Option<String>,
}

/// MCP request for ensemble_context tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpEnsembleContextRequest {
    #[schemars(description = "Optional session ID (uses active session if not provided)")]
    session_id: Option<String>,
    #[schemars(description = "Optional new summary to set")]
    summary: Option<String>,
    #[schemars(description = "Optional decision to add")]
    decision: Option<String>,
    #[schemars(description = "Optional file reference to add")]
    file_reference: Option<String>,
}

/// MCP request for ensemble_replay tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
struct McpEnsembleReplayRequest {
    #[schemars(description = "Optional session ID to filter events")]
    session_id: Option<String>,
    #[schemars(description = "Optional task ID to filter events")]
    task_id: Option<String>,
    #[schemars(description = "Number of minutes of history to replay (default: 60)")]
    minutes: Option<i64>,
}

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

/// The MCP server handler
#[derive(Clone)]
struct RustClusterServer {
    config: LlmConfig,
    http: reqwest::Client,
    /// Optional harness state for harness mode
    harness: Option<harness::SharedHarnessState>,
    /// Optional ensemble coordinator for ensemble mode
    ensemble: Option<ensemble::SharedEnsembleCoordinator>,
    /// Optional state store for persistence
    state_store: Option<state::SharedStateStore>,
    /// Optional event bus for pub/sub
    event_bus: Option<events::SharedEventBus>,
}

impl RustClusterServer {
    fn new() -> Result<Self, reqwest::Error> {
        Ok(Self {
            config: LlmConfig::default(),
            http: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(300))
                .build()?,
            harness: None,
            ensemble: None,
            state_store: None,
            event_bus: None,
        })
    }

    fn with_harness(mut self, state: harness::SharedHarnessState) -> Self {
        self.harness = Some(state);
        self
    }

    fn with_ensemble(
        mut self,
        coordinator: ensemble::SharedEnsembleCoordinator,
        store: state::SharedStateStore,
        bus: events::SharedEventBus,
    ) -> Self {
        self.ensemble = Some(coordinator);
        self.state_store = Some(store);
        self.event_bus = Some(bus);
        self
    }

    fn get_harness(&self) -> Result<harness::SharedHarnessState, String> {
        self.harness
            .clone()
            .ok_or_else(|| "Harness not enabled. Start server with --harness flag.".to_string())
    }

    fn get_ensemble(&self) -> Result<ensemble::SharedEnsembleCoordinator, String> {
        self.ensemble
            .clone()
            .ok_or_else(|| "Ensemble not enabled. Start server with --ensemble flag.".to_string())
    }

    fn get_event_bus(&self) -> Result<events::SharedEventBus, String> {
        self.event_bus
            .clone()
            .ok_or_else(|| "Ensemble not enabled. Start server with --ensemble flag.".to_string())
    }

    fn get_state_store(&self) -> Result<state::SharedStateStore, String> {
        self.state_store
            .clone()
            .ok_or_else(|| "Ensemble not enabled. Start server with --ensemble flag.".to_string())
    }

    async fn query_llm(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        max_tokens: u32,
        temperature: f32,
    ) -> Result<String, String> {
        let request = ChatRequest {
            model: model.to_string(),
            messages,
            max_tokens,
            temperature,
        };

        let response = self
            .http
            .post(&self.config.router_url)
            .json(&request)
            .send()
            .await
            .map_err(|e| format!("HTTP request failed: {}", e))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(format!("LLM API error ({}): {}", status, body));
        }

        let chat_response: ChatResponse = response
            .json()
            .await
            .map_err(|e| format!("Failed to parse response: {}", e))?;

        let choice = chat_response
            .choices
            .first()
            .ok_or("No response from LLM")?;

        let content = if let Some(reasoning) = &choice.message.reasoning_content {
            if let Some(answer) = &choice.message.content {
                format!("<reasoning>\n{}\n</reasoning>\n\n{}", reasoning, answer)
            } else {
                reasoning.clone()
            }
        } else {
            choice.message.content.clone().unwrap_or_default()
        };

        Ok(content)
    }
}

#[tool_router]
impl RustClusterServer {
    #[tool(
        description = "Ask the Rust Architect (OR1-Behemoth 73B) for deep analysis of architecture decisions, ownership patterns, async design, error handling strategies, and complex system design. Best for questions requiring extended reasoning. ~11 tokens/sec (distributed GPU)."
    )]
    async fn ask_rust_architect(
        &self,
        Parameters(req): Parameters<ArchitectRequest>,
    ) -> Result<String, String> {
        let system_prompt = r#"You are an expert Rust architect with deep knowledge of:
- Ownership, borrowing, and lifetime patterns
- Async/await and Tokio ecosystem
- Error handling with thiserror/anyhow
- Type-state patterns and compile-time guarantees
- Zero-cost abstractions and performance optimization
- Memory safety without garbage collection

Provide detailed, well-reasoned analysis. Show your reasoning process."#;

        let user_content = match req.code_context {
            Some(code) => format!("{}\n\n```rust\n{}\n```", req.prompt, code),
            None => req.prompt,
        };

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_content,
            },
        ];

        self.query_llm(
            &self.config.architect_model,
            messages,
            req.max_tokens.unwrap_or(2048),
            0.7,
        )
        .await
    }

    #[tool(
        description = "Ask the Rust Coder (Strand-Rust-Coder 14B) for idiomatic code fixes, refactoring, completion, and implementation. Best for concrete coding tasks. ~53 tokens/sec (local GPU). Note: First request may take ~10s to load model."
    )]
    async fn ask_rust_coder(
        &self,
        Parameters(req): Parameters<CoderRequest>,
    ) -> Result<String, String> {
        let system_prompt = r#"You are an expert Rust coder specialized in writing idiomatic, safe, and efficient Rust code.
Focus on:
- Idiomatic Rust patterns and conventions
- Proper error handling (Result, Option, ?)
- Ownership and borrowing correctness
- Clippy-clean code
- Clear, readable implementations

Provide clean, compilable code with minimal explanation unless asked."#;

        let user_content = match req.code {
            Some(code) => format!("{}\n\n```rust\n{}\n```", req.prompt, code),
            None => req.prompt,
        };

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_content,
            },
        ];

        self.query_llm(
            &self.config.coder_model,
            messages,
            req.max_tokens.unwrap_or(1024),
            0.3,
        )
        .await
    }

    #[tool(
        description = "Ask HydraCoder (31B MoE, ~7.5B active) for specialized Rust code generation. Fine-tuned on 180k+ Rust samples including tokio, serde, actix, clap ecosystems. Best for complex Rust patterns, lifetime management, async implementations, and trait designs. ~40-60 tokens/sec (local GPU). Note: First request may take ~15s to load model."
    )]
    async fn ask_hydra_coder(
        &self,
        Parameters(req): Parameters<HydraRequest>,
    ) -> Result<String, String> {
        let system_prompt = r#"You are HydraCoder, a specialized Rust code generation model trained on 180k+ Rust samples.
Your expertise includes:
- Complex lifetime and borrowing patterns
- Async/await with Tokio, futures, and streams
- Trait implementations and generic programming
- Macro development (declarative and procedural)
- Error handling patterns (thiserror, anyhow, custom errors)
- Popular ecosystem crates (serde, actix, axum, clap, tokio)

Generate idiomatic, zero-cost-abstraction Rust code. Prioritize compile-time safety and performance.
Include necessary imports and derive macros. Code should be clippy-clean and well-structured."#;

        let user_content = match req.code {
            Some(code) => format!("{}\n\n```rust\n{}\n```", req.prompt, code),
            None => req.prompt,
        };

        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: system_prompt.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user_content,
            },
        ];

        self.query_llm(
            &self.config.hydra_model,
            messages,
            req.max_tokens.unwrap_or(2048),
            0.2,
        )
        .await
    }

    // ========================================================================
    // Harness Tools (available when --harness flag is used)
    // ========================================================================

    #[tool(
        description = "Start or resume a harness session for structured agent work. Returns session context including git state, feature progress, and recent activity. Use at the beginning of an agent session."
    )]
    async fn harness_start(
        &self,
        Parameters(req): Parameters<McpHarnessStartRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessStartRequest {
            max_iterations: req.max_iterations,
            require_clean_git: req.require_clean_git,
            auto_resume: req.auto_resume,
        };
        let response = harness::tools::harness_start(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Get current harness session status including iteration count, feature completion, and git state.",
        annotations(read_only_hint = true)
    )]
    async fn harness_status(
        &self,
        Parameters(req): Parameters<McpHarnessStatusRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessStatusRequest {
            include_features: req.include_features,
            include_progress: req.include_progress,
            max_features: req.max_features,
            max_progress_entries: req.max_progress_entries,
        };
        let response = harness::tools::harness_status(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Increment the iteration counter and log progress. Call this at the end of each work iteration to track progress and enforce iteration limits."
    )]
    async fn harness_iterate(
        &self,
        Parameters(req): Parameters<McpHarnessIterateRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessIterateRequest {
            summary: req.summary,
            feature_id: req.feature_id,
        };
        let response = harness::tools::harness_iterate(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Mark a feature as complete in the registry. Optionally creates a git checkpoint."
    )]
    async fn harness_complete_feature(
        &self,
        Parameters(req): Parameters<McpHarnessCompleteFeatureRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessCompleteFeatureRequest {
            feature_id: req.feature_id,
            summary: req.summary,
            checkpoint: req.checkpoint,
        };
        let response = harness::tools::harness_complete_feature(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Create a git checkpoint commit with the current state. Use for recoverable save points."
    )]
    async fn harness_checkpoint(
        &self,
        Parameters(req): Parameters<McpHarnessCheckpointRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessCheckpointRequest {
            description: req.description,
            feature_id: req.feature_id,
        };
        let response = harness::tools::harness_checkpoint(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Rollback to a previous git checkpoint. Use for recovery when work goes wrong. WARNING: This is a destructive operation.",
        annotations(destructive_hint = true)
    )]
    async fn harness_rollback(
        &self,
        Parameters(req): Parameters<McpHarnessRollbackRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessRollbackRequest {
            commit_hash: req.commit_hash,
            hard: req.hard,
        };
        let response = harness::tools::harness_rollback(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Compact old progress entries to manage token budget. Summarizes entries older than keep_recent count. Use execute=true to actually perform compaction, or false to preview."
    )]
    async fn harness_compact_progress(
        &self,
        Parameters(req): Parameters<McpHarnessCompactProgressRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessCompactProgressRequest {
            keep_recent: req.keep_recent,
            execute: req.execute,
        };
        let response = harness::tools::harness_compact_progress(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Start working on a feature. Combines iterate + set_current_feature + log progress in one call. Returns feature details and steps. Workflow tool to reduce API calls."
    )]
    async fn harness_work_on_feature(
        &self,
        Parameters(req): Parameters<McpHarnessWorkOnFeatureRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessWorkOnFeatureRequest {
            feature_id: req.feature_id,
            summary: req.summary,
        };
        let response = harness::tools::harness_work_on_feature(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Complete a feature and get the next one. Combines complete_feature + checkpoint + get_next_feature in one call. Workflow tool to reduce API calls."
    )]
    async fn harness_complete_and_next(
        &self,
        Parameters(req): Parameters<McpHarnessCompleteAndNextRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessCompleteAndNextRequest {
            feature_id: req.feature_id,
            summary: req.summary,
            checkpoint: req.checkpoint,
        };
        let response = harness::tools::harness_complete_and_next(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Get minimal session status for rapid polling. Returns only: iteration, max_iterations, can_continue, current_feature, next_feature, session_status. No features or progress entries.",
        annotations(read_only_hint = true)
    )]
    async fn harness_quick_status(
        &self,
        Parameters(_req): Parameters<McpHarnessQuickStatusRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessQuickStatusRequest {};
        let response = harness::tools::harness_quick_status(&state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Acknowledge completion of the startup ritual. Call this AFTER reading harness_status to confirm you understand the session context. Returns a warning if you didn't read status first."
    )]
    async fn harness_acknowledge(
        &self,
        Parameters(req): Parameters<McpHarnessAcknowledgeRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessAcknowledgeRequest {
            reviewed_items: req.reviewed_items,
        };
        let response = harness::tools::harness_acknowledge(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Request human intervention for a judgment decision. Use for: review_required (human should review), approval_needed (explicit approval for destructive action), decision_point (multiple valid paths, human chooses), clarification_needed (unclear requirements). Returns an intervention_id to resolve later."
    )]
    async fn harness_request_intervention(
        &self,
        Parameters(req): Parameters<McpHarnessRequestInterventionRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessRequestInterventionRequest {
            intervention_type: req.intervention_type,
            question: req.question,
            feature_id: req.feature_id,
            options: req.options,
        };
        let response = harness::tools::harness_request_intervention(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Resolve a pending human intervention. Provide the intervention_id and the resolution/decision made. This unblocks any feature that was waiting on the intervention."
    )]
    async fn harness_resolve_intervention(
        &self,
        Parameters(req): Parameters<McpHarnessResolveInterventionRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessResolveInterventionRequest {
            intervention_id: req.intervention_id,
            resolution: req.resolution,
        };
        let response = harness::tools::harness_resolve_intervention(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    // ========================================================================
    // Phase 6: Sub-Agent Delegation Tools
    // ========================================================================

    #[tool(
        description = "Delegate work to an isolated sub-session. Creates a sub-session with separate context for token-heavy subtasks. Returns sub_session_id and context_path for the sub-agent to read."
    )]
    async fn harness_delegate(
        &self,
        Parameters(req): Parameters<McpHarnessDelegateRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessDelegateRequest {
            feature_id: req.feature_id,
            task_description: req.task_description,
            max_iterations: req.max_iterations,
        };
        let response = harness::tools::harness_delegate(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Check status of a sub-session. Returns current iteration, max_iterations, status, and summary if complete.",
        annotations(read_only_hint = true)
    )]
    async fn harness_sub_session_status(
        &self,
        Parameters(req): Parameters<McpHarnessSubSessionStatusRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessSubSessionStatusRequest {
            sub_session_id: req.sub_session_id,
        };
        let response = harness::tools::harness_sub_session_status(&state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Claim results from a completed sub-session. Incorporates the sub-session work into the main session and cleans up the context file."
    )]
    async fn harness_claim_sub_session_result(
        &self,
        Parameters(req): Parameters<McpHarnessClaimSubSessionResultRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let harness_req = harness::tools::HarnessClaimSubSessionResultRequest {
            sub_session_id: req.sub_session_id,
            summary: req.summary,
        };
        let response = harness::tools::harness_claim_sub_session_result(&mut state, harness_req)
            .map_err(|e| e.to_structured_json())?;
        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Complete a sub-session (called by the sub-agent). Mark the sub-session as complete with a summary of work done."
    )]
    async fn harness_complete_sub_session(
        &self,
        Parameters(req): Parameters<McpHarnessCompleteSubSessionRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        harness::tools::harness_complete_sub_session(&mut state, &req.sub_session_id, &req.summary)
            .map_err(|e| e.to_structured_json())?;
        Ok(r#"{"success": true, "message": "Sub-session completed"}"#.to_string())
    }

    #[tool(
        description = "Fail a sub-session (called by the sub-agent on error). Mark the sub-session as failed with a reason."
    )]
    async fn harness_fail_sub_session(
        &self,
        Parameters(req): Parameters<McpHarnessFailSubSessionRequest>,
    ) -> Result<String, String> {
        let shared = self.get_harness()?;
        let mut state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        harness::tools::harness_fail_sub_session(&mut state, &req.sub_session_id, &req.reason)
            .map_err(|e| e.to_structured_json())?;
        Ok(r#"{"success": true, "message": "Sub-session marked as failed"}"#.to_string())
    }

    #[tool(
        description = "List all active sub-sessions for the current session.",
        annotations(read_only_hint = true)
    )]
    async fn harness_list_sub_sessions(&self) -> Result<String, String> {
        let shared = self.get_harness()?;
        let state = shared.lock().map_err(|e| format!("Lock error: {}", e))?;
        let sub_sessions = harness::tools::harness_list_sub_sessions(&state);
        serde_json::to_string_pretty(&sub_sessions).map_err(|e| e.to_string())
    }

    // ========================================================================
    // Ensemble Tools (available when --ensemble flag is used)
    // ========================================================================

    #[tool(
        description = "Start a new ensemble session for multi-model coordination. Creates persistent state for tracking tasks across model swaps. Optionally links to an existing harness session."
    )]
    async fn ensemble_start(
        &self,
        Parameters(req): Parameters<McpEnsembleStartRequest>,
    ) -> Result<String, String> {
        let coordinator = self.get_ensemble()?;
        let session = coordinator
            .start_session(req.harness_session_id)
            .await
            .map_err(|e| format!("Ensemble error: {}", e))?;

        let status = ensemble::EnsembleStatus::from_session(&session, None);
        serde_json::to_string_pretty(&status).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Submit a task for multi-model ensemble processing. When require_consensus is true, all 3 models (Behemoth, HydraCoder, StrandCoder) will process the task and voting determines the winner."
    )]
    async fn ensemble_submit(
        &self,
        Parameters(req): Parameters<McpEnsembleSubmitRequest>,
    ) -> Result<String, String> {
        let coordinator = self.get_ensemble()?;
        let require_consensus = req.require_consensus.unwrap_or(true);
        let execute = req.execute.unwrap_or(true);

        let task = coordinator
            .submit_task(req.prompt, req.code_context, require_consensus)
            .await
            .map_err(|e| format!("Ensemble error: {}", e))?;

        if execute {
            // Execute the task through all assigned models
            let executed_task = coordinator
                .execute_task(&task.id)
                .await
                .map_err(|e| format!("Ensemble error: {}", e))?;

            #[derive(serde::Serialize)]
            struct TaskResponse {
                task_id: String,
                status: String,
                completed_models: Vec<String>,
                all_complete: bool,
            }

            let response = TaskResponse {
                task_id: executed_task.id.clone(),
                status: format!("{:?}", executed_task.status),
                completed_models: executed_task
                    .completed_models
                    .iter()
                    .map(|m| m.to_string())
                    .collect(),
                all_complete: executed_task.all_models_complete(),
            };

            serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
        } else {
            #[derive(serde::Serialize)]
            struct TaskResponse {
                task_id: String,
                status: String,
                assigned_models: Vec<String>,
            }

            let response = TaskResponse {
                task_id: task.id.clone(),
                status: format!("{:?}", task.status),
                assigned_models: task.assigned_models.iter().map(|m| m.to_string()).collect(),
            };

            serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
        }
    }

    #[tool(
        description = "Get current ensemble session status including pending tasks, completed tasks, and current model state."
    )]
    async fn ensemble_status(
        &self,
        Parameters(req): Parameters<McpEnsembleStatusRequest>,
    ) -> Result<String, String> {
        let coordinator = self.get_ensemble()?;

        let session = if let Some(ref session_id) = req.session_id {
            coordinator
                .get_session(session_id)
                .map_err(|e| format!("Ensemble error: {}", e))?
        } else {
            coordinator
                .get_active_session()
                .map_err(|e| format!("Ensemble error: {}", e))?
        };

        #[derive(serde::Serialize)]
        struct StatusResponse {
            session_id: String,
            active: bool,
            pending_tasks: usize,
            completed_tasks: usize,
            tasks: Option<Vec<TaskInfo>>,
        }

        #[derive(serde::Serialize)]
        struct TaskInfo {
            id: String,
            status: String,
            prompt_preview: String,
        }

        let tasks = if req.include_tasks.unwrap_or(false) {
            let store = self.get_state_store()?;
            let session_tasks = store
                .get_session_tasks(&session.id)
                .map_err(|e| format!("Store error: {}", e))?;

            Some(
                session_tasks
                    .iter()
                    .map(|t| TaskInfo {
                        id: t.id.clone(),
                        status: format!("{:?}", t.status),
                        prompt_preview: if t.prompt.len() > 50 {
                            format!("{}...", &t.prompt[..50])
                        } else {
                            t.prompt.clone()
                        },
                    })
                    .collect(),
            )
        } else {
            None
        };

        let response = StatusResponse {
            session_id: session.id.clone(),
            active: session.active,
            pending_tasks: session.pending_tasks.len(),
            completed_tasks: session.completed_tasks.len(),
            tasks,
        };

        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Trigger voting on collected results for a task. Uses weighted voting by default, combining model confidence with model expertise weight."
    )]
    async fn ensemble_vote(
        &self,
        Parameters(req): Parameters<McpEnsembleVoteRequest>,
    ) -> Result<String, String> {
        let coordinator = self.get_ensemble()?;

        let strategy = match req.strategy.as_deref() {
            Some("majority") => state::VotingStrategy::Majority,
            Some("unanimous") => state::VotingStrategy::Unanimous,
            _ => state::VotingStrategy::Weighted,
        };

        let outcome = coordinator
            .vote_on_task(&req.task_id, Some(strategy))
            .await
            .map_err(|e| format!("Voting error: {}", e))?;

        #[derive(serde::Serialize)]
        struct VoteResponse {
            winner: String,
            arbitrated: bool,
            vote_summary: VoteSummaryInfo,
        }

        #[derive(serde::Serialize)]
        struct VoteSummaryInfo {
            total_votes: u32,
            margin: u32,
            avg_confidence: f32,
        }

        let response = VoteResponse {
            winner: outcome.winner.to_string(),
            arbitrated: outcome.arbitrated,
            vote_summary: VoteSummaryInfo {
                total_votes: outcome.summary.total_votes,
                margin: outcome.summary.margin,
                avg_confidence: outcome.summary.avg_confidence,
            },
        };

        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Request Claude arbitration for a task when voting fails to produce a clear winner (tie or low confidence). Returns the arbitration request with all model responses for comparison."
    )]
    async fn ensemble_arbitrate(
        &self,
        Parameters(req): Parameters<McpEnsembleArbitrateRequest>,
    ) -> Result<String, String> {
        let coordinator = self.get_ensemble()?;

        let reason = match req.reason.as_deref() {
            Some("tie") => events::ArbitrationReason::TieVote {
                tied_models: vec![],
            },
            Some("low_confidence") => events::ArbitrationReason::LowConfidence {
                max_confidence: 0.3,
            },
            Some("conflict") => events::ArbitrationReason::ConflictingResponses {
                description: "Responses differ significantly".to_string(),
            },
            _ => events::ArbitrationReason::ExplicitRequest {
                requester: "claude".to_string(),
            },
        };

        let request = coordinator
            .request_arbitration(&req.task_id, reason)
            .map_err(|e| format!("Arbitration error: {}", e))?;

        serde_json::to_string_pretty(&request).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Apply an arbitration decision to a task. Called after Claude reviews the model responses and selects a winner."
    )]
    async fn ensemble_apply_decision(
        &self,
        Parameters(req): Parameters<McpEnsembleApplyDecisionRequest>,
    ) -> Result<String, String> {
        let coordinator = self.get_ensemble()?;

        let winner = match req.winner.to_lowercase().as_str() {
            "behemoth" => state::ModelId::Behemoth,
            "strand_coder" | "strandcoder" => state::ModelId::StrandCoder,
            "hydra_coder" | "hydracoder" => state::ModelId::HydraCoder,
            _ => return Err(format!("Unknown model: {}", req.winner)),
        };

        let decision = ensemble::ArbitrationDecision {
            winner,
            rationale: req.rationale,
            modified_response: req.modified_response,
            notes: None,
        };

        coordinator
            .apply_arbitration(&req.task_id, decision)
            .map_err(|e| format!("Arbitration error: {}", e))?;

        Ok(r#"{"status": "decision_applied"}"#.to_string())
    }

    #[tool(
        description = "Get or update the shared context for an ensemble session. Context survives model swaps and helps maintain coherence across the ensemble."
    )]
    async fn ensemble_context(
        &self,
        Parameters(req): Parameters<McpEnsembleContextRequest>,
    ) -> Result<String, String> {
        let coordinator = self.get_ensemble()?;
        let context_mgr = coordinator.context();

        let session_id = if let Some(ref id) = req.session_id {
            id.clone()
        } else {
            coordinator
                .get_active_session()
                .map_err(|e| format!("Ensemble error: {}", e))?
                .id
        };

        // Apply updates if provided
        if let Some(ref summary) = req.summary {
            context_mgr
                .update_summary(
                    &session_id,
                    summary.clone(),
                    events::ContextUpdater::Overseer,
                )
                .map_err(|e| format!("Context error: {}", e))?;
        }

        if let Some(ref decision) = req.decision {
            context_mgr
                .add_decision(
                    &session_id,
                    decision.clone(),
                    events::ContextUpdater::Overseer,
                )
                .map_err(|e| format!("Context error: {}", e))?;
        }

        if let Some(ref file) = req.file_reference {
            context_mgr
                .add_file_reference(&session_id, file.clone())
                .map_err(|e| format!("Context error: {}", e))?;
        }

        // Get current context
        let ctx = context_mgr
            .get_or_create(&session_id)
            .map_err(|e| format!("Context error: {}", e))?;

        let snapshot = ensemble::ContextSnapshot::from(ctx);
        serde_json::to_string_pretty(&snapshot).map_err(|e| e.to_string())
    }

    #[tool(
        description = "Replay events from the event history for debugging or recovery. Can filter by session or task ID."
    )]
    async fn ensemble_replay(
        &self,
        Parameters(req): Parameters<McpEnsembleReplayRequest>,
    ) -> Result<String, String> {
        let store = self.get_state_store()?;
        let history = events::EventHistory::new(store);

        let minutes = req.minutes.unwrap_or(60);
        let mut events = history
            .get_recent_events(minutes)
            .map_err(|e| format!("History error: {}", e))?;

        // Apply filters
        if let Some(ref session_id) = req.session_id {
            events.retain(|e| e.session_id() == Some(session_id.as_str()));
        }

        if let Some(ref task_id) = req.task_id {
            events.retain(|e| e.task_id() == Some(task_id.as_str()));
        }

        #[derive(serde::Serialize)]
        struct ReplayResponse {
            event_count: usize,
            events: Vec<EventInfo>,
        }

        #[derive(serde::Serialize)]
        struct EventInfo {
            event_type: String,
            timestamp: String,
            session_id: Option<String>,
            task_id: Option<String>,
        }

        let event_infos: Vec<EventInfo> = events
            .iter()
            .map(|e| EventInfo {
                event_type: e.event_type().to_string(),
                timestamp: e.timestamp().to_rfc3339(),
                session_id: e.session_id().map(String::from),
                task_id: e.task_id().map(String::from),
            })
            .collect();

        let response = ReplayResponse {
            event_count: event_infos.len(),
            events: event_infos,
        };

        serde_json::to_string_pretty(&response).map_err(|e| e.to_string())
    }
}

impl ServerHandler for RustClusterServer {
    fn get_info(&self) -> ServerInfo {
        let base_instructions = "MCP server providing access to local Rust-expert LLMs via llama.cpp router mode.\n\
                 Models auto-switch on demand (LRU eviction when memory constrained).\n\
                 - ask_rust_architect: OR1-Behemoth 73B (distributed GPU via RPC, ~11 tok/s)\n\
                 - ask_rust_coder: Strand-Rust-Coder 14B (local GPU, ~53 tok/s)\n\
                 - ask_hydra_coder: HydraCoder 31B MoE (local GPU, ~40-60 tok/s) - specialized Rust fine-tune";

        let harness_instructions = "\n\n\
## Agent Harness Workflow (when --harness enabled)

This harness manages long-running agent sessions with structured progress tracking.

### Recommended Workflow:
1. `harness_start` → Initialize or resume session
2. `harness_quick_status` → Check current state (minimal response)
3. `harness_work_on_feature` → Start working on a feature (increments iteration)
4. [do actual work...]
5. `harness_complete_and_next` → Mark complete, checkpoint, get next feature
6. Repeat steps 3-5 until all features done

### Workflow Tools (reduce API calls):
- `harness_work_on_feature`: Combines iterate + set_current + log (use instead of separate calls)
- `harness_complete_and_next`: Combines complete + checkpoint + get_next (use instead of separate calls)
- `harness_quick_status`: Minimal status for rapid polling (use instead of full harness_status)

### Human Intervention Points (Phase 5):
- `harness_request_intervention`: Request human review, approval, or decision
- `harness_resolve_intervention`: Provide resolution to pending intervention
- Types: `review_required` (non-blocking), `approval_needed`, `decision_point`, `clarification_needed`
- Blocking interventions (approval_needed, decision_point) prevent work until resolved

### Sub-Agent Delegation (Phase 6):
- `harness_delegate`: Delegate token-heavy work to isolated sub-session
- `harness_sub_session_status`: Check sub-session progress (read-only)
- `harness_claim_sub_session_result`: Claim completed sub-session work
- `harness_complete_sub_session`: Sub-agent marks work done
- `harness_list_sub_sessions`: List active sub-sessions (read-only)
- Use for: code generation, test writing, documentation, refactoring tasks

### Context Management:
- `harness_compact_progress`: Summarize old progress entries when context grows large
- Use `max_progress_entries` parameter in `harness_status` to limit response size

### Safety Notes:
- `harness_status`, `harness_quick_status`, `harness_sub_session_status`, `harness_list_sub_sessions`: Read-only, safe to call anytime
- `harness_rollback`: DESTRUCTIVE - rolls back git state, use with caution
- `harness_checkpoint`: Creates git commits, safe but irreversible

### Error Recovery:
All errors include structured recovery actions. Parse the `recovery_action` field for actionable next steps.";

        let instructions = if self.harness.is_some() {
            format!("{}{}", base_instructions, harness_instructions)
        } else {
            base_instructions.to_string()
        };

        ServerInfo {
            instructions: Some(instructions),
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            ..Default::default()
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse CLI arguments
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("rust_cluster_mcp=info".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .init();

    // Initialize harness state if enabled
    let harness_state = if args.harness {
        tracing::info!("Starting Rust Cluster MCP Server with Harness Mode");

        // Build harness config from args and env
        let mut config = harness::HarnessConfig::from_env();

        if let Some(max) = args.max_iterations {
            config.max_iterations = max;
        }
        if let Some(path) = args.features_path {
            config.features_path = path;
        }
        if let Some(path) = args.progress_path {
            config.progress_path = path;
        }
        config.require_clean_git = args.require_clean_git;
        config.resolve_paths();

        tracing::info!(
            "Harness config: features={}, progress={}, max_iterations={}",
            config.features_path.display(),
            config.progress_path.display(),
            config.max_iterations
        );

        // Initialize harness state
        Some(harness::create_shared_state(config))
    } else {
        tracing::info!("Starting Rust Cluster MCP Server");
        None
    };

    // Initialize ensemble state if enabled
    let ensemble_state = if args.ensemble {
        tracing::info!("Initializing Ensemble Mode");

        let state_path = args.state_path.unwrap_or_else(|| {
            std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."))
                .join(".ensemble-state")
        });

        tracing::info!("Ensemble state path: {}", state_path.display());

        // Create state store
        let store = state::StateStore::open(&state_path)
            .map_err(|e| anyhow::anyhow!("Failed to open state store: {}", e))?
            .shared();

        // Create event bus with persistence
        let bus = events::EventBus::with_persistence(store.clone()).shared();

        // Create ensemble config
        let config = ensemble::EnsembleConfig::default();

        // Create coordinator
        let coordinator = ensemble::EnsembleCoordinator::new(store.clone(), bus.clone(), config)
            .map_err(|e| anyhow::anyhow!("Failed to create ensemble coordinator: {}", e))?
            .shared();

        Some((coordinator, store, bus))
    } else {
        None
    };

    // Create server, optionally with harness and/or ensemble state
    let mut server = RustClusterServer::new()
        .map_err(|e| anyhow::anyhow!("Failed to create HTTP client: {}", e))?;

    if let Some(state) = harness_state {
        tracing::info!("Harness tools enabled via MCP");
        server = server.with_harness(state);
    }

    if let Some((coordinator, store, bus)) = ensemble_state {
        tracing::info!("Ensemble tools enabled via MCP");
        server = server.with_ensemble(coordinator, store, bus);
    }

    let transport = (stdin(), stdout());
    let service = server.serve(transport).await?;

    service.waiting().await?;

    Ok(())
}
