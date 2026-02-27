//! Manager Council for AI Escalation
//!
//! Provides abstraction for escalating complex problems to peer manager models
//! (Gemini 3 Pro, Claude Opus 4.5, Qwen3.5) using concurrent queries and delegation.
//!
//! # Design Note: Intentional LLM calls
//!
//! The `coordination/` crate is deterministic by design — no LLM calls in routing,
//! escalation, feedback, verifier, or ensemble modules.
//!
//! `council/` is the **sole intentional exception**: it is the cloud AI escalation
//! adapter, whose entire purpose is to delegate to external model providers when
//! local tiers are exhausted. Network I/O here is load-bearing, not accidental.
//! All other modules in `coordination/` remain LLM-free.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

/// Errors from Manager Council operations
#[derive(Debug, Error)]
pub enum CouncilError {
    #[error("API request failed: {0}")]
    RequestFailed(String),

    #[error("API key not configured for {0}")]
    MissingApiKey(String),

    #[error("Response parse error: {0}")]
    ParseError(String),

    #[error("Rate limited: retry after {0:?}")]
    RateLimited(Duration),

    #[error("Council member unavailable: {0}")]
    Unavailable(String),

    #[error("No consensus reached after {0} attempts")]
    NoConsensus(u32),
}

/// Role of each council member
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CouncilRole {
    /// Librarian: Repository context and code understanding (Gemini 3 Pro)
    Librarian,
    /// Architect: Design decisions and safety contracts (Claude Opus 4.5)
    Architect,
    /// Strategist: Reasoning, planning, task decomposition (Qwen3.5)
    Strategist,
}

impl CouncilRole {
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Librarian => "gemini-3-pro",
            Self::Architect => "claude-opus-4-5",
            Self::Strategist => "qwen3.5-397b",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Librarian => "Repository context, code understanding, documentation",
            Self::Architect => "Design decisions, safety contracts, architecture review",
            Self::Strategist => "Reasoning, planning, task decomposition, strategy",
        }
    }
}

impl std::fmt::Display for CouncilRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Librarian => write!(f, "librarian"),
            Self::Architect => write!(f, "architect"),
            Self::Strategist => write!(f, "strategist"),
        }
    }
}

/// Context for escalation requests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationContext {
    /// The original task or question
    pub task: String,
    /// Code context (files, snippets)
    pub code_context: Option<String>,
    /// Error history from local model attempts
    pub error_history: Vec<ErrorAttempt>,
    /// Why escalation was triggered
    pub escalation_reason: EscalationReason,
    /// Specific constraints or requirements
    pub constraints: Vec<String>,
}

/// Record of a failed local model attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorAttempt {
    /// Which model was used
    pub model: String,
    /// What the model tried
    pub attempted_fix: String,
    /// The resulting error
    pub error: String,
    /// Attempt number at this tier
    pub attempt: u32,
}

/// Why escalation was triggered
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationReason {
    /// No progress after max attempts at tier
    NoProgress { tier: String, attempts: u32 },
    /// Environment/linker error (not a code fix)
    EnvironmentError { error_type: String },
    /// Timeout exceeded
    Timeout { wall_clock_minutes: u32 },
    /// Explicit request from user/system
    ExplicitRequest { requester: String },
    /// Complex architectural decision needed
    ArchitecturalDecision { description: String },
}

/// Decision from the Manager Council
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilDecision {
    /// The recommended action or fix
    pub recommendation: String,
    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,
    /// Which council member(s) contributed
    pub contributors: Vec<CouncilRole>,
    /// Detailed rationale
    pub rationale: String,
    /// Alternative approaches considered
    pub alternatives: Vec<String>,
    /// Warnings or caveats
    pub warnings: Vec<String>,
}

/// Trait for council members (individual AI models)
#[async_trait]
pub trait CouncilMember: Send + Sync {
    /// Get the role of this council member
    fn role(&self) -> CouncilRole;

    /// Query this council member
    async fn query(&self, context: &EscalationContext) -> Result<CouncilResponse, CouncilError>;

    /// Check if this member is available
    async fn is_available(&self) -> bool;
}

/// Response from a single council member
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouncilResponse {
    /// The response content
    pub content: String,
    /// Confidence in this response
    pub confidence: f32,
    /// Model used
    pub model: String,
    /// Token usage
    pub tokens_used: u32,
    /// Response time in ms
    pub response_time_ms: u64,
}

/// Configuration for the Manager Council
#[derive(Debug, Clone)]
pub struct CouncilConfig {
    /// API keys for each provider
    pub api_keys: HashMap<String, String>,
    /// Qwen3.5 local endpoint (OpenAI-compatible)
    pub qwen35_endpoint: String,
    /// Request timeout
    pub timeout: Duration,
    /// Maximum retries per member
    pub max_retries: u32,
    /// Whether to require consensus
    pub require_consensus: bool,
    /// Minimum number of members that must respond for a valid decision
    pub min_quorum: usize,
}

impl Default for CouncilConfig {
    fn default() -> Self {
        let mut api_keys = HashMap::new();

        // Load from environment
        if let Ok(key) = std::env::var("GEMINI_API_KEY") {
            api_keys.insert("gemini".to_string(), key);
        }
        if let Ok(key) = std::env::var("ANTHROPIC_API_KEY") {
            api_keys.insert("anthropic".to_string(), key);
        }

        let qwen35_endpoint = std::env::var("QWEN35_ENDPOINT")
            .unwrap_or_else(|_| "http://vasp-01:8081/v1/chat/completions".to_string());

        Self {
            api_keys,
            qwen35_endpoint,
            timeout: Duration::from_secs(300),
            max_retries: 2,
            require_consensus: false,
            min_quorum: 2,
        }
    }
}

/// Request to delegate work to another manager
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationRequest {
    pub from: crate::state::types::ModelId,
    pub to: Option<Vec<crate::state::types::ModelId>>,
    pub context: EscalationContext,
    pub reason: DelegationReason,
}

/// Reason for delegation between managers
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelegationReason {
    LowConfidence { confidence: f32, threshold: f32 },
    SecondOpinion,
    Specialization { area: String },
    LoadBalance,
}

/// Gemini-based Librarian council member
pub struct GeminiLibrarian {
    api_key: String,
    client: reqwest::Client,
}

impl GeminiLibrarian {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }
}

#[async_trait]
impl CouncilMember for GeminiLibrarian {
    fn role(&self) -> CouncilRole {
        CouncilRole::Librarian
    }

    async fn query(&self, context: &EscalationContext) -> Result<CouncilResponse, CouncilError> {
        let start = std::time::Instant::now();

        let system_prompt = r"You are the Librarian of a development council, specializing in:
- Repository structure and code navigation
- Understanding existing patterns and conventions
- Documentation and API references
- Finding relevant code examples

Given an escalated coding problem, provide context and insights about:
1. Relevant files and code patterns
2. Similar problems solved elsewhere in the codebase
3. Documentation that might help
4. Suggested approach based on codebase conventions";

        let user_prompt = format!(
            "## Escalated Problem\n\n{}\n\n## Escalation Reason\n\n{:?}\n\n## Previous Attempts\n\n{}\n\n## Code Context\n\n{}",
            context.task,
            context.escalation_reason,
            context.error_history.iter()
                .map(|e| format!("- {} (attempt {}): {}", e.model, e.attempt, e.error))
                .collect::<Vec<_>>()
                .join("\n"),
            context.code_context.as_deref().unwrap_or("No code context provided")
        );

        // Gemini API call
        let request_body = serde_json::json!({
            "contents": [{
                "parts": [{
                    "text": format!("{}\n\n{}", system_prompt, user_prompt)
                }]
            }],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 2048
            }
        });

        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro:generateContent?key={}",
            self.api_key
        );

        let response = self
            .client
            .post(&url)
            .json(&request_body)
            .send()
            .await
            .map_err(|e| CouncilError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CouncilError::RequestFailed(format!(
                "Gemini API error ({}): {}",
                status, body
            )));
        }

        let resp_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CouncilError::ParseError(e.to_string()))?;

        let response_text = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(CouncilResponse {
            content: response_text,
            confidence: 0.8,
            model: "gemini-3-pro".to_string(),
            tokens_used: 0, // Would parse from response
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}

/// Claude-based Architect council member
pub struct ClaudeArchitect {
    api_key: String,
    client: reqwest::Client,
}

impl ClaudeArchitect {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(120))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }
}

#[async_trait]
impl CouncilMember for ClaudeArchitect {
    fn role(&self) -> CouncilRole {
        CouncilRole::Architect
    }

    async fn query(&self, context: &EscalationContext) -> Result<CouncilResponse, CouncilError> {
        let start = std::time::Instant::now();

        let system_prompt = r"You are the Architect of a development council, specializing in:
- System design and architecture decisions
- Safety contracts and invariants
- Code review and quality assessment
- Rust-specific patterns (ownership, lifetimes, async)

Given an escalated coding problem, provide:
1. Architectural analysis of the issue
2. Recommended design approach
3. Safety considerations and invariants to maintain
4. Specific code fix or implementation guidance";

        let user_prompt = format!(
            "## Escalated Problem\n\n{}\n\n## Escalation Reason\n\n{:?}\n\n## Previous Attempts\n\n{}\n\n## Code Context\n\n{}",
            context.task,
            context.escalation_reason,
            context.error_history.iter()
                .map(|e| format!("- {} (attempt {}): {}", e.model, e.attempt, e.error))
                .collect::<Vec<_>>()
                .join("\n"),
            context.code_context.as_deref().unwrap_or("No code context provided")
        );

        let request_body = serde_json::json!({
            "model": "claude-opus-4-5-20250514",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [{
                "role": "user",
                "content": user_prompt
            }]
        });

        let response = self
            .client
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| CouncilError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CouncilError::RequestFailed(format!(
                "Claude API error ({}): {}",
                status, body
            )));
        }

        let resp_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CouncilError::ParseError(e.to_string()))?;

        let response_text = resp_json["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(CouncilResponse {
            content: response_text,
            confidence: 0.9,
            model: "claude-opus-4-5".to_string(), // Actual model used in API call
            tokens_used: 0,
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}

/// Qwen3.5-based Strategist council member (local model via OpenAI-compatible API)
pub struct Qwen35Strategist {
    endpoint: String,
    client: reqwest::Client,
}

impl Qwen35Strategist {
    pub fn new(endpoint: String) -> Self {
        Self {
            endpoint,
            client: reqwest::Client::builder()
                .timeout(Duration::from_secs(300))
                .build()
                .expect("Failed to create HTTP client"),
        }
    }
}

#[async_trait]
impl CouncilMember for Qwen35Strategist {
    fn role(&self) -> CouncilRole {
        CouncilRole::Strategist
    }

    async fn query(&self, context: &EscalationContext) -> Result<CouncilResponse, CouncilError> {
        let start = std::time::Instant::now();

        let system_prompt = r"You are the Strategist of a development council, specializing in:
- Deep reasoning about complex coding problems
- Strategic planning and task decomposition
- Breaking down large problems into actionable steps
- Identifying root causes and systemic issues

Given an escalated coding problem, provide:
1. Root cause analysis with reasoning chain
2. Strategic task decomposition if the problem is too large
3. Priority ordering of fixes with rationale
4. Coordination advice for multi-file changes";

        let user_prompt = format!(
            "## Escalated Problem\n\n{}\n\n## Escalation Reason\n\n{:?}\n\n## Previous Attempts\n\n{}\n\n## Code Context\n\n{}",
            context.task,
            context.escalation_reason,
            context.error_history.iter()
                .map(|e| format!("- {} (attempt {}): {}", e.model, e.attempt, e.error))
                .collect::<Vec<_>>()
                .join("\n"),
            context.code_context.as_deref().unwrap_or("No code context provided")
        );

        let request_body = serde_json::json!({
            "model": "Qwen3.5-397B-A17B",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 2048,
            "temperature": 0.3
        });

        let response = self
            .client
            .post(&self.endpoint)
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| CouncilError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CouncilError::RequestFailed(format!(
                "Qwen3.5 API error ({}): {}",
                status, body
            )));
        }

        let resp_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CouncilError::ParseError(e.to_string()))?;

        let response_text = resp_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(CouncilResponse {
            content: response_text,
            confidence: 0.85,
            model: "qwen3.5-397b".to_string(),
            tokens_used: 0,
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn is_available(&self) -> bool {
        // Try HTTP GET to base URL health endpoint
        let base_url = self
            .endpoint
            .trim_end_matches("/chat/completions")
            .trim_end_matches("/v1");
        let health_url = format!("{}/health", base_url);
        if let Ok(resp) = self.client.get(&health_url).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        // Fall back to checking endpoint is configured
        !self.endpoint.is_empty()
    }
}

/// The Manager Council - coordinates peer manager models for complex decisions
pub struct ManagerCouncil {
    librarian: Option<Box<dyn CouncilMember>>,
    architect: Option<Box<dyn CouncilMember>>,
    strategist: Option<Box<dyn CouncilMember>>,
    config: CouncilConfig,
}

impl ManagerCouncil {
    /// Create a new Manager Council from configuration
    pub fn from_config(config: CouncilConfig) -> Self {
        let librarian = config
            .api_keys
            .get("gemini")
            .map(|key| Box::new(GeminiLibrarian::new(key.clone())) as Box<dyn CouncilMember>);

        let architect = config
            .api_keys
            .get("anthropic")
            .map(|key| Box::new(ClaudeArchitect::new(key.clone())) as Box<dyn CouncilMember>);

        // Qwen3.5 is a local model — no API key needed
        let strategist = Some(
            Box::new(Qwen35Strategist::new(config.qwen35_endpoint.clone()))
                as Box<dyn CouncilMember>,
        );

        Self {
            librarian,
            architect,
            strategist,
            config,
        }
    }

    /// Create with default configuration (from environment)
    pub fn new() -> Self {
        Self::from_config(CouncilConfig::default())
    }

    /// Check which council members are available
    pub async fn available_members(&self) -> Vec<CouncilRole> {
        let mut available = Vec::new();

        if let Some(ref member) = self.librarian {
            if member.is_available().await {
                available.push(CouncilRole::Librarian);
            }
        }
        if let Some(ref member) = self.architect {
            if member.is_available().await {
                available.push(CouncilRole::Architect);
            }
        }
        if let Some(ref member) = self.strategist {
            if member.is_available().await {
                available.push(CouncilRole::Strategist);
            }
        }

        available
    }

    /// Escalate a problem to the council — queries all members concurrently
    pub async fn escalate(
        &self,
        context: &EscalationContext,
    ) -> Result<CouncilDecision, CouncilError> {
        use futures::future::join_all;

        // Collect available members for concurrent querying
        let members: Vec<(&dyn CouncilMember, CouncilRole)> = [
            self.architect
                .as_ref()
                .map(|m| (m.as_ref(), CouncilRole::Architect)),
            self.librarian
                .as_ref()
                .map(|m| (m.as_ref(), CouncilRole::Librarian)),
            self.strategist
                .as_ref()
                .map(|m| (m.as_ref(), CouncilRole::Strategist)),
        ]
        .into_iter()
        .flatten()
        .collect();

        // Query all members concurrently
        let futures: Vec<_> = members
            .iter()
            .map(|(member, role)| {
                let role = *role;
                async move { (role, member.query(context).await) }
            })
            .collect();

        let results = join_all(futures).await;

        let mut responses = Vec::new();
        let mut contributors = Vec::new();

        for (role, result) in results {
            match result {
                Ok(resp) => {
                    responses.push(resp);
                    contributors.push(role);
                }
                Err(e) => tracing::warn!("{} query failed: {}", role, e),
            }
        }

        if responses.is_empty() {
            return Err(CouncilError::Unavailable(
                "No council members available".to_string(),
            ));
        }

        // Synthesize decision from responses
        self.synthesize_decision(responses, contributors)
    }

    /// Query a specific council member
    pub async fn query_member(
        &self,
        role: CouncilRole,
        context: &EscalationContext,
    ) -> Result<CouncilResponse, CouncilError> {
        let member: &dyn CouncilMember = match role {
            CouncilRole::Librarian => {
                self.librarian.as_ref().map(|m| m.as_ref()).ok_or_else(|| {
                    CouncilError::Unavailable("Librarian not configured".to_string())
                })?
            }
            CouncilRole::Architect => {
                self.architect.as_ref().map(|m| m.as_ref()).ok_or_else(|| {
                    CouncilError::Unavailable("Architect not configured".to_string())
                })?
            }
            CouncilRole::Strategist => {
                self.strategist
                    .as_ref()
                    .map(|m| m.as_ref())
                    .ok_or_else(|| {
                        CouncilError::Unavailable("Strategist not configured".to_string())
                    })?
            }
        };

        member.query(context).await
    }

    /// Synthesize a decision from multiple council responses
    fn synthesize_decision(
        &self,
        responses: Vec<CouncilResponse>,
        contributors: Vec<CouncilRole>,
    ) -> Result<CouncilDecision, CouncilError> {
        // All managers are equal weight (1.0) — use first response as primary
        let primary = &responses[0];

        let avg_confidence =
            responses.iter().map(|r| r.confidence).sum::<f32>() / responses.len() as f32;

        // Combine insights
        let mut rationale = String::new();
        for (resp, role) in responses.iter().zip(contributors.iter()) {
            rationale.push_str(&format!("\n## {} Perspective\n\n{}\n", role, resp.content));
        }

        Ok(CouncilDecision {
            recommendation: primary.content.clone(),
            confidence: avg_confidence,
            contributors,
            rationale,
            alternatives: vec![],
            warnings: vec![],
        })
    }
}

impl Default for ManagerCouncil {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_council_role_display() {
        assert_eq!(CouncilRole::Librarian.to_string(), "librarian");
        assert_eq!(CouncilRole::Architect.to_string(), "architect");
        assert_eq!(CouncilRole::Strategist.to_string(), "strategist");
    }

    #[test]
    fn test_council_config_default() {
        let config = CouncilConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(300));
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.min_quorum, 2);
        assert!(!config.qwen35_endpoint.is_empty());
    }

    #[test]
    fn test_escalation_context_serialize() {
        let ctx = EscalationContext {
            task: "Fix compilation error".to_string(),
            code_context: Some("fn main() {}".to_string()),
            error_history: vec![],
            escalation_reason: EscalationReason::NoProgress {
                tier: "fast".to_string(),
                attempts: 2,
            },
            constraints: vec![],
        };

        let json = serde_json::to_string(&ctx).unwrap();
        assert!(json.contains("Fix compilation error"));
    }
}
