//! Cloud Council Adapter for AI Escalation
//!
//! Provides abstraction for escalating complex problems to cloud-based AI models
//! (Gemini 3 Pro, Claude Opus 4.5, GPT-5.2) when local models can't solve them.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;
use thiserror::Error;

/// Errors from Cloud Council operations
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
    /// Manager: Task decomposition and project management (GPT-5.2)
    Manager,
}

impl CouncilRole {
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Librarian => "gemini-3-pro",
            Self::Architect => "claude-opus-4-5",
            Self::Manager => "gpt-5.2",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            Self::Librarian => "Repository context, code understanding, documentation",
            Self::Architect => "Design decisions, safety contracts, architecture review",
            Self::Manager => "Task decomposition, project management, coordination",
        }
    }
}

impl std::fmt::Display for CouncilRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Librarian => write!(f, "librarian"),
            Self::Architect => write!(f, "architect"),
            Self::Manager => write!(f, "manager"),
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

/// Decision from the Cloud Council
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

/// Configuration for the Cloud Council
#[derive(Debug, Clone)]
pub struct CouncilConfig {
    /// API keys for each provider
    pub api_keys: HashMap<String, String>,
    /// Request timeout
    pub timeout: Duration,
    /// Maximum retries per member
    pub max_retries: u32,
    /// Whether to require consensus
    pub require_consensus: bool,
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
        if let Ok(key) = std::env::var("OPENAI_API_KEY") {
            api_keys.insert("openai".to_string(), key);
        }

        Self {
            api_keys,
            timeout: Duration::from_secs(120),
            max_retries: 2,
            require_consensus: false,
        }
    }
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

        let system_prompt = r#"You are the Librarian of a development council, specializing in:
- Repository structure and code navigation
- Understanding existing patterns and conventions
- Documentation and API references
- Finding relevant code examples

Given an escalated coding problem, provide context and insights about:
1. Relevant files and code patterns
2. Similar problems solved elsewhere in the codebase
3. Documentation that might help
4. Suggested approach based on codebase conventions"#;

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
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={}",
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

        let content = resp_json["candidates"][0]["content"]["parts"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(CouncilResponse {
            content,
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

        let system_prompt = r#"You are the Architect of a development council, specializing in:
- System design and architecture decisions
- Safety contracts and invariants
- Code review and quality assessment
- Rust-specific patterns (ownership, lifetimes, async)

Given an escalated coding problem, provide:
1. Architectural analysis of the issue
2. Recommended design approach
3. Safety considerations and invariants to maintain
4. Specific code fix or implementation guidance"#;

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
            "model": "claude-sonnet-4-20250514",
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

        let content = resp_json["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(CouncilResponse {
            content,
            confidence: 0.9,
            model: "claude-sonnet-4".to_string(), // Actual model used in API call
            tokens_used: 0,
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}

/// GPT-based Manager council member
pub struct GptManager {
    api_key: String,
    client: reqwest::Client,
}

impl GptManager {
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
impl CouncilMember for GptManager {
    fn role(&self) -> CouncilRole {
        CouncilRole::Manager
    }

    async fn query(&self, context: &EscalationContext) -> Result<CouncilResponse, CouncilError> {
        let start = std::time::Instant::now();

        let system_prompt = r#"You are the Manager of a development council, specializing in:
- Task decomposition and planning
- Project coordination
- Resource allocation
- Process optimization

Given an escalated coding problem, provide:
1. Step-by-step action plan
2. Task decomposition if the problem is too large
3. Priority ordering of fixes
4. Coordination advice for multi-file changes"#;

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
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 2048,
            "temperature": 0.3
        });

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await
            .map_err(|e| CouncilError::RequestFailed(e.to_string()))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            return Err(CouncilError::RequestFailed(format!(
                "OpenAI API error ({}): {}",
                status, body
            )));
        }

        let resp_json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| CouncilError::ParseError(e.to_string()))?;

        let content = resp_json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        Ok(CouncilResponse {
            content,
            confidence: 0.85,
            model: "gpt-4o".to_string(), // Actual model used in API call
            tokens_used: 0,
            response_time_ms: start.elapsed().as_millis() as u64,
        })
    }

    async fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }
}

/// The Cloud Council - coordinates multiple AI models for complex decisions
pub struct CloudCouncil {
    librarian: Option<Box<dyn CouncilMember>>,
    architect: Option<Box<dyn CouncilMember>>,
    manager: Option<Box<dyn CouncilMember>>,
    config: CouncilConfig,
}

impl CloudCouncil {
    /// Create a new Cloud Council from configuration
    pub fn from_config(config: CouncilConfig) -> Self {
        let librarian = config
            .api_keys
            .get("gemini")
            .map(|key| Box::new(GeminiLibrarian::new(key.clone())) as Box<dyn CouncilMember>);

        let architect = config
            .api_keys
            .get("anthropic")
            .map(|key| Box::new(ClaudeArchitect::new(key.clone())) as Box<dyn CouncilMember>);

        let manager = config
            .api_keys
            .get("openai")
            .map(|key| Box::new(GptManager::new(key.clone())) as Box<dyn CouncilMember>);

        Self {
            librarian,
            architect,
            manager,
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
        if let Some(ref member) = self.manager {
            if member.is_available().await {
                available.push(CouncilRole::Manager);
            }
        }

        available
    }

    /// Escalate a problem to the council
    pub async fn escalate(
        &self,
        context: &EscalationContext,
    ) -> Result<CouncilDecision, CouncilError> {
        let mut responses = Vec::new();
        let mut contributors = Vec::new();

        // Query architect first (most relevant for code fixes)
        if let Some(ref architect) = self.architect {
            if architect.is_available().await {
                match architect.query(context).await {
                    Ok(resp) => {
                        responses.push(resp);
                        contributors.push(CouncilRole::Architect);
                    }
                    Err(e) => tracing::warn!("Architect query failed: {}", e),
                }
            }
        }

        // Query librarian for context
        if let Some(ref librarian) = self.librarian {
            if librarian.is_available().await {
                match librarian.query(context).await {
                    Ok(resp) => {
                        responses.push(resp);
                        contributors.push(CouncilRole::Librarian);
                    }
                    Err(e) => tracing::warn!("Librarian query failed: {}", e),
                }
            }
        }

        // Query manager for coordination
        if let Some(ref manager) = self.manager {
            if manager.is_available().await {
                match manager.query(context).await {
                    Ok(resp) => {
                        responses.push(resp);
                        contributors.push(CouncilRole::Manager);
                    }
                    Err(e) => tracing::warn!("Manager query failed: {}", e),
                }
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
            CouncilRole::Manager => {
                self.manager.as_ref().map(|m| m.as_ref()).ok_or_else(|| {
                    CouncilError::Unavailable("Manager not configured".to_string())
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
        // Weight architect highest for code fixes
        let primary = responses
            .iter()
            .zip(contributors.iter())
            .find(|(_, role)| **role == CouncilRole::Architect)
            .map(|(r, _)| r)
            .unwrap_or(&responses[0]);

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

impl Default for CloudCouncil {
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
        assert_eq!(CouncilRole::Manager.to_string(), "manager");
    }

    #[test]
    fn test_council_config_default() {
        let config = CouncilConfig::default();
        assert_eq!(config.timeout, Duration::from_secs(120));
        assert_eq!(config.max_retries, 2);
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
