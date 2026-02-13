use anyhow::{Context, Result};
use rig::providers::openai;
use serde::Deserialize;
use std::path::PathBuf;

/// Inference tier for model routing.
#[derive(Debug, Clone, Deserialize)]
pub enum Tier {
    /// Fast 14B model on vasp-02 (~53 tok/s)
    Fast,
    /// General coder model (Qwen3-Coder-Next) on vasp-02
    Coder,
    /// Reasoning 72B model on vasp-01+vasp-03 (~13 tok/s)
    Reasoning,
    /// Cloud models via CLIAPIProxy
    Cloud,
}

/// Cluster inference endpoint configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Endpoint {
    pub url: String,
    pub model: String,
    pub tier: Tier,
}

/// Cloud escalation endpoint (CLIAPIProxy on ai-proxy).
#[derive(Debug, Clone, Deserialize)]
pub struct CloudEndpoint {
    pub url: String,
    pub api_key: String,
    pub model: String,
}

/// Top-level swarm configuration.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// strand-rust-coder-14B (Rust specialist)
    pub fast_endpoint: Endpoint,
    /// Qwen3-Coder-Next (general coding, 256K context)
    pub coder_endpoint: Endpoint,
    /// OR1-Behemoth 72B (reasoning/manager)
    pub reasoning_endpoint: Endpoint,
    /// CLIAPIProxy cloud escalation (optional)
    pub cloud_endpoint: Option<CloudEndpoint>,
    /// Maximum retries per issue before giving up.
    pub max_retries: u32,
    /// Base directory for worktrees (None = auto-detect).
    pub worktree_base: Option<PathBuf>,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            fast_endpoint: Endpoint {
                url: std::env::var("SWARM_FAST_URL")
                    .unwrap_or_else(|_| "http://vasp-02:8080/v1".into()),
                model: std::env::var("SWARM_FAST_MODEL")
                    .unwrap_or_else(|_| "strand-rust-coder-14b-q8_0.gguf".into()),
                tier: Tier::Fast,
            },
            coder_endpoint: Endpoint {
                url: std::env::var("SWARM_CODER_URL")
                    .unwrap_or_else(|_| "http://vasp-02:8080/v1".into()),
                model: std::env::var("SWARM_CODER_MODEL")
                    .unwrap_or_else(|_| "Qwen3-Coder-Next-UD-Q4_K_XL.gguf".into()),
                tier: Tier::Coder,
            },
            reasoning_endpoint: Endpoint {
                url: std::env::var("SWARM_REASONING_URL")
                    .unwrap_or_else(|_| "http://vasp-01:8081/v1".into()),
                model: std::env::var("SWARM_REASONING_MODEL")
                    .unwrap_or_else(|_| "or1-behemoth-q4_k_m.gguf".into()),
                tier: Tier::Reasoning,
            },
            cloud_endpoint: Self::cloud_from_env(),
            max_retries: 6,
            worktree_base: None,
        }
    }
}

impl SwarmConfig {
    fn cloud_from_env() -> Option<CloudEndpoint> {
        let url = std::env::var("SWARM_CLOUD_URL").ok()?;
        let api_key = std::env::var("SWARM_CLOUD_API_KEY").ok()?;
        let model =
            std::env::var("SWARM_CLOUD_MODEL").unwrap_or_else(|_| "claude-sonnet-4-5".into());
        Some(CloudEndpoint {
            url,
            api_key,
            model,
        })
    }
}

/// Pre-built rig CompletionsClients, deduplicated by endpoint URL.
///
/// vasp-02:8080 serves both strand-14B and Qwen3-Coder-Next, so one
/// client handles both — model selection happens via the model name in
/// the request JSON.
pub struct ClientSet {
    /// Client for vasp-02:8080 (serves Fast + Coder models)
    pub local: openai::CompletionsClient,
    /// Client for vasp-01:8081 (serves Reasoning model)
    pub reasoning: openai::CompletionsClient,
    /// Client for ai-proxy:8317 (cloud escalation, optional)
    pub cloud: Option<openai::CompletionsClient>,
}

impl ClientSet {
    pub fn from_config(config: &SwarmConfig) -> Result<Self> {
        let local = openai::CompletionsClient::builder()
            .api_key("not-needed")
            .base_url(&config.fast_endpoint.url)
            .build()
            .context("Failed to build local client (vasp-02)")?;

        let reasoning = openai::CompletionsClient::builder()
            .api_key("not-needed")
            .base_url(&config.reasoning_endpoint.url)
            .build()
            .context("Failed to build reasoning client (vasp-01)")?;

        let cloud = if let Some(ref ce) = config.cloud_endpoint {
            Some(
                openai::CompletionsClient::builder()
                    .api_key(&ce.api_key)
                    .base_url(&ce.url)
                    .build()
                    .context("Failed to build cloud client (ai-proxy)")?,
            )
        } else {
            None
        };

        Ok(Self {
            local,
            reasoning,
            cloud,
        })
    }
}

/// Check if an inference endpoint is reachable (GET /health or /v1/models).
pub async fn check_endpoint(url: &str) -> bool {
    let models_url = format!("{url}/models");
    match reqwest::Client::new()
        .get(&models_url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}
