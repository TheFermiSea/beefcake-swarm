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
    pub api_key: String,
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
                api_key: std::env::var("SWARM_FAST_API_KEY")
                    .unwrap_or_else(|_| "not-needed".into()),
            },
            coder_endpoint: Endpoint {
                url: std::env::var("SWARM_CODER_URL")
                    .unwrap_or_else(|_| "http://vasp-02:8080/v1".into()),
                model: std::env::var("SWARM_CODER_MODEL")
                    .unwrap_or_else(|_| "Qwen3-Coder-Next-UD-Q4_K_XL.gguf".into()),
                tier: Tier::Coder,
                api_key: std::env::var("SWARM_CODER_API_KEY")
                    .unwrap_or_else(|_| "not-needed".into()),
            },
            reasoning_endpoint: Endpoint {
                url: std::env::var("SWARM_REASONING_URL")
                    .unwrap_or_else(|_| "http://vasp-01:8081/v1".into()),
                model: std::env::var("SWARM_REASONING_MODEL")
                    .unwrap_or_else(|_| "or1-behemoth-q4_k_m.gguf".into()),
                tier: Tier::Reasoning,
                api_key: std::env::var("SWARM_REASONING_API_KEY")
                    .unwrap_or_else(|_| "not-needed".into()),
            },
            cloud_endpoint: Self::cloud_from_env(),
            max_retries: 10,
            worktree_base: None,
        }
    }
}

impl SwarmConfig {
    fn cloud_from_env() -> Option<CloudEndpoint> {
        let url = std::env::var("SWARM_CLOUD_URL").ok()?;
        let api_key = std::env::var("SWARM_CLOUD_API_KEY").ok()?;
        let model = std::env::var("SWARM_CLOUD_MODEL").unwrap_or_else(|_| "claude-opus-4-5-20251101".into());
        Some(CloudEndpoint {
            url,
            api_key,
            model,
        })
    }

    /// Configuration pointing all tiers at the local CLIAPIProxy.
    ///
    /// Used for integration tests that run against `localhost:8317`.
    pub fn proxy_config() -> Self {
        let proxy_url = "http://localhost:8317/v1".to_string();
        let proxy_key = "rust-daq-proxy-key".to_string();

        Self {
            fast_endpoint: Endpoint {
                url: proxy_url.clone(),
                model: "gemini-2.5-flash".into(),
                tier: Tier::Fast,
                api_key: proxy_key.clone(),
            },
            coder_endpoint: Endpoint {
                url: proxy_url.clone(),
                model: "claude-sonnet-4-5".into(),
                tier: Tier::Coder,
                api_key: proxy_key.clone(),
            },
            reasoning_endpoint: Endpoint {
                url: proxy_url.clone(),
                model: "claude-opus-4-5-20251101".into(),
                tier: Tier::Reasoning,
                api_key: proxy_key.clone(),
            },
            cloud_endpoint: Some(CloudEndpoint {
                url: proxy_url,
                api_key: proxy_key,
                model: "claude-opus-4-5-20251101".into(),
            }),
            max_retries: 3,
            worktree_base: None,
        }
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
    /// Client for CLIAPIProxy (cloud models: Opus 4.6, G3-Pro, etc.)
    /// Used as the Manager tier when available.
    pub cloud: Option<openai::CompletionsClient>,
}

impl ClientSet {
    pub fn from_config(config: &SwarmConfig) -> Result<Self> {
        let local = openai::CompletionsClient::builder()
            .api_key(&config.fast_endpoint.api_key)
            .base_url(&config.fast_endpoint.url)
            .build()
            .context("Failed to build local client (vasp-02)")?;

        let reasoning = openai::CompletionsClient::builder()
            .api_key(&config.reasoning_endpoint.api_key)
            .base_url(&config.reasoning_endpoint.url)
            .build()
            .context("Failed to build reasoning client (vasp-01)")?;

        let cloud = config
            .cloud_endpoint
            .as_ref()
            .map(|ep| {
                openai::CompletionsClient::builder()
                    .api_key(&ep.api_key)
                    .base_url(&ep.url)
                    .build()
            })
            .transpose()
            .context("Failed to build cloud client (CLIAPIProxy)")?;

        Ok(Self {
            local,
            reasoning,
            cloud,
        })
    }
}

/// Check if an inference endpoint is reachable (GET /v1/models).
///
/// If `api_key` is provided (and not `"not-needed"`), sends a Bearer auth header.
pub async fn check_endpoint(url: &str, api_key: Option<&str>) -> bool {
    let models_url = format!("{url}/models");
    let client = reqwest::Client::new();
    let mut req = client
        .get(&models_url)
        .timeout(std::time::Duration::from_secs(5));

    if let Some(key) = api_key {
        if key != "not-needed" {
            req = req.bearer_auth(key);
        }
    }

    match req.send().await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SwarmConfig::default();
        assert_eq!(config.max_retries, 10);
        assert!(config.fast_endpoint.url.contains("vasp-02"));
        assert!(config.reasoning_endpoint.url.contains("vasp-01"));
        assert_eq!(config.fast_endpoint.api_key, "not-needed");
    }

    #[test]
    fn test_proxy_config() {
        let config = SwarmConfig::proxy_config();
        assert_eq!(config.max_retries, 3);
        assert!(config.fast_endpoint.url.contains("localhost:8317"));
        assert!(config.coder_endpoint.url.contains("localhost:8317"));
        assert!(config.reasoning_endpoint.url.contains("localhost:8317"));
        assert_eq!(config.fast_endpoint.api_key, "rust-daq-proxy-key");
        assert!(config.cloud_endpoint.is_some());
    }

    #[test]
    fn test_client_set_from_config() {
        let config = SwarmConfig::proxy_config();
        let clients = ClientSet::from_config(&config);
        assert!(clients.is_ok());
    }
}
