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
    /// HydraCoder (Rust specialist)
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
    /// Path to the notebook registry TOML file.
    pub notebook_registry_path: Option<PathBuf>,
    /// Scope verifier to specific cargo packages.
    /// When empty, targets the entire workspace.
    /// Populated from `--package` CLI flag or `SWARM_VERIFIER_PACKAGES` env var (comma-separated).
    pub verifier_packages: Vec<String>,
    /// Maximum retries for cloud (CLIAPIProxy) HTTP calls.
    /// Exponential backoff: 2s, 4s, 8s, ...
    /// Populated from `SWARM_CLOUD_MAX_RETRIES` env var (default: 3).
    pub cloud_max_retries: u32,
    /// Cloud-only mode: skip local endpoint health checks, route all work through cloud.
    /// Requires `cloud_endpoint` to be configured.
    /// Populated from `--cloud-only` CLI flag or `SWARM_CLOUD_ONLY=1` env var.
    pub cloud_only: bool,
    /// Maximum consecutive no-change iterations before circuit breaker fires.
    /// When an agent responds without producing any file changes for this many
    /// iterations in a row, the loop breaks and creates a stuck intervention.
    /// Populated from `SWARM_MAX_NO_CHANGE` env var (default: 3).
    pub max_consecutive_no_change: u32,
    /// Minimum objective length (characters) to accept for processing.
    /// Issues with titles shorter than this are rejected before worktree creation.
    /// Populated from `SWARM_MIN_OBJECTIVE_LEN` env var (default: 10).
    pub min_objective_len: usize,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            fast_endpoint: Endpoint {
                url: std::env::var("SWARM_FAST_URL")
                    .unwrap_or_else(|_| "http://vasp-02:8080/v1".into()),
                model: std::env::var("SWARM_FAST_MODEL")
                    .unwrap_or_else(|_| "HydraCoder.i1-Q4_K_M".into()),
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
            max_retries: std::env::var("SWARM_MAX_RETRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|v| *v > 0)
                .unwrap_or(10),
            worktree_base: None,
            notebook_registry_path: Some(PathBuf::from("./notebook_registry.toml")),
            verifier_packages: std::env::var("SWARM_VERIFIER_PACKAGES")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| s.split(',').map(|p| p.trim().to_string()).collect())
                .unwrap_or_default(),
            cloud_max_retries: std::env::var("SWARM_CLOUD_MAX_RETRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
            cloud_only: std::env::var("SWARM_CLOUD_ONLY")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            max_consecutive_no_change: std::env::var("SWARM_MAX_NO_CHANGE")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|v| *v > 0)
                .unwrap_or(3),
            min_objective_len: std::env::var("SWARM_MIN_OBJECTIVE_LEN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
        }
    }
}

impl SwarmConfig {
    fn cloud_from_env() -> Option<CloudEndpoint> {
        let url = std::env::var("SWARM_CLOUD_URL").ok()?;
        let api_key = std::env::var("SWARM_CLOUD_API_KEY").ok()?;
        let model = std::env::var("SWARM_CLOUD_MODEL")
            .unwrap_or_else(|_| "claude-opus-4-6-thinking".into());
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
                model: "claude-opus-4-6-thinking".into(),
                tier: Tier::Reasoning,
                api_key: proxy_key.clone(),
            },
            cloud_endpoint: Some(CloudEndpoint {
                url: proxy_url,
                api_key: proxy_key,
                model: "claude-opus-4-6-thinking".into(),
            }),
            max_retries: 3,
            worktree_base: None,
            notebook_registry_path: None,
            verifier_packages: Vec::new(),
            cloud_max_retries: 3,
            cloud_only: false,
            max_consecutive_no_change: 3,
            min_objective_len: 10,
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
        // In cloud-only mode, reuse the cloud client for all tiers
        if config.cloud_only {
            let cloud_ep = config
                .cloud_endpoint
                .as_ref()
                .context("cloud_only requires cloud_endpoint to be configured")?;
            let cloud = openai::CompletionsClient::builder()
                .api_key(&cloud_ep.api_key)
                .base_url(&cloud_ep.url)
                .build()
                .context("Failed to build cloud client")?;
            // Clone the client for local and reasoning slots
            let local = openai::CompletionsClient::builder()
                .api_key(&cloud_ep.api_key)
                .base_url(&cloud_ep.url)
                .build()
                .context("Failed to build cloud-as-local client")?;
            let reasoning = openai::CompletionsClient::builder()
                .api_key(&cloud_ep.api_key)
                .base_url(&cloud_ep.url)
                .build()
                .context("Failed to build cloud-as-reasoning client")?;
            return Ok(Self {
                local,
                reasoning,
                cloud: Some(cloud),
            });
        }

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

/// Check if an inference endpoint is reachable and has a model loaded.
///
/// Queries `GET /v1/models` and optionally verifies that `expected_model` is in
/// the response. Returns `true` only if the endpoint responds and the model check
/// passes.
///
/// If `api_key` is provided (and not `"not-needed"`), sends a Bearer auth header.
pub async fn check_endpoint(url: &str, api_key: Option<&str>) -> bool {
    check_endpoint_with_model(url, api_key, None).await
}

/// Like [`check_endpoint`] but also verifies a specific model is loaded.
pub async fn check_endpoint_with_model(
    url: &str,
    api_key: Option<&str>,
    expected_model: Option<&str>,
) -> bool {
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
        Ok(resp) if resp.status().is_success() => {
            // If no model check requested, just return reachable
            let Some(expected) = expected_model else {
                return true;
            };

            // Parse the models list to verify expected model is loaded
            if let Ok(body) = resp.json::<serde_json::Value>().await {
                let has_model = body["data"]
                    .as_array()
                    .map(|models| {
                        models.iter().any(|m| {
                            m["id"]
                                .as_str()
                                .map(|id| id.contains(expected))
                                .unwrap_or(false)
                        })
                    })
                    .unwrap_or(false);

                if !has_model {
                    tracing::warn!(
                        endpoint = url,
                        expected_model = expected,
                        "Endpoint reachable but expected model not loaded"
                    );
                }
                has_model
            } else {
                // Couldn't parse body but endpoint is reachable
                true
            }
        }
        Ok(resp) => {
            tracing::warn!(
                endpoint = url,
                status = %resp.status(),
                "Endpoint returned non-success status"
            );
            false
        }
        Err(e) => {
            tracing::warn!(
                endpoint = url,
                error = %e,
                "Endpoint unreachable"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        // Unset the environment variable so we test the default value
        std::env::remove_var("SWARM_MAX_RETRIES");
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
