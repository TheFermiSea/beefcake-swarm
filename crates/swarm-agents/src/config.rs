use serde::Deserialize;
use std::path::PathBuf;

/// Inference tier for model routing.
#[derive(Debug, Clone, Deserialize)]
pub enum Tier {
    /// Fast 14B model on vasp-02 (~53 tok/s)
    Fast,
    /// Reasoning 72B model on vasp-01+vasp-03 (~13 tok/s)
    Reasoning,
}

/// Cluster inference endpoint configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Endpoint {
    pub url: String,
    pub model: String,
    pub tier: Tier,
}

/// Top-level swarm configuration.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    pub fast_endpoint: Endpoint,
    pub reasoning_endpoint: Endpoint,
    /// Maximum retries per issue before giving up.
    pub max_retries: u32,
    /// Base directory for worktrees (None = auto-detect).
    pub worktree_base: Option<PathBuf>,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            fast_endpoint: Endpoint {
                url: "http://vasp-02:8080/v1".to_string(),
                model: "strand-rust-coder-14b-q8_0".to_string(),
                tier: Tier::Fast,
            },
            reasoning_endpoint: Endpoint {
                url: "http://vasp-01:8081/v1".to_string(),
                model: "or1-behemoth-q4_k_m".to_string(),
                tier: Tier::Reasoning,
            },
            max_retries: 3,
            worktree_base: None,
        }
    }
}
