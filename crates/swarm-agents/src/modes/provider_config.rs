//! NS-1.5: Provider and model runtime configuration for local vLLM and cloud fallback.
//!
//! This module extends `crate::config` with mode-specific configuration so that
//! Contextual / Deepthink / Agentic mode runners can retrieve pre-built Rig
//! clients without bespoke wiring.
//!
//! ## Precedence (highest to lowest)
//!
//! 1. Environment variable overrides (e.g. `SWARM_GENERATOR_MODEL`)
//! 2. Values in this struct
//! 3. Built-in defaults (Qwen3.5-397B on vasp-02:8081)
//!
//! ## Model roles
//!
//! | Role       | Used by                | Default                       |
//! |------------|------------------------|-------------------------------|
//! | generator  | Contextual, Agentic    | Qwen3.5-397B on vasp-02:8081  |
//! | critique   | Contextual             | Qwen3.5-397B on vasp-02:8081  |
//! | strategy   | Deepthink phase 1      | Qwen3.5-397B on vasp-02:8081  |
//! | worker     | Deepthink phase 2      | Qwen3.5-397B on vasp-02:8081  |
//! | judge      | Deepthink phase 3      | Qwen3.5-397B on vasp-02:8081  |
//! | compactor  | Memory manager         | Qwen3.5-397B on vasp-02:8081  |

use std::env;

use serde::{Deserialize, Serialize};

/// Default local inference base URL (Qwen3.5-397B on vasp-02).
const DEFAULT_LOCAL_BASE_URL: &str = "http://vasp-02:8081/v1";
/// Default model alias (Qwen3.5-397B-A17B — primary reasoning model).
const DEFAULT_LOCAL_MODEL: &str = "Qwen3.5-397B-A17B";
/// Maximum number of parallel worker slots on vasp-02.
const DEFAULT_MAX_PARALLEL_WORKERS: usize = 4;
/// Default context window size (tokens) for compaction trigger.
const DEFAULT_CONTEXT_WINDOW: u64 = 32_768;
/// Fraction of context window that triggers compaction (0.75 = 75%).
const DEFAULT_COMPACTION_THRESHOLD: f64 = 0.75;

/// Environment-variable names for model role overrides.
const ENV_GENERATOR_MODEL: &str = "SWARM_GENERATOR_MODEL";
const ENV_CRITIQUE_MODEL: &str = "SWARM_CRITIQUE_MODEL";
const ENV_STRATEGY_MODEL: &str = "SWARM_STRATEGY_MODEL";
const ENV_WORKER_MODEL: &str = "SWARM_WORKER_MODEL";
const ENV_JUDGE_MODEL: &str = "SWARM_JUDGE_MODEL";
const ENV_COMPACTOR_MODEL: &str = "SWARM_COMPACTOR_MODEL";
const ENV_LOCAL_BASE_URL: &str = "SWARM_LOCAL_BASE_URL";
const ENV_LOCAL_API_KEY: &str = "SWARM_LOCAL_API_KEY";

/// Per-role model assignment.  All roles default to the local Qwen3.5-397B endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeModelConfig {
    /// Generator agent — produces code artifacts.
    pub generator: String,
    /// Critique agent — evaluates and gates artifacts.
    pub critique: String,
    /// Strategy agent — decomposes problems in Deepthink mode.
    pub strategy: String,
    /// Worker sub-agent template — executes a single strategy.
    pub worker: String,
    /// Judge agent — synthesises Deepthink results.
    pub judge: String,
    /// Compactor agent — produces `CompactionSummary`.
    pub compactor: String,
}

impl Default for ModeModelConfig {
    fn default() -> Self {
        Self {
            generator: env::var(ENV_GENERATOR_MODEL)
                .unwrap_or_else(|_| DEFAULT_LOCAL_MODEL.to_string()),
            critique: env::var(ENV_CRITIQUE_MODEL)
                .unwrap_or_else(|_| DEFAULT_LOCAL_MODEL.to_string()),
            strategy: env::var(ENV_STRATEGY_MODEL)
                .unwrap_or_else(|_| DEFAULT_LOCAL_MODEL.to_string()),
            worker: env::var(ENV_WORKER_MODEL).unwrap_or_else(|_| DEFAULT_LOCAL_MODEL.to_string()),
            judge: env::var(ENV_JUDGE_MODEL).unwrap_or_else(|_| DEFAULT_LOCAL_MODEL.to_string()),
            compactor: env::var(ENV_COMPACTOR_MODEL)
                .unwrap_or_else(|_| DEFAULT_LOCAL_MODEL.to_string()),
        }
    }
}

/// Configuration for the local vLLM / llama.cpp OpenAI-compatible endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalProviderConfig {
    /// Base URL for the OpenAI-compatible API (e.g. `http://vasp-02:8080/v1`).
    pub base_url: String,
    /// API key — most local servers accept any non-empty value.
    pub api_key: String,
    /// Maximum number of parallel inference requests (matches server slot count).
    pub max_parallel_workers: usize,
}

impl Default for LocalProviderConfig {
    fn default() -> Self {
        Self {
            base_url: env::var(ENV_LOCAL_BASE_URL)
                .unwrap_or_else(|_| DEFAULT_LOCAL_BASE_URL.to_string()),
            api_key: env::var(ENV_LOCAL_API_KEY).unwrap_or_else(|_| "local".to_string()),
            max_parallel_workers: DEFAULT_MAX_PARALLEL_WORKERS,
        }
    }
}

/// Optional cloud provider fallback configuration.
///
/// When `enabled = true` and local inference is unavailable, the mode runner
/// will fall back to the configured cloud endpoint using the key from the
/// corresponding environment variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudFallbackConfig {
    /// Enable cloud fallback.
    pub enabled: bool,
    /// Cloud provider to use (`"anthropic"`, `"openai"`, `"gemini"`).
    pub provider: CloudProvider,
    /// Model to use for all roles when falling back to cloud.
    pub model: String,
    /// Maximum tokens per cloud request (cost guard).
    pub max_tokens: u64,
}

impl Default for CloudFallbackConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            provider: CloudProvider::Anthropic,
            model: "claude-sonnet-4-20250514".to_string(),
            max_tokens: 8_192,
        }
    }
}

/// Supported cloud providers for fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CloudProvider {
    Anthropic,
    OpenAi,
    Gemini,
}

impl std::fmt::Display for CloudProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Anthropic => write!(f, "anthropic"),
            Self::OpenAi => write!(f, "openai"),
            Self::Gemini => write!(f, "gemini"),
        }
    }
}

/// Memory / compaction configuration shared across all modes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionConfig {
    /// Total context window size in tokens for the target model.
    pub context_window_tokens: u64,
    /// Fraction of `context_window_tokens` that triggers compaction.
    ///
    /// Must be in `(0.0, 1.0)`.
    pub compaction_threshold: f64,
    /// Minimum number of messages before compaction is eligible.
    pub min_messages_before_compaction: usize,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            context_window_tokens: DEFAULT_CONTEXT_WINDOW,
            compaction_threshold: DEFAULT_COMPACTION_THRESHOLD,
            min_messages_before_compaction: 4,
        }
    }
}

impl CompactionConfig {
    /// Token count at which compaction should trigger.
    pub fn trigger_threshold_tokens(&self) -> u64 {
        (self.context_window_tokens as f64 * self.compaction_threshold) as u64
    }

    /// Validate the config; return an error string if invalid.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0 < self.compaction_threshold && self.compaction_threshold < 1.0) {
            return Err(format!(
                "compaction_threshold must be in (0, 1), got {}",
                self.compaction_threshold
            ));
        }
        if self.context_window_tokens == 0 {
            return Err("context_window_tokens must be > 0".to_string());
        }
        Ok(())
    }
}

/// Top-level configuration consumed by mode runners.
///
/// Constructed via `ModeRunnerConfig::default()` or `from_env()` for
/// environment-variable-driven configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeRunnerConfig {
    pub models: ModeModelConfig,
    pub local: LocalProviderConfig,
    pub cloud_fallback: CloudFallbackConfig,
    pub compaction: CompactionConfig,
    /// Maximum iterations before the mode runner gives up.
    pub max_iterations: u32,
    /// Temperature for generator/worker agents.
    pub generator_temperature: f64,
    /// Temperature for critique/judge agents (lower = more deterministic).
    pub critic_temperature: f64,
}

impl Default for ModeRunnerConfig {
    fn default() -> Self {
        Self {
            models: ModeModelConfig::default(),
            local: LocalProviderConfig::default(),
            cloud_fallback: CloudFallbackConfig::default(),
            compaction: CompactionConfig::default(),
            max_iterations: 10,
            generator_temperature: 0.4,
            critic_temperature: 0.1,
        }
    }
}

impl ModeRunnerConfig {
    /// Build from environment, falling back to defaults.
    pub fn from_env() -> Self {
        Self::default()
    }

    /// Validate all sub-configs.
    pub fn validate(&self) -> Result<(), String> {
        self.compaction.validate()?;
        if !(0.0..=1.0).contains(&self.generator_temperature) {
            return Err(format!(
                "generator_temperature must be in [0, 1], got {}",
                self.generator_temperature
            ));
        }
        if !(0.0..=1.0).contains(&self.critic_temperature) {
            return Err(format!(
                "critic_temperature must be in [0, 1], got {}",
                self.critic_temperature
            ));
        }
        if self.max_iterations == 0 {
            return Err("max_iterations must be > 0".to_string());
        }
        Ok(())
    }

    /// Build a Rig OpenAI-compatible client pointed at the local inference endpoint.
    pub fn local_client(&self) -> anyhow::Result<rig::providers::openai::CompletionsClient> {
        rig::providers::openai::CompletionsClient::builder()
            .api_key(&self.local.api_key)
            .base_url(&self.local.base_url)
            .build()
            .map_err(|e| anyhow::anyhow!("failed to build local client: {e}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_validates() {
        let cfg = ModeRunnerConfig::default();
        cfg.validate().expect("default config should be valid");
    }

    #[test]
    fn compaction_threshold_tokens() {
        let cfg = CompactionConfig {
            context_window_tokens: 32_768,
            compaction_threshold: 0.75,
            min_messages_before_compaction: 4,
        };
        assert_eq!(cfg.trigger_threshold_tokens(), 24_576);
    }

    #[test]
    fn invalid_compaction_threshold_rejected() {
        let cfg = CompactionConfig {
            context_window_tokens: 32_768,
            compaction_threshold: 1.5,
            min_messages_before_compaction: 4,
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn zero_max_iterations_rejected() {
        let mut cfg = ModeRunnerConfig::default();
        cfg.max_iterations = 0;
        assert!(cfg.validate().is_err());
    }
}
