use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue};
use rig::providers::openai;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Duration;

/// Named stack-profile system for the swarm.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmStackProfile {
    /// Qwen3-Coder-Next on vasp-03, Qwen3.5-122B-A10B on vasp-01 (coder) + vasp-02 (reasoning)
    #[default]
    #[serde(rename = "hybrid_balanced_v1")]
    HybridBalancedV1,
    /// Test if 27B should absorb more early planning and tool-use work.
    #[serde(rename = "small_specialist_v1")]
    SmallSpecialistV1,
    /// Test if 397B belongs only in a non-writing strategist arbitration lane.
    #[serde(rename = "strategist_hybrid_v1")]
    StrategistHybridV1,
}

impl FromStr for SwarmStackProfile {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hybrid_balanced_v1" | "balanced" => Ok(Self::HybridBalancedV1),
            "small_specialist_v1" | "specialist" => Ok(Self::SmallSpecialistV1),
            "strategist_hybrid_v1" | "strategist" => Ok(Self::StrategistHybridV1),
            _ => anyhow::bail!("Unknown stack profile: {}", s),
        }
    }
}

/// Swarm agent roles used for model routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SwarmRole {
    Scout,
    Reviewer,
    RustWorker,
    GeneralWorker,
    Planner,
    Fixer,
    ReasoningWorker,
    Strategist,
    LocalManagerFallback,
    Council,
}

/// Inference tier for model routing.
#[derive(Debug, Clone, Deserialize)]
pub enum Tier {
    /// Qwen3.5-27B-Distilled on vasp-03 — Scout/fast tier (192K context, VRAM-resident)
    Fast,
    /// Qwen3.5-122B-A10B on vasp-01 — Coder/integrator tier (65K context, expert-offload)
    Coder,
    /// Qwen3.5-122B-A10B on vasp-02 — Reasoning/integrator tier (65K context, expert-offload)
    Reasoning,
    /// Cloud models via CLIAPIProxy
    Cloud,
    /// Qwen3.5-397B-A17B (advisor/strategist tier)
    Strategist,
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
///
/// CLIAPIProxy v6.8+ authenticates via `x-api-key` header (not `Authorization: Bearer`).
/// The Rig OpenAI client sends Bearer by default, so we inject `x-api-key` via
/// `.http_headers()` — the proxy prioritizes `x-api-key` and ignores the Bearer token.
#[derive(Debug, Clone, Deserialize)]
pub struct CloudEndpoint {
    pub url: String,
    pub api_key: String,
    pub model: String,
}

/// A cloud model entry with cost and capability metadata.
///
/// Used by the phase-based model selector and learning router to choose
/// the best model for each workflow phase. Part of the Ensemble Swarm architecture.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CloudModelEntry {
    /// Model ID as sent to CLIAPIProxy/OpenRouter (e.g., "claude-haiku-4-5-20251001").
    pub model: String,
    /// Human-readable label for logging.
    pub label: String,
    /// Cost per million input tokens (USD). Used for cost-aware routing.
    pub cost_input_per_m: f64,
    /// Cost per million output tokens (USD).
    pub cost_output_per_m: f64,
    /// Maximum context window (tokens).
    pub context_window: usize,
    /// Capability tags for phase-based selection.
    pub capabilities: Vec<String>,
    /// Provider: "cliapi" (CLIAPIProxy) or "openrouter".
    pub provider: String,
    /// Explicit capability ranking (higher = more capable). Used instead of cost as proxy.
    pub capability_score: u32,
}

/// Catalog of all available cloud models.
///
/// Built from env vars or defaults. The phase-based model selector and learning
/// router query this catalog to find the best model for each workflow phase.
#[derive(Debug, Clone)]
pub struct CloudModelCatalog {
    pub models: Vec<CloudModelEntry>,
}

impl CloudModelCatalog {
    /// Build the default catalog from CLIAPIProxy + optional OpenRouter.
    ///
    /// All models available via CLIAPIProxy are included with approximate costs.
    /// OpenRouter models are added when `SWARM_OPENROUTER_URL` is set.
    pub fn default_catalog() -> Self {
        let mut models = vec![
            // --- Anthropic (via CLIAPIProxy) ---
            CloudModelEntry {
                model: "claude-opus-4-6".into(),
                label: "Opus 4.6".into(),
                cost_input_per_m: 15.0,
                cost_output_per_m: 75.0,
                context_window: 200_000,
                capabilities: vec!["plan".into(), "architect".into(), "review".into(), "reason".into()],
                provider: "cliapi".into(),
                capability_score: 100,
            },
            CloudModelEntry {
                model: "claude-sonnet-4-6".into(),
                label: "Sonnet 4.6".into(),
                cost_input_per_m: 3.0,
                cost_output_per_m: 15.0,
                context_window: 200_000,
                capabilities: vec!["plan".into(), "implement".into(), "review".into()],
                provider: "cliapi".into(),
                capability_score: 80,
            },
            CloudModelEntry {
                model: "claude-haiku-4-5-20251001".into(),
                label: "Haiku 4.5".into(),
                cost_input_per_m: 0.80,
                cost_output_per_m: 4.0,
                context_window: 200_000,
                capabilities: vec!["triage".into(), "review".into(), "scout".into()],
                provider: "cliapi".into(),
                capability_score: 40,
            },
            // --- Antigravity / Google (via CLIAPIProxy) ---
            CloudModelEntry {
                model: "gemini-3.1-pro-high".into(),
                label: "Gemini 3.1 Pro".into(),
                cost_input_per_m: 1.25,
                cost_output_per_m: 10.0,
                context_window: 2_000_000,
                capabilities: vec!["explore".into(), "plan".into(), "review".into(), "architect".into()],
                provider: "cliapi".into(),
                capability_score: 85,
            },
            CloudModelEntry {
                model: "gemini-3-flash-preview".into(),
                label: "Gemini 3 Flash".into(),
                cost_input_per_m: 0.10,
                cost_output_per_m: 0.40,
                context_window: 1_000_000,
                capabilities: vec!["triage".into(), "explore".into(), "scout".into()],
                provider: "cliapi".into(),
                capability_score: 35,
            },
            CloudModelEntry {
                model: "gemini-2.5-flash".into(),
                label: "Gemini 2.5 Flash".into(),
                cost_input_per_m: 0.15,
                cost_output_per_m: 0.60,
                context_window: 1_000_000,
                capabilities: vec!["triage".into(), "scout".into()],
                provider: "cliapi".into(),
                capability_score: 30,
            },
        ];

        // --- OpenRouter models (when configured) ---
        if std::env::var("SWARM_OPENROUTER_URL")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .is_some()
        {
            models.push(CloudModelEntry {
                model: "minimax/minimax-m2.7".into(),
                label: "MiniMax M2.7".into(),
                cost_input_per_m: 0.30,
                cost_output_per_m: 1.20,
                context_window: 204_800,
                capabilities: vec!["implement".into(), "review".into(), "plan".into()],
                provider: "openrouter".into(),
                capability_score: 60,
            });
        }

        Self { models }
    }

    /// Find models with a specific capability.
    pub fn with_capability(&self, capability: &str) -> Vec<&CloudModelEntry> {
        self.models
            .iter()
            .filter(|m| m.capabilities.iter().any(|c| c == capability))
            .collect()
    }

    /// Find the cheapest model with a given capability.
    pub fn cheapest_for(&self, capability: &str) -> Option<&CloudModelEntry> {
        self.with_capability(capability)
            .into_iter()
            .min_by(|a, b| {
                a.cost_input_per_m
                    .partial_cmp(&b.cost_input_per_m)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Find the most capable model (highest capability_score).
    pub fn strongest_for(&self, capability: &str) -> Option<&CloudModelEntry> {
        self.with_capability(capability)
            .into_iter()
            .max_by_key(|m| m.capability_score)
    }
}

/// Build a [`HeaderMap`] with the `x-api-key` header for CLIAPIProxy authentication.
///
/// CLIAPIProxy v6.8+ requires `x-api-key` instead of `Authorization: Bearer` for
/// static API key auth. This header map is injected into Rig's OpenAI client via
/// `.http_headers()`, coexisting with the Bearer token that Rig adds automatically.
fn cloud_headers(api_key: &str) -> HeaderMap {
    let mut headers = HeaderMap::new();
    if let Ok(val) = HeaderValue::from_str(api_key) {
        headers.insert("x-api-key", val);
    }
    headers
}

/// Helper function to build a rig OpenAI CompletionsClient.
fn build_oai_client(
    api_key: &str,
    url: &str,
    http_client: reqwest::Client,
    headers: Option<HeaderMap>,
) -> Result<openai::CompletionsClient> {
    let mut builder = openai::CompletionsClient::<reqwest::Client>::builder()
        .api_key(api_key)
        .base_url(url)
        .http_client(http_client);

    if let Some(h) = headers {
        builder = builder.http_headers(h);
    }

    Ok(builder.build()?)
}

/// Default per-request HTTP timeout for cloud API calls (CLIAPIProxy).
///
/// Cloud APIs (Opus 4.6 via CLIAPIProxy) respond within seconds to a few minutes.
/// 5 minutes is generous; a hung proxy connection should fail fast so retry logic
/// can kick in rather than burning the entire 45-minute manager budget.
const DEFAULT_CLOUD_HTTP_TIMEOUT_SECS: u64 = 300;

/// Default per-request HTTP timeout for local LLM calls (Qwen3.5 on vasp nodes).
///
/// Local models at ~6 tok/s with max_tokens=4096 need ~11 minutes worst case.
/// With 20-turn write deadline and 2 min/turn on 122B expert-offload,
/// workers need up to 40 minutes. 45 minutes provides headroom.
const DEFAULT_LOCAL_HTTP_TIMEOUT_SECS: u64 = 2700;

/// Default TCP connect timeout for all endpoints.
const DEFAULT_CONNECT_TIMEOUT_SECS: u64 = 15;

/// Build a `reqwest::Client` with per-request and connect timeouts.
///
/// The per-request timeout covers the entire HTTP lifecycle (connect + send + receive).
/// Without this, a hung endpoint silently consumes the entire manager/worker budget.
fn http_client_with_timeout(timeout_secs: u64) -> Result<reqwest::Client> {
    reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .connect_timeout(Duration::from_secs(DEFAULT_CONNECT_TIMEOUT_SECS))
        .build()
        .context("Failed to build reqwest::Client with timeout")
}

/// Read an HTTP timeout from an environment variable, falling back to a default.
fn http_timeout_from_env(var: &str, default: u64) -> u64 {
    std::env::var(var)
        .ok()
        .and_then(|s| s.parse().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

/// A single entry in the cloud model fallback matrix.
#[derive(Debug, Clone, Deserialize)]
pub struct CloudFallbackEntry {
    /// Model name as used in API requests.
    pub model: String,
    /// Human-readable tier label (e.g., "primary", "fallback-1").
    pub tier_label: String,
    /// Maximum tokens for responses from this model.
    pub max_tokens: u32,
}

/// Ordered list of cloud models to attempt, with automatic fallback.
///
/// When the primary cloud model fails (rate limit, timeout, error),
/// the orchestrator tries the next model in the matrix.
#[derive(Debug, Clone)]
pub struct CloudFallbackMatrix {
    /// Ordered entries: first is primary, rest are fallbacks.
    pub entries: Vec<CloudFallbackEntry>,
}

impl CloudFallbackMatrix {
    /// The primary (first-choice) cloud model, if any.
    pub fn primary(&self) -> Option<&CloudFallbackEntry> {
        self.entries.first()
    }

    /// Fallback models (everything after the primary).
    pub fn fallbacks(&self) -> &[CloudFallbackEntry] {
        if self.entries.len() > 1 {
            &self.entries[1..]
        } else {
            &[]
        }
    }

    /// Number of models in the matrix.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the matrix is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Build from `SWARM_CLOUD_FALLBACK_MODELS` env var (comma-separated model names).
    ///
    /// Falls back to [`Self::default_matrix`] if the env var is unset or empty.
    pub fn from_env() -> Self {
        match std::env::var("SWARM_CLOUD_FALLBACK_MODELS") {
            Ok(val) if !val.trim().is_empty() => {
                let entries = val
                    .split(',')
                    .enumerate()
                    .map(|(i, m)| {
                        let model = m.trim().to_string();
                        let tier_label = if i == 0 {
                            "primary".to_string()
                        } else {
                            format!("fallback-{i}")
                        };
                        CloudFallbackEntry {
                            model,
                            tier_label,
                            max_tokens: 4096,
                        }
                    })
                    .collect();
                Self { entries }
            }
            _ => Self::default_matrix(),
        }
    }

    /// Default matrix: Opus 4.6 → Gemini 3.1 Pro High → Sonnet 4.6 → Gemini 3.1 Flash Lite.
    ///
    /// Gemini 3.1 Pro has 2M context for whole-codebase understanding. Sonnet 4.6
    /// is faster than Opus for simpler delegation. Gemini 3.1 Flash Lite is the
    /// latest fast/cheap model for last-resort fallback.
    pub fn default_matrix() -> Self {
        Self {
            entries: vec![
                CloudFallbackEntry {
                    model: "claude-opus-4-6".to_string(),
                    tier_label: "primary".to_string(),
                    max_tokens: 4096,
                },
                CloudFallbackEntry {
                    model: "gemini-3.1-pro-high".to_string(),
                    tier_label: "fallback-1".to_string(),
                    max_tokens: 4096,
                },
                CloudFallbackEntry {
                    model: "claude-sonnet-4-6".to_string(),
                    tier_label: "fallback-2".to_string(),
                    max_tokens: 4096,
                },
                CloudFallbackEntry {
                    model: "gemini-3.1-flash-lite-preview".to_string(),
                    tier_label: "fallback-3".to_string(),
                    max_tokens: 4096,
                },
            ],
        }
    }
}

/// Top-level swarm configuration.
#[derive(Debug, Clone)]
pub struct SwarmConfig {
    /// Qwen3-Coder-Next on vasp-03:8081 (Scout/fast tier, 65K context, expert-offload MoE)
    pub fast_endpoint: Endpoint,
    /// Qwen3.5-122B-A10B on vasp-01:8081 (Coder/integrator, 65K context, expert-offload)
    pub coder_endpoint: Endpoint,
    /// Qwen3.5-122B-A10B on vasp-02:8081 (Reasoning/integrator, 65K context, expert-offload)
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
    /// Ordered cloud model fallback matrix.
    /// When the primary cloud model fails, the orchestrator tries the next model.
    /// Populated from `SWARM_CLOUD_FALLBACK_MODELS` env var (comma-separated) or defaults.
    pub cloud_fallback_matrix: CloudFallbackMatrix,
    /// Maximum consecutive no-change iterations before circuit breaker fires.
    /// When an agent responds without producing any file changes for this many
    /// iterations in a row, the loop breaks and creates a stuck intervention.
    /// Populated from `SWARM_MAX_NO_CHANGE` env var (default: 3).
    pub max_consecutive_no_change: u32,
    /// Minimum objective length (characters) to accept for processing.
    /// Issues with titles shorter than this are rejected before worktree creation.
    /// Populated from `SWARM_MIN_OBJECTIVE_LEN` env var (default: 10).
    pub min_objective_len: usize,
    /// Maximum estimated cost (in USD) per issue before the loop aborts.
    /// Uses approximate token costs: cloud=$15/M input + $75/M output, local=$0 (self-hosted).
    /// Set to 0.0 to disable cost budgeting (default).
    /// Populated from `SWARM_MAX_COST_PER_ISSUE` env var.
    pub max_cost_per_issue: f64,
    /// Pre-parsed, lowercased reject patterns from `SWARM_REJECT_PATTERNS` env var.
    /// Issues matching any pattern (substring match) are rejected before worktree creation.
    pub reject_patterns: Vec<String>,
    /// Skip LLM-based triage, use keyword heuristics only.
    /// Populated from `SWARM_SKIP_TRIAGE` env var.
    pub skip_triage: bool,
    /// Number of iterations after which task prompts are pruned to save context.
    /// After this many iterations, only the system prompt, last 2 iteration results,
    /// and the latest verifier output are included in the prompt.
    /// Populated from `SWARM_PRUNE_AFTER_ITERATION` env var (default: 3).
    pub prune_after_iteration: u32,
    /// Number of issues to process concurrently.
    /// Each issue gets its own worktree and uses the next node in round-robin order.
    /// Populated from `SWARM_PARALLEL_ISSUES` env var (default: 3 = one per node).
    pub parallel_issues: usize,
    /// Enable concurrent subtask dispatch for multi-file issues.
    /// When true, the planner decomposes issues into non-overlapping subtasks
    /// and workers execute them concurrently on the 2-node 122B cluster.
    /// Single-subtask plans fall through to the sequential loop.
    /// Populated from `SWARM_CONCURRENT_SUBTASKS` env var (default: true).
    pub concurrent_subtasks: bool,
    /// Named stack profile for model routing.
    /// Selected by `SWARM_STACK_PROFILE` env var.
    pub stack_profile: SwarmStackProfile,
    /// Qwen3.5-397B-A17B (strategist/advisor, optional).
    /// Populated from `SWARM_STRATEGIST_URL` and `SWARM_STRATEGIST_MODEL`.
    pub strategist_endpoint: Option<Endpoint>,
    /// Repository identifier (e.g., "rust-daq", "CF-LIBS") for adapter selection.
    /// Populated from `SWARM_REPO_ID`.
    pub repo_id: Option<String>,
    /// QLoRA/LoRA adapter identifier for the coder model.
    /// Populated from `SWARM_ADAPTER_ID`.
    pub adapter_id: Option<String>,
    /// TensorZero gateway URL (e.g., "http://localhost:3000").
    /// When set, cloud inference calls are routed through TensorZero for
    /// experiment tracking, A/B testing, and feedback collection.
    /// Populated from `SWARM_TENSORZERO_URL` env var.
    pub tensorzero_url: Option<String>,
    /// TensorZero Postgres URL for reading performance insights.
    /// Auto-detected when `SWARM_TENSORZERO_URL` is set (defaults to
    /// `postgres://tensorzero:tensorzero@localhost:5433/tensorzero`).
    /// Explicitly overridden via `SWARM_TENSORZERO_PG_URL` env var.
    pub tensorzero_pg_url: Option<String>,
    /// Cache TTL for TZ insights (seconds). Default: 1800 (30 min).
    /// Populated from `SWARM_TZ_INSIGHTS_TTL_SECS` env var.
    pub tz_insights_ttl_secs: u64,
    /// OpenRouter API endpoint for external models (e.g., MiniMax M2.7).
    /// When set, the cloud model catalog includes OpenRouter models.
    /// Populated from `SWARM_OPENROUTER_URL` env var.
    pub openrouter_url: Option<String>,
    /// OpenRouter API key.
    /// Populated from `SWARM_OPENROUTER_API_KEY` env var.
    pub openrouter_api_key: Option<String>,
    /// Cloud model catalog with all available models and their capabilities/costs.
    /// Built automatically from CLIAPIProxy + OpenRouter configuration.
    pub cloud_model_catalog: CloudModelCatalog,
    /// Enable UCB1 bandit-based adaptive model routing.
    /// When true, `mutation_archive.recommend_model()` is consulted before each
    /// worker dispatch and the recommendation is logged for analysis.
    /// Populated from `SWARM_ADAPTIVE_ROUTING` env var (default: false).
    pub adaptive_routing: bool,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            fast_endpoint: Endpoint {
                url: std::env::var("SWARM_FAST_URL")
                    .unwrap_or_else(|_| "http://vasp-03:8081/v1".into()),
                model: std::env::var("SWARM_FAST_MODEL")
                    .unwrap_or_else(|_| "Qwen3-Coder-Next".into()),
                tier: Tier::Fast,
                api_key: std::env::var("SWARM_FAST_API_KEY")
                    .unwrap_or_else(|_| "not-needed".into()),
            },
            coder_endpoint: Endpoint {
                url: std::env::var("SWARM_CODER_URL")
                    .unwrap_or_else(|_| "http://vasp-01:8081/v1".into()),
                model: std::env::var("SWARM_CODER_MODEL")
                    .unwrap_or_else(|_| "Qwen3.5-122B-A10B".into()),
                tier: Tier::Coder,
                api_key: std::env::var("SWARM_CODER_API_KEY")
                    .unwrap_or_else(|_| "not-needed".into()),
            },
            reasoning_endpoint: Endpoint {
                url: std::env::var("SWARM_REASONING_URL")
                    .unwrap_or_else(|_| "http://vasp-02:8081/v1".into()),
                model: std::env::var("SWARM_REASONING_MODEL")
                    .unwrap_or_else(|_| "Qwen3.5-122B-A10B".into()),
                tier: Tier::Reasoning,
                api_key: std::env::var("SWARM_REASONING_API_KEY")
                    .unwrap_or_else(|_| "not-needed".into()),
            },
            cloud_endpoint: Self::cloud_from_env(),
            max_retries: std::env::var("SWARM_MAX_RETRIES")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|v| *v > 0)
                .unwrap_or(6),
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
            cloud_fallback_matrix: CloudFallbackMatrix::from_env(),
            max_consecutive_no_change: std::env::var("SWARM_MAX_NO_CHANGE")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|v| *v > 0)
                .unwrap_or(3),
            min_objective_len: std::env::var("SWARM_MIN_OBJECTIVE_LEN")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(10),
            max_cost_per_issue: std::env::var("SWARM_MAX_COST_PER_ISSUE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.0),
            reject_patterns: std::env::var("SWARM_REJECT_PATTERNS")
                .ok()
                .filter(|s| !s.is_empty())
                .map(|s| s.split(',').map(|p| p.trim().to_lowercase()).collect())
                .unwrap_or_default(),
            skip_triage: std::env::var("SWARM_SKIP_TRIAGE")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
            prune_after_iteration: std::env::var("SWARM_PRUNE_AFTER_ITERATION")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(3),
            parallel_issues: std::env::var("SWARM_PARALLEL_ISSUES")
                .ok()
                .and_then(|s| s.parse().ok())
                .filter(|v: &usize| *v > 0)
                .unwrap_or(3),
            concurrent_subtasks: std::env::var("SWARM_CONCURRENT_SUBTASKS")
                .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
                .unwrap_or(true),
            stack_profile: std::env::var("SWARM_STACK_PROFILE")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or_default(),
            strategist_endpoint: Self::strategist_from_env(),
            repo_id: std::env::var("SWARM_REPO_ID").ok(),
            adapter_id: std::env::var("SWARM_ADAPTER_ID").ok(),
            tensorzero_url: std::env::var("SWARM_TENSORZERO_URL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            tensorzero_pg_url: std::env::var("SWARM_TENSORZERO_PG_URL")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .or_else(|| {
                    std::env::var("SWARM_TENSORZERO_URL")
                        .ok()
                        .filter(|s| !s.trim().is_empty())
                        .map(|_| {
                            "postgres://tensorzero:tensorzero@localhost:5433/tensorzero".into()
                        })
                }),
            tz_insights_ttl_secs: std::env::var("SWARM_TZ_INSIGHTS_TTL_SECS")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1800),
            openrouter_url: std::env::var("SWARM_OPENROUTER_URL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            openrouter_api_key: std::env::var("SWARM_OPENROUTER_API_KEY")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            cloud_model_catalog: CloudModelCatalog::default_catalog(),
            adaptive_routing: std::env::var("SWARM_ADAPTIVE_ROUTING")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false),
        }
    }
}

impl SwarmConfig {
    /// Resolve the appropriate rig client for a given swarm role based on the active stack profile.
    pub fn resolve_role_client(
        &self,
        role: SwarmRole,
        clients: &ClientSet,
    ) -> Result<openai::CompletionsClient> {
        match self.stack_profile {
            SwarmStackProfile::HybridBalancedV1 => match role {
                SwarmRole::Scout | SwarmRole::Reviewer | SwarmRole::RustWorker | SwarmRole::Fixer => {
                    Ok(clients.local.clone())
                }
                SwarmRole::GeneralWorker
                | SwarmRole::Planner
                | SwarmRole::ReasoningWorker
                | SwarmRole::LocalManagerFallback => Ok(clients.coder.clone()), // Use coder/reasoning pool via factory
                SwarmRole::Council => clients
                    .cloud_tz
                    .clone()
                    .or_else(|| clients.cloud.clone())
                    .context("Council role requested but no cloud endpoint configured"),
                SwarmRole::Strategist => clients
                    .strategist
                    .clone()
                    .or_else(|| clients.cloud.clone())
                    .context("Strategist role requested but no strategist or cloud endpoint configured"),
            },
            SwarmStackProfile::SmallSpecialistV1 => match role {
                SwarmRole::Scout
                | SwarmRole::Reviewer
                | SwarmRole::RustWorker
                | SwarmRole::Fixer
                | SwarmRole::Planner => Ok(clients.local.clone()),
                SwarmRole::GeneralWorker
                | SwarmRole::ReasoningWorker
                | SwarmRole::LocalManagerFallback => Ok(clients.coder.clone()),
                SwarmRole::Council => clients
                    .cloud_tz
                    .clone()
                    .or_else(|| clients.cloud.clone())
                    .context("Council role requested but no cloud endpoint configured"),
                SwarmRole::Strategist => clients
                    .strategist
                    .clone()
                    .or_else(|| clients.cloud.clone())
                    .context("Strategist role requested but no strategist or cloud endpoint configured"),
            },
            SwarmStackProfile::StrategistHybridV1 => match role {
                SwarmRole::Scout | SwarmRole::Reviewer | SwarmRole::RustWorker | SwarmRole::Fixer => {
                    Ok(clients.local.clone())
                }
                SwarmRole::GeneralWorker
                | SwarmRole::Planner
                | SwarmRole::ReasoningWorker
                | SwarmRole::LocalManagerFallback => Ok(clients.coder.clone()),
                SwarmRole::Strategist => clients.strategist.clone().context(
                    "Strategist role requested for StrategistHybridV1 but SWARM_STRATEGIST_URL is not set",
                ),
                SwarmRole::Council => clients
                    .cloud_tz
                    .clone()
                    .or_else(|| clients.cloud.clone())
                    .context("Council role requested but no cloud endpoint configured"),
            },
        }
    }

    /// Resolve the appropriate model name for a given swarm role based on the active stack profile.
    /// Includes the LoRA/QLoRA adapter suffix if configured for worker roles.
    pub fn resolve_role_model(&self, role: SwarmRole) -> String {
        if self.cloud_only {
            if let Some(ref cloud_ep) = self.cloud_endpoint {
                return cloud_ep.model.clone();
            }
        }

        // When TZ is configured, map every role to a TZ function name so all
        // inferences are tracked and A/B tested. TZ routes each function to the
        // configured model variants (Strand-14B, Tessa-7B, HydraCoder, 122B, etc.)
        if self.tensorzero_url.is_some() {
            return match role {
                SwarmRole::Council => "tensorzero::function_name::architect_plan",
                SwarmRole::Scout | SwarmRole::Reviewer => "tensorzero::function_name::code_review",
                SwarmRole::RustWorker | SwarmRole::GeneralWorker => {
                    "tensorzero::function_name::worker_code_edit"
                }
                SwarmRole::Fixer => "tensorzero::function_name::code_fixing",
                SwarmRole::Planner => "tensorzero::function_name::task_planning",
                SwarmRole::ReasoningWorker => "tensorzero::function_name::deep_reasoning",
                SwarmRole::Strategist => "tensorzero::function_name::deep_reasoning",
                SwarmRole::LocalManagerFallback => "tensorzero::function_name::worker_code_edit",
            }
            .to_string();
        }

        let base_model = match self.stack_profile {
            SwarmStackProfile::HybridBalancedV1 => match role {
                SwarmRole::Scout
                | SwarmRole::Reviewer
                | SwarmRole::RustWorker
                | SwarmRole::Fixer => &self.fast_endpoint.model,
                SwarmRole::GeneralWorker
                | SwarmRole::Planner
                | SwarmRole::ReasoningWorker
                | SwarmRole::LocalManagerFallback => &self.coder_endpoint.model,
                SwarmRole::Council => self
                    .cloud_endpoint
                    .as_ref()
                    .map(|e| e.model.as_str())
                    .unwrap_or("unknown-cloud-model"),
                SwarmRole::Strategist => self
                    .strategist_endpoint
                    .as_ref()
                    .map(|e| e.model.as_str())
                    .or_else(|| self.cloud_endpoint.as_ref().map(|e| e.model.as_str()))
                    .unwrap_or("unknown-strategist-model"),
            },
            SwarmStackProfile::SmallSpecialistV1 => match role {
                SwarmRole::Scout
                | SwarmRole::Reviewer
                | SwarmRole::RustWorker
                | SwarmRole::Fixer
                | SwarmRole::Planner => &self.fast_endpoint.model,
                SwarmRole::GeneralWorker
                | SwarmRole::ReasoningWorker
                | SwarmRole::LocalManagerFallback => &self.coder_endpoint.model,
                SwarmRole::Council => self
                    .cloud_endpoint
                    .as_ref()
                    .map(|e| e.model.as_str())
                    .unwrap_or("unknown-cloud-model"),
                SwarmRole::Strategist => self
                    .strategist_endpoint
                    .as_ref()
                    .map(|e| e.model.as_str())
                    .or_else(|| self.cloud_endpoint.as_ref().map(|e| e.model.as_str()))
                    .unwrap_or("unknown-strategist-model"),
            },
            SwarmStackProfile::StrategistHybridV1 => match role {
                SwarmRole::Scout
                | SwarmRole::Reviewer
                | SwarmRole::RustWorker
                | SwarmRole::Fixer => &self.fast_endpoint.model,
                SwarmRole::GeneralWorker
                | SwarmRole::Planner
                | SwarmRole::ReasoningWorker
                | SwarmRole::LocalManagerFallback => &self.coder_endpoint.model,
                SwarmRole::Strategist => self
                    .strategist_endpoint
                    .as_ref()
                    .map(|e| e.model.as_str())
                    .unwrap_or("Qwen3.5-397B-A17B"),
                SwarmRole::Council => self
                    .cloud_endpoint
                    .as_ref()
                    .map(|e| e.model.as_str())
                    .unwrap_or("unknown-cloud-model"),
            },
        };

        // Apply adapter if configured for worker/fixer roles
        if let Some(ref adapter) = self.adapter_id {
            if matches!(role, SwarmRole::RustWorker | SwarmRole::Fixer) {
                return format!("{base_model}:{adapter}");
            }
        }

        base_model.to_string()
    }

    fn cloud_from_env() -> Option<CloudEndpoint> {
        let url = std::env::var("SWARM_CLOUD_URL").ok()?;
        let api_key = std::env::var("SWARM_CLOUD_API_KEY").ok()?;
        let model = std::env::var("SWARM_CLOUD_MODEL").unwrap_or_else(|_| "claude-opus-4-6".into());
        Some(CloudEndpoint {
            url,
            api_key,
            model,
        })
    }

    fn strategist_from_env() -> Option<Endpoint> {
        let url = std::env::var("SWARM_STRATEGIST_URL").ok()?;
        let model =
            std::env::var("SWARM_STRATEGIST_MODEL").unwrap_or_else(|_| "Qwen3.5-397B-A17B".into());
        Some(Endpoint {
            url,
            model,
            tier: Tier::Strategist,
            api_key: std::env::var("SWARM_STRATEGIST_API_KEY")
                .unwrap_or_else(|_| "not-needed".into()),
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
                model: "gemini-3.1-flash-lite-preview".into(),
                tier: Tier::Fast,
                api_key: proxy_key.clone(),
            },
            coder_endpoint: Endpoint {
                url: proxy_url.clone(),
                model: "claude-sonnet-4-5-20250929".into(),
                tier: Tier::Coder,
                api_key: proxy_key.clone(),
            },
            reasoning_endpoint: Endpoint {
                url: proxy_url.clone(),
                model: "claude-opus-4-6".into(),
                tier: Tier::Reasoning,
                api_key: proxy_key.clone(),
            },
            cloud_endpoint: Some(CloudEndpoint {
                url: proxy_url.clone(),
                api_key: proxy_key.clone(),
                model: "claude-opus-4-6".into(),
            }),
            strategist_endpoint: Some(Endpoint {
                url: proxy_url,
                model: "gemini-3-pro-preview".into(),
                tier: Tier::Strategist,
                api_key: proxy_key,
            }),
            max_retries: 3,
            worktree_base: None,
            notebook_registry_path: None,
            verifier_packages: Vec::new(),
            cloud_max_retries: 3,
            cloud_only: false,
            cloud_fallback_matrix: CloudFallbackMatrix::default_matrix(),
            max_consecutive_no_change: 3,
            min_objective_len: 10,
            max_cost_per_issue: 0.0,
            reject_patterns: Vec::new(),
            skip_triage: false,
            prune_after_iteration: 3,
            parallel_issues: 1,
            concurrent_subtasks: true,
            stack_profile: SwarmStackProfile::HybridBalancedV1,
            repo_id: None,
            adapter_id: None,
            tensorzero_url: None,
            tensorzero_pg_url: None,
            tz_insights_ttl_secs: 1800,
            openrouter_url: std::env::var("SWARM_OPENROUTER_URL")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            openrouter_api_key: std::env::var("SWARM_OPENROUTER_API_KEY")
                .ok()
                .filter(|s| !s.trim().is_empty()),
            cloud_model_catalog: CloudModelCatalog::default_catalog(),
            adaptive_routing: false,
        }
    }
}

/// Pre-built rig CompletionsClients for the three-node inference cluster.
///
/// Each tier maps to a different node/model:
/// - `local`     -> vasp-03:8081 (Qwen3.5-27B-Distilled, Scout/fast tier, 192K context)
/// - `coder`     -> vasp-01:8081 (Qwen3.5-122B-A10B, Integrator/coder tier, 65K context)
/// - `reasoning` -> vasp-02:8081 (Qwen3.5-122B-A10B, Integrator/reasoning tier, 65K context)
#[derive(Clone)]
pub struct ClientSet {
    /// Client for vasp-03:8081 (Qwen3.5-27B-Distilled -- scout/fast tier: analysis, routing, review)
    pub local: openai::CompletionsClient,
    /// Client for vasp-01:8081 (Qwen3.5-122B-A10B -- coder/integrator tier: code generation)
    pub coder: openai::CompletionsClient,
    /// Client for vasp-02:8081 (Qwen3.5-122B-A10B -- reasoning/integrator tier: deep analysis)
    pub reasoning: openai::CompletionsClient,
    /// Client for Qwen3.5-397B-A17B (strategist/advisor tier)
    pub strategist: Option<openai::CompletionsClient>,
    /// Direct client for CLIAPIProxy (cloud models: Opus 4.6, G3-Pro, etc.).
    /// Used for fallback model matrix, cloud validation, and all non-manager cloud calls.
    pub cloud: Option<openai::CompletionsClient>,
    /// TensorZero gateway client — present when `SWARM_TENSORZERO_URL` is set.
    ///
    /// Routes the primary cloud manager and Architect calls through TZ for
    /// experiment tracking and A/B testing. The fallback model matrix always
    /// uses `cloud` (direct CLIAPIProxy) to bypass TZ function-name routing.
    pub cloud_tz: Option<openai::CompletionsClient>,
}

impl ClientSet {
    pub fn from_config(config: &SwarmConfig) -> Result<Self> {
        let cloud_timeout = http_timeout_from_env(
            "SWARM_CLOUD_HTTP_TIMEOUT_SECS",
            DEFAULT_CLOUD_HTTP_TIMEOUT_SECS,
        );
        let local_timeout = http_timeout_from_env(
            "SWARM_LOCAL_HTTP_TIMEOUT_SECS",
            DEFAULT_LOCAL_HTTP_TIMEOUT_SECS,
        );

        tracing::info!(
            cloud_http_timeout_secs = cloud_timeout,
            local_http_timeout_secs = local_timeout,
            "HTTP client timeouts configured"
        );

        let cloud_http = http_client_with_timeout(cloud_timeout)?;
        let local_http = http_client_with_timeout(local_timeout)?;

        // In cloud-only mode, reuse the cloud client for all tiers
        if config.cloud_only {
            let cloud_ep = config
                .cloud_endpoint
                .as_ref()
                .context("cloud_only requires cloud_endpoint to be configured")?;
            let headers = cloud_headers(&cloud_ep.api_key);

            let cloud =
                build_oai_client(&cloud_ep.api_key, &cloud_ep.url, cloud_http, Some(headers))
                    .context("Failed to build cloud client")?;

            return Ok(Self {
                local: cloud.clone(),
                coder: cloud.clone(),
                reasoning: cloud.clone(),
                strategist: Some(cloud.clone()),
                cloud: Some(cloud),
                cloud_tz: None, // TZ doesn't apply in cloud_only mode
            });
        }

        // When TZ is configured, route ALL local inference through TZ's OpenAI
        // endpoint so every call is logged with function/variant for A/B testing.
        // Without this, only cloud manager calls go through TZ and local model
        // experiments are invisible.
        let (local, coder, reasoning) = if let Some(ref tz_url) = config.tensorzero_url {
            let tz_base = format!("{tz_url}/openai/v1");
            tracing::info!(url = %tz_url, "Routing ALL local clients through TensorZero");
            let tz_local = build_oai_client("not-needed", &tz_base, local_http.clone(), None)
                .context("Failed to build TZ local client")?;
            // All three local clients point at TZ — the model name (set per-agent
            // via resolve_role_model) determines which TZ function handles each call.
            (tz_local.clone(), tz_local.clone(), tz_local)
        } else {
            let local = build_oai_client(
                &config.fast_endpoint.api_key,
                &config.fast_endpoint.url,
                local_http.clone(),
                None,
            )
            .context("Failed to build local/fast client (vasp-03)")?;

            let coder = build_oai_client(
                &config.coder_endpoint.api_key,
                &config.coder_endpoint.url,
                local_http.clone(),
                None,
            )
            .context("Failed to build coder client (vasp-01)")?;

            let reasoning = build_oai_client(
                &config.reasoning_endpoint.api_key,
                &config.reasoning_endpoint.url,
                local_http.clone(),
                None,
            )
            .context("Failed to build reasoning client (vasp-02)")?;

            (local, coder, reasoning)
        };

        let strategist = config
            .strategist_endpoint
            .as_ref()
            .map(|ep| {
                build_oai_client(&ep.api_key, &ep.url, local_http.clone(), None)
                    .context("Failed to build strategist client")
            })
            .transpose()?;

        let cloud = config
            .cloud_endpoint
            .as_ref()
            .map(|ep| {
                build_oai_client(
                    &ep.api_key,
                    &ep.url,
                    cloud_http,
                    Some(cloud_headers(&ep.api_key)),
                )
            })
            .transpose()
            .context("Failed to build cloud client (CLIAPIProxy)")?;

        // Build a TZ gateway client when SWARM_TENSORZERO_URL is set and a cloud
        // endpoint is configured. TZ handles auth to CLIAPIProxy internally via
        // the SWARM_CLOUD_API_KEY env var in tensorzero.toml — no x-api-key needed here.
        let cloud_tz = config
            .tensorzero_url
            .as_ref()
            .filter(|_| config.cloud_endpoint.is_some())
            .map(|tz_url| {
                let tz_http = http_client_with_timeout(cloud_timeout)?;
                tracing::info!(url = %tz_url, "Routing primary cloud client through TensorZero");
                build_oai_client("not-needed", &format!("{tz_url}/openai/v1"), tz_http, None)
            })
            .transpose()
            .context("Failed to build TensorZero cloud client")?;

        Ok(Self {
            local,
            coder,
            reasoning,
            strategist,
            cloud,
            cloud_tz,
        })
    }
}

/// Check if an inference endpoint is reachable and has a model loaded.
///
/// Queries `GET /v1/models` and optionally verifies that `expected_model` is in
/// the response. Returns `true` only if the endpoint responds and the model check
/// passes.
///
/// If `api_key` is provided (and not `"not-needed"`), sends both `x-api-key` and
/// `Authorization: Bearer` headers. CLIAPIProxy v6.8+ requires `x-api-key`; local
/// llama-server endpoints accept either or ignore auth entirely.
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
            req = req.bearer_auth(key).header("x-api-key", key);
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
                tracing::warn!(
                    endpoint = url,
                    expected_model = expected,
                    "Endpoint reachable but /v1/models response could not be parsed"
                );
                false
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

    // Serialize tests that mutate process-wide environment variables.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    #[test]
    fn test_default_config() {
        let _guard = ENV_LOCK.lock().unwrap();
        // Unset environment variables so we test the compiled-in defaults.
        // Without this, the test fails in dogfood where run-swarm.sh sets these.
        for var in [
            "SWARM_MAX_RETRIES",
            "SWARM_FAST_URL",
            "SWARM_FAST_MODEL",
            "SWARM_FAST_API_KEY",
            "SWARM_CODER_URL",
            "SWARM_CODER_MODEL",
            "SWARM_REASONING_URL",
            "SWARM_REASONING_MODEL",
            "SWARM_CLOUD_URL",
            "SWARM_CLOUD_API_KEY",
            "SWARM_CLOUD_MODEL",
            "SWARM_MAX_NO_CHANGE",
        ] {
            std::env::remove_var(var);
        }
        let config = SwarmConfig::default();
        assert_eq!(config.max_retries, 6);
        assert!(config.fast_endpoint.url.contains("vasp-03"));
        assert!(config.coder_endpoint.url.contains("vasp-01"));
        assert!(config.reasoning_endpoint.url.contains("vasp-02"));
        assert_eq!(config.fast_endpoint.model, "Qwen3-Coder-Next");
        assert_eq!(config.coder_endpoint.model, "Qwen3.5-122B-A10B");
        assert_eq!(config.reasoning_endpoint.model, "Qwen3.5-122B-A10B");
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

    #[test]
    fn test_cloud_fallback_matrix_default() {
        let matrix = CloudFallbackMatrix::default_matrix();
        assert_eq!(matrix.len(), 4);
        assert!(!matrix.is_empty());
        let primary = matrix.primary().unwrap();
        assert_eq!(primary.model, "claude-opus-4-6");
        assert_eq!(primary.tier_label, "primary");
        let fallbacks = matrix.fallbacks();
        assert_eq!(fallbacks.len(), 3);
        assert_eq!(fallbacks[0].model, "gemini-3.1-pro-high");
        assert_eq!(fallbacks[1].model, "claude-sonnet-4-6");
        assert_eq!(fallbacks[2].model, "gemini-3.1-flash-lite-preview");
    }

    #[test]
    fn test_cloud_fallback_matrix_empty() {
        let matrix = CloudFallbackMatrix { entries: vec![] };
        assert!(matrix.is_empty());
        assert_eq!(matrix.len(), 0);
        assert!(matrix.primary().is_none());
        assert!(matrix.fallbacks().is_empty());
    }

    #[test]
    fn test_cloud_fallback_matrix_in_config() {
        let config = SwarmConfig::proxy_config();
        assert_eq!(config.cloud_fallback_matrix.len(), 4);
        assert_eq!(
            config.cloud_fallback_matrix.primary().unwrap().model,
            "claude-opus-4-6"
        );
    }
}
