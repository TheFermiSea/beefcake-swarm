//! Provider Registry — model capability and health metadata
//!
//! Tracks which providers are available, their capabilities (context window,
//! supported features), and live health metadata (availability, latency, error rates).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::state::types::{ModelId, ModelKind};

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Capabilities of a model provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    /// Maximum context window in tokens
    pub context_window: u32,
    /// Maximum output tokens
    pub max_output_tokens: u32,
    /// Whether the model supports streaming
    pub supports_streaming: bool,
    /// Whether the model supports function/tool calling
    pub supports_tool_calls: bool,
    /// Whether the model produces reasoning/chain-of-thought
    pub supports_reasoning: bool,
    /// Approximate tokens per second (for local models)
    pub tokens_per_sec: Option<u32>,
    /// Whether the model runs locally (vs cloud API)
    pub is_local: bool,
    /// Human-readable description of the model's specialty
    pub specialty: String,
}

impl ProviderCapabilities {
    /// Default capabilities for each known ModelId
    pub fn for_model(model: ModelId) -> Self {
        match model {
            ModelId::Opus45 => Self {
                context_window: 200_000,
                max_output_tokens: 32_000,
                supports_streaming: true,
                supports_tool_calls: true,
                supports_reasoning: true,
                tokens_per_sec: None,
                is_local: false,
                specialty: "Architecture, safety, code review".to_string(),
            },
            ModelId::Gemini3Pro => Self {
                context_window: 1_000_000,
                max_output_tokens: 8_192,
                supports_streaming: true,
                supports_tool_calls: true,
                supports_reasoning: false,
                tokens_per_sec: None,
                is_local: false,
                specialty: "Repository context, documentation, code navigation".to_string(),
            },
            ModelId::Qwen35 => Self {
                context_window: 32_768,
                max_output_tokens: 8_192,
                supports_streaming: true,
                supports_tool_calls: false,
                supports_reasoning: true,
                tokens_per_sec: Some(8),
                is_local: true,
                specialty: "Reasoning, planning, task decomposition".to_string(),
            },
            ModelId::HydraCoder => Self {
                context_window: 16_384,
                max_output_tokens: 4_096,
                supports_streaming: true,
                supports_tool_calls: false,
                supports_reasoning: false,
                tokens_per_sec: Some(40),
                is_local: true,
                specialty: "Rust code generation and error fixing".to_string(),
            },
        }
    }
}

/// Live health metadata for a provider
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderHealth {
    /// Whether the provider is currently reachable
    pub available: bool,
    /// Average response latency in milliseconds (rolling window)
    pub avg_latency_ms: u64,
    /// Number of successful requests in the last window
    pub success_count: u64,
    /// Number of failed requests in the last window
    pub error_count: u64,
    /// Last time health was checked (as Unix timestamp seconds)
    pub last_checked_secs: u64,
    /// Optional human-readable status message
    pub status_message: Option<String>,
}

impl ProviderHealth {
    /// Create a default healthy state
    pub fn healthy() -> Self {
        Self {
            available: true,
            avg_latency_ms: 0,
            success_count: 0,
            error_count: 0,
            last_checked_secs: unix_now(),
            status_message: None,
        }
    }

    /// Create an unavailable state with a reason
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            available: false,
            avg_latency_ms: 0,
            success_count: 0,
            error_count: 0,
            last_checked_secs: unix_now(),
            status_message: Some(reason.into()),
        }
    }

    /// Compute success rate (0.0 - 1.0)
    pub fn success_rate(&self) -> f32 {
        let total = self.success_count + self.error_count;
        if total == 0 {
            1.0
        } else {
            self.success_count as f32 / total as f32
        }
    }

    /// Record a successful request with latency
    pub fn record_success(&mut self, latency_ms: u64) {
        self.avg_latency_ms =
            (self.avg_latency_ms * self.success_count + latency_ms) / (self.success_count + 1);
        self.success_count += 1;
        self.last_checked_secs = unix_now();
    }

    /// Record a failed request
    pub fn record_failure(&mut self) {
        self.error_count += 1;
        self.last_checked_secs = unix_now();
    }
}

/// A registered provider entry combining identity, capabilities, and health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderEntry {
    pub model_id: ModelId,
    pub kind: ModelKind,
    pub api_name: String,
    pub capabilities: ProviderCapabilities,
    pub health: ProviderHealth,
}

impl ProviderEntry {
    pub fn new(model_id: ModelId) -> Self {
        Self {
            kind: model_id.kind(),
            api_name: model_id.api_name().to_string(),
            capabilities: ProviderCapabilities::for_model(model_id),
            health: ProviderHealth::healthy(),
            model_id,
        }
    }

    /// Whether this provider is usable (available + healthy enough)
    pub fn is_usable(&self) -> bool {
        self.health.available && self.health.success_rate() >= 0.5
    }
}

/// Registry of all known providers with their capabilities and health
pub struct ProviderRegistry {
    entries: HashMap<ModelId, ProviderEntry>,
}

impl ProviderRegistry {
    /// Create a registry pre-populated with all known models
    pub fn new() -> Self {
        let mut entries = HashMap::new();
        for &model in ModelId::all() {
            entries.insert(model, ProviderEntry::new(model));
        }
        Self { entries }
    }

    /// Get a provider entry by model ID
    pub fn get(&self, model: ModelId) -> Option<&ProviderEntry> {
        self.entries.get(&model)
    }

    /// Get a mutable provider entry for health updates
    pub fn get_mut(&mut self, model: ModelId) -> Option<&mut ProviderEntry> {
        self.entries.get_mut(&model)
    }

    /// Update health for a provider
    pub fn update_health(&mut self, model: ModelId, health: ProviderHealth) {
        if let Some(entry) = self.entries.get_mut(&model) {
            entry.health = health;
        }
    }

    /// Get all usable providers of a given kind
    pub fn usable_by_kind(&self, kind: ModelKind) -> Vec<&ProviderEntry> {
        self.entries
            .values()
            .filter(|e| e.kind == kind && e.is_usable())
            .collect()
    }

    /// Get all providers sorted by success rate (best first)
    pub fn ranked_by_health(&self) -> Vec<&ProviderEntry> {
        let mut entries: Vec<&ProviderEntry> = self.entries.values().collect();
        entries.sort_by(|a, b| {
            b.health
                .success_rate()
                .partial_cmp(&a.health.success_rate())
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.health.avg_latency_ms.cmp(&b.health.avg_latency_ms))
        });
        entries
    }

    /// Mark a provider as unavailable
    pub fn mark_unavailable(&mut self, model: ModelId, reason: impl Into<String>) {
        if let Some(entry) = self.entries.get_mut(&model) {
            entry.health = ProviderHealth::unavailable(reason);
        }
    }

    /// Mark a provider as available
    pub fn mark_available(&mut self, model: ModelId) {
        if let Some(entry) = self.entries.get_mut(&model) {
            entry.health.available = true;
            entry.health.status_message = None;
            entry.health.last_checked_secs = unix_now();
        }
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_populated() {
        let registry = ProviderRegistry::new();
        assert!(registry.get(ModelId::Opus45).is_some());
        assert!(registry.get(ModelId::Gemini3Pro).is_some());
        assert!(registry.get(ModelId::Qwen35).is_some());
        assert!(registry.get(ModelId::HydraCoder).is_some());
    }

    #[test]
    fn test_provider_capabilities() {
        let caps = ProviderCapabilities::for_model(ModelId::Opus45);
        assert_eq!(caps.context_window, 200_000);
        assert!(caps.supports_tool_calls);
        assert!(!caps.is_local);

        let caps = ProviderCapabilities::for_model(ModelId::HydraCoder);
        assert_eq!(caps.context_window, 16_384);
        assert_eq!(caps.tokens_per_sec, Some(40));
        assert!(caps.is_local);
    }

    #[test]
    fn test_health_success_rate() {
        let mut h = ProviderHealth::healthy();
        assert_eq!(h.success_rate(), 1.0);

        h.record_success(100);
        h.record_failure();
        assert_eq!(h.success_rate(), 0.5);
    }

    #[test]
    fn test_provider_entry_usable() {
        let entry = ProviderEntry::new(ModelId::Opus45);
        assert!(entry.is_usable());
    }

    #[test]
    fn test_mark_unavailable() {
        let mut registry = ProviderRegistry::new();
        registry.mark_unavailable(ModelId::HydraCoder, "maintenance");
        let entry = registry.get(ModelId::HydraCoder).unwrap();
        assert!(!entry.health.available);
        assert!(!entry.is_usable());
    }

    #[test]
    fn test_usable_by_kind() {
        let mut registry = ProviderRegistry::new();
        registry.mark_unavailable(ModelId::HydraCoder, "down");
        let workers = registry.usable_by_kind(ModelKind::Worker);
        assert!(workers.is_empty());

        let managers = registry.usable_by_kind(ModelKind::Manager);
        assert_eq!(managers.len(), 3);
    }

    #[test]
    fn test_ranked_by_health() {
        let mut registry = ProviderRegistry::new();
        // Degrade Opus45 by recording failures
        if let Some(entry) = registry.get_mut(ModelId::Opus45) {
            entry.health.record_failure();
            entry.health.record_failure();
        }
        let ranked = registry.ranked_by_health();
        assert_eq!(ranked.len(), 4);
        // Opus45 should be ranked lower due to failures
        let opus_pos = ranked
            .iter()
            .position(|e| e.model_id == ModelId::Opus45)
            .unwrap();
        assert!(opus_pos > 0);
    }
}
