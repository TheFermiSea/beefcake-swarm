//! Circuit breaker and fallback ladder for model routing.
//!
//! The circuit breaker tracks consecutive failures per [`ModelId`]. When
//! failures exceed a configurable threshold the circuit *opens* and the
//! model is temporarily skipped. After a cooldown the circuit enters
//! *half-open* state to probe recovery.
//!
//! The [`FallbackLadder`] walks an ordered list of models, skipping any
//! whose circuit is currently open.

use crate::state::types::ModelId;
use std::collections::HashMap;

/// Circuit breaker state for a single model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Healthy — requests allowed.
    Closed,
    /// Tripped — requests blocked until cooldown expires.
    Open,
    /// Cooldown expired — one probe request allowed.
    HalfOpen,
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Per-model circuit breaker tracking consecutive failures.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    consecutive_failures: HashMap<ModelId, u32>,
    last_failure_secs: HashMap<ModelId, u64>,
    /// Consecutive failures before the circuit opens.
    pub failure_threshold: u32,
    /// Seconds after last failure before Open → HalfOpen.
    pub cooldown_secs: u64,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(failure_threshold: u32, cooldown_secs: u64) -> Self {
        Self {
            consecutive_failures: HashMap::new(),
            last_failure_secs: HashMap::new(),
            failure_threshold,
            cooldown_secs,
        }
    }

    /// Record a success — resets circuit to Closed.
    pub fn record_success(&mut self, model: ModelId) {
        self.consecutive_failures.remove(&model);
        self.last_failure_secs.remove(&model);
    }

    /// Record a failure — may trip circuit to Open.
    pub fn record_failure(&mut self, model: ModelId) {
        let count = self.consecutive_failures.entry(model).or_insert(0);
        *count += 1;
        self.last_failure_secs.insert(model, unix_now());
    }

    /// Current state of the circuit for `model`.
    pub fn state(&self, model: ModelId) -> CircuitState {
        let failures = self.consecutive_failures.get(&model).copied().unwrap_or(0);
        if failures < self.failure_threshold {
            return CircuitState::Closed;
        }
        let last = self.last_failure_secs.get(&model).copied().unwrap_or(0);
        if unix_now().saturating_sub(last) >= self.cooldown_secs {
            CircuitState::HalfOpen
        } else {
            CircuitState::Open
        }
    }

    /// Whether the model is available (Closed or HalfOpen).
    pub fn is_available(&self, model: ModelId) -> bool {
        !matches!(self.state(model), CircuitState::Open)
    }

    /// Consecutive failures recorded for `model`.
    pub fn failure_count(&self, model: ModelId) -> u32 {
        self.consecutive_failures.get(&model).copied().unwrap_or(0)
    }
}

impl Default for CircuitBreaker {
    fn default() -> Self {
        Self::new(3, 60)
    }
}

/// Ordered fallback ladder of models.
///
/// The first model whose circuit is not open is returned.
#[derive(Debug, Clone)]
pub struct FallbackLadder {
    models: Vec<ModelId>,
}

impl FallbackLadder {
    /// Create a ladder from an ordered list of models.
    pub fn new(models: Vec<ModelId>) -> Self {
        Self { models }
    }

    /// Default ladder: HydraCoder → Qwen35 → Opus45 → Gemini3Pro.
    pub fn default_ladder() -> Self {
        Self::new(vec![
            ModelId::HydraCoder,
            ModelId::Qwen35,
            ModelId::Opus45,
            ModelId::Gemini3Pro,
        ])
    }

    /// First model in the ladder whose circuit is not open.
    pub fn next_available(&self, breaker: &CircuitBreaker) -> Option<ModelId> {
        self.models
            .iter()
            .copied()
            .find(|m| breaker.is_available(*m))
    }

    /// The ordered list of models.
    pub fn models(&self) -> &[ModelId] {
        &self.models
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_starts_closed() {
        let cb = CircuitBreaker::default();
        assert_eq!(cb.state(ModelId::HydraCoder), CircuitState::Closed);
        assert!(cb.is_available(ModelId::HydraCoder));
    }

    #[test]
    fn test_circuit_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(2, 9999);
        cb.record_failure(ModelId::HydraCoder);
        assert_eq!(cb.state(ModelId::HydraCoder), CircuitState::Closed);
        cb.record_failure(ModelId::HydraCoder);
        assert_eq!(cb.state(ModelId::HydraCoder), CircuitState::Open);
        assert!(!cb.is_available(ModelId::HydraCoder));
    }

    #[test]
    fn test_success_resets_circuit() {
        let mut cb = CircuitBreaker::new(2, 9999);
        cb.record_failure(ModelId::Opus45);
        cb.record_failure(ModelId::Opus45);
        assert_eq!(cb.state(ModelId::Opus45), CircuitState::Open);
        cb.record_success(ModelId::Opus45);
        assert_eq!(cb.state(ModelId::Opus45), CircuitState::Closed);
    }

    #[test]
    fn test_half_open_after_cooldown() {
        let mut cb = CircuitBreaker::new(1, 0);
        cb.record_failure(ModelId::Qwen35);
        assert_eq!(cb.state(ModelId::Qwen35), CircuitState::HalfOpen);
        assert!(cb.is_available(ModelId::Qwen35));
    }

    #[test]
    fn test_fallback_skips_open() {
        let mut cb = CircuitBreaker::new(1, 9999);
        cb.record_failure(ModelId::HydraCoder);
        let ladder = FallbackLadder::default_ladder();
        assert_eq!(ladder.next_available(&cb), Some(ModelId::Qwen35));
    }

    #[test]
    fn test_fallback_all_open() {
        let mut cb = CircuitBreaker::new(1, 9999);
        for &m in ModelId::all() {
            cb.record_failure(m);
        }
        let ladder = FallbackLadder::default_ladder();
        assert_eq!(ladder.next_available(&cb), None);
    }

    #[test]
    fn test_fallback_returns_first() {
        let cb = CircuitBreaker::default();
        let ladder = FallbackLadder::default_ladder();
        assert_eq!(ladder.next_available(&cb), Some(ModelId::HydraCoder));
    }
}
