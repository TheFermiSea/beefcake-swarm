//! Round-robin pool of local inference endpoints with optional health awareness.
//!
//! Cloning an `EndpointPool` shares the `Arc<AtomicUsize>` counter, so parallel
//! `AgentFactory` clones all draw from the same sequence — each parallel issue
//! naturally lands on the next node.
//!
//! When a `ClusterHealth` monitor is attached, `next()` skips endpoints that
//! are marked `Down`. Falls back to normal round-robin when all nodes are down
//! (health check may be stale) or when health data is unavailable.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rig::providers::openai;
use tracing::debug;

use crate::cluster_health::ClusterHealth;
use crate::config::{ClientSet, SwarmConfig};

/// Tier names matching ClusterHealth's status map keys (all 3 nodes).
const TIER_NAMES_ALL: [&str; 3] = ["fast", "coder", "reasoning"];

pub struct EndpointPool {
    workers: Vec<(openai::CompletionsClient, String)>, // (client, model_name)
    tier_names: Vec<&'static str>,
    counter: Arc<AtomicUsize>,
    health: Option<ClusterHealth>,
}

impl EndpointPool {
    /// Create a pool with all 3 nodes (fast + coder + reasoning).
    ///
    /// When `SWARM_TENSORZERO_URL` is set, all model names are replaced with the
    /// TZ function name `tensorzero::function_name::worker_code_edit` so every
    /// inference is routed through TZ for experiment tracking.
    pub fn new(clients: &ClientSet, config: &SwarmConfig) -> Self {
        let tz_model = config
            .tensorzero_url
            .as_ref()
            .map(|_| "tensorzero::function_name::worker_code_edit".to_string());
        Self {
            workers: vec![
                (
                    clients.local.clone(),
                    tz_model
                        .clone()
                        .unwrap_or_else(|| config.fast_endpoint.model.clone()),
                ),
                (
                    clients.coder.clone(),
                    tz_model
                        .clone()
                        .unwrap_or_else(|| config.coder_endpoint.model.clone()),
                ),
                (
                    clients.reasoning.clone(),
                    tz_model.unwrap_or_else(|| config.reasoning_endpoint.model.clone()),
                ),
            ],
            tier_names: TIER_NAMES_ALL.to_vec(),
            counter: Arc::new(AtomicUsize::new(0)),
            health: None,
        }
    }

    /// Attach a health monitor. When set, `next()` skips down nodes.
    pub fn with_health(mut self, health: ClusterHealth) -> Self {
        self.health = Some(health);
        self
    }

    /// Select the next (client, model) pair, skipping unhealthy endpoints.
    ///
    /// Tries up to `workers.len()` candidates. If all are down (or health data
    /// is unavailable), returns the original round-robin pick — stale health
    /// data shouldn't block all work.
    ///
    /// Thread-safe: uses `fetch_add(Relaxed)` for the counter and `try_read()`
    /// on the health RwLock (non-blocking).
    pub fn next(&self) -> (&openai::CompletionsClient, &str) {
        let n = self.workers.len();
        let base = self.counter.fetch_add(1, Ordering::Relaxed);
        let default_idx = base % n;

        // Without health data, pure round-robin.
        let health = match &self.health {
            Some(h) => h,
            None => {
                let (client, model) = &self.workers[default_idx];
                return (client, model.as_str());
            }
        };

        // Try to read health status without blocking.
        let status_guard = match health.try_read_status() {
            Some(guard) => guard,
            None => {
                // Write lock held (health check in progress) — use default.
                let (client, model) = &self.workers[default_idx];
                return (client, model.as_str());
            }
        };

        // Try each candidate in round-robin order, skipping down nodes.
        for offset in 0..n {
            let idx = (base + offset) % n;
            let tier_name = self.tier_names[idx];
            let is_usable = status_guard
                .get(tier_name)
                .map(|s| s.is_usable())
                .unwrap_or(true); // unknown = assume usable

            if is_usable {
                if offset > 0 {
                    debug!(
                        skipped = offset,
                        selected = tier_name,
                        "Skipped unhealthy endpoint(s)"
                    );
                }
                let (client, model) = &self.workers[idx];
                return (client, model.as_str());
            }
        }

        // All down — use original pick anyway (health may be stale).
        debug!("All endpoints report down — using default round-robin");
        let (client, model) = &self.workers[default_idx];
        (client, model.as_str())
    }

    /// Select the next (client, model) pair, returning `None` if ALL endpoints
    /// are currently known-down according to the health monitor.
    ///
    /// Unlike `next()` — which falls back to the default round-robin pick even
    /// when all nodes are down — this variant lets the caller detect total
    /// failure and fall back to cloud models instead of dispatching to a dead
    /// local endpoint.
    ///
    /// Returns `None` only when a `ClusterHealth` monitor is attached AND all
    /// endpoints report `Down` status with a fresh (non-locked) read.  When
    /// health data is unavailable (no monitor, write-lock held, or any status is
    /// `Unknown`/`Degraded`) it falls back to the standard round-robin pick.
    pub fn next_or_none(&self) -> Option<(&openai::CompletionsClient, &str)> {
        let n = self.workers.len();
        let base = self.counter.fetch_add(1, Ordering::Relaxed);
        let default_idx = base % n;

        // Without health data, always return the default pick.
        let health = match &self.health {
            Some(h) => h,
            None => {
                let (client, model) = &self.workers[default_idx];
                return Some((client, model.as_str()));
            }
        };

        // Non-blocking read — fall back to round-robin if write lock is held.
        let status_guard = match health.try_read_status() {
            Some(guard) => guard,
            None => {
                let (client, model) = &self.workers[default_idx];
                return Some((client, model.as_str()));
            }
        };

        // Walk through all candidates in round-robin order.
        // Return the first usable endpoint; treat Unknown status as optimistically
        // usable (not yet probed — health check may not have run yet).
        // Only return None when every endpoint is *confirmed* Down.
        for offset in 0..n {
            let idx = (base + offset) % n;
            let tier_name = self.tier_names[idx];
            match status_guard.get(tier_name) {
                Some(s) if s.is_down() => {
                    // Confirmed down — keep searching.
                }
                _ => {
                    // Usable or Unknown — return this endpoint.
                    if offset > 0 {
                        debug!(
                            skipped = offset,
                            selected = tier_name,
                            "next_or_none: skipped unhealthy endpoint(s)"
                        );
                    }
                    let (client, model) = &self.workers[idx];
                    return Some((client, model.as_str()));
                }
            }
        }

        // Every endpoint is confirmed Down — signal total failure to caller.
        debug!("next_or_none: all endpoints confirmed down — returning None");
        None
    }

    /// Number of nodes in the pool (= 3 for the default cluster).
    pub fn capacity(&self) -> usize {
        self.workers.len()
    }

    /// Return a reference to the attached health monitor, if any.
    pub fn cluster_health(&self) -> Option<&ClusterHealth> {
        self.health.as_ref()
    }
}

impl Clone for EndpointPool {
    fn clone(&self) -> Self {
        Self {
            workers: self.workers.clone(), // clones CompletionsClient (cheap Arc)
            tier_names: self.tier_names.clone(),
            counter: Arc::clone(&self.counter), // shares the counter — key for round-robin
            health: self.health.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
}
