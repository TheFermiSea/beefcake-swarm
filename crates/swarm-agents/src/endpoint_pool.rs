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

/// Tier names for the worker-only pool (excludes 27B scout).
const TIER_NAMES_WORKERS: [&str; 2] = ["coder", "reasoning"];

pub struct EndpointPool {
    workers: Vec<(openai::CompletionsClient, String)>, // (client, model_name)
    tier_names: Vec<&'static str>,
    counter: Arc<AtomicUsize>,
    health: Option<ClusterHealth>,
}

impl EndpointPool {
    /// Create a pool with all 3 nodes (fast + coder + reasoning).
    /// Use for health monitoring and non-code-generation tasks.
    pub fn new(clients: &ClientSet, config: &SwarmConfig) -> Self {
        Self {
            workers: vec![
                (clients.local.clone(), config.fast_endpoint.model.clone()),
                (clients.coder.clone(), config.coder_endpoint.model.clone()),
                (
                    clients.reasoning.clone(),
                    config.reasoning_endpoint.model.clone(),
                ),
            ],
            tier_names: TIER_NAMES_ALL.to_vec(),
            counter: Arc::new(AtomicUsize::new(0)),
            health: None,
        }
    }

    /// Create a worker-only pool with coder + reasoning (both 122B).
    /// Excludes the 27B scout/reviewer which is not tuned for code generation.
    pub fn new_workers(clients: &ClientSet, config: &SwarmConfig) -> Self {
        Self {
            workers: vec![
                (clients.coder.clone(), config.coder_endpoint.model.clone()),
                (
                    clients.reasoning.clone(),
                    config.reasoning_endpoint.model.clone(),
                ),
            ],
            tier_names: TIER_NAMES_WORKERS.to_vec(),
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

    /// Verify round-robin wraps correctly across 3 nodes.
    ///
    /// We can't easily build real `CompletionsClient`s in unit tests without
    /// network access, so we test the counter logic directly.
    #[test]
    fn round_robin_counter_cycles() {
        let counter = Arc::new(AtomicUsize::new(0));
        let n = 3usize;

        let indices: Vec<usize> = (0..6)
            .map(|_| counter.fetch_add(1, Ordering::Relaxed) % n)
            .collect();

        // Should cycle: 0, 1, 2, 0, 1, 2
        assert_eq!(indices, vec![0, 1, 2, 0, 1, 2]);
        // Position 0 == position 3 (wraps at 3)
        assert_eq!(indices[0], indices[3]);
        // Adjacent positions differ
        assert_ne!(indices[0], indices[1]);
    }

    /// Verify worker-only pool cycles across 2 nodes (coder, reasoning).
    #[test]
    fn worker_pool_cycles_two_nodes() {
        let counter = Arc::new(AtomicUsize::new(0));
        let n = 2usize;

        let indices: Vec<usize> = (0..6)
            .map(|_| counter.fetch_add(1, Ordering::Relaxed) % n)
            .collect();

        // Should cycle: 0, 1, 0, 1, 0, 1
        assert_eq!(indices, vec![0, 1, 0, 1, 0, 1]);
    }

    #[test]
    fn shared_counter_across_clones() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter2 = Arc::clone(&counter);
        let n = 3usize;

        // Simulate two parallel clones sharing the counter
        let idx_a = counter.fetch_add(1, Ordering::Relaxed) % n;
        let idx_b = counter2.fetch_add(1, Ordering::Relaxed) % n;
        let idx_c = counter.fetch_add(1, Ordering::Relaxed) % n;

        // Each clone draws the next slot
        assert_eq!(idx_a, 0);
        assert_eq!(idx_b, 1);
        assert_eq!(idx_c, 2);
    }
}
