//! Round-robin pool of local inference endpoints.
//!
//! Cloning an `EndpointPool` shares the `Arc<AtomicUsize>` counter, so parallel
//! `AgentFactory` clones all draw from the same sequence — each parallel issue
//! naturally lands on the next node.
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use rig::providers::openai;

use crate::config::{ClientSet, SwarmConfig};

pub struct EndpointPool {
    workers: Vec<(openai::CompletionsClient, String)>, // (client, model_name)
    counter: Arc<AtomicUsize>,
}

impl EndpointPool {
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
            counter: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Select the next (client, model) pair in round-robin order.
    ///
    /// Thread-safe: `fetch_add(Relaxed)` is sufficient — we only need global
    /// progress, not strict ordering between tasks.
    pub fn next(&self) -> (&openai::CompletionsClient, &str) {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.workers.len();
        let (client, model) = &self.workers[idx];
        (client, model.as_str())
    }

    /// Number of nodes in the pool (= 3 for the default cluster).
    pub fn capacity(&self) -> usize {
        self.workers.len()
    }
}

impl Clone for EndpointPool {
    fn clone(&self) -> Self {
        Self {
            workers: self.workers.clone(), // clones CompletionsClient (cheap Arc)
            counter: Arc::clone(&self.counter), // shares the counter — key for round-robin
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
