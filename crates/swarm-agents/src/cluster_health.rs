//! Cluster endpoint health monitoring.
//!
//! Background task pings `/health` on each inference endpoint every 30s.
//! `ClusterHealth` provides synchronous health queries for the orchestrator
//! to check before dispatching work to a potentially dead endpoint.
//!
//! This prevents the cloud manager from burning API credits by delegating
//! to workers that have silently crashed — the primary failure mode from
//! the 2026-03-02 parallel dogfood run.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::RwLock;
use tracing::{debug, info, warn};

use crate::config::{Endpoint, SwarmConfig};

/// How often to probe endpoints (seconds).
const HEALTH_CHECK_INTERVAL_SECS: u64 = 30;

/// Timeout for a single health check request.
const HEALTH_CHECK_TIMEOUT_SECS: u64 = 5;

/// Status of an inference endpoint.
#[derive(Debug, Clone)]
pub enum EndpointStatus {
    /// Endpoint is responding normally.
    Healthy {
        last_check: Instant,
        latency_ms: u64,
    },
    /// Endpoint is responding but slowly or with errors.
    Degraded {
        reason: String,
        since: Instant,
        last_check: Instant,
    },
    /// Endpoint is not responding.
    Down {
        reason: String,
        since: Instant,
        last_check: Instant,
    },
    /// Not yet checked.
    Unknown,
}

impl EndpointStatus {
    /// Returns true if the endpoint is healthy or degraded (still usable).
    pub fn is_usable(&self) -> bool {
        matches!(self, Self::Healthy { .. } | Self::Degraded { .. })
    }

    /// Returns true if the endpoint is down.
    pub fn is_down(&self) -> bool {
        matches!(self, Self::Down { .. })
    }

    /// Human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Healthy { .. } => "healthy",
            Self::Degraded { .. } => "degraded",
            Self::Down { .. } => "down",
            Self::Unknown => "unknown",
        }
    }
}

/// Descriptor for a monitored endpoint.
#[derive(Debug, Clone)]
pub struct MonitoredEndpoint {
    /// Display name (e.g., "fast/vasp-03").
    pub name: String,
    /// Base URL (e.g., "http://vasp-03:8081/v1").
    pub url: String,
    /// Expected model name (optional verification).
    pub expected_model: Option<String>,
    /// API key for auth (None or "not-needed" = skip auth).
    pub api_key: Option<String>,
}

impl MonitoredEndpoint {
    fn from_endpoint(ep: &Endpoint, name: &str) -> Self {
        let api_key = if ep.api_key == "not-needed" {
            None
        } else {
            Some(ep.api_key.clone())
        };
        Self {
            name: name.to_string(),
            url: ep.url.clone(),
            expected_model: Some(ep.model.clone()),
            api_key,
        }
    }
}

/// Shared cluster health state.
///
/// Thread-safe: uses `Arc<RwLock<...>>` internally so it can be cloned
/// and shared between the background monitor and the orchestrator.
#[derive(Clone)]
pub struct ClusterHealth {
    endpoints: Arc<Vec<MonitoredEndpoint>>,
    status: Arc<RwLock<HashMap<String, EndpointStatus>>>,
}

impl ClusterHealth {
    /// Build from SwarmConfig — monitors all 3 local inference endpoints.
    pub fn from_config(config: &SwarmConfig) -> Self {
        let endpoints = vec![
            MonitoredEndpoint::from_endpoint(&config.fast_endpoint, "fast"),
            MonitoredEndpoint::from_endpoint(&config.coder_endpoint, "coder"),
            MonitoredEndpoint::from_endpoint(&config.reasoning_endpoint, "reasoning"),
        ];

        let mut status = HashMap::new();
        for ep in &endpoints {
            status.insert(ep.name.clone(), EndpointStatus::Unknown);
        }

        Self {
            endpoints: Arc::new(endpoints),
            status: Arc::new(RwLock::new(status)),
        }
    }

    /// Run a one-shot health check on all endpoints. Returns the number of healthy endpoints.
    ///
    /// Use this for preflight validation before entering the main loop.
    pub async fn check_all_now(&self) -> usize {
        let mut healthy = 0;
        for ep in self.endpoints.iter() {
            let result = probe_endpoint(&ep.url, ep.api_key.as_deref()).await;
            let status = match result {
                ProbeResult::Healthy { latency_ms } => {
                    healthy += 1;
                    info!(
                        endpoint = %ep.name,
                        url = %ep.url,
                        latency_ms,
                        "Endpoint healthy"
                    );
                    EndpointStatus::Healthy {
                        last_check: Instant::now(),
                        latency_ms,
                    }
                }
                ProbeResult::Degraded { reason } => {
                    healthy += 1; // still usable
                    warn!(
                        endpoint = %ep.name,
                        url = %ep.url,
                        reason = %reason,
                        "Endpoint degraded"
                    );
                    EndpointStatus::Degraded {
                        reason,
                        since: Instant::now(),
                        last_check: Instant::now(),
                    }
                }
                ProbeResult::Down { reason } => {
                    warn!(
                        endpoint = %ep.name,
                        url = %ep.url,
                        reason = %reason,
                        "Endpoint DOWN"
                    );
                    EndpointStatus::Down {
                        reason,
                        since: Instant::now(),
                        last_check: Instant::now(),
                    }
                }
            };
            self.status.write().await.insert(ep.name.clone(), status);
        }
        healthy
    }

    /// Check if a specific tier's endpoint is usable (healthy or degraded).
    pub async fn is_tier_usable(&self, tier_name: &str) -> bool {
        let map = self.status.read().await;
        map.get(tier_name).map(|s| s.is_usable()).unwrap_or(false)
    }

    /// Get status for a specific tier.
    pub async fn tier_status(&self, tier_name: &str) -> EndpointStatus {
        let map = self.status.read().await;
        map.get(tier_name)
            .cloned()
            .unwrap_or(EndpointStatus::Unknown)
    }

    /// Get a summary of all endpoint statuses (for logging/tools).
    pub async fn summary(&self) -> String {
        let map = self.status.read().await;
        let mut parts = Vec::new();
        for ep in self.endpoints.iter() {
            let status = map.get(&ep.name).map(|s| s.label()).unwrap_or("unknown");
            parts.push(format!("{}={}", ep.name, status));
        }
        parts.join(", ")
    }

    /// Non-blocking read of the status map for synchronous callers.
    ///
    /// Returns `None` if a write lock is held (health check in progress).
    /// Used by `EndpointPool::next()` to skip down nodes without blocking.
    pub fn try_read_status(
        &self,
    ) -> Option<tokio::sync::RwLockReadGuard<'_, HashMap<String, EndpointStatus>>> {
        self.status.try_read().ok()
    }

    /// Count how many endpoints are usable (healthy or degraded).
    pub async fn usable_count(&self) -> usize {
        let map = self.status.read().await;
        map.values().filter(|s| s.is_usable()).count()
    }

    /// Spawn a background task that periodically checks all endpoints.
    ///
    /// Returns a `JoinHandle` that can be aborted to stop monitoring.
    pub fn spawn_monitor(&self) -> tokio::task::JoinHandle<()> {
        let health = self.clone();
        tokio::spawn(async move {
            let interval = Duration::from_secs(HEALTH_CHECK_INTERVAL_SECS);
            loop {
                tokio::time::sleep(interval).await;
                for ep in health.endpoints.iter() {
                    let result = probe_endpoint(&ep.url, ep.api_key.as_deref()).await;
                    let now = Instant::now();

                    let new_status = {
                        let current = health.status.read().await;
                        let prev = current.get(&ep.name);

                        match result {
                            ProbeResult::Healthy { latency_ms } => {
                                if let Some(EndpointStatus::Down { .. }) = prev {
                                    info!(
                                        endpoint = %ep.name,
                                        latency_ms,
                                        "Endpoint recovered"
                                    );
                                }
                                EndpointStatus::Healthy {
                                    last_check: now,
                                    latency_ms,
                                }
                            }
                            ProbeResult::Degraded { reason } => {
                                let since = match prev {
                                    Some(EndpointStatus::Degraded { since, .. }) => *since,
                                    _ => now,
                                };
                                debug!(
                                    endpoint = %ep.name,
                                    reason = %reason,
                                    "Endpoint degraded"
                                );
                                EndpointStatus::Degraded {
                                    reason,
                                    since,
                                    last_check: now,
                                }
                            }
                            ProbeResult::Down { reason } => {
                                let since = match prev {
                                    Some(EndpointStatus::Down { since, .. }) => *since,
                                    _ => {
                                        warn!(
                                            endpoint = %ep.name,
                                            reason = %reason,
                                            "Endpoint went DOWN"
                                        );
                                        now
                                    }
                                };
                                EndpointStatus::Down {
                                    reason,
                                    since,
                                    last_check: now,
                                }
                            }
                        }
                    };

                    health
                        .status
                        .write()
                        .await
                        .insert(ep.name.clone(), new_status);
                }
            }
        })
    }
}

/// Result of probing an endpoint.
enum ProbeResult {
    Healthy { latency_ms: u64 },
    Degraded { reason: String },
    Down { reason: String },
}

/// Probe an endpoint's health by hitting GET /health.
///
/// Falls back to GET /v1/models if /health is not available (llama-server
/// serves /health but CLIAPIProxy may not).
async fn probe_endpoint(base_url: &str, api_key: Option<&str>) -> ProbeResult {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(HEALTH_CHECK_TIMEOUT_SECS))
        .build()
        .unwrap_or_default();

    // Try /health first (llama-server native).
    // TZ OpenAI compat uses /openai/v1 as base — strip that to reach /health.
    let health_url = if base_url.contains("/openai/v1") {
        base_url
            .trim_end_matches("/v1")
            .trim_end_matches("/openai")
            .to_string()
            + "/health"
    } else {
        base_url.trim_end_matches("/v1").to_string() + "/health"
    };
    let start = Instant::now();

    let mut req = client.get(&health_url);
    if let Some(key) = api_key {
        req = req.bearer_auth(key).header("x-api-key", key);
    }

    match req.send().await {
        Ok(resp) => {
            let latency_ms = start.elapsed().as_millis() as u64;
            if resp.status().is_success() {
                // Check for degraded state in response body
                if let Ok(body) = resp.text().await {
                    if body.contains("no slot available") || body.contains("loading") {
                        return ProbeResult::Degraded {
                            reason: "all slots busy or model loading".to_string(),
                        };
                    }
                }
                ProbeResult::Healthy { latency_ms }
            } else if resp.status().as_u16() == 503 {
                ProbeResult::Degraded {
                    reason: "HTTP 503 (busy/loading)".to_string(),
                }
            } else {
                ProbeResult::Down {
                    reason: format!("HTTP {}", resp.status()),
                }
            }
        }
        Err(e) => {
            // /health not available — try /v1/models as fallback
            let models_url = format!("{base_url}/models");
            let start2 = Instant::now();
            let mut req2 = client.get(&models_url);
            if let Some(key) = api_key {
                req2 = req2.bearer_auth(key).header("x-api-key", key);
            }

            match req2.send().await {
                Ok(resp) if resp.status().is_success() => {
                    let latency_ms = start2.elapsed().as_millis() as u64;
                    ProbeResult::Healthy { latency_ms }
                }
                Ok(resp) => ProbeResult::Down {
                    reason: format!("/health failed ({}), /models HTTP {}", e, resp.status()),
                },
                Err(e2) => ProbeResult::Down {
                    reason: format!("unreachable: {e2}"),
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_endpoint_status_labels() {
        let healthy = EndpointStatus::Healthy {
            last_check: Instant::now(),
            latency_ms: 42,
        };
        assert!(healthy.is_usable());
        assert!(!healthy.is_down());
        assert_eq!(healthy.label(), "healthy");

        let degraded = EndpointStatus::Degraded {
            reason: "busy".into(),
            since: Instant::now(),
            last_check: Instant::now(),
        };
        assert!(degraded.is_usable());
        assert!(!degraded.is_down());
        assert_eq!(degraded.label(), "degraded");

        let down = EndpointStatus::Down {
            reason: "timeout".into(),
            since: Instant::now(),
            last_check: Instant::now(),
        };
        assert!(!down.is_usable());
        assert!(down.is_down());
        assert_eq!(down.label(), "down");

        let unknown = EndpointStatus::Unknown;
        assert!(!unknown.is_usable());
        assert!(!unknown.is_down());
        assert_eq!(unknown.label(), "unknown");
    }

    #[test]
    fn test_from_config_creates_3_endpoints() {
        let config = SwarmConfig::default();
        let health = ClusterHealth::from_config(&config);
        assert_eq!(health.endpoints.len(), 3);
        assert_eq!(health.endpoints[0].name, "fast");
        assert_eq!(health.endpoints[1].name, "coder");
        assert_eq!(health.endpoints[2].name, "reasoning");
    }
}
