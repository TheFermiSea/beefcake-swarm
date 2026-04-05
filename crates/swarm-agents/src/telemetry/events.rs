use serde::{Deserialize, Serialize};
use std::path::Path;

/// A structured swarm event emitted in real-time during orchestration.
///
/// Each event is a self-contained JSON record written to `telemetry.jsonl` and
/// optionally POSTed to a webhook URL. This supplements the batch
/// `SessionMetrics` with live, per-action granularity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmEvent {
    /// Event type (dot-notation, e.g. `swarm.issue.started`).
    pub event: String,
    /// ISO 8601 timestamp.
    pub timestamp: String,
    /// Issue ID this event relates to.
    pub issue_id: String,
    /// Typed payload.
    #[serde(flatten)]
    pub payload: SwarmEventPayload,
}

/// Typed payloads for each event kind.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SwarmEventPayload {
    IssueStarted {
        title: String,
        priority: Option<u8>,
        tier: String,
    },
    IterationCompleted {
        iteration: u32,
        tier: String,
        error_count: usize,
        no_change: bool,
        elapsed_ms: u64,
    },
    WorkerFailed {
        model: String,
        error: String,
        iteration: u32,
    },
    HealthCheck {
        endpoint: String,
        healthy: bool,
        latency_ms: Option<u64>,
    },
    IssueResolved {
        success: bool,
        total_iterations: u32,
        elapsed_ms: u64,
    },
}

/// Emits structured events to a JSONL file and optional webhook.
pub struct SwarmEventEmitter {
    /// Path to the JSONL event log (typically `<repo_root>/.swarm-events.jsonl`).
    event_log_path: std::path::PathBuf,
    /// Optional webhook URL for critical events.
    webhook_url: Option<String>,
    /// HTTP client for webhook delivery (reused across calls).
    http_client: Option<reqwest::Client>,
}

impl SwarmEventEmitter {
    /// Create a new emitter writing to the given repo root.
    ///
    /// Reads `SWARM_WEBHOOK_URL` from the environment for webhook delivery.
    pub fn new(repo_root: &Path) -> Self {
        let webhook_url = std::env::var("SWARM_WEBHOOK_URL")
            .ok()
            .filter(|u| !u.is_empty());
        let http_client = webhook_url.as_ref().map(|_| reqwest::Client::new());
        Self {
            event_log_path: repo_root.join(".swarm-events.jsonl"),
            webhook_url,
            http_client,
        }
    }

    /// Emit a structured event. Writes to the event log and optionally fires a webhook.
    pub fn emit(&self, event: SwarmEvent) {
        // Write to JSONL
        if let Ok(json) = serde_json::to_string(&event) {
            use std::io::Write;
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.event_log_path)
            {
                Ok(mut file) => {
                    if let Err(e) = writeln!(file, "{json}") {
                        tracing::warn!(
                            error = %e,
                            path = %self.event_log_path.display(),
                            "Failed to write JSONL event"
                        );
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        path = %self.event_log_path.display(),
                        "Failed to open JSONL event log"
                    );
                }
            }

            // Also emit as a structured tracing event for log aggregation
            tracing::info!(
                target: "swarm.events",
                event_type = %event.event,
                issue_id = %event.issue_id,
                "structured_event"
            );
        }

        // Webhook delivery for critical events (non-blocking fire-and-forget)
        if let (Some(url), Some(client)) = (&self.webhook_url, &self.http_client) {
            if Self::is_critical(&event) {
                if tokio::runtime::Handle::try_current().is_ok() {
                    let url = url.clone();
                    let client = client.clone();
                    let event_clone = event;
                    tokio::spawn(async move {
                        let _ = client
                            .post(&url)
                            .json(&event_clone)
                            .timeout(std::time::Duration::from_secs(5))
                            .send()
                            .await;
                    });
                } else {
                    tracing::debug!("No Tokio runtime — skipping webhook delivery");
                }
            }
        }
    }

    /// Helper to emit an issue-started event.
    pub fn issue_started(&self, issue_id: &str, title: &str, priority: Option<u8>, tier: &str) {
        self.emit(SwarmEvent {
            event: "swarm.issue.started".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::IssueStarted {
                title: title.into(),
                priority,
                tier: tier.into(),
            },
        });
    }

    /// Helper to emit an iteration-completed event.
    pub fn iteration_completed(
        &self,
        issue_id: &str,
        iteration: u32,
        tier: &str,
        error_count: usize,
        no_change: bool,
        elapsed_ms: u64,
    ) {
        self.emit(SwarmEvent {
            event: "swarm.iteration.completed".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::IterationCompleted {
                iteration,
                tier: tier.into(),
                error_count,
                no_change,
                elapsed_ms,
            },
        });
    }

    /// Helper to emit a worker-failed event.
    pub fn worker_failed(&self, issue_id: &str, model: &str, error: &str, iteration: u32) {
        self.emit(SwarmEvent {
            event: "swarm.worker.failed".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::WorkerFailed {
                model: model.into(),
                error: error.into(),
                iteration,
            },
        });
    }

    /// Helper to emit a health-check event.
    pub fn health_check(
        &self,
        issue_id: &str,
        endpoint: &str,
        healthy: bool,
        latency_ms: Option<u64>,
    ) {
        self.emit(SwarmEvent {
            event: "swarm.health.check".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::HealthCheck {
                endpoint: endpoint.into(),
                healthy,
                latency_ms,
            },
        });
    }

    /// Helper to emit an issue-resolved event.
    pub fn issue_resolved(
        &self,
        issue_id: &str,
        success: bool,
        total_iterations: u32,
        elapsed_ms: u64,
    ) {
        self.emit(SwarmEvent {
            event: "swarm.issue.resolved".into(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            issue_id: issue_id.into(),
            payload: SwarmEventPayload::IssueResolved {
                success,
                total_iterations,
                elapsed_ms,
            },
        });
    }

    /// Whether an event is critical enough to warrant webhook delivery.
    fn is_critical(event: &SwarmEvent) -> bool {
        matches!(
            event.payload,
            SwarmEventPayload::WorkerFailed { .. }
                | SwarmEventPayload::IssueResolved { .. }
                | SwarmEventPayload::HealthCheck { healthy: false, .. }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;



    #[test]
    fn test_swarm_event_serialization() {
        let event = SwarmEvent {
            event: "swarm.issue.started".into(),
            timestamp: "2026-03-03T00:00:00Z".into(),
            issue_id: "test-123".into(),
            payload: SwarmEventPayload::IssueStarted {
                title: "Fix borrow checker error".into(),
                priority: Some(1),
                tier: "Worker".into(),
            },
        };
        let json = serde_json::to_string(&event).unwrap();
        assert!(json.contains("swarm.issue.started"));
        assert!(json.contains("issue_started"));
        assert!(json.contains("test-123"));

        // Round-trip
        let deserialized: SwarmEvent = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.event, "swarm.issue.started");
        assert_eq!(deserialized.issue_id, "test-123");
    }

    #[test]
    fn test_event_emitter_writes_jsonl() {
        let dir = tempfile::tempdir().unwrap();
        let emitter = SwarmEventEmitter::new(dir.path());

        emitter.issue_started("test-456", "Add feature X", Some(2), "Worker");
        emitter.iteration_completed("test-456", 1, "Worker", 3, false, 5000);
        emitter.issue_resolved("test-456", true, 2, 10000);

        let content = std::fs::read_to_string(dir.path().join(".swarm-events.jsonl")).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 3);

        // Each line is valid JSON
        for line in &lines {
            let _: SwarmEvent = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn test_is_critical() {
        let critical = SwarmEvent {
            event: "swarm.worker.failed".into(),
            timestamp: "2026-03-03T00:00:00Z".into(),
            issue_id: "test".into(),
            payload: SwarmEventPayload::WorkerFailed {
                model: "Qwen3.5".into(),
                error: "timeout".into(),
                iteration: 1,
            },
        };
        assert!(SwarmEventEmitter::is_critical(&critical));

        let non_critical = SwarmEvent {
            event: "swarm.iteration.completed".into(),
            timestamp: "2026-03-03T00:00:00Z".into(),
            issue_id: "test".into(),
            payload: SwarmEventPayload::IterationCompleted {
                iteration: 1,
                tier: "Worker".into(),
                error_count: 0,
                no_change: false,
                elapsed_ms: 5000,
            },
        };
        assert!(!SwarmEventEmitter::is_critical(&non_critical));

        let unhealthy = SwarmEvent {
            event: "swarm.health.check".into(),
            timestamp: "2026-03-03T00:00:00Z".into(),
            issue_id: "test".into(),
            payload: SwarmEventPayload::HealthCheck {
                endpoint: "vasp-03:8081".into(),
                healthy: false,
                latency_ms: None,
            },
        };
        assert!(SwarmEventEmitter::is_critical(&unhealthy));
    }
}
