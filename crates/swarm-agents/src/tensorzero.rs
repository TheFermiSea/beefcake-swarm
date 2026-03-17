//! TensorZero feedback client for the swarm orchestrator.
//!
//! Posts episode-level feedback to TensorZero's `/feedback` endpoint after
//! each issue run completes. This links outcomes (resolved/failed, iterations,
//! wall time) back to the specific prompt variants and model configurations
//! that TensorZero selected during inference.
//!
//! TensorZero uses this feedback to optimize prompt selection via its GEPA
//! (Generative Engineering with Production Analytics) pipeline.

use serde::Serialize;
use tracing::{info, warn};

/// Feedback payload for TensorZero's `/feedback` endpoint.
#[derive(Debug, Serialize)]
struct FeedbackRequest {
    /// Metric name as defined in tensorzero.toml.
    metric_name: String,
    /// Episode ID linking this feedback to all inferences in the run.
    episode_id: String,
    /// The feedback value (type depends on the metric: boolean or float).
    value: serde_json::Value,
}

/// Post feedback to a TensorZero gateway instance.
///
/// Sends one feedback call per metric. Non-fatal: logs warnings on failure
/// but does not propagate errors (feedback is best-effort).
pub async fn post_episode_feedback(
    gateway_url: &str,
    episode_id: &str,
    success: bool,
    iterations: u32,
    wall_time_secs: f64,
) {
    let client = reqwest::Client::new();
    let feedback_url = format!("{gateway_url}/feedback");

    // task_resolved: boolean — did the issue get resolved?
    let feedbacks = vec![
        FeedbackRequest {
            metric_name: "task_resolved".to_string(),
            episode_id: episode_id.to_string(),
            value: serde_json::Value::Bool(success),
        },
        FeedbackRequest {
            metric_name: "iterations_used".to_string(),
            episode_id: episode_id.to_string(),
            value: serde_json::json!(iterations as f64),
        },
        FeedbackRequest {
            metric_name: "wall_time_seconds".to_string(),
            episode_id: episode_id.to_string(),
            value: serde_json::json!(wall_time_secs),
        },
    ];

    for fb in &feedbacks {
        match client
            .post(&feedback_url)
            .json(fb)
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await
        {
            Ok(resp) if resp.status().is_success() => {
                info!(
                    metric = %fb.metric_name,
                    episode_id = %fb.episode_id,
                    "Posted TensorZero feedback"
                );
            }
            Ok(resp) => {
                warn!(
                    metric = %fb.metric_name,
                    status = %resp.status(),
                    "TensorZero feedback rejected"
                );
            }
            Err(e) => {
                warn!(
                    metric = %fb.metric_name,
                    error = %e,
                    "Failed to post TensorZero feedback"
                );
            }
        }
    }
}

/// Generate an episode ID for a given issue and session.
///
/// Format: `{issue_id}_{session_short_id}` — unique per issue run,
/// stable within a session for grouping all inferences together.
pub fn generate_episode_id(issue_id: &str, session_id: &str) -> String {
    // Use first 8 chars of session ID for brevity
    let short_session = if session_id.len() > 8 {
        &session_id[..8]
    } else {
        session_id
    };
    format!("{issue_id}_{short_session}")
}

/// Check if a TensorZero gateway is reachable.
pub async fn check_gateway(gateway_url: &str) -> bool {
    let health_url = format!("{gateway_url}/health");
    match reqwest::Client::new()
        .get(&health_url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            info!(url = gateway_url, "TensorZero gateway is healthy");
            true
        }
        Ok(resp) => {
            warn!(
                url = gateway_url,
                status = %resp.status(),
                "TensorZero gateway returned non-success"
            );
            false
        }
        Err(e) => {
            warn!(
                url = gateway_url,
                error = %e,
                "TensorZero gateway unreachable — inference will bypass TensorZero"
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_episode_id() {
        let ep = generate_episode_id("beads-abc1", "12345678-90ab-cdef");
        assert_eq!(ep, "beads-abc1_12345678");
    }

    #[test]
    fn test_generate_episode_id_short_session() {
        let ep = generate_episode_id("beads-xyz9", "abc");
        assert_eq!(ep, "beads-xyz9_abc");
    }
}
