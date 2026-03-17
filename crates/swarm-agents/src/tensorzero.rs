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

/// Generate a deterministic UUID episode ID for a given issue and session.
///
/// TensorZero requires episode IDs to be valid UUIDs. We derive one
/// deterministically from `blake3(issue_id + session_id)` so the same
/// run always produces the same UUID. The version nibble is set to 4
/// and the variant bits to `10xx` for RFC 4122 compliance.
pub fn generate_episode_id(issue_id: &str, session_id: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(issue_id.as_bytes());
    hasher.update(b":");
    hasher.update(session_id.as_bytes());
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();

    // Format first 16 bytes as UUID with proper version/variant bits
    let mut uuid_bytes: [u8; 16] = [0; 16];
    uuid_bytes.copy_from_slice(&bytes[..16]);
    // Version 4 (random): set bits 48-51 to 0100
    uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x40;
    // Variant 1 (RFC 4122): set bits 64-65 to 10
    uuid_bytes[8] = (uuid_bytes[8] & 0x3f) | 0x80;

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        uuid_bytes[0], uuid_bytes[1], uuid_bytes[2], uuid_bytes[3],
        uuid_bytes[4], uuid_bytes[5],
        uuid_bytes[6], uuid_bytes[7],
        uuid_bytes[8], uuid_bytes[9],
        uuid_bytes[10], uuid_bytes[11], uuid_bytes[12], uuid_bytes[13], uuid_bytes[14], uuid_bytes[15],
    )
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
    fn test_generate_episode_id_is_valid_uuid() {
        let ep = generate_episode_id("beads-abc1", "12345678-90ab-cdef");
        // Must be 36 chars: 8-4-4-4-12
        assert_eq!(ep.len(), 36);
        assert_eq!(&ep[8..9], "-");
        assert_eq!(&ep[13..14], "-");
        assert_eq!(&ep[18..19], "-");
        assert_eq!(&ep[23..24], "-");
        // Version nibble must be 4
        assert_eq!(&ep[14..15], "4");
        // Variant nibble must be 8, 9, a, or b
        let variant = u8::from_str_radix(&ep[19..20], 16).unwrap();
        assert!((0x8..=0xb).contains(&variant), "variant={variant:x}");
    }

    #[test]
    fn test_generate_episode_id_deterministic() {
        let ep1 = generate_episode_id("beads-xyz9", "session-abc");
        let ep2 = generate_episode_id("beads-xyz9", "session-abc");
        assert_eq!(ep1, ep2);
    }

    #[test]
    fn test_generate_episode_id_unique() {
        let ep1 = generate_episode_id("issue-1", "session-a");
        let ep2 = generate_episode_id("issue-1", "session-b");
        assert_ne!(ep1, ep2);
    }
}
