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

/// Generate a UUIDv7 episode ID for a given issue and session.
///
/// TensorZero requires episode IDs to be UUIDv7 (RFC 9562), which
/// encodes a Unix timestamp in the first 48 bits for time-ordered
/// storage. Random bits are derived from `blake3(issue_id:session_id)`
/// for reproducibility within the same millisecond.
pub fn generate_episode_id(issue_id: &str, session_id: &str) -> String {
    // blake3 hash for the random portion
    let mut hasher = blake3::Hasher::new();
    hasher.update(issue_id.as_bytes());
    hasher.update(b":");
    hasher.update(session_id.as_bytes());
    let hash = hasher.finalize();
    let rand_bytes = hash.as_bytes();

    // Unix timestamp in milliseconds (48 bits)
    let ts_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let mut uuid_bytes: [u8; 16] = [0; 16];

    // Bytes 0-5: 48-bit timestamp (big-endian)
    uuid_bytes[0] = ((ts_ms >> 40) & 0xff) as u8;
    uuid_bytes[1] = ((ts_ms >> 32) & 0xff) as u8;
    uuid_bytes[2] = ((ts_ms >> 24) & 0xff) as u8;
    uuid_bytes[3] = ((ts_ms >> 16) & 0xff) as u8;
    uuid_bytes[4] = ((ts_ms >> 8) & 0xff) as u8;
    uuid_bytes[5] = (ts_ms & 0xff) as u8;

    // Bytes 6-15: random from blake3 hash
    uuid_bytes[6..16].copy_from_slice(&rand_bytes[..10]);

    // Version 7: set bits 48-51 to 0111
    uuid_bytes[6] = (uuid_bytes[6] & 0x0f) | 0x70;
    // Variant 1 (RFC 4122/9562): set bits 64-65 to 10
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
    fn test_generate_episode_id_is_valid_uuidv7() {
        let ep = generate_episode_id("beads-abc1", "12345678-90ab-cdef");
        // Must be 36 chars: 8-4-4-4-12
        assert_eq!(ep.len(), 36);
        assert_eq!(&ep[8..9], "-");
        assert_eq!(&ep[13..14], "-");
        assert_eq!(&ep[18..19], "-");
        assert_eq!(&ep[23..24], "-");
        // Version nibble must be 7 (UUIDv7)
        assert_eq!(&ep[14..15], "7");
        // Variant nibble must be 8, 9, a, or b
        let variant = u8::from_str_radix(&ep[19..20], 16).unwrap();
        assert!((0x8..=0xb).contains(&variant), "variant={variant:x}");
    }

    #[test]
    fn test_generate_episode_id_unique_across_sessions() {
        let ep1 = generate_episode_id("issue-1", "session-a");
        let ep2 = generate_episode_id("issue-1", "session-b");
        // Different sessions produce different random bits (even if same ms)
        assert_ne!(ep1, ep2);
    }

    #[test]
    fn test_generate_episode_id_time_ordered() {
        let ep1 = generate_episode_id("issue-1", "session-a");
        std::thread::sleep(std::time::Duration::from_millis(2));
        let ep2 = generate_episode_id("issue-2", "session-b");
        // UUIDv7 sorts lexicographically by time — ep2 should be greater
        assert!(ep2 > ep1, "ep1={ep1} ep2={ep2}");
    }
}
