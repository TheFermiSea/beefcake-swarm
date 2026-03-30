//! TensorZero feedback client for the swarm orchestrator.
//!
//! Posts episode-level feedback to TensorZero's `/feedback` endpoint after
//! each issue run completes. This links outcomes (resolved/failed, iterations,
//! wall time) back to the specific prompt variants and model configurations
//! that TensorZero selected during inference.
//!
//! TensorZero uses this feedback to optimize prompt selection via its GEPA
//! (Generative Engineering with Production Analytics) pipeline.

use std::collections::HashMap;

use serde::Serialize;
use tracing::{info, warn};

/// Feedback payload for TensorZero's `/feedback` endpoint.
///
/// TZ distinguishes episode-level feedback (keyed by `episode_id`) from
/// inference-level feedback (keyed by `inference_id`). Exactly one of
/// the two ID fields should be `Some`.
#[derive(Debug, Serialize)]
struct FeedbackRequest {
    /// Metric name as defined in tensorzero.toml (or `"demonstration"` / `"comment"`).
    metric_name: String,
    /// Episode ID — for episode-level metrics like `task_resolved`.
    #[serde(skip_serializing_if = "Option::is_none")]
    episode_id: Option<String>,
    /// Inference ID — for inference-level metrics like `verifier_pass` and demonstrations.
    #[serde(skip_serializing_if = "Option::is_none")]
    inference_id: Option<String>,
    /// The feedback value (type depends on the metric: boolean, float, or string).
    value: serde_json::Value,
    /// Optional segmentation tags stored in the `tags jsonb` column.
    #[serde(skip_serializing_if = "Option::is_none")]
    tags: Option<HashMap<String, String>>,
}

/// Segmentation tags attached to TensorZero feedback for slice-based analysis.
///
/// All fields are optional so callers can fill in what they know. `None`
/// values are omitted from the serialised `tags` map.
#[derive(Debug, Default, Clone)]
pub struct FeedbackTags {
    /// Beads issue ID (e.g. `beefcake-grg4`).
    pub issue_id: Option<String>,
    /// Primary language of the target repo (e.g. `rust`, `python`).
    pub language: Option<String>,
    /// Triage complexity bucket: `simple`, `medium`, `complex`, or `critical`.
    pub triage_complexity: Option<String>,
    /// Primary model used for this run (e.g. `claude-opus-4-6`).
    pub model: Option<String>,
    /// Repository identifier for project isolation (e.g. `beefcake-swarm`, `rust-daq`).
    /// Prevents TZ feedback from one project contaminating routing for another.
    /// Set from `SWARM_REPO_ID` env var (auto-populated from `--repo-root` basename).
    pub repo_id: Option<String>,
}

impl FeedbackTags {
    /// Convert to a `HashMap` suitable for JSON serialisation.
    /// Only entries that are `Some` are included.
    pub fn into_map(self) -> HashMap<String, String> {
        let mut map = HashMap::new();
        if let Some(v) = self.issue_id {
            map.insert("issue_id".to_string(), v);
        }
        if let Some(v) = self.language {
            map.insert("language".to_string(), v);
        }
        if let Some(v) = self.triage_complexity {
            map.insert("triage_complexity".to_string(), v);
        }
        if let Some(v) = self.model {
            map.insert("model".to_string(), v);
        }
        if let Some(v) = self.repo_id {
            map.insert("repo_id".to_string(), v);
        }
        map
    }

    /// Returns `None` when all fields are empty (so the tags key is omitted).
    pub fn into_option_map(self) -> Option<HashMap<String, String>> {
        let map = self.into_map();
        if map.is_empty() {
            None
        } else {
            Some(map)
        }
    }
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
    tags: Option<FeedbackTags>,
) {
    let client = reqwest::Client::new();
    let feedback_url = format!("{gateway_url}/feedback");

    // Convert tags once; clone the resulting map for each metric.
    let tags_map: Option<HashMap<String, String>> = tags.and_then(|t| t.into_option_map());

    // Episode-level metrics — all keyed by episode_id.
    let feedbacks = vec![
        FeedbackRequest {
            metric_name: "task_resolved".to_string(),
            episode_id: Some(episode_id.to_string()),
            inference_id: None,
            value: serde_json::Value::Bool(success),
            tags: tags_map.clone(),
        },
        FeedbackRequest {
            metric_name: "iterations_used".to_string(),
            episode_id: Some(episode_id.to_string()),
            inference_id: None,
            value: serde_json::json!(iterations as f64),
            tags: tags_map.clone(),
        },
        FeedbackRequest {
            metric_name: "wall_time_seconds".to_string(),
            episode_id: Some(episode_id.to_string()),
            inference_id: None,
            value: serde_json::json!(wall_time_secs),
            tags: tags_map.clone(),
        },
        FeedbackRequest {
            metric_name: "iteration_efficiency".to_string(),
            episode_id: Some(episode_id.to_string()),
            inference_id: None,
            value: serde_json::json!(iterations as f64),
            tags: tags_map,
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
                    episode_id = episode_id,
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

/// Resolve actual episode IDs from TZ Postgres for feedback posting.
///
/// The swarm generates its own episode_id via `generate_episode_id()`, but TZ's
/// OpenAI-compat endpoint auto-generates different episode_ids per inference
/// (since Rig can't inject the `tensorzero::episode_id` extra body param).
///
/// This function queries TZ Postgres for distinct episode_ids from recent
/// inferences and returns them so feedback can be posted to the correct targets.
///
/// Returns empty Vec on any error (fail-safe).
pub async fn resolve_episode_ids(pg_url: &str, session_start_secs: f64) -> Vec<String> {
    let Ok((client, connection)) = tokio_postgres::connect(pg_url, tokio_postgres::NoTls).await
    else {
        warn!("Failed to connect to TZ Postgres for episode ID resolution");
        return Vec::new();
    };

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            warn!(error = %e, "TZ Postgres connection error during episode resolution");
        }
    });

    // Query distinct episode_ids from inferences created after session start.
    // We use a generous 60s buffer before session start to catch early inferences.
    let cutoff_secs = (session_start_secs - 60.0).max(0.0);
    let rows = match client
        .query(
            r#"
SELECT DISTINCT episode_id::text
FROM tensorzero.chat_inferences
WHERE created_at > to_timestamp($1)
ORDER BY episode_id::text
"#,
            &[&cutoff_secs],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Failed to query TZ episode IDs");
            return Vec::new();
        }
    };

    let ids: Vec<String> = rows.iter().filter_map(|row| row.get(0)).collect();
    if ids.is_empty() {
        warn!("No TZ episode IDs found for this session — feedback will be skipped");
    } else {
        info!(count = ids.len(), "Resolved TZ episode IDs for feedback");
    }
    ids
}

/// Post feedback to all resolved TZ episode IDs.
///
/// Wraps `post_episode_feedback` to handle the many-episode case where each
/// TZ inference gets its own episode_id. Posts the same outcome metrics to
/// every episode in the session.
pub async fn post_resolved_feedback(
    gateway_url: &str,
    pg_url: &str,
    session_start_secs: f64,
    success: bool,
    iterations: u32,
    wall_time_secs: f64,
    tags: Option<FeedbackTags>,
) {
    let episode_ids = resolve_episode_ids(pg_url, session_start_secs).await;
    if episode_ids.is_empty() {
        return;
    }

    // Pre-materialise the tags map so we can clone it cheaply per episode.
    let tags_map: Option<HashMap<String, String>> = tags.and_then(|t| t.into_option_map());

    for ep_id in &episode_ids {
        // Reconstruct a FeedbackTags-compatible value from the pre-built map.
        // We pass None here and supply the map directly via a one-off helper
        // to avoid re-serialising every iteration.
        let per_ep_tags = tags_map.as_ref().map(|m| FeedbackTags {
            issue_id: m.get("issue_id").cloned(),
            language: m.get("language").cloned(),
            triage_complexity: m.get("triage_complexity").cloned(),
            model: m.get("model").cloned(),
            repo_id: m.get("repo_id").cloned(),
        });
        post_episode_feedback(
            gateway_url,
            ep_id,
            success,
            iterations,
            wall_time_secs,
            per_ep_tags,
        )
        .await;
    }
    info!(
        episodes = episode_ids.len(),
        success, "Posted TZ feedback to all resolved episodes"
    );
}

/// Post inference-level feedback (e.g. `verifier_pass` after each iteration).
///
/// Unlike episode-level feedback, this targets a specific inference via its
/// `inference_id`. The inference_id comes from TZ's response (the `id` field
/// in the ChatCompletion) or from Postgres resolution.
pub async fn post_inference_feedback(
    gateway_url: &str,
    inference_id: &str,
    metric_name: &str,
    value: serde_json::Value,
    tags: Option<FeedbackTags>,
) {
    let client = reqwest::Client::new();
    let feedback_url = format!("{gateway_url}/feedback");
    let tags_map = tags.and_then(|t| t.into_option_map());

    let fb = FeedbackRequest {
        metric_name: metric_name.to_string(),
        episode_id: None,
        inference_id: Some(inference_id.to_string()),
        value,
        tags: tags_map,
    };

    match client
        .post(&feedback_url)
        .json(&fb)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            info!(
                metric = metric_name,
                inference_id, "Posted TZ inference-level feedback"
            );
        }
        Ok(resp) => {
            warn!(
                metric = metric_name,
                status = %resp.status(),
                "TZ inference feedback rejected"
            );
        }
        Err(e) => {
            warn!(
                metric = metric_name,
                error = %e,
                "Failed to post TZ inference feedback"
            );
        }
    }
}

/// Post a demonstration (ideal output) for a specific inference.
///
/// Demonstrations are the foundation for DICL (Dynamic In-Context Learning)
/// and improve DPO preference pair quality. TZ stores them and retrieves
/// similar examples at inference time when a DICL variant is configured.
///
/// `value` should be the successful worker output — typically the git diff
/// of the changes that passed verification.
pub async fn post_demonstration(gateway_url: &str, inference_id: &str, value: &str) {
    let client = reqwest::Client::new();
    let feedback_url = format!("{gateway_url}/feedback");

    let fb = FeedbackRequest {
        metric_name: "demonstration".to_string(),
        episode_id: None,
        inference_id: Some(inference_id.to_string()),
        value: serde_json::Value::String(value.to_string()),
        tags: None,
    };

    match client
        .post(&feedback_url)
        .json(&fb)
        .timeout(std::time::Duration::from_secs(15))
        .send()
        .await
    {
        Ok(resp) if resp.status().is_success() => {
            info!(
                inference_id,
                demo_len = value.len(),
                "Posted TZ demonstration feedback"
            );
        }
        Ok(resp) => {
            warn!(
                status = %resp.status(),
                "TZ demonstration feedback rejected"
            );
        }
        Err(e) => {
            warn!(error = %e, "Failed to post TZ demonstration");
        }
    }
}

/// Resolve the most recent inference IDs from TZ Postgres.
///
/// Returns inference IDs ordered newest-first. Used to target inference-level
/// feedback (verifier_pass, demonstrations) at the correct TZ inference record.
pub async fn resolve_recent_inference_ids(
    pg_url: &str,
    since_secs: f64,
    limit: i64,
) -> Vec<String> {
    let Ok((client, connection)) = tokio_postgres::connect(pg_url, tokio_postgres::NoTls).await
    else {
        warn!("Failed to connect to TZ Postgres for inference ID resolution");
        return Vec::new();
    };

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            warn!(error = %e, "TZ Postgres connection error during inference resolution");
        }
    });

    let cutoff_secs = (since_secs - 10.0).max(0.0);
    let rows = match client
        .query(
            r#"
SELECT id::text
FROM tensorzero.chat_inferences
WHERE created_at > to_timestamp($1)
ORDER BY created_at DESC
LIMIT $2
"#,
            &[&cutoff_secs, &limit],
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Failed to query TZ inference IDs");
            return Vec::new();
        }
    };

    rows.iter().filter_map(|row| row.get(0)).collect()
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
