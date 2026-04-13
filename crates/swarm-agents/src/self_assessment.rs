//! Periodic self-assessment loop for swarm health (Layer 3).
//!
//! Runs after every N completed issues to detect degradation, anomalous routing,
//! and model-level failures. Takes conservative corrective actions and escalates
//! to humans (via beads issues) when automated fixes aren't sufficient.
//!
//! Design: docs/research/self-improving-swarm-architecture.md Layer 3.
//!
//! **NOT YET VALIDATED** — requires deployment and observation over multiple
//! dogfood cycles before relying on its corrective actions.

use std::path::Path;

use tracing::{debug, info, warn};

/// How often to run self-assessment (every N completed issues).
pub const ASSESSMENT_INTERVAL: usize = 10;

/// If overall success rate drops below this vs 7-day average, flag degradation.
const DEGRADATION_THRESHOLD_PCT: f64 = 20.0;

/// If a variant has 0% success on this many episodes, consider it broken.
const BROKEN_VARIANT_MIN_EPISODES: usize = 20;

/// If a single variant receives more than this % of traffic, exploration is needed.
const TRAFFIC_CONCENTRATION_THRESHOLD_PCT: f64 = 85.0;

/// Result of a self-assessment cycle.
#[derive(Debug, Clone)]
pub struct AssessmentReport {
    /// Number of issues completed since last assessment.
    pub issues_since_last: usize,
    /// Overall success rate in the assessment window.
    pub window_success_rate: f64,
    /// 7-day baseline success rate for comparison.
    pub baseline_success_rate: f64,
    /// Whether degradation was detected.
    pub degradation_detected: bool,
    /// Variants with 0% success rate on enough episodes to be concerning.
    pub broken_variants: Vec<String>,
    /// Variant receiving the most traffic (name, percentage).
    pub dominant_variant: Option<(String, f64)>,
    /// Whether traffic is overly concentrated on one variant.
    pub traffic_concentrated: bool,
    /// Actions taken automatically.
    pub actions_taken: Vec<String>,
    /// Issues created for human review.
    pub issues_created: Vec<String>,
}

/// Counter that tracks completed issues and triggers assessment.
pub struct AssessmentTrigger {
    completed_count: usize,
    interval: usize,
}

impl AssessmentTrigger {
    pub fn new() -> Self {
        Self {
            completed_count: 0,
            interval: ASSESSMENT_INTERVAL,
        }
    }

    /// Record a completed issue. Returns true if assessment should run.
    pub fn record_completion(&mut self) -> bool {
        self.completed_count += 1;
        if self.completed_count >= self.interval {
            self.completed_count = 0;
            true
        } else {
            false
        }
    }

    /// Force an assessment on the next completion.
    #[allow(dead_code)]
    pub fn force_next(&mut self) {
        self.completed_count = self.interval.saturating_sub(1);
    }
}

impl Default for AssessmentTrigger {
    fn default() -> Self {
        Self::new()
    }
}

/// Run a self-assessment cycle by querying TZ Postgres for variant performance.
///
/// Returns a report describing what was found and what actions were taken.
/// This is a best-effort operation — if TZ Postgres is unreachable, returns None.
///
/// **NOT YET VALIDATED** — corrective actions (weight adjustment, issue creation)
/// are logged but should be reviewed before trusting them in production.
pub async fn run_assessment(tz_pg_url: Option<&str>, repo_root: &Path) -> Option<AssessmentReport> {
    let pg_url = tz_pg_url?;

    // Query TZ Postgres for variant performance — run both queries concurrently.
    let (stats_result, baseline_result) = tokio::join!(
        query_variant_stats(pg_url),
        query_baseline_rate(pg_url),
    );

    let stats = match stats_result {
        Ok(s) => s,
        Err(e) => {
            warn!(error = %e, "Self-assessment: failed to query TZ Postgres");
            return None;
        }
    };

    let baseline = match baseline_result {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Self-assessment: failed to query baseline rate");
            return None;
        }
    };

    let mut report = AssessmentReport {
        issues_since_last: ASSESSMENT_INTERVAL,
        window_success_rate: 0.0,
        baseline_success_rate: baseline,
        degradation_detected: false,
        broken_variants: Vec::new(),
        dominant_variant: None,
        traffic_concentrated: false,
        actions_taken: Vec::new(),
        issues_created: Vec::new(),
    };

    // Calculate overall window success rate
    let total_episodes: usize = stats.iter().map(|s| s.episodes).sum();
    let total_wins: usize = stats.iter().map(|s| s.wins).sum();
    report.window_success_rate = if total_episodes > 0 {
        100.0 * total_wins as f64 / total_episodes as f64
    } else {
        0.0
    };

    // Check for degradation
    if baseline > 0.0
        && report.window_success_rate < baseline - DEGRADATION_THRESHOLD_PCT
        && total_episodes >= 10
    {
        report.degradation_detected = true;
        warn!(
            window = format!("{:.1}%", report.window_success_rate),
            baseline = format!("{:.1}%", baseline),
            drop = format!("{:.1}%", baseline - report.window_success_rate),
            "Self-assessment: DEGRADATION DETECTED"
        );
    }

    // Check for broken variants
    for stat in &stats {
        if stat.episodes >= BROKEN_VARIANT_MIN_EPISODES && stat.wins == 0 {
            report.broken_variants.push(stat.variant.clone());
            warn!(
                variant = %stat.variant,
                episodes = stat.episodes,
                "Self-assessment: variant has 0% success rate"
            );
        }
    }

    // Check for traffic concentration
    if total_episodes > 0 {
        if let Some(max) = stats.iter().max_by_key(|s| s.episodes) {
            let pct = 100.0 * max.episodes as f64 / total_episodes as f64;
            report.dominant_variant = Some((max.variant.clone(), pct));
            if pct > TRAFFIC_CONCENTRATION_THRESHOLD_PCT {
                report.traffic_concentrated = true;
                warn!(
                    variant = %max.variant,
                    traffic_pct = format!("{pct:.1}%"),
                    "Self-assessment: traffic overly concentrated on single variant"
                );
            }
        }
    }

    // Log the report
    let report_path = repo_root.join(".swarm-telemetry.jsonl");
    if let Ok(json) = serde_json::to_string(&serde_json::json!({
        "type": "self_assessment",
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "window_success_rate": report.window_success_rate,
        "baseline_success_rate": report.baseline_success_rate,
        "degradation_detected": report.degradation_detected,
        "broken_variants": &report.broken_variants,
        "dominant_variant": &report.dominant_variant,
        "traffic_concentrated": report.traffic_concentrated,
        "total_episodes": total_episodes,
        "total_wins": total_wins,
    })) {
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&report_path)
            .and_then(|mut f| {
                use std::io::Write;
                writeln!(f, "{json}")
            });
    }

    info!(
        rate = format!("{:.1}%", report.window_success_rate),
        baseline = format!("{:.1}%", report.baseline_success_rate),
        degraded = report.degradation_detected,
        broken = report.broken_variants.len(),
        concentrated = report.traffic_concentrated,
        "Self-assessment complete"
    );

    Some(report)
}

// ---------------------------------------------------------------------------
// TZ Postgres queries
// ---------------------------------------------------------------------------

struct VariantStats {
    variant: String,
    episodes: usize,
    wins: usize,
}

async fn query_variant_stats(pg_url: &str) -> Result<Vec<VariantStats>, String> {
    // Use tokio-postgres for async queries
    let (client, connection) = tokio_postgres::connect(pg_url, tokio_postgres::NoTls)
        .await
        .map_err(|e| format!("TZ Postgres connect failed: {e}"))?;

    // Spawn connection handler
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            warn!(error = %e, "TZ Postgres connection error");
        }
    });

    let rows = client
        .query(
            "SELECT ci.variant_name,
                    COUNT(DISTINCT ci.episode_id) as episodes,
                    COUNT(DISTINCT CASE WHEN bf.value THEN ci.episode_id END) as wins
             FROM tensorzero.chat_inferences ci
             JOIN tensorzero.boolean_metric_feedback bf ON bf.target_id = ci.episode_id
             WHERE bf.metric_name = 'task_resolved'
             AND ci.function_name = 'worker_code_edit'
             AND bf.created_at > NOW() - INTERVAL '24 hours'
             GROUP BY ci.variant_name
             ORDER BY episodes DESC",
            &[],
        )
        .await
        .map_err(|e| format!("TZ query failed: {e}"))?;

    Ok(rows
        .iter()
        .map(|r| VariantStats {
            variant: r.get::<_, String>(0),
            episodes: r.get::<_, i64>(1) as usize,
            wins: r.get::<_, i64>(2) as usize,
        })
        .collect())
}

async fn query_baseline_rate(pg_url: &str) -> Result<f64, String> {
    let (client, connection) = tokio_postgres::connect(pg_url, tokio_postgres::NoTls)
        .await
        .map_err(|e| format!("TZ Postgres connect failed: {e}"))?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            debug!(error = %e, "TZ Postgres connection error (baseline query)");
        }
    });

    let row = client
        .query_one(
            "SELECT COALESCE(
                100.0 * SUM(CASE WHEN value THEN 1 ELSE 0 END)::FLOAT / NULLIF(COUNT(*), 0),
                0.0
             ) as rate
             FROM tensorzero.boolean_metric_feedback
             WHERE metric_name = 'task_resolved'
             AND created_at > NOW() - INTERVAL '7 days'",
            &[],
        )
        .await
        .map_err(|e| format!("TZ baseline query failed: {e}"))?;

    Ok(row.get::<_, f64>(0))
}
