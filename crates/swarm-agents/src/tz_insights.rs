//! TensorZero performance insights reader.
//!
//! Queries TZ's Postgres database directly to extract per-variant latency and
//! (once feedback is flowing) success-rate statistics. Generates deterministic
//! directives injected into worker prompts so the cloud manager can make
//! data-informed delegation decisions.
//!
//! Fail-safe: all errors are logged and produce an empty directive set.
//! The swarm runs normally when Postgres is unreachable.

use std::sync::Mutex;
use std::time::{Duration, Instant};
use tracing::{info, warn};

/// Errors that can occur when constructing or querying TZ insights.
#[derive(Debug, thiserror::Error)]
pub enum TzInsightsError {
    #[error("invalid configuration: {0}")]
    Config(String),
}

/// Cached performance insights from TensorZero Postgres.
pub struct TzInsights {
    pg_url: String,
    ttl: Duration,
    cache: Mutex<Option<CachedInsights>>,
    /// Optional repo_id filter. When set, insights only include data from
    /// this repository, preventing cross-project contamination.
    repo_id: Option<String>,
}

struct CachedInsights {
    generated_at: Instant,
    directives: Vec<String>,
}

/// Performance statistics for one function+variant combination.
#[derive(Debug, Clone)]
pub struct VariantStat {
    pub function_name: String,
    pub variant_name: String,
    pub call_count: u64,
    pub avg_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub min_latency_ms: f64,
    pub max_latency_ms: f64,
    /// Populated after Phase 1 episode fix enables feedback.
    pub success_rate: Option<f64>,
    pub avg_iterations: Option<f64>,
}

impl TzInsights {
    /// Create a new TzInsights reader.
    ///
    /// When `repo_id` is set, only insights from that repository are included.
    pub fn new(pg_url: &str, ttl_secs: u64) -> Result<Self, TzInsightsError> {
        if pg_url.is_empty() {
            return Err(TzInsightsError::Config("empty Postgres URL".into()));
        }
        Ok(Self {
            pg_url: pg_url.to_string(),
            ttl: Duration::from_secs(ttl_secs),
            cache: Mutex::new(None),
            repo_id: std::env::var("SWARM_REPO_ID")
                .ok()
                .filter(|s| !s.is_empty()),
        })
    }

    /// Get cached directives, refreshing from Postgres if stale.
    ///
    /// Never fails — returns empty Vec on any error (fail-safe pattern from
    /// `helpers.rs:query_kb_with_failsafe` and `tensorzero.rs:check_gateway`).
    pub async fn get_directives(&self) -> Vec<String> {
        // Check cache freshness
        {
            let guard = self.cache.lock().unwrap();
            if let Some(ref cached) = *guard {
                if cached.generated_at.elapsed() < self.ttl {
                    return cached.directives.clone();
                }
            }
        }

        // Cache miss or stale — refresh from Postgres
        match self.refresh().await {
            Ok(directives) => {
                let mut guard = self.cache.lock().unwrap();
                *guard = Some(CachedInsights {
                    generated_at: Instant::now(),
                    directives: directives.clone(),
                });
                directives
            }
            Err(e) => {
                warn!(error = %e, "TZ insights refresh failed — returning empty");
                Vec::new()
            }
        }
    }

    async fn refresh(&self) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let (client, connection) =
            tokio_postgres::connect(&self.pg_url, tokio_postgres::NoTls).await?;

        // Drive the connection in the background
        tokio::spawn(async move {
            if let Err(e) = connection.await {
                warn!(error = %e, "TZ Postgres connection error");
            }
        });

        let stats = query_variant_stats(&client, self.repo_id.as_deref()).await?;
        let global_model_stats = query_global_model_effectiveness(&client).await?;

        if stats.is_empty() && global_model_stats.is_empty() {
            info!("TZ insights: no variant stats found (need 3+ calls per variant)");
            return Ok(Vec::new());
        }

        let mut directives = generate_directives(&stats);
        directives.extend(generate_global_directives(&global_model_stats));
        directives.truncate(8); // cap total directives
        Ok(directives)
    }
}

/// Query per-variant latency statistics from the last 7 days.
///
/// Uses `::float8` casts so Postgres returns `double precision` directly,
/// avoiding the need for a `rust_decimal` dependency in tokio-postgres.
async fn query_variant_stats(
    client: &tokio_postgres::Client,
    repo_id: Option<&str>,
) -> Result<Vec<VariantStat>, Box<dyn std::error::Error + Send + Sync>> {
    let rows = if let Some(repo) = repo_id {
        client
            .query(
                r#"
SELECT
    ci.function_name,
    ci.variant_name,
    COUNT(*)::float8 AS call_count,
    AVG(ci.processing_time_ms)::float8 AS avg_latency_ms,
    (PERCENTILE_CONT(0.95) WITHIN GROUP
          (ORDER BY ci.processing_time_ms))::float8 AS p95_latency_ms,
    MIN(ci.processing_time_ms)::float8 AS min_latency_ms,
    MAX(ci.processing_time_ms)::float8 AS max_latency_ms
FROM tensorzero.chat_inferences ci
INNER JOIN (
    SELECT DISTINCT episode_id
    FROM tensorzero.boolean_metric_feedback
    WHERE tags->>'repo_id' = $1
) bmf ON ci.episode_id = bmf.episode_id
WHERE ci.created_at > NOW() - INTERVAL '7 days'
  AND ci.processing_time_ms IS NOT NULL
GROUP BY ci.function_name, ci.variant_name
HAVING COUNT(*) >= 3
ORDER BY ci.function_name, call_count DESC
"#,
                &[&repo],
            )
            .await?
    } else {
        client
            .query(
                r#"
SELECT
    function_name,
    variant_name,
    COUNT(*)::float8 AS call_count,
    AVG(processing_time_ms)::float8 AS avg_latency_ms,
    (PERCENTILE_CONT(0.95) WITHIN GROUP
          (ORDER BY processing_time_ms))::float8 AS p95_latency_ms,
    MIN(processing_time_ms)::float8 AS min_latency_ms,
    MAX(processing_time_ms)::float8 AS max_latency_ms
FROM tensorzero.chat_inferences
WHERE created_at > NOW() - INTERVAL '7 days'
  AND processing_time_ms IS NOT NULL
GROUP BY function_name, variant_name
HAVING COUNT(*) >= 3
ORDER BY function_name, call_count DESC
"#,
                &[],
            )
            .await?
    };

    let stats = rows
        .iter()
        .map(|row| {
            let call_count: f64 = row.get("call_count");
            let avg_latency: f64 = row.get("avg_latency_ms");
            let p95_latency: f64 = row.get("p95_latency_ms");
            let min_latency: f64 = row.get("min_latency_ms");
            let max_latency: f64 = row.get("max_latency_ms");

            VariantStat {
                function_name: row.get("function_name"),
                variant_name: row.get("variant_name"),
                call_count: call_count as u64,
                avg_latency_ms: avg_latency,
                p95_latency_ms: p95_latency,
                min_latency_ms: min_latency,
                max_latency_ms: max_latency,
                success_rate: None,
                avg_iterations: None,
            }
        })
        .collect();

    Ok(stats)
}

/// Global model effectiveness: success rate per model across ALL projects.
///
/// Queries the boolean_metric_feedback table (task_resolved metric) joined
/// with model_inferences to get per-model success rates. No repo_id filter —
/// this is intentionally cross-project to capture universal model strengths.
#[derive(Debug, Clone)]
struct GlobalModelStat {
    model_name: String,
    total_episodes: u64,
    success_rate: f64,
    avg_iterations: f64,
}

async fn query_global_model_effectiveness(
    client: &tokio_postgres::Client,
) -> Result<Vec<GlobalModelStat>, Box<dyn std::error::Error + Send + Sync>> {
    // This query joins feedback (task_resolved) with the model that handled
    // the episode. It aggregates across all repos for universal insights.
    let rows = client
        .query(
            r#"
SELECT
    mi.model_name,
    COUNT(DISTINCT bmf.episode_id)::float8 AS total_episodes,
    AVG(CASE WHEN bmf.value THEN 1.0 ELSE 0.0 END)::float8 AS success_rate,
    COALESCE(AVG(fmf.value), 0)::float8 AS avg_iterations
FROM tensorzero.boolean_metric_feedback bmf
JOIN tensorzero.chat_inferences ci ON bmf.episode_id = ci.episode_id
JOIN tensorzero.model_inferences mi ON ci.id = mi.inference_id
LEFT JOIN tensorzero.float_metric_feedback fmf
    ON bmf.episode_id = fmf.episode_id AND fmf.metric_name = 'iterations_used'
WHERE bmf.metric_name = 'task_resolved'
  AND bmf.created_at > NOW() - INTERVAL '14 days'
GROUP BY mi.model_name
HAVING COUNT(DISTINCT bmf.episode_id) >= 5
ORDER BY success_rate DESC
"#,
            &[],
        )
        .await;

    match rows {
        Ok(rows) => Ok(rows
            .iter()
            .map(|row| GlobalModelStat {
                model_name: row.get("model_name"),
                total_episodes: row.get::<_, f64>("total_episodes") as u64,
                success_rate: row.get("success_rate"),
                avg_iterations: row.get("avg_iterations"),
            })
            .collect()),
        Err(e) => {
            // Non-fatal: the feedback tables may not have data yet
            tracing::debug!(error = %e, "Global model effectiveness query failed (non-fatal)");
            Ok(Vec::new())
        }
    }
}

/// Generate directives from global cross-project model effectiveness data.
fn generate_global_directives(stats: &[GlobalModelStat]) -> Vec<String> {
    let mut directives = Vec::new();

    if stats.len() < 2 {
        return directives;
    }

    // Best model by success rate (with 10+ episodes for significance)
    let significant: Vec<&GlobalModelStat> =
        stats.iter().filter(|s| s.total_episodes >= 10).collect();
    if let Some(best) = significant.first() {
        if best.success_rate >= 0.5 {
            directives.push(format!(
                "[global] Best model across all projects: {} ({:.0}% success, {} episodes, {:.1} avg iterations)",
                best.model_name,
                best.success_rate * 100.0,
                best.total_episodes,
                best.avg_iterations,
            ));
        }
    }

    // Model with fewest iterations (efficiency champion)
    if let Some(efficient) = significant
        .iter()
        .filter(|s| s.avg_iterations > 0.0)
        .min_by(|a, b| {
            a.avg_iterations
                .partial_cmp(&b.avg_iterations)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        if efficient.avg_iterations < 3.0 {
            directives.push(format!(
                "[global] Most efficient model: {} ({:.1} avg iterations, {:.0}% success)",
                efficient.model_name,
                efficient.avg_iterations,
                efficient.success_rate * 100.0,
            ));
        }
    }

    directives
}

/// Generate human-readable directives from variant statistics.
///
/// Deterministic threshold-based rules (no LLM). Returns at most 5 directives.
pub fn generate_directives(stats: &[VariantStat]) -> Vec<String> {
    let mut directives = Vec::new();

    // Group stats by function
    let mut by_function: std::collections::HashMap<&str, Vec<&VariantStat>> =
        std::collections::HashMap::new();
    for s in stats {
        by_function.entry(&s.function_name).or_default().push(s);
    }

    for (func, variants) in &by_function {
        // Rule 1: Speed comparison — variant A >2x faster than B (5+ samples each)
        if variants.len() >= 2 {
            let qualified: Vec<&&VariantStat> =
                variants.iter().filter(|v| v.call_count >= 5).collect();
            if qualified.len() >= 2 {
                let fastest = qualified
                    .iter()
                    .min_by(|a, b| a.avg_latency_ms.partial_cmp(&b.avg_latency_ms).unwrap())
                    .unwrap();
                let slowest = qualified
                    .iter()
                    .max_by(|a, b| a.avg_latency_ms.partial_cmp(&b.avg_latency_ms).unwrap())
                    .unwrap();
                if slowest.avg_latency_ms > 0.0 {
                    let ratio = slowest.avg_latency_ms / fastest.avg_latency_ms;
                    if ratio >= 2.0 {
                        directives.push(format!(
                            "For {func}: {} ({:.0}ms avg) is {:.1}x faster than {} ({:.0}ms avg)",
                            fastest.variant_name,
                            fastest.avg_latency_ms,
                            ratio,
                            slowest.variant_name,
                            slowest.avg_latency_ms,
                        ));
                    }
                }
            }
        }

        // Rule 2: Dominant variant — one handles >70% of calls
        let total_calls: u64 = variants.iter().map(|v| v.call_count).sum();
        if total_calls > 0 {
            for v in variants {
                let pct = (v.call_count as f64 / total_calls as f64) * 100.0;
                if pct > 70.0 && variants.len() > 1 {
                    directives.push(format!(
                        "For {func}: {} handles {:.0}% of calls ({} total)",
                        v.variant_name, pct, total_calls,
                    ));
                }
            }
        }

        // Rule 3: Latency anomaly — P95 > 5x avg with 5+ samples
        for v in variants {
            if v.call_count >= 5
                && v.avg_latency_ms > 0.0
                && v.p95_latency_ms / v.avg_latency_ms > 5.0
            {
                directives.push(format!(
                    "WARNING: {func}/{} has P95 {:.0}ms vs avg {:.0}ms — possible timeout",
                    v.variant_name, v.p95_latency_ms, v.avg_latency_ms,
                ));
            }
        }

        // Rule 4: Success winner — variant A >20% higher success rate (10+ episodes each)
        if variants.len() >= 2 {
            let with_success: Vec<&&VariantStat> = variants
                .iter()
                .filter(|v| v.success_rate.is_some() && v.call_count >= 10)
                .collect();
            if with_success.len() >= 2 {
                let best = with_success
                    .iter()
                    .max_by(|a, b| {
                        a.success_rate
                            .partial_cmp(&b.success_rate)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                let worst = with_success
                    .iter()
                    .min_by(|a, b| {
                        a.success_rate
                            .partial_cmp(&b.success_rate)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                let best_rate = best.success_rate.unwrap_or(0.0);
                let worst_rate = worst.success_rate.unwrap_or(0.0);
                if best_rate - worst_rate > 20.0 {
                    directives.push(format!(
                        "For {func}: {} resolves {:.0}% vs {} {:.0}%",
                        best.variant_name, best_rate, worst.variant_name, worst_rate,
                    ));
                }
            }
        }

        // Rule 5: Zero success — function has 0% success across all variants
        let all_zero = variants
            .iter()
            .all(|v| v.success_rate.map(|r| r < 0.01).unwrap_or(false));
        let any_has_success = variants.iter().any(|v| v.success_rate.is_some());
        if all_zero && any_has_success && total_calls >= 10 {
            directives.push(format!(
                "WARNING: {func} has 0% resolution rate — tasks may be misrouted",
            ));
        }
    }

    // Sort for determinism, truncate to 5
    directives.sort();
    directives.truncate(5);
    directives
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stat(func: &str, variant: &str, count: u64, avg: f64, p95: f64) -> VariantStat {
        VariantStat {
            function_name: func.into(),
            variant_name: variant.into(),
            call_count: count,
            avg_latency_ms: avg,
            p95_latency_ms: p95,
            min_latency_ms: avg * 0.5,
            max_latency_ms: p95 * 1.2,
            success_rate: None,
            avg_iterations: None,
        }
    }

    #[test]
    fn test_generate_directives_empty() {
        assert!(generate_directives(&[]).is_empty());
    }

    #[test]
    fn test_speed_comparison_2x() {
        let stats = vec![
            make_stat("f", "fast", 10, 100.0, 200.0),
            make_stat("f", "slow", 10, 500.0, 1000.0),
        ];
        let d = generate_directives(&stats);
        assert!(d.iter().any(|s| s.contains("faster")), "directives: {d:?}");
    }

    #[test]
    fn test_speed_comparison_under_2x_no_directive() {
        let stats = vec![
            make_stat("f", "a", 10, 100.0, 200.0),
            make_stat("f", "b", 10, 150.0, 300.0),
        ];
        let d = generate_directives(&stats);
        assert!(
            !d.iter().any(|s| s.contains("faster")),
            "should not trigger for <2x ratio: {d:?}"
        );
    }

    #[test]
    fn test_latency_anomaly() {
        let stats = vec![make_stat("f", "v", 10, 100.0, 60000.0)];
        let d = generate_directives(&stats);
        assert!(d.iter().any(|s| s.contains("WARNING")), "directives: {d:?}");
    }

    #[test]
    fn test_latency_anomaly_under_5x_no_directive() {
        let stats = vec![make_stat("f", "v", 10, 100.0, 400.0)];
        let d = generate_directives(&stats);
        assert!(
            !d.iter().any(|s| s.contains("WARNING")),
            "should not trigger under 5x: {d:?}"
        );
    }

    #[test]
    fn test_max_five_directives() {
        let mut stats = Vec::new();
        // Create 10 functions, each with 2 variants where slow is 3x the fast
        for i in 0..10 {
            stats.push(make_stat(&format!("f{i}"), "fast", 10, 100.0, 200.0));
            stats.push(make_stat(&format!("f{i}"), "slow", 10, 500.0, 1000.0));
        }
        let d = generate_directives(&stats);
        assert!(d.len() <= 5, "got {} directives: {d:?}", d.len());
    }

    #[test]
    fn test_single_variant_no_comparison() {
        let stats = vec![make_stat("f", "only", 50, 100.0, 200.0)];
        let d = generate_directives(&stats);
        assert!(
            d.is_empty() || d.iter().all(|s| !s.contains("faster")),
            "single variant can't trigger speed comparison: {d:?}"
        );
    }

    #[test]
    fn test_dominant_variant() {
        let stats = vec![
            make_stat("f", "dominant", 80, 100.0, 200.0),
            make_stat("f", "minor", 10, 100.0, 200.0),
        ];
        let d = generate_directives(&stats);
        assert!(
            d.iter().any(|s| s.contains("handles") && s.contains("89%")),
            "should detect dominant variant: {d:?}"
        );
    }

    #[test]
    fn test_success_winner() {
        let stats = vec![
            VariantStat {
                success_rate: Some(65.0),
                ..make_stat("f", "good", 15, 100.0, 200.0)
            },
            VariantStat {
                success_rate: Some(30.0),
                ..make_stat("f", "bad", 15, 100.0, 200.0)
            },
        ];
        let d = generate_directives(&stats);
        assert!(
            d.iter().any(|s| s.contains("resolves")),
            "should detect success winner: {d:?}"
        );
    }

    #[test]
    fn test_zero_success() {
        let stats = vec![
            VariantStat {
                success_rate: Some(0.0),
                ..make_stat("f", "a", 10, 100.0, 200.0)
            },
            VariantStat {
                success_rate: Some(0.0),
                ..make_stat("f", "b", 10, 100.0, 200.0)
            },
        ];
        let d = generate_directives(&stats);
        assert!(
            d.iter()
                .any(|s| s.contains("WARNING") && s.contains("0% resolution")),
            "should detect zero success: {d:?}"
        );
    }

    #[test]
    fn test_cache_fresh_returns_cached() {
        let insights = TzInsights {
            pg_url: "postgres://invalid:5432/test".into(),
            ttl: Duration::from_secs(60),
            cache: Mutex::new(Some(CachedInsights {
                generated_at: Instant::now(),
                directives: vec!["cached directive".into()],
            })),
            repo_id: None,
        };
        // Cache is fresh — should not attempt Postgres connection
        let guard = insights.cache.lock().unwrap();
        let cached = guard.as_ref().unwrap();
        assert!(cached.generated_at.elapsed().as_secs() < 60);
        assert_eq!(cached.directives, vec!["cached directive"]);
    }

    #[test]
    fn test_new_rejects_empty_url() {
        assert!(TzInsights::new("", 1800).is_err());
    }

    #[test]
    fn test_new_accepts_valid_url() {
        assert!(TzInsights::new("postgres://localhost:5433/tensorzero", 1800).is_ok());
    }

    #[tokio::test]
    async fn test_get_directives_cache_hit() {
        let insights = TzInsights {
            pg_url: "postgres://invalid:5432/nonexistent".into(),
            ttl: Duration::from_secs(3600),
            cache: Mutex::new(Some(CachedInsights {
                generated_at: Instant::now(),
                directives: vec!["test insight".into()],
            })),
            repo_id: None,
        };
        // Should return cached value without hitting Postgres
        let d = insights.get_directives().await;
        assert_eq!(d, vec!["test insight"]);
    }

    #[tokio::test]
    async fn test_get_directives_pg_unreachable() {
        let insights = TzInsights::new("postgres://invalid:5432/nonexistent", 0).unwrap();
        // TTL=0 forces refresh, but Postgres is unreachable — should return empty
        let d = insights.get_directives().await;
        assert!(d.is_empty());
    }
}
