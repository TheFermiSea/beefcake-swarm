//! Autopilot runner: end-to-end analysis → recommendations → artifacts.
//!
//! Stitches together the analysis stages in dependency order:
//!
//! 1. **MutationArchive** — load historical outcomes
//! 2. **MetaReflector** — generate insights from outcomes
//! 3. **Recommendations** — turn insights into experiment candidates
//! 4. **TzInsights** — query TensorZero for variant performance data
//! 5. **Artifact output** — write JSON artifacts to `.swarm/autopilot/`
//! 6. **Operator summary** — concise report of what changed and what to try next
//!
//! # Usage
//!
//! ```ignore
//! let report = AutopilotRunner::new(repo_root)
//!     .with_tz_postgres("postgresql://...")
//!     .run()
//!     .await;
//! println!("{}", report.operator_summary());
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use tracing::{info, warn};

use crate::meta_reflection::{MetaInsight, MetaReflector};
use crate::mutation_archive::{ArchiveSummary, MutationArchive};
use crate::recommendations::{
    generate_recommendations, to_runner_inputs, ExperimentRecommendation,
};

/// Configuration for the autopilot runner.
pub struct AutopilotRunner {
    repo_root: PathBuf,
    /// Number of recent mutation records to analyze.
    window_size: usize,
    /// TZ Postgres URL for performance insights (optional).
    pg_url: Option<String>,
    /// TZ insights cache TTL in seconds.
    tz_ttl_secs: u64,
}

/// The complete output of an autopilot run.
#[derive(Debug, Serialize, Deserialize)]
pub struct AutopilotReport {
    pub timestamp: String,
    pub archive_summary: ArchiveSummarySnapshot,
    pub insights: Vec<MetaInsight>,
    pub recommendations: Vec<ExperimentRecommendation>,
    pub tz_directives: Vec<String>,
    pub artifacts_dir: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trends: Option<TrendData>,
}

/// Week-over-week trend deltas computed from previous autopilot reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendData {
    /// Change in resolution rate (current - previous), e.g. +0.05 means +5%.
    pub resolution_rate_delta: f64,
    /// Change in average iterations to resolve.
    pub avg_iterations_delta: f64,
    /// Change in insight count (current - previous).
    pub insights_count_delta: i32,
    /// How many previous reports were available for comparison.
    pub reports_compared: usize,
}

/// Serializable snapshot of archive stats (ArchiveSummary has HashMap which
/// serializes fine, but we want a clean top-level view).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveSummarySnapshot {
    pub total_attempts: usize,
    pub resolved: usize,
    pub failed: usize,
    pub resolution_rate: f64,
    pub avg_iterations: f64,
}

impl From<ArchiveSummary> for ArchiveSummarySnapshot {
    fn from(s: ArchiveSummary) -> Self {
        let rate = if s.total_attempts > 0 {
            s.resolved as f64 / s.total_attempts as f64
        } else {
            0.0
        };
        Self {
            total_attempts: s.total_attempts,
            resolved: s.resolved,
            failed: s.failed,
            resolution_rate: rate,
            avg_iterations: s.avg_iterations_to_resolve,
        }
    }
}

/// Load previous autopilot report JSON files from the artifacts directory.
///
/// Reads `report-*.json` files, parses each into an [`AutopilotReport`],
/// sorts by timestamp descending, and returns at most `limit` entries.
fn load_previous_reports(artifacts_dir: &Path, limit: usize) -> Vec<AutopilotReport> {
    let entries = match std::fs::read_dir(artifacts_dir) {
        Ok(rd) => rd,
        Err(e) => {
            warn!(error = %e, "Failed to read autopilot artifacts directory");
            return Vec::new();
        }
    };

    let mut reports: Vec<AutopilotReport> = entries
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            name.starts_with("report-") && name.ends_with(".json")
        })
        .filter_map(|entry| {
            let path = entry.path();
            let contents = std::fs::read_to_string(&path).ok()?;
            serde_json::from_str::<AutopilotReport>(&contents)
                .map_err(|e| {
                    warn!(path = %path.display(), error = %e, "Failed to parse autopilot report");
                    e
                })
                .ok()
        })
        .collect();

    // Sort by timestamp descending (ISO 8601 / RFC 3339 sorts lexicographically)
    reports.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
    reports.truncate(limit);
    reports
}

/// Compute trend deltas between the current report and the most recent previous one.
fn compute_trends(current: &AutopilotReport, previous: &[AutopilotReport]) -> Option<TrendData> {
    let prev = previous.first()?;

    Some(TrendData {
        resolution_rate_delta: current.archive_summary.resolution_rate
            - prev.archive_summary.resolution_rate,
        avg_iterations_delta: current.archive_summary.avg_iterations
            - prev.archive_summary.avg_iterations,
        insights_count_delta: current.insights.len() as i32 - prev.insights.len() as i32,
        reports_compared: previous.len(),
    })
}

impl AutopilotRunner {
    pub fn new(repo_root: &Path) -> Self {
        Self {
            repo_root: repo_root.to_path_buf(),
            window_size: 50,
            pg_url: std::env::var("SWARM_TENSORZERO_PG_URL").ok(),
            tz_ttl_secs: 1800,
        }
    }

    pub fn with_window_size(mut self, n: usize) -> Self {
        self.window_size = n;
        self
    }

    pub fn with_tz_postgres(mut self, url: &str) -> Self {
        self.pg_url = Some(url.to_string());
        self
    }

    /// Run the full autopilot pipeline.
    ///
    /// # Errors
    /// Returns an error if the artifacts directory cannot be created or if
    /// any artifact file cannot be serialized or written to disk.
    pub async fn run(&self) -> Result<AutopilotReport> {
        let now = Utc::now();
        info!("Autopilot: starting analysis pipeline");

        // --- Stage 1: Load mutation archive ---
        let archive = MutationArchive::new(&self.repo_root);
        let summary: ArchiveSummarySnapshot = archive.summary().into();
        info!(
            total = summary.total_attempts,
            resolved = summary.resolved,
            rate = format!("{:.0}%", summary.resolution_rate * 100.0),
            "Autopilot: mutation archive loaded"
        );

        // --- Stage 2: Generate meta-insights ---
        let reflector = MetaReflector::new(&self.repo_root);
        let insights = reflector.reflect(self.window_size);
        info!(
            count = insights.len(),
            "Autopilot: meta-reflection complete"
        );

        // Save insights to the JSONL file for future sessions
        if !insights.is_empty() {
            reflector.save_insights(&insights);
        }

        // --- Stage 3: Generate experiment recommendations ---
        let recommendations = generate_recommendations(&insights);
        info!(
            count = recommendations.len(),
            "Autopilot: experiment recommendations generated"
        );

        // --- Stage 4: Query TZ performance directives ---
        let tz_directives = if let Some(ref pg_url) = self.pg_url {
            match crate::tz_insights::TzInsights::new(pg_url, self.tz_ttl_secs) {
                Ok(tz) => tz.get_directives().await,
                Err(e) => {
                    warn!(error = %e, "Autopilot: TZ insights unavailable");
                    Vec::new()
                }
            }
        } else {
            info!("Autopilot: no TZ Postgres URL — skipping performance directives");
            Vec::new()
        };
        info!(
            count = tz_directives.len(),
            "Autopilot: TZ performance directives collected"
        );

        // --- Stage 5: Write artifacts ---
        let artifacts_dir = self.repo_root.join(".swarm").join("autopilot");
        std::fs::create_dir_all(&artifacts_dir).with_context(|| {
            format!(
                "Failed to create autopilot artifacts directory: {}",
                artifacts_dir.display()
            )
        })?;

        let run_id = now.format("%Y%m%d-%H%M%S").to_string();

        // Write insights
        let insights_path = artifacts_dir.join(format!("insights-{run_id}.json"));
        write_json(&insights_path, &insights)?;

        // Write recommendations
        let recs_path = artifacts_dir.join(format!("recommendations-{run_id}.json"));
        write_json(&recs_path, &recommendations)?;

        // Write flat runner-input records (one row per variant, ready for adaptive runner)
        let runner_inputs = to_runner_inputs(&recommendations);
        let runner_path = artifacts_dir.join(format!("runner-inputs-{run_id}.json"));
        write_json(&runner_path, &runner_inputs)?;

        // Write full report (trends computed from previous reports)
        let mut report = AutopilotReport {
            timestamp: now.to_rfc3339(),
            archive_summary: summary,
            insights,
            recommendations,
            tz_directives,
            artifacts_dir: artifacts_dir.display().to_string(),
            trends: None,
        };

        // --- Stage 6: Compute weekly trends ---
        let previous_reports = load_previous_reports(&artifacts_dir, 7);
        if let Some(trends) = compute_trends(&report, &previous_reports) {
            info!(
                resolution_rate_delta = format!("{:+.1}%", trends.resolution_rate_delta * 100.0),
                avg_iterations_delta = format!("{:+.1}", trends.avg_iterations_delta),
                insights_delta = trends.insights_count_delta,
                compared = trends.reports_compared,
                "Autopilot: trend data computed"
            );
            report.trends = Some(trends);
        }

        let report_path = artifacts_dir.join(format!("report-{run_id}.json"));
        write_json(&report_path, &report)?;

        // Write a symlink-like "latest" pointer
        let latest_path = artifacts_dir.join("latest-report.json");
        write_json(&latest_path, &report)?;

        info!(
            dir = %artifacts_dir.display(),
            run_id,
            "Autopilot: artifacts written"
        );

        Ok(report)
    }
}

impl AutopilotReport {
    /// Generate a concise operator-facing summary.
    pub fn operator_summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push("═══ Autopilot Report ═══".to_string());
        lines.push(format!("  Timestamp: {}", self.timestamp));
        lines.push(String::new());

        // Archive health
        let s = &self.archive_summary;
        lines.push(format!(
            "  Archive: {}/{} resolved ({:.0}%), avg {:.1} iterations",
            s.resolved,
            s.total_attempts,
            s.resolution_rate * 100.0,
            s.avg_iterations,
        ));

        // Insights
        if self.insights.is_empty() {
            lines.push("  Insights: none (need more data)".to_string());
        } else {
            lines.push(format!("  Insights: {} found", self.insights.len()));
            for insight in &self.insights {
                lines.push(format!(
                    "    [{:.0}%] {:?}: {}",
                    insight.confidence * 100.0,
                    insight.insight_type,
                    insight.description,
                ));
            }
        }
        lines.push(String::new());

        // Recommendations
        if self.recommendations.is_empty() {
            lines.push("  Experiments: none recommended".to_string());
        } else {
            lines.push(format!(
                "  Experiments: {} recommended",
                self.recommendations.len()
            ));
            for rec in &self.recommendations {
                lines.push(format!(
                    "    [{:?}] {} — {}",
                    rec.priority, rec.experiment_id, rec.hypothesis,
                ));
            }
        }
        lines.push(String::new());

        // TZ directives
        if !self.tz_directives.is_empty() {
            lines.push(format!(
                "  TZ Directives: {} active",
                self.tz_directives.len()
            ));
            for d in &self.tz_directives {
                lines.push(format!("    • {d}"));
            }
            lines.push(String::new());
        }

        // Trends
        if let Some(ref trends) = self.trends {
            let prev_rate = s.resolution_rate - trends.resolution_rate_delta;
            let prev_iters = s.avg_iterations - trends.avg_iterations_delta;
            let prev_insights = self.insights.len() as i32 - trends.insights_count_delta;

            lines.push(format!(
                "  Trends (vs last report, {} compared):",
                trends.reports_compared
            ));
            lines.push(format!(
                "    Resolution rate: {:.0}% -> {:.0}% ({:+.0}%)",
                prev_rate * 100.0,
                s.resolution_rate * 100.0,
                trends.resolution_rate_delta * 100.0,
            ));
            lines.push(format!(
                "    Avg iterations:  {:.1} -> {:.1} ({:+.1})",
                prev_iters, s.avg_iterations, trends.avg_iterations_delta,
            ));
            lines.push(format!(
                "    Insights: {} -> {} ({:+})",
                prev_insights,
                self.insights.len(),
                trends.insights_count_delta,
            ));
            lines.push(String::new());
        }

        // Next steps
        lines.push(format!("  Artifacts: {}", self.artifacts_dir));

        if !self.recommendations.is_empty() {
            let high_priority: Vec<_> = self
                .recommendations
                .iter()
                .filter(|r| matches!(r.priority, crate::recommendations::ExperimentPriority::High))
                .collect();
            if !high_priority.is_empty() {
                lines.push(format!(
                    "  ⚡ {} high-priority experiments ready — run GEPA or update tensorzero.toml",
                    high_priority.len()
                ));
            }
        }

        if s.total_attempts >= 50 && s.resolution_rate < 0.5 {
            lines.push(
                "  ⚠ Resolution rate below 50% — consider running GEPA prompt optimization"
                    .to_string(),
            );
        }

        lines.push("═══════════════════════".to_string());
        lines.join("\n")
    }
}

/// Convenience entry-point for the full autopilot operational loop.
///
/// Constructs an [`AutopilotRunner`] from the supplied parameters, executes
/// every analysis stage in dependency order, writes artifacts to
/// `<repo_root>/.swarm/autopilot/`, and returns the completed
/// [`AutopilotReport`].
///
/// # Parameters
/// - `repo_root`   – root of the repository being analysed.
/// - `window_size` – number of recent mutation records to analyse (0 → default of 50).
/// - `pg_url`      – optional TensorZero Postgres URL for performance directives.
///
/// # Example
/// ```ignore
/// let report = run_autopilot_loop(Path::new("."), 50, None).await?;
/// println!("{}", report.operator_summary());
/// ```
///
/// # Errors
/// Propagates any error returned by [`AutopilotRunner::run`].
pub async fn run_autopilot_loop(
    repo_root: &Path,
    window_size: usize,
    pg_url: Option<&str>,
) -> Result<AutopilotReport> {
    let effective_window = if window_size == 0 { 50 } else { window_size };

    let mut runner = AutopilotRunner::new(repo_root).with_window_size(effective_window);

    if let Some(url) = pg_url {
        runner = runner.with_tz_postgres(url);
    }

    runner.run().await
}

fn write_json<T: Serialize>(path: &Path, data: &T) -> Result<()> {
    let json = serde_json::to_string_pretty(data).with_context(|| {
        format!(
            "Failed to serialize autopilot artifact for {}",
            path.display()
        )
    })?;
    std::fs::write(path, json)
        .with_context(|| format!("Failed to write autopilot artifact: {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_report_summary() {
        let report = AutopilotReport {
            timestamp: "2026-03-30T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 0,
                resolved: 0,
                failed: 0,
                resolution_rate: 0.0,
                avg_iterations: 0.0,
            },
            insights: Vec::new(),
            recommendations: Vec::new(),
            tz_directives: Vec::new(),
            artifacts_dir: "/tmp/test".to_string(),
            trends: None,
        };
        let summary = report.operator_summary();
        assert!(summary.contains("Autopilot Report"));
        assert!(summary.contains("0/0 resolved"));
        assert!(summary.contains("none (need more data)"));
        assert!(summary.contains("none recommended"));
    }

    #[test]
    fn report_with_insights_and_recs() {
        use crate::meta_reflection::InsightType;

        let insight = MetaInsight {
            timestamp: Utc::now(),
            insight_type: InsightType::ModelPerformance,
            description: "Test model underperforms".into(),
            recommendation: "Switch model".into(),
            confidence: 0.9,
            evidence: vec!["issue-1".into()],
        };
        let recs = generate_recommendations(&[insight.clone()]);

        let report = AutopilotReport {
            timestamp: "2026-03-30T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 100,
                resolved: 60,
                failed: 40,
                resolution_rate: 0.6,
                avg_iterations: 2.5,
            },
            insights: vec![insight],
            recommendations: recs,
            tz_directives: vec!["Variant A is 3x faster than B".into()],
            artifacts_dir: "/tmp/test".to_string(),
            trends: None,
        };
        let summary = report.operator_summary();
        assert!(summary.contains("1 found"));
        assert!(summary.contains("1 recommended"));
        assert!(summary.contains("TZ Directives: 1 active"));
        assert!(summary.contains("high-priority"));
    }

    #[test]
    fn low_resolution_rate_warning() {
        let report = AutopilotReport {
            timestamp: "2026-03-30T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 100,
                resolved: 30,
                failed: 70,
                resolution_rate: 0.3,
                avg_iterations: 4.0,
            },
            insights: Vec::new(),
            recommendations: Vec::new(),
            tz_directives: Vec::new(),
            artifacts_dir: "/tmp/test".to_string(),
            trends: None,
        };
        let summary = report.operator_summary();
        assert!(summary.contains("below 50%"));
    }

    #[tokio::test]
    async fn runner_with_empty_archive() {
        let tmp = tempfile::tempdir().unwrap();
        let swarm_dir = tmp.path().join(".swarm");
        std::fs::create_dir_all(&swarm_dir).unwrap();

        let runner = AutopilotRunner::new(tmp.path()).with_window_size(10);
        let report = runner.run().await.unwrap();

        assert_eq!(report.archive_summary.total_attempts, 0);
        assert!(report.insights.is_empty());
        assert!(report.recommendations.is_empty());

        // Check artifacts were written
        let autopilot_dir = swarm_dir.join("autopilot");
        assert!(autopilot_dir.exists());
        assert!(autopilot_dir.join("latest-report.json").exists());
    }

    #[test]
    fn trend_computation_from_two_reports() {
        let previous = AutopilotReport {
            timestamp: "2026-03-23T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 80,
                resolved: 48,
                failed: 32,
                resolution_rate: 0.6,
                avg_iterations: 2.5,
            },
            insights: vec![],
            recommendations: vec![],
            tz_directives: vec![],
            artifacts_dir: "/tmp/old".to_string(),
            trends: None,
        };

        use crate::meta_reflection::InsightType;
        let current_insights = vec![
            MetaInsight {
                timestamp: Utc::now(),
                insight_type: InsightType::ModelPerformance,
                description: "insight-1".into(),
                recommendation: "rec".into(),
                confidence: 0.8,
                evidence: vec![],
            },
            MetaInsight {
                timestamp: Utc::now(),
                insight_type: InsightType::ErrorPattern,
                description: "insight-2".into(),
                recommendation: "rec".into(),
                confidence: 0.7,
                evidence: vec![],
            },
        ];

        let current = AutopilotReport {
            timestamp: "2026-03-30T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 100,
                resolved: 65,
                failed: 35,
                resolution_rate: 0.65,
                avg_iterations: 2.1,
            },
            insights: current_insights,
            recommendations: vec![],
            tz_directives: vec![],
            artifacts_dir: "/tmp/new".to_string(),
            trends: None,
        };

        let trends = compute_trends(&current, &[previous]).unwrap();

        // resolution_rate: 0.65 - 0.6 = 0.05
        assert!((trends.resolution_rate_delta - 0.05).abs() < 1e-9);
        // avg_iterations: 2.1 - 2.5 = -0.4
        assert!((trends.avg_iterations_delta - (-0.4)).abs() < 1e-9);
        // insights: 2 - 0 = 2
        assert_eq!(trends.insights_count_delta, 2);
        assert_eq!(trends.reports_compared, 1);
    }

    #[test]
    fn trend_summary_display() {
        let report = AutopilotReport {
            timestamp: "2026-03-30T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 100,
                resolved: 65,
                failed: 35,
                resolution_rate: 0.65,
                avg_iterations: 2.1,
            },
            insights: vec![],
            recommendations: vec![],
            tz_directives: vec![],
            artifacts_dir: "/tmp/test".to_string(),
            trends: Some(TrendData {
                resolution_rate_delta: 0.05,
                avg_iterations_delta: -0.4,
                insights_count_delta: 2,
                reports_compared: 1,
            }),
        };
        let summary = report.operator_summary();
        assert!(summary.contains("Trends (vs last report"));
        assert!(summary.contains("Resolution rate:"));
        assert!(summary.contains("+5%"));
        assert!(summary.contains("-0.4"));
    }

    #[test]
    fn no_trends_when_no_previous_reports() {
        let current = AutopilotReport {
            timestamp: "2026-03-30T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 10,
                resolved: 5,
                failed: 5,
                resolution_rate: 0.5,
                avg_iterations: 3.0,
            },
            insights: vec![],
            recommendations: vec![],
            tz_directives: vec![],
            artifacts_dir: "/tmp/test".to_string(),
            trends: None,
        };
        let trends = compute_trends(&current, &[]);
        assert!(trends.is_none());
    }

    #[test]
    fn load_previous_reports_from_disk() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path();

        // Write two mock report files
        let report_old = AutopilotReport {
            timestamp: "2026-03-16T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 50,
                resolved: 25,
                failed: 25,
                resolution_rate: 0.5,
                avg_iterations: 3.0,
            },
            insights: vec![],
            recommendations: vec![],
            tz_directives: vec![],
            artifacts_dir: dir.display().to_string(),
            trends: None,
        };
        let report_new = AutopilotReport {
            timestamp: "2026-03-23T12:00:00Z".to_string(),
            archive_summary: ArchiveSummarySnapshot {
                total_attempts: 80,
                resolved: 48,
                failed: 32,
                resolution_rate: 0.6,
                avg_iterations: 2.5,
            },
            insights: vec![],
            recommendations: vec![],
            tz_directives: vec![],
            artifacts_dir: dir.display().to_string(),
            trends: None,
        };

        std::fs::write(
            dir.join("report-20260316-120000.json"),
            serde_json::to_string_pretty(&report_old).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.join("report-20260323-120000.json"),
            serde_json::to_string_pretty(&report_new).unwrap(),
        )
        .unwrap();

        let loaded = load_previous_reports(dir, 7);
        assert_eq!(loaded.len(), 2);
        // Most recent first
        assert_eq!(loaded[0].timestamp, "2026-03-23T12:00:00Z");
        assert_eq!(loaded[1].timestamp, "2026-03-16T12:00:00Z");

        // Limit works
        let loaded_1 = load_previous_reports(dir, 1);
        assert_eq!(loaded_1.len(), 1);
        assert_eq!(loaded_1[0].timestamp, "2026-03-23T12:00:00Z");
    }
}
