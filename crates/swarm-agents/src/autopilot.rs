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

use chrono::Utc;
use serde::Serialize;
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
#[derive(Debug, Serialize)]
pub struct AutopilotReport {
    pub timestamp: String,
    pub archive_summary: ArchiveSummarySnapshot,
    pub insights: Vec<MetaInsight>,
    pub recommendations: Vec<ExperimentRecommendation>,
    pub tz_directives: Vec<String>,
    pub artifacts_dir: String,
}

/// Serializable snapshot of archive stats (ArchiveSummary has HashMap which
/// serializes fine, but we want a clean top-level view).
#[derive(Debug, Serialize)]
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
    pub async fn run(&self) -> AutopilotReport {
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
        if let Err(e) = std::fs::create_dir_all(&artifacts_dir) {
            warn!(error = %e, "Failed to create autopilot artifacts directory");
        }

        let run_id = now.format("%Y%m%d-%H%M%S").to_string();

        // Write insights
        let insights_path = artifacts_dir.join(format!("insights-{run_id}.json"));
        write_json(&insights_path, &insights);

        // Write recommendations
        let recs_path = artifacts_dir.join(format!("recommendations-{run_id}.json"));
        write_json(&recs_path, &recommendations);

        // Write flat runner-input records (one row per variant, ready for adaptive runner)
        let runner_inputs = to_runner_inputs(&recommendations);
        let runner_path = artifacts_dir.join(format!("runner-inputs-{run_id}.json"));
        write_json(&runner_path, &runner_inputs);

        // Write full report
        let report = AutopilotReport {
            timestamp: now.to_rfc3339(),
            archive_summary: summary,
            insights,
            recommendations,
            tz_directives,
            artifacts_dir: artifacts_dir.display().to_string(),
        };
        let report_path = artifacts_dir.join(format!("report-{run_id}.json"));
        write_json(&report_path, &report);

        // Write a symlink-like "latest" pointer
        let latest_path = artifacts_dir.join("latest-report.json");
        write_json(&latest_path, &report);

        info!(
            dir = %artifacts_dir.display(),
            run_id,
            "Autopilot: artifacts written"
        );

        report
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

fn write_json<T: Serialize>(path: &Path, data: &T) {
    match serde_json::to_string_pretty(data) {
        Ok(json) => {
            if let Err(e) = std::fs::write(path, json) {
                warn!(path = %path.display(), error = %e, "Failed to write autopilot artifact");
            }
        }
        Err(e) => {
            warn!(error = %e, "Failed to serialize autopilot artifact");
        }
    }
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
        let report = runner.run().await;

        assert_eq!(report.archive_summary.total_attempts, 0);
        assert!(report.insights.is_empty());
        assert!(report.recommendations.is_empty());

        // Check artifacts were written
        let autopilot_dir = swarm_dir.join("autopilot");
        assert!(autopilot_dir.exists());
        assert!(autopilot_dir.join("latest-report.json").exists());
    }
}
