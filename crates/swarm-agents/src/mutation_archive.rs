//! Mutation Archive — Evolutionary tracking of issue resolution outcomes.
//!
//! Records every issue attempt's outcome (success/failure, model used, iterations,
//! error types, files changed) as append-only JSONL. This data feeds:
//!
//! - **UCB model selection** (Phase 4b): dynamically route issues to models
//!   based on historical success rates per error type.
//! - **Prompt co-evolution** (Phase 4c): track which prompt versions produce
//!   better outcomes and evolve them over time.
//! - **Fitness scoring** (Phase 4d): multi-dimensional quality metrics beyond pass/fail.
//!
//! Inspired by ShinkaEvolve's population-based evolution and mutation tracking.
//!
//! # Storage
//!
//! Archive is stored as `.swarm/mutation-archive.jsonl` in the target repo root.
//! Each line is a self-contained JSON record. The file is append-only and can be
//! rotated or compacted externally.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use tracing::{info, warn};

/// A single mutation outcome record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationRecord {
    /// Timestamp of the attempt.
    pub timestamp: DateTime<Utc>,
    /// Issue ID (e.g., "CF-LIBS-improved-q14").
    pub issue_id: String,
    /// Issue title (for human readability in the archive).
    pub issue_title: String,
    /// Target repo language (from profile).
    pub language: String,
    /// Whether the issue was resolved.
    pub resolved: bool,
    /// Number of iterations used.
    pub iterations: u32,
    /// Which model tier resolved it (or the last tier attempted).
    pub tier: String,
    /// Primary model used (e.g., "claude-opus-4-6", "Qwen3.5-397B-A17B").
    pub model: String,
    /// Prompt version at time of attempt.
    pub prompt_version: String,
    /// Error categories encountered (from verifier).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub error_categories: Vec<String>,
    /// Files changed by the agent.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub files_changed: Vec<String>,
    /// Number of lines added.
    pub lines_added: u32,
    /// Number of lines removed.
    pub lines_removed: u32,
    /// Whether auto-fix resolved it (no LLM needed).
    pub auto_fix_only: bool,
    /// Wall-clock duration in seconds.
    pub duration_secs: u64,
    /// First failure gate (if any).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_failure_gate: Option<String>,
    /// Reason for failure (if not resolved).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failure_reason: Option<String>,
}

/// Append-only mutation archive stored as JSONL.
pub struct MutationArchive {
    path: PathBuf,
}

impl MutationArchive {
    /// Create an archive writer for the given repo root.
    ///
    /// The archive file is created at `{repo_root}/.swarm/mutation-archive.jsonl`.
    /// Creates the `.swarm/` directory if needed.
    pub fn new(repo_root: &Path) -> Self {
        let path = repo_root.join(".swarm").join("mutation-archive.jsonl");
        Self { path }
    }

    /// Append a record to the archive.
    pub fn record(&self, record: &MutationRecord) {
        // Ensure .swarm/ exists
        if let Some(parent) = self.path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        match serde_json::to_string(record) {
            Ok(json) => {
                match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&self.path)
                {
                    Ok(mut file) => {
                        if let Err(e) = writeln!(file, "{json}") {
                            warn!(error = %e, "Failed to write mutation record");
                        } else {
                            info!(
                                issue = %record.issue_id,
                                resolved = record.resolved,
                                iterations = record.iterations,
                                "Mutation archive: recorded outcome"
                            );
                        }
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            path = %self.path.display(),
                            "Failed to open mutation archive"
                        );
                    }
                }
            }
            Err(e) => {
                warn!(error = %e, "Failed to serialize mutation record");
            }
        }
    }

    /// Load all records from the archive.
    pub fn load_all(&self) -> Vec<MutationRecord> {
        match std::fs::read_to_string(&self.path) {
            Ok(content) => content
                .lines()
                .filter_map(|line| serde_json::from_str(line).ok())
                .collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Query the archive for records matching an error category.
    ///
    /// Returns the most recent `limit` records where `error_categories`
    /// contains the given category. Useful for seeding prompts with
    /// successful fix patterns for similar errors.
    pub fn query_by_error(&self, category: &str, limit: usize) -> Vec<MutationRecord> {
        let mut records: Vec<MutationRecord> = self
            .load_all()
            .into_iter()
            .filter(|r| r.resolved && r.error_categories.iter().any(|c| c == category))
            .collect();
        records.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        records.truncate(limit);
        records
    }

    /// Compute success rate for a given model across all recorded attempts.
    ///
    /// Returns `(success_count, total_count, rate)`. Used for UCB model selection.
    pub fn model_success_rate(&self, model: &str) -> (usize, usize, f64) {
        let records = self.load_all();
        let total = records.iter().filter(|r| r.model == model).count();
        let success = records
            .iter()
            .filter(|r| r.model == model && r.resolved)
            .count();
        let rate = if total > 0 {
            success as f64 / total as f64
        } else {
            0.0
        };
        (success, total, rate)
    }

    /// Compute UCB1 score for a model.
    ///
    /// `UCB = success_rate + sqrt(2 * ln(total_attempts) / model_attempts)`
    ///
    /// Higher score = more attractive (balances exploitation + exploration).
    /// Models with zero attempts get infinite UCB (always explore first).
    pub fn ucb_score(&self, model: &str) -> f64 {
        let records = self.load_all();
        let total_attempts = records.len();
        let model_records: Vec<&MutationRecord> =
            records.iter().filter(|r| r.model == model).collect();
        let model_attempts = model_records.len();

        if model_attempts == 0 {
            return f64::INFINITY; // Explore untested models
        }

        let success = model_records.iter().filter(|r| r.resolved).count();
        let success_rate = success as f64 / model_attempts as f64;
        let exploration = (2.0 * (total_attempts as f64).ln() / model_attempts as f64).sqrt();

        success_rate + exploration
    }

    /// Get a summary of archive statistics.
    pub fn summary(&self) -> ArchiveSummary {
        let records = self.load_all();
        let total = records.len();
        let resolved = records.iter().filter(|r| r.resolved).count();
        let auto_fix = records.iter().filter(|r| r.auto_fix_only).count();
        let avg_iterations = if resolved > 0 {
            records
                .iter()
                .filter(|r| r.resolved)
                .map(|r| r.iterations as f64)
                .sum::<f64>()
                / resolved as f64
        } else {
            0.0
        };

        // Model breakdown
        let mut model_stats: std::collections::HashMap<String, (usize, usize)> =
            std::collections::HashMap::new();
        for r in &records {
            let entry = model_stats.entry(r.model.clone()).or_insert((0, 0));
            entry.1 += 1; // total
            if r.resolved {
                entry.0 += 1; // success
            }
        }

        ArchiveSummary {
            total_attempts: total,
            resolved,
            failed: total - resolved,
            auto_fix_only: auto_fix,
            avg_iterations_to_resolve: avg_iterations,
            model_stats,
        }
    }

    /// Archive file path.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Summary statistics from the mutation archive.
#[derive(Debug, Clone, Serialize)]
pub struct ArchiveSummary {
    pub total_attempts: usize,
    pub resolved: usize,
    pub failed: usize,
    pub auto_fix_only: usize,
    pub avg_iterations_to_resolve: f64,
    /// Model → (successes, total_attempts)
    pub model_stats: std::collections::HashMap<String, (usize, usize)>,
}

impl std::fmt::Display for ArchiveSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Archive: {} attempts, {} resolved ({:.0}%), {} auto-fix, avg {:.1} iterations",
            self.total_attempts,
            self.resolved,
            if self.total_attempts > 0 {
                self.resolved as f64 / self.total_attempts as f64 * 100.0
            } else {
                0.0
            },
            self.auto_fix_only,
            self.avg_iterations_to_resolve,
        )
    }
}

/// Build a MutationRecord from orchestration context.
///
/// This is the main entry point called from the orchestrator at each
/// success/failure exit point.
pub fn build_record(
    issue_id: &str,
    issue_title: &str,
    language: &str,
    resolved: bool,
    iterations: u32,
    tier: &str,
    model: &str,
    duration_secs: u64,
) -> MutationRecord {
    MutationRecord {
        timestamp: Utc::now(),
        issue_id: issue_id.to_string(),
        issue_title: issue_title.to_string(),
        language: language.to_string(),
        resolved,
        iterations,
        tier: tier.to_string(),
        model: model.to_string(),
        prompt_version: crate::prompts::PROMPT_VERSION.to_string(),
        error_categories: Vec::new(),
        files_changed: Vec::new(),
        lines_added: 0,
        lines_removed: 0,
        auto_fix_only: false,
        duration_secs,
        first_failure_gate: None,
        failure_reason: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let swarm_dir = dir.path().join(".swarm");
        std::fs::create_dir_all(&swarm_dir).unwrap();

        let archive = MutationArchive::new(dir.path());

        let record = build_record(
            "test-001",
            "Fix import error",
            "python",
            true,
            2,
            "Worker",
            "claude-opus-4-6",
            120,
        );
        archive.record(&record);

        let loaded = archive.load_all();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].issue_id, "test-001");
        assert!(loaded[0].resolved);
    }

    #[test]
    fn test_model_success_rate() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        // 2 successes, 1 failure for model A
        for (resolved, model) in [(true, "A"), (true, "A"), (false, "A"), (true, "B")] {
            let mut record = build_record("id", "title", "python", resolved, 1, "W", model, 60);
            record.resolved = resolved;
            archive.record(&record);
        }

        let (success, total, rate) = archive.model_success_rate("A");
        assert_eq!(success, 2);
        assert_eq!(total, 3);
        assert!((rate - 0.6667).abs() < 0.01);

        let (success, total, _) = archive.model_success_rate("B");
        assert_eq!(success, 1);
        assert_eq!(total, 1);
    }

    #[test]
    fn test_ucb_score_untested_model() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        // Record some data for model A
        let record = build_record("id", "t", "python", true, 1, "W", "A", 60);
        archive.record(&record);

        // Untested model B should have infinite UCB (explore first)
        assert!(archive.ucb_score("B").is_infinite());
        // Tested model A should have finite UCB
        assert!(archive.ucb_score("A").is_finite());
    }

    #[test]
    fn test_query_by_error() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        let mut record1 = build_record("id1", "t", "python", true, 1, "W", "A", 60);
        record1.error_categories = vec!["LintViolation".into()];
        archive.record(&record1);

        let mut record2 = build_record("id2", "t", "python", true, 2, "W", "A", 120);
        record2.error_categories = vec!["TypeCheckFailure".into()];
        archive.record(&record2);

        let mut record3 = build_record("id3", "t", "python", false, 5, "W", "A", 300);
        record3.error_categories = vec!["LintViolation".into()];
        archive.record(&record3);

        let lint_records = archive.query_by_error("LintViolation", 10);
        // Only resolved records
        assert_eq!(lint_records.len(), 1);
        assert_eq!(lint_records[0].issue_id, "id1");
    }

    #[test]
    fn test_summary() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        archive.record(&build_record("1", "t", "py", true, 2, "W", "A", 60));
        archive.record(&build_record("2", "t", "py", true, 4, "W", "B", 120));
        archive.record(&build_record("3", "t", "py", false, 10, "C", "A", 300));

        let summary = archive.summary();
        assert_eq!(summary.total_attempts, 3);
        assert_eq!(summary.resolved, 2);
        assert_eq!(summary.failed, 1);
        assert!((summary.avg_iterations_to_resolve - 3.0).abs() < 0.01);
        assert_eq!(format!("{summary}").contains("3 attempts"), true);
    }

    #[test]
    fn test_empty_archive() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());
        assert!(archive.load_all().is_empty());
        assert_eq!(archive.summary().total_attempts, 0);
    }
}
