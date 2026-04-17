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
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::info;

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
    /// Pivot decisions made during the session (strategy changes between iterations).
    ///
    /// Each entry records when the manager decided to change approach mid-session
    /// rather than refining the current approach. Non-empty means the session
    /// was adaptive rather than linear.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pivot_decisions: Vec<PivotDecision>,
}

/// A record of an explicit pivot decision made by the manager.
///
/// Created when the manager judges that the current approach has a fundamental
/// flaw (not just a fixable bug) and chooses a different strategy. The pivot
/// is recorded so meta-reflection can analyze which pivot strategies succeed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PivotDecision {
    /// Iteration number when the pivot was decided.
    pub iteration: usize,
    /// Why the current approach was judged unworkable.
    pub rationale: String,
    /// The new strategy description.
    pub pivot_strategy: String,
    /// Confidence in the new strategy (0.0–1.0 as reported by the manager).
    pub strategy_confidence: f32,
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
        crate::jsonl::append(&self.path, record);
        info!(
            issue = %record.issue_id,
            resolved = record.resolved,
            iterations = record.iterations,
            "Mutation archive: recorded outcome"
        );
    }

    /// Load all records from the archive.
    pub fn load_all(&self) -> Vec<MutationRecord> {
        crate::jsonl::load_all(&self.path)
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
        records.sort_by_key(|record| std::cmp::Reverse(record.timestamp));
        records.truncate(limit);
        records
    }

    /// Query the archive for resolved records similar to the current issue.
    ///
    /// Similarity is scored by overlap in error categories AND file paths.
    /// Returns the top `limit` records sorted by similarity score (descending).
    ///
    /// Inspired by Robin (Future House): downstream results actively reshape
    /// upstream proposals. This transforms the passive archive into an active
    /// feedback loop — successful fix patterns are injected into prompts.
    pub fn query_similar(
        &self,
        error_categories: &[String],
        files_changed: &[String],
        limit: usize,
    ) -> Vec<MutationRecord> {
        if error_categories.is_empty() && files_changed.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(f64, MutationRecord)> = self
            .load_all()
            .into_iter()
            .filter(|r| r.resolved) // Only learn from successes
            .map(|r| {
                // Score: error category overlap + file path overlap
                let cat_overlap = error_categories
                    .iter()
                    .filter(|c| r.error_categories.contains(c))
                    .count() as f64;
                let file_overlap = files_changed
                    .iter()
                    .filter(|f| {
                        r.files_changed
                            .iter()
                            .any(|rf| rf.contains(f.as_str()) || f.contains(rf.as_str()))
                    })
                    .count() as f64;
                let score = cat_overlap * 2.0 + file_overlap; // Weight categories higher
                (score, r)
            })
            .filter(|(score, _)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored.into_iter().map(|(_, r)| r).collect()
    }

    /// Query for anti-patterns: records with matching error categories that FAILED.
    ///
    /// Returns failure reasons to inject as "do not try" blocklist in prompts.
    pub fn query_anti_patterns(&self, error_categories: &[String], limit: usize) -> Vec<String> {
        self.load_all()
            .into_iter()
            .filter(|r| !r.resolved && r.failure_reason.is_some())
            .filter(|r| {
                error_categories
                    .iter()
                    .any(|c| r.error_categories.contains(c))
            })
            .rev() // Most recent first
            .take(limit)
            .filter_map(|r| {
                r.failure_reason.map(|reason| {
                    format!(
                        "{} (issue {}, {} iterations): {}",
                        r.issue_title, r.issue_id, r.iterations, reason
                    )
                })
            })
            .collect()
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

    /// Success rate grouped by prompt version (Hyperagents prompt coevolution).
    pub fn success_rate_by_prompt_version(&self) -> HashMap<String, (usize, usize, f64)> {
        let records = self.load_all();
        let mut stats: HashMap<String, (usize, usize)> = HashMap::new();
        for r in &records {
            let entry = stats.entry(r.prompt_version.clone()).or_insert((0, 0));
            entry.1 += 1;
            if r.resolved {
                entry.0 += 1;
            }
        }
        stats
            .into_iter()
            .map(|(version, (successes, total))| {
                let rate = if total > 0 {
                    successes as f64 / total as f64
                } else {
                    0.0
                };
                (version, (successes, total, rate))
            })
            .collect()
    }

    /// UCB1 score for a specific model+error_category lineage (Hyperagents generational selection).
    pub fn ucb_score_for_lineage(&self, model: &str, error_category: &str) -> f64 {
        let records = self.load_all();
        let total_all = records.len() as f64;
        if total_all == 0.0 {
            return f64::MAX; // Encourage exploration
        }

        let lineage: Vec<&MutationRecord> = records
            .iter()
            .filter(|r| r.model == model && r.error_categories.iter().any(|c| c == error_category))
            .collect();

        let n = lineage.len() as f64;
        if n == 0.0 {
            return f64::MAX; // Never tried → explore
        }

        let successes = lineage.iter().filter(|r| r.resolved).count() as f64;
        let mean_reward = successes / n;
        let exploration = (2.0 * total_all.ln() / n).sqrt();
        mean_reward + exploration
    }

    /// Recommend the best model for a given set of error categories using UCB1 (Hyperagents).
    ///
    /// Returns `None` if insufficient data (< `min_samples` per candidate).
    pub fn recommend_model(
        &self,
        error_categories: &[String],
        candidate_models: &[String],
        min_samples: usize,
    ) -> Option<String> {
        if error_categories.is_empty() || candidate_models.is_empty() {
            return None;
        }

        let records = self.load_all();

        // Check that at least one candidate has enough samples
        let has_sufficient_data = candidate_models.iter().any(|model| {
            let count = records
                .iter()
                .filter(|r| {
                    r.model == *model
                        && r.error_categories
                            .iter()
                            .any(|c| error_categories.contains(c))
                })
                .count();
            count >= min_samples
        });

        if !has_sufficient_data {
            return None;
        }

        // Compute aggregate UCB score for each candidate using the pre-loaded records.
        // Inline the UCB1 formula to avoid N×M load_all() calls via ucb_score_for_lineage.
        let total_all = records.len() as f64;
        candidate_models
            .iter()
            .map(|model| {
                let avg_ucb: f64 = error_categories
                    .iter()
                    .map(|cat| {
                        let lineage_count = records
                            .iter()
                            .filter(|r| {
                                r.model == *model
                                    && r.error_categories.iter().any(|c| c == cat.as_str())
                            })
                            .count() as f64;
                        if lineage_count == 0.0 {
                            return f64::MAX;
                        }
                        let successes = records
                            .iter()
                            .filter(|r| {
                                r.model == *model
                                    && r.resolved
                                    && r.error_categories.iter().any(|c| c == cat.as_str())
                            })
                            .count() as f64;
                        let mean_reward = successes / lineage_count;
                        let exploration = (2.0 * total_all.ln() / lineage_count).sqrt();
                        mean_reward + exploration
                    })
                    .sum::<f64>()
                    / error_categories.len() as f64;
                (model.clone(), avg_ucb)
            })
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(model, _)| model)
    }

    /// Format similar past fixes and anti-patterns as a prompt context section.
    ///
    /// Returns empty string if no relevant history exists.
    pub fn format_feedback_context(
        &self,
        error_categories: &[String],
        files_changed: &[String],
    ) -> String {
        let successes = self.query_similar(error_categories, files_changed, 3);
        let anti_patterns = self.query_anti_patterns(error_categories, 2);

        if successes.is_empty() && anti_patterns.is_empty() {
            return String::new();
        }

        let mut ctx = String::from("## Feedback from Past Issues (Robin pattern)\n\n");

        if !successes.is_empty() {
            ctx.push_str("**Successful patterns** (similar issues that were resolved):\n");
            for r in &successes {
                ctx.push_str(&format!(
                    "- `{}` ({} iters, {}): changed {}\n",
                    r.issue_title,
                    r.iterations,
                    r.tier,
                    r.files_changed.join(", "),
                ));
            }
            ctx.push('\n');
        }

        if !anti_patterns.is_empty() {
            ctx.push_str(
                "**Anti-patterns** (approaches that FAILED on similar errors — do NOT repeat):\n",
            );
            for ap in &anti_patterns {
                ctx.push_str(&format!("- {ap}\n"));
            }
            ctx.push('\n');
        }

        ctx
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
        let mut model_stats: HashMap<String, (usize, usize)> = HashMap::new();
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

    /// Generate context about past similar issues for the manager prompt.
    ///
    /// Returns a short text block describing past resolutions of similar
    /// error types, which models worked, and how many iterations they took.
    /// Returns None if no relevant history exists.
    pub fn context_for_issue(&self, error_categories: &[String]) -> Option<String> {
        let records = self.load_all();
        if records.is_empty() {
            return None;
        }

        let mut parts = Vec::new();

        // Summary stats
        let summary = self.summary();
        parts.push(format!(
            "Prior run history: {} attempts, {} resolved ({:.0}% success rate), \n             avg {:.1} iterations when successful.",
            summary.total_attempts,
            summary.resolved,
            if summary.total_attempts > 0 {
                summary.resolved as f64 / summary.total_attempts as f64 * 100.0
            } else {
                0.0
            },
            summary.avg_iterations_to_resolve,
        ));

        // Strategy advice from successful runs
        let successful: Vec<&MutationRecord> = records.iter().filter(|r| r.resolved).collect();
        if !successful.is_empty() {
            let avg_iter: f64 = successful.iter().map(|r| r.iterations as f64).sum::<f64>()
                / successful.len() as f64;
            if avg_iter <= 2.0 {
                parts.push(
                    "Strategy: past fixes resolved quickly (<=2 iterations). \n                     Read the target file first, make a focused edit, and let the verifier confirm.".to_string(),
                );
            } else {
                parts.push(format!(
                    "Strategy: past fixes averaged {avg_iter:.0} iterations. \n                     Consider reading the file structure carefully before editing.",
                ));
            }
        }

        // Failure pattern analysis
        let failed: Vec<&MutationRecord> = records.iter().filter(|r| !r.resolved).collect();
        if !failed.is_empty() {
            // Check common failure reasons
            let timeout_count = failed.iter().filter(|r| r.iterations >= 10).count();
            if timeout_count > failed.len() / 2 {
                parts.push(
                    "WARNING: Most past failures exhausted all iterations. \n                     Make your edits early — do not spend turns only reading files.".to_string(),
                );
            }
        }

        // Similar error category matches
        if !error_categories.is_empty() {
            for cat in error_categories {
                let similar = self.query_by_error(cat, 2);
                for r in &similar {
                    parts.push(format!(
                        "Similar fix: \"{}\" resolved in {} iteration(s).",
                        r.issue_title, r.iterations,
                    ));
                }
            }
        }

        if parts.is_empty() {
            None
        } else {
            Some(format!(
                "## Lessons from Past Runs\n\n{}\n",
                parts.join("\n")
            ))
        }
    }

    /// Query resolved records whose title shares keywords with `query`.
    ///
    /// Used at session start (before any errors exist) to inject iteration-count
    /// estimates based on similar past issues. Returns up to `limit` records
    /// sorted by keyword overlap score (descending).
    pub fn query_by_keywords(&self, query: &str, limit: usize) -> Vec<MutationRecord> {
        let keywords: Vec<String> = query
            .split_whitespace()
            .filter(|w| w.len() > 3)
            .map(|w| w.to_lowercase())
            .collect();
        if keywords.is_empty() {
            return Vec::new();
        }
        let mut scored: Vec<(usize, MutationRecord)> = self
            .load_all()
            .into_iter()
            .filter(|r| r.resolved)
            .map(|r| {
                let title_lower = r.issue_title.to_lowercase();
                let score = keywords
                    .iter()
                    .filter(|kw| title_lower.contains(kw.as_str()))
                    .count();
                (score, r)
            })
            .filter(|(score, _)| *score > 0)
            .collect();
        scored.sort_by_key(|entry| std::cmp::Reverse(entry.0));
        scored.truncate(limit);
        scored.into_iter().map(|(_, r)| r).collect()
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
    pub model_stats: HashMap<String, (usize, usize)>,
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

/// A candidate for promotion to the skill library (Hyperagents pattern).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCandidate {
    pub error_categories: Vec<String>,
    pub files_changed: Vec<String>,
    pub approach_summary: String,
    pub model_used: String,
    pub iterations: u32,
}

impl MutationArchive {
    /// Extract a skill candidate from a successful mutation record.
    /// Only promotes records that resolved quickly (≤3 iterations) with known error categories.
    pub fn extract_skill_candidate(record: &MutationRecord) -> Option<SkillCandidate> {
        if !record.resolved || record.iterations > 3 {
            return None;
        }
        if record.error_categories.is_empty() {
            return None;
        }
        Some(SkillCandidate {
            error_categories: record.error_categories.clone(),
            files_changed: record.files_changed.clone(),
            approach_summary: format!(
                "Resolved '{}' in {} iteration(s) using {}",
                record.issue_title, record.iterations, record.model
            ),
            model_used: record.model.clone(),
            iterations: record.iterations,
        })
    }
}

/// Build a MutationRecord from orchestration context.
///
/// This is the main entry point called from the orchestrator at each
/// success/failure exit point.
#[allow(clippy::too_many_arguments)]
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
        pivot_decisions: Vec::new(),
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

    #[test]
    fn test_success_rate_by_prompt_version() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        // v1: 2 successes, 1 failure → 66.7% success rate
        for resolved in [true, true, false] {
            let mut r = build_record("id", "title", "rust", resolved, 1, "W", "model-a", 60);
            r.prompt_version = "v1".into();
            archive.record(&r);
        }

        // v2: 1 success, 0 failures → 100% success rate
        let mut r = build_record("id", "title", "rust", true, 1, "W", "model-a", 60);
        r.prompt_version = "v2".into();
        archive.record(&r);

        let rates = archive.success_rate_by_prompt_version();

        let (successes, total, rate) = rates["v1"];
        assert_eq!(successes, 2);
        assert_eq!(total, 3);
        assert!((rate - 2.0 / 3.0).abs() < 0.01);

        let (successes, total, rate) = rates["v2"];
        assert_eq!(successes, 1);
        assert_eq!(total, 1);
        assert!((rate - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_ucb_score_for_lineage() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        // Add one record for model-a with category "borrow_checker"
        let mut r = build_record("id1", "title", "rust", true, 1, "W", "model-a", 60);
        r.error_categories = vec!["borrow_checker".into()];
        archive.record(&r);

        // model-a with borrow_checker should have a finite UCB score
        let score_a = archive.ucb_score_for_lineage("model-a", "borrow_checker");
        assert!(score_a.is_finite());

        // model-b (never tried for borrow_checker) should get MAX (exploration)
        let score_b = archive.ucb_score_for_lineage("model-b", "borrow_checker");
        assert_eq!(score_b, f64::MAX);

        // model-a with a different category should also get MAX
        let score_a_other = archive.ucb_score_for_lineage("model-a", "type_mismatch");
        assert_eq!(score_a_other, f64::MAX);
    }

    #[test]
    fn test_recommend_model_picks_best() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        let cats = vec!["borrow_checker".to_string()];

        // model-a: 3 successes out of 3 (100% success rate)
        for _ in 0..3 {
            let mut r = build_record("id", "title", "rust", true, 1, "W", "model-a", 60);
            r.error_categories = cats.clone();
            archive.record(&r);
        }

        // model-b: 0 successes out of 3 (0% success rate)
        for _ in 0..3 {
            let mut r = build_record("id", "title", "rust", false, 5, "W", "model-b", 300);
            r.error_categories = cats.clone();
            archive.record(&r);
        }

        let candidates = vec!["model-a".to_string(), "model-b".to_string()];
        let recommended = archive.recommend_model(&cats, &candidates, 1);
        assert_eq!(recommended.as_deref(), Some("model-a"));
    }

    #[test]
    fn test_recommend_model_returns_none_insufficient_data() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        let cats = vec!["type_mismatch".to_string()];

        // Only 1 sample for model-a, but min_samples = 5
        let mut r = build_record("id", "title", "rust", true, 1, "W", "model-a", 60);
        r.error_categories = cats.clone();
        archive.record(&r);

        let candidates = vec!["model-a".to_string(), "model-b".to_string()];
        let recommended = archive.recommend_model(&cats, &candidates, 5);
        assert!(recommended.is_none());
    }

    #[test]
    fn test_recommend_model_empty_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let archive = MutationArchive::new(dir.path());

        // Empty error_categories → None
        assert!(archive
            .recommend_model(&[], &["model-a".to_string()], 1)
            .is_none());

        // Empty candidates → None
        assert!(archive
            .recommend_model(&["borrow_checker".to_string()], &[], 1)
            .is_none());
    }
}
