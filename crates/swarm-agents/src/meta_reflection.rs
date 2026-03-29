//! Meta-agent reflection loop (Hyperagents pattern).
//!
//! Periodically analyzes mutation archive outcomes and produces structured
//! `MetaInsight` records. These insights are injected into future task prompts
//! via `WorkPacket.relevant_heuristics`, enabling the swarm to learn from its
//! own performance patterns.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

use crate::mutation_archive::{MutationArchive, MutationRecord};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InsightType {
    /// "Model X underperforms on borrow-checker issues"
    ModelPerformance,
    /// "Error category Y has <30% success rate"
    ErrorPattern,
    /// "Route borrow-checker to fast tier instead of coder"
    RoutingAdjustment,
    /// "Prompt version N outperforms version N-1"
    PromptPerformance,
    /// "Skill Z exceeded 80% confidence, recommend as default"
    SkillPromotion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaInsight {
    pub timestamp: DateTime<Utc>,
    pub insight_type: InsightType,
    pub description: String,
    pub recommendation: String,
    pub confidence: f64,
    /// Issue IDs that support this insight.
    pub evidence: Vec<String>,
}

impl MetaInsight {
    /// Human-readable label for the insight type.
    pub fn insight_type_label(&self) -> &'static str {
        match self.insight_type {
            InsightType::ModelPerformance => "model",
            InsightType::ErrorPattern => "error",
            InsightType::RoutingAdjustment => "routing",
            InsightType::PromptPerformance => "prompt",
            InsightType::SkillPromotion => "skill",
        }
    }
}

pub struct MetaReflector {
    archive: MutationArchive,
    insights_path: PathBuf,
}

impl MetaReflector {
    pub fn new(repo_root: &Path) -> Self {
        Self {
            archive: MutationArchive::new(repo_root),
            insights_path: repo_root.join(".swarm").join("meta-insights.jsonl"),
        }
    }

    /// Run reflection on the last `window_size` records. Returns new insights.
    pub fn reflect(&self, window_size: usize) -> Vec<MetaInsight> {
        self.reflect_with_slo_violations(window_size, &[])
    }

    /// Run reflection on the last `window_size` records, incorporating any SLO
    /// violations as additional context. Each entry in `slo_violations` is a
    /// human-readable SLO name (e.g. "Overall success rate") that was violated
    /// in the current session. Returns new insights.
    pub fn reflect_with_slo_violations(
        &self,
        window_size: usize,
        slo_violations: &[String],
    ) -> Vec<MetaInsight> {
        let all = self.archive.load_all();
        let records: Vec<&MutationRecord> = all.iter().rev().take(window_size).collect();
        if records.len() < 5 {
            debug!("Too few records ({}) for reflection", records.len());
            // Still emit SLO-violation insights even with a thin archive.
            return self.analyze_slo_violations(slo_violations);
        }

        let mut insights = Vec::new();
        insights.extend(self.analyze_model_performance(&records));
        insights.extend(self.analyze_error_trends(&records));
        insights.extend(self.analyze_prompt_performance(&records));
        insights.extend(self.analyze_slo_violations(slo_violations));
        insights
    }

    /// Convert SLO violations into `ErrorPattern` insights so they are
    /// persisted and injected into future task prompts.
    fn analyze_slo_violations(&self, slo_violations: &[String]) -> Vec<MetaInsight> {
        slo_violations
            .iter()
            .map(|name| MetaInsight {
                timestamp: chrono::Utc::now(),
                insight_type: InsightType::ErrorPattern,
                description: format!("SLO violated in recent session: {name}"),
                recommendation: format!(
                    "Investigate '{name}' SLO breach — consider adjusting routing or escalation thresholds"
                ),
                confidence: 0.8,
                evidence: vec![],
            })
            .collect()
    }

    /// Persist insights to the JSONL file.
    pub fn save_insights(&self, insights: &[MetaInsight]) {
        if insights.is_empty() {
            return;
        }
        for insight in insights {
            crate::jsonl::append(&self.insights_path, insight);
        }
        info!(count = insights.len(), "Saved meta-insights");
    }

    /// Load the most recent N insights.
    pub fn load_recent_insights(&self, limit: usize) -> Vec<MetaInsight> {
        crate::jsonl::load_tail(&self.insights_path, limit)
    }

    /// Which models underperform on which error categories?
    fn analyze_model_performance(&self, records: &[&MutationRecord]) -> Vec<MetaInsight> {
        // Group by model → (successes, total)
        let mut model_stats: HashMap<&str, (usize, usize)> = HashMap::new();
        for r in records {
            let entry = model_stats.entry(r.model.as_str()).or_insert((0, 0));
            entry.1 += 1;
            if r.resolved {
                entry.0 += 1;
            }
        }

        let mut insights = Vec::new();
        for (model, (successes, total)) in &model_stats {
            if *total < 3 {
                continue;
            }
            let rate = *successes as f64 / *total as f64;
            if rate < 0.3 {
                let evidence: Vec<String> = records
                    .iter()
                    .filter(|r| r.model == *model && !r.resolved)
                    .map(|r| r.issue_id.clone())
                    .collect();
                insights.push(MetaInsight {
                    timestamp: Utc::now(),
                    insight_type: InsightType::ModelPerformance,
                    description: format!(
                        "Model '{}' has {:.0}% success rate ({}/{} issues)",
                        model,
                        rate * 100.0,
                        successes,
                        total
                    ),
                    recommendation: format!(
                        "Consider routing away from '{}' for these task types",
                        model
                    ),
                    confidence: 1.0 - rate,
                    evidence,
                });
            }
        }
        insights
    }

    /// Identify error categories with low success rates.
    fn analyze_error_trends(&self, records: &[&MutationRecord]) -> Vec<MetaInsight> {
        let mut cat_stats: HashMap<&str, (usize, usize)> = HashMap::new();
        for r in records {
            for cat in &r.error_categories {
                let entry = cat_stats.entry(cat.as_str()).or_insert((0, 0));
                entry.1 += 1;
                if r.resolved {
                    entry.0 += 1;
                }
            }
        }

        let mut insights = Vec::new();
        for (cat, (successes, total)) in &cat_stats {
            if *total < 3 {
                continue;
            }
            let rate = *successes as f64 / *total as f64;
            if rate < 0.4 {
                insights.push(MetaInsight {
                    timestamp: Utc::now(),
                    insight_type: InsightType::ErrorPattern,
                    description: format!(
                        "Error category '{}' has {:.0}% resolution rate ({}/{})",
                        cat,
                        rate * 100.0,
                        successes,
                        total
                    ),
                    recommendation: format!(
                        "Issues with '{}' errors may need specialized handling or escalation",
                        cat
                    ),
                    confidence: 1.0 - rate,
                    evidence: vec![],
                });
            }
        }
        insights
    }

    /// Compare prompt version performance.
    fn analyze_prompt_performance(&self, records: &[&MutationRecord]) -> Vec<MetaInsight> {
        let mut version_stats: HashMap<&str, (usize, usize)> = HashMap::new();
        for r in records {
            let entry = version_stats
                .entry(r.prompt_version.as_str())
                .or_insert((0, 0));
            entry.1 += 1;
            if r.resolved {
                entry.0 += 1;
            }
        }

        // Only emit if we have 2+ versions to compare
        if version_stats.len() < 2 {
            return vec![];
        }

        let mut insights = Vec::new();
        let best = version_stats
            .iter()
            .filter(|(_, (_, total))| *total >= 3)
            .max_by(|(_, (s1, t1)), (_, (s2, t2))| {
                let r1 = *s1 as f64 / *t1 as f64;
                let r2 = *s2 as f64 / *t2 as f64;
                r1.partial_cmp(&r2).unwrap_or(std::cmp::Ordering::Equal)
            });

        if let Some((version, (successes, total))) = best {
            let rate = *successes as f64 / *total as f64;
            insights.push(MetaInsight {
                timestamp: Utc::now(),
                insight_type: InsightType::PromptPerformance,
                description: format!(
                    "Prompt version '{}' has the highest success rate: {:.0}% ({}/{})",
                    version,
                    rate * 100.0,
                    successes,
                    total
                ),
                recommendation: format!("Prefer prompt version '{}'", version),
                confidence: rate,
                evidence: vec![],
            });
        }

        insights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(
        model: &str,
        resolved: bool,
        iterations: u32,
        error_cats: Vec<&str>,
    ) -> MutationRecord {
        MutationRecord {
            timestamp: Utc::now(),
            issue_id: format!("test-{}", rand_id()),
            issue_title: "test issue".into(),
            language: "rust".into(),
            resolved,
            iterations,
            tier: "worker".into(),
            model: model.into(),
            prompt_version: "v9.0.0".into(),
            error_categories: error_cats.into_iter().map(String::from).collect(),
            files_changed: vec![],
            lines_added: 0,
            lines_removed: 0,
            auto_fix_only: false,
            duration_secs: 60,
            first_failure_gate: None,
            failure_reason: None,
            pivot_decisions: Vec::new(),
        }
    }

    fn rand_id() -> String {
        use std::time::SystemTime;
        format!(
            "{:x}",
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        )
    }

    #[test]
    fn test_model_performance_low_rate() {
        let records: Vec<MutationRecord> = (0..5)
            .map(|i| make_record("bad-model", i == 0, 5, vec!["borrow_checker"]))
            .collect();
        let refs: Vec<&MutationRecord> = records.iter().collect();

        let reflector = MetaReflector {
            archive: MutationArchive::new(Path::new("/tmp/nonexistent")),
            insights_path: PathBuf::from("/tmp/nonexistent-insights.jsonl"),
        };
        let insights = reflector.analyze_model_performance(&refs);
        assert_eq!(insights.len(), 1);
        assert!(insights[0].description.contains("bad-model"));
        assert!(insights[0].description.contains("20%"));
    }

    #[test]
    fn test_error_trend_detection() {
        let records: Vec<MutationRecord> = (0..4)
            .map(|i| make_record("model-a", i == 0, 3, vec!["lifetime"]))
            .collect();
        let refs: Vec<&MutationRecord> = records.iter().collect();

        let reflector = MetaReflector {
            archive: MutationArchive::new(Path::new("/tmp/nonexistent")),
            insights_path: PathBuf::from("/tmp/nonexistent-insights.jsonl"),
        };
        let insights = reflector.analyze_error_trends(&refs);
        assert_eq!(insights.len(), 1);
        assert!(insights[0].description.contains("lifetime"));
    }

    #[test]
    fn test_no_insights_for_good_performance() {
        let records: Vec<MutationRecord> = (0..5)
            .map(|_| make_record("good-model", true, 1, vec!["syntax"]))
            .collect();
        let refs: Vec<&MutationRecord> = records.iter().collect();

        let reflector = MetaReflector {
            archive: MutationArchive::new(Path::new("/tmp/nonexistent")),
            insights_path: PathBuf::from("/tmp/nonexistent-insights.jsonl"),
        };
        assert!(reflector.analyze_model_performance(&refs).is_empty());
        assert!(reflector.analyze_error_trends(&refs).is_empty());
    }
}
