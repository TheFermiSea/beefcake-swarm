//! Experiment History Database — stores past resolution outcomes for the LDEA loop.
//!
//! Each successful (or failed) swarm resolution is recorded as an [`ExperimentNode`].
//! The database enables:
//! - Retrieval of similar past resolutions by embedding cosine similarity
//! - UCB1-based sampling balancing exploitation vs exploration
//! - Structured context for worker task prompts
//!
//! Source: ASI-Evolve (arxiv:2603.29640) database/database.py

use crate::embedding::cosine_similarity;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::jsonl;

/// A single experiment node — one swarm resolution attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentNode {
    pub issue_id: String,
    pub title: String,
    pub error_category: Option<String>,
    pub model_used: Option<String>,
    pub iterations: u32,
    pub success: bool,
    /// 0.0-1.0 composite quality score (e.g. verifier health_score).
    pub score: f64,
    pub diff_summary: Option<String>,
    /// Analysis from the Analyzer agent.
    pub analysis: Option<String>,
    pub timestamp: DateTime<Utc>,
    /// Embedding of the issue title+description for similarity search.
    #[serde(default)]
    pub embedding: Vec<f32>,
    /// UCB1 visit counter — how many times this node was sampled as context.
    #[serde(default)]
    pub visit_count: u32,
}

/// The experiment history database.
pub struct ExperimentDb {
    nodes: Vec<ExperimentNode>,
    storage_path: PathBuf,
}

impl ExperimentDb {
    /// Load from disk or create empty.
    pub fn load_or_create(storage_dir: &Path) -> Result<Self> {
        let storage_path = storage_dir.join("experiment_history.jsonl");
        let nodes: Vec<ExperimentNode> = jsonl::load_all(&storage_path);
        if !storage_path.exists() {
            std::fs::create_dir_all(storage_dir)?;
        }
        Ok(Self {
            nodes,
            storage_path,
        })
    }

    /// Record a new experiment outcome.
    pub fn record(&mut self, node: ExperimentNode) -> Result<()> {
        jsonl::append(&self.storage_path, &node);
        self.nodes.push(node);
        Ok(())
    }

    /// Find top-k most similar past experiments by embedding cosine similarity.
    pub fn find_similar(&self, query_embedding: &[f32], top_k: usize) -> Vec<&ExperimentNode> {
        let mut scored: Vec<_> = self
            .nodes
            .iter()
            .filter(|n| !n.embedding.is_empty())
            .map(|n| {
                let sim = cosine_similarity(query_embedding, &n.embedding);
                (n, sim)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(n, _)| n).collect()
    }

    /// UCB1-based sampling of `n` experiment nodes.
    ///
    /// Balances exploitation (high-scoring) vs exploration (under-sampled).
    /// Increments `visit_count` for each sampled node.
    pub fn sample_ucb1(&mut self, n: usize, exploration_c: f64) -> Vec<usize> {
        if self.nodes.is_empty() {
            return vec![];
        }

        let total_visits: u32 = self.nodes.iter().map(|nd| nd.visit_count).sum();
        let max_score = self
            .nodes
            .iter()
            .map(|nd| nd.score)
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.001); // avoid division by zero

        let mut scored: Vec<(usize, f64)> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(idx, node)| {
                let normalized_score = node.score / max_score;
                let ucb1 = if node.visit_count == 0 {
                    f64::INFINITY // prioritize unvisited
                } else {
                    let total = (total_visits.max(1) as f64).ln();
                    normalized_score + exploration_c * (total / node.visit_count as f64).sqrt()
                };
                (idx, ucb1)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);

        let indices: Vec<usize> = scored.iter().map(|(idx, _)| *idx).collect();

        // Increment visit counts for sampled nodes.
        for &idx in &indices {
            self.nodes[idx].visit_count += 1;
        }

        indices
    }

    /// Sample and return references using UCB1.
    pub fn sample_ucb1_refs(&mut self, n: usize, exploration_c: f64) -> Vec<&ExperimentNode> {
        let indices = self.sample_ucb1(n, exploration_c);
        indices.into_iter().map(|idx| &self.nodes[idx]).collect()
    }

    /// Get success rate for a given error category.
    pub fn category_success_rate(&self, category: &str) -> Option<f64> {
        let matching: Vec<_> = self
            .nodes
            .iter()
            .filter(|n| n.error_category.as_deref() == Some(category))
            .collect();
        if matching.is_empty() {
            return None;
        }
        let successes = matching.iter().filter(|n| n.success).count();
        Some(successes as f64 / matching.len() as f64)
    }

    /// Format experiment nodes as context for a worker prompt.
    pub fn format_context(nodes: &[&ExperimentNode]) -> String {
        if nodes.is_empty() {
            return String::new();
        }
        let mut lines = vec!["## Similar Past Resolutions".to_string()];
        for (i, n) in nodes.iter().enumerate() {
            let status = if n.success { "PASS" } else { "FAIL" };
            let category = n.error_category.as_deref().unwrap_or("unknown");
            let model = n.model_used.as_deref().unwrap_or("?");
            let analysis_preview = n
                .analysis
                .as_deref()
                .map(|a| {
                    if a.len() > 150 {
                        format!("{}...", &a[..a.floor_char_boundary(150)])
                    } else {
                        a.to_string()
                    }
                })
                .unwrap_or_default();
            lines.push(format!(
                "[{}] {} {} ({}, {}iter, {}) {}",
                i + 1,
                status,
                n.title,
                category,
                n.iterations,
                model,
                analysis_preview,
            ));
        }
        lines.join("\n")
    }

    /// Total number of recorded experiments.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether the database is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_node(id: &str, success: bool, score: f64) -> ExperimentNode {
        ExperimentNode {
            issue_id: id.to_string(),
            title: format!("Fix {}", id),
            error_category: Some("type_mismatch".to_string()),
            model_used: Some("Qwen3.5-27B".to_string()),
            iterations: 2,
            success,
            score,
            diff_summary: Some("Changed foo.rs".to_string()),
            analysis: Some("Applied type conversion".to_string()),
            timestamp: Utc::now(),
            embedding: vec![],
            visit_count: 0,
        }
    }

    fn make_node_with_embedding(id: &str, embedding: Vec<f32>) -> ExperimentNode {
        let mut node = make_node(id, true, 0.8);
        node.embedding = embedding;
        node
    }

    #[test]
    fn test_load_save_roundtrip() {
        let dir = TempDir::new().unwrap();
        let mut db = ExperimentDb::load_or_create(dir.path()).unwrap();
        assert!(db.is_empty());

        db.record(make_node("issue-1", true, 0.9)).unwrap();
        db.record(make_node("issue-2", false, 0.3)).unwrap();
        assert_eq!(db.len(), 2);

        // Reload from disk.
        let db2 = ExperimentDb::load_or_create(dir.path()).unwrap();
        assert_eq!(db2.len(), 2);
        assert_eq!(db2.nodes[0].issue_id, "issue-1");
        assert!(db2.nodes[0].success);
        assert_eq!(db2.nodes[1].issue_id, "issue-2");
        assert!(!db2.nodes[1].success);
    }

    #[test]
    fn test_find_similar() {
        let dir = TempDir::new().unwrap();
        let mut db = ExperimentDb::load_or_create(dir.path()).unwrap();

        // Three nodes with known embeddings.
        db.record(make_node_with_embedding("a", vec![1.0, 0.0, 0.0]))
            .unwrap();
        db.record(make_node_with_embedding("b", vec![0.9, 0.1, 0.0]))
            .unwrap();
        db.record(make_node_with_embedding("c", vec![0.0, 0.0, 1.0]))
            .unwrap();

        // Query close to "a" and "b".
        let results = db.find_similar(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].issue_id, "a"); // exact match first
        assert_eq!(results[1].issue_id, "b"); // close second
    }

    #[test]
    fn test_find_similar_no_embeddings() {
        let dir = TempDir::new().unwrap();
        let mut db = ExperimentDb::load_or_create(dir.path()).unwrap();
        db.record(make_node("no-emb", true, 0.5)).unwrap();

        let results = db.find_similar(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_sample_ucb1_prioritizes_unvisited() {
        let dir = TempDir::new().unwrap();
        let mut db = ExperimentDb::load_or_create(dir.path()).unwrap();

        let mut n1 = make_node("visited", true, 1.0);
        n1.visit_count = 10;
        db.record(n1).unwrap();

        let n2 = make_node("unvisited", true, 0.5);
        db.record(n2).unwrap();

        // Unvisited should be prioritized (UCB1 = infinity for visit_count=0).
        let indices = db.sample_ucb1(1, 1.4);
        assert_eq!(indices.len(), 1);
        assert_eq!(db.nodes[indices[0]].issue_id, "unvisited");
        // Visit count should be incremented.
        assert_eq!(db.nodes[indices[0]].visit_count, 1);
    }

    #[test]
    fn test_sample_ucb1_empty() {
        let dir = TempDir::new().unwrap();
        let mut db = ExperimentDb::load_or_create(dir.path()).unwrap();
        let indices = db.sample_ucb1(5, 1.4);
        assert!(indices.is_empty());
    }

    #[test]
    fn test_sample_ucb1_refs() {
        let dir = TempDir::new().unwrap();
        let mut db = ExperimentDb::load_or_create(dir.path()).unwrap();
        db.record(make_node("a", true, 0.9)).unwrap();
        db.record(make_node("b", false, 0.2)).unwrap();

        let refs = db.sample_ucb1_refs(2, 1.4);
        assert_eq!(refs.len(), 2);
    }

    #[test]
    fn test_category_success_rate() {
        let dir = TempDir::new().unwrap();
        let mut db = ExperimentDb::load_or_create(dir.path()).unwrap();

        db.record(make_node("a", true, 0.9)).unwrap(); // type_mismatch, success
        db.record(make_node("b", false, 0.2)).unwrap(); // type_mismatch, failure
        db.record(make_node("c", true, 0.8)).unwrap(); // type_mismatch, success

        let rate = db.category_success_rate("type_mismatch").unwrap();
        assert!((rate - 2.0 / 3.0).abs() < 1e-9);

        assert!(db.category_success_rate("borrow_checker").is_none());
    }

    #[test]
    fn test_format_context() {
        let n1 = make_node("x", true, 1.0);
        let n2 = make_node("y", false, 0.1);
        let refs = vec![&n1, &n2];

        let ctx = ExperimentDb::format_context(&refs);
        assert!(ctx.contains("## Similar Past Resolutions"));
        assert!(ctx.contains("PASS"));
        assert!(ctx.contains("FAIL"));
        assert!(ctx.contains("Fix x"));
        assert!(ctx.contains("Fix y"));
    }

    #[test]
    fn test_format_context_empty() {
        let ctx = ExperimentDb::format_context(&[]);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let sim = cosine_similarity(&[1.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let sim = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_mismatched_length() {
        let sim = cosine_similarity(&[1.0, 0.0], &[1.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let sim = cosine_similarity(&[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_format_context_long_analysis() {
        let mut node = make_node("long", true, 0.9);
        node.analysis = Some("x".repeat(200));
        let refs = vec![&node];
        let ctx = ExperimentDb::format_context(&refs);
        assert!(ctx.contains("..."));
    }
}
