//! Cognition Base -- semantic knowledge retrieval for the LDEA loop.
//!
//! Stores domain knowledge items with embeddings and enables semantic
//! retrieval of relevant priors before each swarm iteration.
//!
//! Source: ASI-Evolve (arxiv:2603.29640) -- Cognition Base with FAISS.
//! Our implementation uses brute-force cosine similarity (<1K items).

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single knowledge item in the Cognition Base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionItem {
    pub id: String,
    pub content: String,
    pub source: CognitionSource,
    pub domain: Option<String>,
    /// Pre-computed embedding vector (populated lazily or on insert).
    #[serde(default)]
    pub embedding: Vec<f32>,
}

/// Where a cognition item originated.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CognitionSource {
    Wiki,
    Memory,
    Notebook,
    Experiment,
    Literature,
}

/// Retrieval result with relevance score.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub item: CognitionItem,
    pub score: f32,
}

impl RetrievalResult {
    /// Format retrieval results as a context block for prompts.
    pub fn format_context(results: &[RetrievalResult]) -> String {
        if results.is_empty() {
            return String::new();
        }
        let mut lines = vec!["## Relevant Prior Knowledge".to_string()];
        for (i, r) in results.iter().enumerate() {
            let source = match r.item.source {
                CognitionSource::Wiki => "wiki",
                CognitionSource::Experiment => "experiment",
                CognitionSource::Memory => "memory",
                CognitionSource::Notebook => "notebook",
                CognitionSource::Literature => "literature",
            };
            let preview = if r.item.content.len() > 200 {
                format!("{}...", &r.item.content[..200])
            } else {
                r.item.content.clone()
            };
            lines.push(format!(
                "[{}] ({}, score={:.2}) {}",
                i + 1,
                source,
                r.score,
                preview
            ));
        }
        lines.join("\n")
    }
}

/// The Cognition Base -- stores and retrieves knowledge items.
pub struct CognitionBase {
    items: Vec<CognitionItem>,
    storage_path: PathBuf,
}

impl CognitionBase {
    /// Load from disk or create empty.
    pub fn load_or_create(storage_dir: &Path) -> Result<Self> {
        let storage_path = storage_dir.join("cognition_items.json");
        let items = if storage_path.exists() {
            let data = std::fs::read_to_string(&storage_path)?;
            serde_json::from_str(&data)?
        } else {
            std::fs::create_dir_all(storage_dir)?;
            Vec::new()
        };
        Ok(Self {
            items,
            storage_path,
        })
    }

    /// Add an item (with optional pre-computed embedding).
    pub fn add(&mut self, item: CognitionItem) -> Result<()> {
        self.items.retain(|i| i.id != item.id);
        self.items.push(item);
        self.save()?;
        Ok(())
    }

    /// Retrieve top-k items most similar to the query embedding.
    pub fn retrieve(
        &self,
        query_embedding: &[f32],
        top_k: usize,
        threshold: f32,
    ) -> Vec<RetrievalResult> {
        let mut scored: Vec<_> = self
            .items
            .iter()
            .filter(|item| !item.embedding.is_empty())
            .map(|item| {
                let score = cosine_similarity(query_embedding, &item.embedding);
                (item.clone(), score)
            })
            .filter(|(_, score)| *score >= threshold)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(item, score)| RetrievalResult { item, score })
            .collect()
    }

    /// Retrieve by text query (requires an embedding function).
    /// Convenience wrapper -- the caller provides the embedding.
    pub fn retrieve_by_text(&self, query_embedding: &[f32], top_k: usize) -> Vec<RetrievalResult> {
        self.retrieve(query_embedding, top_k, 0.3)
    }

    /// Get all items (for iteration/export).
    pub fn items(&self) -> &[CognitionItem] {
        &self.items
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    fn save(&self) -> Result<()> {
        let data = serde_json::to_string_pretty(&self.items)?;
        std::fs::write(&self.storage_path, data)?;
        Ok(())
    }
}

/// Seed the Cognition Base from docs/wiki/ markdown files.
pub fn seed_from_wiki(base: &mut CognitionBase, repo_root: &Path) -> Result<usize> {
    let wiki_dir = repo_root.join("docs/wiki");
    if !wiki_dir.exists() {
        return Ok(0);
    }
    let mut count = 0;
    for entry in std::fs::read_dir(&wiki_dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "md") {
            let content = std::fs::read_to_string(&path)?;
            let slug = path.file_stem().unwrap_or_default().to_string_lossy();
            let item = CognitionItem {
                id: format!("wiki-{slug}"),
                content,
                source: CognitionSource::Wiki,
                domain: Some("swarm-architecture".into()),
                embedding: vec![],
            };
            base.add(item)?;
            count += 1;
        }
    }
    Ok(count)
}

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_item(id: &str, content: &str, embedding: Vec<f32>) -> CognitionItem {
        CognitionItem {
            id: id.to_string(),
            content: content.to_string(),
            source: CognitionSource::Wiki,
            domain: None,
            embedding,
        }
    }

    #[test]
    fn test_load_or_create_empty() {
        let dir = TempDir::new().unwrap();
        let base = CognitionBase::load_or_create(dir.path()).unwrap();
        assert!(base.is_empty());
        assert_eq!(base.len(), 0);
    }

    #[test]
    fn test_add_and_persist() {
        let dir = TempDir::new().unwrap();
        {
            let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
            base.add(make_item("a", "hello", vec![1.0, 0.0])).unwrap();
            base.add(make_item("b", "world", vec![0.0, 1.0])).unwrap();
            assert_eq!(base.len(), 2);
        }
        // Reload from disk
        let base = CognitionBase::load_or_create(dir.path()).unwrap();
        assert_eq!(base.len(), 2);
        assert_eq!(base.items()[0].id, "a");
        assert_eq!(base.items()[1].id, "b");
    }

    #[test]
    fn test_dedup_by_id() {
        let dir = TempDir::new().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        base.add(make_item("x", "first", vec![1.0])).unwrap();
        base.add(make_item("x", "second", vec![2.0])).unwrap();
        assert_eq!(base.len(), 1);
        assert_eq!(base.items()[0].content, "second");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!((cosine_similarity(&a, &b)).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_empty_vectors() {
        assert_eq!(cosine_similarity(&[], &[]), 0.0);
    }

    #[test]
    fn test_cosine_length_mismatch() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_cosine_zero_norm() {
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
    }

    #[test]
    fn test_retrieve_top_k() {
        let dir = TempDir::new().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        // Item aligned with [1,0]
        base.add(make_item("aligned", "aligned doc", vec![1.0, 0.0]))
            .unwrap();
        // Item orthogonal
        base.add(make_item("ortho", "ortho doc", vec![0.0, 1.0]))
            .unwrap();
        // Item partially aligned
        base.add(make_item("partial", "partial doc", vec![0.7071, 0.7071]))
            .unwrap();

        let query = vec![1.0, 0.0];
        let results = base.retrieve(&query, 2, 0.0);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].item.id, "aligned");
        assert!((results[0].score - 1.0).abs() < 1e-4);
        assert_eq!(results[1].item.id, "partial");
    }

    #[test]
    fn test_retrieve_threshold_filters() {
        let dir = TempDir::new().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        base.add(make_item("low", "low relevance", vec![0.1, 0.995]))
            .unwrap();
        base.add(make_item("high", "high relevance", vec![0.99, 0.14]))
            .unwrap();

        let query = vec![1.0, 0.0];
        let results = base.retrieve(&query, 10, 0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item.id, "high");
    }

    #[test]
    fn test_retrieve_skips_no_embedding() {
        let dir = TempDir::new().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        base.add(make_item("no_emb", "no embedding", vec![]))
            .unwrap();
        base.add(make_item("has_emb", "has embedding", vec![1.0, 0.0]))
            .unwrap();

        let results = base.retrieve(&[1.0, 0.0], 10, 0.0);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item.id, "has_emb");
    }

    #[test]
    fn test_retrieve_by_text_default_threshold() {
        let dir = TempDir::new().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        // Score will be ~0.1 (below 0.3 threshold)
        base.add(make_item("low", "low", vec![0.1, 0.995])).unwrap();
        // Score will be ~0.99 (above 0.3 threshold)
        base.add(make_item("high", "high", vec![0.99, 0.14]))
            .unwrap();

        let results = base.retrieve_by_text(&[1.0, 0.0], 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].item.id, "high");
    }

    #[test]
    fn test_seed_from_wiki() {
        let dir = TempDir::new().unwrap();
        let repo = TempDir::new().unwrap();
        let wiki_dir = repo.path().join("docs/wiki");
        std::fs::create_dir_all(&wiki_dir).unwrap();
        std::fs::write(wiki_dir.join("architecture.md"), "# Architecture\nStuff").unwrap();
        std::fs::write(wiki_dir.join("patterns.md"), "# Patterns\nMore stuff").unwrap();
        std::fs::write(wiki_dir.join("readme.txt"), "not markdown").unwrap();

        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        let count = seed_from_wiki(&mut base, repo.path()).unwrap();
        assert_eq!(count, 2);
        assert_eq!(base.len(), 2);

        let ids: Vec<&str> = base.items().iter().map(|i| i.id.as_str()).collect();
        assert!(ids.contains(&"wiki-architecture"));
        assert!(ids.contains(&"wiki-patterns"));
    }

    #[test]
    fn test_seed_from_wiki_no_dir() {
        let dir = TempDir::new().unwrap();
        let repo = TempDir::new().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        let count = seed_from_wiki(&mut base, repo.path()).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_format_context_empty() {
        assert_eq!(RetrievalResult::format_context(&[]), "");
    }

    #[test]
    fn test_format_context_with_results() {
        let results = vec![
            RetrievalResult {
                item: CognitionItem {
                    id: "a".into(),
                    content: "Short content".into(),
                    source: CognitionSource::Wiki,
                    domain: None,
                    embedding: vec![],
                },
                score: 0.95,
            },
            RetrievalResult {
                item: CognitionItem {
                    id: "b".into(),
                    content: "x".repeat(250),
                    source: CognitionSource::Experiment,
                    domain: None,
                    embedding: vec![],
                },
                score: 0.42,
            },
        ];
        let ctx = RetrievalResult::format_context(&results);
        assert!(ctx.starts_with("## Relevant Prior Knowledge"));
        assert!(ctx.contains("[1] (wiki, score=0.95) Short content"));
        assert!(ctx.contains("[2] (experiment, score=0.42)"));
        assert!(ctx.contains("..."));
    }

    #[test]
    fn test_format_context_all_sources() {
        let sources = [
            (CognitionSource::Wiki, "wiki"),
            (CognitionSource::Memory, "memory"),
            (CognitionSource::Notebook, "notebook"),
            (CognitionSource::Experiment, "experiment"),
            (CognitionSource::Literature, "literature"),
        ];
        for (source, label) in sources {
            let results = vec![RetrievalResult {
                item: CognitionItem {
                    id: "test".into(),
                    content: "test".into(),
                    source,
                    domain: None,
                    embedding: vec![],
                },
                score: 0.5,
            }];
            let ctx = RetrievalResult::format_context(&results);
            assert!(
                ctx.contains(label),
                "Expected label '{label}' in context: {ctx}"
            );
        }
    }
}

/// Retrieve cognition items by keyword overlap (fallback when embeddings unavailable).
pub fn retrieve_by_keywords(
    base: &CognitionBase,
    query: &str,
    top_k: usize,
) -> Vec<RetrievalResult> {
    let query_lower = query.to_lowercase();
    let query_words: std::collections::HashSet<&str> = query_lower.split_whitespace().collect();
    if query_words.is_empty() {
        return vec![];
    }
    let mut scored: Vec<_> = base
        .items()
        .iter()
        .map(|item| {
            let content_lower = item.content.to_lowercase();
            let item_words: std::collections::HashSet<&str> =
                content_lower.split_whitespace().collect();
            let overlap = query_words.intersection(&item_words).count();
            let score = overlap as f32 / query_words.len().max(1) as f32;
            (item.clone(), score)
        })
        .filter(|(_, score)| *score > 0.1)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);
    scored
        .into_iter()
        .map(|(item, score)| RetrievalResult { item, score })
        .collect()
}
