//! Cognition Base — persistent knowledge retrieval for worker task prompts.
//!
//! Stores structured knowledge items (error patterns, architecture decisions,
//! resolution playbooks) and retrieves relevant ones before each worker dispatch.
//! Uses keyword overlap for retrieval until an embedding model is wired up.
//!
//! Lifecycle:
//! 1. `CognitionBase::load_or_create` loads persisted items from disk
//! 2. `seed_from_wiki` indexes `docs/wiki/*.md` pages as cognition items
//! 3. `retrieve_by_keywords` finds relevant items for a query
//! 4. `RetrievalResult::format_context` renders them as structured prompt context

use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// A single knowledge item in the cognition base.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitionItem {
    /// Unique identifier (e.g., wiki page slug or hash).
    pub id: String,
    /// Category tag: `"error_pattern"`, `"architecture"`, `"playbook"`, etc.
    pub category: String,
    /// Human-readable title.
    pub title: String,
    /// Full text content used for retrieval matching.
    pub content: String,
    /// Source provenance (e.g., `"wiki:error-patterns"`, `"mutation_archive"`).
    pub source: String,
}

/// A retrieval result pairing an item with a relevance score.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    pub item: CognitionItem,
    /// Relevance score in [0.0, 1.0] — keyword overlap fraction.
    pub score: f32,
}

impl RetrievalResult {
    /// Format a list of retrieval results as structured context for injection
    /// into a worker task prompt.
    ///
    /// Returns an empty string when the results list is empty.
    pub fn format_context(results: &[RetrievalResult]) -> String {
        if results.is_empty() {
            return String::new();
        }

        let mut out = String::from("## Cognition Context (retrieved knowledge)\n\n");
        out.push_str(
            "_Relevant knowledge items from the project cognition base. \
             Use these patterns and context to inform your approach._\n\n",
        );
        for (i, r) in results.iter().enumerate() {
            out.push_str(&format!(
                "### {}. {} [{}] (relevance: {:.0}%)\n",
                i + 1,
                r.item.title,
                r.item.category,
                r.score * 100.0,
            ));
            // Truncate long content to avoid blowing up the prompt.
            let content = if r.item.content.len() > 600 {
                let cut = crate::str_util::safe_truncate(&r.item.content, 600);
                format!("{cut}...")
            } else {
                r.item.content.clone()
            };
            out.push_str(&content);
            out.push_str("\n\n");
        }
        out
    }
}

/// Persistent knowledge store backed by a JSONL file on disk.
pub struct CognitionBase {
    /// Directory where `items.jsonl` lives.
    path: PathBuf,
    /// In-memory item store.
    items: Vec<CognitionItem>,
}

impl CognitionBase {
    /// Load an existing cognition base from disk, or create an empty one.
    pub fn load_or_create(dir: &Path) -> Result<Self> {
        if !dir.exists() {
            fs::create_dir_all(dir)
                .with_context(|| format!("creating cognition dir {}", dir.display()))?;
        }
        let items_path = dir.join("items.jsonl");
        let items = if items_path.exists() {
            let content = fs::read_to_string(&items_path)
                .with_context(|| format!("reading {}", items_path.display()))?;
            content
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|line| match serde_json::from_str::<CognitionItem>(line) {
                    Ok(item) => Some(item),
                    Err(e) => {
                        warn!(error = %e, "Skipping malformed cognition item");
                        None
                    }
                })
                .collect()
        } else {
            Vec::new()
        };

        debug!(count = items.len(), dir = %dir.display(), "Loaded cognition base");
        Ok(Self {
            path: dir.to_path_buf(),
            items,
        })
    }

    /// Access the in-memory items.
    pub fn items(&self) -> &[CognitionItem] {
        &self.items
    }

    /// Add an item if no item with the same id already exists.
    pub fn add_if_new(&mut self, item: CognitionItem) {
        if self.items.iter().any(|existing| existing.id == item.id) {
            return;
        }
        self.items.push(item);
    }

    /// Persist the current items to disk as JSONL.
    pub fn save(&self) -> Result<()> {
        let items_path = self.path.join("items.jsonl");
        let content: String = self
            .items
            .iter()
            .filter_map(|item| serde_json::to_string(item).ok())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&items_path, content)
            .with_context(|| format!("writing {}", items_path.display()))?;
        Ok(())
    }
}

/// Retrieve cognition items by keyword overlap (fallback for no embeddings).
///
/// Tokenizes the query and each item's content into lowercase words, computes
/// the fraction of query words that appear in the item, and returns the top-k
/// items above a minimum threshold.
pub fn retrieve_by_keywords(
    base: &CognitionBase,
    query: &str,
    top_k: usize,
) -> Vec<RetrievalResult> {
    let query_lower = query.to_lowercase();
    let query_words: HashSet<&str> = query_lower.split_whitespace().collect();
    if query_words.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<_> = base
        .items()
        .iter()
        .map(|item| {
            let item_lower = item.content.to_lowercase();
            let item_words: HashSet<&str> = item_lower.split_whitespace().collect();
            let overlap = query_words
                .iter()
                .filter(|w| item_words.contains(*w))
                .count();
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

/// Seed the cognition base from `docs/wiki/*.md` pages.
///
/// Each markdown file becomes one cognition item. The file's first `# Heading`
/// line is used as the title; the slug (filename without `.md`) becomes the id.
/// Existing items with the same id are skipped (idempotent).
pub fn seed_from_wiki(base: &mut CognitionBase, repo_root: &Path) -> Result<()> {
    let wiki_dir = repo_root.join("docs/wiki");
    if !wiki_dir.exists() {
        debug!(dir = %wiki_dir.display(), "No wiki directory — skipping seed");
        return Ok(());
    }

    let entries = fs::read_dir(&wiki_dir)
        .with_context(|| format!("reading wiki dir {}", wiki_dir.display()))?;

    let mut added = 0usize;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "md") {
            let slug = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            // Skip index/log files — they're metadata, not knowledge.
            if slug == "index" || slug == "log" {
                continue;
            }

            let content = match fs::read_to_string(&path) {
                Ok(c) => c,
                Err(e) => {
                    warn!(path = %path.display(), error = %e, "Failed to read wiki page");
                    continue;
                }
            };

            // Extract title from first `# Heading` line.
            let title = content
                .lines()
                .find(|l| l.starts_with("# "))
                .map(|l| l.trim_start_matches("# ").trim().to_string())
                .unwrap_or_else(|| slug.replace('-', " "));

            // Derive category from filename pattern.
            let category = if slug.contains("error") || slug.contains("pattern") {
                "error_pattern"
            } else if slug.contains("architecture") || slug.contains("design") {
                "architecture"
            } else if slug.contains("performance") || slug.contains("model") {
                "performance"
            } else if slug.contains("harness") || slug.contains("config") {
                "configuration"
            } else {
                "knowledge"
            };

            base.add_if_new(CognitionItem {
                id: format!("wiki:{slug}"),
                category: category.to_string(),
                title,
                content,
                source: format!("wiki:{slug}"),
            });
            added += 1;
        }
    }

    if added > 0 {
        base.save()?;
        info!(
            added,
            total = base.items().len(),
            "Seeded cognition base from wiki"
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_item(id: &str, content: &str) -> CognitionItem {
        CognitionItem {
            id: id.to_string(),
            category: "test".to_string(),
            title: id.to_string(),
            content: content.to_string(),
            source: "test".to_string(),
        }
    }

    #[test]
    fn test_retrieve_by_keywords_basic() {
        let dir = tempfile::tempdir().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        base.add_if_new(make_item(
            "borrow",
            "borrow checker errors require clone or Arc to fix ownership issues",
        ));
        base.add_if_new(make_item(
            "imports",
            "import resolution errors need use statements or Cargo.toml deps",
        ));
        base.add_if_new(make_item(
            "unrelated",
            "this item has completely unrelated content about cooking recipes",
        ));

        let results = retrieve_by_keywords(&base, "borrow checker ownership", 5);
        assert!(!results.is_empty());
        assert_eq!(results[0].item.id, "borrow");
        assert!(results[0].score > 0.3);
    }

    #[test]
    fn test_retrieve_by_keywords_empty_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        base.add_if_new(make_item("item", "some content"));

        let results = retrieve_by_keywords(&base, "", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_retrieve_by_keywords_no_match() {
        let dir = tempfile::tempdir().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        base.add_if_new(make_item("item", "rust borrow checker lifetime"));

        let results = retrieve_by_keywords(&base, "javascript webpack bundler react", 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_format_context_empty() {
        assert_eq!(RetrievalResult::format_context(&[]), "");
    }

    #[test]
    fn test_format_context_with_results() {
        let results = vec![RetrievalResult {
            item: make_item("borrow", "fix with clone"),
            score: 0.75,
        }];
        let ctx = RetrievalResult::format_context(&results);
        assert!(ctx.contains("Cognition Context"));
        assert!(ctx.contains("borrow"));
        assert!(ctx.contains("75%"));
    }

    #[test]
    fn test_add_if_new_deduplicates() {
        let dir = tempfile::tempdir().unwrap();
        let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
        base.add_if_new(make_item("a", "first"));
        base.add_if_new(make_item("a", "second"));
        assert_eq!(base.items().len(), 1);
        assert_eq!(base.items()[0].content, "first");
    }

    #[test]
    fn test_save_and_reload() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut base = CognitionBase::load_or_create(dir.path()).unwrap();
            base.add_if_new(make_item("a", "content a"));
            base.add_if_new(make_item("b", "content b"));
            base.save().unwrap();
        }
        let base = CognitionBase::load_or_create(dir.path()).unwrap();
        assert_eq!(base.items().len(), 2);
    }
}
