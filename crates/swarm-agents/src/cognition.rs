//! Cognition Base — persistent memory of reusable insights from resolutions.
//!
//! Stores structured insight items (patterns, principles, guidance) extracted
//! by the Analyzer agent after each successful resolution. Items are keyed by
//! a unique ID and can be queried by domain (error category).
//!
//! This is a lightweight in-process store. A future version may back it with
//! RocksDB or the wiki module for cross-session persistence.

use tracing::debug;

/// Source of a cognition item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CognitionSource {
    /// Automatically extracted from a successful resolution.
    Experiment,
    /// Manually added by a human operator.
    Manual,
}

/// A single insight stored in the Cognition Base.
#[derive(Debug, Clone)]
pub struct CognitionItem {
    pub id: String,
    pub content: String,
    pub source: CognitionSource,
    /// Optional domain tag (e.g., error category like "borrow_checker").
    pub domain: Option<String>,
    /// Reserved for future semantic search.
    pub embedding: Vec<f32>,
}

/// In-memory Cognition Base.
#[derive(Debug, Default)]
pub struct CognitionBase {
    items: Vec<CognitionItem>,
}

impl CognitionBase {
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Add an insight item to the base.
    pub fn add(&mut self, item: CognitionItem) {
        debug!(id = %item.id, domain = ?item.domain, "Cognition Base: adding item");
        self.items.push(item);
    }

    /// Return all items, most recent first.
    pub fn items(&self) -> &[CognitionItem] {
        &self.items
    }

    /// Return items matching a domain tag.
    pub fn by_domain(&self, domain: &str) -> Vec<&CognitionItem> {
        self.items
            .iter()
            .filter(|i| i.domain.as_deref() == Some(domain))
            .collect()
    }

    /// Number of stored items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the base is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_retrieve() {
        let mut base = CognitionBase::new();
        assert!(base.is_empty());

        base.add(CognitionItem {
            id: "exp-001".into(),
            content: "Fixed borrow checker issue by cloning".into(),
            source: CognitionSource::Experiment,
            domain: Some("borrow_checker".into()),
            embedding: vec![],
        });

        assert_eq!(base.len(), 1);
        assert_eq!(base.items()[0].id, "exp-001");
    }

    #[test]
    fn test_by_domain() {
        let mut base = CognitionBase::new();
        base.add(CognitionItem {
            id: "exp-001".into(),
            content: "borrow fix".into(),
            source: CognitionSource::Experiment,
            domain: Some("borrow_checker".into()),
            embedding: vec![],
        });
        base.add(CognitionItem {
            id: "exp-002".into(),
            content: "type fix".into(),
            source: CognitionSource::Experiment,
            domain: Some("type_mismatch".into()),
            embedding: vec![],
        });
        base.add(CognitionItem {
            id: "exp-003".into(),
            content: "another borrow fix".into(),
            source: CognitionSource::Experiment,
            domain: Some("borrow_checker".into()),
            embedding: vec![],
        });

        let borrow_items = base.by_domain("borrow_checker");
        assert_eq!(borrow_items.len(), 2);

        let type_items = base.by_domain("type_mismatch");
        assert_eq!(type_items.len(), 1);

        let unknown = base.by_domain("unknown");
        assert!(unknown.is_empty());
    }
}
