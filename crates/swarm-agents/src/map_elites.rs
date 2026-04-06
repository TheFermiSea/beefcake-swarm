//! MAP-Elites quality-diversity archive for diverse fix exploration.
//!
//! Bins past experiments by behavioral features (complexity x strategy),
//! preventing the swarm from always trying the same approach. Each cell
//! in the 4x4 feature grid keeps only the highest-scoring experiment,
//! and `sample_diverse` selects from different bins to maximise coverage.
//!
//! Standalone module — experiment_db integration in a follow-up.
//!
//! Source: ASI-Evolve database/algorithms/island.py

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of complexity bins: [0-5, 5-20, 20-50, 50+].
pub const COMPLEXITY_BINS: usize = 4;

/// Number of strategy bins: single_line, multi_line, multi_file, refactor.
pub const STRATEGY_BINS: usize = 4;

/// Total cells in the feature grid.
pub const GRID_SIZE: usize = COMPLEXITY_BINS * STRATEGY_BINS;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Edit strategy category — the *kind* of change an experiment applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EditStrategy {
    /// A single line was changed.
    SingleLine,
    /// Multiple lines within the same file were changed.
    MultiLine,
    /// Changes spanned multiple files.
    MultiFile,
    /// Large-scale refactor (many files or structural change).
    Refactor,
}

impl EditStrategy {
    /// Map strategy to its bin index (0..3).
    pub fn bin(self) -> usize {
        match self {
            Self::SingleLine => 0,
            Self::MultiLine => 1,
            Self::MultiFile => 2,
            Self::Refactor => 3,
        }
    }
}

/// Lightweight experiment node that can be stored in the archive.
///
/// Intentionally decoupled from any particular database schema so the
/// archive stays standalone.  Integration code maps richer types into
/// this struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperimentNode {
    /// Unique experiment identifier.
    pub id: String,
    /// Human-readable description of the attempted fix.
    pub description: String,
    /// Fitness / quality score (higher is better).
    pub score: f64,
    /// Total lines changed (added + removed).
    pub lines_changed: usize,
    /// Number of files touched.
    pub files_changed: usize,
    /// Which edit strategy was used.
    pub strategy: EditStrategy,
}

/// Behavioral feature vector used to place an experiment in the grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BehavioralFeatures {
    /// Complexity bin (0..COMPLEXITY_BINS-1).
    pub complexity_bin: usize,
    /// Strategy bin (0..STRATEGY_BINS-1).
    pub strategy_bin: usize,
}

impl BehavioralFeatures {
    /// Create features from raw values, clamping to valid bin ranges.
    pub fn new(complexity_bin: usize, strategy_bin: usize) -> Self {
        Self {
            complexity_bin: complexity_bin.min(COMPLEXITY_BINS - 1),
            strategy_bin: strategy_bin.min(STRATEGY_BINS - 1),
        }
    }
}

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------

/// Computes [`BehavioralFeatures`] from experiment metadata.
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Bin `lines_changed` into complexity buckets.
    ///
    /// | Range  | Bin |
    /// |--------|-----|
    /// | 0–5    | 0   |
    /// | 6–20   | 1   |
    /// | 21–50  | 2   |
    /// | 51+    | 3   |
    pub fn complexity_bin(lines_changed: usize) -> usize {
        match lines_changed {
            0..=5 => 0,
            6..=20 => 1,
            21..=50 => 2,
            _ => 3,
        }
    }

    /// Determine the edit strategy from file/line counts.
    ///
    /// Heuristic:
    /// - 1 file, <=1 line changed  → SingleLine
    /// - 1 file, >1 line changed   → MultiLine
    /// - 2–4 files                  → MultiFile
    /// - 5+ files                   → Refactor
    pub fn infer_strategy(files_changed: usize, lines_changed: usize) -> EditStrategy {
        match files_changed {
            0..=1 if lines_changed <= 1 => EditStrategy::SingleLine,
            0..=1 => EditStrategy::MultiLine,
            2..=4 => EditStrategy::MultiFile,
            _ => EditStrategy::Refactor,
        }
    }

    /// Extract behavioral features from an [`ExperimentNode`].
    pub fn extract(node: &ExperimentNode) -> BehavioralFeatures {
        BehavioralFeatures::new(
            Self::complexity_bin(node.lines_changed),
            node.strategy.bin(),
        )
    }

    /// Extract features from raw metadata (useful when you don't have a node yet).
    pub fn extract_from_metadata(lines_changed: usize, files_changed: usize) -> BehavioralFeatures {
        let strategy = Self::infer_strategy(files_changed, lines_changed);
        BehavioralFeatures::new(Self::complexity_bin(lines_changed), strategy.bin())
    }
}

// ---------------------------------------------------------------------------
// Archive
// ---------------------------------------------------------------------------

/// MAP-Elites quality-diversity archive.
///
/// A 2-D grid (complexity × strategy) where each cell holds the best-scoring
/// experiment that falls into that behavioural niche.  The archive ensures the
/// swarm explores *diverse* fix strategies rather than converging on a single
/// approach.
#[derive(Debug, Default)]
pub struct QualityDiversityArchive {
    grid: HashMap<(usize, usize), ExperimentNode>,
}

impl QualityDiversityArchive {
    /// Create an empty archive.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert an experiment into the archive.
    ///
    /// The node is placed in the cell identified by `features`. If the cell
    /// is empty the node is inserted unconditionally. If occupied, the node
    /// replaces the incumbent only when it has a strictly higher score.
    ///
    /// Returns `true` if the node was inserted (new cell or better score).
    pub fn insert(&mut self, node: ExperimentNode, features: BehavioralFeatures) -> bool {
        let key = (features.complexity_bin, features.strategy_bin);
        match self.grid.get(&key) {
            Some(existing) if existing.score >= node.score => false,
            _ => {
                self.grid.insert(key, node);
                true
            }
        }
    }

    /// Sample up to `n` experiments from *different* bins for maximum diversity.
    ///
    /// Returns experiments sorted by score (descending) so the caller gets the
    /// highest-quality diverse set.  If the archive contains fewer than `n`
    /// occupied bins, all occupied bins are returned.
    pub fn sample_diverse(&self, n: usize) -> Vec<&ExperimentNode> {
        let mut entries: Vec<&ExperimentNode> = self.grid.values().collect();
        // Sort by score descending — take the best from each unique bin.
        entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        entries.truncate(n);
        entries
    }

    /// Fraction of grid cells that are occupied (0.0 .. 1.0).
    pub fn coverage(&self) -> f64 {
        self.grid.len() as f64 / GRID_SIZE as f64
    }

    /// Number of occupied cells.
    pub fn len(&self) -> usize {
        self.grid.len()
    }

    /// Whether the archive is empty.
    pub fn is_empty(&self) -> bool {
        self.grid.is_empty()
    }

    /// Iterate over all (features, node) pairs in the archive.
    pub fn iter(&self) -> impl Iterator<Item = (BehavioralFeatures, &ExperimentNode)> {
        self.grid.iter().map(|(&(c, s), node)| {
            (
                BehavioralFeatures {
                    complexity_bin: c,
                    strategy_bin: s,
                },
                node,
            )
        })
    }

    /// Get the experiment occupying a specific cell, if any.
    pub fn get(&self, features: &BehavioralFeatures) -> Option<&ExperimentNode> {
        self.grid
            .get(&(features.complexity_bin, features.strategy_bin))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id: &str, score: f64, lines: usize, files: usize) -> ExperimentNode {
        ExperimentNode {
            id: id.to_string(),
            description: format!("experiment {id}"),
            score,
            lines_changed: lines,
            files_changed: files,
            strategy: FeatureExtractor::infer_strategy(files, lines),
        }
    }

    // -- FeatureExtractor ---------------------------------------------------

    #[test]
    fn complexity_bins() {
        assert_eq!(FeatureExtractor::complexity_bin(0), 0);
        assert_eq!(FeatureExtractor::complexity_bin(3), 0);
        assert_eq!(FeatureExtractor::complexity_bin(5), 0);
        assert_eq!(FeatureExtractor::complexity_bin(6), 1);
        assert_eq!(FeatureExtractor::complexity_bin(20), 1);
        assert_eq!(FeatureExtractor::complexity_bin(21), 2);
        assert_eq!(FeatureExtractor::complexity_bin(50), 2);
        assert_eq!(FeatureExtractor::complexity_bin(51), 3);
        assert_eq!(FeatureExtractor::complexity_bin(1000), 3);
    }

    #[test]
    fn strategy_inference() {
        // 0 files, 0 lines → SingleLine (trivial no-op)
        assert_eq!(
            FeatureExtractor::infer_strategy(0, 0),
            EditStrategy::SingleLine
        );
        // 1 file, 1 line → SingleLine
        assert_eq!(
            FeatureExtractor::infer_strategy(1, 1),
            EditStrategy::SingleLine
        );
        // 1 file, 10 lines → MultiLine
        assert_eq!(
            FeatureExtractor::infer_strategy(1, 10),
            EditStrategy::MultiLine
        );
        // 3 files → MultiFile
        assert_eq!(
            FeatureExtractor::infer_strategy(3, 25),
            EditStrategy::MultiFile
        );
        // 7 files → Refactor
        assert_eq!(
            FeatureExtractor::infer_strategy(7, 100),
            EditStrategy::Refactor
        );
    }

    #[test]
    fn extract_from_node() {
        let node = make_node("a", 1.0, 12, 1);
        let features = FeatureExtractor::extract(&node);
        assert_eq!(features.complexity_bin, 1); // 12 lines → bin 1
        assert_eq!(features.strategy_bin, 1); // 1 file, >1 line → MultiLine
    }

    #[test]
    fn extract_from_metadata() {
        let features = FeatureExtractor::extract_from_metadata(55, 6);
        assert_eq!(features.complexity_bin, 3); // 55 lines → bin 3
        assert_eq!(features.strategy_bin, 3); // 6 files → Refactor
    }

    #[test]
    fn behavioral_features_clamp() {
        let f = BehavioralFeatures::new(99, 99);
        assert_eq!(f.complexity_bin, COMPLEXITY_BINS - 1);
        assert_eq!(f.strategy_bin, STRATEGY_BINS - 1);
    }

    // -- QualityDiversityArchive --------------------------------------------

    #[test]
    fn empty_archive() {
        let archive = QualityDiversityArchive::new();
        assert!(archive.is_empty());
        assert_eq!(archive.len(), 0);
        assert_eq!(archive.coverage(), 0.0);
        assert!(archive.sample_diverse(5).is_empty());
    }

    #[test]
    fn insert_new_cell() {
        let mut archive = QualityDiversityArchive::new();
        let node = make_node("a", 0.8, 3, 1);
        let features = FeatureExtractor::extract(&node);

        assert!(archive.insert(node, features));
        assert_eq!(archive.len(), 1);
        assert!(archive.coverage() > 0.0);
    }

    #[test]
    fn insert_replaces_worse() {
        let mut archive = QualityDiversityArchive::new();
        let node_a = make_node("a", 0.5, 3, 1);
        let node_b = make_node("b", 0.9, 4, 1);
        let features = BehavioralFeatures::new(0, 0);

        archive.insert(node_a, features);
        assert!(archive.insert(node_b, features));
        assert_eq!(archive.get(&features).unwrap().id, "b");
    }

    #[test]
    fn insert_keeps_better() {
        let mut archive = QualityDiversityArchive::new();
        let node_a = make_node("a", 0.9, 3, 1);
        let node_b = make_node("b", 0.5, 4, 1);
        let features = BehavioralFeatures::new(0, 0);

        archive.insert(node_a, features);
        assert!(!archive.insert(node_b, features));
        assert_eq!(archive.get(&features).unwrap().id, "a");
    }

    #[test]
    fn insert_keeps_equal_score() {
        let mut archive = QualityDiversityArchive::new();
        let node_a = make_node("a", 0.7, 3, 1);
        let node_b = make_node("b", 0.7, 4, 1);
        let features = BehavioralFeatures::new(0, 0);

        archive.insert(node_a, features);
        // Equal score does NOT replace — incumbent wins ties.
        assert!(!archive.insert(node_b, features));
        assert_eq!(archive.get(&features).unwrap().id, "a");
    }

    #[test]
    fn sample_diverse_returns_from_different_bins() {
        let mut archive = QualityDiversityArchive::new();

        // Insert into 3 different bins.
        let n1 = make_node("small-single", 0.6, 2, 1); // bin (0, 0)
        let n2 = make_node("medium-multi", 0.8, 15, 3); // bin (1, 2)
        let n3 = make_node("large-refactor", 0.7, 60, 8); // bin (3, 3)

        archive.insert(
            n1,
            FeatureExtractor::extract(&make_node("small-single", 0.6, 2, 1)),
        );
        archive.insert(
            n2,
            FeatureExtractor::extract(&make_node("medium-multi", 0.8, 15, 3)),
        );
        archive.insert(
            n3,
            FeatureExtractor::extract(&make_node("large-refactor", 0.7, 60, 8)),
        );

        let diverse = archive.sample_diverse(10);
        assert_eq!(diverse.len(), 3);

        // Sorted by score descending.
        assert_eq!(diverse[0].id, "medium-multi"); // 0.8
        assert_eq!(diverse[1].id, "large-refactor"); // 0.7
        assert_eq!(diverse[2].id, "small-single"); // 0.6
    }

    #[test]
    fn sample_diverse_respects_limit() {
        let mut archive = QualityDiversityArchive::new();

        for i in 0..GRID_SIZE {
            let c = i / STRATEGY_BINS;
            let s = i % STRATEGY_BINS;
            let node = ExperimentNode {
                id: format!("n{i}"),
                description: format!("node {i}"),
                score: i as f64 / GRID_SIZE as f64,
                lines_changed: 10,
                files_changed: 1,
                strategy: EditStrategy::MultiLine,
            };
            archive.insert(node, BehavioralFeatures::new(c, s));
        }

        assert_eq!(archive.len(), GRID_SIZE);
        let sample = archive.sample_diverse(3);
        assert_eq!(sample.len(), 3);
    }

    #[test]
    fn coverage_full_grid() {
        let mut archive = QualityDiversityArchive::new();

        for c in 0..COMPLEXITY_BINS {
            for s in 0..STRATEGY_BINS {
                let node = ExperimentNode {
                    id: format!("n{c}{s}"),
                    description: String::new(),
                    score: 1.0,
                    lines_changed: 10,
                    files_changed: 1,
                    strategy: EditStrategy::MultiLine,
                };
                archive.insert(node, BehavioralFeatures::new(c, s));
            }
        }

        assert!((archive.coverage() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn iter_visits_all_cells() {
        let mut archive = QualityDiversityArchive::new();

        let n1 = make_node("a", 1.0, 2, 1);
        let n2 = make_node("b", 1.0, 30, 5);
        let f1 = FeatureExtractor::extract(&n1);
        let f2 = FeatureExtractor::extract(&n2);
        archive.insert(n1, f1);
        archive.insert(n2, f2);

        let collected: Vec<_> = archive.iter().collect();
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn get_missing_cell() {
        let archive = QualityDiversityArchive::new();
        let features = BehavioralFeatures::new(2, 3);
        assert!(archive.get(&features).is_none());
    }
}
