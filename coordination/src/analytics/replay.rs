//! Experience replay: indexes successful traces and replays them as
//! structured hints when a new task resembles a past success.
//!
//! Unlike skills (which capture the final pattern), experience traces
//! preserve the full sequence of actions — useful for multi-step fixes
//! where order matters.

use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::feedback::error_parser::ErrorCategory;
use crate::work_packet::types::IterationDelta;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Outcome of a traced session.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TraceOutcome {
    Success,
    Failure,
    Escalated,
}

/// A recorded experience trace from a completed session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExperienceTrace {
    /// Unique trace identifier.
    pub id: String,
    /// The task context at the start of the session.
    pub context: TraceContext,
    /// Ordered sequence of iteration deltas (the "replay").
    pub iteration_sequence: Vec<IterationDelta>,
    /// Final outcome.
    pub outcome: TraceOutcome,
    /// Total duration in seconds.
    pub duration_secs: u64,
    /// Number of iterations used.
    pub iterations_used: u32,
}

/// Context snapshot at the start of a traced session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceContext {
    /// Error categories present at session start.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub error_categories: Vec<ErrorCategory>,
    /// Files involved in the task.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub file_patterns: Vec<String>,
    /// Task type (e.g., "bug", "feature").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_type: Option<String>,
}

/// A condensed hint derived from an experience trace, injected into work packets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayHint {
    /// ID of the originating trace.
    pub trace_id: String,
    /// Similarity score (0.0–1.0) to the current task.
    pub similarity: f64,
    /// Number of iterations the original session took.
    pub iterations_used: u32,
    /// Ordered summary of strategies used.
    pub strategy_sequence: Vec<String>,
    /// Files that were modified across all iterations.
    pub files_modified: Vec<String>,
}

// ---------------------------------------------------------------------------
// Similarity scoring
// ---------------------------------------------------------------------------

/// Weights for similarity scoring dimensions.
const WEIGHT_ERROR_CATEGORIES: f64 = 0.5;
const WEIGHT_FILE_PATTERNS: f64 = 0.3;
const WEIGHT_TASK_TYPE: f64 = 0.2;

/// Compute similarity between a trace context and a query context.
///
/// Weighted Jaccard-like overlap across error categories, file patterns,
/// and task type.
fn compute_similarity(trace: &TraceContext, query: &TraceContext) -> f64 {
    let mut score = 0.0;
    let mut total_weight = 0.0;

    // Error category overlap (Jaccard)
    if !trace.error_categories.is_empty() || !query.error_categories.is_empty() {
        let intersection = trace
            .error_categories
            .iter()
            .filter(|c| query.error_categories.contains(c))
            .count();
        let mut union_set = trace.error_categories.clone();
        for c in &query.error_categories {
            if !union_set.contains(c) {
                union_set.push(*c);
            }
        }
        let union = union_set.len();
        if union > 0 {
            score += WEIGHT_ERROR_CATEGORIES * (intersection as f64 / union as f64);
        }
        total_weight += WEIGHT_ERROR_CATEGORIES;
    }

    // File pattern overlap (simple string-prefix matching)
    if !trace.file_patterns.is_empty() || !query.file_patterns.is_empty() {
        let intersection = trace
            .file_patterns
            .iter()
            .filter(|f| query.file_patterns.iter().any(|qf| shares_directory(f, qf)))
            .count();
        let max_len = trace.file_patterns.len().max(query.file_patterns.len());
        if max_len > 0 {
            score += WEIGHT_FILE_PATTERNS * (intersection as f64 / max_len as f64);
        }
        total_weight += WEIGHT_FILE_PATTERNS;
    }

    // Task type match (binary)
    if trace.task_type.is_some() || query.task_type.is_some() {
        let matches = match (&trace.task_type, &query.task_type) {
            (Some(a), Some(b)) => a.eq_ignore_ascii_case(b),
            _ => false,
        };
        score += WEIGHT_TASK_TYPE * if matches { 1.0 } else { 0.0 };
        total_weight += WEIGHT_TASK_TYPE;
    }

    if total_weight > 0.0 {
        score / total_weight
    } else {
        0.0
    }
}

/// Check if two file paths share the same parent directory.
fn shares_directory(a: &str, b: &str) -> bool {
    let dir_a = a.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    let dir_b = b.rsplit_once('/').map(|(d, _)| d).unwrap_or("");
    !dir_a.is_empty() && dir_a == dir_b
}

// ---------------------------------------------------------------------------
// TraceIndex (experience replay library)
// ---------------------------------------------------------------------------

/// Minimum similarity threshold for a trace to be considered a match.
const DEFAULT_MIN_SIMILARITY: f64 = 0.3;

/// Indexed store of experience traces with similarity-based retrieval.
pub struct TraceIndex {
    traces: Vec<ExperienceTrace>,
    min_similarity: f64,
}

impl TraceIndex {
    /// Create an empty trace index.
    pub fn new() -> Self {
        Self {
            traces: Vec::new(),
            min_similarity: DEFAULT_MIN_SIMILARITY,
        }
    }

    /// Load traces from a JSON file. Returns empty index if file doesn't exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let data = std::fs::read_to_string(path).context(format!(
            "Failed to read trace index from {}",
            path.display()
        ))?;
        let traces: Vec<ExperienceTrace> =
            serde_json::from_str(&data).context("Failed to parse trace index JSON")?;
        Ok(Self {
            traces,
            min_similarity: DEFAULT_MIN_SIMILARITY,
        })
    }

    /// Persist traces to a JSON file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let data =
            serde_json::to_string_pretty(&self.traces).context("Failed to serialize traces")?;
        std::fs::write(path, data)
            .context(format!("Failed to write trace index to {}", path.display()))?;
        Ok(())
    }

    /// Override the minimum similarity threshold.
    pub fn with_min_similarity(mut self, min: f64) -> Self {
        self.min_similarity = min;
        self
    }

    /// Add a completed trace to the index.
    pub fn add_trace(&mut self, trace: ExperienceTrace) {
        self.traces.push(trace);
    }

    /// Find the top-k most similar successful traces for the given context.
    ///
    /// Only returns traces with outcome == Success and similarity >= min_similarity.
    pub fn find_similar(&self, context: &TraceContext, k: usize) -> Vec<ReplayHint> {
        let mut scored: Vec<(&ExperienceTrace, f64)> = self
            .traces
            .iter()
            .filter(|t| t.outcome == TraceOutcome::Success)
            .map(|t| {
                let sim = compute_similarity(&t.context, context);
                (t, sim)
            })
            .filter(|(_, sim)| *sim >= self.min_similarity)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        scored
            .into_iter()
            .map(|(trace, similarity)| {
                let strategy_sequence: Vec<String> = trace
                    .iteration_sequence
                    .iter()
                    .filter(|d| !d.strategy_used.is_empty())
                    .map(|d| d.strategy_used.clone())
                    .collect();

                let mut files_modified: Vec<String> = trace
                    .iteration_sequence
                    .iter()
                    .flat_map(|d| d.files_modified.iter().cloned())
                    .collect();
                files_modified.sort();
                files_modified.dedup();

                ReplayHint {
                    trace_id: trace.id.clone(),
                    similarity,
                    iterations_used: trace.iterations_used,
                    strategy_sequence,
                    files_modified,
                }
            })
            .collect()
    }

    /// Number of traces in the index.
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Get all traces.
    pub fn traces(&self) -> &[ExperienceTrace] {
        &self.traces
    }
}

impl Default for TraceIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn borrow_context() -> TraceContext {
        TraceContext {
            error_categories: vec![ErrorCategory::BorrowChecker, ErrorCategory::Lifetime],
            file_patterns: vec!["src/agents/coder.rs".into()],
            task_type: Some("bug".into()),
        }
    }

    fn trait_context() -> TraceContext {
        TraceContext {
            error_categories: vec![ErrorCategory::TraitBound],
            file_patterns: vec!["src/tools/bundles.rs".into()],
            task_type: Some("feature".into()),
        }
    }

    fn sample_delta(iter: u32, strategy: &str) -> IterationDelta {
        IterationDelta {
            iteration: iter,
            fixed_errors: vec![],
            new_errors: vec![],
            files_modified: vec!["src/agents/coder.rs".into()],
            hypothesis: None,
            result_summary: format!("Iteration {iter}"),
            strategy_used: strategy.into(),
        }
    }

    fn sample_trace(id: &str, context: TraceContext, outcome: TraceOutcome) -> ExperienceTrace {
        ExperienceTrace {
            id: id.into(),
            context,
            iteration_sequence: vec![
                sample_delta(1, "Wrap in Arc"),
                sample_delta(2, "Add lifetime annotation"),
            ],
            outcome,
            duration_secs: 120,
            iterations_used: 2,
        }
    }

    // --- Similarity scoring ---

    #[test]
    fn test_similarity_identical_contexts() {
        let ctx = borrow_context();
        let sim = compute_similarity(&ctx, &ctx);
        assert!((sim - 1.0).abs() < 0.01, "Expected ~1.0, got {sim}");
    }

    #[test]
    fn test_similarity_no_overlap() {
        let sim = compute_similarity(&borrow_context(), &trait_context());
        assert!(sim < 0.3, "Expected low similarity, got {sim}");
    }

    #[test]
    fn test_similarity_partial_category_overlap() {
        let ctx1 = TraceContext {
            error_categories: vec![ErrorCategory::BorrowChecker, ErrorCategory::Lifetime],
            file_patterns: vec![],
            task_type: None,
        };
        let ctx2 = TraceContext {
            error_categories: vec![ErrorCategory::BorrowChecker, ErrorCategory::TraitBound],
            file_patterns: vec![],
            task_type: None,
        };
        let sim = compute_similarity(&ctx1, &ctx2);
        // Jaccard: 1 intersection / 3 union = 0.333
        assert!(sim > 0.2 && sim < 0.5, "Expected ~0.33, got {sim}");
    }

    #[test]
    fn test_similarity_empty_contexts() {
        let empty = TraceContext {
            error_categories: vec![],
            file_patterns: vec![],
            task_type: None,
        };
        let sim = compute_similarity(&empty, &empty);
        assert!((sim - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_shares_directory() {
        assert!(shares_directory(
            "src/agents/coder.rs",
            "src/agents/manager.rs"
        ));
        assert!(!shares_directory(
            "src/agents/coder.rs",
            "src/tools/bundles.rs"
        ));
        assert!(!shares_directory("lib.rs", "main.rs")); // no parent dir
    }

    // --- TraceIndex ---

    #[test]
    fn test_index_add_and_find() {
        let mut index = TraceIndex::new().with_min_similarity(0.1);
        index.add_trace(sample_trace(
            "trace-1",
            borrow_context(),
            TraceOutcome::Success,
        ));

        let hints = index.find_similar(&borrow_context(), 5);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].trace_id, "trace-1");
        assert!(hints[0].similarity > 0.9);
        assert_eq!(hints[0].iterations_used, 2);
        assert_eq!(
            hints[0].strategy_sequence,
            vec!["Wrap in Arc", "Add lifetime annotation"]
        );
    }

    #[test]
    fn test_index_filters_failures() {
        let mut index = TraceIndex::new();
        index.add_trace(sample_trace(
            "success",
            borrow_context(),
            TraceOutcome::Success,
        ));
        index.add_trace(sample_trace(
            "failure",
            borrow_context(),
            TraceOutcome::Failure,
        ));
        index.add_trace(sample_trace(
            "escalated",
            borrow_context(),
            TraceOutcome::Escalated,
        ));

        let hints = index.find_similar(&borrow_context(), 10);
        assert_eq!(hints.len(), 1);
        assert_eq!(hints[0].trace_id, "success");
    }

    #[test]
    fn test_index_respects_min_similarity() {
        let mut index = TraceIndex::new().with_min_similarity(0.9);
        index.add_trace(sample_trace(
            "trace-1",
            borrow_context(),
            TraceOutcome::Success,
        ));

        // Different context → low similarity → filtered out
        let hints = index.find_similar(&trait_context(), 5);
        assert!(hints.is_empty());
    }

    #[test]
    fn test_index_top_k_limit() {
        let mut index = TraceIndex::new().with_min_similarity(0.1);
        for i in 0..10 {
            index.add_trace(sample_trace(
                &format!("trace-{i}"),
                borrow_context(),
                TraceOutcome::Success,
            ));
        }

        let hints = index.find_similar(&borrow_context(), 3);
        assert_eq!(hints.len(), 3);
    }

    #[test]
    fn test_index_ranked_by_similarity() {
        let mut index = TraceIndex::new().with_min_similarity(0.1);

        // Perfect match
        index.add_trace(sample_trace(
            "exact",
            borrow_context(),
            TraceOutcome::Success,
        ));

        // Partial match (only error categories overlap)
        let partial = TraceContext {
            error_categories: vec![ErrorCategory::BorrowChecker],
            file_patterns: vec!["tests/test.rs".into()],
            task_type: Some("feature".into()),
        };
        index.add_trace(sample_trace("partial", partial, TraceOutcome::Success));

        let hints = index.find_similar(&borrow_context(), 5);
        assert_eq!(hints.len(), 2);
        assert_eq!(hints[0].trace_id, "exact");
        assert!(hints[0].similarity > hints[1].similarity);
    }

    #[test]
    fn test_index_persistence_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("traces.json");

        let mut index = TraceIndex::new();
        index.add_trace(sample_trace(
            "persisted",
            borrow_context(),
            TraceOutcome::Success,
        ));
        index.save(&path).unwrap();

        let loaded = TraceIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded.traces()[0].id, "persisted");
    }

    #[test]
    fn test_index_load_nonexistent() {
        let index = TraceIndex::load(Path::new("/nonexistent/traces.json")).unwrap();
        assert!(index.is_empty());
    }

    #[test]
    fn test_replay_hint_deduplicates_files() {
        let mut index = TraceIndex::new().with_min_similarity(0.1);
        let mut trace = sample_trace("dup-files", borrow_context(), TraceOutcome::Success);
        // Both iterations modify the same file
        trace.iteration_sequence[0].files_modified = vec!["src/lib.rs".into()];
        trace.iteration_sequence[1].files_modified =
            vec!["src/lib.rs".into(), "src/main.rs".into()];
        index.add_trace(trace);

        let hints = index.find_similar(&borrow_context(), 1);
        assert_eq!(hints[0].files_modified, vec!["src/lib.rs", "src/main.rs"]);
    }

    #[test]
    fn test_trace_serde_roundtrip() {
        let trace = sample_trace("serde-test", borrow_context(), TraceOutcome::Success);
        let json = serde_json::to_string(&trace).unwrap();
        let deserialized: ExperienceTrace = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "serde-test");
        assert_eq!(deserialized.outcome, TraceOutcome::Success);
        assert_eq!(deserialized.iteration_sequence.len(), 2);
    }

    #[test]
    fn test_replay_hint_serde_roundtrip() {
        let hint = ReplayHint {
            trace_id: "trace-001".into(),
            similarity: 0.85,
            iterations_used: 3,
            strategy_sequence: vec!["step1".into(), "step2".into()],
            files_modified: vec!["src/lib.rs".into()],
        };
        let json = serde_json::to_string(&hint).unwrap();
        let deserialized: ReplayHint = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.trace_id, "trace-001");
        assert!((deserialized.similarity - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_index_returns_empty() {
        let index = TraceIndex::new();
        let hints = index.find_similar(&borrow_context(), 5);
        assert!(hints.is_empty());
    }
}
