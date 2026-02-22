//! Probe-based context quality evaluation.
//!
//! Validates that packed context (summaries, work packets) preserves
//! task-relevant information. Generates deterministic probe questions
//! from the full session history, then checks if answers are recoverable
//! from the compressed summary alone.
//!
//! If probe pass rate drops below threshold, the summary is too lossy
//! and should trigger a warning or repack.

use crate::feedback::error_parser::ErrorCategory;
use crate::harness::types::StructuredSessionSummary;
use crate::work_packet::types::IterationDelta;

/// Default minimum pass rate for probe evaluation.
const DEFAULT_MIN_PASS_RATE: f64 = 0.8;

/// A probe question with its expected answer and source location.
#[derive(Debug, Clone)]
pub struct Probe {
    /// The factual question to check.
    pub question: String,
    /// The expected answer (substring or exact value).
    pub expected_answer: String,
    /// Where in the session this fact originates (e.g., "iteration 3, files_modified").
    pub answer_source: String,
    /// The kind of probe (determines evaluation strategy).
    pub kind: ProbeKind,
}

/// Classification of what a probe tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeKind {
    /// Did a specific file appear in the summary?
    FilePresence,
    /// Did a specific error category appear?
    ErrorPresence,
    /// Did a specific strategy/approach appear?
    StrategyPresence,
    /// Was iteration count preserved?
    IterationCount,
    /// Was outcome/status preserved?
    OutcomePresence,
}

/// Results of evaluating probes against a summary.
#[derive(Debug, Clone)]
pub struct ProbeResults {
    /// Number of probes that passed.
    pub pass_count: usize,
    /// Number of probes that failed.
    pub fail_count: usize,
    /// Pass rate (0.0 to 1.0).
    pub pass_rate: f64,
    /// Details of failed probes (for debugging).
    pub failed_probes: Vec<FailedProbe>,
}

/// A probe that failed evaluation, with context for debugging.
#[derive(Debug, Clone)]
pub struct FailedProbe {
    /// The original probe question.
    pub question: String,
    /// What we expected to find.
    pub expected: String,
    /// Where the answer should have come from.
    pub source: String,
}

/// Generates deterministic probe questions from session history.
pub struct ProbeGenerator {
    _private: (),
}

impl ProbeGenerator {
    pub fn new() -> Self {
        Self { _private: () }
    }

    /// Generate probes from a full iteration history.
    ///
    /// Creates template-based factual questions that any adequate summary
    /// should be able to answer. Probes cover:
    /// - File presence (were modified files mentioned?)
    /// - Error categories (were error types preserved?)
    /// - Strategies used (were approaches documented?)
    /// - Iteration count (was session length preserved?)
    pub fn generate_probes(&self, deltas: &[IterationDelta]) -> Vec<Probe> {
        let mut probes = Vec::new();

        if deltas.is_empty() {
            return probes;
        }

        // Probe: total iteration count
        probes.push(Probe {
            question: format!(
                "How many iterations were in this session? Expected: {}",
                deltas.len()
            ),
            expected_answer: deltas.len().to_string(),
            answer_source: "iteration count".to_string(),
            kind: ProbeKind::IterationCount,
        });

        // Collect unique files, error categories, and strategies across all iterations.
        let mut all_files: Vec<String> = Vec::new();
        let mut all_fixed: Vec<ErrorCategory> = Vec::new();
        let mut all_new_errors: Vec<ErrorCategory> = Vec::new();
        let mut strategies: Vec<(u32, String)> = Vec::new();

        for delta in deltas {
            for f in &delta.files_modified {
                if !all_files.contains(f) {
                    all_files.push(f.clone());
                }
            }
            for cat in &delta.fixed_errors {
                if !all_fixed.contains(cat) {
                    all_fixed.push(*cat);
                }
            }
            for cat in &delta.new_errors {
                if !all_new_errors.contains(cat) {
                    all_new_errors.push(*cat);
                }
            }
            if !delta.strategy_used.is_empty() {
                strategies.push((delta.iteration, delta.strategy_used.clone()));
            }
        }

        // Probe: file presence (up to 5 most frequently modified files)
        let mut file_freq: Vec<(String, usize)> = all_files
            .iter()
            .map(|f| {
                let count = deltas
                    .iter()
                    .filter(|d| d.files_modified.contains(f))
                    .count();
                (f.clone(), count)
            })
            .collect();
        file_freq.sort_by(|a, b| b.1.cmp(&a.1));

        for (file, _) in file_freq.iter().take(5) {
            probes.push(Probe {
                question: format!("Was the file '{}' modified during the session?", file),
                expected_answer: file.clone(),
                answer_source: "files_modified across iterations".to_string(),
                kind: ProbeKind::FilePresence,
            });
        }

        // Probe: error categories that were fixed
        for cat in &all_fixed {
            let cat_str = format!("{:?}", cat);
            probes.push(Probe {
                question: format!("Was a {:?} error fixed during the session?", cat),
                expected_answer: cat_str,
                answer_source: "fixed_errors across iterations".to_string(),
                kind: ProbeKind::ErrorPresence,
            });
        }

        // Probe: error categories that were introduced (regressions)
        for cat in &all_new_errors {
            let cat_str = format!("{:?}", cat);
            probes.push(Probe {
                question: format!("Was a {:?} error introduced during the session?", cat),
                expected_answer: cat_str,
                answer_source: "new_errors across iterations".to_string(),
                kind: ProbeKind::ErrorPresence,
            });
        }

        // Probe: strategies used (up to 3 most recent)
        for (iter, strategy) in strategies.iter().rev().take(3) {
            probes.push(Probe {
                question: format!(
                    "What strategy was used in iteration {}? Expected: '{}'",
                    iter, strategy
                ),
                expected_answer: strategy.clone(),
                answer_source: format!("iteration {}, strategy_used", iter),
                kind: ProbeKind::StrategyPresence,
            });
        }

        // Probe: final iteration outcome
        if let Some(last) = deltas.last() {
            probes.push(Probe {
                question: format!(
                    "What was the result of the final iteration? Expected: '{}'",
                    last.result_summary
                ),
                expected_answer: last.result_summary.clone(),
                answer_source: format!("iteration {}, result_summary", last.iteration),
                kind: ProbeKind::OutcomePresence,
            });
        }

        probes
    }
}

impl Default for ProbeGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Evaluates probes against a structured session summary.
pub struct ProbeEvaluator {
    /// Minimum pass rate to consider the summary adequate.
    pub min_pass_rate: f64,
}

impl ProbeEvaluator {
    pub fn new() -> Self {
        Self {
            min_pass_rate: DEFAULT_MIN_PASS_RATE,
        }
    }

    /// Create an evaluator with a custom minimum pass rate.
    pub fn with_threshold(min_pass_rate: f64) -> Self {
        Self {
            min_pass_rate: min_pass_rate.clamp(0.0, 1.0),
        }
    }

    /// Evaluate probes against a structured session summary.
    ///
    /// Serializes the summary to a searchable string and checks
    /// whether each probe's expected answer is recoverable.
    pub fn evaluate(&self, summary: &StructuredSessionSummary, probes: &[Probe]) -> ProbeResults {
        if probes.is_empty() {
            return ProbeResults {
                pass_count: 0,
                fail_count: 0,
                pass_rate: 1.0,
                failed_probes: Vec::new(),
            };
        }

        let searchable = self.summary_to_searchable(summary);
        let mut pass_count = 0usize;
        let mut failed_probes = Vec::new();

        for probe in probes {
            let passed = self.check_probe(probe, summary, &searchable);
            if passed {
                pass_count += 1;
            } else {
                failed_probes.push(FailedProbe {
                    question: probe.question.clone(),
                    expected: probe.expected_answer.clone(),
                    source: probe.answer_source.clone(),
                });
            }
        }

        let fail_count = probes.len() - pass_count;
        let pass_rate = pass_count as f64 / probes.len() as f64;

        ProbeResults {
            pass_count,
            fail_count,
            pass_rate,
            failed_probes,
        }
    }

    /// Check if the summary meets the minimum pass rate.
    pub fn is_adequate(&self, results: &ProbeResults) -> bool {
        results.pass_rate >= self.min_pass_rate
    }

    /// Check a single probe against the summary.
    fn check_probe(
        &self,
        probe: &Probe,
        summary: &StructuredSessionSummary,
        searchable: &str,
    ) -> bool {
        match probe.kind {
            ProbeKind::IterationCount => {
                // Check if the iteration count is preserved in the summary.
                let expected: u32 = probe.expected_answer.parse().unwrap_or(0);
                summary.total_iterations == expected
            }
            ProbeKind::FilePresence => {
                // Check if the filename appears anywhere in the searchable summary.
                let needle = &probe.expected_answer;
                searchable.contains(needle)
            }
            ProbeKind::ErrorPresence => {
                // Check if the error category name appears (case-insensitive).
                let needle = probe.expected_answer.to_lowercase();
                let haystack = searchable.to_lowercase();
                haystack.contains(&needle)
            }
            ProbeKind::StrategyPresence => {
                // Check if key words from the strategy appear in the summary.
                // We check that at least 50% of significant words (>3 chars) match.
                let words: Vec<&str> = probe
                    .expected_answer
                    .split_whitespace()
                    .filter(|w| w.len() > 3)
                    .collect();
                if words.is_empty() {
                    // Short strategy — just do substring match.
                    searchable.contains(&probe.expected_answer)
                } else {
                    let haystack = searchable.to_lowercase();
                    let matched = words
                        .iter()
                        .filter(|w| haystack.contains(&w.to_lowercase()))
                        .count();
                    matched * 2 >= words.len() // ≥50% word match
                }
            }
            ProbeKind::OutcomePresence => {
                // Similar to strategy — check key words from result summary.
                let words: Vec<&str> = probe
                    .expected_answer
                    .split_whitespace()
                    .filter(|w| w.len() > 3)
                    .collect();
                if words.is_empty() {
                    searchable.contains(&probe.expected_answer)
                } else {
                    let haystack = searchable.to_lowercase();
                    let matched = words
                        .iter()
                        .filter(|w| haystack.contains(&w.to_lowercase()))
                        .count();
                    matched * 2 >= words.len()
                }
            }
        }
    }

    /// Convert a structured summary to a flat searchable string.
    ///
    /// Concatenates all text fields so probes can search across them.
    fn summary_to_searchable(&self, summary: &StructuredSessionSummary) -> String {
        let mut parts = Vec::new();

        parts.push(format!("session_id: {}", summary.session_id));
        parts.push(format!("status: {:?}", summary.status));
        parts.push(format!("total_iterations: {}", summary.total_iterations));

        for feature in &summary.features {
            parts.push(format!("feature: {}", feature.feature_id));
            parts.push(format!("feature_status: {:?}", feature.status));
            parts.push(format!("start_iteration: {}", feature.start_iteration));
            if let Some(end) = feature.end_iteration {
                parts.push(format!("end_iteration: {}", end));
            }
            for step in &feature.iterative_steps {
                parts.push(step.clone());
            }
        }

        for checkpoint in &summary.checkpoints {
            parts.push(format!(
                "checkpoint: iteration {} commit {}",
                checkpoint.iteration, checkpoint.commit_hash
            ));
            if let Some(ref fid) = checkpoint.feature_id {
                parts.push(format!("checkpoint_feature: {}", fid));
            }
        }

        for error in &summary.errors {
            parts.push(error.clone());
        }

        parts.join("\n")
    }
}

impl Default for ProbeEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harness::types::{
        CheckpointSummary, FeatureProgressSummary, FeatureWorkStatus, SessionStatus,
    };

    fn make_delta(
        iteration: u32,
        files: &[&str],
        fixed: &[ErrorCategory],
        new_errors: &[ErrorCategory],
        strategy: &str,
        result: &str,
    ) -> IterationDelta {
        IterationDelta {
            iteration,
            fixed_errors: fixed.to_vec(),
            new_errors: new_errors.to_vec(),
            files_modified: files.iter().map(|s| s.to_string()).collect(),
            hypothesis: None,
            result_summary: result.to_string(),
            strategy_used: strategy.to_string(),
        }
    }

    fn make_summary(
        iterations: u32,
        features: Vec<FeatureProgressSummary>,
        errors: Vec<String>,
    ) -> StructuredSessionSummary {
        StructuredSessionSummary {
            session_id: "test-session-001".to_string(),
            status: SessionStatus::Active,
            total_iterations: iterations,
            features,
            checkpoints: vec![],
            errors,
        }
    }

    fn make_feature(id: &str, steps: &[&str], status: FeatureWorkStatus) -> FeatureProgressSummary {
        FeatureProgressSummary {
            feature_id: id.to_string(),
            start_iteration: 1,
            end_iteration: None,
            status,
            iterative_steps: steps.iter().map(|s| s.to_string()).collect(),
        }
    }

    // ── ProbeGenerator tests ──

    #[test]
    fn test_generate_probes_empty_deltas() {
        let gen = ProbeGenerator::new();
        let probes = gen.generate_probes(&[]);
        assert!(probes.is_empty());
    }

    #[test]
    fn test_generate_probes_single_iteration() {
        let gen = ProbeGenerator::new();
        let deltas = vec![make_delta(
            1,
            &["src/lib.rs"],
            &[ErrorCategory::TypeMismatch],
            &[],
            "added type annotation",
            "type error fixed",
        )];
        let probes = gen.generate_probes(&deltas);

        // Should have: iteration count + file + fixed error + strategy + outcome = 5
        assert_eq!(probes.len(), 5);

        // Check iteration count probe
        let iter_probe = probes.iter().find(|p| p.kind == ProbeKind::IterationCount);
        assert!(iter_probe.is_some());
        assert_eq!(iter_probe.unwrap().expected_answer, "1");

        // Check file presence probe
        let file_probe = probes.iter().find(|p| p.kind == ProbeKind::FilePresence);
        assert!(file_probe.is_some());
        assert_eq!(file_probe.unwrap().expected_answer, "src/lib.rs");

        // Check error presence probe
        let error_probe = probes.iter().find(|p| p.kind == ProbeKind::ErrorPresence);
        assert!(error_probe.is_some());
        assert_eq!(error_probe.unwrap().expected_answer, "TypeMismatch");
    }

    #[test]
    fn test_generate_probes_deduplicates_files() {
        let gen = ProbeGenerator::new();
        let deltas = vec![
            make_delta(1, &["src/lib.rs"], &[], &[], "fix A", "done A"),
            make_delta(2, &["src/lib.rs"], &[], &[], "fix B", "done B"),
        ];
        let probes = gen.generate_probes(&deltas);

        let file_probes: Vec<_> = probes
            .iter()
            .filter(|p| p.kind == ProbeKind::FilePresence)
            .collect();
        // Only one file probe even though file appears in both iterations
        assert_eq!(file_probes.len(), 1);
    }

    #[test]
    fn test_generate_probes_caps_files_at_five() {
        let gen = ProbeGenerator::new();
        let files: Vec<String> = (0..10).map(|i| format!("src/mod_{}.rs", i)).collect();
        let file_refs: Vec<&str> = files.iter().map(|s| s.as_str()).collect();
        let deltas = vec![make_delta(1, &file_refs, &[], &[], "bulk change", "done")];
        let probes = gen.generate_probes(&deltas);

        let file_probes: Vec<_> = probes
            .iter()
            .filter(|p| p.kind == ProbeKind::FilePresence)
            .collect();
        assert_eq!(file_probes.len(), 5);
    }

    #[test]
    fn test_generate_probes_includes_regressions() {
        let gen = ProbeGenerator::new();
        let deltas = vec![make_delta(
            1,
            &["src/lib.rs"],
            &[ErrorCategory::TypeMismatch],
            &[ErrorCategory::BorrowChecker],
            "added Arc wrapper",
            "type fixed but introduced borrow error",
        )];
        let probes = gen.generate_probes(&deltas);

        let error_probes: Vec<_> = probes
            .iter()
            .filter(|p| p.kind == ProbeKind::ErrorPresence)
            .collect();
        // Fixed: TypeMismatch, New: BorrowChecker
        assert_eq!(error_probes.len(), 2);
    }

    #[test]
    fn test_generate_probes_strategies_recent_first() {
        let gen = ProbeGenerator::new();
        let deltas = vec![
            make_delta(1, &["a.rs"], &[], &[], "strategy alpha", "done"),
            make_delta(2, &["b.rs"], &[], &[], "strategy beta", "done"),
            make_delta(3, &["c.rs"], &[], &[], "strategy gamma", "done"),
            make_delta(4, &["d.rs"], &[], &[], "strategy delta", "done"),
        ];
        let probes = gen.generate_probes(&deltas);

        let strategy_probes: Vec<_> = probes
            .iter()
            .filter(|p| p.kind == ProbeKind::StrategyPresence)
            .collect();
        // At most 3, most recent first
        assert_eq!(strategy_probes.len(), 3);
        assert!(strategy_probes[0].expected_answer.contains("delta"));
        assert!(strategy_probes[1].expected_answer.contains("gamma"));
        assert!(strategy_probes[2].expected_answer.contains("beta"));
    }

    // ── ProbeEvaluator tests ──

    #[test]
    fn test_evaluate_empty_probes() {
        let eval = ProbeEvaluator::new();
        let summary = make_summary(0, vec![], vec![]);
        let results = eval.evaluate(&summary, &[]);
        assert_eq!(results.pass_rate, 1.0);
        assert!(results.failed_probes.is_empty());
    }

    #[test]
    fn test_evaluate_good_summary_passes() {
        let eval = ProbeEvaluator::new();

        let feature = make_feature(
            "add-telemetry",
            &[
                "Modified src/telemetry.rs: added TypeMismatch fix using type annotation strategy",
                "Fixed BorrowChecker error, all tests passing now",
            ],
            FeatureWorkStatus::Completed,
        );
        let summary = make_summary(
            2,
            vec![feature],
            vec!["TypeMismatch in src/telemetry.rs".to_string()],
        );

        let gen = ProbeGenerator::new();
        let deltas = vec![
            make_delta(
                1,
                &["src/telemetry.rs"],
                &[ErrorCategory::TypeMismatch],
                &[],
                "type annotation",
                "type error fixed",
            ),
            make_delta(
                2,
                &["src/telemetry.rs"],
                &[ErrorCategory::BorrowChecker],
                &[],
                "borrow fix",
                "all tests passing now",
            ),
        ];
        let probes = gen.generate_probes(&deltas);
        let results = eval.evaluate(&summary, &probes);

        // Most probes should pass since the summary contains the key info.
        assert!(
            results.pass_rate >= 0.7,
            "Expected pass_rate >= 0.7, got {} (failed: {:?})",
            results.pass_rate,
            results.failed_probes
        );
    }

    #[test]
    fn test_evaluate_bad_summary_fails() {
        let eval = ProbeEvaluator::new();

        // Empty summary — preserves nothing
        let summary = make_summary(0, vec![], vec![]);

        let gen = ProbeGenerator::new();
        let deltas = vec![
            make_delta(
                1,
                &["src/lib.rs"],
                &[ErrorCategory::TypeMismatch],
                &[],
                "added annotation",
                "fixed type error",
            ),
            make_delta(
                2,
                &["src/main.rs"],
                &[ErrorCategory::BorrowChecker],
                &[ErrorCategory::Lifetime],
                "added Arc wrapper",
                "borrow fixed but lifetime introduced",
            ),
        ];
        let probes = gen.generate_probes(&deltas);
        let results = eval.evaluate(&summary, &probes);

        // Should fail most probes
        assert!(
            results.pass_rate < 0.5,
            "Expected pass_rate < 0.5, got {}",
            results.pass_rate
        );
        assert!(!eval.is_adequate(&results));
    }

    #[test]
    fn test_evaluate_iteration_count_probe() {
        let eval = ProbeEvaluator::new();

        // Summary says 3 iterations, probes expect 3
        let summary = make_summary(3, vec![], vec![]);

        let probes = vec![Probe {
            question: "How many iterations?".to_string(),
            expected_answer: "3".to_string(),
            answer_source: "iteration count".to_string(),
            kind: ProbeKind::IterationCount,
        }];

        let results = eval.evaluate(&summary, &probes);
        assert_eq!(results.pass_count, 1);
        assert_eq!(results.fail_count, 0);
    }

    #[test]
    fn test_evaluate_iteration_count_mismatch() {
        let eval = ProbeEvaluator::new();

        // Summary says 2, probe expects 3
        let summary = make_summary(2, vec![], vec![]);
        let probes = vec![Probe {
            question: "How many iterations?".to_string(),
            expected_answer: "3".to_string(),
            answer_source: "iteration count".to_string(),
            kind: ProbeKind::IterationCount,
        }];

        let results = eval.evaluate(&summary, &probes);
        assert_eq!(results.pass_count, 0);
        assert_eq!(results.fail_count, 1);
    }

    #[test]
    fn test_evaluate_file_presence() {
        let eval = ProbeEvaluator::new();

        let feature = make_feature(
            "fix-bug",
            &["Modified src/lib.rs to fix the compile error"],
            FeatureWorkStatus::Completed,
        );
        let summary = make_summary(1, vec![feature], vec![]);

        let probes = vec![
            Probe {
                question: "Was src/lib.rs modified?".to_string(),
                expected_answer: "src/lib.rs".to_string(),
                answer_source: "files_modified".to_string(),
                kind: ProbeKind::FilePresence,
            },
            Probe {
                question: "Was src/main.rs modified?".to_string(),
                expected_answer: "src/main.rs".to_string(),
                answer_source: "files_modified".to_string(),
                kind: ProbeKind::FilePresence,
            },
        ];

        let results = eval.evaluate(&summary, &probes);
        assert_eq!(results.pass_count, 1); // lib.rs found
        assert_eq!(results.fail_count, 1); // main.rs not found
    }

    #[test]
    fn test_evaluate_strategy_word_matching() {
        let eval = ProbeEvaluator::new();

        let feature = make_feature(
            "fix",
            &["Used Arc wrapper to resolve shared ownership issue"],
            FeatureWorkStatus::Completed,
        );
        let summary = make_summary(1, vec![feature], vec![]);

        let probes = vec![Probe {
            question: "What strategy was used?".to_string(),
            expected_answer: "added Arc wrapper for shared ownership".to_string(),
            answer_source: "iteration 1".to_string(),
            kind: ProbeKind::StrategyPresence,
        }];

        let results = eval.evaluate(&summary, &probes);
        // "wrapper", "shared", "ownership" should match ≥50%
        assert_eq!(results.pass_count, 1);
    }

    #[test]
    fn test_threshold_configuration() {
        let strict = ProbeEvaluator::with_threshold(0.95);
        let lenient = ProbeEvaluator::with_threshold(0.5);

        let results = ProbeResults {
            pass_count: 8,
            fail_count: 2,
            pass_rate: 0.8,
            failed_probes: vec![],
        };

        assert!(!strict.is_adequate(&results)); // 0.8 < 0.95
        assert!(lenient.is_adequate(&results)); // 0.8 >= 0.5
    }

    #[test]
    fn test_threshold_clamped() {
        let eval = ProbeEvaluator::with_threshold(1.5);
        assert_eq!(eval.min_pass_rate, 1.0);

        let eval = ProbeEvaluator::with_threshold(-0.5);
        assert_eq!(eval.min_pass_rate, 0.0);
    }

    #[test]
    fn test_evaluate_error_presence_case_insensitive() {
        let eval = ProbeEvaluator::new();

        let summary = make_summary(1, vec![], vec!["typemismatch error in line 42".to_string()]);

        let probes = vec![Probe {
            question: "Was there a TypeMismatch?".to_string(),
            expected_answer: "TypeMismatch".to_string(),
            answer_source: "fixed_errors".to_string(),
            kind: ProbeKind::ErrorPresence,
        }];

        let results = eval.evaluate(&summary, &probes);
        assert_eq!(results.pass_count, 1);
    }

    #[test]
    fn test_checkpoint_info_searchable() {
        let eval = ProbeEvaluator::new();

        let summary = StructuredSessionSummary {
            session_id: "sess-abc".to_string(),
            status: SessionStatus::Active,
            total_iterations: 2,
            features: vec![],
            checkpoints: vec![CheckpointSummary {
                iteration: 1,
                commit_hash: "abc123def".to_string(),
                feature_id: Some("my-feature".to_string()),
            }],
            errors: vec![],
        };

        let probes = vec![Probe {
            question: "Was there a checkpoint at abc123def?".to_string(),
            expected_answer: "abc123def".to_string(),
            answer_source: "checkpoints".to_string(),
            kind: ProbeKind::FilePresence, // Reuse file presence for substring check
        }];

        let results = eval.evaluate(&summary, &probes);
        assert_eq!(results.pass_count, 1);
    }

    #[test]
    fn test_roundtrip_generate_evaluate() {
        // End-to-end: generate probes from deltas, build a summary that
        // preserves the key facts, evaluate.
        let gen = ProbeGenerator::new();
        let eval = ProbeEvaluator::new();

        let deltas = vec![
            make_delta(
                1,
                &["src/parser.rs", "src/ast.rs"],
                &[],
                &[ErrorCategory::Syntax],
                "refactored parser grammar",
                "syntax error introduced in ast module",
            ),
            make_delta(
                2,
                &["src/ast.rs"],
                &[ErrorCategory::Syntax],
                &[],
                "fixed ast node construction",
                "all tests passing",
            ),
        ];

        let probes = gen.generate_probes(&deltas);

        // Build a summary that preserves all the key facts
        let feature = make_feature(
            "parser-refactor",
            &[
                "Iter 1: refactored parser grammar in src/parser.rs and src/ast.rs, introduced Syntax error",
                "Iter 2: fixed ast node construction in src/ast.rs, Syntax error resolved, all tests passing",
            ],
            FeatureWorkStatus::Completed,
        );
        let summary = make_summary(
            2,
            vec![feature],
            vec!["Syntax error in src/ast.rs".to_string()],
        );

        let results = eval.evaluate(&summary, &probes);

        assert!(
            eval.is_adequate(&results),
            "Expected adequate summary, got pass_rate={} (failed: {:?})",
            results.pass_rate,
            results.failed_probes
        );
    }
}
