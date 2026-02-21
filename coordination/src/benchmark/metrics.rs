//! Benchmark metrics tracking
//!
//! Collects and aggregates metrics for benchmark runs.

use crate::benchmark::problem::{Difficulty, ProblemStatus};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Metrics for a single problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemMetrics {
    /// Problem identifier
    pub problem_id: String,
    /// Problem difficulty
    pub difficulty: Difficulty,
    /// Final status
    pub status: ProblemStatus,
    /// Whether first attempt compiled
    pub first_attempt_compiled: bool,
    /// Total iterations used
    pub total_iterations: u32,
    /// Total tokens consumed
    pub total_tokens: u64,
    /// Total time in milliseconds
    pub total_time_ms: u64,
    /// Final model tier used
    pub final_model_tier: Option<String>,
}

impl ProblemMetrics {
    /// Check if problem succeeded (compiled)
    pub fn succeeded(&self) -> bool {
        matches!(
            self.status,
            ProblemStatus::PassedFirst | ProblemStatus::PassedCorrected
        )
    }

    /// Get tokens per iteration
    pub fn tokens_per_iteration(&self) -> f32 {
        if self.total_iterations > 0 {
            self.total_tokens as f32 / self.total_iterations as f32
        } else {
            0.0
        }
    }
}

/// Metrics for a single attempt within the correction loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptMetrics {
    /// Attempt number
    pub attempt_number: u32,
    /// Model tier used
    pub model_tier: String,
    /// Tokens used
    pub tokens: u32,
    /// Time in milliseconds
    pub time_ms: u64,
    /// Error count before attempt
    pub errors_before: usize,
    /// Error count after attempt
    pub errors_after: usize,
    /// Whether this attempt succeeded
    pub compiled: bool,
    /// The code artifact produced during this iteration
    pub generated_code: Option<String>,
    /// The compiler error output from this iteration
    pub compiler_output: Option<String>,
}

impl AttemptMetrics {
    /// Create a new `AttemptMetrics`; artifact fields default to `None`
    pub fn new(
        attempt_number: u32,
        model_tier: String,
        tokens: u32,
        time_ms: u64,
        errors_before: usize,
        errors_after: usize,
        compiled: bool,
    ) -> Self {
        Self {
            attempt_number,
            model_tier,
            tokens,
            time_ms,
            errors_before,
            errors_after,
            compiled,
            generated_code: None,
            compiler_output: None,
        }
    }

    /// Attach the generated code artifact to this attempt record
    pub fn with_generated_code(mut self, code: String) -> Self {
        self.generated_code = Some(code);
        self
    }

    /// Attach the compiler output to this attempt record
    pub fn with_compiler_output(mut self, output: String) -> Self {
        self.compiler_output = Some(output);
        self
    }

    /// Check if this attempt made progress (reduced errors)
    pub fn made_progress(&self) -> bool {
        self.compiled || self.errors_after < self.errors_before
    }
}

/// Aggregated metrics for an entire benchmark session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetrics {
    /// Session identifier
    pub session_id: String,
    /// Total problems in benchmark
    pub total_problems: usize,
    /// Problems completed (attempted)
    pub completed_problems: usize,
    /// Problems that passed on first attempt
    pub passed_first_attempt: usize,
    /// Problems that passed after correction
    pub passed_with_correction: usize,
    /// Problems that failed all attempts
    pub failed_problems: usize,
    /// First attempt success rate (%)
    pub first_attempt_rate: f32,
    /// Overall success rate (%)
    pub overall_success_rate: f32,
    /// Easy problem success rate (%)
    pub easy_success_rate: f32,
    /// Hard problem success rate (%)
    pub hard_success_rate: f32,
    /// Total tokens consumed
    pub total_tokens: u64,
    /// Total time in milliseconds
    pub total_time_ms: u64,
    /// Average iterations per problem
    pub average_iterations: f32,
    /// Model usage breakdown
    pub model_usage: HashMap<String, u32>,
}

impl BenchmarkMetrics {
    /// Format as a summary report
    pub fn format_report(&self) -> String {
        let mut report = String::new();

        report.push_str("# Benchmark Results\n\n");

        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "| Metric | Value |\n\
             |--------|-------|\n\
             | Total Problems | {} |\n\
             | Completed | {} |\n\
             | Passed (First) | {} |\n\
             | Passed (Corrected) | {} |\n\
             | Failed | {} |\n\n",
            self.total_problems,
            self.completed_problems,
            self.passed_first_attempt,
            self.passed_with_correction,
            self.failed_problems
        ));

        report.push_str("## Success Rates\n\n");
        report.push_str(&format!(
            "| Category | Rate |\n\
             |----------|------|\n\
             | First Attempt | {:.1}% |\n\
             | Overall | {:.1}% |\n\
             | Easy Problems | {:.1}% |\n\
             | Hard Problems | {:.1}% |\n\n",
            self.first_attempt_rate,
            self.overall_success_rate,
            self.easy_success_rate,
            self.hard_success_rate
        ));

        report.push_str("## Resource Usage\n\n");
        report.push_str(&format!(
            "- Total Tokens: {}\n\
             - Total Time: {:.1}s\n\
             - Avg Iterations: {:.2}\n\n",
            self.total_tokens,
            self.total_time_ms as f64 / 1000.0,
            self.average_iterations
        ));

        if !self.model_usage.is_empty() {
            report.push_str("## Model Usage\n\n");
            report.push_str("| Model | Attempts |\n|-------|----------|\n");
            for (model, count) in &self.model_usage {
                report.push_str(&format!("| {} | {} |\n", model, count));
            }
            report.push('\n');
        }

        // Target comparison
        report.push_str("## Target Comparison\n\n");
        let targets = [
            ("First-attempt compilation", 65.0, self.first_attempt_rate),
            ("With correction loop", 85.0, self.overall_success_rate),
            ("Easy problems", 95.0, self.easy_success_rate),
            ("Hard problems", 75.0, self.hard_success_rate),
            ("Avg iterations to fix", 2.5, self.average_iterations),
        ];

        report.push_str("| Metric | Target | Actual | Status |\n");
        report.push_str("|--------|--------|--------|--------|\n");

        for (name, target, actual) in targets {
            let status = if name.contains("iterations") {
                if actual <= target {
                    "✅"
                } else {
                    "❌"
                }
            } else if actual >= target {
                "✅"
            } else {
                "❌"
            };

            if name.contains("iterations") {
                report.push_str(&format!(
                    "| {} | <{:.1} | {:.2} | {} |\n",
                    name, target, actual, status
                ));
            } else {
                report.push_str(&format!(
                    "| {} | {:.0}%+ | {:.1}% | {} |\n",
                    name, target, actual, status
                ));
            }
        }

        report
    }

    /// Check if all targets are met
    pub fn meets_targets(&self) -> bool {
        self.first_attempt_rate >= 65.0
            && self.overall_success_rate >= 85.0
            && self.easy_success_rate >= 95.0
            && self.hard_success_rate >= 75.0
            && self.average_iterations <= 2.5
    }

    /// Get metrics as JSON
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

/// Builder for tracking metrics during a benchmark run
pub struct MetricsTracker {
    /// Per-problem metrics
    problems: Vec<ProblemMetrics>,
    /// Session ID
    session_id: String,
}

impl MetricsTracker {
    /// Create a new tracker
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            problems: Vec::new(),
            session_id: session_id.into(),
        }
    }

    /// Add problem metrics
    pub fn add_problem(&mut self, metrics: ProblemMetrics) {
        self.problems.push(metrics);
    }

    /// Build aggregated metrics
    pub fn build(&self) -> BenchmarkMetrics {
        let total = self.problems.len();
        let completed = self
            .problems
            .iter()
            .filter(|p| p.status != ProblemStatus::Pending)
            .count();

        let passed_first = self
            .problems
            .iter()
            .filter(|p| p.status == ProblemStatus::PassedFirst)
            .count();

        let passed_corrected = self
            .problems
            .iter()
            .filter(|p| p.status == ProblemStatus::PassedCorrected)
            .count();

        let failed = self
            .problems
            .iter()
            .filter(|p| p.status == ProblemStatus::Failed)
            .count();

        let easy_total = self
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Easy && p.status != ProblemStatus::Pending)
            .count();

        let easy_passed = self
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Easy && p.succeeded())
            .count();

        let hard_total = self
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Hard && p.status != ProblemStatus::Pending)
            .count();

        let hard_passed = self
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Hard && p.succeeded())
            .count();

        let total_tokens: u64 = self.problems.iter().map(|p| p.total_tokens).sum();
        let total_time_ms: u64 = self.problems.iter().map(|p| p.total_time_ms).sum();
        let total_iterations: u32 = self.problems.iter().map(|p| p.total_iterations).sum();

        let avg_iterations = if completed > 0 {
            total_iterations as f32 / completed as f32
        } else {
            0.0
        };

        let mut model_usage = HashMap::new();
        for problem in &self.problems {
            if let Some(tier) = &problem.final_model_tier {
                *model_usage.entry(tier.clone()).or_insert(0) += 1;
            }
        }

        BenchmarkMetrics {
            session_id: self.session_id.clone(),
            total_problems: total,
            completed_problems: completed,
            passed_first_attempt: passed_first,
            passed_with_correction: passed_corrected,
            failed_problems: failed,
            first_attempt_rate: if completed > 0 {
                passed_first as f32 / completed as f32 * 100.0
            } else {
                0.0
            },
            overall_success_rate: if completed > 0 {
                (passed_first + passed_corrected) as f32 / completed as f32 * 100.0
            } else {
                0.0
            },
            easy_success_rate: if easy_total > 0 {
                easy_passed as f32 / easy_total as f32 * 100.0
            } else {
                0.0
            },
            hard_success_rate: if hard_total > 0 {
                hard_passed as f32 / hard_total as f32 * 100.0
            } else {
                0.0
            },
            total_tokens,
            total_time_ms,
            average_iterations: avg_iterations,
            model_usage,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attempt_metrics_artifact_tracking() {
        let attempt = AttemptMetrics::new(1, "worker".to_string(), 512, 1500, 3, 0, true)
            .with_generated_code("fn main() {}".to_string())
            .with_compiler_output(String::new());

        assert_eq!(attempt.attempt_number, 1);
        assert!(attempt.compiled);
        assert!(attempt.made_progress());
        assert_eq!(attempt.generated_code.as_deref(), Some("fn main() {}"));
        assert_eq!(attempt.compiler_output.as_deref(), Some(""));
    }

    #[test]
    fn test_attempt_metrics_artifact_fields_default_to_none() {
        let attempt = AttemptMetrics::new(2, "council".to_string(), 256, 800, 1, 1, false);
        assert!(attempt.generated_code.is_none());
        assert!(attempt.compiler_output.is_none());
        assert!(!attempt.made_progress());
    }

    #[test]
    fn test_problem_metrics() {
        let metrics = ProblemMetrics {
            problem_id: "test".to_string(),
            difficulty: Difficulty::Easy,
            status: ProblemStatus::PassedFirst,
            first_attempt_compiled: true,
            total_iterations: 1,
            total_tokens: 100,
            total_time_ms: 500,
            final_model_tier: Some("fast".to_string()),
        };

        assert!(metrics.succeeded());
        assert_eq!(metrics.tokens_per_iteration(), 100.0);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new("test-session");

        tracker.add_problem(ProblemMetrics {
            problem_id: "p1".to_string(),
            difficulty: Difficulty::Easy,
            status: ProblemStatus::PassedFirst,
            first_attempt_compiled: true,
            total_iterations: 1,
            total_tokens: 100,
            total_time_ms: 500,
            final_model_tier: Some("fast".to_string()),
        });

        tracker.add_problem(ProblemMetrics {
            problem_id: "p2".to_string(),
            difficulty: Difficulty::Hard,
            status: ProblemStatus::PassedCorrected,
            first_attempt_compiled: false,
            total_iterations: 3,
            total_tokens: 400,
            total_time_ms: 2000,
            final_model_tier: Some("specialized".to_string()),
        });

        let metrics = tracker.build();
        assert_eq!(metrics.total_problems, 2);
        assert_eq!(metrics.passed_first_attempt, 1);
        assert_eq!(metrics.passed_with_correction, 1);
        assert_eq!(metrics.overall_success_rate, 100.0);
    }

    #[test]
    fn test_report_format() {
        let metrics = BenchmarkMetrics {
            session_id: "test".to_string(),
            total_problems: 10,
            completed_problems: 10,
            passed_first_attempt: 7,
            passed_with_correction: 2,
            failed_problems: 1,
            first_attempt_rate: 70.0,
            overall_success_rate: 90.0,
            easy_success_rate: 100.0,
            hard_success_rate: 80.0,
            total_tokens: 5000,
            total_time_ms: 30000,
            average_iterations: 1.5,
            model_usage: HashMap::new(),
        };

        let report = metrics.format_report();
        assert!(report.contains("Benchmark Results"));
        assert!(report.contains("90.0%"));
        assert!(report.contains("✅"));
    }
}
