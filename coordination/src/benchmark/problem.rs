//! Benchmark problem types and session management
//!
//! Handles loading problems, tracking attempts, and managing benchmark sessions.

use crate::benchmark::metrics::{BenchmarkMetrics, ProblemMetrics};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use uuid::Uuid;

/// Problem difficulty level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Difficulty {
    /// Easy problems - straightforward implementations
    Easy,
    /// Hard problems - complex patterns, edge cases
    Hard,
}

impl std::fmt::Display for Difficulty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Easy => write!(f, "easy"),
            Self::Hard => write!(f, "hard"),
        }
    }
}

/// Status of a benchmark problem
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProblemStatus {
    /// Not yet attempted
    Pending,
    /// Currently being worked on
    InProgress,
    /// Compiled on first attempt
    PassedFirst,
    /// Compiled after correction loop
    PassedCorrected,
    /// Failed all correction attempts
    Failed,
    /// Skipped (e.g., due to timeout)
    Skipped,
}

impl std::fmt::Display for ProblemStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::InProgress => write!(f, "in_progress"),
            Self::PassedFirst => write!(f, "passed_first"),
            Self::PassedCorrected => write!(f, "passed_corrected"),
            Self::Failed => write!(f, "failed"),
            Self::Skipped => write!(f, "skipped"),
        }
    }
}

/// A benchmark problem from rust-bench
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkProblem {
    /// Unique identifier (from rust-bench)
    pub id: String,
    /// Problem difficulty
    pub difficulty: Difficulty,
    /// Problem description/prompt
    pub description: String,
    /// Expected function signature
    pub function_signature: String,
    /// Test code for verification (if available)
    pub test_code: Option<String>,
    /// Current status
    pub status: ProblemStatus,
    /// First attempt result
    pub first_attempt: Option<AttemptResult>,
    /// Correction loop result (if first attempt failed)
    pub correction_result: Option<CorrectionAttemptResult>,
    /// Problem metrics
    pub metrics: Option<ProblemMetrics>,
}

impl BenchmarkProblem {
    /// Create a new problem
    pub fn new(
        id: impl Into<String>,
        difficulty: Difficulty,
        description: impl Into<String>,
        signature: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            difficulty,
            description: description.into(),
            function_signature: signature.into(),
            test_code: None,
            status: ProblemStatus::Pending,
            first_attempt: None,
            correction_result: None,
            metrics: None,
        }
    }

    /// Set test code
    pub fn with_test_code(mut self, test: impl Into<String>) -> Self {
        self.test_code = Some(test.into());
        self
    }

    /// Check if problem is complete (passed or failed)
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            ProblemStatus::PassedFirst
                | ProblemStatus::PassedCorrected
                | ProblemStatus::Failed
                | ProblemStatus::Skipped
        )
    }

    /// Check if problem compiled successfully
    pub fn compiled(&self) -> bool {
        matches!(
            self.status,
            ProblemStatus::PassedFirst | ProblemStatus::PassedCorrected
        )
    }

    /// Get total iterations used
    pub fn total_iterations(&self) -> u32 {
        let first = if self.first_attempt.is_some() { 1 } else { 0 };
        let correction = self
            .correction_result
            .as_ref()
            .map(|r| r.iterations)
            .unwrap_or(0);
        first + correction
    }

    /// Record first attempt result
    pub fn record_first_attempt(&mut self, result: AttemptResult) {
        self.status = if result.compiled {
            ProblemStatus::PassedFirst
        } else {
            ProblemStatus::InProgress
        };
        self.first_attempt = Some(result);
    }

    /// Record correction loop result
    pub fn record_correction(&mut self, result: CorrectionAttemptResult) {
        self.status = if result.compiled {
            ProblemStatus::PassedCorrected
        } else {
            ProblemStatus::Failed
        };
        self.correction_result = Some(result);
    }

    /// Calculate and store metrics
    pub fn calculate_metrics(&mut self) {
        let mut total_tokens = 0u64;
        let mut total_time_ms = 0u64;

        if let Some(first) = &self.first_attempt {
            total_tokens += first.tokens_used as u64;
            total_time_ms += first.duration_ms;
        }

        if let Some(correction) = &self.correction_result {
            total_tokens += correction.total_tokens as u64;
            total_time_ms += correction.duration_ms;
        }

        self.metrics = Some(ProblemMetrics {
            problem_id: self.id.clone(),
            difficulty: self.difficulty,
            status: self.status,
            first_attempt_compiled: self
                .first_attempt
                .as_ref()
                .map(|a| a.compiled)
                .unwrap_or(false),
            total_iterations: self.total_iterations(),
            total_tokens,
            total_time_ms,
            final_model_tier: self
                .correction_result
                .as_ref()
                .map(|c| c.final_model_tier.clone())
                .or_else(|| self.first_attempt.as_ref().map(|a| a.model_tier.clone())),
        });
    }
}

/// Result of a single attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptResult {
    /// Timestamp of attempt
    pub timestamp: DateTime<Utc>,
    /// Whether the code compiled
    pub compiled: bool,
    /// Error count (0 if compiled)
    pub error_count: usize,
    /// Model tier used
    pub model_tier: String,
    /// Tokens used in this attempt
    pub tokens_used: u32,
    /// Time taken in milliseconds
    pub duration_ms: u64,
    /// Generated code
    pub code: String,
    /// Error message if compilation failed
    pub error_message: Option<String>,
}

impl AttemptResult {
    /// Create a successful attempt
    pub fn success(
        model_tier: impl Into<String>,
        code: String,
        tokens: u32,
        duration_ms: u64,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            compiled: true,
            error_count: 0,
            model_tier: model_tier.into(),
            tokens_used: tokens,
            duration_ms,
            code,
            error_message: None,
        }
    }

    /// Create a failed attempt
    pub fn failure(
        model_tier: impl Into<String>,
        code: String,
        error_count: usize,
        error_message: String,
        tokens: u32,
        duration_ms: u64,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            compiled: false,
            error_count,
            model_tier: model_tier.into(),
            tokens_used: tokens,
            duration_ms,
            code,
            error_message: Some(error_message),
        }
    }
}

/// Result of the correction loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionAttemptResult {
    /// Whether correction succeeded
    pub compiled: bool,
    /// Number of iterations used
    pub iterations: u32,
    /// Maximum iterations allowed
    pub max_iterations: u32,
    /// Total tokens used across all iterations
    pub total_tokens: u32,
    /// Total time in milliseconds
    pub duration_ms: u64,
    /// Final code (whether compiled or not)
    pub final_code: String,
    /// Final model tier used
    pub final_model_tier: String,
    /// History of each correction attempt
    pub attempt_history: Vec<AttemptResult>,
}

/// Configuration for benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Maximum iterations for correction loop
    pub max_correction_iterations: u32,
    /// Enable model escalation
    pub enable_escalation: bool,
    /// Escalation threshold (failures before escalating)
    pub escalation_threshold: u32,
    /// Problem timeout in seconds
    pub timeout_seconds: u64,
    /// Whether to run all problems or just a subset
    pub subset: Option<ProblemSubset>,
    /// Working directory for temporary crate
    pub work_dir: PathBuf,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            max_correction_iterations: 5,
            enable_escalation: true,
            escalation_threshold: 2,
            timeout_seconds: 300,
            subset: None,
            work_dir: std::env::temp_dir().join("rust-bench-harness"),
        }
    }
}

/// Subset specification for partial benchmark runs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProblemSubset {
    /// Only run problems with these IDs
    pub problem_ids: Option<Vec<String>>,
    /// Only run problems of this difficulty
    pub difficulty: Option<Difficulty>,
    /// Maximum number of problems to run
    pub limit: Option<usize>,
}

/// Active benchmark session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkSession {
    /// Unique session ID
    pub id: String,
    /// Session start time
    pub started_at: DateTime<Utc>,
    /// Session configuration
    pub config: BenchmarkConfig,
    /// All problems in this benchmark
    pub problems: Vec<BenchmarkProblem>,
    /// Index of current problem
    pub current_index: usize,
    /// Session status
    pub status: SessionStatus,
    /// Aggregated metrics
    pub metrics: Option<BenchmarkMetrics>,
}

/// Session status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Session is active
    Active,
    /// Session completed successfully
    Completed,
    /// Session was aborted
    Aborted,
    /// Session failed with error
    Failed,
}

impl BenchmarkSession {
    /// Create a new session
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            started_at: Utc::now(),
            config,
            problems: Vec::new(),
            current_index: 0,
            status: SessionStatus::Active,
            metrics: None,
        }
    }

    /// Load problems from rust-bench directory
    pub fn load_problems(&mut self, bench_dir: impl AsRef<Path>) -> Result<usize, std::io::Error> {
        let bench_path = bench_dir.as_ref();

        // Look for problems in the standard rust-bench structure
        let problems_dir = bench_path.join("problems");
        if !problems_dir.exists() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Problems directory not found: {:?}", problems_dir),
            ));
        }

        // Parse problems (simplified - real implementation would parse TOML/JSON)
        let mut problems = Vec::new();

        for entry in std::fs::read_dir(&problems_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(problem) = self.parse_problem_dir(&path)? {
                    problems.push(problem);
                }
            }
        }

        // Apply subset filter if configured
        if let Some(subset) = &self.config.subset {
            problems = self.apply_subset_filter(problems, subset);
        }

        let count = problems.len();
        self.problems = problems;
        Ok(count)
    }

    /// Parse a problem directory
    fn parse_problem_dir(&self, path: &Path) -> Result<Option<BenchmarkProblem>, std::io::Error> {
        let id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Look for description file
        let desc_path = path.join("description.md");
        let description = if desc_path.exists() {
            std::fs::read_to_string(&desc_path)?
        } else {
            // Try alternative names
            let prompt_path = path.join("prompt.txt");
            if prompt_path.exists() {
                std::fs::read_to_string(&prompt_path)?
            } else {
                return Ok(None); // Skip problems without description
            }
        };

        // Look for signature file or extract from description
        let sig_path = path.join("signature.rs");
        let signature = if sig_path.exists() {
            std::fs::read_to_string(&sig_path)?
        } else {
            self.extract_signature(&description)
        };

        // Determine difficulty from path or metadata
        let difficulty = if id.contains("hard") || path.join("hard").exists() {
            Difficulty::Hard
        } else {
            Difficulty::Easy
        };

        // Look for test code
        let test_path = path.join("test.rs");
        let test_code = if test_path.exists() {
            Some(std::fs::read_to_string(&test_path)?)
        } else {
            None
        };

        let mut problem = BenchmarkProblem::new(id, difficulty, description, signature);
        if let Some(test) = test_code {
            problem = problem.with_test_code(test);
        }

        Ok(Some(problem))
    }

    /// Extract function signature from description
    fn extract_signature(&self, description: &str) -> String {
        // Look for code blocks with fn
        for line in description.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("fn ") && trimmed.contains("->") {
                return trimmed.to_string();
            }
        }

        // Default to placeholder
        "fn solution() -> ()".to_string()
    }

    /// Apply subset filter to problems
    fn apply_subset_filter(
        &self,
        mut problems: Vec<BenchmarkProblem>,
        subset: &ProblemSubset,
    ) -> Vec<BenchmarkProblem> {
        // Filter by IDs
        if let Some(ids) = &subset.problem_ids {
            problems.retain(|p| ids.contains(&p.id));
        }

        // Filter by difficulty
        if let Some(diff) = subset.difficulty {
            problems.retain(|p| p.difficulty == diff);
        }

        // Apply limit
        if let Some(limit) = subset.limit {
            problems.truncate(limit);
        }

        problems
    }

    /// Add problems manually (for testing or custom sets)
    pub fn add_problem(&mut self, problem: BenchmarkProblem) {
        self.problems.push(problem);
    }

    /// Get the next pending problem
    pub fn next_problem(&self) -> Option<&BenchmarkProblem> {
        self.problems
            .iter()
            .skip(self.current_index)
            .find(|p| p.status == ProblemStatus::Pending)
    }

    /// Get the current problem
    pub fn current_problem(&self) -> Option<&BenchmarkProblem> {
        self.problems.get(self.current_index)
    }

    /// Get mutable reference to current problem
    pub fn current_problem_mut(&mut self) -> Option<&mut BenchmarkProblem> {
        self.problems.get_mut(self.current_index)
    }

    /// Advance to next problem
    pub fn advance(&mut self) -> bool {
        if self.current_index + 1 < self.problems.len() {
            self.current_index += 1;
            true
        } else {
            self.status = SessionStatus::Completed;
            false
        }
    }

    /// Calculate aggregated metrics
    pub fn calculate_metrics(&mut self) {
        // Ensure all problems have metrics
        for problem in &mut self.problems {
            if problem.metrics.is_none() {
                problem.calculate_metrics();
            }
        }

        let total = self.problems.len();
        let completed = self.problems.iter().filter(|p| p.is_complete()).count();
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

        let easy_passed = self
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Easy && p.compiled())
            .count();
        let easy_total = self
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Easy && p.is_complete())
            .count();

        let hard_passed = self
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Hard && p.compiled())
            .count();
        let hard_total = self
            .problems
            .iter()
            .filter(|p| p.difficulty == Difficulty::Hard && p.is_complete())
            .count();

        let total_tokens: u64 = self
            .problems
            .iter()
            .filter_map(|p| p.metrics.as_ref())
            .map(|m| m.total_tokens)
            .sum();

        let total_time_ms: u64 = self
            .problems
            .iter()
            .filter_map(|p| p.metrics.as_ref())
            .map(|m| m.total_time_ms)
            .sum();

        let total_iterations: u32 = self.problems.iter().map(|p| p.total_iterations()).sum();

        let avg_iterations = if completed > 0 {
            total_iterations as f32 / completed as f32
        } else {
            0.0
        };

        self.metrics = Some(BenchmarkMetrics {
            session_id: self.id.clone(),
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
            model_usage: self.calculate_model_usage(),
        });
    }

    /// Calculate model usage statistics
    fn calculate_model_usage(&self) -> HashMap<String, u32> {
        let mut usage = HashMap::new();

        for problem in &self.problems {
            if let Some(first) = &problem.first_attempt {
                *usage.entry(first.model_tier.clone()).or_insert(0) += 1;
            }
            if let Some(correction) = &problem.correction_result {
                for attempt in &correction.attempt_history {
                    *usage.entry(attempt.model_tier.clone()).or_insert(0) += 1;
                }
            }
        }

        usage
    }

    /// Get progress summary
    pub fn progress_summary(&self) -> String {
        let completed = self.problems.iter().filter(|p| p.is_complete()).count();
        let passed = self.problems.iter().filter(|p| p.compiled()).count();

        format!(
            "Progress: {}/{} completed, {}/{} passed ({:.1}%)",
            completed,
            self.problems.len(),
            passed,
            completed,
            if completed > 0 {
                passed as f32 / completed as f32 * 100.0
            } else {
                0.0
            }
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_problem_creation() {
        let problem = BenchmarkProblem::new(
            "test-1",
            Difficulty::Easy,
            "Write a function to add two numbers",
            "fn add(a: i32, b: i32) -> i32",
        );

        assert_eq!(problem.id, "test-1");
        assert_eq!(problem.difficulty, Difficulty::Easy);
        assert_eq!(problem.status, ProblemStatus::Pending);
        assert!(!problem.is_complete());
    }

    #[test]
    fn test_session_creation() {
        let session = BenchmarkSession::new(BenchmarkConfig::default());
        assert_eq!(session.status, SessionStatus::Active);
        assert!(session.problems.is_empty());
    }

    #[test]
    fn test_problem_status_transitions() {
        let mut problem = BenchmarkProblem::new(
            "test-2",
            Difficulty::Hard,
            "Complex problem",
            "fn complex() -> ()",
        );

        // First attempt fails
        problem.record_first_attempt(AttemptResult::failure(
            "fast",
            "fn complex() {}".to_string(),
            2,
            "error".to_string(),
            100,
            500,
        ));
        assert_eq!(problem.status, ProblemStatus::InProgress);

        // Correction succeeds
        problem.record_correction(CorrectionAttemptResult {
            compiled: true,
            iterations: 2,
            max_iterations: 5,
            total_tokens: 500,
            duration_ms: 2000,
            final_code: "fn complex() -> () {}".to_string(),
            final_model_tier: "specialized".to_string(),
            attempt_history: vec![],
        });
        assert_eq!(problem.status, ProblemStatus::PassedCorrected);
        assert!(problem.compiled());
    }

    #[test]
    fn test_metrics_calculation() {
        let mut session = BenchmarkSession::new(BenchmarkConfig::default());

        let mut p1 = BenchmarkProblem::new("p1", Difficulty::Easy, "desc", "sig");
        p1.record_first_attempt(AttemptResult::success("fast", "code".to_string(), 100, 500));

        let mut p2 = BenchmarkProblem::new("p2", Difficulty::Hard, "desc", "sig");
        p2.record_first_attempt(AttemptResult::failure(
            "fast",
            "code".to_string(),
            1,
            "err".to_string(),
            100,
            500,
        ));
        p2.record_correction(CorrectionAttemptResult {
            compiled: true,
            iterations: 2,
            max_iterations: 5,
            total_tokens: 300,
            duration_ms: 1500,
            final_code: "fixed".to_string(),
            final_model_tier: "specialized".to_string(),
            attempt_history: vec![],
        });

        session.add_problem(p1);
        session.add_problem(p2);
        session.calculate_metrics();

        let metrics = session.metrics.unwrap();
        assert_eq!(metrics.total_problems, 2);
        assert_eq!(metrics.passed_first_attempt, 1);
        assert_eq!(metrics.passed_with_correction, 1);
        assert_eq!(metrics.overall_success_rate, 100.0);
    }
}
