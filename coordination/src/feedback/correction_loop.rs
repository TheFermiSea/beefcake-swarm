//! Iterative correction loop for fixing compilation errors
//!
//! Implements a feedback loop that:
//! 1. Attempts compilation
//! 2. Parses errors
//! 3. Routes to appropriate model
//! 4. Applies fix
//! 5. Repeats until success or max iterations

use crate::feedback::compiler::{CompileResult, Compiler};
use crate::feedback::error_parser::{ErrorCategory, ErrorSummary, ParsedError, RustcErrorParser};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Configuration for the correction loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionConfig {
    /// Maximum correction iterations
    pub max_iterations: u32,
    /// Whether to escalate to more powerful models on failure
    pub enable_escalation: bool,
    /// Number of failures before escalating
    pub escalation_threshold: u32,
    /// Whether to focus on single errors at a time
    pub single_error_focus: bool,
    /// Whether to use clippy in addition to check
    pub use_clippy: bool,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5,
            enable_escalation: true,
            escalation_threshold: 2,
            single_error_focus: true,
            use_clippy: false, // Start with just cargo check
        }
    }
}

/// Result of a single correction attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttemptResult {
    /// Attempt number (1-indexed)
    pub attempt: u32,
    /// Timestamp of attempt
    pub timestamp: DateTime<Utc>,
    /// Whether compilation succeeded
    pub compiled: bool,
    /// Number of errors before this attempt
    pub errors_before: usize,
    /// Number of errors after this attempt
    pub errors_after: usize,
    /// Primary error category addressed
    pub error_category: Option<ErrorCategory>,
    /// Model tier used for fix
    pub model_tier: String,
    /// The fix that was applied (code diff or description)
    pub fix_applied: Option<String>,
    /// Error message if attempt failed
    pub error_message: Option<String>,
}

/// Result of the complete correction loop
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionResult {
    /// Whether the code compiles after correction
    pub success: bool,
    /// Total iterations used
    pub iterations: u32,
    /// All attempt results
    pub attempts: Vec<AttemptResult>,
    /// Final compile result
    pub final_compile: Option<CompileResultSummary>,
    /// Total time spent
    pub duration_ms: u64,
    /// Final error summary (if not successful)
    pub remaining_errors: Option<ErrorSummary>,
}

impl CorrectionResult {
    /// Create a successful result with no correction needed
    pub fn already_compiles() -> Self {
        Self {
            success: true,
            iterations: 0,
            attempts: vec![],
            final_compile: Some(CompileResultSummary {
                success: true,
                error_count: 0,
                warning_count: 0,
            }),
            duration_ms: 0,
            remaining_errors: None,
        }
    }
}

/// Summarized compile result for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompileResultSummary {
    /// Whether compilation succeeded
    pub success: bool,
    /// Number of errors
    pub error_count: usize,
    /// Number of warnings
    pub warning_count: usize,
}

impl From<&CompileResult> for CompileResultSummary {
    fn from(result: &CompileResult) -> Self {
        Self {
            success: result.success,
            error_count: result.error_count(),
            warning_count: result.warnings().len(),
        }
    }
}

/// Model tier for escalation (local models)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelTier {
    /// Worker model (HydraCoder 30B-A3B MoE)
    Worker,
    /// Manager Council (Opus 4.5, Gemini 3 Pro, Qwen3.5)
    Council,
}

impl ModelTier {
    /// Get the model name for this tier
    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Worker => "HydraCoder-Q6_K",
            Self::Council => "manager-council",
        }
    }

    /// Escalate to next tier
    pub fn escalate(&self) -> Self {
        match self {
            Self::Worker => Self::Council,
            Self::Council => Self::Council, // Already at max
        }
    }

    /// Parse tier from string
    pub fn parse_tier(s: &str) -> Self {
        match s {
            "worker" => Self::Worker,
            "council" => Self::Council,
            _ => Self::Worker,
        }
    }
}

impl std::fmt::Display for ModelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Worker => write!(f, "worker"),
            Self::Council => write!(f, "council"),
        }
    }
}

/// Escalation tier for the full correction pipeline
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationTier {
    /// Worker tier (HydraCoder) — 2 attempts max
    Worker,
    /// Manager Council (Opus 4.5, Gemini 3 Pro, Qwen3.5) — final escalation
    ManagerCouncil,
}

impl EscalationTier {
    /// Maximum attempts allowed at this tier
    pub fn max_attempts(&self) -> u32 {
        match self {
            Self::Worker => 2,
            Self::ManagerCouncil => 1,
        }
    }

    /// Escalate to next tier
    pub fn escalate(&self) -> Self {
        match self {
            Self::Worker => Self::ManagerCouncil,
            Self::ManagerCouncil => Self::ManagerCouncil, // Already at max
        }
    }

    /// Check if this is the final tier
    pub fn is_final(&self) -> bool {
        matches!(self, Self::ManagerCouncil)
    }

    /// Get the corresponding local ModelTier (None for ManagerCouncil)
    pub fn local_tier(&self) -> Option<ModelTier> {
        match self {
            Self::Worker => Some(ModelTier::Worker),
            Self::ManagerCouncil => None,
        }
    }
}

impl std::fmt::Display for EscalationTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Worker => write!(f, "worker"),
            Self::ManagerCouncil => write!(f, "manager_council"),
        }
    }
}

/// Triggers for early escalation (before max attempts reached)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EscalationTrigger {
    /// No progress for N consecutive attempts
    NoProgress { attempts: u32 },
    /// Linker or environment error (not fixable by code changes)
    LinkerError,
    /// Wall-clock timeout exceeded
    Timeout { minutes: u32 },
    /// Explicit escalation request
    Explicit { reason: String },
}

impl EscalationTrigger {
    /// Check if this trigger should cause escalation
    pub fn should_escalate(&self, context: &TieredCorrectionContext) -> bool {
        match self {
            Self::NoProgress { attempts } => context.consecutive_no_progress >= *attempts,
            Self::LinkerError => context.last_error_is_linker,
            Self::Timeout { minutes } => context.wall_clock_minutes >= *minutes,
            Self::Explicit { .. } => true,
        }
    }
}

/// Context for tiered correction loop
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TieredCorrectionContext {
    /// Consecutive attempts with no error reduction
    pub consecutive_no_progress: u32,
    /// Whether the last error was a linker/environment error
    pub last_error_is_linker: bool,
    /// Wall-clock time in minutes
    pub wall_clock_minutes: u32,
    /// Previous error count for progress tracking
    pub previous_error_count: usize,
}

/// Tiered correction loop with local and cloud escalation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TieredCorrectionLoop {
    /// Current escalation tier
    current_tier: EscalationTier,
    /// Attempts at current tier
    attempts_at_tier: u32,
    /// Total attempts across all tiers
    total_attempts: u32,
    /// Context for escalation decisions
    context: TieredCorrectionContext,
    /// Early escalation triggers
    triggers: Vec<EscalationTrigger>,
    /// Start time for wall-clock tracking
    #[serde(skip)]
    start_time: Option<std::time::Instant>,
}

impl Default for TieredCorrectionLoop {
    fn default() -> Self {
        Self::new()
    }
}

impl TieredCorrectionLoop {
    /// Create a new tiered correction loop
    pub fn new() -> Self {
        Self {
            current_tier: EscalationTier::Worker,
            attempts_at_tier: 0,
            total_attempts: 0,
            context: TieredCorrectionContext::default(),
            triggers: vec![
                EscalationTrigger::NoProgress { attempts: 2 },
                EscalationTrigger::LinkerError,
                EscalationTrigger::Timeout { minutes: 8 },
            ],
            start_time: None,
        }
    }

    /// Start the correction loop (call before first attempt)
    pub fn start(&mut self) {
        self.start_time = Some(std::time::Instant::now());
    }

    /// Get current escalation tier
    pub fn current_tier(&self) -> EscalationTier {
        self.current_tier
    }

    /// Get attempts at current tier
    pub fn attempts_at_tier(&self) -> u32 {
        self.attempts_at_tier
    }

    /// Get total attempts across all tiers
    pub fn total_attempts(&self) -> u32 {
        self.total_attempts
    }

    /// Check if we should escalate based on triggers or attempt limits
    pub fn should_escalate(&self) -> bool {
        // Already at final tier
        if self.current_tier.is_final() {
            return false;
        }

        // Max attempts at current tier
        if self.attempts_at_tier >= self.current_tier.max_attempts() {
            return true;
        }

        // Check early triggers
        for trigger in &self.triggers {
            if trigger.should_escalate(&self.context) {
                return true;
            }
        }

        false
    }

    /// Escalate to next tier
    pub fn escalate(&mut self) -> EscalationTier {
        let old_tier = self.current_tier;
        self.current_tier = self.current_tier.escalate();
        self.attempts_at_tier = 0;
        self.context.consecutive_no_progress = 0;

        tracing::info!(
            "Escalating from {} to {} (total attempts: {})",
            old_tier,
            self.current_tier,
            self.total_attempts
        );

        self.current_tier
    }

    /// Record an attempt result
    pub fn record_attempt(&mut self, error_count: usize, is_linker_error: bool) {
        self.attempts_at_tier += 1;
        self.total_attempts += 1;

        // Track progress
        if error_count >= self.context.previous_error_count && self.context.previous_error_count > 0
        {
            self.context.consecutive_no_progress += 1;
        } else {
            self.context.consecutive_no_progress = 0;
        }
        self.context.previous_error_count = error_count;
        self.context.last_error_is_linker = is_linker_error;

        // Update wall clock
        if let Some(start) = self.start_time {
            self.context.wall_clock_minutes = (start.elapsed().as_secs() / 60) as u32;
        }
    }

    /// Check if we can continue (not exhausted)
    pub fn can_continue(&self) -> bool {
        // Final tier with attempts exhausted
        if self.current_tier.is_final() && self.attempts_at_tier >= self.current_tier.max_attempts()
        {
            return false;
        }
        true
    }

    /// Get remaining budget at current tier
    pub fn remaining_at_tier(&self) -> u32 {
        self.current_tier
            .max_attempts()
            .saturating_sub(self.attempts_at_tier)
    }

    /// Get summary for logging/display
    pub fn summary(&self) -> String {
        format!(
            "Tier: {}, Attempt: {}/{}, Total: {}, No-progress streak: {}",
            self.current_tier,
            self.attempts_at_tier,
            self.current_tier.max_attempts(),
            self.total_attempts,
            self.context.consecutive_no_progress
        )
    }
}

/// The correction loop controller
pub struct CorrectionLoop {
    /// Compiler instance
    compiler: Compiler,
    /// Configuration
    config: CorrectionConfig,
    /// Current model tier
    current_tier: ModelTier,
    /// Consecutive failures at current tier
    failures_at_tier: u32,
}

impl CorrectionLoop {
    /// Create a new correction loop for a crate directory
    pub fn new(crate_dir: impl AsRef<Path>, config: CorrectionConfig) -> Self {
        Self {
            compiler: Compiler::new(crate_dir),
            config,
            current_tier: ModelTier::Worker,
            failures_at_tier: 0,
        }
    }

    /// Check if code compiles (initial check)
    pub fn check_compiles(&self) -> CompileResult {
        self.compiler.check()
    }

    /// Get the current model tier
    pub fn current_model(&self) -> ModelTier {
        self.current_tier
    }

    /// Get the recommended model for a set of errors
    pub fn recommend_model(&self, errors: &[ParsedError]) -> ModelTier {
        let summary = RustcErrorParser::summarize(errors);
        let base_tier = ModelTier::parse_tier(summary.recommended_tier());

        // Consider escalation from failures
        if self.failures_at_tier >= self.config.escalation_threshold {
            return self.current_tier.escalate();
        }

        // Use base recommendation if it's higher than current
        match (base_tier, self.current_tier) {
            (ModelTier::Council, _) => ModelTier::Council,
            _ => self.current_tier,
        }
    }

    /// Record an attempt result and update state
    pub fn record_attempt(&mut self, attempt: &AttemptResult) {
        if attempt.compiled || attempt.errors_after < attempt.errors_before {
            // Success or progress - reset failure counter
            self.failures_at_tier = 0;
        } else {
            // No progress - increment failure counter
            self.failures_at_tier += 1;

            // Escalate if threshold reached
            if self.config.enable_escalation
                && self.failures_at_tier >= self.config.escalation_threshold
            {
                self.current_tier = self.current_tier.escalate();
                self.failures_at_tier = 0;
            }
        }
    }

    /// Get the errors to focus on (respects single_error_focus config)
    pub fn get_focus_errors<'a>(&self, all_errors: &'a [ParsedError]) -> Vec<&'a ParsedError> {
        if self.config.single_error_focus && !all_errors.is_empty() {
            vec![&all_errors[0]]
        } else {
            all_errors.iter().collect()
        }
    }

    /// Format a fix prompt for the model
    pub fn format_fix_prompt(
        &self,
        code: &str,
        errors: &[&ParsedError],
        context: Option<&str>,
    ) -> String {
        let mut prompt = String::new();

        prompt.push_str("Fix the following Rust compilation error(s).\n\n");

        if let Some(ctx) = context {
            prompt.push_str(&format!("Context: {}\n\n", ctx));
        }

        prompt.push_str("## Current Code\n\n```rust\n");
        prompt.push_str(code);
        prompt.push_str("\n```\n\n");

        prompt.push_str("## Compilation Error(s)\n\n");

        for (i, error) in errors.iter().enumerate() {
            if errors.len() > 1 {
                prompt.push_str(&format!("### Error {} ({:?})\n\n", i + 1, error.category));
            }
            prompt.push_str(&error.format_for_fix_prompt());
            prompt.push_str("\n\n");
        }

        prompt.push_str("## Instructions\n\n");
        prompt.push_str("1. Fix the error(s) above\n");
        prompt.push_str("2. Return ONLY the corrected Rust code\n");
        prompt.push_str("3. Do not include explanations or markdown formatting\n");
        prompt.push_str("4. Ensure the code compiles cleanly\n");

        if errors.iter().any(|e| e.category == ErrorCategory::Lifetime) {
            prompt.push_str("\nNote: This involves lifetime annotations. Consider:\n");
            prompt.push_str("- Adding explicit lifetime parameters\n");
            prompt.push_str("- Using owned types instead of references\n");
            prompt.push_str("- Cloning data if appropriate\n");
        }

        if errors
            .iter()
            .any(|e| e.category == ErrorCategory::BorrowChecker)
        {
            prompt.push_str("\nNote: This involves borrow checking. Consider:\n");
            prompt.push_str("- Splitting borrows across scopes\n");
            prompt.push_str("- Using clone() to avoid borrow conflicts\n");
            prompt.push_str("- Restructuring to avoid simultaneous borrows\n");
        }

        prompt
    }

    /// Parse errors from the last compile result
    pub fn parse_errors(&self, result: &CompileResult) -> Vec<ParsedError> {
        RustcErrorParser::parse_cargo_messages(&result.messages)
    }

    /// Create an attempt result from a compile result
    pub fn create_attempt_result(
        &self,
        attempt_num: u32,
        errors_before: usize,
        compile_result: &CompileResult,
        model_tier: ModelTier,
        fix_applied: Option<String>,
    ) -> AttemptResult {
        let errors = self.parse_errors(compile_result);
        let error_category = errors.first().map(|e| e.category);

        AttemptResult {
            attempt: attempt_num,
            timestamp: Utc::now(),
            compiled: compile_result.success,
            errors_before,
            errors_after: errors.len(),
            error_category,
            model_tier: model_tier.to_string(),
            fix_applied,
            error_message: if compile_result.success {
                None
            } else {
                Some(compile_result.format_for_llm())
            },
        }
    }
}

/// Builder for correction loop results
pub struct CorrectionResultBuilder {
    start_time: std::time::Instant,
    attempts: Vec<AttemptResult>,
}

impl CorrectionResultBuilder {
    /// Start building a result
    pub fn new() -> Self {
        Self {
            start_time: std::time::Instant::now(),
            attempts: Vec::new(),
        }
    }

    /// Add an attempt
    pub fn add_attempt(&mut self, attempt: AttemptResult) {
        self.attempts.push(attempt);
    }

    /// Build the final result
    pub fn build(self, final_compile: &CompileResult) -> CorrectionResult {
        let duration_ms = self.start_time.elapsed().as_millis() as u64;
        let errors = RustcErrorParser::parse_cargo_messages(&final_compile.messages);

        CorrectionResult {
            success: final_compile.success,
            iterations: self.attempts.len() as u32,
            attempts: self.attempts,
            final_compile: Some(CompileResultSummary::from(final_compile)),
            duration_ms,
            remaining_errors: if final_compile.success {
                None
            } else {
                Some(RustcErrorParser::summarize(&errors))
            },
        }
    }
}

impl Default for CorrectionResultBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_tier_escalation() {
        assert_eq!(ModelTier::Worker.escalate(), ModelTier::Council);
        assert_eq!(ModelTier::Council.escalate(), ModelTier::Council);
    }

    #[test]
    fn test_correction_config_default() {
        let config = CorrectionConfig::default();
        assert_eq!(config.max_iterations, 5);
        assert!(config.enable_escalation);
        assert_eq!(config.escalation_threshold, 2);
    }

    #[test]
    fn test_fix_prompt_format() {
        let loop_ = CorrectionLoop::new("/tmp", CorrectionConfig::default());

        let error = ParsedError {
            category: ErrorCategory::TypeMismatch,
            code: Some("E0308".to_string()),
            message: "mismatched types".to_string(),
            file: Some("src/main.rs".to_string()),
            line: Some(10),
            column: Some(5),
            suggestion: None,
            rendered: "error[E0308]: mismatched types".to_string(),
            labels: vec!["expected i32, found &str".to_string()],
        };

        let prompt = loop_.format_fix_prompt("fn main() {}", &[&error], None);

        assert!(prompt.contains("Fix the following"));
        assert!(prompt.contains("E0308"));
        assert!(prompt.contains("fn main()"));
    }

    #[test]
    fn test_already_compiles() {
        let result = CorrectionResult::already_compiles();
        assert!(result.success);
        assert_eq!(result.iterations, 0);
    }
}
