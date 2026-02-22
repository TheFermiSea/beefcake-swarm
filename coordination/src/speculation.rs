//! Multi-Proposal Speculative Execution
//!
//! For hard/risky tasks, a single implementer often gets stuck in local
//! minima. This module provides the planning and selection logic for
//! spawning multiple implementations in parallel with different strategies.
//!
//! # Flow
//!
//! ```text
//! Task Assessment → SpeculativePlanner → Vec<ProposalSpec>
//!                                              ↓
//!                                    [Orchestrator executes each]
//!                                              ↓
//!                                    Vec<ProposalResult>
//!                                              ↓
//!                                    ProposalSelector → Winner
//! ```
//!
//! # Strategy Diversity
//!
//! Proposals are diversified along three axes:
//! - **Temperature**: Conservative (0.2) vs creative (0.8)
//! - **Approach**: Minimal fix vs refactor for clarity
//! - **Model tier**: Fast (14B) vs deep (72B) vs cloud
//!
//! # Selection Criteria
//!
//! When multiple proposals pass the verifier, the selector ranks by:
//! 1. Verifier pass (hard gate)
//! 2. Fewest remaining warnings
//! 3. Smallest diff (most concise change)
//! 4. Lowest token cost

use crate::escalation::worker_first::TaskComplexity;
use serde::{Deserialize, Serialize};

/// Configuration for speculative execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativeConfig {
    /// Minimum task complexity to trigger speculation.
    pub min_complexity: TaskComplexity,
    /// Maximum number of parallel proposals.
    pub max_proposals: usize,
    /// Maximum total token budget across all proposals.
    pub token_budget: u64,
    /// Whether speculation is enabled at all.
    pub enabled: bool,
}

impl Default for SpeculativeConfig {
    fn default() -> Self {
        Self {
            min_complexity: TaskComplexity::Complex,
            max_proposals: 3,
            token_budget: 200_000,
            enabled: false,
        }
    }
}

impl SpeculativeConfig {
    /// Read from environment variables.
    pub fn from_env() -> Self {
        let enabled = std::env::var("SWARM_SPECULATION_ENABLED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let max_proposals = std::env::var("SWARM_SPECULATION_MAX_PROPOSALS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);
        let token_budget = std::env::var("SWARM_SPECULATION_TOKEN_BUDGET")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200_000);

        Self {
            min_complexity: TaskComplexity::Complex,
            max_proposals,
            token_budget,
            enabled,
        }
    }

    /// Whether speculation should activate for a given complexity.
    pub fn should_speculate(&self, complexity: TaskComplexity) -> bool {
        if !self.enabled {
            return false;
        }
        matches!(
            (complexity, self.min_complexity),
            (TaskComplexity::Complex, _) | (TaskComplexity::Medium, TaskComplexity::Medium)
        )
    }
}

/// Strategy axis for differentiating proposals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProposalStrategy {
    /// Conservative: minimal changes, low temperature.
    Conservative,
    /// Standard: balanced approach, moderate temperature.
    Balanced,
    /// Creative: broader refactoring allowed, high temperature.
    Creative,
}

impl std::fmt::Display for ProposalStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Conservative => write!(f, "conservative"),
            Self::Balanced => write!(f, "balanced"),
            Self::Creative => write!(f, "creative"),
        }
    }
}

/// Model tier preference for a proposal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProposalTier {
    /// Fast local model (strand-14B).
    Fast,
    /// Reasoning local model (OR1-Behemoth 72B).
    Reasoning,
    /// Cloud model (Opus 4.6 etc).
    Cloud,
}

impl std::fmt::Display for ProposalTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fast => write!(f, "fast"),
            Self::Reasoning => write!(f, "reasoning"),
            Self::Cloud => write!(f, "cloud"),
        }
    }
}

/// Specification for a single proposal attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalSpec {
    /// Unique proposal identifier (e.g., "prop-0", "prop-1").
    pub id: String,
    /// Strategy axis for this proposal.
    pub strategy: ProposalStrategy,
    /// Model tier to use.
    pub tier: ProposalTier,
    /// Temperature setting.
    pub temperature: f64,
    /// System prompt variant description.
    pub prompt_variant: String,
    /// Maximum tokens for this proposal.
    pub max_tokens: u64,
}

/// Result of a single proposal execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposalResult {
    /// Which proposal spec this corresponds to.
    pub proposal_id: String,
    /// Whether the verifier passed.
    pub verifier_passed: bool,
    /// Number of remaining errors.
    pub error_count: usize,
    /// Number of remaining warnings.
    pub warning_count: usize,
    /// Number of files changed.
    pub files_changed: usize,
    /// Total lines added + removed.
    pub diff_size: usize,
    /// Tokens consumed.
    pub tokens_used: u64,
    /// Duration of the attempt.
    pub duration_secs: f64,
    /// Strategy used.
    pub strategy: ProposalStrategy,
    /// Model tier used.
    pub tier: ProposalTier,
}

/// Outcome of the speculative execution selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionOutcome {
    /// A clear winner was selected.
    Winner { winner_id: String, reason: String },
    /// No proposals passed the verifier.
    NonePassedVerifier {
        /// Combined insights from failed attempts for retry context.
        combined_insights: Vec<String>,
    },
    /// Only one proposal was submitted (no competition).
    SingleProposal { proposal_id: String, passed: bool },
}

/// Plan generated by the speculative planner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculativePlan {
    /// Proposals to execute.
    pub proposals: Vec<ProposalSpec>,
    /// Token budget allocated per proposal.
    pub per_proposal_budget: u64,
    /// Total token budget.
    pub total_budget: u64,
    /// Task description used for planning.
    pub task_summary: String,
    /// Assessed complexity.
    pub complexity: TaskComplexity,
}

/// Generate a speculative plan for a task.
///
/// Creates diversified proposals based on task complexity and available
/// budget. Returns `None` if speculation should not be used.
pub fn plan_proposals(
    task_description: &str,
    complexity: TaskComplexity,
    config: &SpeculativeConfig,
) -> Option<SpeculativePlan> {
    if !config.should_speculate(complexity) {
        return None;
    }

    let max_proposals = config.max_proposals.min(3);
    let per_proposal_budget = config.token_budget / max_proposals as u64;

    let mut proposals = Vec::with_capacity(max_proposals);

    // Proposal 0: Conservative (minimal fix, low temp, fast model)
    proposals.push(ProposalSpec {
        id: "prop-0".to_string(),
        strategy: ProposalStrategy::Conservative,
        tier: ProposalTier::Fast,
        temperature: 0.2,
        prompt_variant: "Minimal targeted fix. Change as few lines as possible.".to_string(),
        max_tokens: per_proposal_budget,
    });

    // Proposal 1: Balanced (standard approach, reasoning model)
    if max_proposals >= 2 {
        proposals.push(ProposalSpec {
            id: "prop-1".to_string(),
            strategy: ProposalStrategy::Balanced,
            tier: ProposalTier::Reasoning,
            temperature: 0.5,
            prompt_variant: "Balanced implementation. Fix the issue with clean code.".to_string(),
            max_tokens: per_proposal_budget,
        });
    }

    // Proposal 2: Creative (refactor, high temp, cloud model)
    if max_proposals >= 3 {
        proposals.push(ProposalSpec {
            id: "prop-2".to_string(),
            strategy: ProposalStrategy::Creative,
            tier: ProposalTier::Cloud,
            temperature: 0.8,
            prompt_variant:
                "Refactor for clarity. You may restructure code to solve the root cause."
                    .to_string(),
            max_tokens: per_proposal_budget,
        });
    }

    Some(SpeculativePlan {
        proposals,
        per_proposal_budget,
        total_budget: config.token_budget,
        task_summary: task_description.to_string(),
        complexity,
    })
}

/// Select the best proposal from a set of results.
///
/// Ranking:
/// 1. Verifier passed (hard gate)
/// 2. Fewest errors
/// 3. Fewest warnings
/// 4. Smallest diff (most concise)
/// 5. Lowest token cost
pub fn select_winner(results: &[ProposalResult]) -> SelectionOutcome {
    if results.is_empty() {
        return SelectionOutcome::NonePassedVerifier {
            combined_insights: vec!["No proposals were executed.".to_string()],
        };
    }

    if results.len() == 1 {
        return SelectionOutcome::SingleProposal {
            proposal_id: results[0].proposal_id.clone(),
            passed: results[0].verifier_passed,
        };
    }

    let mut passed: Vec<&ProposalResult> = results.iter().filter(|r| r.verifier_passed).collect();

    if passed.is_empty() {
        // None passed — combine insights
        let insights = results
            .iter()
            .map(|r| {
                format!(
                    "{} ({}): {} errors, {} warnings, {} files changed",
                    r.proposal_id, r.strategy, r.error_count, r.warning_count, r.files_changed,
                )
            })
            .collect();

        return SelectionOutcome::NonePassedVerifier {
            combined_insights: insights,
        };
    }

    // Sort passed proposals by quality (errors → warnings → diff_size → tokens)
    passed.sort_by(|a, b| {
        a.error_count
            .cmp(&b.error_count)
            .then(a.warning_count.cmp(&b.warning_count))
            .then(a.diff_size.cmp(&b.diff_size))
            .then(a.tokens_used.cmp(&b.tokens_used))
    });

    let winner = passed[0];
    let reason = if passed.len() == 1 {
        format!(
            "Only passing proposal: {} ({})",
            winner.proposal_id, winner.strategy,
        )
    } else {
        format!(
            "Best of {} passing proposals: {} ({}) — {} errors, {} warnings, {} diff lines, {} tokens",
            passed.len(),
            winner.proposal_id,
            winner.strategy,
            winner.error_count,
            winner.warning_count,
            winner.diff_size,
            winner.tokens_used,
        )
    };

    SelectionOutcome::Winner {
        winner_id: winner.proposal_id.clone(),
        reason,
    }
}

/// Summary of a speculative execution run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeculationSummary {
    /// Number of proposals attempted.
    pub proposals_attempted: usize,
    /// Number that passed the verifier.
    pub proposals_passed: usize,
    /// Total tokens consumed across all proposals.
    pub total_tokens: u64,
    /// Total duration across all proposals.
    pub total_duration_secs: f64,
    /// The selection outcome.
    pub outcome: SelectionOutcome,
}

/// Compute a summary from proposal results.
pub fn summarize_results(results: &[ProposalResult]) -> SpeculationSummary {
    let proposals_attempted = results.len();
    let proposals_passed = results.iter().filter(|r| r.verifier_passed).count();
    let total_tokens: u64 = results.iter().map(|r| r.tokens_used).sum();
    let total_duration_secs: f64 = results.iter().map(|r| r.duration_secs).sum();
    let outcome = select_winner(results);

    SpeculationSummary {
        proposals_attempted,
        proposals_passed,
        total_tokens,
        total_duration_secs,
        outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default_disabled() {
        let config = SpeculativeConfig::default();
        assert!(!config.enabled);
        assert!(!config.should_speculate(TaskComplexity::Complex));
    }

    #[test]
    fn test_config_enabled_for_complex() {
        let config = SpeculativeConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(config.should_speculate(TaskComplexity::Complex));
        assert!(!config.should_speculate(TaskComplexity::Simple));
        assert!(!config.should_speculate(TaskComplexity::Unknown));
    }

    #[test]
    fn test_config_enabled_for_medium() {
        let config = SpeculativeConfig {
            enabled: true,
            min_complexity: TaskComplexity::Medium,
            ..Default::default()
        };
        assert!(config.should_speculate(TaskComplexity::Complex));
        assert!(config.should_speculate(TaskComplexity::Medium));
        assert!(!config.should_speculate(TaskComplexity::Simple));
    }

    #[test]
    fn test_plan_returns_none_when_disabled() {
        let config = SpeculativeConfig::default();
        assert!(plan_proposals("task", TaskComplexity::Complex, &config).is_none());
    }

    #[test]
    fn test_plan_returns_none_for_simple_tasks() {
        let config = SpeculativeConfig {
            enabled: true,
            ..Default::default()
        };
        assert!(plan_proposals("task", TaskComplexity::Simple, &config).is_none());
    }

    #[test]
    fn test_plan_generates_proposals_for_complex() {
        let config = SpeculativeConfig {
            enabled: true,
            max_proposals: 3,
            token_budget: 300_000,
            ..Default::default()
        };
        let plan = plan_proposals("Fix async lifetimes", TaskComplexity::Complex, &config);
        assert!(plan.is_some());
        let plan = plan.unwrap();

        assert_eq!(plan.proposals.len(), 3);
        assert_eq!(plan.per_proposal_budget, 100_000);
        assert_eq!(plan.complexity, TaskComplexity::Complex);

        // Verify diversity
        assert_eq!(plan.proposals[0].strategy, ProposalStrategy::Conservative);
        assert_eq!(plan.proposals[1].strategy, ProposalStrategy::Balanced);
        assert_eq!(plan.proposals[2].strategy, ProposalStrategy::Creative);

        assert_eq!(plan.proposals[0].tier, ProposalTier::Fast);
        assert_eq!(plan.proposals[1].tier, ProposalTier::Reasoning);
        assert_eq!(plan.proposals[2].tier, ProposalTier::Cloud);

        assert!(plan.proposals[0].temperature < plan.proposals[1].temperature);
        assert!(plan.proposals[1].temperature < plan.proposals[2].temperature);
    }

    #[test]
    fn test_plan_respects_max_proposals() {
        let config = SpeculativeConfig {
            enabled: true,
            max_proposals: 2,
            token_budget: 200_000,
            ..Default::default()
        };
        let plan = plan_proposals("task", TaskComplexity::Complex, &config).unwrap();
        assert_eq!(plan.proposals.len(), 2);
    }

    #[test]
    fn test_select_winner_single_passing() {
        let results = vec![
            make_result("prop-0", true, 0, 2, 5, 30, 10_000),
            make_result("prop-1", false, 3, 0, 10, 50, 15_000),
        ];
        match select_winner(&results) {
            SelectionOutcome::Winner { winner_id, .. } => {
                assert_eq!(winner_id, "prop-0");
            }
            other => panic!("Expected Winner, got {:?}", other),
        }
    }

    #[test]
    fn test_select_winner_multiple_passing() {
        let results = vec![
            make_result("prop-0", true, 0, 5, 20, 100, 10_000),
            make_result("prop-1", true, 0, 2, 10, 50, 15_000),
            make_result("prop-2", true, 0, 3, 15, 80, 20_000),
        ];
        match select_winner(&results) {
            SelectionOutcome::Winner { winner_id, reason } => {
                // prop-1 wins: 0 errors, 2 warnings (fewest), 50 diff
                assert_eq!(winner_id, "prop-1");
                assert!(reason.contains("Best of 3"));
            }
            other => panic!("Expected Winner, got {:?}", other),
        }
    }

    #[test]
    fn test_select_winner_none_passing() {
        let results = vec![
            make_result("prop-0", false, 5, 2, 10, 30, 10_000),
            make_result("prop-1", false, 3, 1, 8, 20, 15_000),
        ];
        match select_winner(&results) {
            SelectionOutcome::NonePassedVerifier { combined_insights } => {
                assert_eq!(combined_insights.len(), 2);
                assert!(combined_insights[0].contains("prop-0"));
            }
            other => panic!("Expected NonePassedVerifier, got {:?}", other),
        }
    }

    #[test]
    fn test_select_winner_single_proposal() {
        let results = vec![make_result("prop-0", true, 0, 0, 5, 30, 10_000)];
        match select_winner(&results) {
            SelectionOutcome::SingleProposal {
                proposal_id,
                passed,
            } => {
                assert_eq!(proposal_id, "prop-0");
                assert!(passed);
            }
            other => panic!("Expected SingleProposal, got {:?}", other),
        }
    }

    #[test]
    fn test_select_winner_empty_results() {
        match select_winner(&[]) {
            SelectionOutcome::NonePassedVerifier { .. } => {}
            other => panic!("Expected NonePassedVerifier, got {:?}", other),
        }
    }

    #[test]
    fn test_summarize_results() {
        let results = vec![
            make_result("prop-0", true, 0, 1, 5, 30, 10_000),
            make_result("prop-1", false, 3, 0, 10, 50, 15_000),
            make_result("prop-2", true, 0, 0, 3, 20, 20_000),
        ];
        let summary = summarize_results(&results);

        assert_eq!(summary.proposals_attempted, 3);
        assert_eq!(summary.proposals_passed, 2);
        assert_eq!(summary.total_tokens, 45_000);
        assert!((summary.total_duration_secs - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_proposal_strategy_display() {
        assert_eq!(ProposalStrategy::Conservative.to_string(), "conservative");
        assert_eq!(ProposalStrategy::Balanced.to_string(), "balanced");
        assert_eq!(ProposalStrategy::Creative.to_string(), "creative");
    }

    #[test]
    fn test_proposal_tier_display() {
        assert_eq!(ProposalTier::Fast.to_string(), "fast");
        assert_eq!(ProposalTier::Reasoning.to_string(), "reasoning");
        assert_eq!(ProposalTier::Cloud.to_string(), "cloud");
    }

    #[test]
    fn test_plan_json_roundtrip() {
        let config = SpeculativeConfig {
            enabled: true,
            ..Default::default()
        };
        let plan = plan_proposals("Fix issue", TaskComplexity::Complex, &config).unwrap();
        let json = serde_json::to_string(&plan).unwrap();
        let restored: SpeculativePlan = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.proposals.len(), plan.proposals.len());
        assert_eq!(restored.total_budget, plan.total_budget);
    }

    #[test]
    fn test_selection_tiebreaker_by_diff_size() {
        // Same errors and warnings, different diff sizes
        let results = vec![
            make_result("prop-0", true, 0, 0, 20, 200, 10_000),
            make_result("prop-1", true, 0, 0, 5, 30, 15_000),
        ];
        match select_winner(&results) {
            SelectionOutcome::Winner { winner_id, .. } => {
                // prop-1 wins: smaller diff (30 vs 200)
                assert_eq!(winner_id, "prop-1");
            }
            other => panic!("Expected Winner, got {:?}", other),
        }
    }

    #[test]
    fn test_selection_tiebreaker_by_tokens() {
        // Same everything except tokens
        let results = vec![
            make_result("prop-0", true, 0, 0, 5, 30, 20_000),
            make_result("prop-1", true, 0, 0, 5, 30, 10_000),
        ];
        match select_winner(&results) {
            SelectionOutcome::Winner { winner_id, .. } => {
                // prop-1 wins: fewer tokens
                assert_eq!(winner_id, "prop-1");
            }
            other => panic!("Expected Winner, got {:?}", other),
        }
    }

    /// Helper to construct ProposalResult for tests.
    fn make_result(
        id: &str,
        passed: bool,
        errors: usize,
        warnings: usize,
        files: usize,
        diff: usize,
        tokens: u64,
    ) -> ProposalResult {
        ProposalResult {
            proposal_id: id.to_string(),
            verifier_passed: passed,
            error_count: errors,
            warning_count: warnings,
            files_changed: files,
            diff_size: diff,
            tokens_used: tokens,
            duration_secs: 1.0,
            strategy: match id {
                "prop-0" => ProposalStrategy::Conservative,
                "prop-1" => ProposalStrategy::Balanced,
                _ => ProposalStrategy::Creative,
            },
            tier: match id {
                "prop-0" => ProposalTier::Fast,
                "prop-1" => ProposalTier::Reasoning,
                _ => ProposalTier::Cloud,
            },
        }
    }
}
