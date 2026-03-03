//! NS-1: Foundation contracts and orchestration scaffolding.
//!
//! Re-exports the four foundation sub-modules for external consumers.
//!
//! ## Sub-modules
//!
//! | Module          | NS task | Purpose                                          |
//! |-----------------|---------|--------------------------------------------------|
//! | `errors`        | NS-1.4  | Unified error taxonomy with retry classification |
//! | `types`         | NS-1.3  | Domain types: Artifact, Critique, Strategy, etc. |
//! | `provider_config` | NS-1.5| Provider/model runtime configuration             |
//! | `runner`        | NS-1.2  | ModeRunner trait and ModeOrchestrator driver     |

pub mod agentic;
pub mod apply_diff;
pub mod contextual;
pub mod deepthink;
pub mod errors;
pub mod memory;
pub mod provider_config;
pub mod runner;
pub mod types;

// Convenience re-exports used by mode implementations.
pub use errors::{OrchestrationError, RetryCategory};
pub use provider_config::ModeRunnerConfig;
pub use runner::{ModeContext, ModeOrchestrator, ModeRequest, ModeRunner, StepResult};
pub use types::{
    Artifact, CompactionSummary, CritiqueSeverity, CritiqueVerdict, ModeOutcome, Strategy,
    StrategyOutcome, StrategyResult, SynthesisResult,
};

// ── CLI / runtime mode selection ─────────────────────────────────────────────

/// Orchestration mode requested via `--mode` CLI flag.
///
/// Each variant maps to one of the NS-2/3/4 `ModeRunner` implementations.
/// When no mode is specified the classic implement→verify loop runs unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SwarmMode {
    /// Iterative Drafting → Critiquing → Condensing FSM (NS-2).
    Contextual,
    /// JoinSet fan-out across parallel strategy branches (NS-3).
    Deepthink,
    /// LLM-driven unified-diff file editing loop (NS-4).
    Agentic,
}

impl SwarmMode {
    /// Select a mode based on issue metadata.
    ///
    /// Heuristic:
    /// - bug with priority 0-1 → Agentic (fast, focused edits)
    /// - feature/task with label "modify_large" or "architecture" → Deepthink (parallel strategies)
    /// - everything else → Contextual (iterative refinement)
    pub fn from_issue(
        issue_type: Option<&str>,
        priority: Option<u8>,
        labels: &[String],
    ) -> Self {
        // High-priority bugs need fast, targeted fixes
        if issue_type == Some("bug") && priority.is_some_and(|p| p <= 1) {
            return SwarmMode::Agentic;
        }

        // Architecture/refactor tasks benefit from parallel strategy exploration
        let is_architecture = labels.iter().any(|l| {
            l == "architecture" || l == "refactor" || l == "modify_large"
        });
        if is_architecture {
            return SwarmMode::Deepthink;
        }

        // Feature tasks that are explicitly large → Deepthink
        if issue_type == Some("feature") && labels.iter().any(|l| l == "modify_large") {
            return SwarmMode::Deepthink;
        }

        // Default: iterative refinement
        SwarmMode::Contextual
    }

    /// Suggest a mode switch when the current mode is stuck.
    ///
    /// Called after consecutive no-change iterations. The idea is to try a
    /// different approach when the current one isn't making progress.
    pub fn switch_on_stuck(current: SwarmMode, consecutive_no_change: u32) -> Option<SwarmMode> {
        if consecutive_no_change < 2 {
            return None;
        }
        match current {
            // Contextual stuck → try Agentic (more direct edits)
            SwarmMode::Contextual => Some(SwarmMode::Agentic),
            // Agentic stuck → try Deepthink (explore alternatives)
            SwarmMode::Agentic => Some(SwarmMode::Deepthink),
            // Deepthink stuck → try Contextual (iterative refinement)
            SwarmMode::Deepthink => Some(SwarmMode::Contextual),
        }
    }

    /// Create the appropriate `ModeRunner` for this mode.
    pub fn into_runner(self, config: ModeRunnerConfig, working_dir: std::path::PathBuf) -> Box<dyn ModeRunner> {
        match self {
            SwarmMode::Contextual => Box::new(contextual::ContextualRunner::new(config)),
            SwarmMode::Deepthink => Box::new(deepthink::DeepthinkRunner::new(config)),
            SwarmMode::Agentic => Box::new(agentic::AgenticRunner::new(config, working_dir)),
        }
    }
}

#[cfg(test)]
mod mode_selection_tests {
    use super::*;

    #[test]
    fn high_priority_bug_selects_agentic() {
        assert_eq!(
            SwarmMode::from_issue(Some("bug"), Some(0), &[]),
            SwarmMode::Agentic,
        );
        assert_eq!(
            SwarmMode::from_issue(Some("bug"), Some(1), &[]),
            SwarmMode::Agentic,
        );
    }

    #[test]
    fn low_priority_bug_selects_contextual() {
        assert_eq!(
            SwarmMode::from_issue(Some("bug"), Some(3), &[]),
            SwarmMode::Contextual,
        );
    }

    #[test]
    fn architecture_label_selects_deepthink() {
        let labels = vec!["architecture".to_string()];
        assert_eq!(
            SwarmMode::from_issue(Some("task"), Some(2), &labels),
            SwarmMode::Deepthink,
        );
    }

    #[test]
    fn modify_large_selects_deepthink() {
        let labels = vec!["modify_large".to_string()];
        assert_eq!(
            SwarmMode::from_issue(Some("feature"), Some(2), &labels),
            SwarmMode::Deepthink,
        );
    }

    #[test]
    fn default_selects_contextual() {
        assert_eq!(
            SwarmMode::from_issue(Some("task"), Some(2), &[]),
            SwarmMode::Contextual,
        );
    }

    #[test]
    fn stuck_switches_modes() {
        // No switch before 2 consecutive no-change
        assert_eq!(SwarmMode::switch_on_stuck(SwarmMode::Contextual, 1), None);

        // Contextual → Agentic
        assert_eq!(
            SwarmMode::switch_on_stuck(SwarmMode::Contextual, 2),
            Some(SwarmMode::Agentic),
        );

        // Agentic → Deepthink
        assert_eq!(
            SwarmMode::switch_on_stuck(SwarmMode::Agentic, 3),
            Some(SwarmMode::Deepthink),
        );

        // Deepthink → Contextual
        assert_eq!(
            SwarmMode::switch_on_stuck(SwarmMode::Deepthink, 2),
            Some(SwarmMode::Contextual),
        );
    }
}
