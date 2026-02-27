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

pub mod apply_diff;
pub mod agentic;
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
