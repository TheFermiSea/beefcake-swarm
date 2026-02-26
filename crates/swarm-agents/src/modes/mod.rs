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

pub mod errors;
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
