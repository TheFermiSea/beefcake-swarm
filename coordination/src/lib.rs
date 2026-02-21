//! Rust Cluster MCP Library
//!
//! This library provides:
//! - MCP tools for delegating to local Rust-expert LLMs
//! - Agent harness implementing Anthropic's patterns for effective long-running agents
//! - Multi-agent ensemble coordination with voting and arbitration
//!
//! # Features
//!
//! ## LLM Tools
//! - `ask_rust_architect`: Deep analysis with OR1-Behemoth 73B
//! - `ask_rust_coder`: Idiomatic code with Strand-Rust-Coder 14B
//! - `ask_hydra_coder`: Specialized generation with HydraCoder 31B MoE
//!
//! ## Harness Tools (when enabled with --harness)
//! - `harness_start`: Initialize or resume a harness session
//! - `harness_status`: Get session and feature status
//! - `harness_complete_feature`: Mark a feature as complete
//! - `harness_checkpoint`: Create a git checkpoint
//! - `harness_rollback`: Rollback to a previous checkpoint
//! - `harness_iterate`: Increment iteration counter
//!
//! ## Ensemble Tools (when enabled with --ensemble)
//! - `ensemble_start`: Create ensemble session for multi-model coordination
//! - `ensemble_submit`: Submit task for multi-model processing
//! - `ensemble_status`: Get session state and pending tasks
//! - `ensemble_vote`: Trigger voting on collected results
//! - `ensemble_arbitrate`: Request Claude arbitration for disputes
//! - `ensemble_context`: Get/update shared cross-model context
//! - `ensemble_replay`: Replay events for debugging/recovery
//!
//! # Usage
//!
//! ```bash
//! # Run in normal MCP mode
//! rust-cluster-mcp
//!
//! # Run with harness tools enabled
//! rust-cluster-mcp --harness
//!
//! # Run with ensemble coordination enabled
//! rust-cluster-mcp --ensemble --state-path ./ensemble-state
//!
//! # Run with both harness and ensemble
//! rust-cluster-mcp --harness --ensemble
//! ```

#![allow(dead_code)]
#![allow(clippy::uninlined_format_args)]

pub mod agent_profile;
pub mod benchmark;
pub mod context_packer;
pub mod council;
pub mod debate;
pub mod ensemble;
pub mod escalation;
pub mod events;
pub mod feedback;
pub mod harness;
pub mod memory;
pub mod perf_control;
pub mod registry;
pub mod resilience;
pub mod reviewer_policy;
pub mod reviewer_tools;
pub mod rollout;
pub mod router;
pub mod shell_safety;
pub mod slurm;
pub mod state;
pub mod tool_schema;
pub mod verifier;
pub mod work_packet;

// Re-export key harness types
pub use harness::{load_session_state, save_session_state};
pub use harness::{
    FeatureCategory, FeatureSpec, GitManager, HarnessConfig, HarnessError, HarnessResult,
    HarnessState, InterventionType, PendingIntervention, ProgressEntry, ProgressMarker,
    ProgressTracker, SessionManager, SessionState, SessionStatus, SessionSummary, StartupContext,
};

// Re-export key ensemble types
pub use ensemble::{
    ArbitrationDecision, ArbitrationRequest, EnsembleConfig, EnsembleCoordinator, EnsembleStatus,
    SharedEnsembleCoordinator, VoteOutcome,
};

// Re-export key state types
pub use state::{
    EnsembleSession, EnsembleTask, ModelId, ModelResult, SharedContext, SharedStateStore,
    StateStore, TaskStatus, VoteRecord, VotingStrategy,
};

// Re-export key event types
pub use events::{
    ArbitrationReason, ContextUpdater, EnsembleEvent, EventBus, EventHistory, SessionEndReason,
    SharedEventBus,
};

// Re-export key SLURM types
pub use slurm::{
    EndpointHealth, EndpointHealthDetails, EndpointInfo, HealthCheckConfig,
    HealthCheckMetricsSnapshot, InferenceTier, JobInfo, JobState, SlurmConfig, SlurmError,
    SlurmInferenceManager,
};

// Re-export key Council types
pub use council::{
    CouncilConfig, CouncilDecision, CouncilError, CouncilMember, CouncilResponse, CouncilRole,
    DelegationReason, DelegationRequest, ErrorAttempt, EscalationContext, EscalationReason,
    ManagerCouncil,
};

// Re-export tiered correction types
pub use feedback::correction_loop::{
    EscalationTier, EscalationTrigger, TieredCorrectionContext, TieredCorrectionLoop,
};

// Re-export verifier types
pub use verifier::{GateOutcome, GateResult, Verifier, VerifierConfig, VerifierReport};

// Re-export escalation types
pub use escalation::{
    EscalationDecision, EscalationEngine, EscalationState, SwarmTier, TierBudget, TurnPolicy,
};

// Re-export telemetry heuristic types
pub use escalation::{compute_heuristics, SessionSample, TelemetryHeuristics};

// Re-export friction detection types
pub use escalation::{FrictionDetector, FrictionKind, FrictionSeverity, FrictionSignal};

// Re-export delight detection types
pub use escalation::{DelightDetector, DelightIntensity, DelightKind, DelightSignal};

// Re-export work packet types
pub use work_packet::{Constraint, FileContext, KeySymbol, WorkPacket, WorkPacketGenerator};

// Re-export context packer types
pub use context_packer::{ContextPacker, FileWalker};

// Re-export pre-routing classifier types
pub use router::{
    ComplexityFactors, PreRoutingAnalysis, PreRoutingClassifier, RiskFactor, RiskKind, RiskLevel,
};

// Re-export provider registry types
pub use registry::{ProviderCapabilities, ProviderEntry, ProviderHealth, ProviderRegistry};

// Re-export resilience types
pub use resilience::{DegradationLevel, DegradedResponse, FallbackChain, FallbackTier, ToolHealth};

// Re-export rollout types
pub use rollout::{
    Cohort, FeatureFlag, RolloutError, RolloutManager, RolloutStage, RolloutSummary, SafetyGate,
};
