//! Coordination Library
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
//! coordination
//!
//! # Run with harness tools enabled
//! coordination --harness
//!
//! # Run with ensemble coordination enabled
//! coordination --ensemble --state-path ./ensemble-state
//!
//! # Run with both harness and ensemble
//! coordination --harness --ensemble
//! ```

// ── Always-compiled modules (used by swarm-agents) ──
pub mod analytics;
pub mod benchmark;
pub mod context_packer;
pub mod escalation;
pub mod feedback;
pub mod fim;
pub mod harness;
pub mod otel;
pub mod reformulation;
pub mod rollout;
pub mod router;
pub mod state;
pub mod verifier;
pub mod work_packet;

// ── Full-only modules (unused by swarm-agents, needed by MCP binary) ──
#[cfg(feature = "full")]
pub mod agent_profile;
#[cfg(feature = "full")]
pub mod council;
#[cfg(feature = "full")]
pub mod debate;
#[cfg(feature = "heavy-state")]
pub mod ensemble;
#[cfg(feature = "heavy-state")]
pub mod events;
#[cfg(feature = "full")]
pub mod memory;
#[cfg(feature = "full")]
pub mod patch;
#[cfg(feature = "full")]
pub mod perf_control;
#[cfg(feature = "full")]
pub mod registry;
#[cfg(feature = "full")]
pub mod resilience;
#[cfg(feature = "full")]
pub mod reviewer_policy;
#[cfg(feature = "full")]
pub mod reviewer_tools;
#[cfg(feature = "full")]
pub mod shell_safety;
#[cfg(feature = "full")]
pub mod slurm;
#[cfg(feature = "full")]
pub mod speculation;
#[cfg(feature = "full")]
pub mod tool_bundle;
#[cfg(feature = "full")]
pub mod tool_schema;

// Re-export key harness types
pub use harness::{load_session_state, save_session_state};
pub use harness::{
    FeatureCategory, FeatureSpec, GitManager, HarnessConfig, HarnessError, HarnessResult,
    HarnessState, InterventionType, PendingIntervention, ProgressEntry, ProgressMarker,
    ProgressTracker, SessionManager, SessionRetrospective, SessionState, SessionStatus,
    SessionSummary, StartupContext,
};

// Re-export key ensemble types
#[cfg(feature = "heavy-state")]
pub use ensemble::{
    ArbitrationDecision, ArbitrationRequest, EnsembleConfig, EnsembleCoordinator, EnsembleStatus,
    SharedEnsembleCoordinator, VoteOutcome,
};

// Re-export key state types (always available: pure data types)
pub use state::{
    EnsembleSession, EnsembleTask, ModelId, ModelResult, SharedContext, TaskStatus, VoteRecord,
    VotingStrategy,
};

// Re-export RocksDB-backed state store types (only with heavy-state)
#[cfg(feature = "heavy-state")]
pub use state::{SharedStateStore, StateStore};

// Re-export analytics types
pub use analytics::error::{AnalyticsError, AnalyticsResult};
pub use analytics::replay::{ExperienceTrace, ReplayHint, TraceContext, TraceIndex, TraceOutcome};
pub use analytics::skills::{Skill, SkillHint, SkillLibrary, SkillTrigger, TaskContext};
pub use analytics::verification::{
    AcceptancePolicy, AcceptanceStatus, UseOutcome, VerificationTracker,
};

// Re-export key event types
#[cfg(feature = "heavy-state")]
pub use events::{
    ArbitrationReason, ContextUpdater, EnsembleEvent, EventBus, EventHistory, SessionEndReason,
    SharedEventBus,
};

// Re-export key SLURM types
#[cfg(feature = "full")]
pub use slurm::{
    EndpointHealth, EndpointHealthDetails, EndpointInfo, HealthCheckConfig,
    HealthCheckMetricsSnapshot, InferenceTier, JobInfo, JobState, SlurmConfig, SlurmError,
    SlurmInferenceManager,
};

// Re-export key Council types
#[cfg(feature = "full")]
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
pub use verifier::{
    GateOutcome, GateResult, LanguageProfile, ScriptVerifier, ValidatorFeedback,
    ValidatorIssueType, Verifier, VerifierConfig, VerifierReport,
};

// Re-export key escalation types
pub use escalation::{
    EscalationDecision, EscalationEngine, EscalationState, SwarmTier, TierBudget, TurnPolicy,
};

// Re-export benchmark types
pub use benchmark::{load_beefcake_lx2o_manifest, BenchmarkManifest};

// Re-export telemetry heuristic types
pub use escalation::{compute_heuristics, SessionSample, TelemetryHeuristics};

// Re-export friction detection types
pub use escalation::{FrictionDetector, FrictionKind, FrictionSeverity, FrictionSignal};

// Re-export delight detection types
pub use escalation::{DelightDetector, DelightIntensity, DelightKind, DelightSignal};

// Re-export work packet types
pub use work_packet::{
    ChangeContract, Constraint, FileContext, KeySymbol, WorkPacket, WorkPacketGenerator,
};

// Re-export context packer types
#[cfg(feature = "full")]
pub use context_packer::SemanticCodeGraph;
pub use context_packer::{ContextPacker, FileWalker, SourceFileProvider};

// Re-export pre-routing classifier types
pub use router::{
    ComplexityFactors, PreRoutingAnalysis, PreRoutingClassifier, RiskFactor, RiskKind, RiskLevel,
};

// Re-export provider registry types
#[cfg(feature = "full")]
pub use registry::{ProviderCapabilities, ProviderEntry, ProviderHealth, ProviderRegistry};

// Re-export resilience types
#[cfg(feature = "full")]
pub use resilience::{DegradationLevel, DegradedResponse, FallbackChain, FallbackTier, ToolHealth};

// Re-export OTel span helpers
pub use otel::{AgentRole, SpanSummary};

// Re-export rollout types
pub use rollout::{
    Cohort, FeatureFlag, FeatureFlagOverrides, FeatureFlags, RolloutError, RolloutManager,
    RolloutStage, RolloutSummary, SafetyGate,
};
