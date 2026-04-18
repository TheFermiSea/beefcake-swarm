//! Coordination library — surviving MCP server components after Phase 2 Python migration.
//!
//! The multi-agent orchestration that lived in swarm-agents + the escalation /
//! ensemble / council / feedback / router / work_packet / reformulation /
//! debate / context_packer / analytics subsystems was replaced by the
//! mini-SWE-agent-based Python worker (see python/swarm_worker.py,
//! python/run.py, python/dogfood.py). What remains here is the subset still
//! useful as a standalone MCP server: verifier pipeline, SLURM lifecycle,
//! harness session primitives, state types, benchmarks, and OTel helpers.

// ── Always-compiled modules ──
pub mod benchmark;
pub mod fim;
pub mod harness;
pub mod otel;
pub mod rollout;
pub mod state;

// ── Full-only modules (MCP binary surface; not linked by default) ──
#[cfg(feature = "full")]
pub mod agent_profile;
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
pub mod tool_bundle;
#[cfg(feature = "full")]
pub mod tool_schema;

// ── Re-exports ──

pub use harness::{load_session_state, save_session_state};
pub use harness::{
    FeatureCategory, FeatureSpec, GitManager, HarnessConfig, HarnessError, HarnessResult,
    HarnessState, InterventionType, PendingIntervention, ProgressEntry, ProgressMarker,
    ProgressTracker, SessionManager, SessionRetrospective, SessionState, SessionStatus,
    SessionSummary, StartupContext,
};

pub use state::{
    EnsembleSession, EnsembleTask, ModelId, ModelResult, SharedContext, TaskStatus, VoteRecord,
    VotingStrategy,
};

#[cfg(feature = "heavy-state")]
pub use state::{SharedStateStore, StateStore};

#[cfg(feature = "heavy-state")]
pub use events::{
    ArbitrationReason, ContextUpdater, EnsembleEvent, EventBus, EventHistory, SessionEndReason,
    SharedEventBus,
};

#[cfg(feature = "full")]
pub use slurm::{
    EndpointHealth, EndpointHealthDetails, EndpointInfo, HealthCheckConfig,
    HealthCheckMetricsSnapshot, InferenceTier, JobInfo, JobState, SlurmConfig, SlurmError,
    SlurmInferenceManager,
};

pub use benchmark::{load_beefcake_lx2o_manifest, BenchmarkManifest};

#[cfg(feature = "full")]
pub use registry::{ProviderCapabilities, ProviderEntry, ProviderHealth, ProviderRegistry};

#[cfg(feature = "full")]
pub use resilience::{DegradationLevel, DegradedResponse, FallbackChain, FallbackTier, ToolHealth};

pub use otel::{AgentRole, SpanSummary};

pub use rollout::{
    Cohort, FeatureFlag, FeatureFlagOverrides, FeatureFlags, RolloutError, RolloutManager,
    RolloutStage, RolloutSummary, SafetyGate,
};
