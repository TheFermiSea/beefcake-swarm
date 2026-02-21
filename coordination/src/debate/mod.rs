//! Debate Orchestration — Coder-Reviewer Consensus Loop
//!
//! State machine for structured debate between coder and reviewer agents.
//! Compatible with the existing `ensemble::arbitration` module for cases
//! where consensus cannot be reached.
//!
//! # Debate Flow
//!
//! ```text
//! Idle → CoderTurn → ReviewerTurn → [consensus?]
//!   │       │             │              │
//!   │       └─────────────┘              ├─ Yes → Resolved
//!   │         (iterate)                  ├─ No, rounds left → CoderTurn
//!   │                                    └─ No, max rounds → Deadlocked
//!   │                                                           │
//!   │                                                           ▼
//!   │                                                     Escalated
//!   │                                                     (→ arbitration)
//!   └─ abort at any point → Aborted
//! ```
//!
//! # Modules
//!
//! - [`state`] — Phase state machine, session tracking, transitions
//! - [`consensus`] — Verdict protocol, consensus detection, stall detection
//! - [`guardrails`] — Deadlock/timeout guardrails
//! - [`orchestrator`] — High-level debate driver tying everything together
//! - [`critique`] — Structured reviewer→coder feedback plumbing
//! - [`persistence`] — Checkpoint/resume for interrupted debates

pub mod consensus;
pub mod critique;
pub mod guardrails;
pub mod orchestrator;
pub mod persistence;
pub mod state;

pub use consensus::{ConsensusCheck, ConsensusOutcome, ConsensusProtocol, Verdict};
pub use critique::{
    CritiqueCategory, CritiqueItem, CritiqueSeverity, PatchCritique, RepairInstruction,
};
pub use guardrails::{DeadlockOutcome, GuardrailConfig, GuardrailEngine};
pub use orchestrator::{
    CoderOutput, DebateConfig, DebateError, DebateOrchestrator, DebateOutcome, NextAction,
    ReviewerOutput,
};
pub use persistence::{CheckpointManager, DebateCheckpoint, IntegrityStatus, PersistenceError};
pub use state::{DebatePhase, DebateSession, DebateTransition, ParticipantRole, RoundRecord};
