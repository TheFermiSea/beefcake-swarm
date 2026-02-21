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

pub mod consensus;
pub mod guardrails;
pub mod state;

pub use consensus::{ConsensusCheck, ConsensusOutcome, ConsensusProtocol, Verdict};
pub use guardrails::{DeadlockOutcome, GuardrailConfig, GuardrailEngine};
pub use state::{DebatePhase, DebateSession, DebateTransition, ParticipantRole, RoundRecord};
