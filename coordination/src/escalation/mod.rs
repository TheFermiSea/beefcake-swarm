//! Escalation Engine — Deterministic State Machine for Tier Routing
//!
//! Implements the escalation ladder that routes work through model tiers
//! based on error patterns and iteration history. This is a pure state machine
//! with no LLM calls — all decisions are deterministic.
//!
//! # Escalation Ladder
//!
//! ```text
//! Implementer (14B) — 6 iterations max
//!     │
//!     ├─ Error category CHANGES each iteration → stay (making progress)
//!     ├─ Error category REPEATS 2x → escalate to Integrator
//!     ├─ >3 compile failures total → escalate to Integrator
//!     │
//!     ▼
//! Integrator (72B) — 2 consultations max
//!     │  Produces repair plan + minimal edits
//!     │  Hands back to Implementer for execution
//!     │
//!     ├─ Issue resolved → Adversary review → close
//!     ├─ Still stuck → escalate to Cloud
//!     │
//!     ▼
//! Cloud Brain Trust — 1 architecture + 1 review per issue
//!     │  Receives Work Packet (not full transcript)
//!     │  Returns fix strategy → flows back down
//!     │
//!     ▼
//! If still stuck → create blocking beads issue, flag for human intervention
//! ```

pub mod delight;
pub mod engine;
pub mod friction;
pub mod heuristics;
pub mod state;

pub use delight::{DelightDetector, DelightIntensity, DelightKind, DelightSignal};
pub use engine::{EscalationDecision, EscalationEngine};
pub use friction::{FrictionDetector, FrictionKind, FrictionSeverity, FrictionSignal};
pub use heuristics::{compute_heuristics, SessionSample, TelemetryHeuristics};
pub use state::{EscalationState, SwarmTier, TierBudget, TurnPolicy};
