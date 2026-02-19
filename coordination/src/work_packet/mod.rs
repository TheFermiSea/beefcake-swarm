//! Work Packet Generator — Context-efficient handoff between tiers
//!
//! Work Packets are the primary communication format between model tiers.
//! They contain just enough context for a model to understand the task
//! without receiving the full conversation transcript.
//!
//! # Design Principles
//!
//! 1. **Context-efficient**: Models consume Work Packets, not full transcripts
//! 2. **Self-contained**: All information needed to act is in the packet
//! 3. **Structured**: Machine-parseable by any tier
//! 4. **Git-anchored**: Always references a specific commit/branch state

pub mod generator;
pub mod types;

pub use generator::WorkPacketGenerator;
pub use types::{Constraint, DelegationStep, FileContext, KeySymbol, WorkPacket};
