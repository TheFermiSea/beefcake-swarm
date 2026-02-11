//! Context Packer — Builds token-budgeted context for agent tiers
//!
//! Handles both initial context (no prior failures) and retry context
//! (using existing WorkPacketGenerator for error-enriched packets).

pub mod file_walker;
pub mod packer;

pub use file_walker::FileWalker;
pub use packer::ContextPacker;
