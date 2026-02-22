//! Context Packer — Builds token-budgeted context for agent tiers
//!
//! Handles both initial context (no prior failures) and retry context
//! (using existing WorkPacketGenerator for error-enriched packets).

pub mod ast_index;
pub mod file_walker;
pub mod packer;
pub mod source_provider;

pub use ast_index::{FileSymbolIndex, RustSymbol, SymbolKind};
pub use file_walker::FileWalker;
pub use packer::ContextPacker;
pub use source_provider::SourceFileProvider;
