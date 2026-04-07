//! Context Packer — Builds token-budgeted context for agent tiers
//!
//! Handles both initial context (no prior failures) and retry context
//! (using existing WorkPacketGenerator for error-enriched packets).

pub mod ast_index;
pub mod file_walker;
pub mod packer;
pub mod probes;
pub mod repo_map;
pub(crate) mod semantic;
#[cfg(feature = "full")]
pub mod semantic_graph;
pub mod source_provider;

pub use ast_index::{FileSymbolIndex, RustSymbol, SymbolKind};
pub use file_walker::FileWalker;
pub use packer::ContextPacker;
pub use repo_map::generate_repo_map;
#[cfg(feature = "full")]
pub use semantic_graph::SemanticCodeGraph;
pub use source_provider::SourceFileProvider;
