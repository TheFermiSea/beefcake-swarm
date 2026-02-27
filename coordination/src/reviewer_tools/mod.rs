//! Reviewer analysis tools — structured wrappers for ast-grep and GraphRAG.
//!
//! These modules provide bounded, deterministic tool interfaces for the
//! reviewer agent's analysis pipeline.
//!
//! # Modules
//!
//! - [`ast_grep`] — ast-grep (sg) wrapper with bounded output
//! - [`graph_rag`] — GraphRAG/CocoIndex wrapper for dependency queries
//! - [`rule_pack`] — Rule pack mapping to sgconfig

pub mod ast_grep;
pub mod graph_rag;
pub mod rule_pack;

pub use ast_grep::{AstGrepConfig, AstGrepMatch, AstGrepQuery, AstGrepRunner};
pub use graph_rag::{GraphRagConfig, GraphRagEnvBridge, GraphRagQuery, GraphRagResult, GraphRagRunner};
pub use rule_pack::{
    IngestionErrorKind, IngestionSummary, RuleIngestionError, RulePack, RulePackEntry,
    RulePackRegistry, RuleSeverity,
};
