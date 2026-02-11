//! Verifier Module — Deterministic Code Quality Gates
//!
//! The Verifier is the ONLY source of truth for code quality in the swarm.
//! It runs a sequential pipeline of deterministic checks and produces
//! structured reports with classified error categories.
//!
//! # Pipeline
//!
//! ```text
//! cargo fmt --check → cargo clippy -D warnings → cargo check --message-format=json → cargo test
//! ```
//!
//! # Error Classification
//!
//! Errors are classified into categories using rustc JSON output (no LLM involved):
//! - Borrow checker (E0502, E0505, E0382)
//! - Lifetimes (E0106, E0495, E0621)
//! - Trait bounds (E0277, E0599)
//! - Type mismatch (E0308, E0271)
//! - Async/Send (E0277 with Send bound)
//! - Module/visibility (E0603, E0412)
//! - Macro issues (E0658, expansion errors)
//!
//! # Usage
//!
//! ```rust,ignore
//! use rust_cluster_mcp::verifier::{Verifier, VerifierConfig};
//!
//! let verifier = Verifier::new("/path/to/crate", VerifierConfig::default());
//! let report = verifier.run_pipeline().await;
//! println!("Gates passed: {}/{}", report.gates_passed, report.gates_total);
//! ```

pub mod pipeline;
pub mod report;

pub use pipeline::{Verifier, VerifierConfig};
pub use report::{GateOutcome, GateResult, VerifierReport};
