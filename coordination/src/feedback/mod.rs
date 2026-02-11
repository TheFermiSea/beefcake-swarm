//! Compilation Feedback Module
//!
//! Provides a feedback loop for iterative code correction:
//! - Run cargo check/clippy and parse errors
//! - Classify errors by type (borrow checker, lifetime, type mismatch, etc.)
//! - Iterate with LLM corrections until compilation succeeds
//!
//! # Architecture
//!
//! ```text
//! Code → Compiler → ErrorParser → CorrectionLoop → Model Router → Fixed Code
//!                        ↑                                           |
//!                        └───────────────────────────────────────────┘
//! ```

pub mod compiler;
pub mod correction_loop;
pub mod error_parser;

pub use compiler::{CargoOutput, CompileResult, Compiler};
pub use correction_loop::{CorrectionConfig, CorrectionLoop, CorrectionResult};
pub use error_parser::{ErrorCategory, ParsedError, RustcErrorParser};
