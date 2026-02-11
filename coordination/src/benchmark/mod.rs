//! Benchmark Module
//!
//! Provides infrastructure for running rust-bench problems and tracking metrics.
//!
//! # Architecture
//!
//! ```text
//! rust-bench repo → Problem Parser → BenchmarkSession
//!                                          ↓
//!                                    Problem Queue
//!                                          ↓
//!                              ┌─────────────────────┐
//!                              │                     │
//!                              ↓                     ↓
//!                        First Attempt         Correction Loop
//!                              │                     │
//!                              └──────────┬──────────┘
//!                                         ↓
//!                                   Metrics Tracking
//! ```
//!
//! # Workflow
//!
//! 1. Load problems from rust-bench repo
//! 2. For each problem:
//!    a. Generate initial solution
//!    b. Run cargo check
//!    c. If fails, run correction loop
//!    d. Track metrics (tokens, iterations, model used)
//! 3. Generate summary report

pub mod metrics;
pub mod problem;

pub use metrics::{AttemptMetrics, BenchmarkMetrics, ProblemMetrics};
pub use problem::{BenchmarkConfig, BenchmarkProblem, BenchmarkSession, Difficulty, ProblemStatus};
