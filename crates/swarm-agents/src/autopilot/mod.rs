//! Autopilot module: end-to-end analysis → recommendations → artifacts.

pub mod runner;

// Re-export the main types for backward compatibility
pub use runner::{run_autopilot_loop, AutopilotReport, AutopilotRunner, TrendData};
