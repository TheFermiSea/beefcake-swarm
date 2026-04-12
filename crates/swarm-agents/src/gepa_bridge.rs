//! GEPA prompt optimization bridge (Layer 4).
//!
//! Integrates TensorZero's GEPA algorithm for autonomous prompt template
//! optimization. GEPA iteratively samples prompt variations, evaluates them
//! against inference metrics, and mutates prompts based on LLM analysis.
//!
//! **NOT YET VALIDATED** — requires TZ Autopilot API key and stable Layers 1-3.
//! This module provides the scaffold; actual GEPA runs are triggered via
//! the TZ CLI or Autopilot UI.
//!
//! Design: docs/research/self-improving-swarm-architecture.md Layer 4.
//! TZ GEPA docs: https://www.tensorzero.com/docs/optimization/gepa.md

use std::path::Path;

use tracing::{info, warn};

/// Configuration for GEPA optimization runs.
#[derive(Debug, Clone)]
pub struct GepaConfig {
    /// TZ gateway URL (e.g., http://localhost:3000)
    pub gateway_url: String,
    /// TZ Autopilot API key (sk-t0-...)
    pub autopilot_api_key: Option<String>,
    /// Which TZ function to optimize prompts for
    pub function_name: String,
    /// Which evaluation to use for scoring prompt quality
    pub evaluation_name: String,
    /// Minimum episodes required before triggering GEPA
    pub min_episodes_for_optimization: usize,
    /// How often to consider running GEPA (every N self-assessments)
    pub optimization_interval_assessments: usize,
}

impl Default for GepaConfig {
    fn default() -> Self {
        Self {
            gateway_url: "http://localhost:3000".to_string(),
            autopilot_api_key: std::env::var("TENSORZERO_AUTOPILOT_API_KEY").ok(),
            function_name: "worker_code_edit".to_string(),
            evaluation_name: "worker_behavior_quality".to_string(),
            min_episodes_for_optimization: 500,
            optimization_interval_assessments: 5, // every 50 issues (5 * 10)
        }
    }
}

/// Check if GEPA optimization is available and should run.
///
/// Returns a human-readable status message. Does NOT trigger GEPA —
/// that requires manual invocation via TZ Autopilot UI or CLI.
pub fn check_gepa_readiness(config: &GepaConfig, repo_root: &Path) -> GepaReadiness {
    // Check 1: API key
    if config.autopilot_api_key.is_none() {
        return GepaReadiness::NotReady {
            reason: "TENSORZERO_AUTOPILOT_API_KEY not set. \
                     Get one from https://autopilot.tensorzero.com"
                .to_string(),
        };
    }

    // Check 2: Evaluation config exists
    let eval_dir = repo_root
        .join("config")
        .join("evaluations")
        .join(&config.evaluation_name);
    if !eval_dir.exists() {
        return GepaReadiness::NotReady {
            reason: format!(
                "Evaluation '{}' not found at {}",
                config.evaluation_name,
                eval_dir.display()
            ),
        };
    }

    // Check 3: Function has prompt templates
    let func_dir = repo_root
        .join("config")
        .join("functions")
        .join(&config.function_name);
    let has_templates = func_dir
        .read_dir()
        .map(|entries| entries.filter_map(|e| e.ok()).any(|e| e.path().is_dir()))
        .unwrap_or(false);

    if !has_templates {
        return GepaReadiness::NotReady {
            reason: format!(
                "Function '{}' has no variant template directories",
                config.function_name
            ),
        };
    }

    GepaReadiness::Ready {
        function: config.function_name.clone(),
        evaluation: config.evaluation_name.clone(),
        note: "GEPA is available. Trigger via TZ Autopilot UI at /autopilot \
               or use the self-assessment loop to suggest optimization runs."
            .to_string(),
    }
}

/// GEPA readiness status.
#[derive(Debug)]
pub enum GepaReadiness {
    Ready {
        function: String,
        evaluation: String,
        note: String,
    },
    NotReady {
        reason: String,
    },
}

impl GepaReadiness {
    pub fn is_ready(&self) -> bool {
        matches!(self, GepaReadiness::Ready { .. })
    }
}

/// Log GEPA readiness status (called from self-assessment).
pub fn log_gepa_status(config: &GepaConfig, repo_root: &Path) {
    let readiness = check_gepa_readiness(config, repo_root);
    match &readiness {
        GepaReadiness::Ready {
            function,
            evaluation,
            note,
        } => {
            info!(
                function = %function,
                evaluation = %evaluation,
                "GEPA prompt optimization available: {note}"
            );
        }
        GepaReadiness::NotReady { reason } => {
            warn!("GEPA prompt optimization not ready: {reason}");
        }
    }
}
