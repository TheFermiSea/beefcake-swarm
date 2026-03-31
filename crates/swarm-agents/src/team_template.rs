//! Declarative team templates (ClawTeam adoption #4).
//!
//! TOML-based team definitions that specify phases, roles, model tiers,
//! and triggers for the orchestration loop. Replaces hardcoded role sequences
//! with configurable team compositions per repo type.
//!
//! Templates live in `config/teams/` and are selected by:
//! - `SWARM_TEAM_TEMPLATE` env var (e.g., "rust-fix", "multi-file", "python-fix")
//! - Auto-detection from LanguageProfile (Rust repos → rust-fix, Python → python-fix)
//! - Fallback to built-in default if no template matches
//!
//! # Example Template
//!
//! ```toml
//! [team]
//! name = "rust-fix"
//! description = "Standard Rust bug fix"
//! max_iterations = 10
//!
//! [[team.phases]]
//! name = "scout"
//! role = "Scout"
//! model_tier = "fast"
//! objective = "Identify affected files and assess complexity"
//! writes_code = false
//! ```

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::{Path, PathBuf};

/// A team template loaded from TOML.
#[derive(Debug, Clone, Deserialize)]
pub struct TeamTemplate {
    pub team: TeamConfig,
}

/// Top-level team configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct TeamConfig {
    pub name: String,
    pub description: String,
    #[serde(default = "default_max_iterations")]
    pub max_iterations: u32,
    #[serde(default)]
    pub phases: Vec<PhaseConfig>,
}

pub fn default_max_iterations() -> u32 {
    10
}

/// A single phase in the team workflow.
#[derive(Debug, Clone, Deserialize)]
pub struct PhaseConfig {
    /// Phase name (e.g., "scout", "implement", "fix").
    pub name: String,
    /// SwarmRole name (e.g., "Scout", "RustWorker", "Fixer", "GeneralWorker").
    pub role: String,
    /// Model tier: "fast", "coder", "reasoning", "cloud".
    pub model_tier: String,
    /// Objective template. May contain `{issue_title}` and `{issue_id}` placeholders.
    pub objective: String,
    /// Whether this phase writes code (affects tool selection).
    #[serde(default)]
    pub writes_code: bool,
    /// Whether multiple workers run in parallel (for multi-file phases).
    #[serde(default)]
    pub parallel: bool,
    /// When this phase activates: None = always, "verifier_failure", "escalation".
    #[serde(default)]
    pub trigger: Option<String>,
}

impl PhaseConfig {
    /// Whether this phase runs on the initial (happy path) iteration.
    pub fn is_default_phase(&self) -> bool {
        self.trigger.is_none()
    }

    /// Whether this phase is triggered by verifier failures.
    pub fn is_failure_phase(&self) -> bool {
        self.trigger.as_deref() == Some("verifier_failure")
    }

    /// Whether this phase is triggered by escalation.
    pub fn is_escalation_phase(&self) -> bool {
        self.trigger.as_deref() == Some("escalation")
    }
}

impl TeamConfig {
    /// Get the default phases (no trigger — run on initial iterations).
    pub fn default_phases(&self) -> Vec<&PhaseConfig> {
        self.phases
            .iter()
            .filter(|p| p.is_default_phase())
            .collect()
    }

    /// Get the phase to use on verifier failure.
    pub fn failure_phase(&self) -> Option<&PhaseConfig> {
        self.phases.iter().find(|p| p.is_failure_phase())
    }

    /// Get the phase to use on escalation.
    pub fn escalation_phase(&self) -> Option<&PhaseConfig> {
        self.phases.iter().find(|p| p.is_escalation_phase())
    }
}

/// Load a team template from the `config/teams/` directory.
///
/// Searches in order:
/// 1. `{repo_root}/config/teams/{name}.toml`
/// 2. `{repo_root}/.swarm/teams/{name}.toml` (target repo override)
pub fn load_template(name: &str, repo_root: &Path) -> Result<TeamTemplate> {
    let candidates = [
        repo_root.join("config/teams").join(format!("{name}.toml")),
        repo_root.join(".swarm/teams").join(format!("{name}.toml")),
    ];

    for path in &candidates {
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read team template: {}", path.display()))?;
            let template: TeamTemplate = toml::from_str(&content)
                .with_context(|| format!("Failed to parse team template: {}", path.display()))?;
            tracing::info!(
                name,
                path = %path.display(),
                phases = template.team.phases.len(),
                "Loaded team template"
            );
            return Ok(template);
        }
    }

    anyhow::bail!(
        "Team template '{}' not found in config/teams/ or .swarm/teams/",
        name
    )
}

/// Select the best team template for a given context.
///
/// Priority:
/// 1. `SWARM_TEAM_TEMPLATE` env var
/// 2. Language-based auto-detection
/// 3. Default ("rust-fix")
pub fn select_template(_repo_root: &Path, language: Option<&str>) -> String {
    // Explicit override
    if let Ok(name) = std::env::var("SWARM_TEAM_TEMPLATE") {
        if !name.trim().is_empty() {
            return name.trim().to_string();
        }
    }

    // Language-based selection
    match language {
        Some("python") => "python-fix".to_string(),
        Some("typescript") | Some("javascript") => "python-fix".to_string(), // reuse for now
        _ => "rust-fix".to_string(),
    }
}

/// List all available team templates in the config directory.
#[allow(dead_code)]
pub fn list_templates(repo_root: &Path) -> Vec<PathBuf> {
    let dir = repo_root.join("config/teams");
    if !dir.exists() {
        return vec![];
    }
    std::fs::read_dir(&dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "toml"))
        .map(|e| e.path())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_rust_fix_template() {
        let toml_str = r#"
[team]
name = "rust-fix"
description = "Standard Rust bug fix"
max_iterations = 10

[[team.phases]]
name = "scout"
role = "Scout"
model_tier = "fast"
objective = "Read the issue"
writes_code = false

[[team.phases]]
name = "implement"
role = "RustWorker"
model_tier = "coder"
objective = "Fix the bug"
writes_code = true

[[team.phases]]
name = "fix"
role = "Fixer"
model_tier = "coder"
objective = "Fix verifier errors"
writes_code = true
trigger = "verifier_failure"
"#;
        let template: TeamTemplate = toml::from_str(toml_str).unwrap();
        assert_eq!(template.team.name, "rust-fix");
        assert_eq!(template.team.phases.len(), 3);
        assert_eq!(template.team.default_phases().len(), 2);
        assert!(template.team.failure_phase().is_some());
        assert!(template.team.escalation_phase().is_none());
    }

    #[test]
    fn select_template_defaults() {
        // No env var set, Rust language
        std::env::remove_var("SWARM_TEAM_TEMPLATE");
        assert_eq!(select_template(Path::new("/tmp"), Some("rust")), "rust-fix");
        assert_eq!(
            select_template(Path::new("/tmp"), Some("python")),
            "python-fix"
        );
        assert_eq!(select_template(Path::new("/tmp"), None), "rust-fix");
    }
}
