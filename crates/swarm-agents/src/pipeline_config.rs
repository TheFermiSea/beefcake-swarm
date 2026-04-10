//! TOML-configurable pipeline stage registry.
//!
//! Loads pipeline configuration from a TOML file, enabling A/B testing of different
//! pipeline layouts without recompilation. The active config is selected at startup
//! via the `SWARM_PIPELINE_CONFIG` environment variable.
//!
//! # Format
//!
//! ```toml
//! [pipeline]
//! name = "default"
//! description = "Standard 5-stage pipeline"
//!
//! [[pipeline.stages]]
//! name = "context_packing"
//! enabled = true
//! model_role = "Scout"
//! strategy = "ReAct"
//! tool_access = true
//! max_turns = 10
//! timeout_secs = 300
//! ```
//!
//! See `config/pipeline-stages.toml` (default) and `config/pipeline-masai.toml` (MASAI 6-stage).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::info;

/// Inference strategy for a pipeline stage.
///
/// Controls the prompting pattern used when the stage invokes an LLM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineStrategy {
    /// ReAct: Reason + Act loop — model interleaves reasoning with tool calls.
    /// Best for stages that need to explore the codebase or gather context.
    ReAct,
    /// Chain-of-Thought: extended reasoning without tool calls.
    /// Best for planning and evaluation stages where structured thinking matters.
    CoT,
    /// Vanilla: single forward pass, no special scaffolding.
    /// Best for simple assembly or record stages with deterministic output.
    Vanilla,
}

impl std::fmt::Display for PipelineStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineStrategy::ReAct => write!(f, "ReAct"),
            PipelineStrategy::CoT => write!(f, "CoT"),
            PipelineStrategy::Vanilla => write!(f, "Vanilla"),
        }
    }
}

/// Configuration for a single pipeline stage.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageConfig {
    /// Unique stage name (e.g., "context_packing", "localize").
    pub name: String,
    /// Whether this stage runs. Disabled stages are skipped without error.
    pub enabled: bool,
    /// SwarmRole string for model routing (e.g., "Scout", "ReasoningWorker", "RustWorker").
    /// The orchestrator maps this to the appropriate local or cloud endpoint.
    /// Use "Verifier" for the deterministic quality gate (no LLM invoked).
    pub model_role: String,
    /// Prompting strategy for this stage.
    pub strategy: PipelineStrategy,
    /// Whether the model receives tool definitions in this stage.
    /// Set to false for planning stages (MASAI Fixer pattern) to prevent tool-call drift.
    pub tool_access: bool,
    /// Maximum LLM turns allowed in this stage.
    pub max_turns: u32,
    /// Wall-clock timeout for the stage in seconds.
    pub timeout_secs: u64,
    /// Number of candidates to generate (for generation stages).
    /// When > 1, the stage produces multiple outputs for the evaluation stage to rank.
    #[serde(default = "default_candidate_count")]
    pub candidate_count: u32,
}

fn default_candidate_count() -> u32 {
    1
}

impl StageConfig {
    /// Whether this stage uses a deterministic verifier (no LLM).
    pub fn is_verifier_stage(&self) -> bool {
        self.model_role.eq_ignore_ascii_case("Verifier")
    }

    /// Whether this stage generates multiple candidates.
    pub fn is_multi_candidate(&self) -> bool {
        self.candidate_count > 1
    }
}

/// Top-level pipeline configuration loaded from TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Wrapper key matching the TOML `[pipeline]` table.
    pub pipeline: PipelineInner,
}

/// Inner pipeline definition (inside the `[pipeline]` table).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineInner {
    /// Human-readable pipeline name (e.g., "default", "masai_heterogeneous").
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Ordered list of stage configurations.
    pub stages: Vec<StageConfig>,
}

impl PipelineConfig {
    /// Human-readable pipeline name.
    pub fn name(&self) -> &str {
        &self.pipeline.name
    }

    /// All configured stages (including disabled ones).
    pub fn stages(&self) -> &[StageConfig] {
        &self.pipeline.stages
    }

    /// Only the enabled stages, in order.
    pub fn enabled_stages(&self) -> impl Iterator<Item = &StageConfig> {
        self.pipeline.stages.iter().filter(|s| s.enabled)
    }

    /// Look up a stage by name.
    pub fn stage(&self, name: &str) -> Option<&StageConfig> {
        self.pipeline
            .stages
            .iter()
            .find(|s| s.name == name && s.enabled)
    }

    /// Validate the loaded config. Returns an error if:
    /// - There are no stages at all.
    /// - Any stage name is empty.
    /// - Any stage has `max_turns = 0`.
    pub fn validate(&self) -> Result<()> {
        if self.pipeline.stages.is_empty() {
            anyhow::bail!("Pipeline '{}': no stages defined", self.pipeline.name);
        }
        for stage in &self.pipeline.stages {
            if stage.name.trim().is_empty() {
                anyhow::bail!("Pipeline '{}': stage has empty name", self.pipeline.name);
            }
            if stage.max_turns == 0 {
                anyhow::bail!(
                    "Pipeline '{}': stage '{}' has max_turns = 0 (must be >= 1)",
                    self.pipeline.name,
                    stage.name
                );
            }
        }
        Ok(())
    }
}

/// Load and validate a [`PipelineConfig`] from a TOML file.
///
/// Returns an error if the file cannot be read, the TOML is malformed, or
/// the config fails validation.
pub fn load_pipeline_config(path: &Path) -> Result<PipelineConfig> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read pipeline config: {}", path.display()))?;

    let config: PipelineConfig = toml::from_str(&raw)
        .with_context(|| format!("Failed to parse pipeline config: {}", path.display()))?;

    config
        .validate()
        .with_context(|| format!("Pipeline config validation failed: {}", path.display()))?;

    info!(
        path = %path.display(),
        name = %config.name(),
        stages = config.stages().len(),
        enabled = config.enabled_stages().count(),
        "Loaded pipeline config"
    );

    Ok(config)
}

/// Load the pipeline config from the path in `SWARM_PIPELINE_CONFIG`, or use the
/// built-in default if the env var is unset.
///
/// Returns `None` if the env var is set but the file cannot be loaded (logs a warning
/// rather than crashing — the orchestrator continues with the built-in default).
pub fn load_pipeline_config_from_env(repo_root: &Path) -> Option<PipelineConfig> {
    let path_str = std::env::var("SWARM_PIPELINE_CONFIG").ok()?;
    let path = if std::path::Path::new(&path_str).is_absolute() {
        std::path::PathBuf::from(&path_str)
    } else {
        repo_root.join(&path_str)
    };

    match load_pipeline_config(&path) {
        Ok(cfg) => Some(cfg),
        Err(e) => {
            tracing::warn!(
                error = %e,
                path = %path.display(),
                "Failed to load SWARM_PIPELINE_CONFIG — using built-in defaults"
            );
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_toml(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_load_default_pipeline_stages_toml() {
        // Locate the repo root relative to this file's directory (crates/swarm-agents/src/)
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let repo_root = src_dir.parent().unwrap().parent().unwrap();
        let toml_path = repo_root.join("config/pipeline-stages.toml");

        if !toml_path.exists() {
            // Skip if running from a worktree that doesn't have the config dir
            return;
        }

        let cfg = load_pipeline_config(&toml_path).expect("should parse pipeline-stages.toml");
        assert_eq!(cfg.name(), "default");
        assert!(!cfg.stages().is_empty());
        assert!(cfg.stage("context_packing").is_some());
    }

    #[test]
    fn test_load_masai_pipeline_toml() {
        let src_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let repo_root = src_dir.parent().unwrap().parent().unwrap();
        let toml_path = repo_root.join("config/pipeline-masai.toml");

        if !toml_path.exists() {
            return;
        }

        let cfg = load_pipeline_config(&toml_path).expect("should parse pipeline-masai.toml");
        assert_eq!(cfg.name(), "masai_heterogeneous");
        assert!(cfg.stage("localize").is_some());
        assert!(cfg.stage("generate").is_some());

        let generate = cfg.stage("generate").unwrap();
        assert_eq!(generate.candidate_count, 2);
        assert!(generate.is_multi_candidate());

        let verify = cfg.stage("verify").unwrap();
        assert!(verify.is_verifier_stage());
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(PipelineStrategy::ReAct.to_string(), "ReAct");
        assert_eq!(PipelineStrategy::CoT.to_string(), "CoT");
        assert_eq!(PipelineStrategy::Vanilla.to_string(), "Vanilla");
    }

    #[test]
    fn test_validate_empty_stages_rejected() {
        let toml = r#"
[pipeline]
name = "empty"
description = "No stages"
stages = []
"#;
        let f = write_toml(toml);
        let result = load_pipeline_config(f.path());
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("no stages"), "unexpected error: {msg}");
    }

    #[test]
    fn test_validate_zero_max_turns_rejected() {
        let toml = r#"
[pipeline]
name = "bad"
description = "Stage with zero turns"

[[pipeline.stages]]
name = "oops"
enabled = true
model_role = "Scout"
strategy = "Vanilla"
tool_access = false
max_turns = 0
timeout_secs = 30
"#;
        let f = write_toml(toml);
        let result = load_pipeline_config(f.path());
        assert!(result.is_err());
        let msg = format!("{:#}", result.unwrap_err());
        assert!(msg.contains("max_turns = 0"), "unexpected error: {msg}");
    }

    #[test]
    fn test_disabled_stages_skipped() {
        let toml = r#"
[pipeline]
name = "partial"
description = "One disabled stage"

[[pipeline.stages]]
name = "active"
enabled = true
model_role = "Scout"
strategy = "Vanilla"
tool_access = false
max_turns = 1
timeout_secs = 30

[[pipeline.stages]]
name = "inactive"
enabled = false
model_role = "Scout"
strategy = "Vanilla"
tool_access = false
max_turns = 1
timeout_secs = 30
"#;
        let f = write_toml(toml);
        let cfg = load_pipeline_config(f.path()).unwrap();
        assert_eq!(cfg.stages().len(), 2);
        assert_eq!(cfg.enabled_stages().count(), 1);
        assert!(cfg.stage("inactive").is_none()); // disabled — not found
        assert!(cfg.stage("active").is_some());
    }

    #[test]
    fn test_default_candidate_count() {
        let toml = r#"
[pipeline]
name = "defaults"
description = "Test default candidate_count"

[[pipeline.stages]]
name = "gen"
enabled = true
model_role = "RustWorker"
strategy = "ReAct"
tool_access = true
max_turns = 5
timeout_secs = 120
"#;
        let f = write_toml(toml);
        let cfg = load_pipeline_config(f.path()).unwrap();
        let stage = cfg.stage("gen").unwrap();
        assert_eq!(stage.candidate_count, 1);
        assert!(!stage.is_multi_candidate());
    }

    #[test]
    fn test_serde_roundtrip() {
        let cfg = PipelineConfig {
            pipeline: PipelineInner {
                name: "test".into(),
                description: "roundtrip test".into(),
                stages: vec![StageConfig {
                    name: "s1".into(),
                    enabled: true,
                    model_role: "Scout".into(),
                    strategy: PipelineStrategy::ReAct,
                    tool_access: true,
                    max_turns: 5,
                    timeout_secs: 60,
                    candidate_count: 1,
                }],
            },
        };

        let serialized = toml::to_string(&cfg).unwrap();
        let restored: PipelineConfig = toml::from_str(&serialized).unwrap();
        assert_eq!(restored.name(), "test");
        assert_eq!(restored.stages().len(), 1);
        assert_eq!(restored.stages()[0].strategy, PipelineStrategy::ReAct);
    }
}
