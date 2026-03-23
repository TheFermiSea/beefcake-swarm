//! Issue triage phase — lightweight pre-classification using cheap cloud models.
//!
//! Before the orchestrator creates a worktree or claims an issue, a triage step
//! classifies the issue by complexity, language, and suggested models. This enables
//! the phase-based model selector to route each workflow phase to the best-fit model.
//!
//! Cost: ~$0.001 per issue using Haiku or Gemini Flash (cheapest "triage" models).

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openai;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::config::{CloudEndpoint, CloudModelCatalog, CloudModelEntry};

/// Complexity classification for an issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Complexity {
    /// Single-file, straightforward fix (lint, typo, import, doc comment).
    Simple,
    /// Multi-file but bounded scope (borrow checker, trait bounds, refactor within module).
    Medium,
    /// Cross-module, architectural, or multi-concern changes.
    Complex,
    /// Security, breaking API, or correctness-critical changes requiring consensus.
    Critical,
}

impl std::fmt::Display for Complexity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Complexity::Simple => write!(f, "simple"),
            Complexity::Medium => write!(f, "medium"),
            Complexity::Complex => write!(f, "complex"),
            Complexity::Critical => write!(f, "critical"),
        }
    }
}

/// Result of the triage phase — used by the phase-based model selector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriageResult {
    /// Estimated complexity.
    pub complexity: Complexity,
    /// Primary language detected (e.g., "rust", "python", "typescript").
    pub language: String,
    /// Model IDs suggested for implementation phase.
    pub suggested_models: Vec<String>,
    /// Free-form reasoning from the triage model.
    pub reasoning: String,
    /// Whether LLM-based triage was used (false = keyword fallback).
    pub used_llm: bool,
    /// Model that performed the triage (if LLM-based).
    pub triage_model: Option<String>,
}

impl Default for TriageResult {
    fn default() -> Self {
        Self {
            complexity: Complexity::Medium,
            language: "rust".into(),
            suggested_models: vec![],
            reasoning: "default (no triage performed)".into(),
            used_llm: false,
            triage_model: None,
        }
    }
}

/// Workflow phases for the phase-based model selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkflowPhase {
    /// Pre-classification of issue complexity and language.
    Triage,
    /// Codebase exploration and context gathering.
    Explore,
    /// Plan generation and task decomposition.
    Plan,
    /// Code implementation.
    Implement,
    /// Code review (must differ from implementer).
    Review,
}

impl std::fmt::Display for WorkflowPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WorkflowPhase::Triage => write!(f, "triage"),
            WorkflowPhase::Explore => write!(f, "explore"),
            WorkflowPhase::Plan => write!(f, "plan"),
            WorkflowPhase::Implement => write!(f, "implement"),
            WorkflowPhase::Review => write!(f, "review"),
        }
    }
}

/// Phase-based model selector — replaces the fixed SwarmStackProfile.
///
/// Each workflow phase queries the CloudModelCatalog for the best-fit model,
/// using cost and capability matching. Falls back to local workers when cloud
/// is unavailable or the cost budget is exceeded.
///
/// Uses `AtomicU64` for budget tracking so it is safe for concurrent use
/// without locks. The budget is stored as the raw bit pattern of an `f64`;
/// `u64::MAX` represents "unlimited".
#[derive(Debug)]
pub struct PhaseModelSelector {
    catalog: CloudModelCatalog,
    /// Budget in USD stored as f64 bits. u64::MAX = unlimited.
    cost_budget_bits: AtomicU64,
}

impl Clone for PhaseModelSelector {
    fn clone(&self) -> Self {
        Self {
            catalog: self.catalog.clone(),
            cost_budget_bits: AtomicU64::new(self.cost_budget_bits.load(Ordering::Relaxed)),
        }
    }
}

impl PhaseModelSelector {
    pub fn new(catalog: CloudModelCatalog, max_cost: f64) -> Self {
        let bits = if max_cost > 0.0 { max_cost.to_bits() } else { u64::MAX };
        Self { catalog, cost_budget_bits: AtomicU64::new(bits) }
    }

    /// Select the best cloud model for a given workflow phase.
    ///
    /// Returns `None` when no cloud model is available (budget exhausted,
    /// no models with the capability, etc.) — caller should fall back to local.
    pub fn select_for_phase(
        &self,
        phase: WorkflowPhase,
        triage: Option<&TriageResult>,
        implementer_model: Option<&str>,
    ) -> Option<&CloudModelEntry> {
        let capability = phase.to_string();

        let candidate = match phase {
            WorkflowPhase::Triage => self.catalog.cheapest_for(&capability),
            WorkflowPhase::Explore | WorkflowPhase::Plan => {
                self.catalog.strongest_for(&capability)
            }
            WorkflowPhase::Implement => {
                // Use triage suggestion if available; otherwise strongest implementer.
                if let Some(triage) = triage {
                    // Try triage-suggested models first (bypasses budget check).
                    let suggested = triage.suggested_models.iter().find_map(|id| {
                        self.catalog.models.iter().find(|m| m.model == *id)
                    });
                    if suggested.is_some() {
                        return suggested;
                    }
                    // Cost-aware fallback: simple tasks use cheaper models.
                    match triage.complexity {
                        Complexity::Simple => self.catalog.cheapest_for(&capability),
                        _ => self.catalog.strongest_for(&capability),
                    }
                } else {
                    self.catalog.strongest_for(&capability)
                }
            }
            WorkflowPhase::Review => {
                // Must differ from the implementer model for diversity.
                self.catalog
                    .with_capability(&capability)
                    .into_iter()
                    .filter(|m| implementer_model.is_none_or(|imp| m.model != imp))
                    .max_by_key(|m| m.capability_score)
            }
        };

        // Budget check: reject if estimated cost exceeds remaining budget.
        // Conservative estimate: 50K input tokens + 4K output tokens per phase call.
        let candidate = candidate?;
        let bits = self.cost_budget_bits.load(Ordering::Relaxed);
        if bits != u64::MAX {
            let remaining = f64::from_bits(bits);
            let estimated_cost =
                (50_000.0 * candidate.cost_input_per_m + 4_000.0 * candidate.cost_output_per_m)
                    / 1_000_000.0;
            if estimated_cost > remaining {
                info!(
                    model = %candidate.model,
                    estimated_cost,
                    remaining,
                    "Budget exceeded for phase {phase} — falling back to local"
                );
                return None;
            }
        }
        Some(candidate)
    }

    /// Record cost spent, reducing the remaining budget.
    ///
    /// Uses an atomic compare-exchange loop so concurrent callers don't race.
    pub fn record_cost(&self, cost_usd: f64) {
        let _ = self.cost_budget_bits.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |bits| {
                if bits == u64::MAX {
                    None // Unlimited — nothing to update.
                } else {
                    let remaining = f64::from_bits(bits);
                    Some((remaining - cost_usd).max(0.0).to_bits())
                }
            },
        );
    }

    /// Check if any budget remains (or if budgeting is disabled).
    pub fn has_budget(&self) -> bool {
        let bits = self.cost_budget_bits.load(Ordering::Relaxed);
        bits == u64::MAX || f64::from_bits(bits) > 0.0
    }
}

/// Run the triage phase on an issue.
///
/// Attempts LLM-based triage using the cheapest "triage"-capable model from
/// the catalog. Falls back to keyword-based heuristics if cloud is unavailable
/// or the triage call fails.
///
/// When `cloud_client` is provided the call goes through Rig's `.prompt()` API,
/// which participates in OpenTelemetry tracing and retry middleware. When it is
/// `None`, the function falls back to a raw reqwest call (legacy path).
pub async fn triage_issue(
    title: &str,
    description: Option<&str>,
    cloud: Option<&CloudEndpoint>,
    catalog: &CloudModelCatalog,
    cloud_client: Option<&openai::CompletionsClient>,
) -> TriageResult {
    // Skip LLM triage if cloud is unavailable.
    let Some(cloud) = cloud else {
        debug!("No cloud endpoint — using keyword triage");
        return keyword_triage(title, description);
    };

    // Find the cheapest triage-capable model.
    let Some(triage_model) = catalog.cheapest_for("triage") else {
        warn!("No triage-capable model in catalog — using keyword triage");
        return keyword_triage(title, description);
    };

    // Skip LLM triage if explicitly disabled.
    if std::env::var("SWARM_SKIP_TRIAGE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        debug!("SWARM_SKIP_TRIAGE=1 — using keyword triage");
        return keyword_triage(title, description);
    }

    let desc_snippet = description
        .unwrap_or("")
        .chars()
        .take(500)
        .collect::<String>();

    let prompt_text = format!(
        r#"You are a code issue classifier. Analyze this issue and respond with ONLY a JSON object (no markdown, no explanation).

Issue title: {title}
Issue description: {desc_snippet}

Respond with exactly this JSON structure:
{{"complexity": "simple"|"medium"|"complex"|"critical", "language": "<primary language>", "suggested_models": [], "reasoning": "<one sentence>"}}

Guidelines:
- simple: single-file lint fix, typo, doc comment, import fix
- medium: multi-file bounded change, borrow checker fix, trait implementation
- complex: cross-module refactor, architecture change, new feature with tests
- critical: security fix, breaking API change, data integrity
- language: detect from title/description context (default: "rust")
- suggested_models: leave empty (system will fill based on complexity)"#
    );

    // Prefer the pre-built Rig client (participates in OTel tracing + retry middleware).
    // Fall back to the raw reqwest path when the client isn't available.
    let call_result: Result<String> = if let Some(client) = cloud_client {
        let agent = client.agent(&triage_model.model).build();
        agent
            .prompt(prompt_text.as_str())
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))
    } else {
        call_triage_model(cloud, &triage_model.model, &prompt_text).await
    };

    match call_result {
        Ok(response) => match parse_triage_response(&response) {
            Ok(mut result) => {
                result.used_llm = true;
                result.triage_model = Some(triage_model.model.clone());
                info!(
                    complexity = %result.complexity,
                    language = %result.language,
                    model = %triage_model.model,
                    "LLM triage complete"
                );
                result
            }
            Err(e) => {
                warn!(error = %e, "Failed to parse triage response — falling back to keywords");
                keyword_triage(title, description)
            }
        },
        Err(e) => {
            warn!(error = %e, "Triage model call failed — falling back to keywords");
            keyword_triage(title, description)
        }
    }
}

/// Call the triage model via the cloud endpoint (CLIAPIProxy).
async fn call_triage_model(
    cloud: &CloudEndpoint,
    model: &str,
    prompt: &str,
) -> Result<String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;

    let url = format!("{}/chat/completions", cloud.url.trim_end_matches('/'));

    let body = serde_json::json!({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "temperature": 0.0,
    });

    let mut request = client.post(&url).json(&body);
    // CLIAPIProxy uses x-api-key header.
    request = request.header("x-api-key", &cloud.api_key);

    let response = request.send().await.context("triage HTTP request failed")?;

    if !response.status().is_success() {
        anyhow::bail!(
            "triage model returned HTTP {}",
            response.status()
        );
    }

    let json: serde_json::Value = response.json().await?;
    let content = json["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    Ok(content)
}

/// Parse the JSON response from the triage model.
fn parse_triage_response(response: &str) -> Result<TriageResult> {
    // Strip markdown code fences if present.
    let cleaned = response
        .trim()
        .strip_prefix("```json")
        .or_else(|| response.trim().strip_prefix("```"))
        .unwrap_or(response.trim());
    let cleaned = cleaned
        .strip_suffix("```")
        .unwrap_or(cleaned)
        .trim();

    #[derive(Deserialize)]
    struct RawTriage {
        complexity: String,
        language: String,
        #[serde(default)]
        suggested_models: Vec<String>,
        #[serde(default)]
        reasoning: String,
    }

    let raw: RawTriage = serde_json::from_str(cleaned)
        .context("failed to parse triage JSON")?;

    let complexity = match raw.complexity.as_str() {
        "simple" => Complexity::Simple,
        "medium" => Complexity::Medium,
        "complex" => Complexity::Complex,
        "critical" => Complexity::Critical,
        other => {
            warn!(value = other, "Unknown complexity — defaulting to medium");
            Complexity::Medium
        }
    };

    Ok(TriageResult {
        complexity,
        language: raw.language,
        suggested_models: raw.suggested_models,
        reasoning: raw.reasoning,
        used_llm: false, // Caller sets this.
        triage_model: None,
    })
}

/// Keyword-based triage fallback — no LLM call required.
///
/// Uses the same keyword lists as coordination's `classify_initial_tier()` but
/// returns a TriageResult instead of a tier recommendation.
fn keyword_triage(title: &str, description: Option<&str>) -> TriageResult {
    let combined = format!(
        "{} {}",
        title.to_lowercase(),
        description.unwrap_or("").to_lowercase()
    );

    // Language detection.
    let language = if combined.contains("cargo")
        || combined.contains("rustc")
        || combined.contains(".rs")
        || combined.contains("clippy")
        || combined.contains("borrow")
        || combined.contains("lifetime")
    {
        "rust"
    } else if combined.contains("python")
        || combined.contains(".py")
        || combined.contains("pip")
        || combined.contains("pytest")
    {
        "python"
    } else if combined.contains("typescript")
        || combined.contains(".ts")
        || combined.contains("npm")
        || combined.contains("node")
    {
        "typescript"
    } else if combined.contains(".go") || combined.contains("golang") {
        "go"
    } else {
        "rust" // Default for this project.
    };

    // Complexity detection.
    let simple_keywords = [
        "lint", "format", "clippy", "import", "typo", "rename", "doc comment",
        "unused", "dead_code", "allow(", "warn(", "derive(",
    ];
    let complex_keywords = [
        "refactor", "architecture", "async", "migration", "breaking",
        "redesign", "multi-file", "cross-module", "new feature",
    ];
    let critical_keywords = [
        "security", "vulnerability", "injection", "auth", "breaking api",
        "data loss", "corruption",
    ];

    let simple_hits = simple_keywords.iter().filter(|k| combined.contains(*k)).count();
    let complex_hits = complex_keywords.iter().filter(|k| combined.contains(*k)).count();
    let critical_hits = critical_keywords.iter().filter(|k| combined.contains(*k)).count();

    let complexity = if critical_hits > 0 {
        Complexity::Critical
    } else if complex_hits >= 2 || (complex_hits == 1 && simple_hits == 0) {
        Complexity::Complex
    } else if simple_hits >= 1 {
        Complexity::Simple
    } else {
        Complexity::Medium
    };

    TriageResult {
        complexity,
        language: language.into(),
        suggested_models: vec![],
        reasoning: format!("keyword heuristic (simple={simple_hits}, complex={complex_hits}, critical={critical_hits})"),
        used_llm: false,
        triage_model: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyword_triage_simple() {
        let result = keyword_triage("Fix clippy warning in config.rs", None);
        assert_eq!(result.complexity, Complexity::Simple);
        assert_eq!(result.language, "rust");
        assert!(!result.used_llm);
    }

    #[test]
    fn keyword_triage_complex() {
        let result = keyword_triage(
            "Refactor architecture for async migration",
            Some("Multi-file cross-module changes needed"),
        );
        assert_eq!(result.complexity, Complexity::Complex);
    }

    #[test]
    fn keyword_triage_critical() {
        let result = keyword_triage("Fix SQL injection vulnerability in auth", None);
        assert_eq!(result.complexity, Complexity::Critical);
    }

    #[test]
    fn keyword_triage_medium_default() {
        let result = keyword_triage("Implement new endpoint for user settings", None);
        assert_eq!(result.complexity, Complexity::Medium);
    }

    #[test]
    fn keyword_triage_python() {
        let result = keyword_triage("Fix pytest failure in data_pipeline.py", None);
        assert_eq!(result.language, "python");
    }

    #[test]
    fn phase_selector_cheapest_for_triage() {
        let catalog = CloudModelCatalog::default_catalog();
        let selector = PhaseModelSelector::new(catalog, 0.0);
        let model = selector.select_for_phase(WorkflowPhase::Triage, None, None);
        assert!(model.is_some());
        // Should pick the cheapest triage model (Gemini Flash at $0.10/M).
        let m = model.unwrap();
        assert!(m.cost_input_per_m <= 0.15, "Expected cheap triage model, got {}", m.model);
    }

    #[test]
    fn phase_selector_strongest_for_plan() {
        let catalog = CloudModelCatalog::default_catalog();
        let selector = PhaseModelSelector::new(catalog, 0.0);
        let model = selector.select_for_phase(WorkflowPhase::Plan, None, None);
        assert!(model.is_some());
        // Should pick the most expensive plan-capable model (Opus at $15/M).
        let m = model.unwrap();
        assert_eq!(m.model, "claude-opus-4-6");
    }

    #[test]
    fn phase_selector_review_differs_from_implementer() {
        let catalog = CloudModelCatalog::default_catalog();
        let selector = PhaseModelSelector::new(catalog, 0.0);
        let model = selector.select_for_phase(
            WorkflowPhase::Review,
            None,
            Some("claude-opus-4-6"),
        );
        assert!(model.is_some());
        assert_ne!(model.unwrap().model, "claude-opus-4-6");
    }

    #[test]
    fn phase_selector_budget_exhausted() {
        let catalog = CloudModelCatalog::default_catalog();
        let selector = PhaseModelSelector::new(catalog, 0.0001); // Tiny budget.
        let model = selector.select_for_phase(WorkflowPhase::Plan, None, None);
        // Opus costs ~$0.80 per call estimate — should be rejected.
        assert!(model.is_none());
    }

    #[test]
    fn parse_triage_json() {
        let json = r#"{"complexity": "simple", "language": "rust", "suggested_models": [], "reasoning": "lint fix"}"#;
        let result = parse_triage_response(json).unwrap();
        assert_eq!(result.complexity, Complexity::Simple);
        assert_eq!(result.language, "rust");
    }

    #[test]
    fn parse_triage_json_with_fences() {
        let json = "```json\n{\"complexity\": \"complex\", \"language\": \"python\", \"suggested_models\": [], \"reasoning\": \"arch change\"}\n```";
        let result = parse_triage_response(json).unwrap();
        assert_eq!(result.complexity, Complexity::Complex);
        assert_eq!(result.language, "python");
    }
}
