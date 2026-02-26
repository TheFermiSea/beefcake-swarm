//! NS-1.3: Common domain types for artifacts, critiques, strategies, and summaries.
//!
//! These types form the typed handoff layer between the three new orchestration
//! modes (Contextual, Deepthink, Agentic) and any downstream consumers.
//!
//! ## Key types
//!
//! | Type                | Produced by      | Consumed by                 |
//! |---------------------|------------------|-----------------------------|
//! | `Artifact`          | Generator agent  | Critique agent, final merge |
//! | `CritiqueVerdict`   | Critique agent   | Contextual FSM              |
//! | `Strategy`          | Strategy agent   | Deepthink worker pool       |
//! | `StrategyOutcome`   | Worker sub-agent | Judge agent                 |
//! | `SynthesisResult`   | Judge agent      | Deepthink pipeline output   |
//! | `CompactionSummary` | Memory agent     | All modes (history trim)    |
//! | `ModeOutcome`       | Any mode         | Orchestrator                |

use std::time::Duration;

use serde::{Deserialize, Serialize};

// ── Artifact ────────────────────────────────────────────────────────────────

/// A code artifact produced or refined by a generator agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// The full text content of the artifact (code, patch, explanation, etc.).
    pub content: String,
    /// Language / media-type hint (e.g. `"rust"`, `"unified-diff"`).
    pub language: Option<String>,
    /// Generation iteration that produced this artifact (0-indexed).
    pub iteration: u32,
    /// Tokens consumed to produce this artifact, if reported by the backend.
    pub tokens_used: Option<u64>,
}

impl Artifact {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            language: None,
            iteration: 0,
            tokens_used: None,
        }
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = Some(lang.into());
        self
    }

    pub fn with_iteration(mut self, iteration: u32) -> Self {
        self.iteration = iteration;
        self
    }
}

// ── Critique ────────────────────────────────────────────────────────────────

/// Verdict returned by a critique agent after evaluating an `Artifact`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueVerdict {
    /// Artifact passes all critique checks — proceed to done/merge.
    Approved,
    /// Artifact requires specific changes before it can be approved.
    NeedsRevision {
        /// Actionable feedback the generator must address.
        feedback: String,
        /// Whether the critique considers this revision minor or major.
        severity: CritiqueSeverity,
    },
    /// Artifact is fundamentally flawed; restart from scratch is recommended.
    Rejected { reason: String },
}

impl CritiqueVerdict {
    /// `true` if the artifact was approved (no further refinement needed).
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }

    /// `true` if the artifact can be refined (approved or needs revision).
    pub fn is_refinable(&self) -> bool {
        !matches!(self, Self::Rejected { .. })
    }

    /// Parse the literal `"APPROVED"` sentinel used by many critique prompts.
    pub fn from_llm_response(raw: &str) -> Self {
        let trimmed = raw.trim();
        if trimmed.eq_ignore_ascii_case("approved") {
            Self::Approved
        } else {
            Self::NeedsRevision {
                feedback: trimmed.to_string(),
                severity: CritiqueSeverity::Major,
            }
        }
    }
}

/// How severe a `NeedsRevision` critique is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueSeverity {
    /// Small issues — style, minor naming, trivial fixes.
    Minor,
    /// Substantial logic or safety issues that must be corrected.
    Major,
}

// ── Strategy (Deepthink) ─────────────────────────────────────────────────────

/// A single approach produced by the strategy agent in Deepthink mode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Strategy {
    /// Human-readable label for this strategy (e.g. `"Strategy A"`).
    pub label: String,
    /// Detailed description / prompt fragment that the worker sub-agent will execute.
    pub description: String,
    /// Zero-based index within the strategy set (used for JoinSet bookkeeping).
    pub index: usize,
}

/// Outcome from a single Deepthink worker sub-agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyOutcome {
    /// The strategy that was executed.
    pub strategy: Strategy,
    /// Whether the worker succeeded in producing an artifact.
    pub result: StrategyResult,
    /// Wall-clock time spent on this strategy.
    pub elapsed: Duration,
}

/// Result of a strategy execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StrategyResult {
    /// Worker produced a valid artifact.
    Success(Artifact),
    /// Worker encountered an error.
    Failure(String),
}

impl StrategyOutcome {
    pub fn artifact(&self) -> Option<&Artifact> {
        match &self.result {
            StrategyResult::Success(a) => Some(a),
            StrategyResult::Failure(_) => None,
        }
    }

    pub fn is_success(&self) -> bool {
        matches!(self.result, StrategyResult::Success(_))
    }
}

/// Final output of the Deepthink judge synthesis phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynthesisResult {
    /// The winning or synthesized artifact.
    pub artifact: Artifact,
    /// Which strategy (if any) was selected as the primary winner.
    pub winning_strategy: Option<Strategy>,
    /// Number of strategies that ran successfully (out of total attempted).
    pub successful_strategies: usize,
    /// Total strategies attempted.
    pub total_strategies: usize,
    /// Judge's explanation for the synthesis decision.
    pub rationale: String,
}

// ── Compaction / Memory ──────────────────────────────────────────────────────

/// Output of the memory/compaction agent — a dense summary of prior history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionSummary {
    /// Dense bullet-point summary of the trimmed history segment.
    pub summary: String,
    /// Approximate token count of the history segment that was compacted.
    pub tokens_compacted: u64,
    /// Approximate token count of this summary (the replacement).
    pub tokens_summary: u64,
    /// Number of messages that were compacted into this summary.
    pub messages_compacted: usize,
}

impl CompactionSummary {
    /// Compression ratio: how much smaller the summary is vs. the original.
    pub fn compression_ratio(&self) -> f64 {
        if self.tokens_compacted == 0 {
            1.0
        } else {
            self.tokens_summary as f64 / self.tokens_compacted as f64
        }
    }
}

// ── Mode outcome ─────────────────────────────────────────────────────────────

/// Terminal outcome of any mode run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status")]
pub enum ModeOutcome {
    /// Mode completed successfully and produced a final artifact.
    Success {
        artifact: Artifact,
        iterations: u32,
        /// Total tokens consumed across all agents in this run.
        total_tokens: Option<u64>,
    },
    /// Mode failed after exhausting retries or hitting a terminal error.
    Failure {
        reason: String,
        iterations: u32,
        /// Last artifact produced before failure, if any.
        partial_artifact: Option<Artifact>,
    },
}

impl ModeOutcome {
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success { .. })
    }

    pub fn artifact(&self) -> Option<&Artifact> {
        match self {
            Self::Success { artifact, .. } => Some(artifact),
            Self::Failure {
                partial_artifact, ..
            } => partial_artifact.as_ref(),
        }
    }

    pub fn iterations(&self) -> u32 {
        match self {
            Self::Success { iterations, .. } | Self::Failure { iterations, .. } => *iterations,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn critique_verdict_from_approved() {
        let v = CritiqueVerdict::from_llm_response("APPROVED");
        assert!(v.is_approved());
    }

    #[test]
    fn critique_verdict_from_approved_lowercase() {
        let v = CritiqueVerdict::from_llm_response("approved");
        assert!(v.is_approved());
    }

    #[test]
    fn critique_verdict_needs_revision() {
        let v = CritiqueVerdict::from_llm_response("Fix the borrow checker error on line 42.");
        assert!(!v.is_approved());
        assert!(v.is_refinable());
    }

    #[test]
    fn compaction_summary_compression_ratio() {
        let s = CompactionSummary {
            summary: "summary".into(),
            tokens_compacted: 1000,
            tokens_summary: 100,
            messages_compacted: 20,
        };
        assert!((s.compression_ratio() - 0.1).abs() < f64::EPSILON);
    }

    #[test]
    fn mode_outcome_success_has_artifact() {
        let outcome = ModeOutcome::Success {
            artifact: Artifact::new("fn main() {}"),
            iterations: 3,
            total_tokens: Some(2048),
        };
        assert!(outcome.is_success());
        assert!(outcome.artifact().is_some());
    }

    #[test]
    fn strategy_outcome_failure_no_artifact() {
        let outcome = StrategyOutcome {
            strategy: Strategy {
                label: "A".into(),
                description: "try something".into(),
                index: 0,
            },
            result: StrategyResult::Failure("timeout".into()),
            elapsed: Duration::from_secs(5),
        };
        assert!(!outcome.is_success());
        assert!(outcome.artifact().is_none());
    }
}
