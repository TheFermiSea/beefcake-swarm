//! Work Packet types — structured context for model tier handoffs

use crate::analytics::replay::ReplayHint;
use crate::analytics::skills::SkillHint;
use crate::escalation::state::SwarmTier;
use crate::feedback::error_parser::ErrorCategory;
use crate::verifier::report::FailureSignal;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A Work Packet — self-contained context for a model tier to act on
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkPacket {
    /// Beads issue ID
    pub bead_id: String,
    /// Git branch for this work
    pub branch: String,
    /// Git commit SHA at time of packet generation
    pub checkpoint: String,
    /// Human-readable objective
    pub objective: String,
    /// Files that have been modified
    pub files_touched: Vec<String>,
    /// Key symbols (structs, traits, functions) relevant to the task
    pub key_symbols: Vec<KeySymbol>,
    /// File context — relevant code snippets
    pub file_contexts: Vec<FileContext>,
    /// Verification gates that must pass
    pub verification_gates: Vec<String>,
    /// Current failure signals from the Verifier
    pub failure_signals: Vec<FailureSignal>,
    /// Constraints the model must respect
    pub constraints: Vec<Constraint>,
    /// Current iteration number
    pub iteration: u32,
    /// Which tier is receiving this packet
    pub target_tier: SwarmTier,
    /// Why this packet was generated (escalation reason or initial assignment)
    pub escalation_reason: Option<String>,
    /// Error categories encountered so far
    pub error_history: Vec<ErrorCategory>,
    /// Previous fix attempts (brief descriptions)
    pub previous_attempts: Vec<String>,
    /// Structured iteration deltas (last 2-3 iterations).
    /// Captures what changed between iterations rather than flat attempt strings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub iteration_deltas: Vec<IterationDelta>,
    /// Heuristics from the knowledge base relevant to this task
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant_heuristics: Vec<String>,
    /// Playbook entries relevant to the current error categories
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant_playbooks: Vec<String>,
    /// Decision journal entries for this branch
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub decisions: Vec<String>,
    /// Timestamp of packet generation
    pub generated_at: DateTime<Utc>,
    /// Maximum LOC change allowed in the response
    pub max_patch_loc: u32,
    /// Chain of delegation steps (manager-to-manager handoffs)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub delegation_chain: Vec<DelegationStep>,
    /// Skill hints from the skill library — approaches that worked for similar tasks
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub skill_hints: Vec<SkillHint>,
    /// Replay hints from the experience trace index — past successful session traces
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub replay_hints: Vec<ReplayHint>,
}

impl WorkPacket {
    /// Estimate token count of the serialized packet.
    ///
    /// Uses character count (not byte count) for accuracy with multi-byte
    /// UTF-8 content, plus a 10% safety margin to avoid context overflow.
    pub fn estimated_tokens(&self) -> usize {
        // ~4 chars per token for JSON, with 10% safety margin
        let json = serde_json::to_string(self).unwrap_or_default();
        let char_count = json.chars().count();
        (char_count as f64 * 1.1 / 4.0).ceil() as usize
    }

    /// Get a compact summary for logging
    pub fn summary(&self) -> String {
        format!(
            "WorkPacket[bead={}, branch={}, iter={}, tier={}, signals={}, files={}]",
            self.bead_id,
            self.branch,
            self.iteration,
            self.target_tier,
            self.failure_signals.len(),
            self.files_touched.len(),
        )
    }

    /// Check if this packet has failure signals
    pub fn has_failures(&self) -> bool {
        !self.failure_signals.is_empty()
    }

    /// Get unique error categories from failure signals
    pub fn unique_error_categories(&self) -> Vec<ErrorCategory> {
        let mut cats: Vec<ErrorCategory> = self
            .failure_signals
            .iter()
            .map(|s| s.category)
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        cats.sort_by_key(|c| c.to_string());
        cats
    }
}

/// A key symbol referenced in the task
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeySymbol {
    /// Symbol name (e.g., "StreamParser", "ParseError")
    pub name: String,
    /// Symbol kind (struct, trait, fn, enum, impl)
    pub kind: SymbolKind,
    /// File where this symbol is defined
    pub file: String,
    /// Line number
    pub line: Option<usize>,
}

/// Kind of source code symbol
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SymbolKind {
    Struct,
    Trait,
    Function,
    Enum,
    Impl,
    Const,
    Type,
    Mod,
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Struct => write!(f, "struct"),
            Self::Trait => write!(f, "trait"),
            Self::Function => write!(f, "fn"),
            Self::Enum => write!(f, "enum"),
            Self::Impl => write!(f, "impl"),
            Self::Const => write!(f, "const"),
            Self::Type => write!(f, "type"),
            Self::Mod => write!(f, "mod"),
        }
    }
}

/// File context — a relevant code snippet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileContext {
    /// File path relative to crate root
    pub file: String,
    /// Starting line number
    pub start_line: usize,
    /// Ending line number
    pub end_line: usize,
    /// The code content
    pub content: String,
    /// Why this context is relevant
    pub relevance: String,
    /// Trim priority: 0 = error context (highest, never trim first),
    /// 1 = modified file, 2 = structural/header, 3 = reference (lowest).
    #[serde(default = "default_priority")]
    pub priority: u8,
    /// How this context was sourced (for provenance tracking).
    #[serde(default)]
    pub provenance: ContextProvenance,
}

fn default_priority() -> u8 {
    2
}

/// How a FileContext was sourced — enables decay and deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContextProvenance {
    /// Extracted from a compiler error location.
    CompilerError,
    /// From a git diff of modified files.
    Diff,
    /// Dependency/import chain.
    Dependency,
    /// Imported based on usage/reference.
    Import,
    /// File header scan during initial pack.
    #[default]
    Header,
}

impl std::fmt::Display for ContextProvenance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::CompilerError => write!(f, "compiler_error"),
            Self::Diff => write!(f, "diff"),
            Self::Dependency => write!(f, "dependency"),
            Self::Import => write!(f, "import"),
            Self::Header => write!(f, "header"),
        }
    }
}

/// A constraint the model must respect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Constraint type
    pub kind: ConstraintKind,
    /// Description
    pub description: String,
}

/// Types of constraints
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintKind {
    /// Don't add new dependencies
    NoDeps,
    /// Don't break public API
    NoBreakingApi,
    /// Maximum LOC change
    MaxLoc,
    /// Must maintain backward compatibility
    BackwardCompat,
    /// Security constraint
    Security,
    /// Performance constraint
    Performance,
    /// Custom constraint
    Custom,
}

/// Structured delta between consecutive iterations.
///
/// Captures what changed between iterations (not just what happened),
/// enabling models to see "you fixed X but broke Y" patterns.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IterationDelta {
    /// Iteration number this delta describes.
    pub iteration: u32,
    /// Error categories that were fixed (improved) in this iteration.
    pub fixed_errors: Vec<ErrorCategory>,
    /// Error categories that were newly introduced (regressed).
    pub new_errors: Vec<ErrorCategory>,
    /// Files modified in this iteration.
    pub files_modified: Vec<String>,
    /// What the model claimed it was doing (extracted from response).
    pub hypothesis: Option<String>,
    /// Concise summary of the result (e.g., "borrow error fixed, but
    /// introduced lifetime error in return type").
    pub result_summary: String,
    /// Strategy used (e.g., "added Arc wrapper", "changed lifetime annotation").
    pub strategy_used: String,
}

/// Record of a delegation between managers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelegationStep {
    pub from_model: crate::state::types::ModelId,
    pub to_model: crate::state::types::ModelId,
    pub reason: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_work_packet_summary() {
        let packet = WorkPacket {
            bead_id: "beads-123".to_string(),
            branch: "feat/parser".to_string(),
            checkpoint: "abc123f".to_string(),
            objective: "Implement streaming JSON parser".to_string(),
            files_touched: vec!["src/parser.rs".to_string()],
            key_symbols: vec![],
            file_contexts: vec![],
            verification_gates: vec!["check".to_string(), "test".to_string()],
            failure_signals: vec![],
            constraints: vec![],
            iteration: 3,
            target_tier: SwarmTier::Worker,
            escalation_reason: None,
            error_history: vec![],
            previous_attempts: vec![],
            relevant_heuristics: vec![],
            relevant_playbooks: vec![],
            decisions: vec![],
            generated_at: Utc::now(),
            max_patch_loc: 150,
            iteration_deltas: vec![],
            delegation_chain: vec![],
            skill_hints: vec![],
            replay_hints: vec![],
        };

        let summary = packet.summary();
        assert!(summary.contains("beads-123"));
        assert!(summary.contains("feat/parser"));
        assert!(summary.contains("iter=3"));
    }

    #[test]
    fn test_work_packet_serialization() {
        let packet = WorkPacket {
            bead_id: "beads-456".to_string(),
            branch: "fix/lifetime".to_string(),
            checkpoint: "def456".to_string(),
            objective: "Fix lifetime error in parser".to_string(),
            files_touched: vec!["src/parser.rs".to_string()],
            key_symbols: vec![KeySymbol {
                name: "Parser".to_string(),
                kind: SymbolKind::Struct,
                file: "src/parser.rs".to_string(),
                line: Some(10),
            }],
            file_contexts: vec![],
            verification_gates: vec!["check".to_string()],
            failure_signals: vec![],
            constraints: vec![Constraint {
                kind: ConstraintKind::NoDeps,
                description: "No new dependencies".to_string(),
            }],
            iteration: 1,
            target_tier: SwarmTier::Worker,
            escalation_reason: None,
            error_history: vec![],
            previous_attempts: vec![],
            relevant_heuristics: vec![],
            relevant_playbooks: vec![],
            decisions: vec![],
            generated_at: Utc::now(),
            max_patch_loc: 150,
            iteration_deltas: vec![],
            delegation_chain: vec![],
            skill_hints: vec![],
            replay_hints: vec![],
        };

        let json = serde_json::to_string_pretty(&packet).unwrap();
        assert!(json.contains("beads-456"));
        assert!(json.contains("Parser"));

        // Round-trip
        let deserialized: WorkPacket = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.bead_id, "beads-456");
    }

    #[test]
    fn test_estimated_tokens() {
        let packet = WorkPacket {
            bead_id: "beads-1".to_string(),
            branch: "main".to_string(),
            checkpoint: "abc".to_string(),
            objective: "Test".to_string(),
            files_touched: vec![],
            key_symbols: vec![],
            file_contexts: vec![],
            verification_gates: vec![],
            failure_signals: vec![],
            constraints: vec![],
            iteration: 1,
            target_tier: SwarmTier::Worker,
            escalation_reason: None,
            error_history: vec![],
            previous_attempts: vec![],
            relevant_heuristics: vec![],
            relevant_playbooks: vec![],
            decisions: vec![],
            generated_at: Utc::now(),
            max_patch_loc: 150,
            iteration_deltas: vec![],
            delegation_chain: vec![],
            skill_hints: vec![],
            replay_hints: vec![],
        };

        // Should be a reasonable estimate
        let tokens = packet.estimated_tokens();
        assert!(tokens > 0);
        assert!(tokens < 10000);
    }
}
