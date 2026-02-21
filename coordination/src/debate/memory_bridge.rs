//! Debate-memory integration — connects debate turns to the compaction substrate.
//!
//! Records debate events (coder outputs, reviewer checks, phase transitions)
//! as memory entries, and triggers compaction when appropriate.

use crate::memory::compactor::{CompactionEvent, CompactionTriggerKind, MemoryCompactor};
use crate::memory::errors::CompactionError;
use crate::memory::store::{MemoryEntry, MemoryEntryKind, SwarmMemory};
use crate::memory::summarizer::Summarizer;

use super::consensus::ConsensusCheck;
use super::critique::PatchCritique;
use super::orchestrator::CoderOutput;
use super::state::DebatePhase;

/// Records debate events into the memory store and manages compaction.
pub struct DebateMemoryBridge {
    compactor: MemoryCompactor,
    auto_compact: bool,
}

impl DebateMemoryBridge {
    /// Create a new bridge with default compaction settings.
    pub fn new(session_context: &str) -> Self {
        Self {
            compactor: MemoryCompactor::new(session_context),
            auto_compact: true,
        }
    }

    /// Create with a custom compactor.
    pub fn with_compactor(compactor: MemoryCompactor, auto_compact: bool) -> Self {
        Self {
            compactor,
            auto_compact,
        }
    }

    /// Record a coder turn output into memory.
    pub fn record_coder_turn(
        &self,
        store: &mut dyn SwarmMemory,
        output: &CoderOutput,
        round: u32,
        tokens: u32,
    ) -> u64 {
        let content = format!(
            "Round {} coder output: {} files changed. {}",
            round,
            output.files_changed.len(),
            output.explanation
        );
        let entry = MemoryEntry::new(MemoryEntryKind::AgentTurn, &content, "coder", tokens);
        store.append(entry)
    }

    /// Record a reviewer consensus check into memory.
    pub fn record_reviewer_turn(
        &self,
        store: &mut dyn SwarmMemory,
        check: &ConsensusCheck,
        round: u32,
        tokens: u32,
    ) -> u64 {
        let content = format!(
            "Round {} review: verdict={}, confidence={:.2}, blocking_issues={}",
            round,
            check.verdict,
            check.confidence,
            check.blocking_issues.len()
        );
        let entry = MemoryEntry::new(MemoryEntryKind::AgentTurn, &content, "reviewer", tokens);
        store.append(entry)
    }

    /// Record a critique into memory (more detailed than the consensus check).
    pub fn record_critique(
        &self,
        store: &mut dyn SwarmMemory,
        critique: &PatchCritique,
        tokens: u32,
    ) -> u64 {
        let content = format!(
            "Round {} critique: {} blocking, {} warnings. Assessment: {}",
            critique.round,
            critique.blocking_count(),
            critique.non_blocking_count(),
            critique.overall_assessment
        );
        let entry = MemoryEntry::new(MemoryEntryKind::AgentTurn, &content, "reviewer", tokens);
        store.append(entry)
    }

    /// Record a phase transition into memory.
    pub fn record_phase_change(
        &self,
        store: &mut dyn SwarmMemory,
        from: DebatePhase,
        to: DebatePhase,
        reason: &str,
    ) -> u64 {
        let content = format!("Phase transition: {} → {} ({})", from, to, reason);
        let entry = MemoryEntry::new(MemoryEntryKind::AgentTurn, &content, "orchestrator", 20);
        store.append(entry)
    }

    /// Record an error context entry.
    pub fn record_error(
        &self,
        store: &mut dyn SwarmMemory,
        error: &str,
        round: u32,
        tokens: u32,
    ) -> u64 {
        let content = format!("Round {} error: {}", round, error);
        let entry = MemoryEntry::new(MemoryEntryKind::ErrorContext, &content, "verifier", tokens);
        store.append(entry)
    }

    /// Try to compact if the event warrants it. Returns compaction result on success.
    pub fn maybe_compact(
        &self,
        store: &mut dyn SwarmMemory,
        summarizer: &dyn Summarizer,
        event: CompactionEvent,
    ) -> Option<Result<crate::memory::compactor::CompactionResult, CompactionError>> {
        if !self.auto_compact {
            return None;
        }

        if !self.compactor.check_event(&event) {
            return None;
        }

        let trigger_kind = match &event {
            CompactionEvent::BudgetExceeded { .. } => CompactionTriggerKind::BudgetThreshold,
            _ => CompactionTriggerKind::Event,
        };

        Some(self.compactor.compact(store, summarizer, trigger_kind))
    }

    /// Force a budget check and compact if needed.
    pub fn check_and_compact(
        &self,
        store: &mut dyn SwarmMemory,
        summarizer: &dyn Summarizer,
    ) -> Option<Result<crate::memory::compactor::CompactionResult, CompactionError>> {
        let current_tokens = store.active_token_count();
        let decision = self.compactor.check_budget(current_tokens);

        if decision.should_compact() {
            Some(
                self.compactor
                    .compact(store, summarizer, CompactionTriggerKind::BudgetThreshold),
            )
        } else {
            None
        }
    }

    /// Get a reference to the compactor.
    pub fn compactor(&self) -> &MemoryCompactor {
        &self.compactor
    }
}

#[cfg(test)]
mod tests {
    use super::super::consensus::Verdict;
    use super::*;
    use crate::memory::budget::TokenBudget;
    use crate::memory::compactor::CompactionPolicy;
    use crate::memory::store::SwarmMemoryStore;
    use crate::memory::summarizer::MockSummarizer;

    fn make_coder_output() -> CoderOutput {
        CoderOutput {
            code: "fn main() {}".to_string(),
            files_changed: vec!["src/main.rs".to_string()],
            explanation: "Initial implementation".to_string(),
        }
    }

    fn make_consensus_check() -> ConsensusCheck {
        ConsensusCheck {
            verdict: Verdict::RequestChanges,
            confidence: 0.8,
            blocking_issues: vec!["missing error handling".to_string()],
            suggestions: vec![],
            approach_aligned: true,
        }
    }

    fn make_critique() -> PatchCritique {
        use super::super::critique::{CritiqueCategory, CritiqueItem};
        let mut critique = PatchCritique::new(1, "Needs improvement");
        critique.add_item(CritiqueItem::blocking(
            CritiqueCategory::ErrorHandling,
            "Missing Result type",
        ));
        critique
    }

    #[test]
    fn test_record_coder_turn() {
        let bridge = DebateMemoryBridge::new("test session");
        let mut store = SwarmMemoryStore::new();
        let output = make_coder_output();

        let seq = bridge.record_coder_turn(&mut store, &output, 1, 50);
        assert_eq!(seq, 1);

        let entry = store.get(1).unwrap();
        assert_eq!(entry.kind, MemoryEntryKind::AgentTurn);
        assert!(entry.content.contains("Round 1"));
        assert!(entry.content.contains("1 files changed"));
        assert_eq!(entry.source, "coder");
    }

    #[test]
    fn test_record_reviewer_turn() {
        let bridge = DebateMemoryBridge::new("test session");
        let mut store = SwarmMemoryStore::new();
        let check = make_consensus_check();

        let seq = bridge.record_reviewer_turn(&mut store, &check, 1, 40);
        assert_eq!(seq, 1);

        let entry = store.get(1).unwrap();
        assert!(entry.content.contains("verdict=request_changes"));
        assert!(entry.content.contains("blocking_issues=1"));
        assert_eq!(entry.source, "reviewer");
    }

    #[test]
    fn test_record_critique() {
        let bridge = DebateMemoryBridge::new("test session");
        let mut store = SwarmMemoryStore::new();
        let critique = make_critique();

        let seq = bridge.record_critique(&mut store, &critique, 30);
        assert_eq!(seq, 1);

        let entry = store.get(1).unwrap();
        assert!(entry.content.contains("1 blocking"));
        assert!(entry.content.contains("Needs improvement"));
    }

    #[test]
    fn test_record_phase_change() {
        let bridge = DebateMemoryBridge::new("test session");
        let mut store = SwarmMemoryStore::new();

        let seq = bridge.record_phase_change(
            &mut store,
            DebatePhase::CoderTurn,
            DebatePhase::ReviewerTurn,
            "code submitted",
        );
        assert_eq!(seq, 1);

        let entry = store.get(1).unwrap();
        assert!(entry.content.contains("coder_turn → reviewer_turn"));
        assert_eq!(entry.source, "orchestrator");
    }

    #[test]
    fn test_record_error() {
        let bridge = DebateMemoryBridge::new("test session");
        let mut store = SwarmMemoryStore::new();

        let seq = bridge.record_error(&mut store, "cargo check failed: E0308", 2, 25);
        assert_eq!(seq, 1);

        let entry = store.get(1).unwrap();
        assert_eq!(entry.kind, MemoryEntryKind::ErrorContext);
        assert!(entry.content.contains("Round 2 error"));
        assert!(entry.content.contains("E0308"));
    }

    #[test]
    fn test_memory_continuity_across_rounds() {
        let bridge = DebateMemoryBridge::new("test session");
        let mut store = SwarmMemoryStore::new();

        // Simulate a 3-round debate
        for round in 1..=3 {
            bridge.record_coder_turn(&mut store, &make_coder_output(), round, 50);
            bridge.record_reviewer_turn(&mut store, &make_consensus_check(), round, 40);
        }

        // All 6 entries should be active
        assert_eq!(store.active_entries().len(), 6);
        assert_eq!(store.active_token_count(), 270); // 3 * (50 + 40)

        // Entries should be in order
        let entries = store.active_entries();
        for i in 1..entries.len() {
            assert!(entries[i].seq > entries[i - 1].seq);
        }
    }

    #[test]
    fn test_auto_compact_disabled() {
        let bridge = DebateMemoryBridge::with_compactor(MemoryCompactor::new("test"), false);
        let mut store = SwarmMemoryStore::new();
        let summarizer = MockSummarizer::new();

        let result = bridge.maybe_compact(&mut store, &summarizer, CompactionEvent::SessionEnding);
        assert!(result.is_none());
    }

    #[test]
    fn test_auto_compact_on_event() {
        let budget = TokenBudget {
            max_tokens: 200,
            target_tokens: 50,
            min_retained_entries: 1,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 500);
        let bridge = DebateMemoryBridge::with_compactor(compactor, true);
        let mut store = SwarmMemoryStore::new();

        // Fill store with enough data
        for i in 0..10 {
            store.append(MemoryEntry::new(
                MemoryEntryKind::AgentTurn,
                &format!("Entry {}", i),
                "coder",
                20,
            ));
        }

        let summarizer = MockSummarizer::new();
        let result = bridge.maybe_compact(&mut store, &summarizer, CompactionEvent::SessionEnding);
        assert!(result.is_some());
        // The compaction should have run (store was over target)
        assert!(result.unwrap().is_ok());
    }

    #[test]
    fn test_check_and_compact_under_budget() {
        let bridge = DebateMemoryBridge::new("test");
        let mut store = SwarmMemoryStore::new();
        store.append(MemoryEntry::new(
            MemoryEntryKind::AgentTurn,
            "small",
            "coder",
            10,
        ));
        let summarizer = MockSummarizer::new();

        let result = bridge.check_and_compact(&mut store, &summarizer);
        assert!(result.is_none()); // under budget, no compaction
    }
}
