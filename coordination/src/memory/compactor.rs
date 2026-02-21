//! Memory compactor — orchestrates history rewrite with summary sentinels.
//!
//! Ties together the budget trigger, summarizer, and memory store
//! to perform compaction while preserving recent context.
//!
//! Also provides event-driven compaction support for integration
//! with the orchestration event bus.

use serde::{Deserialize, Serialize};

use super::budget::{BudgetDecision, CompactionTrigger, TokenBudget};
use super::errors::{CompactionError, CompactionErrorKind};
use super::store::{MemoryEntry, SwarmMemory};
use super::summarizer::{build_summary_request, Summarizer};

/// Result of a compaction operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    /// Number of entries compacted.
    pub entries_compacted: usize,
    /// Tokens freed.
    pub tokens_freed: u64,
    /// Summary tokens added.
    pub summary_tokens_added: u32,
    /// Compression ratio achieved.
    pub compression_ratio: f64,
    /// Sequence range that was compacted.
    pub compacted_range: (u64, u64),
    /// Whether compaction was triggered by budget or event.
    pub trigger: CompactionTriggerKind,
}

/// What triggered the compaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionTriggerKind {
    /// Budget threshold exceeded.
    BudgetThreshold,
    /// Explicit event trigger (e.g., session end, phase change).
    Event,
    /// Manual trigger.
    Manual,
}

impl std::fmt::Display for CompactionTriggerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetThreshold => write!(f, "budget_threshold"),
            Self::Event => write!(f, "event"),
            Self::Manual => write!(f, "manual"),
        }
    }
}

/// Orchestration event that can trigger compaction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompactionEvent {
    /// A debate round completed.
    RoundCompleted { round: u32 },
    /// Session phase changed.
    PhaseChanged { from: String, to: String },
    /// Agent tier escalated.
    TierEscalated { from: String, to: String },
    /// Session is ending.
    SessionEnding,
    /// Memory threshold crossed (from budget check).
    BudgetExceeded { current_tokens: u64, budget: u64 },
}

impl std::fmt::Display for CompactionEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RoundCompleted { round } => write!(f, "round_completed({})", round),
            Self::PhaseChanged { from, to } => write!(f, "phase_changed({} → {})", from, to),
            Self::TierEscalated { from, to } => write!(f, "tier_escalated({} → {})", from, to),
            Self::SessionEnding => write!(f, "session_ending"),
            Self::BudgetExceeded {
                current_tokens,
                budget,
            } => write!(f, "budget_exceeded({}/{})", current_tokens, budget),
        }
    }
}

/// Policy for event-driven compaction decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionPolicy {
    /// Compact on every Nth round.
    pub compact_every_n_rounds: u32,
    /// Always compact on phase change.
    pub compact_on_phase_change: bool,
    /// Always compact on escalation.
    pub compact_on_escalation: bool,
    /// Always compact on session end.
    pub compact_on_session_end: bool,
}

impl Default for CompactionPolicy {
    fn default() -> Self {
        Self {
            compact_every_n_rounds: 3,
            compact_on_phase_change: true,
            compact_on_escalation: true,
            compact_on_session_end: true,
        }
    }
}

impl CompactionPolicy {
    /// Evaluate whether an event should trigger compaction.
    pub fn should_compact(&self, event: &CompactionEvent) -> bool {
        match event {
            CompactionEvent::RoundCompleted { round } => {
                self.compact_every_n_rounds > 0 && round % self.compact_every_n_rounds == 0
            }
            CompactionEvent::PhaseChanged { .. } => self.compact_on_phase_change,
            CompactionEvent::TierEscalated { .. } => self.compact_on_escalation,
            CompactionEvent::SessionEnding => self.compact_on_session_end,
            CompactionEvent::BudgetExceeded { .. } => true, // always compact on budget
        }
    }
}

/// The memory compactor — orchestrates compaction operations.
pub struct MemoryCompactor {
    trigger: CompactionTrigger,
    policy: CompactionPolicy,
    session_context: String,
    /// Max tokens for generated summaries.
    max_summary_tokens: u32,
}

impl MemoryCompactor {
    /// Create a new compactor with default settings.
    pub fn new(session_context: &str) -> Self {
        Self {
            trigger: CompactionTrigger::new(TokenBudget::default()),
            policy: CompactionPolicy::default(),
            session_context: session_context.to_string(),
            max_summary_tokens: 2000,
        }
    }

    /// Create with custom budget and policy.
    pub fn with_config(
        budget: TokenBudget,
        policy: CompactionPolicy,
        session_context: &str,
        max_summary_tokens: u32,
    ) -> Self {
        Self {
            trigger: CompactionTrigger::new(budget),
            policy,
            session_context: session_context.to_string(),
            max_summary_tokens,
        }
    }

    /// Check whether compaction should run based on current token count.
    pub fn check_budget(&self, current_tokens: u64) -> BudgetDecision {
        self.trigger.evaluate(current_tokens)
    }

    /// Check whether an event should trigger compaction.
    pub fn check_event(&self, event: &CompactionEvent) -> bool {
        self.policy.should_compact(event)
    }

    /// Execute compaction on the memory store.
    ///
    /// This is the main entry point. It:
    /// 1. Determines which entries to compact (oldest, respecting min retained)
    /// 2. Summarizes them via the provided summarizer
    /// 3. Validates the summary against the contract
    /// 4. Inserts a summary sentinel and marks old entries as compacted
    pub fn compact(
        &self,
        store: &mut dyn SwarmMemory,
        summarizer: &dyn Summarizer,
        trigger_kind: CompactionTriggerKind,
    ) -> Result<CompactionResult, CompactionError> {
        let active = store.active_entries();

        if active.is_empty() {
            return Err(CompactionError::new(
                CompactionErrorKind::EmptyInput,
                "no active entries to compact",
            ));
        }

        let current_tokens = store.active_token_count();
        let entry_tokens: Vec<u64> = active.iter().map(|e| e.estimated_tokens as u64).collect();
        let entries_to_compact = self
            .trigger
            .entries_to_compact(&entry_tokens, current_tokens);

        if entries_to_compact == 0 {
            return Err(CompactionError::new(
                CompactionErrorKind::EmptyInput,
                "no entries eligible for compaction (under target or min retained)",
            ));
        }

        // Collect the entries to summarize
        let to_summarize: Vec<&MemoryEntry> = active.into_iter().take(entries_to_compact).collect();

        let last_seq = to_summarize.last().map(|e| e.seq).unwrap_or(0);

        let first_seq = to_summarize.first().map(|e| e.seq).unwrap_or(0);

        let tokens_in_range: u64 = to_summarize.iter().map(|e| e.estimated_tokens as u64).sum();

        // Build and execute summary request
        let request = build_summary_request(
            &to_summarize,
            self.max_summary_tokens,
            &self.session_context,
        );

        let response = summarizer.summarize(&request).map_err(|e| {
            CompactionError::new(CompactionErrorKind::SummarizationFailed, &e.to_string())
                .with_range(first_seq, last_seq)
        })?;

        // Validate the response
        response
            .validate(&request)
            .map_err(|e| e.with_range(first_seq, last_seq))?;

        // Insert summary sentinel and compact old entries
        let summary_entry = MemoryEntry::summary(&response.summary, response.summary_tokens);
        store.insert_summary(summary_entry, last_seq);

        Ok(CompactionResult {
            entries_compacted: entries_to_compact,
            tokens_freed: tokens_in_range,
            summary_tokens_added: response.summary_tokens,
            compression_ratio: response.compression_ratio,
            compacted_range: (first_seq, last_seq),
            trigger: trigger_kind,
        })
    }

    /// Get the compaction policy.
    pub fn policy(&self) -> &CompactionPolicy {
        &self.policy
    }
}

#[cfg(test)]
mod tests {
    use super::super::budget::TokenBudget;
    use super::super::store::{MemoryEntryKind, SwarmMemoryStore};
    use super::super::summarizer::MockSummarizer;
    use super::*;

    fn make_store_with_entries(count: usize, tokens_each: u32) -> SwarmMemoryStore {
        let mut store = SwarmMemoryStore::new();
        for i in 0..count {
            store.append(MemoryEntry::new(
                MemoryEntryKind::AgentTurn,
                &format!("Entry {}", i),
                "coder",
                tokens_each,
            ));
        }
        store
    }

    #[test]
    fn test_compact_basic() {
        let budget = TokenBudget {
            max_tokens: 100,
            target_tokens: 30,
            min_retained_entries: 2,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test session", 500);
        let mut store = make_store_with_entries(5, 20); // 100 tokens total
        let summarizer = MockSummarizer::new();

        let result = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap();

        assert!(result.entries_compacted > 0);
        assert!(result.tokens_freed > 0);
        assert!(result.summary_tokens_added > 0);

        // Should have summary sentinel in active entries
        let active = store.active_entries();
        assert!(active.iter().any(|e| e.kind == MemoryEntryKind::Summary));
    }

    #[test]
    fn test_compact_empty_store() {
        let compactor = MemoryCompactor::new("test");
        let mut store = SwarmMemoryStore::new();
        let summarizer = MockSummarizer::new();

        let err = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::EmptyInput);
    }

    #[test]
    fn test_compact_under_target() {
        let budget = TokenBudget {
            max_tokens: 1000,
            target_tokens: 500,
            min_retained_entries: 2,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 500);
        let mut store = make_store_with_entries(3, 10); // only 30 tokens
        let summarizer = MockSummarizer::new();

        let err = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::EmptyInput);
    }

    #[test]
    fn test_compact_summarizer_failure() {
        let budget = TokenBudget {
            max_tokens: 100,
            target_tokens: 30,
            min_retained_entries: 1,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 500);
        let mut store = make_store_with_entries(5, 20);
        let summarizer = MockSummarizer::failing();

        let err = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::SummarizationFailed);
    }

    #[test]
    fn test_compact_oversize_summary() {
        let budget = TokenBudget {
            max_tokens: 100,
            target_tokens: 30,
            min_retained_entries: 1,
            system_reserve: 0,
        };
        let compactor = MemoryCompactor::with_config(
            budget,
            CompactionPolicy::default(),
            "test",
            50, // very small budget
        );
        let mut store = make_store_with_entries(5, 20);
        let summarizer = MockSummarizer::oversize();

        let err = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::SummaryTooLarge);
    }

    #[test]
    fn test_compact_preserves_recent_entries() {
        let budget = TokenBudget {
            max_tokens: 100,
            target_tokens: 30,
            min_retained_entries: 3,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 500);
        let mut store = make_store_with_entries(5, 20);
        let summarizer = MockSummarizer::new();

        compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap();

        // Recent entries (last 3) should still be active
        let active = store.active_entries();
        let non_summary: Vec<_> = active
            .iter()
            .filter(|e| e.kind != MemoryEntryKind::Summary)
            .collect();
        assert!(non_summary.len() >= 3);
    }

    #[test]
    fn test_summary_sentinel_ordering() {
        let budget = TokenBudget {
            max_tokens: 100,
            target_tokens: 30,
            min_retained_entries: 2,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 500);
        let mut store = make_store_with_entries(5, 20);
        let summarizer = MockSummarizer::new();

        compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap();

        // Summary sentinel should have a higher sequence than compacted entries
        let all = store.all_entries();
        let summary = all
            .iter()
            .find(|e| e.kind == MemoryEntryKind::Summary)
            .unwrap();
        let compacted: Vec<_> = all.iter().filter(|e| e.compacted).collect();
        for entry in compacted {
            assert!(
                summary.seq > entry.seq,
                "Summary seq {} should be > compacted entry seq {}",
                summary.seq,
                entry.seq
            );
        }
    }

    // --- Event-driven compaction tests (hx0.2.6) ---

    #[test]
    fn test_policy_round_trigger() {
        let policy = CompactionPolicy {
            compact_every_n_rounds: 3,
            ..Default::default()
        };
        assert!(!policy.should_compact(&CompactionEvent::RoundCompleted { round: 1 }));
        assert!(!policy.should_compact(&CompactionEvent::RoundCompleted { round: 2 }));
        assert!(policy.should_compact(&CompactionEvent::RoundCompleted { round: 3 }));
        assert!(!policy.should_compact(&CompactionEvent::RoundCompleted { round: 4 }));
        assert!(policy.should_compact(&CompactionEvent::RoundCompleted { round: 6 }));
    }

    #[test]
    fn test_policy_disabled_rounds() {
        let policy = CompactionPolicy {
            compact_every_n_rounds: 0,
            ..Default::default()
        };
        assert!(!policy.should_compact(&CompactionEvent::RoundCompleted { round: 3 }));
    }

    #[test]
    fn test_policy_phase_change() {
        let policy = CompactionPolicy::default();
        assert!(policy.should_compact(&CompactionEvent::PhaseChanged {
            from: "coder_turn".to_string(),
            to: "reviewer_turn".to_string(),
        }));
    }

    #[test]
    fn test_policy_escalation() {
        let policy = CompactionPolicy::default();
        assert!(policy.should_compact(&CompactionEvent::TierEscalated {
            from: "strand-14b".to_string(),
            to: "or1-behemoth".to_string(),
        }));
    }

    #[test]
    fn test_policy_session_end() {
        let policy = CompactionPolicy::default();
        assert!(policy.should_compact(&CompactionEvent::SessionEnding));
    }

    #[test]
    fn test_policy_budget_always_triggers() {
        let policy = CompactionPolicy {
            compact_every_n_rounds: 0,
            compact_on_phase_change: false,
            compact_on_escalation: false,
            compact_on_session_end: false,
        };
        // Budget events always trigger regardless of policy
        assert!(policy.should_compact(&CompactionEvent::BudgetExceeded {
            current_tokens: 100000,
            budget: 80000,
        }));
    }

    // --- Compaction failure mode tests (hx0.2.8) ---

    #[test]
    fn test_failure_empty_history() {
        let compactor = MemoryCompactor::new("test");
        let mut store = SwarmMemoryStore::new();
        let summarizer = MockSummarizer::new();

        let err = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::EmptyInput);
    }

    #[test]
    fn test_failure_summarizer_error() {
        let budget = TokenBudget {
            max_tokens: 100,
            target_tokens: 30,
            min_retained_entries: 1,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 500);
        let mut store = make_store_with_entries(5, 20);
        let summarizer = MockSummarizer::failing();

        let err = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::SummarizationFailed);
        assert!(err.seq_range.is_some()); // should have range context
    }

    #[test]
    fn test_failure_oversize_summary() {
        let budget = TokenBudget {
            max_tokens: 100,
            target_tokens: 30,
            min_retained_entries: 1,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 50);
        let mut store = make_store_with_entries(5, 20);
        let summarizer = MockSummarizer::oversize();

        let err = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::SummaryTooLarge);
    }

    #[test]
    fn test_failure_all_entries_protected() {
        // When min_retained >= entry count, nothing can be compacted
        let budget = TokenBudget {
            max_tokens: 100,
            target_tokens: 30,
            min_retained_entries: 10, // more than entries
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 500);
        let mut store = make_store_with_entries(5, 20);
        let summarizer = MockSummarizer::new();

        let err = compactor
            .compact(&mut store, &summarizer, CompactionTriggerKind::Manual)
            .unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::EmptyInput);
    }

    #[test]
    fn test_double_compaction() {
        let budget = TokenBudget {
            max_tokens: 200,
            target_tokens: 50,
            min_retained_entries: 1,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "test", 500);
        let mut store = make_store_with_entries(10, 20); // 200 tokens
        let summarizer = MockSummarizer::new();

        // First compaction
        let result1 = compactor
            .compact(
                &mut store,
                &summarizer,
                CompactionTriggerKind::BudgetThreshold,
            )
            .unwrap();
        assert!(result1.entries_compacted > 0);

        // Add more entries
        for i in 0..5 {
            store.append(MemoryEntry::new(
                MemoryEntryKind::AgentTurn,
                &format!("New entry {}", i),
                "coder",
                20,
            ));
        }

        // Second compaction — should work on new entries
        let result2 = compactor
            .compact(
                &mut store,
                &summarizer,
                CompactionTriggerKind::BudgetThreshold,
            )
            .unwrap();
        assert!(result2.entries_compacted > 0);

        // Should have 2 summaries now
        let all = store.all_entries();
        let summary_count = all
            .iter()
            .filter(|e| e.kind == MemoryEntryKind::Summary)
            .count();
        assert_eq!(summary_count, 2);
    }

    // --- Serde tests ---

    #[test]
    fn test_compaction_result_serde() {
        let result = CompactionResult {
            entries_compacted: 5,
            tokens_freed: 500,
            summary_tokens_added: 50,
            compression_ratio: 10.0,
            compacted_range: (1, 5),
            trigger: CompactionTriggerKind::BudgetThreshold,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: CompactionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.entries_compacted, 5);
        assert_eq!(parsed.trigger, CompactionTriggerKind::BudgetThreshold);
    }

    #[test]
    fn test_compaction_event_display() {
        assert_eq!(
            CompactionEvent::RoundCompleted { round: 3 }.to_string(),
            "round_completed(3)"
        );
        assert!(CompactionEvent::SessionEnding
            .to_string()
            .contains("session_ending"));
    }

    #[test]
    fn test_policy_serde() {
        let policy = CompactionPolicy::default();
        let json = serde_json::to_string(&policy).unwrap();
        let parsed: CompactionPolicy = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.compact_every_n_rounds, 3);
        assert!(parsed.compact_on_session_end);
    }
}
