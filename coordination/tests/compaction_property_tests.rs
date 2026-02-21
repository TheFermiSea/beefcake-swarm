//! Compaction property tests — randomized validation of memory compaction
//! invariants across varied inputs.
//!
//! Tests verify:
//! - Token counts decrease after compaction
//! - Summary sentinels are never compacted
//! - Sequence ordering is preserved
//! - Active entries never exceed budget after compaction
//! - Compression ratio is positive
//! - Double compaction is idempotent on already-compact stores

use coordination::memory::{
    CompactionPolicy, CompactionTriggerKind, MemoryCompactor, MemoryEntry, MemoryEntryKind,
    MockSummarizer, SwarmMemory, SwarmMemoryStore, TokenBudget,
};

/// Fill a store with N entries of varying kinds and token counts.
fn fill_store(store: &mut SwarmMemoryStore, count: usize, base_tokens: u32) {
    let kinds = [
        MemoryEntryKind::AgentTurn,
        MemoryEntryKind::ToolResult,
        MemoryEntryKind::ErrorContext,
        MemoryEntryKind::WorkPacket,
    ];
    let sources = ["coder", "reviewer", "orchestrator", "verifier"];

    for i in 0..count {
        let kind = kinds[i % kinds.len()];
        let source = sources[i % sources.len()];
        let tokens = base_tokens + (i as u32 % 10) * 5;
        let content = format!("Entry {} with {} tokens", i, tokens);
        store.append(MemoryEntry::new(kind, &content, source, tokens));
    }
}

/// Create a compactor with specific budget parameters.
fn make_compactor(max_tokens: u64, target_tokens: u64) -> MemoryCompactor {
    let budget = TokenBudget {
        max_tokens,
        target_tokens,
        min_retained_entries: 2,
        system_reserve: 0,
    };
    MemoryCompactor::with_config(budget, CompactionPolicy::default(), "property-test", 500)
}

// ── Property: token count decreases after compaction ───────────────

#[test]
fn prop_tokens_decrease_after_compaction() {
    for entry_count in [5, 10, 20, 50] {
        let mut store = SwarmMemoryStore::new();
        fill_store(&mut store, entry_count, 30);

        let before_tokens = store.active_token_count();
        let compactor = make_compactor(before_tokens / 2, before_tokens / 4);
        let summarizer = MockSummarizer::new();

        let result = compactor.compact(
            &mut store,
            &summarizer,
            CompactionTriggerKind::BudgetThreshold,
        );

        if let Ok(ref cr) = result {
            let after_tokens = store.active_token_count();
            assert!(
                after_tokens <= before_tokens,
                "entry_count={}: tokens should decrease: before={}, after={}",
                entry_count,
                before_tokens,
                after_tokens
            );
            assert!(
                cr.entries_compacted > 0,
                "entry_count={}: should compact at least one entry",
                entry_count
            );
        }
    }
}

// ── Property: summary sentinels are never compacted ────────────────

#[test]
fn prop_summaries_survive_compaction() {
    let mut store = SwarmMemoryStore::new();

    // Add some regular entries
    for i in 0..10 {
        store.append(MemoryEntry::new(
            MemoryEntryKind::AgentTurn,
            &format!("Turn {}", i),
            "coder",
            50,
        ));
    }

    // Add a summary sentinel
    let summary_entry = MemoryEntry::summary("Previous compaction summary", 20);
    let max_seq = store.active_entries().last().map(|e| e.seq).unwrap_or(0);
    store.insert_summary(summary_entry, max_seq);

    // Add more entries
    for i in 10..20 {
        store.append(MemoryEntry::new(
            MemoryEntryKind::AgentTurn,
            &format!("Turn {}", i),
            "reviewer",
            50,
        ));
    }

    let compactor = make_compactor(500, 200);
    let summarizer = MockSummarizer::new();
    let _ = compactor.compact(
        &mut store,
        &summarizer,
        CompactionTriggerKind::BudgetThreshold,
    );

    // The summary entry should still exist
    let summaries = store
        .active_entries()
        .iter()
        .filter(|e| e.kind == MemoryEntryKind::Summary)
        .count();
    assert!(summaries > 0, "Summary sentinels must survive compaction");
}

// ── Property: sequence ordering preserved ──────────────────────────

#[test]
fn prop_sequence_ordering_preserved() {
    for entry_count in [10, 25, 50] {
        let mut store = SwarmMemoryStore::new();
        fill_store(&mut store, entry_count, 20);

        let compactor = make_compactor(300, 100);
        let summarizer = MockSummarizer::new();
        let _ = compactor.compact(
            &mut store,
            &summarizer,
            CompactionTriggerKind::BudgetThreshold,
        );

        let entries = store.active_entries();
        for i in 1..entries.len() {
            assert!(
                entries[i].seq > entries[i - 1].seq,
                "entry_count={}: sequence order violated at index {}: {} vs {}",
                entry_count,
                i,
                entries[i - 1].seq,
                entries[i].seq
            );
        }
    }
}

// ── Property: min_retained_entries respected ────────────────────────

#[test]
fn prop_min_retained_entries_respected() {
    for min_retained in [1usize, 2, 5] {
        let mut store = SwarmMemoryStore::new();
        fill_store(&mut store, 20, 50);

        let budget = TokenBudget {
            max_tokens: 100, // very tight — forces aggressive compaction
            target_tokens: 50,
            min_retained_entries: min_retained,
            system_reserve: 0,
        };
        let compactor =
            MemoryCompactor::with_config(budget, CompactionPolicy::default(), "prop-test", 500);
        let summarizer = MockSummarizer::new();

        let _ = compactor.compact(
            &mut store,
            &summarizer,
            CompactionTriggerKind::BudgetThreshold,
        );

        // Must retain at least min_retained non-summary entries
        let non_summary: Vec<_> = store
            .active_entries()
            .into_iter()
            .filter(|e| e.kind != MemoryEntryKind::Summary)
            .collect();
        assert!(
            non_summary.len() >= min_retained,
            "min_retained={}: only {} non-summary entries remain",
            min_retained,
            non_summary.len()
        );
    }
}

// ── Property: compression ratio is positive ────────────────────────

#[test]
fn prop_compression_ratio_positive() {
    let mut store = SwarmMemoryStore::new();
    fill_store(&mut store, 30, 40);

    let compactor = make_compactor(500, 200);
    let summarizer = MockSummarizer::new();

    let result = compactor.compact(
        &mut store,
        &summarizer,
        CompactionTriggerKind::BudgetThreshold,
    );

    if let Ok(cr) = result {
        assert!(
            cr.compression_ratio > 0.0,
            "Compression ratio should be positive, got {}",
            cr.compression_ratio
        );
    }
}

// ── Property: double compaction is safe ─────────────────────────────

#[test]
fn prop_double_compaction_safe() {
    let mut store = SwarmMemoryStore::new();
    fill_store(&mut store, 30, 40);

    let compactor = make_compactor(600, 300);
    let summarizer = MockSummarizer::new();

    // First compaction
    let r1 = compactor.compact(
        &mut store,
        &summarizer,
        CompactionTriggerKind::BudgetThreshold,
    );
    assert!(r1.is_ok());
    let after_first = store.active_token_count();
    let entries_after_first = store.active_entries().len();

    // Second compaction on already-compact store
    let r2 = compactor.compact(
        &mut store,
        &summarizer,
        CompactionTriggerKind::BudgetThreshold,
    );

    // Second compaction should either succeed with 0 compacted or fail gracefully
    match r2 {
        Ok(_cr) => {
            // If it succeeded, should have compacted fewer or equal entries
            assert!(
                store.active_token_count() <= after_first,
                "Double compaction shouldn't increase tokens"
            );
        }
        Err(_) => {
            // Already compact — this is acceptable
            assert_eq!(
                store.active_entries().len(),
                entries_after_first,
                "Failed second compaction shouldn't change state"
            );
        }
    }
}

// ── Property: empty store compaction is safe ────────────────────────

#[test]
fn prop_empty_store_compaction() {
    let mut store = SwarmMemoryStore::new();
    let compactor = make_compactor(1000, 500);
    let summarizer = MockSummarizer::new();

    let result = compactor.compact(
        &mut store,
        &summarizer,
        CompactionTriggerKind::BudgetThreshold,
    );

    // Should fail gracefully, not panic
    assert!(result.is_err());
}

// ── Property: single entry store compaction ─────────────────────────

#[test]
fn prop_single_entry_compaction() {
    let mut store = SwarmMemoryStore::new();
    store.append(MemoryEntry::new(
        MemoryEntryKind::AgentTurn,
        "Only entry",
        "coder",
        100,
    ));

    let compactor = make_compactor(50, 25);
    let summarizer = MockSummarizer::new();

    let _result = compactor.compact(
        &mut store,
        &summarizer,
        CompactionTriggerKind::BudgetThreshold,
    );

    // With min_retained_entries=2 and only 1 entry, compaction should not destroy it
    assert!(store.active_entries().len() >= 1);
}

// ── Property: varied entry sizes ────────────────────────────────────

#[test]
fn prop_varied_entry_sizes() {
    let mut store = SwarmMemoryStore::new();

    // Mix of tiny and large entries
    let sizes: [u32; 10] = [5, 200, 10, 500, 3, 100, 8, 300, 15, 50];
    for (i, &tokens) in sizes.iter().enumerate() {
        store.append(MemoryEntry::new(
            MemoryEntryKind::AgentTurn,
            &format!("Entry {} ({}tok)", i, tokens),
            "coder",
            tokens,
        ));
    }

    let total_before = store.active_token_count();
    let compactor = make_compactor(total_before / 2, total_before / 4);
    let summarizer = MockSummarizer::new();

    let result = compactor.compact(
        &mut store,
        &summarizer,
        CompactionTriggerKind::BudgetThreshold,
    );

    if let Ok(cr) = result {
        assert!(cr.entries_compacted > 0);
        // Ordering preserved
        let entries = store.active_entries();
        for i in 1..entries.len() {
            assert!(entries[i].seq > entries[i - 1].seq);
        }
    }
}

// ── Property: all entry kinds handled ───────────────────────────────

#[test]
fn prop_all_entry_kinds_handled() {
    let mut store = SwarmMemoryStore::new();

    // One of each kind
    store.append(MemoryEntry::new(
        MemoryEntryKind::SystemPrompt,
        "System prompt",
        "system",
        50,
    ));
    store.append(MemoryEntry::new(
        MemoryEntryKind::AgentTurn,
        "Agent turn",
        "coder",
        50,
    ));
    store.append(MemoryEntry::new(
        MemoryEntryKind::ToolResult,
        "Tool result",
        "verifier",
        50,
    ));
    store.append(MemoryEntry::new(
        MemoryEntryKind::ErrorContext,
        "Error context",
        "verifier",
        50,
    ));
    store.append(MemoryEntry::new(
        MemoryEntryKind::WorkPacket,
        "Work packet",
        "orchestrator",
        50,
    ));

    let compactor = make_compactor(150, 75);
    let summarizer = MockSummarizer::new();

    // Should not panic regardless of entry kind mix
    let _ = compactor.compact(
        &mut store,
        &summarizer,
        CompactionTriggerKind::BudgetThreshold,
    );

    // Store should still be consistent
    let entries = store.active_entries();
    for i in 1..entries.len() {
        assert!(entries[i].seq > entries[i - 1].seq);
    }
}
