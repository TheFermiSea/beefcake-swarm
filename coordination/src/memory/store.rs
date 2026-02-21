//! SwarmMemory abstraction — conversation context management.
//!
//! Defines the memory interface and an in-memory implementation
//! compatible with existing message/session structures.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Kind of memory entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryEntryKind {
    /// User/system prompt message.
    SystemPrompt,
    /// Agent (coder/reviewer) turn output.
    AgentTurn,
    /// Tool call and result.
    ToolResult,
    /// Compaction summary (sentinel).
    Summary,
    /// Error context from a failed iteration.
    ErrorContext,
    /// Work packet or context handoff.
    WorkPacket,
}

impl std::fmt::Display for MemoryEntryKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SystemPrompt => write!(f, "system_prompt"),
            Self::AgentTurn => write!(f, "agent_turn"),
            Self::ToolResult => write!(f, "tool_result"),
            Self::Summary => write!(f, "summary"),
            Self::ErrorContext => write!(f, "error_context"),
            Self::WorkPacket => write!(f, "work_packet"),
        }
    }
}

/// A single entry in the memory store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Monotonic sequence number.
    pub seq: u64,
    /// Kind of entry.
    pub kind: MemoryEntryKind,
    /// The content text.
    pub content: String,
    /// Estimated token count.
    pub estimated_tokens: u32,
    /// When this entry was created.
    pub created_at: DateTime<Utc>,
    /// Whether this entry has been compacted (replaced by summary).
    pub compacted: bool,
    /// Source identifier (agent name, tool name, etc.).
    pub source: String,
}

impl MemoryEntry {
    /// Create a new memory entry.
    pub fn new(kind: MemoryEntryKind, content: &str, source: &str, estimated_tokens: u32) -> Self {
        Self {
            seq: 0, // assigned by store
            kind,
            content: content.to_string(),
            estimated_tokens,
            created_at: Utc::now(),
            compacted: false,
            source: source.to_string(),
        }
    }

    /// Create a summary sentinel entry.
    pub fn summary(content: &str, tokens_compacted: u32) -> Self {
        Self {
            seq: 0,
            kind: MemoryEntryKind::Summary,
            content: content.to_string(),
            estimated_tokens: tokens_compacted,
            created_at: Utc::now(),
            compacted: false,
            source: "compactor".to_string(),
        }
    }
}

/// Snapshot of the memory store for inspection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemorySnapshot {
    /// Total entries (including compacted).
    pub total_entries: usize,
    /// Active (non-compacted) entries.
    pub active_entries: usize,
    /// Total estimated tokens in active entries.
    pub active_tokens: u64,
    /// Number of compaction summaries.
    pub summary_count: usize,
    /// Sequence range (min, max).
    pub seq_range: (u64, u64),
}

/// Trait for swarm memory stores.
pub trait SwarmMemory {
    /// Append a new entry, returning its assigned sequence number.
    fn append(&mut self, entry: MemoryEntry) -> u64;

    /// Get all active (non-compacted) entries in order.
    fn active_entries(&self) -> Vec<&MemoryEntry>;

    /// Get the total estimated tokens of active entries.
    fn active_token_count(&self) -> u64;

    /// Mark entries up to `seq` (inclusive) as compacted.
    fn compact_up_to(&mut self, seq: u64);

    /// Insert a summary sentinel and compact preceding entries.
    fn insert_summary(&mut self, summary: MemoryEntry, compact_up_to_seq: u64);

    /// Get a snapshot of the memory state.
    fn snapshot(&self) -> MemorySnapshot;

    /// Get an entry by sequence number.
    fn get(&self, seq: u64) -> Option<&MemoryEntry>;

    /// Get all entries (including compacted) for persistence.
    fn all_entries(&self) -> Vec<&MemoryEntry>;

    /// Clear all entries.
    fn clear(&mut self);
}

/// In-memory implementation of SwarmMemory.
pub struct SwarmMemoryStore {
    entries: Vec<MemoryEntry>,
    next_seq: u64,
}

impl SwarmMemoryStore {
    /// Create a new empty memory store.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            next_seq: 1,
        }
    }

    /// Create from a list of existing entries (for restore).
    pub fn from_entries(entries: Vec<MemoryEntry>) -> Self {
        let next_seq = entries.iter().map(|e| e.seq).max().unwrap_or(0) + 1;
        Self { entries, next_seq }
    }
}

impl Default for SwarmMemoryStore {
    fn default() -> Self {
        Self::new()
    }
}

impl SwarmMemory for SwarmMemoryStore {
    fn append(&mut self, mut entry: MemoryEntry) -> u64 {
        entry.seq = self.next_seq;
        self.next_seq += 1;
        let seq = entry.seq;
        self.entries.push(entry);
        seq
    }

    fn active_entries(&self) -> Vec<&MemoryEntry> {
        self.entries.iter().filter(|e| !e.compacted).collect()
    }

    fn active_token_count(&self) -> u64 {
        self.entries
            .iter()
            .filter(|e| !e.compacted)
            .map(|e| e.estimated_tokens as u64)
            .sum()
    }

    fn compact_up_to(&mut self, seq: u64) {
        for entry in &mut self.entries {
            if entry.seq <= seq && entry.kind != MemoryEntryKind::Summary {
                entry.compacted = true;
            }
        }
    }

    fn insert_summary(&mut self, summary: MemoryEntry, compact_up_to_seq: u64) {
        self.compact_up_to(compact_up_to_seq);
        self.append(summary);
    }

    fn snapshot(&self) -> MemorySnapshot {
        let active: Vec<_> = self.entries.iter().filter(|e| !e.compacted).collect();
        let summary_count = self
            .entries
            .iter()
            .filter(|e| e.kind == MemoryEntryKind::Summary)
            .count();
        let min_seq = self.entries.first().map(|e| e.seq).unwrap_or(0);
        let max_seq = self.entries.last().map(|e| e.seq).unwrap_or(0);

        MemorySnapshot {
            total_entries: self.entries.len(),
            active_entries: active.len(),
            active_tokens: active.iter().map(|e| e.estimated_tokens as u64).sum(),
            summary_count,
            seq_range: (min_seq, max_seq),
        }
    }

    fn get(&self, seq: u64) -> Option<&MemoryEntry> {
        self.entries.iter().find(|e| e.seq == seq)
    }

    fn all_entries(&self) -> Vec<&MemoryEntry> {
        self.entries.iter().collect()
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.next_seq = 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent_entry(content: &str, tokens: u32) -> MemoryEntry {
        MemoryEntry::new(MemoryEntryKind::AgentTurn, content, "coder", tokens)
    }

    #[test]
    fn test_append_and_get() {
        let mut store = SwarmMemoryStore::new();
        let seq = store.append(agent_entry("hello", 10));
        assert_eq!(seq, 1);

        let entry = store.get(1).unwrap();
        assert_eq!(entry.content, "hello");
        assert_eq!(entry.estimated_tokens, 10);
    }

    #[test]
    fn test_sequence_increments() {
        let mut store = SwarmMemoryStore::new();
        let s1 = store.append(agent_entry("a", 5));
        let s2 = store.append(agent_entry("b", 5));
        let s3 = store.append(agent_entry("c", 5));
        assert_eq!(s1, 1);
        assert_eq!(s2, 2);
        assert_eq!(s3, 3);
    }

    #[test]
    fn test_active_entries() {
        let mut store = SwarmMemoryStore::new();
        store.append(agent_entry("a", 10));
        store.append(agent_entry("b", 20));
        store.append(agent_entry("c", 30));

        assert_eq!(store.active_entries().len(), 3);
        assert_eq!(store.active_token_count(), 60);
    }

    #[test]
    fn test_compact_up_to() {
        let mut store = SwarmMemoryStore::new();
        store.append(agent_entry("a", 10));
        store.append(agent_entry("b", 20));
        store.append(agent_entry("c", 30));

        store.compact_up_to(2);

        assert_eq!(store.active_entries().len(), 1);
        assert_eq!(store.active_token_count(), 30);
        assert!(store.get(1).unwrap().compacted);
        assert!(store.get(2).unwrap().compacted);
        assert!(!store.get(3).unwrap().compacted);
    }

    #[test]
    fn test_insert_summary() {
        let mut store = SwarmMemoryStore::new();
        store.append(agent_entry("a", 100));
        store.append(agent_entry("b", 200));
        store.append(agent_entry("c", 300));

        let summary = MemoryEntry::summary("Summary of a+b+c", 50);
        store.insert_summary(summary, 3);

        let snapshot = store.snapshot();
        assert_eq!(snapshot.total_entries, 4);
        assert_eq!(snapshot.active_entries, 1); // just the summary
        assert_eq!(snapshot.summary_count, 1);
    }

    #[test]
    fn test_summary_not_compacted_by_compact_up_to() {
        let mut store = SwarmMemoryStore::new();
        store.append(agent_entry("a", 100));
        let summary = MemoryEntry::summary("Summary", 20);
        store.insert_summary(summary, 1);
        // Now compact everything — summaries should survive
        store.compact_up_to(10);
        let active = store.active_entries();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].kind, MemoryEntryKind::Summary);
    }

    #[test]
    fn test_snapshot() {
        let mut store = SwarmMemoryStore::new();
        store.append(agent_entry("a", 10));
        store.append(agent_entry("b", 20));

        let snap = store.snapshot();
        assert_eq!(snap.total_entries, 2);
        assert_eq!(snap.active_entries, 2);
        assert_eq!(snap.active_tokens, 30);
        assert_eq!(snap.seq_range, (1, 2));
    }

    #[test]
    fn test_clear() {
        let mut store = SwarmMemoryStore::new();
        store.append(agent_entry("a", 10));
        store.clear();
        assert_eq!(store.active_entries().len(), 0);
        assert_eq!(store.snapshot().total_entries, 0);

        // Sequence resets
        let seq = store.append(agent_entry("b", 10));
        assert_eq!(seq, 1);
    }

    #[test]
    fn test_all_entries_includes_compacted() {
        let mut store = SwarmMemoryStore::new();
        store.append(agent_entry("a", 10));
        store.append(agent_entry("b", 20));
        store.compact_up_to(1);

        assert_eq!(store.all_entries().len(), 2);
        assert_eq!(store.active_entries().len(), 1);
    }

    #[test]
    fn test_from_entries() {
        let mut entries = vec![];
        let mut e1 = agent_entry("a", 10);
        e1.seq = 5;
        let mut e2 = agent_entry("b", 20);
        e2.seq = 10;
        entries.push(e1);
        entries.push(e2);

        let mut store = SwarmMemoryStore::from_entries(entries);
        // Next sequence should be 11
        let seq = store.append(agent_entry("c", 30));
        assert_eq!(seq, 11);
    }

    #[test]
    fn test_memory_entry_kind_display() {
        assert_eq!(MemoryEntryKind::SystemPrompt.to_string(), "system_prompt");
        assert_eq!(MemoryEntryKind::AgentTurn.to_string(), "agent_turn");
        assert_eq!(MemoryEntryKind::ToolResult.to_string(), "tool_result");
        assert_eq!(MemoryEntryKind::Summary.to_string(), "summary");
        assert_eq!(MemoryEntryKind::ErrorContext.to_string(), "error_context");
        assert_eq!(MemoryEntryKind::WorkPacket.to_string(), "work_packet");
    }

    #[test]
    fn test_memory_entry_serde() {
        let entry = agent_entry("test content", 42);
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: MemoryEntry = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.content, "test content");
        assert_eq!(parsed.estimated_tokens, 42);
        assert_eq!(parsed.kind, MemoryEntryKind::AgentTurn);
    }

    #[test]
    fn test_snapshot_serde() {
        let snap = MemorySnapshot {
            total_entries: 10,
            active_entries: 5,
            active_tokens: 1000,
            summary_count: 2,
            seq_range: (1, 10),
        };
        let json = serde_json::to_string(&snap).unwrap();
        let parsed: MemorySnapshot = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_entries, 10);
        assert_eq!(parsed.active_tokens, 1000);
    }
}
