//! File-backed append-only session log.
//!
//! Stores orchestrator events as newline-delimited JSON (JSONL).
//! Each line is a complete [`SessionEvent`] that can be independently
//! parsed, making the log resilient to partial writes (a truncated
//! last line is simply skipped on replay).
//!
//! # Durability
//!
//! Each `append()` call writes the JSON line and calls `fsync` to
//! ensure the event reaches stable storage before returning. This
//! makes the log crash-safe: if the process dies between events,
//! all previously-appended events are recoverable.
//!
//! # Concurrency
//!
//! The log is designed for single-writer use (one orchestrator process
//! per session). Multiple readers are safe — `load()` takes a snapshot
//! of the file at read time.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use tracing::{debug, warn};

use super::events::{EventId, EventKind, SessionEvent};

/// Append-only session log backed by a JSONL file.
pub struct SessionLog {
    /// Path to the JSONL file.
    path: PathBuf,
    /// Open file handle for appending (None if read-only).
    writer: Option<File>,
    /// Next event ID to assign (monotonically increasing).
    next_id: AtomicU64,
}

impl SessionLog {
    /// Create a new session log at the given path.
    ///
    /// If the file already exists, the log resumes from the highest
    /// event ID found in it (for crash recovery). If it doesn't exist,
    /// it is created.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();

        // Ensure parent directory exists.
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("creating session log dir: {}", parent.display()))?;
        }

        // Scan existing events to find the highest ID.
        let max_id = if path.exists() {
            Self::scan_max_id(&path)?
        } else {
            0
        };

        let writer = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening session log: {}", path.display()))?;

        debug!(
            path = %path.display(),
            resume_from = max_id,
            "Session log opened"
        );

        Ok(Self {
            path,
            writer: Some(writer),
            next_id: AtomicU64::new(max_id + 1),
        })
    }

    /// Open a session log in read-only mode (for replay/query).
    pub fn open_readonly(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            anyhow::bail!("session log not found: {}", path.display());
        }
        let max_id = Self::scan_max_id(&path)?;
        Ok(Self {
            path,
            writer: None,
            next_id: AtomicU64::new(max_id + 1),
        })
    }

    /// Append an event to the log.
    ///
    /// Assigns a monotonically increasing ID, serializes to JSON,
    /// writes the line, and fsyncs. Returns the assigned event ID.
    pub fn append(&self, kind: EventKind) -> Result<EventId> {
        let writer = self
            .writer
            .as_ref()
            .context("session log opened in read-only mode")?;

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let event = SessionEvent::new(id, kind);

        let mut line = serde_json::to_string(&event)
            .context("serializing session event")?;
        line.push('\n');

        // Write + fsync for durability.
        (&*writer)
            .write_all(line.as_bytes())
            .with_context(|| format!("writing to session log: {}", self.path.display()))?;
        writer
            .sync_data()
            .with_context(|| format!("fsyncing session log: {}", self.path.display()))?;

        Ok(id)
    }

    /// Load all events from the log file.
    ///
    /// Tolerant of a truncated last line (from a crash mid-write).
    pub fn load_all(&self) -> Result<Vec<SessionEvent>> {
        Self::load_from_path(&self.path)
    }

    /// Load events from a path (static, for use before opening).
    pub fn load_from_path(path: &Path) -> Result<Vec<SessionEvent>> {
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(path)
            .with_context(|| format!("reading session log: {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut events = Vec::new();

        for (line_num, line) in reader.lines().enumerate() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    warn!(line = line_num + 1, error = %e, "skipping unreadable line");
                    continue;
                }
            };

            if line.trim().is_empty() {
                continue;
            }

            match serde_json::from_str::<SessionEvent>(&line) {
                Ok(event) => events.push(event),
                Err(e) => {
                    // Tolerate truncated last line (crash during write).
                    warn!(
                        line = line_num + 1,
                        error = %e,
                        "skipping malformed event (truncated write?)"
                    );
                }
            }
        }

        Ok(events)
    }

    /// Load events with IDs greater than `after_id`.
    pub fn load_since(&self, after_id: EventId) -> Result<Vec<SessionEvent>> {
        let all = self.load_all()?;
        Ok(all.into_iter().filter(|e| e.id > after_id).collect())
    }

    /// Load events of a specific type.
    pub fn load_by_type(&self, type_name: &str) -> Result<Vec<SessionEvent>> {
        let all = self.load_all()?;
        Ok(all
            .into_iter()
            .filter(|e| event_type_tag(&e.kind) == type_name)
            .collect())
    }

    /// Get the path to the session log file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Get the current highest event ID.
    pub fn current_id(&self) -> EventId {
        self.next_id.load(Ordering::SeqCst).saturating_sub(1)
    }

    /// Check if a session log exists at the given path and has events.
    pub fn exists_with_events(path: impl AsRef<Path>) -> bool {
        let path = path.as_ref();
        if !path.exists() {
            return false;
        }
        // Quick check: file has at least one non-empty line.
        fs::metadata(path)
            .map(|m| m.len() > 2) // at least "{}\n"
            .unwrap_or(false)
    }

    /// Scan the file for the highest event ID (for resume).
    fn scan_max_id(path: &Path) -> Result<EventId> {
        let file = File::open(path)
            .with_context(|| format!("scanning session log: {}", path.display()))?;
        let reader = BufReader::new(file);
        let mut max_id: EventId = 0;

        for line in reader.lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };
            if line.trim().is_empty() {
                continue;
            }
            // Fast path: extract "id": N without full deserialization.
            if let Some(id) = extract_id_fast(&line) {
                if id > max_id {
                    max_id = id;
                }
            }
        }

        Ok(max_id)
    }
}

/// Extract the event ID from a JSON line without full deserialization.
/// Looks for `"id":N` near the start of the line.
fn extract_id_fast(line: &str) -> Option<EventId> {
    // The ID is always the first field: {"id":123,...
    let start = line.find("\"id\":")?;
    let after = &line[start + 5..];
    let end = after.find(|c: char| !c.is_ascii_digit())?;
    after[..end].parse().ok()
}

/// Get the serde tag name for an event kind (matches #[serde(tag = "type")]).
fn event_type_tag(kind: &EventKind) -> &'static str {
    match kind {
        EventKind::SessionStarted { .. } => "session_started",
        EventKind::StateTransition { .. } => "state_transition",
        EventKind::IterationStarted { .. } => "iteration_started",
        EventKind::WorktreeProvisioned { .. } => "worktree_provisioned",
        EventKind::LlmTurnCompleted { .. } => "llm_turn_completed",
        EventKind::ToolCallCompleted { .. } => "tool_call_completed",
        EventKind::WorkerDelegated { .. } => "worker_delegated",
        EventKind::WorkerCompleted { .. } => "worker_completed",
        EventKind::VerifierResult { .. } => "verifier_result",
        EventKind::EscalationTriggered { .. } => "escalation_triggered",
        EventKind::IterationCompleted { .. } => "iteration_completed",
        EventKind::ContextRebuilt { .. } => "context_rebuilt",
        EventKind::NoChangeDetected { .. } => "no_change_detected",
        EventKind::SessionCompleted { .. } => "session_completed",
        EventKind::CheckpointWritten { .. } => "checkpoint_written",
        EventKind::Note { .. } => "note",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_machine::OrchestratorState;
    use tempfile::TempDir;

    #[test]
    fn test_append_and_load() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test-session.jsonl");

        let log = SessionLog::open(&path).unwrap();

        let id1 = log
            .append(EventKind::SessionStarted {
                issue_id: "test-123".into(),
                objective: "Fix the bug".into(),
                base_commit: Some("abc123".into()),
            })
            .unwrap();
        assert_eq!(id1, 1);

        let id2 = log
            .append(EventKind::StateTransition {
                from: OrchestratorState::SelectingIssue,
                to: OrchestratorState::PreparingWorktree,
                iteration: 0,
                reason: None,
            })
            .unwrap();
        assert_eq!(id2, 2);

        let events = log.load_all().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].id, 1);
        assert_eq!(events[1].id, 2);
    }

    #[test]
    fn test_resume_from_existing() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test-session.jsonl");

        // Write some events.
        {
            let log = SessionLog::open(&path).unwrap();
            log.append(EventKind::Note {
                message: "first".into(),
            })
            .unwrap();
            log.append(EventKind::Note {
                message: "second".into(),
            })
            .unwrap();
        }

        // Reopen — should resume from ID 3.
        let log = SessionLog::open(&path).unwrap();
        let id3 = log
            .append(EventKind::Note {
                message: "third".into(),
            })
            .unwrap();
        assert_eq!(id3, 3);

        let events = log.load_all().unwrap();
        assert_eq!(events.len(), 3);
    }

    #[test]
    fn test_load_since() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test-session.jsonl");

        let log = SessionLog::open(&path).unwrap();
        for i in 0..5 {
            log.append(EventKind::Note {
                message: format!("event {}", i),
            })
            .unwrap();
        }

        let since_3 = log.load_since(3).unwrap();
        assert_eq!(since_3.len(), 2);
        assert_eq!(since_3[0].id, 4);
        assert_eq!(since_3[1].id, 5);
    }

    #[test]
    fn test_tolerates_truncated_last_line() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test-session.jsonl");

        // Write valid events + a truncated line.
        {
            let log = SessionLog::open(&path).unwrap();
            log.append(EventKind::Note {
                message: "valid".into(),
            })
            .unwrap();
        }
        // Append garbage to simulate crash mid-write.
        fs::write(
            &path,
            format!(
                "{}\n{{\"id\":2,\"timestamp\":\"2026-04-11T00:00:00Z\",\"kind\":{{\"type\":\"not",
                fs::read_to_string(&path).unwrap().trim()
            ),
        )
        .unwrap();

        let events = SessionLog::load_from_path(&path).unwrap();
        assert_eq!(events.len(), 1); // Only the valid event.
    }

    #[test]
    fn test_extract_id_fast() {
        assert_eq!(extract_id_fast(r#"{"id":42,"timestamp":"..."}"#), Some(42));
        assert_eq!(extract_id_fast(r#"{"id":1,"kind":{}}"#), Some(1));
        assert_eq!(extract_id_fast(r#"no id here"#), None);
    }

    #[test]
    fn test_exists_with_events() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test-session.jsonl");

        assert!(!SessionLog::exists_with_events(&path));

        let log = SessionLog::open(&path).unwrap();
        // Empty file — exists but no events yet.
        // (Opening creates the file but doesn't write anything.)
        assert!(!SessionLog::exists_with_events(&path));

        log.append(EventKind::Note {
            message: "hello".into(),
        })
        .unwrap();
        assert!(SessionLog::exists_with_events(&path));
    }
}
