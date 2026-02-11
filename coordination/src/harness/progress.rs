//! Progress tracker for claude-progress.txt
//!
//! Handles appending and reading progress entries.

use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::types::{ProgressEntry, ProgressMarker};
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// Progress tracker managing claude-progress.txt
pub struct ProgressTracker {
    path: PathBuf,
}

impl ProgressTracker {
    /// Create tracker for given path
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    /// Append an entry to the progress file
    pub fn append(&self, entry: &ProgressEntry) -> HarnessResult<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        writeln!(file, "{}", entry.to_log_line())
            .map_err(|e| HarnessError::progress(e.to_string()))?;

        Ok(())
    }

    /// Read last N entries from progress file
    pub fn read_last(&self, n: usize) -> HarnessResult<Vec<ProgressEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);

        let lines: Vec<String> = reader.lines().collect::<Result<_, _>>()?;

        let entries: Vec<ProgressEntry> = lines
            .iter()
            .rev()
            .take(n)
            .filter_map(|line| ProgressEntry::from_log_line(line))
            .collect();

        Ok(entries.into_iter().rev().collect())
    }

    /// Read all entries
    pub fn read_all(&self) -> HarnessResult<Vec<ProgressEntry>> {
        if !self.path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);

        let entries: Vec<ProgressEntry> = reader
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| ProgressEntry::from_log_line(&line))
            .collect();

        Ok(entries)
    }

    /// Find last session start entry
    pub fn last_session_start(&self) -> HarnessResult<Option<ProgressEntry>> {
        let entries = self.read_all()?;
        Ok(entries
            .into_iter()
            .rev()
            .find(|e| matches!(e.marker, ProgressMarker::SessionStart)))
    }

    /// Get entries for a specific session
    pub fn session_entries(&self, session_id: &str) -> HarnessResult<Vec<ProgressEntry>> {
        let entries = self.read_all()?;
        Ok(entries
            .into_iter()
            .filter(|e| e.session_id.starts_with(session_id))
            .collect())
    }

    /// Log session start
    pub fn log_session_start(
        &self,
        session_id: &str,
        summary: impl Into<String>,
    ) -> HarnessResult<()> {
        let entry = ProgressEntry::new(session_id, 0, ProgressMarker::SessionStart, summary);
        self.append(&entry)
    }

    /// Log feature start
    pub fn log_feature_start(
        &self,
        session_id: &str,
        iteration: u32,
        feature_id: &str,
        summary: impl Into<String>,
    ) -> HarnessResult<()> {
        let entry =
            ProgressEntry::new(session_id, iteration, ProgressMarker::FeatureStart, summary)
                .with_feature(feature_id);
        self.append(&entry)
    }

    /// Log feature complete
    pub fn log_feature_complete(
        &self,
        session_id: &str,
        iteration: u32,
        feature_id: &str,
        summary: impl Into<String>,
    ) -> HarnessResult<()> {
        let entry = ProgressEntry::new(
            session_id,
            iteration,
            ProgressMarker::FeatureComplete,
            summary,
        )
        .with_feature(feature_id);
        self.append(&entry)
    }

    /// Log checkpoint
    pub fn log_checkpoint(
        &self,
        session_id: &str,
        iteration: u32,
        commit_hash: &str,
    ) -> HarnessResult<()> {
        let entry = ProgressEntry::new(
            session_id,
            iteration,
            ProgressMarker::Checkpoint,
            format!("Created checkpoint at {}", commit_hash),
        )
        .with_metadata("commit", serde_json::Value::String(commit_hash.to_string()));
        self.append(&entry)
    }

    /// Log session end
    pub fn log_session_end(
        &self,
        session_id: &str,
        iteration: u32,
        summary: impl Into<String>,
    ) -> HarnessResult<()> {
        let entry = ProgressEntry::new(session_id, iteration, ProgressMarker::SessionEnd, summary);
        self.append(&entry)
    }

    /// Log error
    pub fn log_error(
        &self,
        session_id: &str,
        iteration: u32,
        error: impl Into<String>,
    ) -> HarnessResult<()> {
        let entry = ProgressEntry::new(session_id, iteration, ProgressMarker::Error, error);
        self.append(&entry)
    }

    /// Clear the progress file (for compaction)
    pub fn clear(&self) -> HarnessResult<()> {
        // Truncate the file to zero length
        File::create(&self.path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_append_and_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("progress.txt");
        let tracker = ProgressTracker::new(&path);

        tracker
            .log_session_start("test-session", "Starting test")
            .unwrap();
        tracker
            .log_feature_start("test-session", 1, "feature-1", "Working on feature")
            .unwrap();
        tracker
            .log_feature_complete("test-session", 1, "feature-1", "Feature done")
            .unwrap();

        let entries = tracker.read_all().unwrap();
        assert_eq!(entries.len(), 3);
        assert!(matches!(entries[0].marker, ProgressMarker::SessionStart));
        assert!(matches!(entries[1].marker, ProgressMarker::FeatureStart));
        assert!(matches!(entries[2].marker, ProgressMarker::FeatureComplete));
    }

    #[test]
    fn test_read_last() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("progress.txt");
        let tracker = ProgressTracker::new(&path);

        for i in 0..10 {
            tracker
                .log_session_start("session", format!("Entry {}", i))
                .unwrap();
        }

        let last3 = tracker.read_last(3).unwrap();
        assert_eq!(last3.len(), 3);
        assert!(last3[2].summary.contains("Entry 9"));
    }

    #[test]
    fn test_empty_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("nonexistent.txt");
        let tracker = ProgressTracker::new(&path);

        let entries = tracker.read_all().unwrap();
        assert!(entries.is_empty());
    }
}
