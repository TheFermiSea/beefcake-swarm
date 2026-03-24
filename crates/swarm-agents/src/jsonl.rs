//! Shared JSONL (JSON Lines) file utilities.
//!
//! Used by mutation_archive, meta_reflection, telemetry, and reformulation
//! for append-only structured logging.

use std::path::Path;
use serde::{Serialize, de::DeserializeOwned};
use tracing::warn;

/// Append a single record as a JSON line to a file. Creates the file and parent dirs if needed.
pub fn append<T: Serialize>(path: &Path, record: &T) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(f) => f,
        Err(e) => {
            warn!(path = %path.display(), error = %e, "Failed to open JSONL file for append");
            return;
        }
    };
    use std::io::Write;
    if let Ok(json) = serde_json::to_string(record) {
        let _ = writeln!(file, "{}", json);
    }
}

/// Load all records from a JSONL file. Returns empty vec if file doesn't exist.
pub fn load_all<T: DeserializeOwned>(path: &Path) -> Vec<T> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}

/// Load the last N records from a JSONL file.
pub fn load_tail<T: DeserializeOwned>(path: &Path, limit: usize) -> Vec<T> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(_) => return vec![],
    };
    content
        .lines()
        .rev()
        .take(limit)
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect()
}
