//! Inter-worker communication via a shared workpad file.
//!
//! Workers executing concurrent subtasks use these tools to announce
//! interface changes (struct fields, function signatures, trait methods)
//! and check what other workers have announced.
//!
//! The workpad is a JSONL file (`.swarm-workpad.jsonl`) in the worktree root.
//! Each line is a structured announcement. File-based so it survives worker
//! restarts and is debuggable via `cat`.

use std::path::{Path, PathBuf};

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

use super::ToolError;

/// Default workpad filename within the worktree.
pub const WORKPAD_FILENAME: &str = ".swarm-workpad.jsonl";

/// A single workpad entry written by a worker.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkpadEntry {
    /// Which worker wrote this (e.g., "subtask-1").
    pub worker: String,
    /// Announcement type: "interface_change", "dependency_added", "done", "note".
    #[serde(rename = "type")]
    pub entry_type: String,
    /// Which file was affected.
    pub file: String,
    /// Human-readable detail about the change.
    pub detail: String,
}

/// Initialize an empty workpad file in the worktree.
///
/// Called by the orchestrator before dispatching concurrent subtasks.
/// If the file already exists (e.g., from a previous attempt), it is truncated.
pub fn init_workpad(wt_path: &Path) -> Result<(), std::io::Error> {
    let path = wt_path.join(WORKPAD_FILENAME);
    std::fs::write(&path, "")?;
    Ok(())
}

/// Read all workpad entries from the file.
pub fn read_workpad(wt_path: &Path) -> Result<Vec<WorkpadEntry>, ToolError> {
    let path = wt_path.join(WORKPAD_FILENAME);
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(e) => return Err(ToolError::Io(e)),
    };

    let mut entries = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Ok(entry) = serde_json::from_str::<WorkpadEntry>(line) {
            entries.push(entry);
        }
    }
    Ok(entries)
}

// ── AnnounceTool ──────────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct AnnounceArgs {
    /// Type of announcement: "interface_change", "dependency_added", "done", "note".
    #[serde(rename = "type")]
    pub entry_type: String,
    /// Which file was changed.
    pub file: String,
    /// Description of the change (e.g., "Added `timeout` field to FooConfig").
    pub detail: String,
}

/// Tool for workers to announce changes to the shared workpad.
pub struct AnnounceTool {
    working_dir: PathBuf,
    worker_id: String,
}

impl AnnounceTool {
    pub fn new(working_dir: &Path, worker_id: &str) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
            worker_id: worker_id.to_string(),
        }
    }
}

impl Tool for AnnounceTool {
    const NAME: &'static str = "announce";
    type Error = ToolError;
    type Args = AnnounceArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "announce".into(),
            description: "Announce a change to other concurrent workers. Use after changing \
                          any public interface (struct fields, function signatures, trait methods) \
                          or adding dependencies. Other workers can see your announcements via \
                          check_announcements."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["type", "file", "detail"],
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": ["interface_change", "dependency_added", "done", "note"],
                        "description": "Type of announcement"
                    },
                    "file": {
                        "type": "string",
                        "description": "File that was changed"
                    },
                    "detail": {
                        "type": "string",
                        "description": "Description of the change"
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let entry = WorkpadEntry {
            worker: self.worker_id.clone(),
            entry_type: args.entry_type,
            file: args.file,
            detail: args.detail.clone(),
        };

        let line = serde_json::to_string(&entry)
            .map_err(|e| ToolError::Io(std::io::Error::other(format!("JSON serialize: {e}"))))?;

        let path = self.working_dir.join(WORKPAD_FILENAME);

        // Append atomically — use OpenOptions to avoid race conditions.
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(ToolError::Io)?;
        writeln!(file, "{line}").map_err(ToolError::Io)?;

        Ok(format!("Announced: {}", args.detail))
    }
}

// ── CheckAnnouncementsTool ───────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct CheckAnnouncementsArgs {}

/// Tool for workers to read announcements from other concurrent workers.
pub struct CheckAnnouncementsTool {
    working_dir: PathBuf,
    worker_id: String,
}

impl CheckAnnouncementsTool {
    pub fn new(working_dir: &Path, worker_id: &str) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
            worker_id: worker_id.to_string(),
        }
    }
}

impl Tool for CheckAnnouncementsTool {
    const NAME: &'static str = "check_announcements";
    type Error = ToolError;
    type Args = CheckAnnouncementsArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "check_announcements".into(),
            description: "Check announcements from other concurrent workers. Call this before \
                          your final edits to see if other workers changed interfaces you depend on."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {}
            }),
        }
    }

    async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
        let entries = read_workpad(&self.working_dir)?;

        // Filter to entries from OTHER workers only.
        let other_entries: Vec<&WorkpadEntry> = entries
            .iter()
            .filter(|e| e.worker != self.worker_id)
            .collect();

        if other_entries.is_empty() {
            return Ok("No announcements from other workers.".to_string());
        }

        let mut output = format!(
            "{} announcement(s) from other workers:\n",
            other_entries.len()
        );
        for entry in &other_entries {
            output.push_str(&format!(
                "- [{}] {} in {}: {}\n",
                entry.worker, entry.entry_type, entry.file, entry.detail
            ));
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_workpad_creates_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        init_workpad(dir.path()).unwrap();
        let path = dir.path().join(WORKPAD_FILENAME);
        assert!(path.exists());
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
    }

    #[test]
    fn read_empty_workpad() {
        let dir = tempfile::tempdir().unwrap();
        init_workpad(dir.path()).unwrap();
        let entries = read_workpad(dir.path()).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn read_workpad_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let entries = read_workpad(dir.path()).unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn check_announcements_filters_own_worker() {
        let dir = tempfile::tempdir().unwrap();
        init_workpad(dir.path()).unwrap();

        // Worker 1 announces
        let tool1 = AnnounceTool::new(dir.path(), "subtask-1");
        tool1
            .call(AnnounceArgs {
                entry_type: "interface_change".to_string(),
                file: "src/types.rs".to_string(),
                detail: "Changed FooConfig".to_string(),
            })
            .await
            .unwrap();

        // Worker 2 announces
        let tool2 = AnnounceTool::new(dir.path(), "subtask-2");
        tool2
            .call(AnnounceArgs {
                entry_type: "done".to_string(),
                file: "src/handler.rs".to_string(),
                detail: "Finished handler update".to_string(),
            })
            .await
            .unwrap();

        // Worker 1 checks — should only see worker 2's announcement
        let check1 = CheckAnnouncementsTool::new(dir.path(), "subtask-1");
        let result = check1.call(CheckAnnouncementsArgs {}).await.unwrap();
        assert!(result.contains("subtask-2"));
        assert!(!result.contains("subtask-1"));
        assert!(result.contains("1 announcement(s)"));

        // Worker 2 checks — should only see worker 1's announcement
        let check2 = CheckAnnouncementsTool::new(dir.path(), "subtask-2");
        let result = check2.call(CheckAnnouncementsArgs {}).await.unwrap();
        assert!(result.contains("subtask-1"));
        assert!(result.contains("1 announcement(s)"));
    }
}
