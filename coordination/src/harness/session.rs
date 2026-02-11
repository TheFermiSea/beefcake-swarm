//! Session manager for tracking agent sessions
//!
//! Handles session lifecycle, iteration tracking, and state management.

use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::types::{SessionState, SessionStatus};
use std::path::PathBuf;

/// Session manager
pub struct SessionManager {
    state: SessionState,
}

impl SessionManager {
    /// Create a new session
    pub fn new(working_directory: PathBuf, max_iterations: u32) -> Self {
        Self {
            state: SessionState::new(working_directory, max_iterations),
        }
    }

    /// Create from existing state (for resuming)
    pub fn from_state(state: SessionState) -> Self {
        Self { state }
    }

    /// Get current session state
    pub fn state(&self) -> &SessionState {
        &self.state
    }

    /// Get mutable session state
    pub fn state_mut(&mut self) -> &mut SessionState {
        &mut self.state
    }

    /// Get session ID
    pub fn session_id(&self) -> &str {
        &self.state.id
    }

    /// Get short session ID (first 8 chars)
    pub fn short_id(&self) -> &str {
        &self.state.id[..8.min(self.state.id.len())]
    }

    /// Get current iteration
    pub fn iteration(&self) -> u32 {
        self.state.iteration
    }

    /// Set initial commit for rollback
    pub fn set_initial_commit(&mut self, commit: String) {
        self.state.initial_commit = Some(commit);
    }

    /// Start the session (transition from Initializing to Active)
    pub fn start(&mut self) -> HarnessResult<()> {
        if self.state.status != SessionStatus::Initializing {
            return Err(HarnessError::InvalidStateTransition {
                from: self.state.status.to_string(),
                to: "active".to_string(),
            });
        }
        self.state.status = SessionStatus::Active;
        Ok(())
    }

    /// Begin next iteration
    pub fn next_iteration(&mut self) -> HarnessResult<u32> {
        if !self.state.can_continue() {
            if self.state.iteration >= self.state.max_iterations {
                self.state.status = SessionStatus::MaxIterationsReached;
                return Err(HarnessError::MaxIterationsReached {
                    max: self.state.max_iterations,
                });
            }
            return Err(HarnessError::session("Session cannot continue"));
        }

        self.state
            .next_iteration()
            .map_err(|max| HarnessError::MaxIterationsReached { max })
    }

    /// Set current feature being worked on
    pub fn set_current_feature(&mut self, feature_id: impl Into<String>) {
        self.state.current_feature = Some(feature_id.into());
    }

    /// Clear current feature
    pub fn clear_current_feature(&mut self) {
        self.state.current_feature = None;
    }

    /// Get current feature
    pub fn current_feature(&self) -> Option<&str> {
        self.state.current_feature.as_deref()
    }

    /// Mark session as completed
    pub fn complete(&mut self) {
        self.state.status = SessionStatus::Completed;
    }

    /// Mark session as failed
    pub fn fail(&mut self) {
        self.state.status = SessionStatus::Failed;
    }

    /// Mark session as paused
    pub fn pause(&mut self) {
        self.state.status = SessionStatus::Paused;
    }

    /// Check if session can continue
    pub fn can_continue(&self) -> bool {
        self.state.can_continue()
    }

    /// Get session status
    pub fn status(&self) -> SessionStatus {
        self.state.status
    }

    /// Get elapsed time as human-readable string
    pub fn elapsed_human(&self) -> String {
        let duration = self.state.elapsed();
        let seconds = duration.num_seconds();

        if seconds < 60 {
            format!("{}s", seconds)
        } else if seconds < 3600 {
            format!("{}m {}s", seconds / 60, seconds % 60)
        } else {
            format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
        }
    }

    /// Generate session summary
    pub fn summary(&self) -> SessionSummary {
        SessionSummary {
            session_id: self.state.id.clone(),
            status: self.state.status,
            iterations: self.state.iteration,
            max_iterations: self.state.max_iterations,
            elapsed: self.elapsed_human(),
            current_feature: self.state.current_feature.clone(),
        }
    }
}

/// Session summary for reporting
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionSummary {
    pub session_id: String,
    pub status: SessionStatus,
    pub iterations: u32,
    pub max_iterations: u32,
    pub elapsed: String,
    pub current_feature: Option<String>,
}

impl std::fmt::Display for SessionSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Session: {} ({})", &self.session_id[..8], self.status)?;
        writeln!(
            f,
            "Progress: {}/{} iterations",
            self.iterations, self.max_iterations
        )?;
        writeln!(f, "Elapsed: {}", self.elapsed)?;
        if let Some(ref feature) = self.current_feature {
            writeln!(f, "Working on: {}", feature)?;
        }
        Ok(())
    }
}

// ============================================================================
// Session State Persistence
// ============================================================================

/// Save session state to a JSON file
pub fn save_session_state(state: &SessionState, path: &std::path::Path) -> HarnessResult<()> {
    let json = serde_json::to_string_pretty(state)
        .map_err(|e| HarnessError::Io(std::io::Error::other(e)))?;
    std::fs::write(path, json).map_err(HarnessError::Io)?;
    Ok(())
}

/// Load session state from a JSON file
pub fn load_session_state(path: &std::path::Path) -> HarnessResult<Option<SessionState>> {
    if !path.exists() {
        return Ok(None);
    }

    let json = std::fs::read_to_string(path).map_err(HarnessError::Io)?;
    let state: SessionState = serde_json::from_str(&json)
        .map_err(|e| HarnessError::Io(std::io::Error::new(std::io::ErrorKind::InvalidData, e)))?;
    Ok(Some(state))
}

/// Delete persisted session state file
pub fn clear_session_state(path: &std::path::Path) -> HarnessResult<()> {
    if path.exists() {
        std::fs::remove_file(path).map_err(HarnessError::Io)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_lifecycle() {
        let mut manager = SessionManager::new(PathBuf::from("/tmp"), 5);

        assert_eq!(manager.status(), SessionStatus::Initializing);

        manager.start().unwrap();
        assert_eq!(manager.status(), SessionStatus::Active);

        assert_eq!(manager.next_iteration().unwrap(), 1);
        assert_eq!(manager.next_iteration().unwrap(), 2);

        manager.set_current_feature("feature-1");
        assert_eq!(manager.current_feature(), Some("feature-1"));

        manager.complete();
        assert_eq!(manager.status(), SessionStatus::Completed);
    }

    #[test]
    fn test_max_iterations() {
        let mut manager = SessionManager::new(PathBuf::from("/tmp"), 2);
        manager.start().unwrap();

        assert_eq!(manager.next_iteration().unwrap(), 1);
        assert_eq!(manager.next_iteration().unwrap(), 2);

        let result = manager.next_iteration();
        assert!(matches!(
            result,
            Err(HarnessError::MaxIterationsReached { max: 2 })
        ));
        assert_eq!(manager.status(), SessionStatus::MaxIterationsReached);
    }

    #[test]
    fn test_session_summary() {
        let mut manager = SessionManager::new(PathBuf::from("/tmp"), 10);
        manager.start().unwrap();
        manager.next_iteration().unwrap();
        manager.set_current_feature("test-feature");

        let summary = manager.summary();
        assert_eq!(summary.iterations, 1);
        assert_eq!(summary.max_iterations, 10);
        assert_eq!(summary.current_feature, Some("test-feature".to_string()));
    }

    #[test]
    fn test_short_id() {
        let manager = SessionManager::new(PathBuf::from("/tmp"), 10);
        let short = manager.short_id();
        assert_eq!(short.len(), 8);
    }

    #[test]
    fn test_session_state_persistence() {
        use tempfile::tempdir;

        let dir = tempdir().unwrap();
        let state_path = dir.path().join("session-state.json");

        // Create a session and modify it
        let mut manager = SessionManager::new(dir.path().to_path_buf(), 20);
        manager.start().unwrap();
        manager.next_iteration().unwrap();
        manager.next_iteration().unwrap();
        manager.set_current_feature("test-feature");

        // Save state
        save_session_state(manager.state(), &state_path).unwrap();
        assert!(state_path.exists());

        // Load state
        let loaded = load_session_state(&state_path).unwrap();
        assert!(loaded.is_some());

        let loaded_state = loaded.unwrap();
        assert_eq!(loaded_state.id, manager.state().id);
        assert_eq!(loaded_state.iteration, 2);
        assert_eq!(loaded_state.max_iterations, 20);
        assert_eq!(
            loaded_state.current_feature,
            Some("test-feature".to_string())
        );
        assert_eq!(loaded_state.status, SessionStatus::Active);

        // Create manager from loaded state
        let resumed_manager = SessionManager::from_state(loaded_state);
        assert_eq!(resumed_manager.iteration(), 2);
        assert_eq!(resumed_manager.current_feature(), Some("test-feature"));

        // Clear state
        clear_session_state(&state_path).unwrap();
        assert!(!state_path.exists());

        // Load should return None now
        let empty = load_session_state(&state_path).unwrap();
        assert!(empty.is_none());
    }
}
