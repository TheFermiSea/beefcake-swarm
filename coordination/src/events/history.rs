//! Event history and replay functionality
//!
//! Provides the ability to replay events from RocksDB for recovery
//! and debugging purposes.

use chrono::{DateTime, Duration, Utc};
use tracing::{debug, info};

use super::types::EnsembleEvent;
use crate::state::SharedStateStore;

/// Error type for history operations
#[derive(Debug, thiserror::Error)]
pub enum HistoryError {
    #[error("Store error: {0}")]
    StoreError(String),

    #[error("Event parsing error: {0}")]
    ParseError(String),
}

/// Result type for history operations
pub type HistoryResult<T> = Result<T, HistoryError>;

/// Event history manager for replay and querying
pub struct EventHistory {
    store: SharedStateStore,
}

impl EventHistory {
    /// Create a new event history manager
    pub fn new(store: SharedStateStore) -> Self {
        Self { store }
    }

    /// Get all events in a time range
    pub fn get_events(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> HistoryResult<Vec<EnsembleEvent>> {
        let start_nanos = start.timestamp_nanos_opt().unwrap_or(0);
        let end_nanos = end.timestamp_nanos_opt().unwrap_or(i64::MAX);

        let events: Vec<EnsembleEvent> = self
            .store
            .get_events_range(start_nanos, end_nanos)
            .map_err(|e| HistoryError::StoreError(e.to_string()))?
            .into_iter()
            .map(|(_, event)| event)
            .collect();

        debug!(
            count = events.len(),
            "Retrieved {} events from history",
            events.len()
        );

        Ok(events)
    }

    /// Get events for the last N minutes
    pub fn get_recent_events(&self, minutes: i64) -> HistoryResult<Vec<EnsembleEvent>> {
        let end = Utc::now();
        let start = end - Duration::minutes(minutes);
        self.get_events(start, end)
    }

    /// Get events for a specific session
    pub fn get_session_events(&self, session_id: &str) -> HistoryResult<Vec<EnsembleEvent>> {
        // Get all events and filter by session
        // In a production system, we might want a secondary index
        let all_events = self.get_recent_events(60 * 24)?; // Last 24 hours

        let session_events: Vec<EnsembleEvent> = all_events
            .into_iter()
            .filter(|e| e.session_id() == Some(session_id))
            .collect();

        Ok(session_events)
    }

    /// Get events for a specific task
    pub fn get_task_events(&self, task_id: &str) -> HistoryResult<Vec<EnsembleEvent>> {
        let all_events = self.get_recent_events(60 * 24)?;

        let task_events: Vec<EnsembleEvent> = all_events
            .into_iter()
            .filter(|e| e.task_id() == Some(task_id))
            .collect();

        Ok(task_events)
    }

    /// Replay events through a callback
    pub async fn replay<F, Fut>(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        mut callback: F,
    ) -> HistoryResult<ReplayStats>
    where
        F: FnMut(EnsembleEvent) -> Fut,
        Fut: std::future::Future<Output = ()>,
    {
        let events = self.get_events(start, end)?;
        let total = events.len();

        info!(total, "Starting event replay");

        let mut stats = ReplayStats::new();
        for event in events {
            stats.record_event(&event);
            callback(event).await;
        }

        info!(
            total = stats.total_events,
            sessions = stats.sessions_seen,
            tasks = stats.tasks_seen,
            "Event replay complete"
        );

        Ok(stats)
    }

    /// Prune old events to manage storage
    pub fn prune_before(&self, cutoff: DateTime<Utc>) -> HistoryResult<usize> {
        let cutoff_nanos = cutoff.timestamp_nanos_opt().unwrap_or(0);
        let count = self
            .store
            .prune_events_before(cutoff_nanos)
            .map_err(|e| HistoryError::StoreError(e.to_string()))?;

        info!(count, cutoff = %cutoff, "Pruned old events");
        Ok(count)
    }

    /// Get event statistics for a time range
    pub fn get_stats(&self, start: DateTime<Utc>, end: DateTime<Utc>) -> HistoryResult<EventStats> {
        let events = self.get_events(start, end)?;
        Ok(EventStats::from_events(&events))
    }
}

/// Statistics from replay or query
#[derive(Debug, Default)]
pub struct ReplayStats {
    pub total_events: usize,
    pub sessions_seen: usize,
    pub tasks_seen: usize,
    pub errors_seen: usize,
    sessions: std::collections::HashSet<String>,
    tasks: std::collections::HashSet<String>,
}

impl ReplayStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_event(&mut self, event: &EnsembleEvent) {
        self.total_events += 1;

        if let Some(session_id) = event.session_id() {
            if self.sessions.insert(session_id.to_string()) {
                self.sessions_seen += 1;
            }
        }

        if let Some(task_id) = event.task_id() {
            if self.tasks.insert(task_id.to_string()) {
                self.tasks_seen += 1;
            }
        }

        if matches!(event, EnsembleEvent::TaskFailed { .. }) {
            self.errors_seen += 1;
        }
    }
}

/// Aggregate statistics for events
#[derive(Debug, Default, serde::Serialize)]
pub struct EventStats {
    pub total_events: usize,
    pub events_by_type: std::collections::HashMap<String, usize>,
    pub unique_sessions: usize,
    pub unique_tasks: usize,
    pub model_loads: usize,
    pub model_unloads: usize,
    pub consensus_reached: usize,
    pub arbitrations: usize,
    pub failures: usize,
}

impl EventStats {
    pub fn from_events(events: &[EnsembleEvent]) -> Self {
        let mut stats = Self::default();
        let mut sessions = std::collections::HashSet::new();
        let mut tasks = std::collections::HashSet::new();

        for event in events {
            stats.total_events += 1;

            let event_type = event.event_type().to_string();
            *stats.events_by_type.entry(event_type).or_insert(0) += 1;

            if let Some(sid) = event.session_id() {
                sessions.insert(sid.to_string());
            }
            if let Some(tid) = event.task_id() {
                tasks.insert(tid.to_string());
            }

            match event {
                EnsembleEvent::ModelLoaded { .. } => stats.model_loads += 1,
                EnsembleEvent::ModelUnloaded { .. } => stats.model_unloads += 1,
                EnsembleEvent::ConsensusReached { .. } => stats.consensus_reached += 1,
                EnsembleEvent::ArbitrationCompleted { .. } => stats.arbitrations += 1,
                EnsembleEvent::TaskFailed { .. } => stats.failures += 1,
                _ => {}
            }
        }

        stats.unique_sessions = sessions.len();
        stats.unique_tasks = tasks.len();

        stats
    }
}

/// Builder for replaying events with transformations
pub struct ReplayBuilder {
    store: SharedStateStore,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    filter_session: Option<String>,
    filter_task: Option<String>,
    filter_types: Option<Vec<String>>,
}

impl ReplayBuilder {
    /// Create a new replay builder
    pub fn new(store: SharedStateStore) -> Self {
        let now = Utc::now();
        Self {
            store,
            start: now - Duration::hours(24),
            end: now,
            filter_session: None,
            filter_task: None,
            filter_types: None,
        }
    }

    /// Set the time range for replay
    pub fn time_range(mut self, start: DateTime<Utc>, end: DateTime<Utc>) -> Self {
        self.start = start;
        self.end = end;
        self
    }

    /// Filter by session ID
    pub fn session(mut self, session_id: &str) -> Self {
        self.filter_session = Some(session_id.to_string());
        self
    }

    /// Filter by task ID
    pub fn task(mut self, task_id: &str) -> Self {
        self.filter_task = Some(task_id.to_string());
        self
    }

    /// Filter by event types
    pub fn event_types(mut self, types: Vec<&str>) -> Self {
        self.filter_types = Some(types.into_iter().map(String::from).collect());
        self
    }

    /// Execute replay and collect events
    pub fn collect(self) -> HistoryResult<Vec<EnsembleEvent>> {
        let history = EventHistory::new(self.store);
        let mut events = history.get_events(self.start, self.end)?;

        // Apply filters
        if let Some(ref session_id) = self.filter_session {
            events.retain(|e| e.session_id() == Some(session_id.as_str()));
        }

        if let Some(ref task_id) = self.filter_task {
            events.retain(|e| e.task_id() == Some(task_id.as_str()));
        }

        if let Some(ref types) = self.filter_types {
            events.retain(|e| types.contains(&e.event_type().to_string()));
        }

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ModelId, StateStore};
    use tempfile::tempdir;

    fn test_history() -> (EventHistory, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let store = StateStore::open(dir.path().join("test.db"))
            .unwrap()
            .shared();
        (EventHistory::new(store), dir)
    }

    #[test]
    fn test_event_stats() {
        let events = vec![
            EnsembleEvent::SessionCreated {
                session_id: "s1".to_string(),
                harness_session_id: None,
                timestamp: Utc::now(),
            },
            EnsembleEvent::TaskCreated {
                task_id: "t1".to_string(),
                session_id: "s1".to_string(),
                prompt_preview: "test".to_string(),
                require_consensus: true,
                timestamp: Utc::now(),
            },
            EnsembleEvent::ModelLoaded {
                model_id: ModelId::Opus45,
                load_time_ms: 1000,
                timestamp: Utc::now(),
            },
        ];

        let stats = EventStats::from_events(&events);

        assert_eq!(stats.total_events, 3);
        assert_eq!(stats.unique_sessions, 1);
        assert_eq!(stats.unique_tasks, 1);
        assert_eq!(stats.model_loads, 1);
    }

    #[test]
    fn test_replay_stats() {
        let mut stats = ReplayStats::new();

        let event1 = EnsembleEvent::TaskCreated {
            task_id: "t1".to_string(),
            session_id: "s1".to_string(),
            prompt_preview: "test".to_string(),
            require_consensus: true,
            timestamp: Utc::now(),
        };

        let event2 = EnsembleEvent::TaskCreated {
            task_id: "t2".to_string(),
            session_id: "s1".to_string(),
            prompt_preview: "test".to_string(),
            require_consensus: true,
            timestamp: Utc::now(),
        };

        stats.record_event(&event1);
        stats.record_event(&event2);

        assert_eq!(stats.total_events, 2);
        assert_eq!(stats.sessions_seen, 1);
        assert_eq!(stats.tasks_seen, 2);
    }
}
