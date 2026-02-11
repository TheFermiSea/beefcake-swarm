//! Column family definitions for RocksDB state store
//!
//! Each column family provides logical separation of data types
//! while sharing the same RocksDB instance.

/// Column family for ensemble sessions
pub const CF_SESSIONS: &str = "sessions";

/// Column family for ensemble tasks
pub const CF_TASKS: &str = "tasks";

/// Column family for model results
pub const CF_RESULTS: &str = "results";

/// Column family for voting records
pub const CF_VOTING: &str = "voting";

/// Column family for shared context
pub const CF_CONTEXT: &str = "context";

/// Column family for event history
pub const CF_EVENTS: &str = "events";

/// All column family names
pub const ALL_CFS: &[&str] = &[
    CF_SESSIONS,
    CF_TASKS,
    CF_RESULTS,
    CF_VOTING,
    CF_CONTEXT,
    CF_EVENTS,
];

/// Key prefixes for compound keys
pub mod keys {
    /// Create a session key
    pub fn session(session_id: &str) -> String {
        format!("sess:{}", session_id)
    }

    /// Create a task key
    pub fn task(task_id: &str) -> String {
        format!("task:{}", task_id)
    }

    /// Create a result key (task + model)
    pub fn result(task_id: &str, model_id: &str) -> String {
        format!("result:{}:{}", task_id, model_id)
    }

    /// Create a vote key
    pub fn vote(task_id: &str) -> String {
        format!("vote:{}", task_id)
    }

    /// Create a context key
    pub fn context(session_id: &str) -> String {
        format!("ctx:{}", session_id)
    }

    /// Create an event key (timestamp-based for ordering)
    pub fn event(timestamp_nanos: i64, event_id: &str) -> String {
        format!("evt:{:020}:{}", timestamp_nanos, event_id)
    }

    /// Parse event timestamp from key
    pub fn parse_event_timestamp(key: &str) -> Option<i64> {
        let parts: Vec<&str> = key.split(':').collect();
        if parts.len() >= 2 && parts[0] == "evt" {
            parts[1].parse().ok()
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_generation() {
        assert_eq!(keys::session("abc123"), "sess:abc123");
        assert_eq!(keys::task("task-1"), "task:task-1");
        assert_eq!(keys::result("task-1", "behemoth"), "result:task-1:behemoth");
        assert_eq!(keys::vote("task-1"), "vote:task-1");
        assert_eq!(keys::context("sess-1"), "ctx:sess-1");
    }

    #[test]
    fn test_event_key_ordering() {
        let key1 = keys::event(1000000000, "evt-1");
        let key2 = keys::event(2000000000, "evt-2");
        assert!(key1 < key2);
    }

    #[test]
    fn test_parse_event_timestamp() {
        let key = keys::event(12345, "evt-1");
        assert_eq!(keys::parse_event_timestamp(&key), Some(12345));
    }
}
