//! RocksDB-backed state store for ensemble coordination
//!
//! Provides persistent storage with column families for logical data separation.
//! Uses bincode for efficient binary serialization internally.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use rocksdb::{ColumnFamilyDescriptor, Options, DB};
use serde::{de::DeserializeOwned, Serialize};

use super::schema::{self, ALL_CFS};
use super::types::*;

/// Error type for state store operations
#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("RocksDB error: {0}")]
    RocksDb(#[from] rocksdb::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Deserialization error: {0}")]
    Deserialization(String),

    #[error("Key not found: {0}")]
    NotFound(String),

    #[error("Lock poisoned")]
    LockPoisoned,

    #[error("Column family not found: {0}")]
    ColumnFamilyNotFound(String),
}

/// Result type for state store operations
pub type StoreResult<T> = Result<T, StoreError>;

/// Shared reference to StateStore
pub type SharedStateStore = Arc<StateStore>;

/// RocksDB-backed persistent state store
pub struct StateStore {
    db: RwLock<DB>,
    path: PathBuf,
}

impl StateStore {
    /// Open or create a state store at the given path
    pub fn open(path: impl Into<PathBuf>) -> StoreResult<Self> {
        let path = path.into();

        // Configure options
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.create_missing_column_families(true);

        // Define column families
        let cf_descriptors: Vec<ColumnFamilyDescriptor> = ALL_CFS
            .iter()
            .map(|name| ColumnFamilyDescriptor::new(*name, Options::default()))
            .collect();

        // Open database with column families
        let db = DB::open_cf_descriptors(&opts, &path, cf_descriptors)?;

        Ok(Self {
            db: RwLock::new(db),
            path,
        })
    }

    /// Create a shared reference to this store
    pub fn shared(self) -> SharedStateStore {
        Arc::new(self)
    }

    /// Get the database path
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    // =========================================================================
    // Generic operations
    // =========================================================================

    /// Store a value in a column family
    fn put<T: Serialize>(&self, cf_name: &str, key: &str, value: &T) -> StoreResult<()> {
        let db = self.db.read().map_err(|_| StoreError::LockPoisoned)?;
        let cf = db
            .cf_handle(cf_name)
            .ok_or_else(|| StoreError::ColumnFamilyNotFound(cf_name.to_string()))?;

        let bytes =
            bincode::serialize(value).map_err(|e| StoreError::Serialization(e.to_string()))?;

        db.put_cf(&cf, key.as_bytes(), bytes)?;
        Ok(())
    }

    /// Get a value from a column family
    fn get<T: DeserializeOwned>(&self, cf_name: &str, key: &str) -> StoreResult<Option<T>> {
        let db = self.db.read().map_err(|_| StoreError::LockPoisoned)?;
        let cf = db
            .cf_handle(cf_name)
            .ok_or_else(|| StoreError::ColumnFamilyNotFound(cf_name.to_string()))?;

        match db.get_cf(&cf, key.as_bytes())? {
            Some(bytes) => {
                let value = bincode::deserialize(&bytes)
                    .map_err(|e| StoreError::Deserialization(e.to_string()))?;
                Ok(Some(value))
            }
            None => Ok(None),
        }
    }

    /// Delete a value from a column family
    fn delete(&self, cf_name: &str, key: &str) -> StoreResult<()> {
        let db = self.db.read().map_err(|_| StoreError::LockPoisoned)?;
        let cf = db
            .cf_handle(cf_name)
            .ok_or_else(|| StoreError::ColumnFamilyNotFound(cf_name.to_string()))?;

        db.delete_cf(&cf, key.as_bytes())?;
        Ok(())
    }

    /// List all keys with a prefix in a column family
    fn list_keys(&self, cf_name: &str, prefix: &str) -> StoreResult<Vec<String>> {
        let db = self.db.read().map_err(|_| StoreError::LockPoisoned)?;
        let cf = db
            .cf_handle(cf_name)
            .ok_or_else(|| StoreError::ColumnFamilyNotFound(cf_name.to_string()))?;

        let mut keys = Vec::new();
        let iter = db.prefix_iterator_cf(&cf, prefix.as_bytes());

        for result in iter {
            let (key, _) = result?;
            if let Ok(key_str) = String::from_utf8(key.to_vec()) {
                if key_str.starts_with(prefix) {
                    keys.push(key_str);
                } else {
                    break; // Prefix no longer matches
                }
            }
        }

        Ok(keys)
    }

    // =========================================================================
    // Session operations
    // =========================================================================

    /// Store an ensemble session
    pub fn put_session(&self, session: &EnsembleSession) -> StoreResult<()> {
        let key = schema::keys::session(&session.id);
        self.put(schema::CF_SESSIONS, &key, session)
    }

    /// Get an ensemble session by ID
    pub fn get_session(&self, session_id: &str) -> StoreResult<Option<EnsembleSession>> {
        let key = schema::keys::session(session_id);
        self.get(schema::CF_SESSIONS, &key)
    }

    /// Get the most recent active session
    pub fn get_active_session(&self) -> StoreResult<Option<EnsembleSession>> {
        let keys = self.list_keys(schema::CF_SESSIONS, "sess:")?;

        let mut sessions: Vec<EnsembleSession> = keys
            .iter()
            .filter_map(|key| self.get::<EnsembleSession>(schema::CF_SESSIONS, key).ok()?)
            .filter(|s| s.active)
            .collect();

        // Sort by updated_at descending
        sessions.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(sessions.into_iter().next())
    }

    /// List all sessions
    pub fn list_sessions(&self) -> StoreResult<Vec<EnsembleSession>> {
        let keys = self.list_keys(schema::CF_SESSIONS, "sess:")?;

        let mut sessions: Vec<EnsembleSession> = keys
            .iter()
            .filter_map(|key| self.get(schema::CF_SESSIONS, key).ok()?)
            .collect();

        sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        Ok(sessions)
    }

    // =========================================================================
    // Task operations
    // =========================================================================

    /// Store an ensemble task
    pub fn put_task(&self, task: &EnsembleTask) -> StoreResult<()> {
        let key = schema::keys::task(&task.id);
        self.put(schema::CF_TASKS, &key, task)
    }

    /// Get a task by ID
    pub fn get_task(&self, task_id: &str) -> StoreResult<Option<EnsembleTask>> {
        let key = schema::keys::task(task_id);
        self.get(schema::CF_TASKS, &key)
    }

    /// Get tasks for a session
    pub fn get_session_tasks(&self, session_id: &str) -> StoreResult<Vec<EnsembleTask>> {
        let keys = self.list_keys(schema::CF_TASKS, "task:")?;

        let tasks: Vec<EnsembleTask> = keys
            .iter()
            .filter_map(|key| self.get::<EnsembleTask>(schema::CF_TASKS, key).ok()?)
            .filter(|t| t.session_id == session_id)
            .collect();

        Ok(tasks)
    }

    /// Get pending tasks for a session
    pub fn get_pending_tasks(&self, session_id: &str) -> StoreResult<Vec<EnsembleTask>> {
        Ok(self
            .get_session_tasks(session_id)?
            .into_iter()
            .filter(|t| matches!(t.status, TaskStatus::Pending | TaskStatus::InProgress))
            .collect())
    }

    // =========================================================================
    // Result operations
    // =========================================================================

    /// Store a model result
    pub fn put_result(&self, result: &ModelResult) -> StoreResult<()> {
        let key = schema::keys::result(&result.task_id, &result.model_id.to_string());
        self.put(schema::CF_RESULTS, &key, result)
    }

    /// Get a specific model's result for a task
    pub fn get_result(
        &self,
        task_id: &str,
        model_id: &ModelId,
    ) -> StoreResult<Option<ModelResult>> {
        let key = schema::keys::result(task_id, &model_id.to_string());
        self.get(schema::CF_RESULTS, &key)
    }

    /// Get all results for a task
    pub fn get_task_results(&self, task_id: &str) -> StoreResult<Vec<ModelResult>> {
        let prefix = format!("result:{}:", task_id);
        let keys = self.list_keys(schema::CF_RESULTS, &prefix)?;

        let results: Vec<ModelResult> = keys
            .iter()
            .filter_map(|key| self.get(schema::CF_RESULTS, key).ok()?)
            .collect();

        Ok(results)
    }

    // =========================================================================
    // Voting operations
    // =========================================================================

    /// Store a vote record
    pub fn put_vote(&self, vote: &VoteRecord) -> StoreResult<()> {
        let key = schema::keys::vote(&vote.task_id);
        self.put(schema::CF_VOTING, &key, vote)
    }

    /// Get a vote record for a task
    pub fn get_vote(&self, task_id: &str) -> StoreResult<Option<VoteRecord>> {
        let key = schema::keys::vote(task_id);
        self.get(schema::CF_VOTING, &key)
    }

    // =========================================================================
    // Context operations
    // =========================================================================

    /// Store shared context
    pub fn put_context(&self, context: &SharedContext) -> StoreResult<()> {
        let key = schema::keys::context(&context.session_id);
        self.put(schema::CF_CONTEXT, &key, context)
    }

    /// Get shared context for a session
    pub fn get_context(&self, session_id: &str) -> StoreResult<Option<SharedContext>> {
        let key = schema::keys::context(session_id);
        self.get(schema::CF_CONTEXT, &key)
    }

    /// Get or create context for a session
    pub fn get_or_create_context(&self, session_id: &str) -> StoreResult<SharedContext> {
        match self.get_context(session_id)? {
            Some(ctx) => Ok(ctx),
            None => {
                let ctx = SharedContext::new(session_id.to_string());
                self.put_context(&ctx)?;
                Ok(ctx)
            }
        }
    }

    // =========================================================================
    // Event operations (for replay)
    // =========================================================================

    /// Store an event (serialized as JSON for debuggability)
    pub fn put_event(
        &self,
        timestamp_nanos: i64,
        event_id: &str,
        event: &impl Serialize,
    ) -> StoreResult<()> {
        let key = schema::keys::event(timestamp_nanos, event_id);
        let bytes =
            serde_json::to_vec(event).map_err(|e| StoreError::Serialization(e.to_string()))?;

        let db = self.db.read().map_err(|_| StoreError::LockPoisoned)?;
        let cf = db
            .cf_handle(schema::CF_EVENTS)
            .ok_or_else(|| StoreError::ColumnFamilyNotFound(schema::CF_EVENTS.to_string()))?;

        db.put_cf(&cf, key.as_bytes(), bytes)?;
        Ok(())
    }

    /// Get events in a time range
    pub fn get_events_range<T: DeserializeOwned>(
        &self,
        start_nanos: i64,
        end_nanos: i64,
    ) -> StoreResult<Vec<(i64, T)>> {
        let db = self.db.read().map_err(|_| StoreError::LockPoisoned)?;
        let cf = db
            .cf_handle(schema::CF_EVENTS)
            .ok_or_else(|| StoreError::ColumnFamilyNotFound(schema::CF_EVENTS.to_string()))?;

        let start_key = schema::keys::event(start_nanos, "");
        let iter = db.iterator_cf(
            &cf,
            rocksdb::IteratorMode::From(start_key.as_bytes(), rocksdb::Direction::Forward),
        );

        let mut events = Vec::new();
        for result in iter {
            let (key, value) = result?;
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| StoreError::Deserialization(e.to_string()))?;

            if let Some(ts) = schema::keys::parse_event_timestamp(&key_str) {
                if ts > end_nanos {
                    break;
                }
                let event: T = serde_json::from_slice(&value)
                    .map_err(|e| StoreError::Deserialization(e.to_string()))?;
                events.push((ts, event));
            }
        }

        Ok(events)
    }

    /// Delete old events before a timestamp
    pub fn prune_events_before(&self, timestamp_nanos: i64) -> StoreResult<usize> {
        let db = self.db.read().map_err(|_| StoreError::LockPoisoned)?;
        let cf = db
            .cf_handle(schema::CF_EVENTS)
            .ok_or_else(|| StoreError::ColumnFamilyNotFound(schema::CF_EVENTS.to_string()))?;

        let start_key = schema::keys::event(0, "");
        let end_key = schema::keys::event(timestamp_nanos, "");

        // Collect keys to delete
        let mut keys_to_delete = Vec::new();
        let iter = db.iterator_cf(
            &cf,
            rocksdb::IteratorMode::From(start_key.as_bytes(), rocksdb::Direction::Forward),
        );

        for result in iter {
            let (key, _) = result?;
            let key_str = String::from_utf8(key.to_vec())
                .map_err(|e| StoreError::Deserialization(e.to_string()))?;

            if key_str >= end_key {
                break;
            }
            keys_to_delete.push(key.to_vec());
        }

        // Delete collected keys
        let count = keys_to_delete.len();
        for key in keys_to_delete {
            db.delete_cf(&cf, key)?;
        }

        Ok(count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn test_store() -> (StateStore, tempfile::TempDir) {
        let dir = tempdir().unwrap();
        let store = StateStore::open(dir.path().join("test.db")).unwrap();
        (store, dir)
    }

    #[test]
    fn test_session_crud() {
        let (store, _dir) = test_store();

        let session = EnsembleSession::new();
        let session_id = session.id.clone();

        store.put_session(&session).unwrap();
        let retrieved = store.get_session(&session_id).unwrap().unwrap();

        assert_eq!(retrieved.id, session_id);
        assert!(retrieved.active);
    }

    #[test]
    fn test_task_crud() {
        let (store, _dir) = test_store();

        let session = EnsembleSession::new();
        store.put_session(&session).unwrap();

        let task = EnsembleTask::new(session.id.clone(), "Test prompt".to_string(), true);
        let task_id = task.id.clone();

        store.put_task(&task).unwrap();
        let retrieved = store.get_task(&task_id).unwrap().unwrap();

        assert_eq!(retrieved.id, task_id);
        assert_eq!(retrieved.prompt, "Test prompt");
    }

    #[test]
    fn test_result_storage() {
        let (store, _dir) = test_store();

        let result = ModelResult::new(
            "task-1".to_string(),
            ModelId::Opus45,
            "Test response".to_string(),
            100,
            500,
        );

        store.put_result(&result).unwrap();
        let retrieved = store
            .get_result("task-1", &ModelId::Opus45)
            .unwrap()
            .unwrap();

        assert_eq!(retrieved.response, "Test response");
        assert_eq!(retrieved.tokens_used, 100);
    }

    #[test]
    fn test_context_versioning() {
        let (store, _dir) = test_store();

        let mut ctx = SharedContext::new("session-1".to_string());
        store.put_context(&ctx).unwrap();

        ctx.update_summary("New summary".to_string());
        store.put_context(&ctx).unwrap();

        let retrieved = store.get_context("session-1").unwrap().unwrap();
        assert_eq!(retrieved.version, 1);
        assert_eq!(retrieved.summary, "New summary");
    }

    #[test]
    fn test_get_or_create_context() {
        let (store, _dir) = test_store();

        // Should create new
        let ctx1 = store.get_or_create_context("session-1").unwrap();
        assert_eq!(ctx1.version, 0);

        // Should retrieve existing
        let ctx2 = store.get_or_create_context("session-1").unwrap();
        assert_eq!(ctx2.session_id, ctx1.session_id);
    }

    #[test]
    fn test_active_session() {
        let (store, _dir) = test_store();

        let session1 = EnsembleSession::new();
        store.put_session(&session1).unwrap();

        // Wait a bit to ensure different timestamps
        std::thread::sleep(std::time::Duration::from_millis(10));

        let session2 = EnsembleSession::new();
        store.put_session(&session2).unwrap();

        let active = store.get_active_session().unwrap().unwrap();
        assert_eq!(active.id, session2.id);
    }
}
