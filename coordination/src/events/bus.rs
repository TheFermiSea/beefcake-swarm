//! Event bus for ensemble coordination
//!
//! Provides pub/sub messaging using Tokio broadcast channels with
//! optional persistence to RocksDB for event replay.

use std::sync::Arc;
use tokio::sync::broadcast;
use tracing::{debug, warn};

use super::types::EnsembleEvent;
use crate::state::SharedStateStore;

/// Channel capacity for broadcast
const CHANNEL_CAPACITY: usize = 256;

/// Error type for event bus operations
#[derive(Debug, thiserror::Error)]
pub enum EventBusError {
    #[error("Failed to send event: {0}")]
    SendFailed(String),

    #[error("Failed to persist event: {0}")]
    PersistFailed(String),

    #[error("Channel closed")]
    ChannelClosed,
}

/// Result type for event bus operations
pub type EventBusResult<T> = Result<T, EventBusError>;

/// Shared reference to EventBus
pub type SharedEventBus = Arc<EventBus>;

/// Event bus with broadcast channels and optional persistence
pub struct EventBus {
    /// Broadcast sender for publishing events
    sender: broadcast::Sender<EnsembleEvent>,

    /// Optional state store for event persistence
    store: Option<SharedStateStore>,

    /// Whether to persist events
    persist_events: bool,
}

impl EventBus {
    /// Create a new event bus without persistence
    pub fn new() -> Self {
        let (sender, _) = broadcast::channel(CHANNEL_CAPACITY);
        Self {
            sender,
            store: None,
            persist_events: false,
        }
    }

    /// Create an event bus with persistence enabled
    pub fn with_persistence(store: SharedStateStore) -> Self {
        let (sender, _) = broadcast::channel(CHANNEL_CAPACITY);
        Self {
            sender,
            store: Some(store),
            persist_events: true,
        }
    }

    /// Create a shared reference to this event bus
    pub fn shared(self) -> SharedEventBus {
        Arc::new(self)
    }

    /// Enable or disable event persistence
    pub fn set_persist_events(&mut self, persist: bool) {
        self.persist_events = persist;
    }

    /// Publish an event to all subscribers
    pub fn publish(&self, event: EnsembleEvent) -> EventBusResult<()> {
        let event_type = event.event_type();
        let timestamp = event.timestamp();

        // Persist if enabled
        if self.persist_events {
            if let Some(store) = &self.store {
                let event_id = EnsembleEvent::new_id();
                let timestamp_nanos = timestamp.timestamp_nanos_opt().unwrap_or(0);

                if let Err(e) = store.put_event(timestamp_nanos, &event_id, &event) {
                    warn!(event_type, "Failed to persist event: {}", e);
                    return Err(EventBusError::PersistFailed(e.to_string()));
                }
                debug!(event_type, event_id, "Event persisted");
            }
        }

        // Broadcast to subscribers (ignore if no receivers)
        match self.sender.send(event) {
            Ok(count) => {
                debug!(event_type, receivers = count, "Event published");
                Ok(())
            }
            Err(_) => {
                // No receivers is OK - we still persisted
                debug!(event_type, "Event published (no receivers)");
                Ok(())
            }
        }
    }

    /// Subscribe to receive events
    pub fn subscribe(&self) -> broadcast::Receiver<EnsembleEvent> {
        self.sender.subscribe()
    }

    /// Get the number of current subscribers
    pub fn subscriber_count(&self) -> usize {
        self.sender.receiver_count()
    }

    /// Check if the bus has any subscribers
    pub fn has_subscribers(&self) -> bool {
        self.sender.receiver_count() > 0
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Event filter for selective subscription
pub struct EventFilter {
    /// Filter by session ID
    pub session_id: Option<String>,
    /// Filter by task ID
    pub task_id: Option<String>,
    /// Filter by event types
    pub event_types: Option<Vec<String>>,
}

impl EventFilter {
    /// Create a new empty filter (matches all events)
    pub fn new() -> Self {
        Self {
            session_id: None,
            task_id: None,
            event_types: None,
        }
    }

    /// Filter by session ID
    pub fn session(mut self, session_id: &str) -> Self {
        self.session_id = Some(session_id.to_string());
        self
    }

    /// Filter by task ID
    pub fn task(mut self, task_id: &str) -> Self {
        self.task_id = Some(task_id.to_string());
        self
    }

    /// Filter by event types
    pub fn types(mut self, event_types: Vec<&str>) -> Self {
        self.event_types = Some(event_types.into_iter().map(String::from).collect());
        self
    }

    /// Check if an event matches this filter
    pub fn matches(&self, event: &EnsembleEvent) -> bool {
        // Check session filter
        if let Some(ref sid) = self.session_id {
            if let Some(event_sid) = event.session_id() {
                if event_sid != sid {
                    return false;
                }
            }
        }

        // Check task filter
        if let Some(ref tid) = self.task_id {
            if let Some(event_tid) = event.task_id() {
                if event_tid != tid {
                    return false;
                }
            }
        }

        // Check event type filter
        if let Some(ref types) = self.event_types {
            if !types.contains(&event.event_type().to_string()) {
                return false;
            }
        }

        true
    }
}

impl Default for EventFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// Filtered event receiver that only yields matching events
pub struct FilteredReceiver {
    receiver: broadcast::Receiver<EnsembleEvent>,
    filter: EventFilter,
}

impl FilteredReceiver {
    /// Create a new filtered receiver
    pub fn new(receiver: broadcast::Receiver<EnsembleEvent>, filter: EventFilter) -> Self {
        Self { receiver, filter }
    }

    /// Receive the next matching event
    pub async fn recv(&mut self) -> Result<EnsembleEvent, broadcast::error::RecvError> {
        loop {
            let event = self.receiver.recv().await?;
            if self.filter.matches(&event) {
                return Ok(event);
            }
        }
    }
}

/// Extension trait for subscribing with filters
pub trait EventBusExt {
    /// Subscribe with a filter
    fn subscribe_filtered(&self, filter: EventFilter) -> FilteredReceiver;
}

impl EventBusExt for EventBus {
    fn subscribe_filtered(&self, filter: EventFilter) -> FilteredReceiver {
        FilteredReceiver::new(self.subscribe(), filter)
    }
}

impl EventBusExt for SharedEventBus {
    fn subscribe_filtered(&self, filter: EventFilter) -> FilteredReceiver {
        FilteredReceiver::new(self.subscribe(), filter)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ModelId;
    use chrono::Utc;

    #[tokio::test]
    async fn test_publish_subscribe() {
        let bus = EventBus::new();
        let mut receiver = bus.subscribe();

        let event = EnsembleEvent::SessionCreated {
            session_id: "test-session".to_string(),
            harness_session_id: None,
            timestamp: Utc::now(),
        };

        bus.publish(event.clone()).unwrap();

        let received = receiver.recv().await.unwrap();
        assert_eq!(received.event_type(), "session_created");
    }

    #[tokio::test]
    async fn test_multiple_subscribers() {
        let bus = EventBus::new().shared();
        let mut rx1 = bus.subscribe();
        let mut rx2 = bus.subscribe();

        assert_eq!(bus.subscriber_count(), 2);

        let event = EnsembleEvent::ModelLoaded {
            model_id: ModelId::Behemoth,
            load_time_ms: 1000,
            timestamp: Utc::now(),
        };

        bus.publish(event).unwrap();

        let e1 = rx1.recv().await.unwrap();
        let e2 = rx2.recv().await.unwrap();

        assert_eq!(e1.event_type(), e2.event_type());
    }

    #[test]
    fn test_event_filter() {
        let filter = EventFilter::new()
            .session("session-1")
            .types(vec!["task_created", "task_completed"]);

        let matching_event = EnsembleEvent::TaskCreated {
            task_id: "task-1".to_string(),
            session_id: "session-1".to_string(),
            prompt_preview: "test".to_string(),
            require_consensus: true,
            timestamp: Utc::now(),
        };

        let non_matching_session = EnsembleEvent::TaskCreated {
            task_id: "task-2".to_string(),
            session_id: "session-2".to_string(),
            prompt_preview: "test".to_string(),
            require_consensus: true,
            timestamp: Utc::now(),
        };

        let non_matching_type = EnsembleEvent::ModelLoaded {
            model_id: ModelId::Behemoth,
            load_time_ms: 1000,
            timestamp: Utc::now(),
        };

        assert!(filter.matches(&matching_event));
        assert!(!filter.matches(&non_matching_session));
        assert!(!filter.matches(&non_matching_type));
    }

    #[tokio::test]
    async fn test_filtered_receiver() {
        let bus = EventBus::new();
        let filter = EventFilter::new().task("target-task");
        let mut filtered = bus.subscribe_filtered(filter);

        // Spawn publisher
        let bus_clone = bus;
        tokio::spawn(async move {
            // Publish non-matching event
            bus_clone
                .publish(EnsembleEvent::TaskCreated {
                    task_id: "other-task".to_string(),
                    session_id: "session-1".to_string(),
                    prompt_preview: "test".to_string(),
                    require_consensus: true,
                    timestamp: Utc::now(),
                })
                .unwrap();

            // Publish matching event
            bus_clone
                .publish(EnsembleEvent::ResultSubmitted {
                    task_id: "target-task".to_string(),
                    model_id: ModelId::Behemoth,
                    confidence: 0.9,
                    tokens_used: 100,
                    latency_ms: 500,
                    timestamp: Utc::now(),
                })
                .unwrap();
        });

        // Should receive only the matching event
        let event = filtered.recv().await.unwrap();
        assert_eq!(event.task_id(), Some("target-task"));
    }
}
