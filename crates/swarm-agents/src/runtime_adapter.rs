//! Rig PromptHook adapter for tool-event visibility, turn accounting, and budget control.
//!
//! Wraps each `agent.prompt()` call to intercept tool invocations, track LLM turns,
//! emit structured traces, and enforce deterministic timeout/cancellation semantics.
//!
//! # Usage
//!
//! ```ignore
//! let adapter = RuntimeAdapter::new(AdapterConfig {
//!     agent_name: "rust_coder".into(),
//!     max_tool_calls: Some(50),
//!     deadline: Some(Instant::now() + Duration::from_secs(1800)),
//!     ..Default::default()
//! });
//!
//! let response = agent
//!     .prompt(&task_prompt)
//!     .with_hook(adapter.clone())
//!     .await?;
//!
//! let report = adapter.report();
//! info!(turns = report.turn_count, tools = report.total_tool_calls, "Agent finished");
//! ```

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rig::agent::{HookAction, PromptHook, ToolCallHookAction};
use rig::completion::CompletionModel;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Configuration for the runtime adapter.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Agent name for structured traces.
    pub agent_name: String,
    /// Maximum tool calls before terminating the agent loop.
    pub max_tool_calls: Option<usize>,
    /// Hard deadline (wall-clock) for the entire prompt request.
    pub deadline: Option<Instant>,
    /// Maximum characters to capture in args/result previews.
    pub preview_len: usize,
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self {
            agent_name: "unknown".to_string(),
            max_tool_calls: None,
            deadline: None,
            preview_len: 200,
        }
    }
}

/// Outcome of a tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolOutcome {
    Success,
    Error,
}

/// A recorded tool call event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolEvent {
    pub tool_name: String,
    pub args_preview: String,
    pub result_preview: String,
    pub duration_ms: u64,
    pub outcome: ToolOutcome,
}

/// Summary report extracted from the adapter after a prompt completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterReport {
    pub agent_name: String,
    pub tool_events: Vec<ToolEvent>,
    pub turn_count: usize,
    pub total_tool_calls: usize,
    pub total_tool_time_ms: u64,
    pub wall_time_ms: u64,
    pub terminated_early: bool,
    pub termination_reason: Option<String>,
}

/// In-flight tool call tracking.
struct InFlight {
    args_preview: String,
    started_at: Instant,
}

/// Shared mutable state for the adapter.
struct AdapterState {
    tool_events: Vec<ToolEvent>,
    turn_count: usize,
    in_flight: HashMap<String, InFlight>,
    total_tool_time: Duration,
    started_at: Instant,
    terminated_early: bool,
    termination_reason: Option<String>,
}

/// Rig [`PromptHook`] implementation for tool-event visibility and budget control.
///
/// Attach to any agent call via `.with_hook(adapter.clone())`. After the call
/// completes, call [`RuntimeAdapter::report()`] to extract the event log.
#[derive(Clone)]
pub struct RuntimeAdapter {
    state: Arc<Mutex<AdapterState>>,
    config: Arc<AdapterConfig>,
}

impl RuntimeAdapter {
    /// Create a new adapter with the given configuration.
    pub fn new(config: AdapterConfig) -> Self {
        Self {
            state: Arc::new(Mutex::new(AdapterState {
                tool_events: Vec::new(),
                turn_count: 0,
                in_flight: HashMap::new(),
                total_tool_time: Duration::ZERO,
                started_at: Instant::now(),
                terminated_early: false,
                termination_reason: None,
            })),
            config: Arc::new(config),
        }
    }

    /// Extract the adapter report after a prompt completes.
    ///
    /// Returns an error if the internal mutex is poisoned (another thread
    /// panicked while holding the lock).
    pub fn report(&self) -> Result<AdapterReport, String> {
        let state = self
            .state
            .lock()
            .map_err(|e| format!("RuntimeAdapter mutex poisoned: {e}"))?;
        Ok(AdapterReport {
            agent_name: self.config.agent_name.clone(),
            tool_events: state.tool_events.clone(),
            turn_count: state.turn_count,
            total_tool_calls: state.tool_events.len(),
            total_tool_time_ms: state.total_tool_time.as_millis() as u64,
            wall_time_ms: state.started_at.elapsed().as_millis() as u64,
            terminated_early: state.terminated_early,
            termination_reason: state.termination_reason.clone(),
        })
    }

    fn truncate(s: &str, max_len: usize) -> String {
        if s.len() <= max_len {
            s.to_string()
        } else {
            format!("{}...", &s[..max_len])
        }
    }
}

impl<M: CompletionModel> PromptHook<M> for RuntimeAdapter {
    fn on_completion_call(
        &self,
        _prompt: &rig::completion::message::Message,
        _history: &[rig::completion::message::Message],
    ) -> impl std::future::Future<Output = HookAction> + Send {
        let state = self.state.clone();
        let agent_name = self.config.agent_name.clone();
        async move {
            let turn = match state.lock() {
                Ok(mut s) => {
                    s.turn_count += 1;
                    s.turn_count
                }
                Err(e) => {
                    warn!(agent = %agent_name, error = %e, "Adapter state poisoned in on_completion_call");
                    0
                }
            };
            info!(agent = %agent_name, turn, "LLM turn started");
            HookAction::cont()
        }
    }

    fn on_tool_call(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        internal_call_id: &str,
        args: &str,
    ) -> impl std::future::Future<Output = ToolCallHookAction> + Send {
        let state = self.state.clone();
        let config = self.config.clone();
        let tool_name = tool_name.to_string();
        let internal_call_id = internal_call_id.to_string();
        let args_preview = Self::truncate(args, config.preview_len);

        async move {
            let mut s = match state.lock() {
                Ok(guard) => guard,
                Err(e) => {
                    warn!(
                        agent = %config.agent_name,
                        tool = %tool_name,
                        error = %e,
                        "Adapter state poisoned in on_tool_call — terminating"
                    );
                    return ToolCallHookAction::terminate(
                        "Runtime adapter: internal state corrupted",
                    );
                }
            };

            // Check deadline
            if let Some(deadline) = config.deadline {
                if Instant::now() >= deadline {
                    s.terminated_early = true;
                    s.termination_reason = Some("deadline exceeded".to_string());
                    warn!(
                        agent = %config.agent_name,
                        tool = %tool_name,
                        "Tool call rejected: deadline exceeded"
                    );
                    return ToolCallHookAction::terminate("Runtime adapter: deadline exceeded");
                }
            }

            // Check tool call budget
            if let Some(max) = config.max_tool_calls {
                if s.tool_events.len() >= max {
                    s.terminated_early = true;
                    let reason = format!("max tool calls ({max}) exceeded");
                    s.termination_reason = Some(reason.clone());
                    warn!(
                        agent = %config.agent_name,
                        tool = %tool_name,
                        max_tool_calls = max,
                        "Tool call rejected: budget exceeded"
                    );
                    return ToolCallHookAction::terminate(format!("Runtime adapter: {reason}"));
                }
            }

            // Track in-flight
            s.in_flight.insert(
                internal_call_id.clone(),
                InFlight {
                    args_preview: args_preview.clone(),
                    started_at: Instant::now(),
                },
            );

            debug!(
                agent = %config.agent_name,
                tool = %tool_name,
                call_id = %internal_call_id,
                args = %args_preview,
                "Tool call started"
            );

            ToolCallHookAction::cont()
        }
    }

    fn on_tool_result(
        &self,
        tool_name: &str,
        _tool_call_id: Option<String>,
        internal_call_id: &str,
        _args: &str,
        result: &str,
    ) -> impl std::future::Future<Output = HookAction> + Send {
        let state = self.state.clone();
        let config = self.config.clone();
        let tool_name = tool_name.to_string();
        let internal_call_id = internal_call_id.to_string();
        let result_preview = Self::truncate(result, config.preview_len);
        let result_len = result.len();
        let is_error = result.starts_with("Error") || result.starts_with("error");

        async move {
            let mut s = match state.lock() {
                Ok(guard) => guard,
                Err(e) => {
                    warn!(
                        agent = %config.agent_name,
                        tool = %tool_name,
                        error = %e,
                        "Adapter state poisoned in on_tool_result — skipping recording"
                    );
                    return HookAction::cont();
                }
            };

            let (args_preview, duration) =
                if let Some(flight) = s.in_flight.remove(&internal_call_id) {
                    (flight.args_preview, flight.started_at.elapsed())
                } else {
                    (String::new(), Duration::ZERO)
                };

            s.total_tool_time += duration;

            let outcome = if is_error {
                ToolOutcome::Error
            } else {
                ToolOutcome::Success
            };

            let duration_ms = duration.as_millis() as u64;

            info!(
                agent = %config.agent_name,
                tool = %tool_name,
                duration_ms,
                outcome = ?outcome,
                result_len,
                "Tool call completed"
            );

            s.tool_events.push(ToolEvent {
                tool_name,
                args_preview,
                result_preview,
                duration_ms,
                outcome,
            });

            HookAction::cont()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_default_report() {
        let adapter = RuntimeAdapter::new(AdapterConfig::default());
        let report = adapter.report().unwrap();
        assert_eq!(report.agent_name, "unknown");
        assert_eq!(report.turn_count, 0);
        assert_eq!(report.total_tool_calls, 0);
        assert!(!report.terminated_early);
        assert!(report.termination_reason.is_none());
    }

    #[test]
    fn test_adapter_config_custom() {
        let config = AdapterConfig {
            agent_name: "strand-14B".into(),
            max_tool_calls: Some(50),
            deadline: Some(Instant::now() + Duration::from_secs(1800)),
            preview_len: 100,
        };
        let adapter = RuntimeAdapter::new(config);
        let report = adapter.report().unwrap();
        assert_eq!(report.agent_name, "strand-14B");
    }

    #[test]
    fn test_truncate() {
        assert_eq!(RuntimeAdapter::truncate("hello", 10), "hello");
        assert_eq!(RuntimeAdapter::truncate("hello world", 5), "hello...");
        assert_eq!(RuntimeAdapter::truncate("", 5), "");
    }

    #[tokio::test]
    async fn test_turn_counting() {
        let adapter = RuntimeAdapter::new(AdapterConfig {
            agent_name: "test".into(),
            ..Default::default()
        });

        // Simulate 3 LLM turns via on_completion_call
        let msg: rig::completion::message::Message = "test".into();
        for _ in 0..3 {
            let action = <RuntimeAdapter as PromptHook<
                rig::providers::openai::completion::CompletionModel,
            >>::on_completion_call(&adapter, &msg, &[])
            .await;
            assert_eq!(action, HookAction::cont());
        }

        let report = adapter.report().unwrap();
        assert_eq!(report.turn_count, 3);
    }

    #[tokio::test]
    async fn test_tool_event_recording() {
        let adapter = RuntimeAdapter::new(AdapterConfig {
            agent_name: "test".into(),
            ..Default::default()
        });

        // on_tool_call
        let action = <RuntimeAdapter as PromptHook<
            rig::providers::openai::completion::CompletionModel,
        >>::on_tool_call(
            &adapter,
            "read_file",
            None,
            "call-1",
            r#"{"path": "src/main.rs"}"#,
        )
        .await;
        assert_eq!(action, ToolCallHookAction::cont());

        // on_tool_result
        let action = <RuntimeAdapter as PromptHook<
            rig::providers::openai::completion::CompletionModel,
        >>::on_tool_result(
            &adapter,
            "read_file",
            None,
            "call-1",
            r#"{"path": "src/main.rs"}"#,
            "fn main() { println!(\"hello\"); }",
        )
        .await;
        assert_eq!(action, HookAction::cont());

        let report = adapter.report().unwrap();
        assert_eq!(report.total_tool_calls, 1);
        assert_eq!(report.tool_events[0].tool_name, "read_file");
        assert_eq!(report.tool_events[0].outcome, ToolOutcome::Success);
    }

    #[tokio::test]
    async fn test_tool_budget_enforcement() {
        let adapter = RuntimeAdapter::new(AdapterConfig {
            agent_name: "test".into(),
            max_tool_calls: Some(2),
            ..Default::default()
        });

        // Record 2 completed tool calls
        for i in 0..2 {
            let call_id = format!("call-{i}");
            let _ = <RuntimeAdapter as PromptHook<
                rig::providers::openai::completion::CompletionModel,
            >>::on_tool_call(&adapter, "write_file", None, &call_id, "{}")
            .await;
            let _ = <RuntimeAdapter as PromptHook<
                rig::providers::openai::completion::CompletionModel,
            >>::on_tool_result(
                &adapter, "write_file", None, &call_id, "{}", "ok"
            )
            .await;
        }

        // 3rd call should be terminated
        let action = <RuntimeAdapter as PromptHook<
            rig::providers::openai::completion::CompletionModel,
        >>::on_tool_call(&adapter, "write_file", None, "call-2", "{}")
        .await;
        assert!(
            matches!(action, ToolCallHookAction::Terminate { .. }),
            "Expected Terminate, got {action:?}"
        );

        let report = adapter.report().unwrap();
        assert!(report.terminated_early);
        assert!(report
            .termination_reason
            .as_ref()
            .unwrap()
            .contains("max tool calls"));
    }

    #[tokio::test]
    async fn test_deadline_enforcement() {
        // Deadline already passed
        let adapter = RuntimeAdapter::new(AdapterConfig {
            agent_name: "test".into(),
            deadline: Some(Instant::now() - Duration::from_secs(1)),
            ..Default::default()
        });

        let action = <RuntimeAdapter as PromptHook<
            rig::providers::openai::completion::CompletionModel,
        >>::on_tool_call(&adapter, "run_command", None, "call-0", "{}")
        .await;

        assert!(
            matches!(action, ToolCallHookAction::Terminate { .. }),
            "Expected Terminate for expired deadline, got {action:?}"
        );

        let report = adapter.report().unwrap();
        assert!(report.terminated_early);
        assert!(report
            .termination_reason
            .as_ref()
            .unwrap()
            .contains("deadline"));
    }

    #[tokio::test]
    async fn test_error_outcome_detection() {
        let adapter = RuntimeAdapter::new(AdapterConfig {
            agent_name: "test".into(),
            ..Default::default()
        });

        let _ = <RuntimeAdapter as PromptHook<
            rig::providers::openai::completion::CompletionModel,
        >>::on_tool_call(&adapter, "run_command", None, "call-0", "{}")
        .await;
        let _ = <RuntimeAdapter as PromptHook<
            rig::providers::openai::completion::CompletionModel,
        >>::on_tool_result(
            &adapter,
            "run_command",
            None,
            "call-0",
            "{}",
            "Error: command failed with exit code 1",
        )
        .await;

        let report = adapter.report().unwrap();
        assert_eq!(report.tool_events[0].outcome, ToolOutcome::Error);
    }

    #[test]
    fn test_report_serialization() {
        let report = AdapterReport {
            agent_name: "strand-14B".into(),
            tool_events: vec![ToolEvent {
                tool_name: "read_file".into(),
                args_preview: r#"{"path":"src/main.rs"}"#.into(),
                result_preview: "fn main() {}".into(),
                duration_ms: 42,
                outcome: ToolOutcome::Success,
            }],
            turn_count: 3,
            total_tool_calls: 1,
            total_tool_time_ms: 42,
            wall_time_ms: 5000,
            terminated_early: false,
            termination_reason: None,
        };

        let json = serde_json::to_string(&report).unwrap();
        let parsed: AdapterReport = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.agent_name, "strand-14B");
        assert_eq!(parsed.tool_events.len(), 1);
        assert_eq!(parsed.turn_count, 3);
        assert!(!parsed.terminated_early);
    }

    #[test]
    fn test_adapter_is_clone_send_sync() {
        fn assert_clone_send_sync<T: Clone + Send + Sync>() {}
        assert_clone_send_sync::<RuntimeAdapter>();
    }
}
