//! Batch (parallel) tool execution — fan out multiple tool calls concurrently.
//!
//! Allows a worker agent to execute several tool operations in a single LLM turn,
//! reducing the total turns spent on sequential reads or searches.

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::{Deserialize, Serialize};

use super::ToolError;

// Type alias for an async handler closure.
type BoxHandler = Arc<
    dyn Fn(serde_json::Value) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send>>
        + Send
        + Sync,
>;

/// A single tool invocation within a batch request.
#[derive(Deserialize)]
pub struct BatchCall {
    /// Registered tool name.
    pub tool: String,
    /// Tool-specific arguments (forwarded verbatim to the handler).
    pub args: serde_json::Value,
}

/// Arguments accepted by the `batch_execute` tool.
#[derive(Deserialize)]
pub struct BatchExecuteArgs {
    /// Tool calls to execute concurrently.
    pub calls: Vec<BatchCall>,
}

/// Per-call outcome returned by the tool.
#[derive(Serialize, Deserialize)]
pub struct BatchCallResult {
    /// Echo of the tool name.
    pub tool: String,
    /// Zero-based position in the input `calls` array.
    pub index: usize,
    /// Tool output on success.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
    /// Error message on failure.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Batch tool executor — runs multiple registered tools concurrently via
/// `tokio::task::JoinSet`.
///
/// Register handlers for specific tool names at construction time using
/// [`BatchExecute::register`]. The agent may then call `batch_execute` with
/// any combination of those tool names.
pub struct BatchExecute {
    handlers: HashMap<String, BoxHandler>,
}

impl BatchExecute {
    /// Create a new, empty `BatchExecute` tool.
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register an async handler for `name`.
    ///
    /// The handler receives the raw JSON `args` value from the batch call
    /// and must return `Result<String, ToolError>`.
    pub fn register<F, Fut>(mut self, name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<String, ToolError>> + Send + 'static,
    {
        self.handlers
            .insert(name.into(), Arc::new(move |args| Box::pin(handler(args))));
        self
    }
}

impl Default for BatchExecute {
    fn default() -> Self {
        Self::new()
    }
}

impl Tool for BatchExecute {
    const NAME: &'static str = "batch_execute";
    type Error = ToolError;
    type Args = BatchExecuteArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        let mut tools: Vec<&str> = self.handlers.keys().map(String::as_str).collect();
        tools.sort_unstable();
        ToolDefinition {
            name: "batch_execute".into(),
            description: format!(
                "Execute multiple tool calls concurrently in a single turn. \
                 Pass a `calls` array where each element is \
                 {{\"tool\": \"<name>\", \"args\": {{...}}}}. \
                 Results are returned as a JSON array in the same order as the input. \
                 Use instead of sequential tool calls when reads or searches are independent. \
                 Registered tools: {tools:?}."
            ),
            parameters: serde_json::json!({
                "type": "object",
                "required": ["calls"],
                "properties": {
                    "calls": {
                        "type": "array",
                        "minItems": 1,
                        "description": "Tool calls to run concurrently",
                        "items": {
                            "type": "object",
                            "required": ["tool", "args"],
                            "properties": {
                                "tool": {
                                    "type": "string",
                                    "description": "Registered tool name"
                                },
                                "args": {
                                    "type": "object",
                                    "description": "Tool-specific arguments"
                                }
                            }
                        }
                    }
                }
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        if args.calls.is_empty() {
            return Ok("[]".to_string());
        }

        let mut set = tokio::task::JoinSet::new();

        for (index, call) in args.calls.into_iter().enumerate() {
            let tool_name = call.tool.clone();
            let call_args = call.args;
            let handler = self.handlers.get(&call.tool).cloned();

            set.spawn(async move {
                let result = match handler {
                    None => Err(ToolError::Policy(format!(
                        "unknown tool: {tool_name:?} — not registered in this BatchExecute"
                    ))),
                    Some(h) => h(call_args).await,
                };
                BatchCallResult {
                    tool: tool_name,
                    index,
                    output: result.as_ref().ok().cloned(),
                    error: result.err().map(|e| e.to_string()),
                }
            });
        }

        let mut results: Vec<BatchCallResult> = Vec::new();
        while let Some(join_result) = set.join_next().await {
            match join_result {
                Ok(r) => results.push(r),
                Err(e) => {
                    return Err(ToolError::Policy(format!("batch task panicked: {e}")));
                }
            }
        }

        // Restore input order (JoinSet completes in arbitrary order).
        results.sort_by_key(|r| r.index);

        serde_json::to_string_pretty(&results)
            .map_err(|e| ToolError::Policy(format!("failed to serialize results: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_batch() -> BatchExecute {
        BatchExecute::new()
            .register("echo", |args| async move {
                Ok(args
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string())
            })
            .register("fail", |_| async move {
                Err(ToolError::Policy("deliberate failure".into()))
            })
    }

    #[tokio::test]
    async fn returns_empty_array_for_empty_calls() {
        let tool = make_batch();
        let result = tool.call(BatchExecuteArgs { calls: vec![] }).await.unwrap();
        assert_eq!(result, "[]");
    }

    #[tokio::test]
    async fn executes_registered_handlers_concurrently() {
        let tool = make_batch();
        let calls = vec![
            BatchCall {
                tool: "echo".into(),
                args: serde_json::json!({"message": "hello"}),
            },
            BatchCall {
                tool: "echo".into(),
                args: serde_json::json!({"message": "world"}),
            },
        ];
        let raw = tool.call(BatchExecuteArgs { calls }).await.unwrap();
        let results: Vec<BatchCallResult> = serde_json::from_str(&raw).unwrap();

        assert_eq!(results.len(), 2);
        // Results must be in input order.
        assert_eq!(results[0].index, 0);
        assert_eq!(results[0].output.as_deref(), Some("hello"));
        assert_eq!(results[1].index, 1);
        assert_eq!(results[1].output.as_deref(), Some("world"));
    }

    #[tokio::test]
    async fn captures_errors_without_propagating() {
        let tool = make_batch();
        let calls = vec![
            BatchCall {
                tool: "echo".into(),
                args: serde_json::json!({"message": "ok"}),
            },
            BatchCall {
                tool: "fail".into(),
                args: serde_json::json!({}),
            },
        ];
        // Should succeed even though one handler fails.
        let raw = tool.call(BatchExecuteArgs { calls }).await.unwrap();
        let results: Vec<BatchCallResult> = serde_json::from_str(&raw).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].output.is_some());
        assert!(results[1].error.is_some());
    }

    #[tokio::test]
    async fn unknown_tool_returns_error_result_not_err() {
        let tool = make_batch();
        let calls = vec![BatchCall {
            tool: "no_such_tool".into(),
            args: serde_json::json!({}),
        }];
        let raw = tool.call(BatchExecuteArgs { calls }).await.unwrap();
        let results: Vec<BatchCallResult> = serde_json::from_str(&raw).unwrap();

        assert!(results[0].error.is_some());
        assert!(results[0].error.as_ref().unwrap().contains("unknown tool"));
    }
}
