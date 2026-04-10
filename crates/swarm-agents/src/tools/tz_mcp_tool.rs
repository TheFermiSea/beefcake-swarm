//! Rig tool for querying TensorZero observability data via its MCP server.
//!
//! Allows the cloud manager to query TZ Autopilot tools (experiment stats,
//! model comparisons, function performance) during task planning. The tool
//! implements the MCP Streamable HTTP transport protocol, establishing a
//! session and forwarding tool calls to the TZ MCP endpoint.

use rig::completion::ToolDefinition;
use rig::tool::Tool;
use serde::Deserialize;
use tracing::{debug, warn};

use super::ToolError;

/// Default TZ MCP endpoint when `SWARM_TENSORZERO_MCP_URL` is not set.
const DEFAULT_MCP_URL: &str = "http://localhost:3000/mcp";

/// MCP protocol version we advertise during initialization.
const MCP_PROTOCOL_VERSION: &str = "2025-03-26";

#[derive(Deserialize)]
pub struct QueryTensorZeroArgs {
    /// The TZ MCP tool to call (e.g. "list_experiments", "get_function_stats").
    pub tool_name: String,
    /// JSON-encoded arguments for the tool call.
    pub arguments: String,
}

/// Query TensorZero observability data through its MCP server.
pub struct QueryTensorZeroTool {
    mcp_url: String,
    client: reqwest::Client,
}

impl Default for QueryTensorZeroTool {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryTensorZeroTool {
    pub fn new() -> Self {
        let mcp_url = std::env::var("SWARM_TENSORZERO_MCP_URL")
            .unwrap_or_else(|_| DEFAULT_MCP_URL.to_string());
        Self {
            mcp_url,
            client: reqwest::Client::new(),
        }
    }

    /// Derive the MCP client name from `SWARM_REPO_ID` env var, or from the
    /// current working directory name, falling back to `"swarm"`.
    fn client_name() -> String {
        std::env::var("SWARM_REPO_ID").ok().unwrap_or_else(|| {
            std::env::current_dir()
                .ok()
                .and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()))
                .unwrap_or_else(|| "swarm".to_string())
        })
    }

    /// Initialize an MCP session and return the session ID.
    async fn init_session(&self) -> Result<String, ToolError> {
        let init_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {
                    "name": Self::client_name(),
                    "version": "1.0.0"
                }
            }
        });

        let resp = self
            .client
            .post(&self.mcp_url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .json(&init_body)
            .send()
            .await
            .map_err(|e| ToolError::External(format!("MCP init request failed: {e}")))?;

        let session_id = resp
            .headers()
            .get("mcp-session-id")
            .and_then(|v| v.to_str().ok())
            .map(String::from)
            .ok_or_else(|| {
                ToolError::External("MCP server did not return Mcp-Session-Id header".into())
            })?;

        // Consume the init response body (may be JSON or SSE).
        let body = resp
            .text()
            .await
            .map_err(|e| ToolError::External(format!("Failed to read init response: {e}")))?;
        debug!(session_id = %session_id, body_len = body.len(), "MCP session initialized");

        // Send the initialized notification (fire-and-forget).
        let notif = serde_json::json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized"
        });
        let _ = self
            .client
            .post(&self.mcp_url)
            .header("Content-Type", "application/json")
            .header("Mcp-Session-Id", &session_id)
            .json(&notif)
            .send()
            .await;

        Ok(session_id)
    }

    /// Call a tool on the MCP server and return the result text.
    async fn call_tool(
        &self,
        session_id: &str,
        tool_name: &str,
        arguments: serde_json::Value,
    ) -> Result<String, ToolError> {
        let call_body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            }
        });

        let resp = self
            .client
            .post(&self.mcp_url)
            .header("Accept", "application/json, text/event-stream")
            .header("Content-Type", "application/json")
            .header("Mcp-Session-Id", session_id)
            .json(&call_body)
            .send()
            .await
            .map_err(|e| ToolError::External(format!("MCP tool call failed: {e}")))?;

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let body = resp
            .text()
            .await
            .map_err(|e| ToolError::External(format!("Failed to read tool response: {e}")))?;

        // Parse response — may be direct JSON or SSE stream.
        if content_type.contains("text/event-stream") {
            parse_sse_result(&body)
        } else {
            parse_json_result(&body)
        }
    }
}

/// Extract the result from a JSON-RPC response body.
fn parse_json_result(body: &str) -> Result<String, ToolError> {
    let parsed: serde_json::Value = serde_json::from_str(body)
        .map_err(|e| ToolError::External(format!("Invalid JSON response: {e}")))?;

    if let Some(error) = parsed.get("error") {
        return Err(ToolError::External(format!(
            "MCP error: {}",
            serde_json::to_string(error).unwrap_or_default()
        )));
    }

    // result.content[0].text is the standard MCP tool result shape.
    if let Some(text) = parsed
        .pointer("/result/content/0/text")
        .and_then(|v| v.as_str())
    {
        return Ok(text.to_string());
    }

    // Fallback: return the entire result object as pretty JSON.
    if let Some(result) = parsed.get("result") {
        Ok(serde_json::to_string_pretty(result).unwrap_or_default())
    } else {
        Ok(body.to_string())
    }
}

/// Extract the result from an SSE (Server-Sent Events) response body.
/// Looks for `data: {...}` lines containing JSON-RPC responses.
fn parse_sse_result(body: &str) -> Result<String, ToolError> {
    for line in body.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data: ") {
            // Try to parse as JSON-RPC result.
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                if parsed.get("result").is_some() || parsed.get("error").is_some() {
                    return parse_json_result(data);
                }
            }
        }
    }
    // No JSON-RPC result found in SSE stream — return raw body.
    Err(ToolError::External(format!(
        "No JSON-RPC result in SSE response: {}",
        body.chars().take(500).collect::<String>()
    )))
}

impl Tool for QueryTensorZeroTool {
    const NAME: &'static str = "query_tensorzero";
    type Error = ToolError;
    type Args = QueryTensorZeroArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "query_tensorzero".into(),
            description: "Query TensorZero observability data via its MCP server. \
                          Use to check experiment results, model performance comparisons, \
                          function-level stats, and inference metrics. Available tools include \
                          list_experiments, get_experiment_stats, list_functions, \
                          get_function_performance, list_models, compare_models, \
                          and more. Pass the TZ MCP tool name and its JSON arguments."
                .into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "description": "The TensorZero MCP tool to call (e.g. 'list_experiments', 'get_function_stats')"
                    },
                    "arguments": {
                        "type": "string",
                        "description": "JSON-encoded arguments for the tool call (e.g. '{\"function_name\": \"code_fixing\"}')"
                    }
                },
                "required": ["tool_name", "arguments"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        // Parse the arguments JSON string.
        let arguments: serde_json::Value = serde_json::from_str(&args.arguments).map_err(|e| {
            ToolError::External(format!(
                "Invalid JSON in arguments: {e}. Expected a JSON object string."
            ))
        })?;

        // Initialize MCP session.
        let session_id = match self.init_session().await {
            Ok(id) => id,
            Err(err) => {
                warn!(
                    error = %err,
                    "TensorZero MCP session init failed; degrading gracefully"
                );
                return Ok(format!(
                    "TensorZero MCP is unavailable \u{2014} cannot query observability data. Error: {err}"
                ));
            }
        };

        // Call the requested tool.
        match self
            .call_tool(&session_id, &args.tool_name, arguments)
            .await
        {
            Ok(result) => {
                debug!(
                    tool = %args.tool_name,
                    result_len = result.len(),
                    "TZ MCP tool call succeeded"
                );
                Ok(format!("## TensorZero: {}\n\n{}", args.tool_name, result))
            }
            Err(err) => {
                warn!(
                    tool = %args.tool_name,
                    error = %err,
                    "TZ MCP tool call failed; degrading gracefully"
                );
                Ok(format!(
                    "TensorZero tool '{}' failed \u{2014} proceeding without TZ data. Error: {err}",
                    args.tool_name
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_json_result_extracts_content_text() {
        let body = r#"{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"experiment data here"}]}}"#;
        let result = parse_json_result(body).unwrap();
        assert_eq!(result, "experiment data here");
    }

    #[test]
    fn parse_json_result_returns_error_on_rpc_error() {
        let body =
            r#"{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found"}}"#;
        let result = parse_json_result(body);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Method not found"), "got: {err}");
    }

    #[test]
    fn parse_json_result_falls_back_to_result_object() {
        let body = r#"{"jsonrpc":"2.0","id":2,"result":{"experiments":["a","b"]}}"#;
        let result = parse_json_result(body).unwrap();
        assert!(result.contains("experiments"), "got: {result}");
    }

    #[test]
    fn parse_sse_result_extracts_from_data_lines() {
        let body = "event: message\ndata: {\"jsonrpc\":\"2.0\",\"id\":2,\"result\":{\"content\":[{\"type\":\"text\",\"text\":\"sse result\"}]}}\n\n";
        let result = parse_sse_result(body).unwrap();
        assert_eq!(result, "sse result");
    }

    #[test]
    fn parse_sse_result_errors_on_no_result() {
        let body = "event: ping\ndata: {}\n\n";
        let result = parse_sse_result(body);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn tool_degrades_gracefully_on_connection_failure() {
        // Point at a port nothing is listening on.
        std::env::set_var("SWARM_TENSORZERO_MCP_URL", "http://127.0.0.1:1/mcp");
        let tool = QueryTensorZeroTool::new();
        std::env::remove_var("SWARM_TENSORZERO_MCP_URL");

        let args = QueryTensorZeroArgs {
            tool_name: "list_experiments".to_string(),
            arguments: "{}".to_string(),
        };

        let result = tool.call(args).await;
        assert!(result.is_ok(), "expected Ok degradation, got: {result:?}");
        let msg = result.unwrap();
        assert!(
            msg.contains("unavailable"),
            "expected degradation message, got: {msg}"
        );
    }
}
