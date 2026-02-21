//! Tool payload schema compatibility tests.
//!
//! Validates that tool definitions and argument payloads remain stable across
//! code changes. Catches accidental schema breaks that would cause LLM tool
//! calls to fail silently.
//!
//! These tests verify:
//! 1. Each tool's `definition()` has the expected name, description, and parameter schema
//! 2. Argument types deserialize correctly from canonical JSON payloads
//! 3. Parameter schemas have required structural fields (type, properties, required)

use rig::tool::Tool;
use swarm_agents::tools::exec_tool::{RunCommandArgs, RunCommandTool};
use swarm_agents::tools::fs_tools::{
    ListFilesArgs, ListFilesTool, ReadFileArgs, ReadFileTool, WriteFileArgs, WriteFileTool,
};
use swarm_agents::tools::patch_tool::{EditFileArgs, EditFileTool};
use swarm_agents::tools::verifier_tool::{RunVerifierArgs, RunVerifierTool};

// ---------------------------------------------------------------------------
// Schema structure validation
// ---------------------------------------------------------------------------

/// Verify a tool definition has the expected name and valid parameter schema.
async fn assert_tool_schema<T: Tool>(tool: &T, expected_name: &str, expected_params: &[&str]) {
    let def = tool.definition(String::new()).await;
    assert_eq!(
        def.name, expected_name,
        "Tool name mismatch for {expected_name}"
    );
    assert!(
        !def.description.is_empty(),
        "Tool {expected_name} must have a description"
    );

    // Parameters must be a JSON object with "type": "object"
    let params = &def.parameters;
    assert_eq!(
        params["type"], "object",
        "Tool {expected_name} parameters must be type 'object'"
    );

    // Must have "properties"
    let props = params["properties"]
        .as_object()
        .unwrap_or_else(|| panic!("Tool {expected_name} must have 'properties'"));

    // Check expected parameters exist
    for param in expected_params {
        assert!(
            props.contains_key(*param),
            "Tool {expected_name} missing expected parameter '{param}'"
        );

        // Each property must have "type" and "description"
        let prop = &props[*param];
        assert!(
            prop.get("type").is_some(),
            "Tool {expected_name} parameter '{param}' missing 'type'"
        );
        assert!(
            prop.get("description").is_some(),
            "Tool {expected_name} parameter '{param}' missing 'description'"
        );
    }

    // "required" array is expected for most tools but not mandatory
    // (tools with all-optional params like run_verifier won't have it)
    if let Some(required) = params.get("required").and_then(|r| r.as_array()) {
        assert!(
            !required.is_empty(),
            "Tool {expected_name} has empty 'required' array — omit it instead"
        );
    }
}

// ---------------------------------------------------------------------------
// Per-tool definition tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_read_file_schema() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool::new(dir.path());
    assert_tool_schema(&tool, "read_file", &["path"]).await;
}

#[tokio::test]
async fn test_write_file_schema() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool::new(dir.path());
    assert_tool_schema(&tool, "write_file", &["path", "content"]).await;
}

#[tokio::test]
async fn test_list_files_schema() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ListFilesTool::new(dir.path());
    assert_tool_schema(&tool, "list_files", &["path"]).await;
}

#[tokio::test]
async fn test_edit_file_schema() {
    let dir = tempfile::tempdir().unwrap();
    let tool = EditFileTool::new(dir.path());
    assert_tool_schema(&tool, "edit_file", &["path", "old_content", "new_content"]).await;
}

#[tokio::test]
async fn test_run_command_schema() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());
    assert_tool_schema(&tool, "run_command", &["command"]).await;
}

#[tokio::test]
async fn test_run_verifier_schema() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunVerifierTool::new(dir.path());
    assert_tool_schema(&tool, "run_verifier", &["mode"]).await;
}

// ---------------------------------------------------------------------------
// Argument deserialization compatibility tests
// ---------------------------------------------------------------------------
// These verify that the canonical JSON payload format LLMs produce can be
// deserialized into the Args types. If someone renames a field or changes
// a type, these tests will catch it.

#[test]
fn test_read_file_args_compat() {
    let json = r#"{"path": "src/main.rs"}"#;
    let args: ReadFileArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.path, "src/main.rs");
}

#[test]
fn test_write_file_args_compat() {
    let json = r#"{"path": "src/lib.rs", "content": "fn main() {}"}"#;
    let args: WriteFileArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.path, "src/lib.rs");
    assert_eq!(args.content, "fn main() {}");
}

#[test]
fn test_list_files_args_compat() {
    let json = r#"{"path": "src"}"#;
    let args: ListFilesArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.path, "src");

    // Empty path (workspace root)
    let json_root = r#"{"path": ""}"#;
    let args_root: ListFilesArgs = serde_json::from_str(json_root).unwrap();
    assert_eq!(args_root.path, "");
}

#[test]
fn test_edit_file_args_compat() {
    let json = r#"{
        "path": "src/lib.rs",
        "old_content": "fn old() {}",
        "new_content": "fn new() {}"
    }"#;
    let args: EditFileArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.path, "src/lib.rs");
    assert_eq!(args.old_content, "fn old() {}");
    assert_eq!(args.new_content, "fn new() {}");
}

#[test]
fn test_run_command_args_compat() {
    // RunCommandArgs takes the full command string (not separate command+args)
    let json = r#"{"command": "cargo build --release"}"#;
    let args: RunCommandArgs = serde_json::from_str(json).unwrap();
    assert_eq!(args.command, "cargo build --release");

    // Simple command
    let json_simple = r#"{"command": "cargo test"}"#;
    let args_simple: RunCommandArgs = serde_json::from_str(json_simple).unwrap();
    assert_eq!(args_simple.command, "cargo test");
}

#[test]
fn test_run_verifier_args_compat() {
    // Full mode
    let json_full = r#"{"mode": "full"}"#;
    let args: RunVerifierArgs = serde_json::from_str(json_full).unwrap();
    assert_eq!(args.mode, Some("full".to_string()));

    // Quick mode
    let json_quick = r#"{"mode": "quick"}"#;
    let args_quick: RunVerifierArgs = serde_json::from_str(json_quick).unwrap();
    assert_eq!(args_quick.mode, Some("quick".to_string()));

    // No mode (defaults to None)
    let json_none = r#"{}"#;
    let args_none: RunVerifierArgs = serde_json::from_str(json_none).unwrap();
    assert_eq!(args_none.mode, None);
}

// ---------------------------------------------------------------------------
// Schema stability snapshot tests
// ---------------------------------------------------------------------------
// These capture the full parameter schema as a JSON string and assert it
// hasn't changed. If a schema changes intentionally, update the snapshot.

#[tokio::test]
async fn test_read_file_schema_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool::new(dir.path());
    let def = tool.definition(String::new()).await;

    let expected = serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file within the workspace"
            }
        },
        "required": ["path"]
    });

    assert_eq!(
        def.parameters, expected,
        "read_file schema changed — update snapshot if intentional"
    );
}

#[tokio::test]
async fn test_write_file_schema_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool::new(dir.path());
    let def = tool.definition(String::new()).await;

    let expected = serde_json::json!({
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file within the workspace"
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file"
            }
        },
        "required": ["path", "content"]
    });

    assert_eq!(
        def.parameters, expected,
        "write_file schema changed — update snapshot if intentional"
    );
}

#[tokio::test]
async fn test_run_command_schema_snapshot() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());
    let def = tool.definition(String::new()).await;

    // Verify key structural properties without full snapshot
    // (run_command has a longer description that might evolve)
    let params = &def.parameters;
    assert_eq!(params["properties"]["command"]["type"], "string");

    let required: Vec<&str> = params["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert!(required.contains(&"command"));
}
