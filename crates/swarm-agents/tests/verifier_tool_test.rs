//! Layer 2: Verifier tool test — needs `cargo` installed but no inference.
//!
//! Creates temp Rust projects and runs the verifier tool against them.

use std::fs;

use rig::tool::Tool;
use swarm_agents::tools::verifier_tool::{RunVerifierArgs, RunVerifierTool};

/// Create a minimal valid Rust crate in the given directory.
fn create_valid_crate(dir: &std::path::Path) {
    fs::write(
        dir.join("Cargo.toml"),
        r#"[package]
name = "test-crate"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(dir.join("src/main.rs"), "fn main() {}\n").unwrap();
}

/// Create a broken Rust crate (type mismatch) in the given directory.
fn create_broken_crate(dir: &std::path::Path) {
    fs::write(
        dir.join("Cargo.toml"),
        r#"[package]
name = "broken-crate"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    fs::create_dir_all(dir.join("src")).unwrap();
    fs::write(
        dir.join("src/main.rs"),
        r#"fn broken() -> String {
    42
}

fn main() {
    broken();
}
"#,
    )
    .unwrap();
}

#[tokio::test]
async fn test_verifier_valid_crate() {
    let dir = tempfile::tempdir().unwrap();
    create_valid_crate(dir.path());

    let tool = RunVerifierTool::new(dir.path());
    let result = tool
        .call(RunVerifierArgs {
            mode: Some("compile".into()),
        })
        .await
        .unwrap();

    assert!(
        result.contains("ALL GREEN"),
        "Expected ALL GREEN in verifier output, got:\n{result}"
    );
}

#[tokio::test]
async fn test_verifier_broken_crate() {
    let dir = tempfile::tempdir().unwrap();
    create_broken_crate(dir.path());

    let tool = RunVerifierTool::new(dir.path());
    let result = tool
        .call(RunVerifierArgs {
            mode: Some("compile".into()),
        })
        .await
        .unwrap();

    assert!(
        result.contains("FAILED"),
        "Expected FAILED in verifier output, got:\n{result}"
    );
    assert!(
        result.contains("TypeMismatch"),
        "Expected TypeMismatch category, got:\n{result}"
    );
}

#[tokio::test]
async fn test_verifier_quick_mode() {
    let dir = tempfile::tempdir().unwrap();
    create_valid_crate(dir.path());

    let tool = RunVerifierTool::new(dir.path());
    let result = tool
        .call(RunVerifierArgs {
            mode: Some("quick".into()),
        })
        .await
        .unwrap();

    assert!(
        result.contains("ALL GREEN"),
        "Expected ALL GREEN in quick mode, got:\n{result}"
    );
}
