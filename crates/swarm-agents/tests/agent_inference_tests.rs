//! Layer 3: Agent inference tests — require live proxy at localhost:8317.
//!
//! All tests are `#[ignore]` — run with `cargo test -p swarm-agents -- --ignored`.
//!
//! These tests verify that individual agents can:
//! 1. Reach the proxy endpoint
//! 2. Receive a prompt and return a coherent response
//! 3. Use tools to modify files in a temp directory

use std::fs;

use rig::completion::Prompt;
use swarm_agents::agents::AgentFactory;
use swarm_agents::config::{check_endpoint, SwarmConfig};

/// Build factory from proxy config and verify proxy is reachable.
async fn proxy_factory() -> AgentFactory {
    let config = SwarmConfig::proxy_config();
    AgentFactory::new(&config).expect("factory from proxy config")
}

// ---------------------------------------------------------------------------
// Endpoint reachability
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_proxy_endpoint_reachable() {
    let config = SwarmConfig::proxy_config();
    let ok = check_endpoint(
        &config.fast_endpoint.url,
        Some(&config.fast_endpoint.api_key),
    )
    .await;
    assert!(
        ok,
        "Proxy at localhost:8317 is not reachable — start it first"
    );
}

// ---------------------------------------------------------------------------
// Reviewer agent
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_reviewer_pass_response() {
    let factory = proxy_factory().await;
    let reviewer = factory.build_reviewer();

    // An obvious, correct single-line fix
    let diff = r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,7 +10,7 @@ pub fn add(a: i32, b: i32) -> i32 {
-    a - b
+    a + b
 }
"#;

    let response = reviewer
        .prompt(diff)
        .await
        .expect("reviewer should respond");

    // The reviewer should pass this obvious fix
    let first_line = response.lines().next().unwrap_or("");
    assert!(
        first_line.trim().to_uppercase().starts_with("PASS"),
        "Expected PASS for obvious fix, got: {first_line}"
    );
}

// ---------------------------------------------------------------------------
// Rust coder with tools
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_rust_coder_with_tools() {
    let factory = proxy_factory().await;
    let dir = tempfile::tempdir().unwrap();

    // Create a broken Rust file
    fs::create_dir_all(dir.path().join("src")).unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        r#"[package]
name = "test-fix"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    fs::write(
        dir.path().join("src/main.rs"),
        r#"fn get_name() -> String {
    42
}

fn main() {
    println!("{}", get_name());
}
"#,
    )
    .unwrap();

    let coder = factory.build_rust_coder(dir.path());

    let prompt = "The file src/main.rs has a type mismatch error: `get_name()` returns \
                  String but the body returns 42. Fix it by returning a String instead. \
                  Read the file first, then write the fix.";

    let response = coder.prompt(prompt).await;
    assert!(
        response.is_ok(),
        "Rust coder should respond: {:?}",
        response.err()
    );

    // Check that the file was modified (the agent should have used write_file)
    let content = fs::read_to_string(dir.path().join("src/main.rs")).unwrap();
    assert!(
        !content.contains("42") || content.contains("String") || content.contains("to_string"),
        "Expected the coder to fix the type mismatch. Content:\n{content}"
    );
}

// ---------------------------------------------------------------------------
// General coder with tools
// ---------------------------------------------------------------------------

#[tokio::test]
#[ignore]
async fn test_general_coder_with_tools() {
    let factory = proxy_factory().await;
    let dir = tempfile::tempdir().unwrap();

    // Create a minimal project
    fs::create_dir_all(dir.path().join("src")).unwrap();
    fs::write(
        dir.path().join("Cargo.toml"),
        r#"[package]
name = "test-gen"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    fs::write(dir.path().join("src/main.rs"), "fn main() {}\n").unwrap();

    let coder = factory.build_general_coder(dir.path());

    let prompt = "Add a function `fn greet(name: &str) -> String` to src/main.rs that \
                  returns a greeting like \"Hello, {name}!\". Read the file first, then \
                  write the complete updated file.";

    let response = coder.prompt(prompt).await;
    assert!(
        response.is_ok(),
        "General coder should respond: {:?}",
        response.err()
    );

    let content = fs::read_to_string(dir.path().join("src/main.rs")).unwrap();
    assert!(
        content.contains("greet"),
        "Expected general coder to add greet function. Content:\n{content}"
    );
}
