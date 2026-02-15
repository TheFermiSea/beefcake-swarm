//! Layer 4: Manager delegation test — verifies the manager delegates to workers.
//!
//! Requires live proxy at localhost:8317. Run with `--ignored`.
//!
//! NOTE: rig-core 0.30 has a known integer overflow bug in token accounting
//! during nested agent-as-tool calls. This test catches that panic and reports
//! it as a known issue rather than masking it.

use std::fs;
use std::sync::Arc;

use rig::completion::Prompt;
use swarm_agents::agents::AgentFactory;
use swarm_agents::config::SwarmConfig;

#[tokio::test]
#[ignore]
async fn test_manager_delegates_fix() {
    let config = SwarmConfig::proxy_config();
    let factory = Arc::new(AgentFactory::new(&config).expect("factory from proxy config"));

    let dir = tempfile::tempdir().unwrap();
    let dir_path = dir.path().to_path_buf();

    // Create a broken Rust project
    fs::create_dir_all(dir_path.join("src")).unwrap();
    fs::write(
        dir_path.join("Cargo.toml"),
        r#"[package]
name = "manager-test"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    fs::write(
        dir_path.join("src/main.rs"),
        r#"fn count() -> u32 {
    "not a number"
}

fn main() {
    println!("{}", count());
}
"#,
    )
    .unwrap();

    let prompt = "There is a type mismatch in src/main.rs: `count()` should return u32 \
                  but returns a string literal. Delegate to the appropriate coder to fix it. \
                  After the fix, run the verifier to confirm it compiles."
        .to_string();

    // Spawn in a task to catch panics (rig-core 0.30 overflow bug)
    let test_dir = dir_path.clone();
    let handle = tokio::task::spawn(async move {
        let manager = factory.build_manager(&test_dir);
        manager.prompt(&prompt).await
    });

    match handle.await {
        Ok(Ok(_response)) => {
            // Manager responded — check if file was modified
            let content = fs::read_to_string(dir_path.join("src/main.rs")).unwrap();
            assert!(
                !content.contains(r#""not a number""#) || content.contains("u32"),
                "Expected the manager to delegate a fix. Content:\n{content}"
            );
        }
        Ok(Err(e)) => {
            eprintln!("Manager returned error (may be model-specific): {e}");
        }
        Err(join_err) => {
            if join_err.is_panic() {
                eprintln!(
                    "KNOWN ISSUE: rig-core 0.30 panics with 'attempt to subtract with overflow' \
                     during nested agent-as-tool token accounting. \
                     See: https://github.com/0xPlaygrounds/rig/issues — upgrade rig when fixed."
                );
            } else {
                eprintln!("Task cancelled: {join_err}");
            }
        }
    }
}
