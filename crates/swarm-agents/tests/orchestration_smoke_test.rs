//! Layer 5: Orchestration smoke test — full loop with MockBeads.
//!
//! Requires live proxy at localhost:8317. Run with `--ignored`.
//!
//! Creates a temp git repo with a broken Rust project, runs process_issue()
//! with a mock IssueTracker, and verifies the loop doesn't panic.

use std::fs;
use std::process::Command;
use std::sync::Mutex;

use anyhow::Result;
use swarm_agents::agents::AgentFactory;
use swarm_agents::beads_bridge::{BeadsIssue, IssueTracker};
use swarm_agents::config::SwarmConfig;
use swarm_agents::orchestrator;
use swarm_agents::worktree_bridge::WorktreeBridge;

/// Mock issue tracker that records calls without shelling out.
struct MockBeads {
    status_updates: Mutex<Vec<(String, String)>>,
    closed: Mutex<Vec<String>>,
}

impl MockBeads {
    fn new() -> Self {
        Self {
            status_updates: Mutex::new(Vec::new()),
            closed: Mutex::new(Vec::new()),
        }
    }

    fn was_claimed(&self, id: &str) -> bool {
        self.status_updates
            .lock()
            .unwrap()
            .iter()
            .any(|(i, s)| i == id && s == "in_progress")
    }
}

impl IssueTracker for MockBeads {
    fn list_ready(&self) -> Result<Vec<BeadsIssue>> {
        Ok(vec![BeadsIssue {
            id: "mock-001".into(),
            title: "Fix type mismatch in count()".into(),
            status: "open".into(),
            priority: Some(1),
            issue_type: Some("bug".into()),
        }])
    }

    fn update_status(&self, id: &str, status: &str) -> Result<()> {
        self.status_updates
            .lock()
            .unwrap()
            .push((id.to_string(), status.to_string()));
        Ok(())
    }

    fn close(&self, id: &str, _reason: Option<&str>) -> Result<()> {
        self.closed.lock().unwrap().push(id.to_string());
        Ok(())
    }
}

/// Initialize a git repo in the given directory with an initial commit.
fn init_git_repo(dir: &std::path::Path) {
    Command::new("git")
        .args(["init"])
        .current_dir(dir)
        .output()
        .unwrap();
    Command::new("git")
        .args(["config", "user.email", "test@swarm.dev"])
        .current_dir(dir)
        .output()
        .unwrap();
    Command::new("git")
        .args(["config", "user.name", "Swarm Test"])
        .current_dir(dir)
        .output()
        .unwrap();
    Command::new("git")
        .args(["add", "."])
        .current_dir(dir)
        .output()
        .unwrap();
    Command::new("git")
        .args(["commit", "-m", "init"])
        .current_dir(dir)
        .output()
        .unwrap();
}

#[tokio::test]
#[ignore]
async fn test_orchestration_smoke() {
    // Set up tracing for visibility
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

    let config = SwarmConfig::proxy_config();
    let factory = AgentFactory::new(&config).expect("factory from proxy config");

    // Create a temp git repo with a broken Rust project
    let repo_dir = tempfile::tempdir().unwrap();
    let wt_base = tempfile::tempdir().unwrap();

    fs::create_dir_all(repo_dir.path().join("src")).unwrap();
    fs::write(
        repo_dir.path().join("Cargo.toml"),
        r#"[package]
name = "smoke-test"
version = "0.1.0"
edition = "2021"
"#,
    )
    .unwrap();
    fs::write(
        repo_dir.path().join("src/main.rs"),
        r#"fn count() -> u32 {
    "not a number"
}

fn main() {
    println!("{}", count());
}
"#,
    )
    .unwrap();

    init_git_repo(repo_dir.path());

    let worktree_bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
        .expect("worktree bridge");

    let beads = MockBeads::new();
    let issue = BeadsIssue {
        id: "smoke-001".into(),
        title: "Fix type mismatch in count()".into(),
        status: "open".into(),
        priority: Some(1),
        issue_type: Some("bug".into()),
    };

    // Run the orchestration loop
    let result =
        orchestrator::process_issue(&config, &factory, &worktree_bridge, &issue, &beads, None)
            .await;

    // The loop should not panic regardless of outcome
    match &result {
        Ok(success) => {
            eprintln!("Orchestration completed: success={success}");
        }
        Err(e) => {
            eprintln!("Orchestration error (acceptable in smoke test): {e}");
        }
    }

    // Verify that the issue was claimed
    assert!(
        beads.was_claimed("smoke-001"),
        "Issue should have been claimed (set to in_progress)"
    );
}
