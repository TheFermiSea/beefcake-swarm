//! hx0.7.4: End-to-end swarm test slice.
//!
//! Exercises `process_issue()` with representative issue types (bug, feature,
//! refactor, short-title rejection) and verifies completion-behavior contracts:
//!
//! - Issue is claimed (set to in_progress) for valid issues.
//! - Short-title issues are NOT claimed (left open for human triage).
//! - Issue tracker always receives at least a status update on valid issues.
//! - Orchestration never panics regardless of outcome.
//!
//! Tests marked `#[ignore]` require a live inference proxy at localhost:8317.
//! All non-ignored tests are deterministic and run in CI without inference.

use std::fs;
use std::process::Command;
use std::sync::Mutex;

use anyhow::Result;
use swarm_agents::agents::AgentFactory;
use swarm_agents::beads_bridge::{BeadsIssue, IssueTracker};
use swarm_agents::config::SwarmConfig;
use swarm_agents::orchestrator;
use swarm_agents::worktree_bridge::WorktreeBridge;

// ── Mock issue tracker ────────────────────────────────────────────────────────

#[derive(Default)]
struct RecordingBeads {
    status_updates: Mutex<Vec<(String, String)>>,
    closed: Mutex<Vec<String>>,
}

impl RecordingBeads {
    fn was_claimed(&self, id: &str) -> bool {
        self.status_updates
            .lock()
            .unwrap()
            .iter()
            .any(|(i, s)| i == id && s == "in_progress")
    }

    fn was_never_claimed(&self, id: &str) -> bool {
        !self.was_claimed(id)
    }
}

impl IssueTracker for RecordingBeads {
    fn list_ready(&self) -> Result<Vec<BeadsIssue>> {
        Ok(vec![])
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

// ── Test fixtures ─────────────────────────────────────────────────────────────

/// Initialise a minimal git repo with an initial commit.
fn init_git_repo(dir: &std::path::Path) {
    for args in &[
        vec!["init"],
        vec!["config", "user.email", "test@swarm.dev"],
        vec!["config", "user.name", "Swarm E2E Test"],
        vec!["add", "."],
        vec!["commit", "-m", "init"],
    ] {
        Command::new("git")
            .args(args)
            .current_dir(dir)
            .output()
            .unwrap();
    }
}

/// Create a minimal Rust project with a compile error in `src/lib.rs`.
fn create_broken_rust_project(repo_dir: &std::path::Path, src: &str) {
    fs::create_dir_all(repo_dir.join("src")).unwrap();
    fs::write(
        repo_dir.join("Cargo.toml"),
        "[package]\nname = \"swarm-e2e-test\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(repo_dir.join("src/lib.rs"), src).unwrap();
    init_git_repo(repo_dir);
}

/// Build a minimal SwarmConfig + AgentFactory that uses the local proxy.
fn make_config_and_factory() -> (SwarmConfig, AgentFactory) {
    let config = SwarmConfig::proxy_config();
    let factory = AgentFactory::new(&config).expect("AgentFactory from proxy config");
    (config, factory)
}

// ── Non-inference tests (always run) ─────────────────────────────────────────

/// Verify that issues with titles shorter than min_objective_len are rejected
/// without claiming them on the tracker.
#[tokio::test]
async fn test_short_title_issue_is_rejected_without_claim() {
    let repo_dir = tempfile::tempdir().unwrap();
    let wt_base = tempfile::tempdir().unwrap();
    create_broken_rust_project(repo_dir.path(), "pub fn add(a: i32, b: i32) -> i32 { a + b }\n");

    let (config, factory) = make_config_and_factory();
    let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
        .expect("WorktreeBridge");
    let beads = RecordingBeads::default();

    let issue = BeadsIssue {
        id: "e2e-short".into(),
        // Deliberately below the default min_objective_len (10)
        title: "Fix".into(),
        status: "open".into(),
        priority: Some(2),
        issue_type: Some("bug".into()),
        labels: vec![],
    };

    let result = orchestrator::process_issue(&config, &factory, &bridge, &issue, &beads, None)
        .await
        .expect("process_issue should not return Err for a short title");

    assert!(!result, "short-title issue should return false (not completed)");
    assert!(
        beads.was_never_claimed("e2e-short"),
        "short-title issue should NOT be claimed — leave it for human triage"
    );
}

// ── Live-inference tests (require proxy at localhost:8317) ────────────────────

/// Representative slice — bug fix issue.
///
/// Passes a type-mismatch bug to the orchestrator and verifies the swarm
/// claims the issue and completes (or errors gracefully) without panicking.
#[tokio::test]
#[ignore]
async fn test_e2e_bug_fix_slice() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

    let repo_dir = tempfile::tempdir().unwrap();
    let wt_base = tempfile::tempdir().unwrap();
    create_broken_rust_project(
        repo_dir.path(),
        r#"pub fn count() -> u32 {
    "not a number"
}
"#,
    );

    let (config, factory) = make_config_and_factory();
    let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
        .expect("WorktreeBridge");
    let beads = RecordingBeads::default();

    let issue = BeadsIssue {
        id: "e2e-bug-001".into(),
        title: "Fix type mismatch: count() returns &str but must return u32".into(),
        status: "open".into(),
        priority: Some(1),
        issue_type: Some("bug".into()),
        labels: vec![],
    };

    let result = orchestrator::process_issue(&config, &factory, &bridge, &issue, &beads, None)
        .await;

    match &result {
        Ok(success) => eprintln!("bug-fix slice: success={success}"),
        Err(e) => eprintln!("bug-fix slice: error (acceptable)={e}"),
    }

    assert!(
        beads.was_claimed("e2e-bug-001"),
        "Bug issue must be claimed before work starts"
    );
}

/// Representative slice — feature addition issue.
#[tokio::test]
#[ignore]
async fn test_e2e_feature_addition_slice() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

    let repo_dir = tempfile::tempdir().unwrap();
    let wt_base = tempfile::tempdir().unwrap();
    create_broken_rust_project(
        repo_dir.path(),
        "pub fn add(a: i32, b: i32) -> i32 { a + b }\n",
    );

    let (config, factory) = make_config_and_factory();
    let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
        .expect("WorktreeBridge");
    let beads = RecordingBeads::default();

    let issue = BeadsIssue {
        id: "e2e-feat-001".into(),
        title: "Add subtract(a, b) -> i32 function alongside the existing add()".into(),
        status: "open".into(),
        priority: Some(2),
        issue_type: Some("feature".into()),
        labels: vec![],
    };

    let result = orchestrator::process_issue(&config, &factory, &bridge, &issue, &beads, None)
        .await;

    match &result {
        Ok(success) => eprintln!("feature slice: success={success}"),
        Err(e) => eprintln!("feature slice: error (acceptable)={e}"),
    }

    assert!(
        beads.was_claimed("e2e-feat-001"),
        "Feature issue must be claimed before work starts"
    );
}

/// Representative slice — refactor issue (rename a function).
#[tokio::test]
#[ignore]
async fn test_e2e_refactor_slice() {
    let _ = tracing_subscriber::fmt().with_env_filter("info").try_init();

    let repo_dir = tempfile::tempdir().unwrap();
    let wt_base = tempfile::tempdir().unwrap();
    create_broken_rust_project(
        repo_dir.path(),
        "pub fn do_add(a: i32, b: i32) -> i32 { a + b }\n",
    );

    let (config, factory) = make_config_and_factory();
    let bridge = WorktreeBridge::new(Some(wt_base.path().to_path_buf()), repo_dir.path())
        .expect("WorktreeBridge");
    let beads = RecordingBeads::default();

    let issue = BeadsIssue {
        id: "e2e-refactor-001".into(),
        title: "Rename do_add to add in src/lib.rs for clarity".into(),
        status: "open".into(),
        priority: Some(3),
        issue_type: Some("task".into()),
        labels: vec![],
    };

    let result = orchestrator::process_issue(&config, &factory, &bridge, &issue, &beads, None)
        .await;

    match &result {
        Ok(success) => eprintln!("refactor slice: success={success}"),
        Err(e) => eprintln!("refactor slice: error (acceptable)={e}"),
    }

    assert!(
        beads.was_claimed("e2e-refactor-001"),
        "Refactor issue must be claimed before work starts"
    );
}
