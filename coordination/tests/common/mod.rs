//! Shared test helpers for harness integration tests.
#![allow(dead_code)] // Not all test binaries use every helper

use coordination::harness::types::HarnessConfig;
use std::process::Command;
use tempfile::TempDir;

/// Initialize a git repo at the given path with user config and an initial commit.
/// Disables GPG signing to prevent failures in CI/test environments.
pub fn init_git_repo(path: &std::path::Path) {
    Command::new("git")
        .args(["init"])
        .current_dir(path)
        .output()
        .expect("git init failed");
    Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(path)
        .output()
        .expect("git config email failed");
    Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(path)
        .output()
        .expect("git config name failed");
    Command::new("git")
        .args(["config", "commit.gpgsign", "false"])
        .current_dir(path)
        .output()
        .expect("git config gpgsign failed");
    std::fs::write(path.join("README.md"), "# Test\n").unwrap();
    Command::new("git")
        .args(["add", "."])
        .current_dir(path)
        .output()
        .expect("git add failed");
    Command::new("git")
        .args(["commit", "-m", "Initial commit"])
        .current_dir(path)
        .output()
        .expect("git commit failed");
}

/// Build a default `HarnessConfig` rooted in the given temp directory.
pub fn harness_config(dir: &TempDir) -> HarnessConfig {
    HarnessConfig {
        features_path: dir.path().join("features.json"),
        progress_path: dir.path().join("claude-progress.txt"),
        session_state_path: dir.path().join(".harness-session.json"),
        working_directory: dir.path().to_path_buf(),
        max_iterations: 20,
        auto_checkpoint: true,
        require_clean_git: false,
        commit_prefix: "[harness]".to_string(),
    }
}
