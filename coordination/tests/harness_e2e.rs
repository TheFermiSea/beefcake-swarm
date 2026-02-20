//! End-to-End Integration Tests for Agent Harness
//!
//! Tests complete workflows as they would be used in production:
//! - Full session lifecycle: start → work → complete
//! - Rollback recovery scenario
//! - Max-iteration abort handling
//! - Session resume from progress file
//! - Feature registry interactions

use coordination::harness::{
    create_shared_state,
    tools::{
        harness_checkpoint, harness_complete_feature, harness_end, harness_iterate,
        harness_rollback, harness_start, harness_status, HarnessCheckpointRequest,
        HarnessCompleteFeatureRequest, HarnessIterateRequest, HarnessRollbackRequest,
        HarnessStartRequest, HarnessStatusRequest,
    },
    HarnessConfig,
};
use std::fs;
use std::process::Command;
use tempfile::tempdir;

/// Setup a test environment with git repo and optional features.json
fn setup_test_env(features: Option<&str>) -> (tempfile::TempDir, HarnessConfig) {
    let dir = tempdir().expect("Failed to create temp dir");

    // Initialize git repo with initial commit
    init_git_repo(dir.path());

    // Create features.json if provided
    if let Some(content) = features {
        fs::write(dir.path().join("features.json"), content).unwrap();
    }

    let config = HarnessConfig {
        features_path: dir.path().join("features.json"),
        progress_path: dir.path().join("claude-progress.txt"),
        session_state_path: dir.path().join(".harness-session.json"),
        working_directory: dir.path().to_path_buf(),
        max_iterations: 10,
        auto_checkpoint: true,
        require_clean_git: false,
        commit_prefix: "[harness]".to_string(),
    };

    (dir, config)
}

fn init_git_repo(path: &std::path::Path) {
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
        .args(["config", "user.name", "Test User"])
        .current_dir(path)
        .output()
        .expect("git config name failed");
    fs::write(path.join("README.md"), "# Test Project\n").unwrap();
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

/// Sample features.json for testing (raw array of FeatureSpec)
const SAMPLE_FEATURES: &str = r#"[
    {
        "id": "feature-1",
        "category": "functional",
        "description": "Implement user login/logout",
        "steps": ["Users can log in", "Users can log out"],
        "passes": false,
        "priority": 0,
        "depends_on": []
    },
    {
        "id": "feature-2",
        "category": "ui",
        "description": "Create main dashboard view",
        "steps": ["Dashboard displays user data"],
        "passes": false,
        "priority": 1,
        "depends_on": ["feature-1"]
    },
    {
        "id": "feature-3",
        "category": "ui",
        "description": "User settings management",
        "steps": ["Users can update settings"],
        "passes": false,
        "priority": 2,
        "depends_on": ["feature-1"]
    }
]"#;

// ============================================================================
// Full Workflow Tests
// ============================================================================

#[test]
fn test_full_workflow_start_to_complete() {
    let (_dir, config) = setup_test_env(Some(SAMPLE_FEATURES));
    let shared_state = create_shared_state(config);

    // Start session
    let mut state = shared_state.lock().unwrap();
    let start_response = harness_start(
        &mut state,
        HarnessStartRequest {
            max_iterations: Some(20),
            require_clean_git: Some(false),
            auto_resume: Some(false),
        },
    )
    .expect("harness_start failed");

    assert!(start_response.success);
    assert!(!start_response.is_resume);
    let session_id = start_response.session_id.clone();
    assert!(!session_id.is_empty());

    // Check initial status
    let status = harness_status(
        &mut state,
        HarnessStatusRequest {
            include_features: Some(true),
            include_progress: Some(true),
            include_structured_summary: None,
            max_features: None,
            max_progress_entries: None,
        },
    )
    .expect("harness_status failed");

    assert!(status.session.is_some());
    assert_eq!(status.features.total, 3);
    assert_eq!(status.features.passing, 0);

    // Iterate through work
    let iter1 = harness_iterate(
        &mut state,
        HarnessIterateRequest {
            summary: "Started working on authentication".to_string(),
            feature_id: Some("feature-1".to_string()),
        },
    )
    .expect("harness_iterate failed");

    assert_eq!(iter1.iteration, 1);
    assert!(iter1.can_continue);

    // Complete first feature
    fs::write(
        state.config.working_directory.join("auth.rs"),
        "// Authentication module\npub fn login() {}\npub fn logout() {}\n",
    )
    .unwrap();

    let complete1 = harness_complete_feature(
        &mut state,
        HarnessCompleteFeatureRequest {
            feature_id: "feature-1".to_string(),
            summary: "Implemented login and logout functions".to_string(),
            checkpoint: Some(true),
        },
    )
    .expect("harness_complete_feature failed");

    assert!(complete1.success);
    assert_eq!(complete1.remaining_features, 2);
    assert!(complete1.checkpoint_commit.is_some());

    // Continue with second feature
    let iter2 = harness_iterate(
        &mut state,
        HarnessIterateRequest {
            summary: "Working on dashboard".to_string(),
            feature_id: Some("feature-2".to_string()),
        },
    )
    .expect("harness_iterate failed");

    assert_eq!(iter2.iteration, 2);

    // Complete second feature
    fs::write(
        state.config.working_directory.join("dashboard.rs"),
        "// Dashboard module\npub fn render() {}\n",
    )
    .unwrap();

    let complete2 = harness_complete_feature(
        &mut state,
        HarnessCompleteFeatureRequest {
            feature_id: "feature-2".to_string(),
            summary: "Built dashboard UI".to_string(),
            checkpoint: Some(true),
        },
    )
    .expect("harness_complete_feature failed");

    assert!(complete2.success);
    assert_eq!(complete2.remaining_features, 1);

    // Complete final feature
    let iter3 = harness_iterate(
        &mut state,
        HarnessIterateRequest {
            summary: "Working on settings".to_string(),
            feature_id: Some("feature-3".to_string()),
        },
    )
    .expect("harness_iterate failed");

    assert_eq!(iter3.iteration, 3);

    fs::write(
        state.config.working_directory.join("settings.rs"),
        "// Settings module\npub fn update_settings() {}\n",
    )
    .unwrap();

    let complete3 = harness_complete_feature(
        &mut state,
        HarnessCompleteFeatureRequest {
            feature_id: "feature-3".to_string(),
            summary: "Implemented settings page".to_string(),
            checkpoint: Some(true),
        },
    )
    .expect("harness_complete_feature failed");

    assert!(complete3.success);
    assert_eq!(complete3.remaining_features, 0);
    assert!((complete3.completion_percent - 100.0).abs() < 0.01);

    // End session successfully
    let summary = harness_end(&mut state, true, "All features completed").unwrap();
    assert_eq!(summary.iterations, 3);

    // Verify progress file was created
    assert!(state.config.progress_path.exists());
    let progress_content = fs::read_to_string(&state.config.progress_path).unwrap();
    assert!(progress_content.contains("SESSION_START"));
    assert!(progress_content.contains("feature-1"));
    assert!(progress_content.contains("feature-2"));
    assert!(progress_content.contains("feature-3"));
    assert!(progress_content.contains("SESSION_END"));
}

// ============================================================================
// Rollback Recovery Tests
// ============================================================================

#[test]
fn test_rollback_recovery_scenario() {
    let (dir, config) = setup_test_env(Some(SAMPLE_FEATURES));
    let shared_state = create_shared_state(config);

    let mut state = shared_state.lock().unwrap();

    // Start session
    harness_start(
        &mut state,
        HarnessStartRequest {
            max_iterations: Some(20),
            require_clean_git: Some(false),
            auto_resume: Some(false),
        },
    )
    .unwrap();

    // Do some work and checkpoint
    harness_iterate(
        &mut state,
        HarnessIterateRequest {
            summary: "Working on auth".to_string(),
            feature_id: Some("feature-1".to_string()),
        },
    )
    .unwrap();

    fs::write(dir.path().join("auth.rs"), "// Good authentication code\n").unwrap();

    let checkpoint1 = harness_checkpoint(
        &mut state,
        HarnessCheckpointRequest {
            description: "Auth module working".to_string(),
            feature_id: Some("feature-1".to_string()),
        },
    )
    .expect("checkpoint failed");

    let good_commit = checkpoint1.commit_hash.clone();

    // Now make a "bad" change
    harness_iterate(
        &mut state,
        HarnessIterateRequest {
            summary: "Refactoring auth - BROKEN".to_string(),
            feature_id: Some("feature-1".to_string()),
        },
    )
    .unwrap();

    fs::write(
        dir.path().join("auth.rs"),
        "// Broken code that doesn't work\ncompile_error!(\"oops\");\n",
    )
    .unwrap();

    let _bad_checkpoint = harness_checkpoint(
        &mut state,
        HarnessCheckpointRequest {
            description: "Broken refactor".to_string(),
            feature_id: Some("feature-1".to_string()),
        },
    )
    .expect("bad checkpoint failed");

    // Verify broken state
    let broken_content = fs::read_to_string(dir.path().join("auth.rs")).unwrap();
    assert!(broken_content.contains("compile_error"));

    // Rollback to good state
    let rollback = harness_rollback(
        &mut state,
        HarnessRollbackRequest {
            commit_hash: good_commit.clone(),
            hard: Some(true),
        },
    )
    .expect("rollback failed");

    assert!(rollback.success);
    assert!(rollback.was_hard_rollback);
    assert_eq!(rollback.rolled_back_to, good_commit);

    // Verify recovered state
    let recovered_content = fs::read_to_string(dir.path().join("auth.rs")).unwrap();
    assert!(recovered_content.contains("Good authentication code"));
    assert!(!recovered_content.contains("compile_error"));

    // Verify progress contains rollback marker
    let progress_content = fs::read_to_string(&state.config.progress_path).unwrap();
    assert!(progress_content.contains("ROLLBACK"));
}

// ============================================================================
// Max-Iteration Abort Tests
// ============================================================================

#[test]
fn test_max_iteration_abort() {
    let (_dir, config) = setup_test_env(Some(SAMPLE_FEATURES));
    let shared_state = create_shared_state(config);

    let mut state = shared_state.lock().unwrap();

    // Start with low max iterations
    harness_start(
        &mut state,
        HarnessStartRequest {
            max_iterations: Some(3),
            require_clean_git: Some(false),
            auto_resume: Some(false),
        },
    )
    .unwrap();

    // Iterate until max
    for i in 1..=3 {
        let resp = harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: format!("Iteration {}", i),
                feature_id: None,
            },
        )
        .unwrap();

        assert_eq!(resp.iteration, i);

        if i < 3 {
            assert!(
                resp.can_continue,
                "Should be able to continue at iteration {}",
                i
            );
        } else {
            assert!(
                !resp.can_continue,
                "Should NOT be able to continue at max iteration"
            );
        }
    }

    // Try to iterate beyond max - should fail
    let beyond_max = harness_iterate(
        &mut state,
        HarnessIterateRequest {
            summary: "Should fail".to_string(),
            feature_id: None,
        },
    );

    assert!(
        beyond_max.is_err(),
        "Should fail when exceeding max iterations"
    );

    // Verify session status
    let status = harness_status(
        &mut state,
        HarnessStatusRequest {
            include_features: None,
            include_progress: None,
            include_structured_summary: None,
            max_features: None,
            max_progress_entries: None,
        },
    )
    .unwrap();

    assert!(status.session.is_some());
    let session = status.session.unwrap();
    assert_eq!(session.iterations, 3);
}

// ============================================================================
// Session Resume Tests
// ============================================================================

#[test]
fn test_session_resume_from_progress() {
    let (dir, config) = setup_test_env(Some(SAMPLE_FEATURES));

    // First session - does partial work
    {
        let shared_state = create_shared_state(config.clone());
        let mut state = shared_state.lock().unwrap();

        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: Some(false),
                auto_resume: Some(false),
            },
        )
        .unwrap();

        // Do some work
        harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Working on feature 1".to_string(),
                feature_id: Some("feature-1".to_string()),
            },
        )
        .unwrap();

        harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Continuing feature 1".to_string(),
                feature_id: Some("feature-1".to_string()),
            },
        )
        .unwrap();

        fs::write(dir.path().join("partial.rs"), "// Partial work\n").unwrap();

        harness_checkpoint(
            &mut state,
            HarnessCheckpointRequest {
                description: "Partial progress".to_string(),
                feature_id: Some("feature-1".to_string()),
            },
        )
        .unwrap();

        // Don't end session - simulate crash/context loss
    }

    // Verify progress file exists and contains first session's work
    assert!(config.progress_path.exists());
    let progress_before = fs::read_to_string(&config.progress_path).unwrap();
    assert!(progress_before.contains("feature-1"));
    assert!(progress_before.contains("Working on feature 1"));
    assert!(progress_before.contains("CHECKPOINT"));

    // Second session - starts fresh (auto_resume may not work without serialized state)
    // This tests that progress file persists and new session can continue from checkpointed state
    {
        let shared_state = create_shared_state(config.clone());
        let mut state = shared_state.lock().unwrap();

        let start_resp = harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: Some(false),
                auto_resume: Some(false), // Start fresh but with existing progress file
            },
        )
        .unwrap();

        assert!(start_resp.success);
        // New session starts fresh
        assert!(!start_resp.is_resume);

        // Continue work from where first session left off
        harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Continuing work in new session".to_string(),
                feature_id: Some("feature-1".to_string()),
            },
        )
        .unwrap();

        // Complete the feature
        fs::write(
            dir.path().join("complete.rs"),
            "// Complete implementation\n",
        )
        .unwrap();

        harness_complete_feature(
            &mut state,
            HarnessCompleteFeatureRequest {
                feature_id: "feature-1".to_string(),
                summary: "Finished in second session".to_string(),
                checkpoint: Some(true),
            },
        )
        .unwrap();

        harness_end(&mut state, true, "Completed in second session").unwrap();
    }

    // Verify progress file contains BOTH sessions' work (appended)
    let progress_after = fs::read_to_string(&config.progress_path).unwrap();
    // First session's entries
    assert!(progress_after.contains("Working on feature 1"));
    // Second session's entries
    assert!(progress_after.contains("Continuing work in new session"));
    assert!(progress_after.contains("FEATURE_COMPLETE"));
    assert!(progress_after.contains("SESSION_END"));
}

// ============================================================================
// Feature Registry Tests
// ============================================================================

#[test]
fn test_feature_dependency_ordering() {
    let features_with_deps = r#"[
        {
            "id": "db",
            "category": "functional",
            "description": "Set up database",
            "steps": ["DB works"],
            "passes": false,
            "priority": 0,
            "depends_on": []
        },
        {
            "id": "api",
            "category": "api",
            "description": "REST API",
            "steps": ["API works"],
            "passes": false,
            "priority": 1,
            "depends_on": ["db"]
        },
        {
            "id": "ui",
            "category": "ui",
            "description": "Frontend",
            "steps": ["UI works"],
            "passes": false,
            "priority": 2,
            "depends_on": ["api"]
        }
    ]"#;

    let (dir, config) = setup_test_env(Some(features_with_deps));
    let shared_state = create_shared_state(config);
    let mut state = shared_state.lock().unwrap();

    harness_start(
        &mut state,
        HarnessStartRequest {
            max_iterations: Some(20),
            require_clean_git: Some(false),
            auto_resume: Some(false),
        },
    )
    .unwrap();

    // First available feature should be "db" (no dependencies)
    let status = harness_status(
        &mut state,
        HarnessStatusRequest {
            include_features: Some(true),
            include_progress: Some(false),
            include_structured_summary: None,
            max_features: None,
            max_progress_entries: None,
        },
    )
    .unwrap();

    assert_eq!(status.next_feature, Some("db".to_string()));

    // Complete db
    fs::write(dir.path().join("db.rs"), "// DB\n").unwrap();
    harness_complete_feature(
        &mut state,
        HarnessCompleteFeatureRequest {
            feature_id: "db".to_string(),
            summary: "DB done".to_string(),
            checkpoint: Some(true),
        },
    )
    .unwrap();

    // Next should be "api" (db complete, so api unblocked)
    let status = harness_status(
        &mut state,
        HarnessStatusRequest {
            include_features: Some(true),
            include_progress: Some(false),
            include_structured_summary: None,
            max_features: None,
            max_progress_entries: None,
        },
    )
    .unwrap();

    assert_eq!(status.next_feature, Some("api".to_string()));

    // Complete api
    fs::write(dir.path().join("api.rs"), "// API\n").unwrap();
    harness_complete_feature(
        &mut state,
        HarnessCompleteFeatureRequest {
            feature_id: "api".to_string(),
            summary: "API done".to_string(),
            checkpoint: Some(true),
        },
    )
    .unwrap();

    // Next should be "ui"
    let status = harness_status(
        &mut state,
        HarnessStatusRequest {
            include_features: Some(true),
            include_progress: Some(false),
            include_structured_summary: None,
            max_features: None,
            max_progress_entries: None,
        },
    )
    .unwrap();

    assert_eq!(status.next_feature, Some("ui".to_string()));
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_operations_without_session_fail() {
    let (_dir, config) = setup_test_env(None);
    let shared_state = create_shared_state(config);
    let mut state = shared_state.lock().unwrap();

    // Try to iterate without starting
    let result = harness_iterate(
        &mut state,
        HarnessIterateRequest {
            summary: "Should fail".to_string(),
            feature_id: None,
        },
    );
    assert!(result.is_err());

    // Try to checkpoint without starting
    let result = harness_checkpoint(
        &mut state,
        HarnessCheckpointRequest {
            description: "Should fail".to_string(),
            feature_id: None,
        },
    );
    assert!(result.is_err());

    // Try to complete feature without starting
    let result = harness_complete_feature(
        &mut state,
        HarnessCompleteFeatureRequest {
            feature_id: "anything".to_string(),
            summary: "Should fail".to_string(),
            checkpoint: None,
        },
    );
    assert!(result.is_err());
}

#[test]
fn test_complete_nonexistent_feature() {
    let (_dir, config) = setup_test_env(Some(SAMPLE_FEATURES));
    let shared_state = create_shared_state(config);
    let mut state = shared_state.lock().unwrap();

    harness_start(
        &mut state,
        HarnessStartRequest {
            max_iterations: Some(10),
            require_clean_git: Some(false),
            auto_resume: Some(false),
        },
    )
    .unwrap();

    // Try to complete feature that doesn't exist
    let result = harness_complete_feature(
        &mut state,
        HarnessCompleteFeatureRequest {
            feature_id: "nonexistent-feature".to_string(),
            summary: "This should fail".to_string(),
            checkpoint: Some(false),
        },
    );

    assert!(result.is_err());
}

// ============================================================================
// Git Integration Tests
// ============================================================================

#[test]
fn test_checkpoint_creates_commits() {
    let (dir, config) = setup_test_env(None);
    let shared_state = create_shared_state(config);
    let mut state = shared_state.lock().unwrap();

    harness_start(
        &mut state,
        HarnessStartRequest {
            max_iterations: Some(10),
            require_clean_git: Some(false),
            auto_resume: Some(false),
        },
    )
    .unwrap();

    // Create file and checkpoint
    fs::write(dir.path().join("new_file.rs"), "// New code\n").unwrap();

    let checkpoint = harness_checkpoint(
        &mut state,
        HarnessCheckpointRequest {
            description: "Added new file".to_string(),
            feature_id: None,
        },
    )
    .expect("checkpoint should succeed");

    assert!(checkpoint.success);
    assert!(!checkpoint.commit_hash.is_empty());

    // Verify commit exists in git
    let log_output = Command::new("git")
        .args(["log", "--oneline", "-n", "1"])
        .current_dir(dir.path())
        .output()
        .expect("git log failed");

    let log_str = String::from_utf8_lossy(&log_output.stdout);
    assert!(
        log_str.contains("[harness]"),
        "Commit should have harness prefix"
    );
    assert!(
        log_str.contains("Added new file"),
        "Commit should contain description"
    );
}

#[test]
fn test_require_clean_git_fails_with_changes() {
    let (dir, config) = setup_test_env(None);

    // Make uncommitted changes
    fs::write(dir.path().join("dirty.txt"), "uncommitted").unwrap();

    let shared_state = create_shared_state(config);
    let mut state = shared_state.lock().unwrap();

    // Start with require_clean_git = true should fail
    let result = harness_start(
        &mut state,
        HarnessStartRequest {
            max_iterations: Some(10),
            require_clean_git: Some(true),
            auto_resume: Some(false),
        },
    );

    assert!(result.is_err());
}

// ============================================================================
// Concurrent Access Tests (Thread Safety)
// ============================================================================

#[test]
fn test_shared_state_is_thread_safe() {
    use std::thread;

    let (_dir, config) = setup_test_env(Some(SAMPLE_FEATURES));
    let shared_state = create_shared_state(config);

    // Start session
    {
        let mut state = shared_state.lock().unwrap();
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(100),
                require_clean_git: Some(false),
                auto_resume: Some(false),
            },
        )
        .unwrap();
    }

    // Spawn multiple threads to read status concurrently
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let state_clone = shared_state.clone();
            thread::spawn(move || {
                for _ in 0..10 {
                    let mut state = state_clone.lock().unwrap();
                    let _ = harness_status(
                        &mut state,
                        HarnessStatusRequest {
                            include_features: Some(true),
                            include_progress: Some(false),
                            include_structured_summary: None,
                            max_features: None,
                            max_progress_entries: None,
                        },
                    );
                }
            })
        })
        .collect();

    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Verify state is still valid
    let mut state = shared_state.lock().unwrap();
    let status = harness_status(
        &mut state,
        HarnessStatusRequest {
            include_features: Some(true),
            include_progress: Some(false),
            include_structured_summary: None,
            max_features: None,
            max_progress_entries: None,
        },
    )
    .unwrap();

    assert!(status.session.is_some());
}
