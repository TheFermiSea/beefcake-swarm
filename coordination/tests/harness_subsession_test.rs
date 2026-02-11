//! Integration tests for Phase 6: Sub-Agent Delegation
//!
//! These tests verify:
//! 1. Sub-session creation and context file generation
//! 2. Sub-session status tracking
//! 3. Sub-session completion and result claiming
//! 4. Sub-session persistence across restarts
//! 5. Blocking behavior respects interventions

use rust_cluster_mcp::harness::{
    error::HarnessError,
    tools::{
        create_shared_state, harness_claim_sub_session_result, harness_complete_sub_session,
        harness_delegate, harness_list_sub_sessions, harness_start, harness_sub_session_status,
        HarnessClaimSubSessionResultRequest, HarnessDelegateRequest, HarnessStartRequest,
        HarnessSubSessionStatusRequest,
    },
    types::HarnessConfig,
};
use std::process::Command;
use tempfile::tempdir;

fn setup_test_repo_with_features() -> (tempfile::TempDir, HarnessConfig) {
    let dir = tempdir().unwrap();

    // Initialize git repo
    Command::new("git")
        .args(["init"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(dir.path())
        .output()
        .unwrap();
    std::fs::write(dir.path().join("README.md"), "# Test").unwrap();
    Command::new("git")
        .args(["add", "."])
        .current_dir(dir.path())
        .output()
        .unwrap();
    Command::new("git")
        .args(["commit", "-m", "Initial commit"])
        .current_dir(dir.path())
        .output()
        .unwrap();

    // Create features.json
    let features = r#"[
        {"id": "f1", "category": "testing", "description": "Feature 1", "steps": ["Step 1", "Step 2"], "passes": false},
        {"id": "f2", "category": "testing", "description": "Feature 2", "steps": ["Step 1"], "depends_on": ["f1"], "passes": false}
    ]"#;
    std::fs::write(dir.path().join("features.json"), features).unwrap();

    let config = HarnessConfig {
        features_path: dir.path().join("features.json"),
        progress_path: dir.path().join("claude-progress.txt"),
        session_state_path: dir.path().join(".harness-session.json"),
        working_directory: dir.path().to_path_buf(),
        max_iterations: 20,
        auto_checkpoint: true,
        require_clean_git: false,
        commit_prefix: "[harness]".to_string(),
    };

    (dir, config)
}

#[test]
fn test_delegate_creates_subsession() {
    let (dir, config) = setup_test_repo_with_features();
    let state = create_shared_state(config);

    // Start session
    {
        let mut state = state.lock().unwrap();
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();
    }

    // Delegate a task
    let delegate_result;
    {
        let mut state = state.lock().unwrap();
        delegate_result = harness_delegate(
            &mut state,
            HarnessDelegateRequest {
                feature_id: "f1".to_string(),
                task_description: "Implement the core algorithm".to_string(),
                max_iterations: Some(5),
            },
        );
    }

    assert!(delegate_result.is_ok(), "Delegate should succeed");
    let response = delegate_result.unwrap();

    assert!(response.success);
    assert!(!response.sub_session_id.is_empty());
    assert_eq!(response.feature_id, "f1");

    // Verify context file was created
    assert!(
        std::path::Path::new(&response.context_path).exists(),
        "Context file should be created"
    );

    // Verify context file contains expected content
    let context_content = std::fs::read_to_string(&response.context_path).unwrap();
    assert!(context_content.contains("Sub-Session ID:"));
    assert!(context_content.contains("Feature 1"));
    assert!(context_content.contains("Implement the core algorithm"));

    drop(dir);
}

#[test]
fn test_subsession_status_tracking() {
    let (_dir, config) = setup_test_repo_with_features();
    let state = create_shared_state(config);

    // Start session
    {
        let mut state = state.lock().unwrap();
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();
    }

    // Delegate a task
    let sub_session_id: String;
    {
        let mut state = state.lock().unwrap();
        let result = harness_delegate(
            &mut state,
            HarnessDelegateRequest {
                feature_id: "f1".to_string(),
                task_description: "Test task".to_string(),
                max_iterations: Some(5),
            },
        )
        .unwrap();
        sub_session_id = result.sub_session_id;
    }

    // Check status
    {
        let state = state.lock().unwrap();
        let status = harness_sub_session_status(
            &state,
            HarnessSubSessionStatusRequest {
                sub_session_id: sub_session_id.clone(),
            },
        )
        .unwrap();

        assert_eq!(status.sub_session_id, sub_session_id);
        assert_eq!(status.status, "active");
        assert_eq!(status.iteration, 0);
        assert_eq!(status.max_iterations, 5);
        assert!(status.summary.is_none());
    }
}

#[test]
fn test_complete_and_claim_subsession() {
    let (dir, config) = setup_test_repo_with_features();
    let state = create_shared_state(config);

    // Start session
    {
        let mut state = state.lock().unwrap();
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();
    }

    // Delegate a task
    let sub_session_id: String;
    let context_path: String;
    {
        let mut state = state.lock().unwrap();
        let result = harness_delegate(
            &mut state,
            HarnessDelegateRequest {
                feature_id: "f1".to_string(),
                task_description: "Test task".to_string(),
                max_iterations: Some(5),
            },
        )
        .unwrap();
        sub_session_id = result.sub_session_id;
        context_path = result.context_path;
    }

    // Complete the sub-session
    {
        let mut state = state.lock().unwrap();
        harness_complete_sub_session(
            &mut state,
            &sub_session_id,
            "Successfully implemented the algorithm",
        )
        .unwrap();
    }

    // Check status shows completed
    {
        let state = state.lock().unwrap();
        let status = harness_sub_session_status(
            &state,
            HarnessSubSessionStatusRequest {
                sub_session_id: sub_session_id.clone(),
            },
        )
        .unwrap();

        assert_eq!(status.status, "completed");
        assert_eq!(
            status.summary,
            Some("Successfully implemented the algorithm".to_string())
        );
    }

    // Claim the result
    {
        let mut state = state.lock().unwrap();
        let claim_result = harness_claim_sub_session_result(
            &mut state,
            HarnessClaimSubSessionResultRequest {
                sub_session_id: sub_session_id.clone(),
                summary: None, // Use sub-session's summary
            },
        )
        .unwrap();

        assert!(claim_result.success);
        assert_eq!(claim_result.feature_id, "f1");
        assert!(claim_result.summary.contains("Successfully implemented"));
        assert!(claim_result.progress_logged);
    }

    // Context file should be cleaned up
    assert!(
        !std::path::Path::new(&context_path).exists(),
        "Context file should be deleted after claiming"
    );

    drop(dir);
}

#[test]
fn test_cannot_claim_incomplete_subsession() {
    let (_dir, config) = setup_test_repo_with_features();
    let state = create_shared_state(config);

    // Start session
    {
        let mut state = state.lock().unwrap();
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();
    }

    // Delegate a task
    let sub_session_id: String;
    {
        let mut state = state.lock().unwrap();
        let result = harness_delegate(
            &mut state,
            HarnessDelegateRequest {
                feature_id: "f1".to_string(),
                task_description: "Test task".to_string(),
                max_iterations: Some(5),
            },
        )
        .unwrap();
        sub_session_id = result.sub_session_id;
    }

    // Try to claim without completing - should fail
    {
        let mut state = state.lock().unwrap();
        let claim_result = harness_claim_sub_session_result(
            &mut state,
            HarnessClaimSubSessionResultRequest {
                sub_session_id: sub_session_id.clone(),
                summary: None,
            },
        );

        assert!(claim_result.is_err());
        match claim_result.unwrap_err() {
            HarnessError::SessionError { message } => {
                assert!(message.contains("not finished"));
            }
            e => panic!("Expected SessionError, got {:?}", e),
        }
    }
}

#[test]
fn test_list_active_subsessions() {
    let (_dir, config) = setup_test_repo_with_features();
    let state = create_shared_state(config);

    // Start session
    {
        let mut state = state.lock().unwrap();
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();
    }

    // Delegate multiple tasks
    {
        let mut state = state.lock().unwrap();
        harness_delegate(
            &mut state,
            HarnessDelegateRequest {
                feature_id: "f1".to_string(),
                task_description: "Task 1".to_string(),
                max_iterations: Some(5),
            },
        )
        .unwrap();

        harness_delegate(
            &mut state,
            HarnessDelegateRequest {
                feature_id: "f1".to_string(),
                task_description: "Task 2".to_string(),
                max_iterations: Some(5),
            },
        )
        .unwrap();
    }

    // List active sub-sessions
    {
        let state = state.lock().unwrap();
        let active = harness_list_sub_sessions(&state);
        assert_eq!(active.len(), 2, "Should have 2 active sub-sessions");
    }

    // Complete one
    {
        let mut state = state.lock().unwrap();
        let active = harness_list_sub_sessions(&state);
        let first_id = active[0].id.clone();
        harness_complete_sub_session(&mut state, &first_id, "Done").unwrap();
    }

    // Now only one active
    {
        let state = state.lock().unwrap();
        let active = harness_list_sub_sessions(&state);
        assert_eq!(
            active.len(),
            1,
            "Should have 1 active sub-session after completing one"
        );
    }
}

#[test]
fn test_delegate_with_nonexistent_feature() {
    let (_dir, config) = setup_test_repo_with_features();
    let state = create_shared_state(config);

    // Start session
    {
        let mut state = state.lock().unwrap();
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();
    }

    // Try to delegate with nonexistent feature
    {
        let mut state = state.lock().unwrap();
        let result = harness_delegate(
            &mut state,
            HarnessDelegateRequest {
                feature_id: "nonexistent".to_string(),
                task_description: "Test task".to_string(),
                max_iterations: Some(5),
            },
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            HarnessError::FeatureNotFound { feature_id } => {
                assert_eq!(feature_id, "nonexistent");
            }
            e => panic!("Expected FeatureNotFound, got {:?}", e),
        }
    }
}

#[test]
fn test_subsession_persists_across_restart() {
    let (dir, config) = setup_test_repo_with_features();
    let state_path = config.session_state_path.clone();

    // First session: create sub-session
    let sub_session_id: String;
    {
        let state = create_shared_state(config.clone());
        {
            let mut state = state.lock().unwrap();
            harness_start(
                &mut state,
                HarnessStartRequest {
                    max_iterations: Some(10),
                    require_clean_git: None,
                    auto_resume: Some(false),
                },
            )
            .unwrap();

            let result = harness_delegate(
                &mut state,
                HarnessDelegateRequest {
                    feature_id: "f1".to_string(),
                    task_description: "Persistent task".to_string(),
                    max_iterations: Some(5),
                },
            )
            .unwrap();
            sub_session_id = result.sub_session_id;
        }
        // State dropped here
    }

    // Verify state file exists
    assert!(state_path.exists(), "Session state should be persisted");

    // Second session: resume and verify sub-session exists
    {
        let state = create_shared_state(config);
        {
            let mut state = state.lock().unwrap();
            let start_result = harness_start(
                &mut state,
                HarnessStartRequest {
                    max_iterations: Some(10),
                    require_clean_git: None,
                    auto_resume: Some(true), // Resume
                },
            )
            .unwrap();

            assert!(start_result.is_resume, "Should resume previous session");

            // Check sub-session still exists
            let status = harness_sub_session_status(
                &state,
                HarnessSubSessionStatusRequest {
                    sub_session_id: sub_session_id.clone(),
                },
            )
            .unwrap();

            assert_eq!(status.sub_session_id, sub_session_id);
            assert_eq!(status.status, "active");
        }
    }

    drop(dir);
}

#[test]
fn test_can_claim_failed_subsession() {
    let (dir, config) = setup_test_repo_with_features();
    let state = create_shared_state(config);

    // Start session
    {
        let mut state = state.lock().unwrap();
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();
    }

    // Delegate a task
    let sub_session_id: String;
    {
        let mut state = state.lock().unwrap();
        let result = harness_delegate(
            &mut state,
            HarnessDelegateRequest {
                feature_id: "f1".to_string(),
                task_description: "Test task that will fail".to_string(),
                max_iterations: Some(5),
            },
        )
        .unwrap();
        sub_session_id = result.sub_session_id;
    }

    // Fail the sub-session
    {
        let mut state = state.lock().unwrap();
        rust_cluster_mcp::harness::tools::harness_fail_sub_session(
            &mut state,
            &sub_session_id,
            "Task failed due to missing dependency",
        )
        .unwrap();
    }

    // Check status shows failed
    {
        let state = state.lock().unwrap();
        let status = harness_sub_session_status(
            &state,
            HarnessSubSessionStatusRequest {
                sub_session_id: sub_session_id.clone(),
            },
        )
        .unwrap();

        assert_eq!(status.status, "failed");
    }

    // Now claim the failed sub-session - should succeed
    {
        let mut state = state.lock().unwrap();
        let claim_result = harness_claim_sub_session_result(
            &mut state,
            HarnessClaimSubSessionResultRequest {
                sub_session_id: sub_session_id.clone(),
                summary: None, // Use sub-session's failure reason
            },
        )
        .unwrap();

        assert!(claim_result.success);
        assert_eq!(claim_result.feature_id, "f1");
        // Summary should indicate failure
        assert!(
            claim_result.summary.contains("failed")
                || claim_result.summary.contains("missing dependency")
        );
    }

    drop(dir);
}
