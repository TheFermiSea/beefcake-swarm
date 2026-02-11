//! Integration tests for Phase 5: Human Intervention Points
//!
//! These tests verify:
//! 1. Intervention persistence across session restarts
//! 2. Blocking logic prevents work when interventions are pending
//! 3. Resolving interventions unblocks work

use rust_cluster_mcp::harness::{
    error::HarnessError,
    tools::{
        create_shared_state, harness_complete_feature, harness_iterate,
        harness_request_intervention, harness_resolve_intervention, harness_start, harness_status,
        harness_work_on_feature, HarnessCompleteFeatureRequest, HarnessIterateRequest,
        HarnessRequestInterventionRequest, HarnessResolveInterventionRequest, HarnessStartRequest,
        HarnessStatusRequest, HarnessWorkOnFeatureRequest,
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
        {"id": "f1", "category": "testing", "description": "Feature 1", "steps": ["Step 1"], "passes": false},
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
fn test_approval_intervention_blocks_iterate() {
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

    // First iteration should succeed
    {
        let mut state = state.lock().unwrap();
        let result = harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Initial work".to_string(),
                feature_id: Some("f1".to_string()),
            },
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().iteration, 1);
    }

    // Request an approval intervention
    let intervention_id: String;
    {
        let mut state = state.lock().unwrap();
        let result = harness_request_intervention(
            &mut state,
            HarnessRequestInterventionRequest {
                intervention_type: "approval_needed".to_string(),
                question: "Should we proceed with risky operation?".to_string(),
                feature_id: Some("f1".to_string()),
                options: None,
            },
        );
        assert!(result.is_ok());
        intervention_id = result.unwrap().intervention_id;
    }

    // Next iteration should be blocked
    {
        let mut state = state.lock().unwrap();
        let result = harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Should be blocked".to_string(),
                feature_id: Some("f1".to_string()),
            },
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            HarnessError::BlockedByIntervention { message } => {
                assert!(message.contains(&intervention_id));
                assert!(message.contains("risky operation"));
            }
            e => panic!("Expected BlockedByIntervention, got {:?}", e),
        }
    }

    // Resolve the intervention
    {
        let mut state = state.lock().unwrap();
        let result = harness_resolve_intervention(
            &mut state,
            HarnessResolveInterventionRequest {
                intervention_id: intervention_id.clone(),
                resolution: "Approved - proceed".to_string(),
            },
        );
        assert!(result.is_ok());
        assert!(result.unwrap().resolved);
    }

    // Now iteration should succeed
    {
        let mut state = state.lock().unwrap();
        let result = harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Should now work".to_string(),
                feature_id: Some("f1".to_string()),
            },
        );
        assert!(result.is_ok());
        assert_eq!(result.unwrap().iteration, 2);
    }
}

#[test]
fn test_decision_point_blocks_work_on_feature() {
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

    // Request a decision point intervention
    let intervention_id: String;
    {
        let mut state = state.lock().unwrap();
        let result = harness_request_intervention(
            &mut state,
            HarnessRequestInterventionRequest {
                intervention_type: "decision_point".to_string(),
                question: "Which approach should we use?".to_string(),
                feature_id: Some("f1".to_string()),
                options: Some(vec![
                    "Approach A: Fast but risky".to_string(),
                    "Approach B: Slow but safe".to_string(),
                ]),
            },
        );
        assert!(result.is_ok());
        intervention_id = result.unwrap().intervention_id;
    }

    // work_on_feature should be blocked
    {
        let mut state = state.lock().unwrap();
        let result = harness_work_on_feature(
            &mut state,
            HarnessWorkOnFeatureRequest {
                feature_id: "f1".to_string(),
                summary: "Starting feature work".to_string(),
            },
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            HarnessError::BlockedByIntervention { message } => {
                assert!(message.contains(&intervention_id));
            }
            e => panic!("Expected BlockedByIntervention, got {:?}", e),
        }
    }

    // Resolve the decision
    {
        let mut state = state.lock().unwrap();
        harness_resolve_intervention(
            &mut state,
            HarnessResolveInterventionRequest {
                intervention_id,
                resolution: "Approach B: Slow but safe".to_string(),
            },
        )
        .unwrap();
    }

    // Now work_on_feature should succeed
    {
        let mut state = state.lock().unwrap();
        let result = harness_work_on_feature(
            &mut state,
            HarnessWorkOnFeatureRequest {
                feature_id: "f1".to_string(),
                summary: "Starting feature work".to_string(),
            },
        );
        assert!(result.is_ok());
    }
}

#[test]
fn test_review_required_does_not_block() {
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

    // Request a review_required intervention (non-blocking)
    {
        let mut state = state.lock().unwrap();
        harness_request_intervention(
            &mut state,
            HarnessRequestInterventionRequest {
                intervention_type: "review_required".to_string(),
                question: "Please review this implementation".to_string(),
                feature_id: Some("f1".to_string()),
                options: None,
            },
        )
        .unwrap();
    }

    // Iteration should NOT be blocked (review_required is non-blocking)
    {
        let mut state = state.lock().unwrap();
        let result = harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Should work despite review pending".to_string(),
                feature_id: Some("f1".to_string()),
            },
        );
        assert!(result.is_ok(), "review_required should not block iteration");
    }
}

#[test]
fn test_intervention_persists_across_restart() {
    let (dir, config) = setup_test_repo_with_features();
    let state_path = config.session_state_path.clone();

    // First session: create intervention
    let intervention_id: String;
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

            let result = harness_request_intervention(
                &mut state,
                HarnessRequestInterventionRequest {
                    intervention_type: "approval_needed".to_string(),
                    question: "Approve database migration?".to_string(),
                    feature_id: None,
                    options: None,
                },
            )
            .unwrap();
            intervention_id = result.intervention_id;
        }
        // State dropped here, simulating process end
    }

    // Verify session state file exists
    assert!(state_path.exists(), "Session state should be persisted");

    // Second session: resume and verify intervention still blocks
    {
        let state = create_shared_state(config);
        {
            let mut state = state.lock().unwrap();
            let start_result = harness_start(
                &mut state,
                HarnessStartRequest {
                    max_iterations: Some(10),
                    require_clean_git: None,
                    auto_resume: Some(true), // Resume previous session
                },
            )
            .unwrap();

            assert!(start_result.is_resume, "Should resume previous session");

            // Verify intervention is still pending via status
            let status = harness_status(
                &mut state,
                HarnessStatusRequest {
                    include_features: Some(false),
                    include_progress: Some(false),
                    max_features: None,
                    max_progress_entries: None,
                },
            )
            .unwrap();

            assert_eq!(
                status.pending_interventions.len(),
                1,
                "Intervention should persist"
            );
            assert_eq!(
                status.pending_interventions[0].id, intervention_id,
                "Same intervention ID"
            );
            assert!(
                !status.pending_interventions[0].resolved,
                "Still unresolved"
            );

            // Should still be blocked
            let result = harness_iterate(
                &mut state,
                HarnessIterateRequest {
                    summary: "Should still be blocked".to_string(),
                    feature_id: None,
                },
            );
            assert!(
                matches!(result, Err(HarnessError::BlockedByIntervention { .. })),
                "Should still be blocked after restart"
            );
        }
    }

    drop(dir); // Cleanup
}

#[test]
fn test_status_shows_pending_interventions() {
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

    // Create multiple interventions
    {
        let mut state = state.lock().unwrap();

        harness_request_intervention(
            &mut state,
            HarnessRequestInterventionRequest {
                intervention_type: "approval_needed".to_string(),
                question: "Approve deployment?".to_string(),
                feature_id: Some("f1".to_string()),
                options: None,
            },
        )
        .unwrap();

        harness_request_intervention(
            &mut state,
            HarnessRequestInterventionRequest {
                intervention_type: "review_required".to_string(),
                question: "Review security changes".to_string(),
                feature_id: Some("f1".to_string()),
                options: None,
            },
        )
        .unwrap();
    }

    // Check status shows both
    {
        let mut state = state.lock().unwrap();
        let status = harness_status(
            &mut state,
            HarnessStatusRequest {
                include_features: Some(false),
                include_progress: Some(false),
                max_features: None,
                max_progress_entries: None,
            },
        )
        .unwrap();

        assert_eq!(
            status.pending_interventions.len(),
            2,
            "Should show 2 pending interventions"
        );

        let types: Vec<_> = status
            .pending_interventions
            .iter()
            .map(|i| i.intervention_type.to_string())
            .collect();
        assert!(types.contains(&"approval_needed".to_string()));
        assert!(types.contains(&"review_required".to_string()));
    }
}

#[test]
fn test_complete_feature_blocked_by_intervention() {
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

    // Create a blocking approval intervention
    let intervention_id: String;
    {
        let mut state = state.lock().unwrap();
        let result = harness_request_intervention(
            &mut state,
            HarnessRequestInterventionRequest {
                intervention_type: "approval_needed".to_string(),
                question: "Approve the feature completion?".to_string(),
                feature_id: Some("f1".to_string()),
                options: None,
            },
        );
        assert!(result.is_ok());
        intervention_id = result.unwrap().intervention_id;
    }

    // harness_complete_feature should be blocked
    {
        let mut state = state.lock().unwrap();
        let result = harness_complete_feature(
            &mut state,
            HarnessCompleteFeatureRequest {
                feature_id: "f1".to_string(),
                summary: "Feature complete".to_string(),
                checkpoint: Some(false),
            },
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            HarnessError::SessionError { message } => {
                assert!(
                    message.contains("blocking interventions"),
                    "Error should mention blocking interventions"
                );
                assert!(
                    message.contains(&intervention_id),
                    "Error should mention the intervention ID"
                );
            }
            e => panic!("Expected SessionError, got {:?}", e),
        }
    }

    // Resolve the intervention
    {
        let mut state = state.lock().unwrap();
        harness_resolve_intervention(
            &mut state,
            HarnessResolveInterventionRequest {
                intervention_id,
                resolution: "Approved".to_string(),
            },
        )
        .unwrap();
    }

    // Now harness_complete_feature should succeed
    {
        let mut state = state.lock().unwrap();
        let result = harness_complete_feature(
            &mut state,
            HarnessCompleteFeatureRequest {
                feature_id: "f1".to_string(),
                summary: "Feature complete".to_string(),
                checkpoint: Some(false),
            },
        );
        assert!(result.is_ok(), "Should succeed after intervention resolved");
    }
}
