use super::*;
use std::collections::HashMap;
use std::time::Duration;

#[test]
fn test_initial_state() {
    let sm = StateMachine::new();
    assert_eq!(sm.current(), OrchestratorState::SelectingIssue);
    assert!(!sm.is_terminal());
    assert_eq!(sm.transitions().len(), 0);
}

#[test]
fn test_happy_path_transitions() {
    let mut sm = StateMachine::new();

    // Full success path
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Validating, Some("all gates green"))
        .unwrap();
    sm.advance(OrchestratorState::Merging, Some("validator passed"))
        .unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    assert!(sm.is_terminal());
    assert_eq!(sm.current(), OrchestratorState::Resolved);
    assert_eq!(sm.transitions().len(), 7);
}

#[test]
fn test_retry_loop() {
    let mut sm = StateMachine::new();

    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();

    // Verifier found errors → retry
    sm.advance(
        OrchestratorState::Implementing,
        Some("errors found, retrying"),
    )
    .unwrap();
    sm.set_iteration(2);
    sm.advance(OrchestratorState::Verifying, None).unwrap();

    // Now green → validate → merge
    sm.advance(OrchestratorState::Validating, None).unwrap();
    sm.advance(OrchestratorState::Merging, None).unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    assert!(sm.is_terminal());
    assert_eq!(sm.transitions().len(), 9);
}

#[test]
fn test_escalation_path() {
    let mut sm = StateMachine::new();

    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();

    // Errors persist → escalate
    sm.advance(
        OrchestratorState::Escalating,
        Some("repeated borrow errors"),
    )
    .unwrap();
    sm.advance(OrchestratorState::Implementing, Some("escalated to Cloud"))
        .unwrap();
    sm.set_iteration(2);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(
        OrchestratorState::Merging,
        Some("all green after escalation"),
    )
    .unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    assert!(sm.is_terminal());
}

#[test]
fn test_failure_from_any_state() {
    for state in [
        OrchestratorState::SelectingIssue,
        OrchestratorState::PreparingWorktree,
        OrchestratorState::Planning,
        OrchestratorState::Implementing,
        OrchestratorState::Verifying,
        OrchestratorState::Validating,
        OrchestratorState::Escalating,
        OrchestratorState::Merging,
    ] {
        let mut sm = StateMachine {
            current: state,
            iteration: 0,
            created_at: Instant::now(),
            transitions: Vec::new(),
        };
        assert!(sm.fail("test failure").is_ok());
        assert_eq!(sm.current(), OrchestratorState::Failed);
        assert!(sm.is_terminal());
    }
}

#[test]
fn test_cannot_transition_from_terminal() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Merging, None).unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    // Cannot transition from Resolved
    let err = sm
        .advance(OrchestratorState::Implementing, None)
        .unwrap_err();
    assert_eq!(err.from, OrchestratorState::Resolved);
    assert_eq!(err.to, OrchestratorState::Implementing);

    // Cannot fail from terminal either
    assert!(sm.fail("nope").is_err());
}

#[test]
fn test_illegal_skip_transition() {
    let mut sm = StateMachine::new();

    // Can't skip directly to Implementing without PreparingWorktree
    let err = sm
        .advance(OrchestratorState::Implementing, None)
        .unwrap_err();
    assert_eq!(err.from, OrchestratorState::SelectingIssue);
    assert_eq!(err.to, OrchestratorState::Implementing);
}

#[test]
fn test_illegal_backward_transition() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();

    // Can't go backward to SelectingIssue
    assert!(sm.advance(OrchestratorState::SelectingIssue, None).is_err());
}

#[test]
fn test_transition_record_has_reason() {
    let mut sm = StateMachine::new();
    sm.advance(
        OrchestratorState::PreparingWorktree,
        Some("issue-123 selected"),
    )
    .unwrap();

    let record = &sm.transitions()[0];
    assert_eq!(record.from, OrchestratorState::SelectingIssue);
    assert_eq!(record.to, OrchestratorState::PreparingWorktree);
    assert_eq!(record.reason.as_deref(), Some("issue-123 selected"));
}

#[test]
fn test_transition_record_serde_roundtrip() {
    let record = TransitionRecord {
        from: OrchestratorState::Verifying,
        to: OrchestratorState::Escalating,
        iteration: 3,
        elapsed_ms: 12345,
        reason: Some("repeated borrow errors".into()),
    };

    let json = serde_json::to_string(&record).unwrap();
    let restored: TransitionRecord = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.from, OrchestratorState::Verifying);
    assert_eq!(restored.to, OrchestratorState::Escalating);
    assert_eq!(restored.iteration, 3);
    assert_eq!(restored.elapsed_ms, 12345);
}

#[test]
fn test_state_display() {
    assert_eq!(
        OrchestratorState::SelectingIssue.to_string(),
        "SelectingIssue"
    );
    assert_eq!(OrchestratorState::Failed.to_string(), "Failed");
}

#[test]
fn test_summary() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.fail("test").unwrap();
    let summary = sm.summary();
    assert!(summary.contains("Failed"));
    assert!(summary.contains("2 transitions"));
}

#[test]
fn test_verifying_can_skip_to_merging() {
    // When verifier is green and no cloud validation needed
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(
        OrchestratorState::Merging,
        Some("all green, no cloud validation needed"),
    )
    .unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();
    assert!(sm.is_terminal());
}

#[test]
fn test_validator_can_trigger_escalation() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Validating, None).unwrap();
    // Validator says needs_escalation
    sm.advance(
        OrchestratorState::Escalating,
        Some("validator: needs_escalation"),
    )
    .unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    assert_eq!(sm.current(), OrchestratorState::Implementing);
}

// ──────────────────────────────────────────────────────────────────────
// Checkpoint / Resume Tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_checkpoint_at_verifying() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();

    let cp = sm.checkpoint("issue-123", Some("abc1234")).unwrap();
    assert_eq!(cp.schema_version, CHECKPOINT_SCHEMA_VERSION);
    assert_eq!(cp.state, OrchestratorState::Verifying);
    assert_eq!(cp.iteration, 1);
    assert_eq!(cp.issue_id, "issue-123");
    assert_eq!(cp.git_hash.as_deref(), Some("abc1234"));
    assert_eq!(cp.transitions.len(), 4);
}

#[test]
fn test_checkpoint_not_allowed_at_terminal() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.fail("test").unwrap();

    // Terminal states are not checkpointable
    assert!(sm.checkpoint("issue", None).is_none());
}

#[test]
fn test_checkpoint_not_allowed_at_pre_loop() {
    let sm = StateMachine::new();
    // SelectingIssue is not checkpointable
    assert!(sm.checkpoint("issue", None).is_none());
}

#[test]
fn test_resume_from_checkpoint() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(2);
    sm.advance(OrchestratorState::Verifying, None).unwrap();

    let cp = sm.checkpoint("issue-456", Some("def5678")).unwrap();

    // Resume from checkpoint
    match StateMachine::resume_from(&cp, Some("def5678")) {
        ResumeResult::Restored(restored) => {
            assert_eq!(restored.current(), OrchestratorState::Verifying);
            assert_eq!(restored.iteration(), 2);
            assert_eq!(restored.transitions().len(), 4);
            // Can continue from restored state
            // (verify we can actually transition)
        }
        other => panic!("Expected Restored, got {other:?}"),
    }
}

#[test]
fn test_resume_continues_transitions() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();

    let cp = sm.checkpoint("issue", None).unwrap();

    match StateMachine::resume_from(&cp, None) {
        ResumeResult::Restored(mut restored) => {
            // Can advance from restored state
            restored
                .advance(OrchestratorState::Implementing, Some("resumed — retrying"))
                .unwrap();
            assert_eq!(restored.current(), OrchestratorState::Implementing);
            // Transition log includes both original and new transitions
            assert_eq!(restored.transitions().len(), 5);
        }
        other => panic!("Expected Restored, got {other:?}"),
    }
}

#[test]
fn test_resume_incompatible_schema() {
    let cp = StateCheckpoint {
        schema_version: 99, // Future version
        checkpoint_id: 0,
        state: OrchestratorState::Verifying,
        iteration: 1,
        transitions: vec![],
        created_at: "2026-01-01T00:00:00Z".into(),
        git_hash: None,
        issue_id: "issue".into(),
    };

    match StateMachine::resume_from(&cp, None) {
        ResumeResult::IncompatibleSchema {
            checkpoint_version,
            current_version,
        } => {
            assert_eq!(checkpoint_version, 99);
            assert_eq!(current_version, CHECKPOINT_SCHEMA_VERSION);
        }
        other => panic!("Expected IncompatibleSchema, got {other:?}"),
    }
}

#[test]
fn test_resume_stale_checkpoint() {
    let cp = StateCheckpoint {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id: 0,
        state: OrchestratorState::Verifying,
        iteration: 1,
        transitions: vec![],
        created_at: "2026-01-01T00:00:00Z".into(),
        git_hash: Some("old_hash".into()),
        issue_id: "issue".into(),
    };

    match StateMachine::resume_from(&cp, Some("new_hash")) {
        ResumeResult::StaleCheckpoint {
            expected_hash,
            actual_hash,
        } => {
            assert_eq!(expected_hash, "new_hash");
            assert_eq!(actual_hash, "old_hash");
        }
        other => panic!("Expected StaleCheckpoint, got {other:?}"),
    }
}

#[test]
fn test_checkpoint_serde_roundtrip() {
    let cp = StateCheckpoint {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id: 5,
        state: OrchestratorState::Implementing,
        iteration: 3,
        transitions: vec![TransitionRecord {
            from: OrchestratorState::Verifying,
            to: OrchestratorState::Implementing,
            iteration: 2,
            elapsed_ms: 5000,
            reason: Some("retry after errors".into()),
        }],
        created_at: "2026-02-21T00:00:00Z".into(),
        git_hash: Some("abc123".into()),
        issue_id: "beefcake-xyz".into(),
    };

    let json = serde_json::to_string_pretty(&cp).unwrap();
    let restored: StateCheckpoint = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.schema_version, CHECKPOINT_SCHEMA_VERSION);
    assert_eq!(restored.state, OrchestratorState::Implementing);
    assert_eq!(restored.iteration, 3);
    assert_eq!(restored.transitions.len(), 1);
    assert_eq!(restored.issue_id, "beefcake-xyz");
}

#[test]
fn test_save_and_load_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join(".swarm-state-checkpoint.json");

    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();

    let cp = sm.checkpoint("test-issue", Some("deadbeef")).unwrap();
    save_checkpoint(&cp, &path);
    assert!(path.exists());

    let loaded = load_checkpoint(&path).unwrap();
    assert_eq!(loaded.state, OrchestratorState::Verifying);
    assert_eq!(loaded.iteration, 1);
    assert_eq!(loaded.issue_id, "test-issue");
    assert_eq!(loaded.git_hash.as_deref(), Some("deadbeef"));
}

#[test]
fn test_load_nonexistent_checkpoint() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no-such-file.json");
    assert!(load_checkpoint(&path).is_none());
}

#[test]
fn test_resume_no_git_hash_skips_staleness() {
    // When checkpoint has no git hash, staleness check is skipped
    let cp = StateCheckpoint {
        schema_version: CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id: 0,
        state: OrchestratorState::Implementing,
        iteration: 1,
        transitions: vec![],
        created_at: "2026-01-01T00:00:00Z".into(),
        git_hash: None,
        issue_id: "issue".into(),
    };

    // Even with a provided expected hash, no staleness error
    match StateMachine::resume_from(&cp, Some("any_hash")) {
        ResumeResult::Restored(sm) => {
            assert_eq!(sm.current(), OrchestratorState::Implementing);
        }
        other => panic!("Expected Restored, got {other:?}"),
    }
}

// ──────────────────────────────────────────────────────────────────────
// Budget / Timeout Tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_budget_config_defaults() {
    let config = BudgetConfig::default();
    assert_eq!(config.global_max_iterations, 10);

    // Implementing has both timeout and iteration limit
    let imp = config
        .budgets
        .get(&OrchestratorState::Implementing)
        .unwrap();
    assert_eq!(imp.timeout_ms, Some(45 * 60 * 1000));
    assert_eq!(imp.max_iterations, Some(6));

    // Verifying has timeout only
    let ver = config.budgets.get(&OrchestratorState::Verifying).unwrap();
    assert!(ver.timeout_ms.is_some());
    assert!(ver.max_iterations.is_none());

    // SelectingIssue has no budget
    assert!(config
        .budgets
        .get(&OrchestratorState::SelectingIssue)
        .is_none());
}

#[test]
fn test_budget_tracker_no_violation() {
    let mut tracker = BudgetTracker::with_defaults();
    tracker.on_state_entered(OrchestratorState::Planning);
    tracker.on_state_entered(OrchestratorState::Implementing);

    // Fresh entry — no violations
    assert!(tracker
        .check_budget(OrchestratorState::Implementing)
        .is_none());
    assert_eq!(tracker.entry_count(OrchestratorState::Implementing), 1);
    // total_iterations counts Planning entries only
    assert_eq!(tracker.total_iterations(), 1);
}

#[test]
fn test_budget_tracker_iteration_exhaustion() {
    let config = BudgetConfig {
        budgets: {
            let mut m = HashMap::new();
            m.insert(
                OrchestratorState::Implementing,
                StateBudget::iterations_only(2),
            );
            m
        },
        global_max_iterations: 100,
    };
    let mut tracker = BudgetTracker::new(config);

    // Enter state 3 times (limit is 2)
    tracker.on_state_entered(OrchestratorState::Implementing);
    assert!(tracker
        .check_budget(OrchestratorState::Implementing)
        .is_none());

    tracker.on_state_entered(OrchestratorState::Implementing);
    assert!(tracker
        .check_budget(OrchestratorState::Implementing)
        .is_none());

    tracker.on_state_entered(OrchestratorState::Implementing);
    match tracker.check_budget(OrchestratorState::Implementing) {
        Some(CancellationReason::BudgetExhausted { state, used, limit }) => {
            assert_eq!(state, OrchestratorState::Implementing);
            assert_eq!(used, 3);
            assert_eq!(limit, 2);
        }
        other => panic!("Expected BudgetExhausted, got {other:?}"),
    }
}

#[test]
fn test_budget_tracker_global_exhaustion() {
    let config = BudgetConfig {
        budgets: HashMap::new(),
        global_max_iterations: 3,
    };
    let mut tracker = BudgetTracker::new(config);

    // Simulate 3 iterations (Planning marks each iteration start)
    tracker.on_state_entered(OrchestratorState::Planning);
    tracker.on_state_entered(OrchestratorState::Implementing);
    tracker.on_state_entered(OrchestratorState::Planning);
    tracker.on_state_entered(OrchestratorState::Implementing);
    tracker.on_state_entered(OrchestratorState::Planning);
    assert!(tracker
        .check_budget(OrchestratorState::Implementing)
        .is_none());

    // 4th Planning entry exceeds global limit of 3
    tracker.on_state_entered(OrchestratorState::Planning);
    match tracker.check_budget(OrchestratorState::Planning) {
        Some(CancellationReason::GlobalBudgetExhausted {
            total_iterations,
            limit,
        }) => {
            assert_eq!(total_iterations, 4);
            assert_eq!(limit, 3);
        }
        other => panic!("Expected GlobalBudgetExhausted, got {other:?}"),
    }
}

#[test]
fn test_budget_tracker_remaining_iterations() {
    let config = BudgetConfig {
        budgets: {
            let mut m = HashMap::new();
            m.insert(
                OrchestratorState::Implementing,
                StateBudget::iterations_only(5),
            );
            m
        },
        global_max_iterations: 100,
    };
    let mut tracker = BudgetTracker::new(config);

    assert_eq!(
        tracker.remaining_iterations(OrchestratorState::Implementing),
        Some(5)
    );

    tracker.on_state_entered(OrchestratorState::Implementing);
    tracker.on_state_entered(OrchestratorState::Implementing);
    assert_eq!(
        tracker.remaining_iterations(OrchestratorState::Implementing),
        Some(3)
    );

    // State without configured budget returns None
    assert!(tracker
        .remaining_iterations(OrchestratorState::Verifying)
        .is_none());
}

#[test]
fn test_budget_tracker_unconfigured_state() {
    let tracker = BudgetTracker::with_defaults();
    // SelectingIssue has no budget — always OK (global check still runs)
    assert!(tracker
        .check_budget(OrchestratorState::SelectingIssue)
        .is_none());
}

#[test]
fn test_state_budget_constructors() {
    let full = StateBudget::new(Duration::from_secs(300), 5);
    assert_eq!(full.timeout_ms, Some(300_000));
    assert_eq!(full.max_iterations, Some(5));

    let timeout = StateBudget::timeout_only(Duration::from_secs(60));
    assert_eq!(timeout.timeout_ms, Some(60_000));
    assert!(timeout.max_iterations.is_none());

    let iters = StateBudget::iterations_only(10);
    assert!(iters.timeout_ms.is_none());
    assert_eq!(iters.max_iterations, Some(10));

    let unlimited = StateBudget::unlimited();
    assert!(unlimited.timeout_ms.is_none());
    assert!(unlimited.max_iterations.is_none());
}

#[test]
fn test_cancellation_reason_display() {
    let timeout = CancellationReason::Timeout {
        state: OrchestratorState::Implementing,
        elapsed_ms: 5000,
        limit_ms: 3000,
    };
    assert!(timeout.to_string().contains("Timeout"));
    assert!(timeout.to_string().contains("5000ms"));

    let budget = CancellationReason::BudgetExhausted {
        state: OrchestratorState::Implementing,
        used: 7,
        limit: 6,
    };
    assert!(budget.to_string().contains("7/6"));

    let global = CancellationReason::GlobalBudgetExhausted {
        total_iterations: 11,
        limit: 10,
    };
    assert!(global.to_string().contains("11/10"));

    let external = CancellationReason::External {
        reason: "operator signal".into(),
    };
    assert!(external.to_string().contains("operator signal"));
}

#[test]
fn test_cancellation_reason_serde_roundtrip() {
    let reasons = vec![
        CancellationReason::Timeout {
            state: OrchestratorState::Verifying,
            elapsed_ms: 12345,
            limit_ms: 10000,
        },
        CancellationReason::BudgetExhausted {
            state: OrchestratorState::Implementing,
            used: 7,
            limit: 6,
        },
        CancellationReason::GlobalBudgetExhausted {
            total_iterations: 11,
            limit: 10,
        },
        CancellationReason::External {
            reason: "test".into(),
        },
    ];

    for reason in &reasons {
        let json = serde_json::to_string(reason).unwrap();
        let restored: CancellationReason = serde_json::from_str(&json).unwrap();
        assert_eq!(&restored, reason);
    }
}

#[test]
fn test_budget_config_serde_roundtrip() {
    let config = BudgetConfig::default();
    let json = serde_json::to_string_pretty(&config).unwrap();
    let restored: BudgetConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.global_max_iterations, config.global_max_iterations);
    assert_eq!(restored.budgets.len(), config.budgets.len());
}

#[test]
fn test_budget_tracker_config_accessor() {
    let tracker = BudgetTracker::with_defaults();
    let config = tracker.config();
    assert_eq!(config.global_max_iterations, 10);
    assert!(config
        .budgets
        .contains_key(&OrchestratorState::Implementing));
}

// ──────────────────────────────────────────────────────────────────────
// Audit Report Tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_audit_report_happy_path() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Validating, None).unwrap();
    sm.advance(OrchestratorState::Merging, None).unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    let report = sm.audit_report();
    assert_eq!(report.final_state, OrchestratorState::Resolved);
    assert_eq!(report.transition_count, 7);
    assert_eq!(report.retry_count, 0);
    assert_eq!(report.escalation_count, 0);
    assert!(report.invariant_violations.is_empty());
}

#[test]
fn test_audit_report_with_retries() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    // Retry from verifying
    sm.advance(OrchestratorState::Implementing, Some("retry"))
        .unwrap();
    sm.set_iteration(2);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Validating, None).unwrap();
    // Retry from validating
    sm.advance(OrchestratorState::Implementing, Some("retry"))
        .unwrap();
    sm.set_iteration(3);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Merging, None).unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    let report = sm.audit_report();
    assert_eq!(report.retry_count, 2);
    assert_eq!(report.escalation_count, 0);
    assert!(report.invariant_violations.is_empty());
}

#[test]
fn test_audit_report_with_escalation() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Escalating, Some("stuck"))
        .unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(2);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Merging, None).unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    let report = sm.audit_report();
    assert_eq!(report.escalation_count, 1);
    assert!(report.invariant_violations.is_empty());
}

#[test]
fn test_audit_report_counts_state_visits() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(2);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Merging, None).unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    let report = sm.audit_report();
    assert_eq!(
        report
            .state_visit_counts
            .get(&OrchestratorState::Implementing),
        Some(&2)
    );
    assert_eq!(
        report.state_visit_counts.get(&OrchestratorState::Verifying),
        Some(&2)
    );
    assert_eq!(
        report.state_visit_counts.get(&OrchestratorState::Merging),
        Some(&1)
    );
}

#[test]
fn test_audit_report_display() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.fail("test").unwrap();

    let report = sm.audit_report();
    let display = report.to_string();
    assert!(display.contains("Audit Report"));
    assert!(display.contains("Failed"));
    assert!(display.contains("Invariants: all passed"));
}

#[test]
fn test_audit_report_serde_roundtrip() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.fail("test failure").unwrap();

    let report = sm.audit_report();
    let json = serde_json::to_string_pretty(&report).unwrap();
    let restored: AuditReport = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.final_state, OrchestratorState::Failed);
    assert_eq!(restored.transition_count, 3);
}

#[test]
fn test_export_transitions_json() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, Some("test"))
        .unwrap();

    let json = sm.export_transitions_json().unwrap();
    let parsed: Vec<TransitionRecord> = serde_json::from_str(&json).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].to, OrchestratorState::PreparingWorktree);
}

// ──────────────────────────────────────────────────────────────────────
// Invariant Tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_invariants_happy_path_clean() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Merging, None).unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    let violations = check_invariants(sm.transitions(), sm.current());
    assert!(violations.is_empty(), "Violations: {violations:?}");
}

#[test]
fn test_invariant_detects_illegal_transition() {
    // Manually construct an illegal log
    let transitions = vec![TransitionRecord {
        from: OrchestratorState::SelectingIssue,
        to: OrchestratorState::Merging, // Illegal skip
        iteration: 0,
        elapsed_ms: 0,
        reason: None,
    }];

    let violations = check_invariants(&transitions, OrchestratorState::Merging);
    assert!(violations.iter().any(|v| v.contains("INV-1")));
}

#[test]
fn test_invariant_detects_post_terminal_transition() {
    let transitions = vec![
        TransitionRecord {
            from: OrchestratorState::SelectingIssue,
            to: OrchestratorState::Failed,
            iteration: 0,
            elapsed_ms: 0,
            reason: None,
        },
        TransitionRecord {
            from: OrchestratorState::Failed,
            to: OrchestratorState::Implementing, // After terminal
            iteration: 1,
            elapsed_ms: 100,
            reason: None,
        },
    ];

    let violations = check_invariants(&transitions, OrchestratorState::Implementing);
    assert!(violations.iter().any(|v| v.contains("INV-2")));
}

#[test]
fn test_invariant_detects_chain_discontinuity() {
    let transitions = vec![
        TransitionRecord {
            from: OrchestratorState::SelectingIssue,
            to: OrchestratorState::PreparingWorktree,
            iteration: 0,
            elapsed_ms: 0,
            reason: None,
        },
        TransitionRecord {
            from: OrchestratorState::Implementing, // Discontinuity
            to: OrchestratorState::Verifying,
            iteration: 1,
            elapsed_ms: 100,
            reason: None,
        },
    ];

    let violations = check_invariants(&transitions, OrchestratorState::Verifying);
    assert!(violations.iter().any(|v| v.contains("INV-4")));
}

#[test]
fn test_invariant_detects_wrong_initial_state() {
    let transitions = vec![TransitionRecord {
        from: OrchestratorState::Implementing, // Wrong start
        to: OrchestratorState::Verifying,
        iteration: 1,
        elapsed_ms: 0,
        reason: None,
    }];

    let violations = check_invariants(&transitions, OrchestratorState::Verifying);
    assert!(violations.iter().any(|v| v.contains("INV-5")));
}

#[test]
fn test_invariant_detects_iteration_decrease() {
    let transitions = vec![
        TransitionRecord {
            from: OrchestratorState::SelectingIssue,
            to: OrchestratorState::PreparingWorktree,
            iteration: 5,
            elapsed_ms: 0,
            reason: None,
        },
        TransitionRecord {
            from: OrchestratorState::PreparingWorktree,
            to: OrchestratorState::Planning,
            iteration: 3, // Decreased
            elapsed_ms: 100,
            reason: None,
        },
    ];

    let violations = check_invariants(&transitions, OrchestratorState::Planning);
    assert!(violations.iter().any(|v| v.contains("INV-6")));
}

#[test]
fn test_invariant_detects_time_decrease() {
    let transitions = vec![
        TransitionRecord {
            from: OrchestratorState::SelectingIssue,
            to: OrchestratorState::PreparingWorktree,
            iteration: 0,
            elapsed_ms: 500,
            reason: None,
        },
        TransitionRecord {
            from: OrchestratorState::PreparingWorktree,
            to: OrchestratorState::Planning,
            iteration: 0,
            elapsed_ms: 200, // Decreased
            reason: None,
        },
    ];

    let violations = check_invariants(&transitions, OrchestratorState::Planning);
    assert!(violations.iter().any(|v| v.contains("INV-7")));
}

#[test]
fn test_invariants_empty_log() {
    let violations = check_invariants(&[], OrchestratorState::SelectingIssue);
    assert!(violations.is_empty());
}

// ──────────────────────────────────────────────────────────────────────
// Property-Style Tests — exhaustive/systematic scenario coverage
// ──────────────────────────────────────────────────────────────────────

/// All non-terminal states can transition to Failed.
#[test]
fn test_property_any_non_terminal_can_fail() {
    let non_terminal = [
        OrchestratorState::SelectingIssue,
        OrchestratorState::PreparingWorktree,
        OrchestratorState::Planning,
        OrchestratorState::Implementing,
        OrchestratorState::Verifying,
        OrchestratorState::Validating,
        OrchestratorState::Escalating,
        OrchestratorState::Merging,
    ];

    for state in non_terminal {
        assert!(
            is_legal_transition(state, OrchestratorState::Failed),
            "{state} → Failed should be legal"
        );
    }
}

/// Terminal states cannot transition to anything.
#[test]
fn test_property_terminal_states_absorbing() {
    let terminals = [OrchestratorState::Resolved, OrchestratorState::Failed];
    let all_states = [
        OrchestratorState::SelectingIssue,
        OrchestratorState::PreparingWorktree,
        OrchestratorState::Planning,
        OrchestratorState::Implementing,
        OrchestratorState::Verifying,
        OrchestratorState::Validating,
        OrchestratorState::Escalating,
        OrchestratorState::Merging,
        OrchestratorState::Resolved,
        OrchestratorState::Failed,
    ];

    for terminal in terminals {
        for target in all_states {
            assert!(
                !is_legal_transition(terminal, target),
                "{terminal} → {target} should be illegal (terminal is absorbing)"
            );
        }
    }
}

/// Every retry loop through the state machine is bounded by budget.
#[test]
fn test_property_retry_loop_bounded_by_budget() {
    let config = BudgetConfig {
        budgets: {
            let mut m = HashMap::new();
            m.insert(
                OrchestratorState::Implementing,
                StateBudget::iterations_only(3),
            );
            m
        },
        global_max_iterations: 100,
    };
    let mut tracker = BudgetTracker::new(config);
    let mut sm = StateMachine::new();

    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();

    let mut retries = 0u32;
    for iter in 1..=10 {
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        tracker.on_state_entered(OrchestratorState::Implementing);
        sm.set_iteration(iter);

        if let Some(_reason) = tracker.check_budget(OrchestratorState::Implementing) {
            sm.fail("budget exhausted").unwrap();
            break;
        }

        sm.advance(OrchestratorState::Verifying, None).unwrap();
        tracker.on_state_entered(OrchestratorState::Verifying);

        // Simulate failure: go back to implementing
        if !sm.is_terminal() {
            retries += 1;
        }
    }

    // Budget was 3, so we should have been stopped
    assert!(sm.is_terminal() || retries <= 3);
    let report = sm.audit_report();
    assert!(report.invariant_violations.is_empty());
}

/// Escalation always returns to Implementing (deterministic trigger).
#[test]
fn test_property_escalation_deterministic_reentry() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(1);
    sm.advance(OrchestratorState::Verifying, None).unwrap();

    // Escalate
    sm.advance(OrchestratorState::Escalating, Some("error repeat"))
        .unwrap();

    // The only legal transition from Escalating (besides Failed) is Implementing
    assert!(is_legal_transition(
        OrchestratorState::Escalating,
        OrchestratorState::Implementing
    ));

    // No other non-fail transitions from Escalating
    for state in [
        OrchestratorState::SelectingIssue,
        OrchestratorState::PreparingWorktree,
        OrchestratorState::Planning,
        OrchestratorState::Verifying,
        OrchestratorState::Validating,
        OrchestratorState::Escalating,
        OrchestratorState::Merging,
        OrchestratorState::Resolved,
    ] {
        assert!(
            !is_legal_transition(OrchestratorState::Escalating, state),
            "Escalating → {state} should be illegal"
        );
    }
}

/// Multiple escalations in a single run maintain invariants.
#[test]
fn test_property_multiple_escalations_maintain_invariants() {
    let mut sm = StateMachine::new();
    sm.advance(OrchestratorState::PreparingWorktree, None)
        .unwrap();
    sm.advance(OrchestratorState::Planning, None).unwrap();

    for iter in 1..=3 {
        sm.advance(OrchestratorState::Implementing, None).unwrap();
        sm.set_iteration(iter);
        sm.advance(OrchestratorState::Verifying, None).unwrap();
        sm.advance(OrchestratorState::Escalating, Some("stuck"))
            .unwrap();
    }

    // Final attempt succeeds
    sm.advance(OrchestratorState::Implementing, None).unwrap();
    sm.set_iteration(4);
    sm.advance(OrchestratorState::Verifying, None).unwrap();
    sm.advance(OrchestratorState::Merging, None).unwrap();
    sm.advance(OrchestratorState::Resolved, None).unwrap();

    let report = sm.audit_report();
    assert_eq!(report.escalation_count, 3);
    assert_eq!(report.retry_count, 0); // Escalations are not retries
    assert!(report.invariant_violations.is_empty());
}

/// Global budget caps total iterations (counted by Planning entries).
#[test]
fn test_property_global_budget_caps_all_states() {
    let config = BudgetConfig {
        budgets: HashMap::new(),
        global_max_iterations: 3,
    };
    let mut tracker = BudgetTracker::new(config);

    // Simulate 3 full iterations: Planning→Implementing→Verifying each
    for _ in 0..3 {
        tracker.on_state_entered(OrchestratorState::Planning);
        tracker.on_state_entered(OrchestratorState::Implementing);
        tracker.on_state_entered(OrchestratorState::Verifying);
    }
    assert!(tracker
        .check_budget(OrchestratorState::Implementing)
        .is_none());

    // 4th Planning entry pushes over the global limit
    tracker.on_state_entered(OrchestratorState::Planning);
    assert!(matches!(
        tracker.check_budget(OrchestratorState::Planning),
        Some(CancellationReason::GlobalBudgetExhausted { .. })
    ));
}
