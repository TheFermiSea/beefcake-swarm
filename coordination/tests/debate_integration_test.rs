//! Mocked debate integration test — exercises the full debate loop
//! with deterministic mock agents (no LLM calls).
//!
//! Covers: orchestrator ↔ consensus ↔ guardrails ↔ critique ↔ persistence
//! running together in a single pass.

use coordination::debate::{
    CheckpointManager, ConsensusCheck, DebateCheckpoint, DebateConfig, DebateOrchestrator,
    DebatePhase, GuardrailConfig, NextAction, PatchCritique, Verdict,
};

use coordination::debate::critique::{CritiqueCategory, CritiqueItem};
use coordination::debate::orchestrator::{CoderOutput, ReviewerOutput};

/// Helper: simulate a coder producing output for a round.
fn mock_coder_output(round: u32) -> CoderOutput {
    CoderOutput {
        code: format!("fn solution_v{}() {{ /* impl */ }}", round),
        files_changed: vec![format!("src/solution_v{}.rs", round)],
        explanation: format!("Implementation attempt {}", round),
    }
}

/// Helper: simulate a reviewer approving.
fn mock_approval_review() -> ReviewerOutput {
    ReviewerOutput {
        check: ConsensusCheck {
            verdict: Verdict::Approve,
            confidence: 0.95,
            blocking_issues: vec![],
            suggestions: vec!["Consider adding docs".to_string()],
            approach_aligned: true,
        },
        summary: "Approved — code meets requirements".to_string(),
    }
}

/// Helper: simulate a reviewer requesting changes.
fn mock_rejection_review() -> ReviewerOutput {
    ReviewerOutput {
        check: ConsensusCheck {
            verdict: Verdict::RequestChanges,
            confidence: 0.85,
            blocking_issues: vec!["Missing error handling".to_string()],
            suggestions: vec!["Use &str for inputs".to_string()],
            approach_aligned: true,
        },
        summary: "Changes needed — missing error handling".to_string(),
    }
}

// ── Single-round approval (happy path) ─────────────────────────────

#[test]
fn test_debate_single_round_approval() {
    let mut orch = DebateOrchestrator::new("d-int-1", "issue-1", "Implement error handling");
    orch.start().unwrap();

    assert_eq!(orch.next_action(), NextAction::AwaitCoder);
    assert_eq!(orch.session().phase, DebatePhase::CoderTurn);

    orch.submit_code(mock_coder_output(1)).unwrap();
    assert_eq!(orch.next_action(), NextAction::AwaitReviewer);
    assert_eq!(orch.session().phase, DebatePhase::ReviewerTurn);

    let action = orch.submit_review(mock_approval_review()).unwrap();
    assert_eq!(action, NextAction::Complete);

    let outcome = orch.outcome().unwrap();
    assert!(outcome.is_success());
    assert!(outcome.consensus_reached);
    assert_eq!(outcome.rounds_completed, 1);
    assert_eq!(outcome.terminal_phase, DebatePhase::Resolved);
}

// ── Multi-round convergence ────────────────────────────────────────

#[test]
fn test_debate_multi_round_convergence() {
    let config = DebateConfig {
        max_rounds: 5,
        guardrails: GuardrailConfig {
            max_rounds: 5,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut orch = DebateOrchestrator::with_config("d-int-2", "issue-2", "Implement retry", config);
    orch.start().unwrap();

    // Round 1: coder submits, reviewer rejects
    orch.submit_code(mock_coder_output(1)).unwrap();
    let action = orch.submit_review(mock_rejection_review()).unwrap();
    assert_eq!(action, NextAction::AwaitCoder);
    assert_eq!(orch.session().current_round, 2);

    // Round 2: coder fixes, reviewer rejects again
    orch.submit_code(mock_coder_output(2)).unwrap();
    let action = orch.submit_review(mock_rejection_review()).unwrap();
    assert_eq!(action, NextAction::AwaitCoder);
    assert_eq!(orch.session().current_round, 3);

    // Round 3: coder fixes, reviewer approves
    orch.submit_code(mock_coder_output(3)).unwrap();
    let action = orch.submit_review(mock_approval_review()).unwrap();
    assert_eq!(action, NextAction::Complete);

    let outcome = orch.outcome().unwrap();
    assert!(outcome.is_success());
    assert!(outcome.consensus_reached);
    assert_eq!(outcome.rounds_completed, 3);
}

// ── Max rounds → deadlock ──────────────────────────────────────────

#[test]
fn test_debate_deadlock_after_max_rounds() {
    let config = DebateConfig {
        max_rounds: 2,
        guardrails: GuardrailConfig {
            max_rounds: 2,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut orch =
        DebateOrchestrator::with_config("d-int-3", "issue-3", "Complex refactor", config);
    orch.start().unwrap();

    // Round 1
    orch.submit_code(mock_coder_output(1)).unwrap();
    let action = orch.submit_review(mock_rejection_review()).unwrap();
    assert_eq!(action, NextAction::AwaitCoder);

    // Round 2 (last round)
    orch.submit_code(mock_coder_output(2)).unwrap();
    let action = orch.submit_review(mock_rejection_review()).unwrap();
    assert_eq!(action, NextAction::Complete);

    let outcome = orch.outcome().unwrap();
    assert!(!outcome.is_success());
    assert!(outcome.needs_escalation());
    assert_eq!(outcome.terminal_phase, DebatePhase::Deadlocked);
}

// ── Abort mid-debate ───────────────────────────────────────────────

#[test]
fn test_debate_abort() {
    let mut orch = DebateOrchestrator::new("d-int-4", "issue-4", "Feature implementation");
    orch.start().unwrap();
    orch.submit_code(mock_coder_output(1)).unwrap();

    // Abort during reviewer turn
    orch.abort("User requested cancellation").unwrap();
    assert_eq!(orch.session().phase, DebatePhase::Aborted);
    assert!(orch.is_complete());

    let outcome = orch.outcome().unwrap();
    assert!(!outcome.is_success());
    assert!(!outcome.needs_escalation());
}

// ── Round records accumulate correctly ─────────────────────────────

#[test]
fn test_debate_round_records() {
    let config = DebateConfig {
        max_rounds: 5,
        guardrails: GuardrailConfig {
            max_rounds: 5,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut orch = DebateOrchestrator::with_config("d-int-5", "issue-5", "Track rounds", config);
    orch.start().unwrap();

    // Round 1
    orch.submit_code(mock_coder_output(1)).unwrap();
    orch.submit_review(mock_rejection_review()).unwrap();

    // Round 2
    orch.submit_code(mock_coder_output(2)).unwrap();
    orch.submit_review(mock_approval_review()).unwrap();

    let session = orch.session();
    assert_eq!(session.rounds.len(), 2);

    // Check round 1 record
    let r1 = &session.rounds[0];
    assert_eq!(r1.round, 1);
    assert!(!r1.approved);
    assert!(!r1.issues.is_empty());

    // Check round 2 record
    let r2 = &session.rounds[1];
    assert_eq!(r2.round, 2);
    assert!(r2.approved);
}

// ── Consensus protocol standalone ──────────────────────────────────

#[test]
fn test_consensus_protocol_integration() {
    use coordination::debate::consensus::{ConsensusOutcome, ConsensusProtocol};

    let protocol = ConsensusProtocol::default();

    // High confidence approve → consensus
    let checks = vec![ConsensusCheck {
        verdict: Verdict::Approve,
        confidence: 0.9,
        blocking_issues: vec![],
        suggestions: vec![],
        approach_aligned: true,
    }];
    assert_eq!(protocol.evaluate(&checks), ConsensusOutcome::Reached);

    // Request changes with blockers → progressing
    let checks = vec![ConsensusCheck {
        verdict: Verdict::RequestChanges,
        confidence: 0.8,
        blocking_issues: vec!["bug".to_string()],
        suggestions: vec![],
        approach_aligned: true,
    }];
    assert_eq!(protocol.evaluate(&checks), ConsensusOutcome::Progressing);
}

// ── Guardrails standalone ──────────────────────────────────────────

#[test]
fn test_guardrails_integration() {
    use coordination::debate::guardrails::{DeadlockOutcome, GuardrailEngine};

    let config = GuardrailConfig {
        max_rounds: 3,
        ..Default::default()
    };
    let engine = GuardrailEngine::new(config);

    // Under limit → continue
    let session = coordination::debate::DebateSession::new("d-g", "i-g", "diff", 5);
    let outcome = engine.evaluate(&session, &[], 0);
    assert_eq!(outcome, DeadlockOutcome::Continue);

    // At limit → deadlock
    let mut session = coordination::debate::DebateSession::new("d-g2", "i-g2", "diff", 5);
    session.start().unwrap();
    session
        .transition(DebatePhase::ReviewerTurn, "code")
        .unwrap();
    session
        .transition(DebatePhase::CoderTurn, "revise")
        .unwrap();
    session
        .transition(DebatePhase::ReviewerTurn, "code")
        .unwrap();
    session
        .transition(DebatePhase::CoderTurn, "revise")
        .unwrap();
    // current_round is now 3
    let outcome = engine.evaluate(&session, &[], 0);
    assert!(matches!(outcome, DeadlockOutcome::MaxRoundsExceeded { .. }));
}

// ── Critique generation integration ────────────────────────────────

#[test]
fn test_critique_repair_instructions() {
    use coordination::debate::critique::{format_critique_for_coder, generate_repair_instructions};

    let mut critique = PatchCritique::new(1, "Several issues found");
    critique.add_item(
        CritiqueItem::blocking(
            CritiqueCategory::BorrowChecker,
            "Dangling reference in closure",
        )
        .in_file("src/lib.rs")
        .at_lines(42, 50)
        .with_fix("Use Arc<Mutex<T>> for shared ownership"),
    );
    critique.add_item(
        CritiqueItem::warning(
            CritiqueCategory::Performance,
            "Unnecessary clone in hot path",
        )
        .in_file("src/lib.rs")
        .at_lines(100, 105),
    );
    critique.add_item(CritiqueItem {
        severity: coordination::debate::CritiqueSeverity::Suggestion,
        category: CritiqueCategory::Style,
        file: None,
        line_range: None,
        description: "Consider using if-let instead of match".to_string(),
        suggested_fix: None,
    });

    // Generate repair instructions
    let instructions = generate_repair_instructions(&critique);
    assert_eq!(instructions.len(), 3);
    assert_eq!(instructions[0].priority, 1); // blocking first (1-indexed)
    assert!(instructions[0].instruction.contains("Dangling reference"));

    // Format for coder
    let formatted = format_critique_for_coder(&critique);
    assert!(formatted.contains("Blocking Issues"));
    assert!(formatted.contains("src/lib.rs"));
    assert!(formatted.contains("Dangling reference"));
}

// ── Persistence checkpoint round-trip ──────────────────────────────

#[test]
fn test_persistence_checkpoint_roundtrip() {
    let config = DebateConfig {
        max_rounds: 5,
        guardrails: GuardrailConfig {
            max_rounds: 5,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut orch =
        DebateOrchestrator::with_config("d-int-cp", "issue-cp", "Checkpointable", config);
    orch.start().unwrap();
    orch.submit_code(mock_coder_output(1)).unwrap();
    orch.submit_review(mock_rejection_review()).unwrap();

    // Create checkpoint
    let checkpoint = DebateCheckpoint::new(
        orch.session(),
        orch.checks(),
        &[], // no critiques in this test
        0,
        "mid-debate checkpoint",
        1,
    );
    let json = serde_json::to_string(&checkpoint).unwrap();

    // Restore from JSON
    let (restored, status) = CheckpointManager::restore(&json).unwrap();

    assert!(status.can_resume());
    assert_eq!(
        restored.session.current_round,
        checkpoint.session.current_round
    );
    assert_eq!(restored.session.phase, checkpoint.session.phase);
    assert_eq!(
        restored.session.rounds.len(),
        checkpoint.session.rounds.len()
    );
}

// ── Full loop with all subsystems ──────────────────────────────────

#[test]
fn test_full_debate_loop_all_subsystems() {
    let config = DebateConfig {
        max_rounds: 4,
        guardrails: GuardrailConfig {
            max_rounds: 4,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut orch =
        DebateOrchestrator::with_config("d-int-full", "issue-full", "Implement feature", config);
    orch.start().unwrap();

    // Round 1: reject
    orch.submit_code(mock_coder_output(1)).unwrap();
    let action = orch.submit_review(mock_rejection_review()).unwrap();
    assert_eq!(action, NextAction::AwaitCoder);

    // Checkpoint after round 1
    let cp1 = DebateCheckpoint::new(orch.session(), orch.checks(), &[], 0, "round 1 done", 1);
    let cp1_json = serde_json::to_string(&cp1).unwrap();
    assert!(!cp1_json.is_empty());

    // Round 2: reject again
    orch.submit_code(mock_coder_output(2)).unwrap();
    orch.submit_review(mock_rejection_review()).unwrap();

    // Round 3: approve
    orch.submit_code(mock_coder_output(3)).unwrap();
    let action = orch.submit_review(mock_approval_review()).unwrap();
    assert_eq!(action, NextAction::Complete);

    // Verify outcome
    let outcome = orch.outcome().unwrap();
    assert!(outcome.is_success());
    assert!(outcome.consensus_reached);
    assert_eq!(outcome.rounds_completed, 3);
    assert_eq!(outcome.terminal_phase, DebatePhase::Resolved);

    // Verify session state
    let session = orch.session();
    assert_eq!(session.phase, DebatePhase::Resolved);
    assert_eq!(session.rounds.len(), 3);
    assert!(!session.rounds[0].approved);
    assert!(!session.rounds[1].approved);
    assert!(session.rounds[2].approved);
}
