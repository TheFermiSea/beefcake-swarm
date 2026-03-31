//! Phase 0c integration tests: verify the orchestrator module split
//! (Slate Phase 0a) and Send-safety refactor (Phase 0b) preserved all
//! public API contracts.
//!
//! These tests import through the crate's public interface, NOT through
//! `super::*`, validating that re-exports from submodules work correctly.

use std::fs;
use std::path::Path;

use coordination::router::classifier::ErrorCategory;
use coordination::verifier::report::{ValidatorFeedback, ValidatorIssueType};
use coordination::{SwarmTier, WorkPacket};
use swarm_agents::orchestrator;
use swarm_agents::orchestrator::dispatch::{
    format_compact_task_prompt, format_task_prompt, route_to_coder, CoderRoute,
};

// ── Test helpers ─────────────────────────────────────────────────────

fn make_packet(objective: &str, iteration: u32) -> WorkPacket {
    WorkPacket {
        bead_id: "test-refactor".into(),
        branch: "swarm/refactor-test".into(),
        checkpoint: "abc123".into(),
        objective: objective.into(),
        files_touched: vec![],
        key_symbols: vec![],
        file_contexts: vec![],
        verification_gates: vec![],
        failure_signals: vec![],
        constraints: vec![],
        iteration,
        target_tier: SwarmTier::Worker,
        escalation_reason: None,
        error_history: vec![],
        previous_attempts: vec![],
        relevant_heuristics: vec![],
        relevant_playbooks: vec![],
        decisions: vec![],
        generated_at: chrono::Utc::now(),
        max_patch_loc: 200,
        iteration_deltas: vec![],
        delegation_chain: vec![],
        skill_hints: vec![],
        replay_hints: vec![],
        validator_feedback: vec![],
        change_contract: None,
        repo_map: None,
        failed_approach_summary: None,
    }
}

fn init_git_repo(dir: &Path) {
    std::process::Command::new("git")
        .args(["init"])
        .current_dir(dir)
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["config", "user.email", "test@test.com"])
        .current_dir(dir)
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["config", "user.name", "Test"])
        .current_dir(dir)
        .output()
        .unwrap();
    fs::write(dir.join("README.md"), "# test\n").unwrap();
    std::process::Command::new("git")
        .args(["add", "."])
        .current_dir(dir)
        .output()
        .unwrap();
    std::process::Command::new("git")
        .args(["commit", "-m", "init"])
        .current_dir(dir)
        .output()
        .unwrap();
}

// ── 1. Re-export integrity ───────────────────────────────────────────
//
// These tests verify that items previously in the monolithic orchestrator.rs
// are accessible through the same public paths after the split.

#[test]
fn reexport_coder_route_variants() {
    // CoderRoute must be accessible through both paths
    let _general: orchestrator::CoderRoute = orchestrator::CoderRoute::GeneralCoder;
    let _rust: CoderRoute = CoderRoute::RustCoder;
    // Variants must be Eq + Debug (used in assertions elsewhere)
    assert_eq!(
        orchestrator::CoderRoute::GeneralCoder,
        CoderRoute::GeneralCoder
    );
}

#[test]
fn reexport_route_to_coder_function() {
    // Must be callable through the re-export path
    let result = orchestrator::route_to_coder(&[]);
    assert_eq!(result, CoderRoute::GeneralCoder);
}

#[test]
fn reexport_format_functions() {
    let packet = make_packet("Test objective", 1);
    // Both prompt formatters must be accessible through re-export
    let prompt1 = orchestrator::format_task_prompt(&packet);
    let prompt2 = format_task_prompt(&packet);
    assert_eq!(prompt1, prompt2, "re-export and direct import must agree");
}

#[test]
fn reexport_cancel_msg_constant() {
    // CANCEL_MSG is used by main.rs — must remain accessible
    assert!(
        !orchestrator::CANCEL_MSG.is_empty(),
        "CANCEL_MSG must be non-empty"
    );
    assert!(
        orchestrator::CANCEL_MSG.contains("cancel"),
        "CANCEL_MSG must mention cancellation"
    );
}

// ── 2. Cross-module routing contracts ────────────────────────────────

#[test]
fn routing_rust_errors_to_rust_coder() {
    let rust_categories: Vec<ErrorCategory> = vec![
        ErrorCategory::BorrowChecker,
        ErrorCategory::Lifetime,
        ErrorCategory::TraitBound,
        ErrorCategory::Async,
    ];
    for cat in &rust_categories {
        assert_eq!(
            route_to_coder(std::slice::from_ref(cat)),
            CoderRoute::RustCoder,
            "{cat:?} should route to RustCoder"
        );
    }
}

#[test]
fn routing_general_errors_to_general_coder() {
    let general_categories: Vec<ErrorCategory> = vec![
        ErrorCategory::ImportResolution,
        ErrorCategory::Syntax,
        ErrorCategory::Macro,
        ErrorCategory::Other,
    ];
    for cat in &general_categories {
        assert_eq!(
            route_to_coder(std::slice::from_ref(cat)),
            CoderRoute::GeneralCoder,
            "{cat:?} should route to GeneralCoder"
        );
    }
}

#[test]
fn routing_empty_errors_to_general() {
    assert_eq!(route_to_coder(&[]), CoderRoute::GeneralCoder);
}

// ── 3. Prompt formatting contracts ───────────────────────────────────

#[test]
fn format_task_prompt_includes_objective_and_branch() {
    let packet = make_packet("Fix borrow checker error in parser", 3);
    let prompt = format_task_prompt(&packet);

    assert!(
        prompt.contains("Fix borrow checker error in parser"),
        "prompt must include the objective"
    );
    assert!(
        prompt.contains("swarm/refactor-test"),
        "prompt must include the branch"
    );
    assert!(
        prompt.contains("200 LOC"),
        "prompt must include the LOC budget"
    );
}

#[test]
fn format_task_prompt_includes_validator_feedback() {
    let mut packet = make_packet("Fix the bug", 2);
    packet.validator_feedback = vec![ValidatorFeedback {
        file: Some("src/lib.rs".into()),
        line_range: Some((10, 15)),
        issue_type: ValidatorIssueType::LogicError,
        description: "Off-by-one in range check".into(),
        suggested_fix: Some("Use <= instead of <".into()),
        source_model: None,
    }];

    let prompt = format_task_prompt(&packet);
    assert!(
        prompt.contains("Off-by-one"),
        "prompt must include validator feedback description"
    );
    assert!(
        prompt.contains("src/lib.rs"),
        "prompt must include feedback file path"
    );
}

#[test]
fn format_compact_task_prompt_includes_essentials() {
    let dir = tempfile::tempdir().unwrap();
    let packet = make_packet("Refactor module structure", 1);
    let prompt = format_compact_task_prompt(&packet, dir.path());

    assert!(
        prompt.contains("Refactor module structure"),
        "compact prompt must include objective"
    );
}

// ── 4. Session resume round-trip ─────────────────────────────────────

#[test]
fn check_for_resume_returns_none_without_file() {
    let dir = tempfile::tempdir().unwrap();
    assert!(
        orchestrator::check_for_resume(dir.path()).is_none(),
        "no resume file → None"
    );
}

#[test]
fn check_for_resume_deserializes_valid_file() {
    let dir = tempfile::tempdir().unwrap();
    let resume_data = serde_json::json!({
        "issue": {
            "id": "beads-test-001",
            "title": "Test issue for resume",
            "status": "in_progress",
            "priority": 2,
            "issue_type": "task",
            "labels": [],
            "description": null
        },
        "worktree_path": "/tmp/beefcake-wt/beads-test-001",
        "iteration": 3,
        "escalation_summary": "Worker tier, no escalation",
        "current_tier": "Worker",
        "total_iterations": 3,
        "saved_at": "2026-03-13T12:00:00Z"
    });
    fs::write(
        dir.path().join(".swarm-resume.json"),
        serde_json::to_string_pretty(&resume_data).unwrap(),
    )
    .unwrap();

    let resume = orchestrator::check_for_resume(dir.path());
    assert!(resume.is_some(), "valid resume file should parse");
    let resume = resume.unwrap();
    assert_eq!(resume.issue.id, "beads-test-001");
    assert_eq!(resume.iteration, 3);
    assert_eq!(resume.worktree_path, "/tmp/beefcake-wt/beads-test-001");
}

// ── 5. Scaffold fallback ─────────────────────────────────────────────

#[test]
fn scaffold_fallback_creates_doc_for_doc_task() {
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(dir.path());

    let created = orchestrator::helpers::try_scaffold_fallback(
        dir.path(),
        "beads-doc-001",
        "Architecture RFC for swarm redesign",
        "Write an RFC describing the new architecture",
        5,
    );

    assert!(created, "doc task should produce a scaffold");

    // Verify the file was created
    let expected_path = dir
        .path()
        .join("docs/architecture-rfc-for-swarm-redesign.md");
    assert!(
        expected_path.exists(),
        "scaffold file should exist at {expected_path:?}"
    );

    let content = fs::read_to_string(&expected_path).unwrap();
    assert!(content.contains("Architecture RFC for swarm redesign"));
    assert!(content.contains("beads-doc-001"));
    assert!(content.contains("iteration 5"));
}

#[test]
fn scaffold_fallback_skips_non_doc_task() {
    let dir = tempfile::tempdir().unwrap();
    init_git_repo(dir.path());

    let created = orchestrator::helpers::try_scaffold_fallback(
        dir.path(),
        "beads-code-001",
        "Fix type mismatch in count()",
        "The return type should be u32 not &str",
        2,
    );

    assert!(!created, "non-doc task should not produce a scaffold");
}

// ── 6. Send-safety (compile-time) ────────────────────────────────────
//
// The Send assertion in mod.rs (_assert_process_issue_core_is_send) is a
// compile-time check. If this file compiles, the assertion holds. This
// test simply documents that the contract exists.

#[test]
fn send_safety_assertion_compiles() {
    // This test is a documentation marker. The actual Send assertion lives
    // in orchestrator/mod.rs and is checked at compile time. If you can run
    // this test, process_issue_core is Send.
}
