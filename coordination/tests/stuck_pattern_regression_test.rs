//! Regression tests for stuck patterns in the escalation engine
//!
//! Validates that the escalation engine correctly detects and handles
//! stuck states: no-change loops, budget exhaustion, error oscillation,
//! and error count plateaus.

use coordination::escalation::engine::{EscalationConfig, EscalationEngine, SuggestedAction};
use coordination::escalation::friction::{FrictionDetector, FrictionKind};
use coordination::escalation::state::{EscalationReason, EscalationState, SwarmTier};
use coordination::feedback::error_parser::{ErrorCategory, ParsedError};
use coordination::verifier::report::{GateOutcome, GateResult, VerifierReport};
use std::time::Duration;

fn failing_report(categories: Vec<ErrorCategory>) -> VerifierReport {
    let errors: Vec<ParsedError> = categories
        .iter()
        .map(|cat| ParsedError {
            category: *cat,
            code: None,
            message: format!("{} error", cat),
            file: Some("src/main.rs".to_string()),
            line: Some(1),
            column: Some(1),
            suggestion: None,
            rendered: String::new(),
            labels: vec![],
        })
        .collect();
    let mut report = VerifierReport::new("/tmp/test".to_string());
    report.add_gate(GateResult {
        gate: "check".to_string(),
        outcome: GateOutcome::Failed,
        duration_ms: 100,
        exit_code: Some(1),
        error_count: errors.len(),
        warning_count: 0,
        errors,
        stderr_excerpt: None,
    });
    report.finalize(Duration::from_millis(100));
    report
}

fn empty_report() -> VerifierReport {
    VerifierReport::new("/tmp/t".to_string())
}

fn stuck_config() -> EscalationConfig {
    EscalationConfig {
        no_change_threshold: 3,
        repeat_threshold: 100,
        failure_threshold: 100,
        ..Default::default()
    }
}

#[test]
fn test_worker_no_change_stuck_escalates_to_human() {
    let engine = EscalationEngine::with_config(stuck_config());
    let mut state = EscalationState::new("stuck-w1");
    for _ in 0..3 {
        state.record_no_change();
    }
    let d = engine.decide(&mut state, &failing_report(vec![ErrorCategory::Other]));
    assert!(d.stuck, "Worker should be stuck after 3 no-changes");
    assert_eq!(d.target_tier, SwarmTier::Human);
}

#[test]
fn test_council_no_change_stuck_escalates_to_human() {
    let engine = EscalationEngine::with_config(stuck_config());
    let mut state = EscalationState::new("stuck-c1");
    let cats = [
        ErrorCategory::TypeMismatch,
        ErrorCategory::BorrowChecker,
        ErrorCategory::TraitBound,
        ErrorCategory::ImportResolution,
    ];
    for cat in &cats {
        state.record_iteration(vec![*cat], 1, false);
    }
    state.record_escalation(
        SwarmTier::Council,
        EscalationReason::BudgetExhausted {
            tier: SwarmTier::Worker,
        },
    );
    for _ in 0..3 {
        state.record_no_change();
    }
    let d = engine.decide(&mut state, &failing_report(vec![ErrorCategory::Other]));
    assert!(d.stuck, "Council should be stuck after 3 no-changes");
    assert_eq!(d.target_tier, SwarmTier::Human);
}

#[test]
fn test_no_change_reset_prevents_stuck() {
    let engine = EscalationEngine::with_config(stuck_config());
    let mut state = EscalationState::new("stuck-r1");
    state.record_no_change();
    state.record_no_change();
    state.reset_no_change();
    state.record_no_change();
    state.record_no_change();
    let d = engine.decide(&mut state, &failing_report(vec![ErrorCategory::Other]));
    assert!(!d.stuck, "Reset should prevent stuck detection");
}

#[test]
fn test_oscillation_detected_as_friction() {
    let mut state = EscalationState::new("osc-1");
    for _ in 0..2 {
        state.record_iteration(vec![ErrorCategory::Lifetime], 2, false);
        state.record_iteration(vec![ErrorCategory::TypeMismatch], 2, false);
    }
    let signals = FrictionDetector::detect(&state, &empty_report());
    assert!(
        signals
            .iter()
            .any(|s| matches!(s.kind, FrictionKind::ErrorOscillation { .. })),
        "Should detect error oscillation, got: {:?}",
        signals.iter().map(|s| &s.kind).collect::<Vec<_>>()
    );
}

#[test]
fn test_plateau_detected_as_friction() {
    let mut state = EscalationState::new("plat-1");
    for _ in 0..4 {
        state.record_iteration(vec![ErrorCategory::BorrowChecker], 5, false);
    }
    let signals = FrictionDetector::detect(&state, &empty_report());
    assert!(
        signals
            .iter()
            .any(|s| matches!(s.kind, FrictionKind::ErrorCountPlateau { .. })),
        "Should detect error count plateau, got: {:?}",
        signals.iter().map(|s| &s.kind).collect::<Vec<_>>()
    );
}

#[test]
fn test_worker_budget_then_council_budget_stuck() {
    let config = EscalationConfig {
        repeat_threshold: 100,
        failure_threshold: 100,
        no_change_threshold: 100,
        ..Default::default()
    };
    let engine = EscalationEngine::with_config(config);
    let mut state = EscalationState::new("exhaust-1");
    let worker_cats = [
        ErrorCategory::TypeMismatch,
        ErrorCategory::BorrowChecker,
        ErrorCategory::TraitBound,
        ErrorCategory::ImportResolution,
    ];
    for cat in &worker_cats {
        state.record_iteration(vec![*cat], 1, false);
    }
    state.record_escalation(
        SwarmTier::Council,
        EscalationReason::BudgetExhausted {
            tier: SwarmTier::Worker,
        },
    );
    for _ in 0..6 {
        state.record_iteration(vec![ErrorCategory::Other], 1, false);
    }
    let d = engine.decide(&mut state, &failing_report(vec![ErrorCategory::Other]));
    assert!(d.stuck, "Should be stuck after exhausting both budgets");
    assert!(
        matches!(d.action, SuggestedAction::FlagForHuman { .. }),
        "Action should be FlagForHuman, got: {:?}",
        d.action
    );
}
