//! Escalation Engine — Deterministic decision-making for tier routing
//!
//! Consumes VerifierReports and EscalationState to produce EscalationDecisions.
//! All decisions are deterministic — no LLM calls in this module.

use crate::escalation::friction::{FrictionDetector, FrictionKind, FrictionSeverity};
use crate::escalation::state::{EscalationReason, EscalationState, SwarmTier};
use crate::feedback::error_parser::ErrorCategory;
use crate::verifier::report::VerifierReport;
use serde::{Deserialize, Serialize};

/// Decision produced by the Escalation Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationDecision {
    /// Which tier should handle the next iteration
    pub target_tier: SwarmTier,
    /// Whether this is an escalation (tier changed)
    pub escalated: bool,
    /// Reason for the decision
    pub reason: String,
    /// Whether the issue is resolved (all-green)
    pub resolved: bool,
    /// Whether the issue is stuck (needs human intervention)
    pub stuck: bool,
    /// Whether adversary review should be triggered
    pub needs_review: bool,
    /// Suggested action for the target tier
    pub action: SuggestedAction,
}

/// Suggested action for the next tier
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SuggestedAction {
    /// Continue implementing (standard compile-fix loop)
    Continue,
    /// Produce a repair plan for the Implementer
    RepairPlan,
    /// Provide architectural guidance
    ArchitecturalGuidance,
    /// Review code before close
    AdversaryReview,
    /// Create blocking beads issue for human
    FlagForHuman { reason: String },
}

/// Configuration for the Escalation Engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationConfig {
    /// Error category repeat threshold to trigger Worker → Council
    pub repeat_threshold: u32,
    /// Total failure threshold to trigger Worker → Council
    pub failure_threshold: u32,
    /// File count threshold for Council escalation
    pub multi_file_threshold: usize,
    /// Whether to require adversary review before close
    pub require_adversary_review: bool,
    /// Consecutive no-change iterations before marking stuck
    pub no_change_threshold: u32,
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            repeat_threshold: 2,
            failure_threshold: 3,
            multi_file_threshold: 8,
            require_adversary_review: true,
            no_change_threshold: 3,
        }
    }
}

/// The Escalation Engine — deterministic state machine
pub struct EscalationEngine {
    config: EscalationConfig,
}

impl EscalationEngine {
    /// Create a new engine with default config
    pub fn new() -> Self {
        Self {
            config: EscalationConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: EscalationConfig) -> Self {
        Self { config }
    }

    /// Process a verifier report and produce a decision
    ///
    /// This is the core decision function. It takes the current escalation state
    /// and a fresh verifier report, and deterministically decides:
    /// 1. Whether to escalate
    /// 2. Which tier should handle the next iteration
    /// 3. What action the tier should take
    pub fn decide(
        &self,
        state: &mut EscalationState,
        report: &VerifierReport,
    ) -> EscalationDecision {
        // Extract error categories from the report
        let error_categories: Vec<ErrorCategory> = report.unique_error_categories();
        let error_count = report.gates.iter().map(|g| g.error_count).sum::<usize>();

        // Record this iteration in state
        state.record_iteration(error_categories, error_count, report.all_green);

        // Check: All green → success path
        if report.all_green {
            return self.success_decision(state);
        }

        // Check: Are we at the Worker tier?
        if state.current_tier == SwarmTier::Worker {
            return self.decide_at_worker(state, report);
        }

        // Check: Are we at the Council tier?
        if state.current_tier == SwarmTier::Council {
            return self.decide_at_council(state, report);
        }

        // Shouldn't reach here, but handle gracefully
        EscalationDecision {
            target_tier: state.current_tier,
            escalated: false,
            reason: "Unknown state".to_string(),
            resolved: false,
            stuck: true,
            needs_review: false,
            action: SuggestedAction::FlagForHuman {
                reason: "Unexpected escalation state".to_string(),
            },
        }
    }

    /// Decision when verification passes (all green)
    fn success_decision(&self, state: &EscalationState) -> EscalationDecision {
        if self.config.require_adversary_review
            && state.remaining_consultations(SwarmTier::Council) > 0
        {
            EscalationDecision {
                target_tier: SwarmTier::Council,
                escalated: state.current_tier != SwarmTier::Council,
                reason: "All gates passed — sending for council review".to_string(),
                resolved: true,
                stuck: false,
                needs_review: true,
                action: SuggestedAction::AdversaryReview,
            }
        } else {
            EscalationDecision {
                target_tier: state.current_tier,
                escalated: false,
                reason: "All gates passed — ready to close".to_string(),
                resolved: true,
                stuck: false,
                needs_review: false,
                action: SuggestedAction::Continue,
            }
        }
    }

    /// Decision-making at the Worker tier
    fn decide_at_worker(
        &self,
        state: &mut EscalationState,
        report: &VerifierReport,
    ) -> EscalationDecision {
        // Trigger T0: Consecutive no-change iterations
        if state.consecutive_no_change >= self.config.no_change_threshold {
            state.stuck = true;
            let reason = EscalationReason::ConsecutiveNoChange {
                count: state.consecutive_no_change,
                threshold: self.config.no_change_threshold,
            };
            return EscalationDecision {
                target_tier: SwarmTier::Human,
                escalated: true,
                reason: format!("Stuck: {}", reason),
                resolved: false,
                stuck: true,
                needs_review: false,
                action: SuggestedAction::FlagForHuman {
                    reason: format!("Issue {} stuck: {}", state.bead_id, reason),
                },
            };
        }

        // Friction detection: catch oscillating/plateauing errors that bypass T1's
        // consecutive-repeat requirement.
        // Guards: (1) require >= 4 iterations for reliable pattern detection,
        //         (2) suppress when error count is strictly decreasing — this
        //             avoids false positives on legitimate cascading error
        //             resolution (A→B→A with dropping counts).
        //   Note: we check error *count* decrease, not `progress_made`, because
        //   `progress_made` treats category changes as progress — but oscillation
        //   by definition changes categories every iteration.
        let error_count_decreasing = {
            let h = &state.iteration_history;
            h.len() >= 2 && h[h.len() - 1].error_count < h[h.len() - 2].error_count
        };
        let friction_signals = FrictionDetector::detect(state, report);
        if state.total_iterations >= 4 && !error_count_decreasing {
            if let Some(signal) = friction_signals.iter().find(|s| {
                s.severity >= FrictionSeverity::Medium
                    && matches!(
                        s.kind,
                        FrictionKind::ErrorOscillation { .. }
                            | FrictionKind::ErrorCountPlateau { .. }
                            | FrictionKind::CategoryChurn { .. }
                    )
            }) {
                let reason = EscalationReason::FrictionDetected {
                    description: signal.description.clone(),
                };
                state.record_escalation(SwarmTier::Council, reason.clone());
                return EscalationDecision {
                    target_tier: SwarmTier::Council,
                    escalated: true,
                    reason: format!("Friction detected: {}", reason),
                    resolved: false,
                    stuck: false,
                    needs_review: false,
                    action: SuggestedAction::RepairPlan,
                };
            }
        }

        // Trigger T1: Same error category repeated >= threshold
        if let Some((category, count)) = state.most_repeated_category() {
            if count >= self.config.repeat_threshold {
                let reason = EscalationReason::RepeatedErrorCategory { category, count };
                state.record_escalation(SwarmTier::Council, reason.clone());
                return EscalationDecision {
                    target_tier: SwarmTier::Council,
                    escalated: true,
                    reason: format!("Escalating: {}", reason),
                    resolved: false,
                    stuck: false,
                    needs_review: false,
                    action: SuggestedAction::RepairPlan,
                };
            }
        }

        // Trigger T3: Total failures exceeded threshold
        if state.total_failures() > self.config.failure_threshold {
            let reason = EscalationReason::TotalFailuresExceeded {
                count: state.total_failures(),
                threshold: self.config.failure_threshold,
            };
            state.record_escalation(SwarmTier::Council, reason.clone());
            return EscalationDecision {
                target_tier: SwarmTier::Council,
                escalated: true,
                reason: format!("Escalating: {}", reason),
                resolved: false,
                stuck: false,
                needs_review: false,
                action: SuggestedAction::RepairPlan,
            };
        }

        // Trigger T6: Multi-file complexity
        let files_touched = report
            .failure_signals
            .iter()
            .filter_map(|s| s.file.as_ref())
            .collect::<std::collections::HashSet<_>>()
            .len();

        if files_touched > self.config.multi_file_threshold {
            let reason = EscalationReason::MultiFileComplexity {
                file_count: files_touched,
            };
            state.record_escalation(SwarmTier::Council, reason.clone());
            return EscalationDecision {
                target_tier: SwarmTier::Council,
                escalated: true,
                reason: format!("Escalating: {}", reason),
                resolved: false,
                stuck: false,
                needs_review: false,
                action: SuggestedAction::ArchitecturalGuidance,
            };
        }

        // Budget check: Worker exhausted
        if state.remaining_budget(SwarmTier::Worker) == 0 {
            let reason = EscalationReason::BudgetExhausted {
                tier: SwarmTier::Worker,
            };
            state.record_escalation(SwarmTier::Council, reason.clone());
            return EscalationDecision {
                target_tier: SwarmTier::Council,
                escalated: true,
                reason: format!("Escalating: {}", reason),
                resolved: false,
                stuck: false,
                needs_review: false,
                action: SuggestedAction::RepairPlan,
            };
        }

        // No escalation — continue at Worker
        EscalationDecision {
            target_tier: SwarmTier::Worker,
            escalated: false,
            reason: format!(
                "Continuing at Worker ({} iterations remaining)",
                state.remaining_budget(SwarmTier::Worker)
            ),
            resolved: false,
            stuck: false,
            needs_review: false,
            action: SuggestedAction::Continue,
        }
    }

    /// Decision-making at the Council tier
    fn decide_at_council(
        &self,
        state: &mut EscalationState,
        report: &VerifierReport,
    ) -> EscalationDecision {
        // Trigger T0: Consecutive no-change iterations
        if state.consecutive_no_change >= self.config.no_change_threshold {
            state.stuck = true;
            let reason = EscalationReason::ConsecutiveNoChange {
                count: state.consecutive_no_change,
                threshold: self.config.no_change_threshold,
            };
            return EscalationDecision {
                target_tier: SwarmTier::Human,
                escalated: true,
                reason: format!("Stuck: {}", reason),
                resolved: false,
                stuck: true,
                needs_review: false,
                action: SuggestedAction::FlagForHuman {
                    reason: format!("Issue {} stuck: {}", state.bead_id, reason),
                },
            };
        }

        // Friction detection: high-severity friction at Council → flag for human
        let friction_signals = FrictionDetector::detect(state, report);
        if friction_signals
            .iter()
            .any(|s| s.severity == FrictionSeverity::High)
        {
            state.stuck = true;
            let desc = friction_signals
                .iter()
                .filter(|s| s.severity == FrictionSeverity::High)
                .map(|s| s.description.as_str())
                .collect::<Vec<_>>()
                .join("; ");
            return EscalationDecision {
                target_tier: SwarmTier::Human,
                escalated: true,
                reason: format!("High friction at Council: {}", desc),
                resolved: false,
                stuck: true,
                needs_review: false,
                action: SuggestedAction::FlagForHuman {
                    reason: format!(
                        "Issue {} stuck: high-severity friction: {}",
                        state.bead_id, desc
                    ),
                },
            };
        }

        // Budget check: Council exhausted → stuck, flag for human
        if state.remaining_budget(SwarmTier::Council) == 0 {
            state.stuck = true;
            return EscalationDecision {
                target_tier: SwarmTier::Human,
                escalated: true,
                reason: "Council budget exhausted".to_string(),
                resolved: false,
                stuck: true,
                needs_review: false,
                action: SuggestedAction::FlagForHuman {
                    reason: format!(
                        "Issue {} stuck: council budget exhausted after {} total iterations",
                        state.bead_id, state.total_iterations
                    ),
                },
            };
        }

        // Continue at Council (still has budget)
        EscalationDecision {
            target_tier: SwarmTier::Council,
            escalated: false,
            reason: format!(
                "Continuing at Council ({} iterations remaining)",
                state.remaining_budget(SwarmTier::Council)
            ),
            resolved: false,
            stuck: false,
            needs_review: false,
            action: SuggestedAction::ArchitecturalGuidance,
        }
    }
}

impl Default for EscalationEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::verifier::report::{GateOutcome, GateResult, VerifierReport};
    use std::time::Duration;

    fn make_failing_report(categories: Vec<ErrorCategory>) -> VerifierReport {
        use crate::feedback::error_parser::ParsedError;

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

    fn make_green_report() -> VerifierReport {
        let mut report = VerifierReport::new("/tmp/test".to_string());
        for gate in &["fmt", "clippy", "check", "test"] {
            report.add_gate(GateResult {
                gate: gate.to_string(),
                outcome: GateOutcome::Passed,
                duration_ms: 50,
                exit_code: Some(0),
                error_count: 0,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: None,
            });
        }
        report.finalize(Duration::from_millis(200));
        report
    }

    #[test]
    fn test_green_report_resolves() {
        let engine = EscalationEngine::new();
        let mut state = EscalationState::new("beads-1");

        let report = make_green_report();
        let decision = engine.decide(&mut state, &report);

        assert!(decision.resolved);
        assert!(!decision.stuck);
        assert!(decision.needs_review); // Adversary review by default
    }

    #[test]
    fn test_repeated_error_escalates() {
        let engine = EscalationEngine::new();
        let mut state = EscalationState::new("beads-2");

        // First iteration with lifetime error
        let report1 = make_failing_report(vec![ErrorCategory::Lifetime]);
        let d1 = engine.decide(&mut state, &report1);
        assert_eq!(d1.target_tier, SwarmTier::Worker);
        assert!(!d1.escalated);

        // Second iteration with same error — should escalate
        let report2 = make_failing_report(vec![ErrorCategory::Lifetime]);
        let d2 = engine.decide(&mut state, &report2);
        assert_eq!(d2.target_tier, SwarmTier::Council);
        assert!(d2.escalated);
    }

    #[test]
    fn test_total_failures_escalates() {
        let config = EscalationConfig {
            failure_threshold: 2,
            repeat_threshold: 10,     // High threshold to avoid repeat trigger
            no_change_threshold: 100, // High threshold to avoid no-change trigger
            ..Default::default()
        };
        let engine = EscalationEngine::with_config(config);
        let mut state = EscalationState::new("beads-3");

        // Different errors each time — no repeat trigger, but total failures accumulate
        let cats = [
            ErrorCategory::TypeMismatch,
            ErrorCategory::BorrowChecker,
            ErrorCategory::TraitBound,
        ];

        for cat in &cats {
            let report = make_failing_report(vec![*cat]);
            let d = engine.decide(&mut state, &report);
            if d.escalated {
                assert_eq!(d.target_tier, SwarmTier::Council);
                return;
            }
        }

        panic!("Expected escalation after {} failures", cats.len());
    }

    #[test]
    fn test_budget_exhaustion_escalates() {
        let engine = EscalationEngine::new();
        let mut state = EscalationState::new("beads-4");

        // Exhaust Implementer budget (6 iterations) with different errors
        let all_cats = [
            ErrorCategory::TypeMismatch,
            ErrorCategory::BorrowChecker,
            ErrorCategory::TraitBound,
            ErrorCategory::ImportResolution,
            ErrorCategory::Syntax,
            ErrorCategory::Macro,
        ];

        let mut escalated = false;
        for cat in &all_cats {
            let report = make_failing_report(vec![*cat]);
            let d = engine.decide(&mut state, &report);
            if d.escalated {
                escalated = true;
                break;
            }
        }

        assert!(escalated, "Should have escalated at some point");
    }

    #[test]
    fn test_stuck_when_all_exhausted() {
        let config = EscalationConfig {
            repeat_threshold: 100,    // Disable repeat trigger
            failure_threshold: 100,   // Disable failure trigger
            no_change_threshold: 100, // Disable no-change trigger
            ..Default::default()
        };
        let engine = EscalationEngine::with_config(config);
        let mut state = EscalationState::new("beads-5");

        // Manually exhaust Worker budget (4 iterations)
        for _ in 0..4 {
            state.record_iteration(vec![ErrorCategory::Other], 1, false);
        }
        state.record_escalation(
            SwarmTier::Council,
            EscalationReason::BudgetExhausted {
                tier: SwarmTier::Worker,
            },
        );

        // Exhaust Council budget (6 iterations)
        for _ in 0..6 {
            state.record_iteration(vec![ErrorCategory::Other], 1, false);
        }

        // Now decide — should be stuck
        let report = make_failing_report(vec![ErrorCategory::Other]);
        let d = engine.decide(&mut state, &report);
        assert!(d.stuck);
        assert!(matches!(d.action, SuggestedAction::FlagForHuman { .. }));
    }

    // ========================================================================
    // No-change circuit breaker tests (Issue 7)
    // ========================================================================

    #[test]
    fn test_no_change_circuit_breaker() {
        let config = EscalationConfig {
            no_change_threshold: 3,
            repeat_threshold: 100, // Disable other triggers
            failure_threshold: 100,
            ..Default::default()
        };
        let engine = EscalationEngine::with_config(config);
        let mut state = EscalationState::new("beads-no-change-1");

        // Simulate 3 consecutive no-change iterations
        state.record_no_change();
        state.record_no_change();
        state.record_no_change();
        assert_eq!(state.consecutive_no_change, 3);

        // Engine should report stuck
        let report = make_failing_report(vec![ErrorCategory::Other]);
        let d = engine.decide(&mut state, &report);
        assert!(d.stuck, "Should be stuck after 3 consecutive no-changes");
        assert_eq!(d.target_tier, SwarmTier::Human);
        assert!(d.reason.contains("no-change"), "Reason: {}", d.reason);
    }

    #[test]
    fn test_no_change_counter_resets_on_change() {
        let config = EscalationConfig {
            no_change_threshold: 3,
            repeat_threshold: 100,
            failure_threshold: 100,
            ..Default::default()
        };
        let engine = EscalationEngine::with_config(config);
        let mut state = EscalationState::new("beads-no-change-2");

        // 2 no-changes, then a reset, then 2 more — should NOT trigger
        state.record_no_change();
        state.record_no_change();
        assert_eq!(state.consecutive_no_change, 2);

        state.reset_no_change(); // Agent produced changes
        assert_eq!(state.consecutive_no_change, 0);

        state.record_no_change();
        state.record_no_change();
        assert_eq!(state.consecutive_no_change, 2);

        // Engine should NOT report stuck (only 2 consecutive, not 3)
        let report = make_failing_report(vec![ErrorCategory::Other]);
        let d = engine.decide(&mut state, &report);
        assert!(
            !d.stuck,
            "Should NOT be stuck — counter was reset mid-sequence"
        );
    }

    #[test]
    fn test_no_change_reason_serialization() {
        let reason = EscalationReason::ConsecutiveNoChange {
            count: 3,
            threshold: 3,
        };

        // Serialize
        let json = serde_json::to_string(&reason).unwrap();
        assert!(json.contains("consecutive_no_change"), "JSON: {json}");
        assert!(json.contains("\"count\":3"), "JSON: {json}");
        assert!(json.contains("\"threshold\":3"), "JSON: {json}");

        // Deserialize
        let roundtrip: EscalationReason = serde_json::from_str(&json).unwrap();
        match roundtrip {
            EscalationReason::ConsecutiveNoChange { count, threshold } => {
                assert_eq!(count, 3);
                assert_eq!(threshold, 3);
            }
            other => panic!("Expected ConsecutiveNoChange, got: {other:?}"),
        }

        // Display
        let display = format!("{reason}");
        assert!(
            display.contains("3 consecutive no-change"),
            "Display: {display}"
        );
    }
}
