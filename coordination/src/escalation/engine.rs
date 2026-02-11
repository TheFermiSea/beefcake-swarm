//! Escalation Engine — Deterministic decision-making for tier routing
//!
//! Consumes VerifierReports and EscalationState to produce EscalationDecisions.
//! All decisions are deterministic — no LLM calls in this module.

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
    /// Error category repeat threshold to trigger Implementer → Integrator
    pub repeat_threshold: u32,
    /// Total failure threshold to trigger Implementer → Integrator
    pub failure_threshold: u32,
    /// File count threshold for Cloud escalation
    pub multi_file_threshold: usize,
    /// Whether to require adversary review before close
    pub require_adversary_review: bool,
}

impl Default for EscalationConfig {
    fn default() -> Self {
        Self {
            repeat_threshold: 2,
            failure_threshold: 3,
            multi_file_threshold: 8,
            require_adversary_review: true,
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

        // Check: Are we at the Implementer tier?
        if state.current_tier == SwarmTier::Implementer {
            return self.decide_at_implementer(state, report);
        }

        // Check: Are we at the Integrator tier?
        if state.current_tier == SwarmTier::Integrator {
            return self.decide_at_integrator(state, report);
        }

        // Check: Are we at the Cloud tier?
        if state.current_tier == SwarmTier::Cloud {
            return self.decide_at_cloud(state, report);
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
            && state.remaining_consultations(SwarmTier::Adversary) > 0
        {
            EscalationDecision {
                target_tier: SwarmTier::Adversary,
                escalated: state.current_tier != SwarmTier::Adversary,
                reason: "All gates passed — sending for adversary review".to_string(),
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

    /// Decision-making at the Implementer tier
    fn decide_at_implementer(
        &self,
        state: &mut EscalationState,
        report: &VerifierReport,
    ) -> EscalationDecision {
        // Trigger T1: Same error category repeated >= threshold
        if let Some((category, count)) = state.most_repeated_category() {
            if count >= self.config.repeat_threshold {
                let reason = EscalationReason::RepeatedErrorCategory { category, count };
                state.record_escalation(SwarmTier::Integrator, reason.clone());
                return EscalationDecision {
                    target_tier: SwarmTier::Integrator,
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
            state.record_escalation(SwarmTier::Integrator, reason.clone());
            return EscalationDecision {
                target_tier: SwarmTier::Integrator,
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
            state.record_escalation(SwarmTier::Cloud, reason.clone());
            return EscalationDecision {
                target_tier: SwarmTier::Cloud,
                escalated: true,
                reason: format!("Escalating: {}", reason),
                resolved: false,
                stuck: false,
                needs_review: false,
                action: SuggestedAction::ArchitecturalGuidance,
            };
        }

        // Budget check: Implementer exhausted
        if state.remaining_budget(SwarmTier::Implementer) == 0 {
            let reason = EscalationReason::BudgetExhausted {
                tier: SwarmTier::Implementer,
            };
            state.record_escalation(SwarmTier::Integrator, reason.clone());
            return EscalationDecision {
                target_tier: SwarmTier::Integrator,
                escalated: true,
                reason: format!("Escalating: {}", reason),
                resolved: false,
                stuck: false,
                needs_review: false,
                action: SuggestedAction::RepairPlan,
            };
        }

        // No escalation — continue at Implementer
        EscalationDecision {
            target_tier: SwarmTier::Implementer,
            escalated: false,
            reason: format!(
                "Continuing at Implementer ({} iterations remaining)",
                state.remaining_budget(SwarmTier::Implementer)
            ),
            resolved: false,
            stuck: false,
            needs_review: false,
            action: SuggestedAction::Continue,
        }
    }

    /// Decision-making at the Integrator tier
    fn decide_at_integrator(
        &self,
        state: &mut EscalationState,
        _report: &VerifierReport,
    ) -> EscalationDecision {
        // Budget check: Integrator exhausted
        if state.remaining_budget(SwarmTier::Integrator) == 0 {
            // Escalate to Cloud
            if state.remaining_budget(SwarmTier::Cloud) > 0 {
                let reason = EscalationReason::IntegratorStuck {
                    consultations: state
                        .tier_consultations
                        .get(&SwarmTier::Integrator)
                        .copied()
                        .unwrap_or(0),
                };
                state.record_escalation(SwarmTier::Cloud, reason.clone());
                return EscalationDecision {
                    target_tier: SwarmTier::Cloud,
                    escalated: true,
                    reason: format!("Escalating: {}", reason),
                    resolved: false,
                    stuck: false,
                    needs_review: false,
                    action: SuggestedAction::ArchitecturalGuidance,
                };
            }

            // Both Integrator and Cloud exhausted → stuck
            state.stuck = true;
            return EscalationDecision {
                target_tier: SwarmTier::Cloud,
                escalated: false,
                reason: "All local and cloud budgets exhausted".to_string(),
                resolved: false,
                stuck: true,
                needs_review: false,
                action: SuggestedAction::FlagForHuman {
                    reason: format!(
                        "Issue {} stuck after {} iterations across all tiers",
                        state.bead_id, state.total_iterations
                    ),
                },
            };
        }

        // Continue at Integrator (still has budget)
        EscalationDecision {
            target_tier: SwarmTier::Integrator,
            escalated: false,
            reason: format!(
                "Continuing at Integrator ({} consultations remaining)",
                state.remaining_budget(SwarmTier::Integrator)
            ),
            resolved: false,
            stuck: false,
            needs_review: false,
            action: SuggestedAction::RepairPlan,
        }
    }

    /// Decision-making at the Cloud tier
    fn decide_at_cloud(
        &self,
        state: &mut EscalationState,
        _report: &VerifierReport,
    ) -> EscalationDecision {
        // Budget check: Cloud exhausted
        if state.remaining_budget(SwarmTier::Cloud) == 0 {
            state.stuck = true;
            return EscalationDecision {
                target_tier: SwarmTier::Cloud,
                escalated: false,
                reason: "Cloud budget exhausted".to_string(),
                resolved: false,
                stuck: true,
                needs_review: false,
                action: SuggestedAction::FlagForHuman {
                    reason: format!(
                        "Issue {} stuck: cloud budget exhausted after {} total iterations",
                        state.bead_id, state.total_iterations
                    ),
                },
            };
        }

        // Cloud strategy flows back down — Implementer executes the cloud's plan
        EscalationDecision {
            target_tier: SwarmTier::Cloud,
            escalated: false,
            reason: format!(
                "Cloud consultation ({} remaining)",
                state.remaining_budget(SwarmTier::Cloud)
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
        assert_eq!(d1.target_tier, SwarmTier::Implementer);
        assert!(!d1.escalated);

        // Second iteration with same error — should escalate
        let report2 = make_failing_report(vec![ErrorCategory::Lifetime]);
        let d2 = engine.decide(&mut state, &report2);
        assert_eq!(d2.target_tier, SwarmTier::Integrator);
        assert!(d2.escalated);
    }

    #[test]
    fn test_total_failures_escalates() {
        let config = EscalationConfig {
            failure_threshold: 2,
            repeat_threshold: 10, // High threshold to avoid repeat trigger
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
                assert_eq!(d.target_tier, SwarmTier::Integrator);
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
            repeat_threshold: 100,  // Disable repeat trigger
            failure_threshold: 100, // Disable failure trigger
            ..Default::default()
        };
        let engine = EscalationEngine::with_config(config);
        let mut state = EscalationState::new("beads-5");

        // Manually exhaust all budgets
        for _ in 0..6 {
            state.record_iteration(vec![ErrorCategory::Other], 1, false);
        }
        state.record_escalation(
            SwarmTier::Integrator,
            EscalationReason::BudgetExhausted {
                tier: SwarmTier::Implementer,
            },
        );

        for _ in 0..2 {
            state.record_iteration(vec![ErrorCategory::Other], 1, false);
        }
        state.record_escalation(
            SwarmTier::Cloud,
            EscalationReason::IntegratorStuck { consultations: 2 },
        );

        for _ in 0..2 {
            state.record_iteration(vec![ErrorCategory::Other], 1, false);
        }

        // Now decide — should be stuck
        let report = make_failing_report(vec![ErrorCategory::Other]);
        let d = engine.decide(&mut state, &report);
        assert!(d.stuck);
        assert!(matches!(d.action, SuggestedAction::FlagForHuman { .. }));
    }
}
