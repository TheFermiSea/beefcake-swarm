//! Friction Detection Signals — detect when the swarm is experiencing difficulty
//!
//! Friction signals are higher-level patterns that indicate the work is harder
//! than expected, beyond what the basic escalation triggers already detect.

use crate::escalation::state::EscalationState;
use crate::feedback::error_parser::ErrorCategory;
use crate::verifier::report::VerifierReport;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrictionKind {
    /// Error categories flip-flopping between iterations
    ErrorOscillation { categories: Vec<ErrorCategory>, oscillation_count: u32 },
    /// Error count not decreasing over N iterations
    ErrorCountPlateau { count: usize, iterations: u32 },
    /// Too many different error categories across recent iterations
    CategoryChurn { unique_categories: usize, iterations: u32 },
    /// High-complexity errors (Lifetime/Async/Macro) dominating the error mix
    HighComplexityDominance { category: ErrorCategory, fraction: f32 },
    /// Multiple escalations in few iterations
    RapidEscalation { escalations: u32, within_iterations: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FrictionSeverity { Low, Medium, High }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrictionSignal {
    pub kind: FrictionKind,
    pub severity: FrictionSeverity,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrictionDetector;

impl FrictionDetector {
    pub fn detect(state: &EscalationState, report: &VerifierReport) -> Vec<FrictionSignal> {
        let mut out = Vec::new();
        let w = &state.recent_error_categories;

        // Oscillation: same categories appear in alternating positions
        if w.len() >= 4 {
            let even: HashSet<ErrorCategory> = w.iter().step_by(2).flat_map(|v| v.iter().copied()).collect();
            let odd: HashSet<ErrorCategory> = w.iter().skip(1).step_by(2).flat_map(|v| v.iter().copied()).collect();
            let cats: Vec<ErrorCategory> = even.intersection(&odd).copied().collect();
            if !cats.is_empty() {
                let n = w.len() as u32;
                out.push(FrictionSignal {
                    kind: FrictionKind::ErrorOscillation { categories: cats.clone(), oscillation_count: n },
                    severity: if n >= 6 { FrictionSeverity::High } else { FrictionSeverity::Medium },
                    description: format!("Error categories {:?} oscillating across {} iterations", cats, n),
                });
            }
        }

        // Plateau: error count not decreasing
        let h = &state.iteration_history;
        if h.len() >= 3 {
            let sz = h.len().min(4);
            let hw = &h[h.len() - sz..];
            if !hw.iter().any(|r| r.all_green) {
                let (first, last) = (hw[0].error_count, hw[sz - 1].error_count);
                if last >= first && first > 0 {
                    out.push(FrictionSignal {
                        kind: FrictionKind::ErrorCountPlateau { count: last, iterations: sz as u32 },
                        severity: if last > first { FrictionSeverity::High } else { FrictionSeverity::Medium },
                        description: format!("Error count stuck at {} over {} iterations", last, sz),
                    });
                }
            }
        }

        // Category churn: too many unique categories in the window
        if w.len() >= 3 {
            let n = w.iter().flat_map(|v| v.iter().copied()).collect::<HashSet<ErrorCategory>>().len();
            if n > 3 {
                out.push(FrictionSignal {
                    kind: FrictionKind::CategoryChurn { unique_categories: n, iterations: w.len() as u32 },
                    severity: if n >= 6 { FrictionSeverity::High } else { FrictionSeverity::Medium },
                    description: format!("{} unique error categories across {} iterations", n, w.len()),
                });
            }
        }

        // High-complexity dominance
        let total: usize = report.error_categories.values().sum();
        if total > 0 {
            for (&cat, &count) in &report.error_categories {
                if cat.complexity() < 3 { continue; }
                let frac = count as f32 / total as f32;
                if frac >= 0.6 {
                    out.push(FrictionSignal {
                        kind: FrictionKind::HighComplexityDominance { category: cat, fraction: frac },
                        severity: if frac >= 0.85 { FrictionSeverity::High } else { FrictionSeverity::Medium },
                        description: format!("{:.0}% of errors are high-complexity {:?}", frac * 100.0, cat),
                    });
                }
            }
        }

        // Rapid escalation: multiple escalations in few iterations
        let esc = &state.escalation_history;
        if esc.len() >= 2 {
            let recent = &esc[esc.len().saturating_sub(3)..];
            let span = state.total_iterations.saturating_sub(recent[0].at_iteration);
            if recent.len() >= 2 && span <= 4 {
                out.push(FrictionSignal {
                    kind: FrictionKind::RapidEscalation { escalations: recent.len() as u32, within_iterations: span },
                    severity: if span <= 2 { FrictionSeverity::High } else { FrictionSeverity::Medium },
                    description: format!("{} escalations within {} iterations", recent.len(), span),
                });
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::escalation::state::{EscalationReason, EscalationState, SwarmTier};

    fn rep() -> VerifierReport { VerifierReport::new("/tmp/t".to_string()) }
    fn has(v: &[FrictionSignal], f: impl Fn(&FrictionKind) -> bool) -> bool { v.iter().any(|s| f(&s.kind)) }

    #[test]
    fn test_oscillation() {
        let mut s = EscalationState::new("t");
        for _ in 0..2 { s.record_iteration(vec![ErrorCategory::Lifetime], 2, false); s.record_iteration(vec![ErrorCategory::TypeMismatch], 2, false); }
        assert!(has(&FrictionDetector::detect(&s, &rep()), |k| matches!(k, FrictionKind::ErrorOscillation { .. })));
    }

    #[test]
    fn test_plateau() {
        let mut s = EscalationState::new("t");
        for _ in 0..4 { s.record_iteration(vec![ErrorCategory::BorrowChecker], 5, false); }
        assert!(has(&FrictionDetector::detect(&s, &rep()), |k| matches!(k, FrictionKind::ErrorCountPlateau { .. })));
        let mut s2 = EscalationState::new("t");
        for i in (1..=4).rev() { s2.record_iteration(vec![ErrorCategory::BorrowChecker], i, false); }
        assert!(!has(&FrictionDetector::detect(&s2, &rep()), |k| matches!(k, FrictionKind::ErrorCountPlateau { .. })));
    }

    #[test]
    fn test_category_churn() {
        let mut s = EscalationState::new("t");
        for cat in [ErrorCategory::Lifetime, ErrorCategory::Async, ErrorCategory::TypeMismatch, ErrorCategory::BorrowChecker] { s.record_iteration(vec![cat], 1, false); }
        assert!(has(&FrictionDetector::detect(&s, &rep()), |k| matches!(k, FrictionKind::CategoryChurn { .. })));
    }

    #[test]
    fn test_high_complexity_dominance() {
        let mut r = rep();
        r.error_categories.insert(ErrorCategory::Lifetime, 9);
        r.error_categories.insert(ErrorCategory::TypeMismatch, 1);
        let v = FrictionDetector::detect(&EscalationState::new("t"), &r);
        assert!(v.iter().any(|s| matches!(&s.kind, FrictionKind::HighComplexityDominance { .. }) && s.severity == FrictionSeverity::High));
    }

    #[test]
    fn test_rapid_escalation() {
        let mut s = EscalationState::new("t");
        s.record_iteration(vec![ErrorCategory::Lifetime], 3, false);
        s.record_escalation(SwarmTier::Council, EscalationReason::RepeatedErrorCategory { category: ErrorCategory::Lifetime, count: 2 });
        assert!(!has(&FrictionDetector::detect(&s, &rep()), |k| matches!(k, FrictionKind::RapidEscalation { .. })));
        s.record_iteration(vec![ErrorCategory::Lifetime], 3, false);
        s.record_escalation(SwarmTier::Human, EscalationReason::BudgetExhausted { tier: SwarmTier::Council });
        assert!(has(&FrictionDetector::detect(&s, &rep()), |k| matches!(k, FrictionKind::RapidEscalation { .. })));
    }
}
