//! Delight Detection Signals — detect when the swarm is making excellent progress
//!
//! Delight signals are higher-level patterns that indicate the work is going
//! better than expected, complementing the friction signals.

use crate::escalation::state::EscalationState;
use crate::verifier::report::VerifierReport;
use serde::{Deserialize, Serialize};

/// Kinds of delight signals
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelightKind {
    /// First attempt succeeded (all-green on iteration 1)
    FirstPassSuccess,
    /// Error count dropped significantly in one iteration
    RapidConvergence {
        from_count: usize,
        to_count: usize,
        drop_fraction: f32,
    },
    /// Consistent error reduction across N iterations
    SteadyProgress {
        iterations: u32,
        total_reduction: usize,
    },
    /// Only low-complexity errors remain
    LowComplexityOnly { max_complexity: u8 },
    /// Resolved within budget (iterations used vs budget)
    EfficientResolution { iterations_used: u32, budget: u32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DelightIntensity {
    Mild,
    Strong,
    Exceptional,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelightSignal {
    pub kind: DelightKind,
    pub intensity: DelightIntensity,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelightDetector;

impl DelightDetector {
    pub fn detect(state: &EscalationState, report: &VerifierReport) -> Vec<DelightSignal> {
        let mut out = Vec::new();

        // FirstPassSuccess: iteration 1 and all-green
        if state.total_iterations == 1 && report.all_green {
            out.push(DelightSignal {
                kind: DelightKind::FirstPassSuccess,
                intensity: DelightIntensity::Exceptional,
                description: "All gates passed on the very first attempt".to_string(),
            });
        }

        // RapidConvergence: last 2 iterations show ≥50% drop in error count
        let h = &state.iteration_history;
        if h.len() >= 2 {
            let prev = &h[h.len() - 2];
            let last = &h[h.len() - 1];
            if prev.error_count > 0 {
                let from = prev.error_count;
                let to = last.error_count;
                if to < from {
                    let drop_fraction = (from - to) as f32 / from as f32;
                    if drop_fraction >= 0.5 {
                        let intensity = if drop_fraction >= 0.8 {
                            DelightIntensity::Exceptional
                        } else {
                            DelightIntensity::Strong
                        };
                        out.push(DelightSignal {
                            kind: DelightKind::RapidConvergence {
                                from_count: from,
                                to_count: to,
                                drop_fraction,
                            },
                            intensity,
                            description: format!(
                                "Error count dropped {:.0}% in one iteration ({} → {})",
                                drop_fraction * 100.0,
                                from,
                                to
                            ),
                        });
                    }
                }
            }
        }

        // SteadyProgress: last 3+ iterations all show strictly decreasing error counts
        if h.len() >= 3 {
            // Find the longest tail of strictly decreasing counts
            let mut run = 1usize;
            for i in (1..h.len()).rev() {
                if h[i].error_count < h[i - 1].error_count {
                    run += 1;
                } else {
                    break;
                }
            }
            if run >= 3 {
                let start = h.len() - run;
                let first_count = h[start].error_count;
                let last_count = h[h.len() - 1].error_count;
                let total_reduction = first_count.saturating_sub(last_count);
                let intensity = if run >= 4 {
                    DelightIntensity::Strong
                } else {
                    DelightIntensity::Mild
                };
                out.push(DelightSignal {
                    kind: DelightKind::SteadyProgress {
                        iterations: run as u32,
                        total_reduction,
                    },
                    intensity,
                    description: format!(
                        "Errors decreased every iteration for {} consecutive iterations (reduced by {})",
                        run, total_reduction
                    ),
                });
            }
        }

        // LowComplexityOnly: not all-green but all error categories have complexity <= 1
        if !report.all_green && !report.error_categories.is_empty() {
            let max_complexity = report
                .error_categories
                .keys()
                .map(|cat| cat.complexity())
                .max()
                .unwrap_or(0);
            if max_complexity <= 1 {
                out.push(DelightSignal {
                    kind: DelightKind::LowComplexityOnly { max_complexity },
                    intensity: DelightIntensity::Mild,
                    description: format!(
                        "Only low-complexity errors remain (max complexity: {})",
                        max_complexity
                    ),
                });
            }
        }

        // EfficientResolution: all-green and used few iterations relative to budget
        if report.all_green && state.total_iterations > 0 {
            let iterations_used = state.total_iterations;
            let budget = state
                .tier_budgets
                .get(&state.current_tier)
                .map(|b| b.max_iterations)
                .unwrap_or(4);
            let intensity = if iterations_used <= budget / 2 {
                DelightIntensity::Exceptional
            } else if iterations_used <= budget * 3 / 4 {
                DelightIntensity::Strong
            } else {
                // Within budget but not especially efficient — still worth noting
                DelightIntensity::Mild
            };
            out.push(DelightSignal {
                kind: DelightKind::EfficientResolution {
                    iterations_used,
                    budget,
                },
                intensity,
                description: format!(
                    "Resolved in {} of {} budgeted iterations",
                    iterations_used, budget
                ),
            });
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::escalation::state::EscalationState;
    use crate::feedback::error_parser::ErrorCategory;

    fn rep() -> VerifierReport {
        VerifierReport::new("/tmp/t".to_string())
    }
    fn green_rep() -> VerifierReport {
        let mut r = rep();
        // simulate all-green by setting all_green directly
        r.all_green = true;
        r
    }
    fn has(v: &[DelightSignal], f: impl Fn(&DelightKind) -> bool) -> bool {
        v.iter().any(|s| f(&s.kind))
    }

    #[test]
    fn test_first_pass_success() {
        let mut s = EscalationState::new("t");
        s.record_iteration(vec![], 0, true);
        let signals = DelightDetector::detect(&s, &green_rep());
        assert!(has(&signals, |k| matches!(
            k,
            DelightKind::FirstPassSuccess
        )));
        let sig = signals
            .iter()
            .find(|s| matches!(s.kind, DelightKind::FirstPassSuccess))
            .unwrap();
        assert_eq!(sig.intensity, DelightIntensity::Exceptional);
    }

    #[test]
    fn test_rapid_convergence() {
        let mut s = EscalationState::new("t");
        s.record_iteration(vec![ErrorCategory::TypeMismatch], 10, false);
        s.record_iteration(vec![ErrorCategory::TypeMismatch], 1, false);
        let signals = DelightDetector::detect(&s, &rep());
        assert!(has(&signals, |k| matches!(
            k,
            DelightKind::RapidConvergence { .. }
        )));
    }

    #[test]
    fn test_steady_progress() {
        let mut s = EscalationState::new("t");
        s.record_iteration(vec![ErrorCategory::TypeMismatch], 6, false);
        s.record_iteration(vec![ErrorCategory::TypeMismatch], 4, false);
        s.record_iteration(vec![ErrorCategory::TypeMismatch], 2, false);
        let signals = DelightDetector::detect(&s, &rep());
        assert!(has(&signals, |k| matches!(
            k,
            DelightKind::SteadyProgress { .. }
        )));
    }

    #[test]
    fn test_efficient_resolution() {
        let mut s = EscalationState::new("t");
        s.record_iteration(vec![], 0, true);
        let signals = DelightDetector::detect(&s, &green_rep());
        assert!(has(&signals, |k| matches!(
            k,
            DelightKind::EfficientResolution { .. }
        )));
    }
}
