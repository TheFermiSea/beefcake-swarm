//! Auto-ticket recurrent failure signatures.
//!
//! Analyzes iteration history to detect recurring failure patterns and
//! generates ticket suggestions for automatic beads issue creation.

use crate::escalation::state::{EscalationState, IterationRecord, SwarmTier};
use crate::feedback::error_parser::ErrorCategory;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Fingerprint of a recurring failure pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FailureSignature {
    /// Dominant error category.
    pub category: ErrorCategory,
    /// Tier where failure recurred.
    pub tier: SwarmTier,
    /// Number of consecutive iterations with this category present.
    pub recurrence_count: u32,
}

/// Priority for auto-generated tickets (maps to beads 0–4).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TicketPriority {
    /// P1 — recurring across tiers or stuck
    High,
    /// P2 — recurring within a single tier
    Medium,
    /// P3 — minor pattern, informational
    Low,
}

impl TicketPriority {
    /// Convert to beads numeric priority.
    pub fn to_beads_priority(self) -> u8 {
        match self {
            Self::High => 1,
            Self::Medium => 2,
            Self::Low => 3,
        }
    }
}

/// Suggestion for auto-creating a beads ticket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TicketSuggestion {
    /// Suggested ticket title.
    pub title: String,
    /// Detailed description with context.
    pub description: String,
    /// Suggested priority.
    pub priority: TicketPriority,
    /// The failure signature that triggered this.
    pub signature: FailureSignature,
    /// Source issue that encountered this failure.
    pub source_issue_id: String,
}

/// Detects recurring failure patterns and suggests tickets.
pub struct RecurrentFailureDetector {
    /// Minimum consecutive recurrences to trigger a ticket.
    pub recurrence_threshold: u32,
}

impl RecurrentFailureDetector {
    pub fn new(recurrence_threshold: u32) -> Self {
        Self {
            recurrence_threshold,
        }
    }

    /// Analyze iteration history and return ticket suggestions.
    pub fn detect(&self, state: &EscalationState) -> Vec<TicketSuggestion> {
        if state.iteration_history.len() < self.recurrence_threshold as usize {
            return vec![];
        }

        // Count consecutive trailing recurrences per category.
        let signatures = self.extract_signatures(&state.iteration_history, state.current_tier);
        let mut suggestions = Vec::new();

        for sig in &signatures {
            if sig.recurrence_count >= self.recurrence_threshold {
                suggestions.push(self.build_suggestion(sig, state));
            }
        }

        suggestions
    }

    /// Extract failure signatures from iteration history.
    fn extract_signatures(
        &self,
        history: &[IterationRecord],
        current_tier: SwarmTier,
    ) -> Vec<FailureSignature> {
        if history.is_empty() {
            return vec![];
        }

        // Count consecutive trailing appearances of each category.
        let mut streak: HashMap<ErrorCategory, u32> = HashMap::new();

        // Walk backwards from most recent non-green iteration.
        for record in history.iter().rev() {
            if record.all_green {
                break; // Success resets all streaks.
            }
            for &cat in &record.error_categories {
                *streak.entry(cat).or_insert(0) += 1;
            }
        }

        streak
            .into_iter()
            .map(|(category, recurrence_count)| FailureSignature {
                category,
                tier: current_tier,
                recurrence_count,
            })
            .collect()
    }

    /// Build a ticket suggestion from a signature.
    fn build_suggestion(
        &self,
        sig: &FailureSignature,
        state: &EscalationState,
    ) -> TicketSuggestion {
        let priority = if sig.recurrence_count >= self.recurrence_threshold * 2
            || state.stuck
            || sig.tier == SwarmTier::Council
        {
            TicketPriority::High
        } else {
            TicketPriority::Medium
        };

        let title = format!(
            "Recurring {} errors in {} ({}x)",
            sig.category, state.bead_id, sig.recurrence_count
        );

        let total_failures = state.total_failures();
        let escalation_count = state.escalation_history.len();

        let description = format!(
            "Recurring failure pattern detected during swarm execution.\n\
             \n\
             - **Error category:** {}\n\
             - **Recurrence count:** {} consecutive iterations\n\
             - **Tier at detection:** {:?}\n\
             - **Total failures:** {}\n\
             - **Escalations:** {}\n\
             - **Stuck:** {}\n\
             \n\
             This pattern suggests the error class needs targeted attention \
             (e.g., a dedicated fix, architectural change, or test coverage).",
            sig.category,
            sig.recurrence_count,
            sig.tier,
            total_failures,
            escalation_count,
            state.stuck,
        );

        TicketSuggestion {
            title,
            description,
            priority,
            signature: sig.clone(),
            source_issue_id: state.bead_id.clone(),
        }
    }
}

impl Default for RecurrentFailureDetector {
    fn default() -> Self {
        Self::new(3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn record(iter: u32, cats: &[ErrorCategory], green: bool) -> IterationRecord {
        IterationRecord {
            iteration: iter,
            tier: SwarmTier::Worker,
            timestamp: Utc::now(),
            error_categories: cats.to_vec(),
            error_count: if green { 0 } else { cats.len() },
            all_green: green,
            progress_made: false,
        }
    }

    fn state_with_history(issue_id: &str, records: Vec<IterationRecord>) -> EscalationState {
        let mut state = EscalationState::new(issue_id.to_string());
        for r in &records {
            state.record_iteration(r.error_categories.clone(), r.error_count, r.all_green);
        }
        state
    }

    #[test]
    fn test_no_suggestions_below_threshold() {
        let detector = RecurrentFailureDetector::new(3);
        let state = state_with_history(
            "test-001",
            vec![
                record(1, &[ErrorCategory::TypeMismatch], false),
                record(2, &[ErrorCategory::TypeMismatch], false),
            ],
        );
        let suggestions = detector.detect(&state);
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_suggestion_at_threshold() {
        let detector = RecurrentFailureDetector::new(3);
        let state = state_with_history(
            "test-002",
            vec![
                record(1, &[ErrorCategory::BorrowChecker], false),
                record(2, &[ErrorCategory::BorrowChecker], false),
                record(3, &[ErrorCategory::BorrowChecker], false),
            ],
        );
        let suggestions = detector.detect(&state);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(
            suggestions[0].signature.category,
            ErrorCategory::BorrowChecker
        );
        assert_eq!(suggestions[0].signature.recurrence_count, 3);
        assert!(suggestions[0].title.contains("borrow_checker"));
        assert_eq!(suggestions[0].priority, TicketPriority::Medium);
    }

    #[test]
    fn test_success_resets_streak() {
        let detector = RecurrentFailureDetector::new(3);
        let state = state_with_history(
            "test-003",
            vec![
                record(1, &[ErrorCategory::Lifetime], false),
                record(2, &[ErrorCategory::Lifetime], false),
                record(3, &[], true), // success resets
                record(4, &[ErrorCategory::Lifetime], false),
                record(5, &[ErrorCategory::Lifetime], false),
            ],
        );
        let suggestions = detector.detect(&state);
        // Only 2 consecutive after the success — below threshold.
        assert!(suggestions.is_empty());
    }

    #[test]
    fn test_multiple_categories() {
        let detector = RecurrentFailureDetector::new(2);
        let state = state_with_history(
            "test-004",
            vec![
                record(
                    1,
                    &[ErrorCategory::BorrowChecker, ErrorCategory::Lifetime],
                    false,
                ),
                record(
                    2,
                    &[ErrorCategory::BorrowChecker, ErrorCategory::Lifetime],
                    false,
                ),
            ],
        );
        let suggestions = detector.detect(&state);
        assert_eq!(suggestions.len(), 2);
        let cats: Vec<_> = suggestions.iter().map(|s| s.signature.category).collect();
        assert!(cats.contains(&ErrorCategory::BorrowChecker));
        assert!(cats.contains(&ErrorCategory::Lifetime));
    }

    #[test]
    fn test_high_priority_for_stuck() {
        let detector = RecurrentFailureDetector::new(2);
        let mut state = state_with_history(
            "test-005",
            vec![
                record(1, &[ErrorCategory::Async], false),
                record(2, &[ErrorCategory::Async], false),
            ],
        );
        state.stuck = true;
        let suggestions = detector.detect(&state);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].priority, TicketPriority::High);
    }

    #[test]
    fn test_high_priority_for_double_threshold() {
        let detector = RecurrentFailureDetector::new(2);
        let state = state_with_history(
            "test-006",
            vec![
                record(1, &[ErrorCategory::TraitBound], false),
                record(2, &[ErrorCategory::TraitBound], false),
                record(3, &[ErrorCategory::TraitBound], false),
                record(4, &[ErrorCategory::TraitBound], false),
            ],
        );
        let suggestions = detector.detect(&state);
        assert_eq!(suggestions.len(), 1);
        assert_eq!(suggestions[0].priority, TicketPriority::High);
    }

    #[test]
    fn test_ticket_priority_to_beads() {
        assert_eq!(TicketPriority::High.to_beads_priority(), 1);
        assert_eq!(TicketPriority::Medium.to_beads_priority(), 2);
        assert_eq!(TicketPriority::Low.to_beads_priority(), 3);
    }

    #[test]
    fn test_ticket_description_contains_context() {
        let detector = RecurrentFailureDetector::new(2);
        let state = state_with_history(
            "beefcake-xyz",
            vec![
                record(1, &[ErrorCategory::ImportResolution], false),
                record(2, &[ErrorCategory::ImportResolution], false),
            ],
        );
        let suggestions = detector.detect(&state);
        assert_eq!(suggestions.len(), 1);
        assert!(suggestions[0].description.contains("import"));
        assert!(suggestions[0].title.contains("beefcake-xyz"));
        assert_eq!(suggestions[0].source_issue_id, "beefcake-xyz");
    }

    #[test]
    fn test_empty_history_no_suggestions() {
        let detector = RecurrentFailureDetector::default();
        let state = EscalationState::new("test-empty".to_string());
        let suggestions = detector.detect(&state);
        assert!(suggestions.is_empty());
    }
}
