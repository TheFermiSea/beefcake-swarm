//! Conservative acceptance for self-modifications (skills and thresholds).
//!
//! New skills and calibrated thresholds must pass a validation period
//! before being promoted to active use. This prevents a single lucky
//! session from permanently biasing routing or escalation behavior.
//!
//! Lifecycle: `Candidate` → `Probation` → `Accepted` (or `Rejected`)

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Lifecycle status of a self-modification (skill or threshold).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcceptanceStatus {
    /// Newly created, not yet used enough to evaluate.
    Candidate,
    /// Under evaluation — `uses_remaining` decrements each use.
    Probation { uses_remaining: u32 },
    /// Promoted to active use — meets policy thresholds.
    Accepted,
    /// Failed validation — soft-deleted (kept for analysis, not matched).
    Rejected,
}

impl AcceptanceStatus {
    /// Whether this status allows the item to be used in matching.
    ///
    /// Accepted items are always used. Probation items can optionally be
    /// included as experimental. Candidate and Rejected are never matched.
    pub fn is_active(&self) -> bool {
        matches!(self, AcceptanceStatus::Accepted)
    }

    /// Whether this is an experimental (probation) item.
    pub fn is_experimental(&self) -> bool {
        matches!(self, AcceptanceStatus::Probation { .. })
    }

    /// Whether the item has been soft-deleted.
    pub fn is_rejected(&self) -> bool {
        matches!(self, AcceptanceStatus::Rejected)
    }
}

impl std::fmt::Display for AcceptanceStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcceptanceStatus::Candidate => write!(f, "Candidate"),
            AcceptanceStatus::Probation { uses_remaining } => {
                write!(f, "Probation({uses_remaining} uses left)")
            }
            AcceptanceStatus::Accepted => write!(f, "Accepted"),
            AcceptanceStatus::Rejected => write!(f, "Rejected"),
        }
    }
}

/// Policy governing how items progress through the acceptance lifecycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptancePolicy {
    /// Minimum total uses before an item can be promoted from Probation to Accepted.
    pub min_uses_before_promotion: u32,
    /// Minimum success rate (0.0-1.0) required for promotion.
    pub min_success_rate: f64,
    /// Number of uses during the probation period.
    pub probation_period: u32,
}

impl Default for AcceptancePolicy {
    fn default() -> Self {
        Self {
            min_uses_before_promotion: 5,
            min_success_rate: 0.7,
            probation_period: 10,
        }
    }
}

/// Outcome of a single use of an item.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseOutcome {
    Success,
    Failure,
}

// ---------------------------------------------------------------------------
// VerificationTracker
// ---------------------------------------------------------------------------

/// Tracks acceptance lifecycle for self-modifying items (skills, thresholds).
///
/// Each tracked item has: an ID, current status, success count, failure count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrackedItem {
    pub id: String,
    pub status: AcceptanceStatus,
    pub success_count: u32,
    pub failure_count: u32,
}

impl TrackedItem {
    fn total_uses(&self) -> u32 {
        self.success_count + self.failure_count
    }

    fn success_rate(&self) -> f64 {
        let total = self.total_uses();
        if total == 0 {
            return 0.0;
        }
        f64::from(self.success_count) / f64::from(total)
    }
}

/// Manages acceptance lifecycle transitions for a set of items.
pub struct VerificationTracker {
    items: Vec<TrackedItem>,
    policy: AcceptancePolicy,
}

impl VerificationTracker {
    /// Create a new tracker with the default policy.
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            policy: AcceptancePolicy::default(),
        }
    }

    /// Create a tracker with a custom policy.
    pub fn with_policy(policy: AcceptancePolicy) -> Self {
        Self {
            items: Vec::new(),
            policy,
        }
    }

    /// Register a new item as a Candidate.
    pub fn register(&mut self, id: &str) {
        // Don't re-register existing items
        if self.items.iter().any(|i| i.id == id) {
            return;
        }
        self.items.push(TrackedItem {
            id: id.to_string(),
            status: AcceptanceStatus::Candidate,
            success_count: 0,
            failure_count: 0,
        });
    }

    /// Record a use of an item and transition its status.
    ///
    /// Returns the new status after the transition.
    pub fn track_usage(&mut self, id: &str, outcome: UseOutcome) -> Option<AcceptanceStatus> {
        let item = self.items.iter_mut().find(|i| i.id == id)?;

        // Terminal states don't change
        if matches!(
            item.status,
            AcceptanceStatus::Accepted | AcceptanceStatus::Rejected
        ) {
            // Still record the usage for analytics
            match outcome {
                UseOutcome::Success => item.success_count += 1,
                UseOutcome::Failure => item.failure_count += 1,
            }
            return Some(item.status.clone());
        }

        // Record outcome
        match outcome {
            UseOutcome::Success => item.success_count += 1,
            UseOutcome::Failure => item.failure_count += 1,
        }

        // State transitions
        match &item.status {
            AcceptanceStatus::Candidate => {
                // First use → enter probation
                item.status = AcceptanceStatus::Probation {
                    uses_remaining: self.policy.probation_period.saturating_sub(1),
                };
            }
            AcceptanceStatus::Probation { uses_remaining } => {
                let remaining = uses_remaining.saturating_sub(1);

                if item.total_uses() >= self.policy.min_uses_before_promotion
                    && item.success_rate() >= self.policy.min_success_rate
                {
                    // Met all criteria → promote
                    item.status = AcceptanceStatus::Accepted;
                } else if remaining == 0 {
                    // Probation period exhausted
                    if item.success_rate() >= self.policy.min_success_rate
                        && item.total_uses() >= self.policy.min_uses_before_promotion
                    {
                        item.status = AcceptanceStatus::Accepted;
                    } else {
                        item.status = AcceptanceStatus::Rejected;
                    }
                } else {
                    item.status = AcceptanceStatus::Probation {
                        uses_remaining: remaining,
                    };
                }
            }
            _ => {} // Accepted/Rejected handled above
        }

        Some(item.status.clone())
    }

    /// Get the current status of an item.
    pub fn status(&self, id: &str) -> Option<&AcceptanceStatus> {
        self.items.iter().find(|i| i.id == id).map(|i| &i.status)
    }

    /// Get a tracked item by ID.
    pub fn get(&self, id: &str) -> Option<&TrackedItem> {
        self.items.iter().find(|i| i.id == id)
    }

    /// Get all items with a given status.
    pub fn items_with_status(&self, status_pred: impl Fn(&AcceptanceStatus) -> bool) -> Vec<&str> {
        self.items
            .iter()
            .filter(|i| status_pred(&i.status))
            .map(|i| i.id.as_str())
            .collect()
    }

    /// Get IDs of all accepted items.
    pub fn accepted(&self) -> Vec<&str> {
        self.items_with_status(|s| s.is_active())
    }

    /// Get IDs of all experimental (probation) items.
    pub fn experimental(&self) -> Vec<&str> {
        self.items_with_status(|s| s.is_experimental())
    }

    /// Get IDs of all rejected items.
    pub fn rejected(&self) -> Vec<&str> {
        self.items_with_status(|s| s.is_rejected())
    }

    /// Get all tracked items.
    pub fn items(&self) -> &[TrackedItem] {
        &self.items
    }

    /// Number of tracked items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Whether the tracker is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for VerificationTracker {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn strict_policy() -> AcceptancePolicy {
        AcceptancePolicy {
            min_uses_before_promotion: 5,
            min_success_rate: 0.7,
            probation_period: 10,
        }
    }

    fn lenient_policy() -> AcceptancePolicy {
        AcceptancePolicy {
            min_uses_before_promotion: 2,
            min_success_rate: 0.5,
            probation_period: 5,
        }
    }

    // --- AcceptanceStatus ---

    #[test]
    fn test_status_display() {
        assert_eq!(AcceptanceStatus::Candidate.to_string(), "Candidate");
        assert_eq!(
            AcceptanceStatus::Probation { uses_remaining: 5 }.to_string(),
            "Probation(5 uses left)"
        );
        assert_eq!(AcceptanceStatus::Accepted.to_string(), "Accepted");
        assert_eq!(AcceptanceStatus::Rejected.to_string(), "Rejected");
    }

    #[test]
    fn test_status_predicates() {
        assert!(!AcceptanceStatus::Candidate.is_active());
        assert!(!AcceptanceStatus::Candidate.is_experimental());
        assert!(!AcceptanceStatus::Candidate.is_rejected());

        assert!(!AcceptanceStatus::Probation { uses_remaining: 3 }.is_active());
        assert!(AcceptanceStatus::Probation { uses_remaining: 3 }.is_experimental());
        assert!(!AcceptanceStatus::Probation { uses_remaining: 3 }.is_rejected());

        assert!(AcceptanceStatus::Accepted.is_active());
        assert!(!AcceptanceStatus::Accepted.is_experimental());

        assert!(!AcceptanceStatus::Rejected.is_active());
        assert!(AcceptanceStatus::Rejected.is_rejected());
    }

    #[test]
    fn test_status_serde_roundtrip() {
        let statuses = vec![
            AcceptanceStatus::Candidate,
            AcceptanceStatus::Probation { uses_remaining: 7 },
            AcceptanceStatus::Accepted,
            AcceptanceStatus::Rejected,
        ];
        for status in statuses {
            let json = serde_json::to_string(&status).unwrap();
            let deserialized: AcceptanceStatus = serde_json::from_str(&json).unwrap();
            assert_eq!(deserialized, status);
        }
    }

    // --- Full lifecycle: Candidate → Probation → Accepted ---

    #[test]
    fn test_lifecycle_candidate_to_accepted() {
        let mut tracker = VerificationTracker::with_policy(lenient_policy());
        tracker.register("skill-001");

        // Candidate → first use → Probation
        let status = tracker
            .track_usage("skill-001", UseOutcome::Success)
            .unwrap();
        assert!(matches!(status, AcceptanceStatus::Probation { .. }));

        // More successes → hit min_uses_before_promotion (2) and min_success_rate (0.5)
        let status = tracker
            .track_usage("skill-001", UseOutcome::Success)
            .unwrap();
        assert_eq!(status, AcceptanceStatus::Accepted);
    }

    #[test]
    fn test_lifecycle_candidate_to_rejected() {
        let mut tracker = VerificationTracker::with_policy(AcceptancePolicy {
            min_uses_before_promotion: 3,
            min_success_rate: 0.8,
            probation_period: 3, // Short probation
        });
        tracker.register("skill-002");

        // All failures during probation
        tracker.track_usage("skill-002", UseOutcome::Failure); // Candidate → Probation(2)
        tracker.track_usage("skill-002", UseOutcome::Failure); // Probation(1)
        let status = tracker
            .track_usage("skill-002", UseOutcome::Failure)
            .unwrap(); // Probation(0) → Rejected
        assert_eq!(status, AcceptanceStatus::Rejected);
    }

    #[test]
    fn test_lifecycle_probation_exhausted_not_enough_uses() {
        let mut tracker = VerificationTracker::with_policy(AcceptancePolicy {
            min_uses_before_promotion: 10, // Very high
            min_success_rate: 0.5,
            probation_period: 3,
        });
        tracker.register("skill-003");

        // Successes but probation ends before min_uses reached
        tracker.track_usage("skill-003", UseOutcome::Success); // Candidate → Probation(2)
        tracker.track_usage("skill-003", UseOutcome::Success); // Probation(1)
        let status = tracker
            .track_usage("skill-003", UseOutcome::Success)
            .unwrap(); // Probation(0) → Rejected (not enough uses)
        assert_eq!(status, AcceptanceStatus::Rejected);
    }

    #[test]
    fn test_lifecycle_early_promotion() {
        let mut tracker = VerificationTracker::with_policy(AcceptancePolicy {
            min_uses_before_promotion: 3,
            min_success_rate: 0.6,
            probation_period: 10, // Long probation
        });
        tracker.register("skill-004");

        tracker.track_usage("skill-004", UseOutcome::Success); // Candidate → Probation(9)
        tracker.track_usage("skill-004", UseOutcome::Success); // Probation(8)
                                                               // 3 total uses, 3 successes = 100% → promoted early
        let status = tracker
            .track_usage("skill-004", UseOutcome::Success)
            .unwrap();
        assert_eq!(status, AcceptanceStatus::Accepted);
    }

    #[test]
    fn test_terminal_states_dont_change() {
        let mut tracker = VerificationTracker::with_policy(lenient_policy());
        tracker.register("accepted");
        // Fast-track to Accepted
        tracker.track_usage("accepted", UseOutcome::Success);
        tracker.track_usage("accepted", UseOutcome::Success);
        assert_eq!(
            tracker.status("accepted"),
            Some(&AcceptanceStatus::Accepted)
        );

        // Further uses don't change status
        tracker.track_usage("accepted", UseOutcome::Failure);
        assert_eq!(
            tracker.status("accepted"),
            Some(&AcceptanceStatus::Accepted)
        );

        // But they do get recorded
        let item = tracker.get("accepted").unwrap();
        assert_eq!(item.failure_count, 1);
    }

    #[test]
    fn test_rejected_still_records_usage() {
        let mut tracker = VerificationTracker::with_policy(AcceptancePolicy {
            min_uses_before_promotion: 5,
            min_success_rate: 0.9,
            probation_period: 2,
        });
        tracker.register("rejected");

        tracker.track_usage("rejected", UseOutcome::Failure); // → Probation(1)
        tracker.track_usage("rejected", UseOutcome::Failure); // → Rejected
        assert_eq!(
            tracker.status("rejected"),
            Some(&AcceptanceStatus::Rejected)
        );

        // Usage after rejection is still recorded
        tracker.track_usage("rejected", UseOutcome::Success);
        let item = tracker.get("rejected").unwrap();
        assert_eq!(item.success_count, 1);
        assert_eq!(item.failure_count, 2);
        assert!(item.status.is_rejected());
    }

    // --- VerificationTracker operations ---

    #[test]
    fn test_register_is_idempotent() {
        let mut tracker = VerificationTracker::new();
        tracker.register("skill-001");
        tracker.register("skill-001"); // duplicate
        assert_eq!(tracker.len(), 1);
    }

    #[test]
    fn test_track_unknown_id_returns_none() {
        let mut tracker = VerificationTracker::new();
        assert!(tracker
            .track_usage("nonexistent", UseOutcome::Success)
            .is_none());
    }

    #[test]
    fn test_query_methods() {
        let mut tracker = VerificationTracker::with_policy(lenient_policy());
        tracker.register("accepted");
        tracker.register("probation");
        tracker.register("rejected");
        tracker.register("candidate");

        // Fast-track accepted
        tracker.track_usage("accepted", UseOutcome::Success);
        tracker.track_usage("accepted", UseOutcome::Success);

        // Move to probation
        tracker.track_usage("probation", UseOutcome::Success);

        // Reject
        tracker.track_usage("rejected", UseOutcome::Failure);
        tracker.track_usage("rejected", UseOutcome::Failure);
        tracker.track_usage("rejected", UseOutcome::Failure);
        tracker.track_usage("rejected", UseOutcome::Failure);
        tracker.track_usage("rejected", UseOutcome::Failure);

        assert_eq!(tracker.accepted(), vec!["accepted"]);
        assert_eq!(tracker.experimental(), vec!["probation"]);
        assert_eq!(tracker.rejected(), vec!["rejected"]);
    }

    #[test]
    fn test_default_policy() {
        let policy = AcceptancePolicy::default();
        assert_eq!(policy.min_uses_before_promotion, 5);
        assert!((policy.min_success_rate - 0.7).abs() < f64::EPSILON);
        assert_eq!(policy.probation_period, 10);
    }

    #[test]
    fn test_tracked_item_serde_roundtrip() {
        let item = TrackedItem {
            id: "test-001".into(),
            status: AcceptanceStatus::Probation { uses_remaining: 5 },
            success_count: 3,
            failure_count: 1,
        };
        let json = serde_json::to_string(&item).unwrap();
        let deserialized: TrackedItem = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.id, "test-001");
        assert_eq!(deserialized.success_count, 3);
        assert_eq!(
            deserialized.status,
            AcceptanceStatus::Probation { uses_remaining: 5 }
        );
    }

    #[test]
    fn test_mixed_outcomes_with_strict_policy() {
        let mut tracker = VerificationTracker::with_policy(strict_policy());
        tracker.register("mixed");

        // 5 successes, 2 failures = 71.4% success rate (just above 70%)
        tracker.track_usage("mixed", UseOutcome::Success); // → Probation(9)
        tracker.track_usage("mixed", UseOutcome::Success); // Probation(8)
        tracker.track_usage("mixed", UseOutcome::Failure); // Probation(7)
        tracker.track_usage("mixed", UseOutcome::Success); // Probation(6)
        tracker.track_usage("mixed", UseOutcome::Failure); // Probation(5)
        tracker.track_usage("mixed", UseOutcome::Success); // 6 uses, 4/6=66% < 70%
                                                           // 6 total, 4 success = 66.7% — not enough yet
        let item = tracker.get("mixed").unwrap();
        assert!(matches!(item.status, AcceptanceStatus::Probation { .. }));

        tracker.track_usage("mixed", UseOutcome::Success); // 7 total, 5/7=71.4% >= 70%
        let status = tracker.status("mixed").unwrap();
        assert_eq!(*status, AcceptanceStatus::Accepted);
    }
}
