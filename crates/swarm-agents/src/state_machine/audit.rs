use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::{is_legal_transition, OrchestratorState, StateMachine, TransitionRecord};

// ──────────────────────────────────────────────────────────────────────────────
// Transition Audit Log — structured export for post-run reasoning
// ──────────────────────────────────────────────────────────────────────────────

/// A structured audit report of a state machine run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditReport {
    /// Final state of the run.
    pub final_state: OrchestratorState,
    /// Total number of transitions.
    pub transition_count: usize,
    /// Final iteration number.
    pub iteration: u32,
    /// Total wall-clock duration in milliseconds.
    pub total_elapsed_ms: u64,
    /// Number of times each state was entered.
    pub state_visit_counts: HashMap<OrchestratorState, u32>,
    /// Number of retry loops (Verifying/Validating → Implementing).
    pub retry_count: u32,
    /// Number of escalations (→ Escalating).
    pub escalation_count: u32,
    /// The full ordered transition log.
    pub transitions: Vec<TransitionRecord>,
    /// Any invariant violations detected.
    pub invariant_violations: Vec<String>,
}

impl StateMachine {
    /// Generate a structured audit report from the state machine's history.
    pub fn audit_report(&self) -> AuditReport {
        let mut state_visits: HashMap<OrchestratorState, u32> = HashMap::new();
        let mut retry_count = 0u32;
        let mut escalation_count = 0u32;

        for t in &self.transitions {
            *state_visits.entry(t.to).or_insert(0) += 1;

            // Count retries: going back to Implementing from Verifying/Validating
            if t.to == OrchestratorState::Implementing
                && (t.from == OrchestratorState::Verifying
                    || t.from == OrchestratorState::Validating)
            {
                retry_count += 1;
            }

            // Count escalations
            if t.to == OrchestratorState::Escalating {
                escalation_count += 1;
            }
        }

        let violations = check_invariants(&self.transitions, self.current);

        AuditReport {
            final_state: self.current,
            transition_count: self.transitions.len(),
            iteration: self.iteration,
            total_elapsed_ms: self.created_at.elapsed().as_millis() as u64,
            state_visit_counts: state_visits,
            retry_count,
            escalation_count,
            transitions: self.transitions.clone(),
            invariant_violations: violations,
        }
    }

    /// Export the transition log as a JSON string.
    pub fn export_transitions_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(&self.transitions)
    }
}

impl fmt::Display for AuditReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== State Machine Audit Report ===")?;
        writeln!(f, "Final state: {}", self.final_state)?;
        writeln!(f, "Transitions: {}", self.transition_count)?;
        writeln!(f, "Iterations: {}", self.iteration)?;
        writeln!(f, "Retries: {}", self.retry_count)?;
        writeln!(f, "Escalations: {}", self.escalation_count)?;

        if !self.invariant_violations.is_empty() {
            writeln!(f, "VIOLATIONS ({}):", self.invariant_violations.len())?;
            for v in &self.invariant_violations {
                writeln!(f, "  - {v}")?;
            }
        } else {
            writeln!(f, "Invariants: all passed")?;
        }

        writeln!(f, "--- Transition Log ---")?;
        for (i, t) in self.transitions.iter().enumerate() {
            let reason = t.reason.as_deref().unwrap_or("");
            writeln!(
                f,
                "  [{i}] {} → {} (iter={}, +{}ms) {reason}",
                t.from, t.to, t.iteration, t.elapsed_ms
            )?;
        }
        Ok(())
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// State Invariants — assertions that should hold for any valid run
// ──────────────────────────────────────────────────────────────────────────────

/// Check structural invariants on a completed transition log.
///
/// Returns a list of violation descriptions (empty = all invariants hold).
pub fn check_invariants(
    transitions: &[TransitionRecord],
    final_state: OrchestratorState,
) -> Vec<String> {
    let mut violations = Vec::new();

    // INV-1: Every transition must be legal according to the transition table.
    for (i, t) in transitions.iter().enumerate() {
        if !is_legal_transition(t.from, t.to) {
            violations.push(format!(
                "INV-1 (legal transitions): transition [{i}] {} → {} is illegal",
                t.from, t.to
            ));
        }
    }

    // INV-2: Terminal states are absorbing — no transitions after Resolved/Failed.
    let mut saw_terminal = false;
    for (i, t) in transitions.iter().enumerate() {
        if saw_terminal {
            violations.push(format!(
                "INV-2 (terminal absorbing): transition [{i}] {} → {} occurs after terminal state",
                t.from, t.to
            ));
        }
        if t.to.is_terminal() {
            saw_terminal = true;
        }
    }

    // INV-3: If run ended in terminal, the last transition should land on it.
    if final_state.is_terminal() && !transitions.is_empty() {
        if let Some(last) = transitions.last() {
            if last.to != final_state {
                violations.push(format!(
                    "INV-3 (terminal consistency): final_state={final_state} but last transition lands on {}",
                    last.to
                ));
            }
        }
    }

    // INV-4: Consecutive transitions must chain (each from == previous to).
    for window in transitions.windows(2) {
        if window[1].from != window[0].to {
            violations.push(format!(
                "INV-4 (chain continuity): expected from={} but got from={} at transition to {}",
                window[0].to, window[1].from, window[1].to
            ));
        }
    }

    // INV-5: First transition (if any) must start from SelectingIssue.
    if let Some(first) = transitions.first() {
        if first.from != OrchestratorState::SelectingIssue {
            violations.push(format!(
                "INV-5 (initial state): first transition starts from {} instead of SelectingIssue",
                first.from
            ));
        }
    }

    // INV-6: Iteration counter must be non-decreasing.
    for window in transitions.windows(2) {
        if window[1].iteration < window[0].iteration {
            violations.push(format!(
                "INV-6 (iteration monotonic): iteration decreased from {} to {} at {} → {}",
                window[0].iteration, window[1].iteration, window[1].from, window[1].to
            ));
        }
    }

    // INV-7: elapsed_ms must be non-decreasing.
    for window in transitions.windows(2) {
        if window[1].elapsed_ms < window[0].elapsed_ms {
            violations.push(format!(
                "INV-7 (time monotonic): elapsed_ms decreased from {} to {} at {} → {}",
                window[0].elapsed_ms, window[1].elapsed_ms, window[1].from, window[1].to
            ));
        }
    }

    violations
}
