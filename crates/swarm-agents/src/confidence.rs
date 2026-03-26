//! Confidence extraction from agent responses (uncertainty-aware LLM pattern).
//!
//! Parses self-reported confidence scores from worker responses. When confidence
//! is below a threshold, the orchestrator escalates to a higher-capability model
//! before wasting iterations on likely-futile attempts.

use tracing::debug;

/// Confidence level extracted from an agent response.
#[derive(Debug, Clone, Copy)]
pub struct AgentConfidence {
    /// Self-reported confidence score (0.0 = no confidence, 1.0 = certain).
    pub score: f64,
    /// Whether the score was explicitly stated vs inferred.
    pub explicit: bool,
}

/// Default confidence threshold below which escalation is recommended.
pub const DEFAULT_CONFIDENCE_THRESHOLD: f64 = 0.55;

/// Extract confidence from an agent response string.
///
/// Looks for patterns like:
/// - "confidence: 0.7" or "Confidence: 85%"
/// - "I'm not confident" / "I'm uncertain" / "I'm not sure"
/// - "CONFIDENCE_SCORE: 0.3"
///
/// Returns None if no confidence signal is found (don't escalate by default).
pub fn extract_confidence(response: &str) -> Option<AgentConfidence> {
    // Pattern 1: explicit numeric "confidence: X.Y" or "confidence: XY%"
    let lower = response.to_lowercase();

    for pattern in &["confidence:", "confidence_score:", "confidence ="] {
        if let Some(pos) = lower.find(pattern) {
            let after = &response[pos + pattern.len()..];
            let after = after.trim_start();
            // Try to parse a number
            let num_str: String = after
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.' || *c == '%')
                .collect();
            if let Some(score) = parse_confidence_value(&num_str) {
                debug!(score, pattern, "Extracted explicit confidence score");
                return Some(AgentConfidence {
                    score,
                    explicit: true,
                });
            }
        }
    }

    // Pattern 2: verbal uncertainty signals
    let uncertainty_phrases = [
        "i'm not confident",
        "i'm uncertain",
        "i'm not sure how to",
        "i don't know how to",
        "this is beyond my",
        "i cannot determine",
        "unable to determine",
        "i need more context",
        "this requires expertise",
    ];

    let certainty_phrases = [
        "i'm confident",
        "straightforward",
        "simple fix",
        "clear solution",
    ];

    let uncertainty_count = uncertainty_phrases
        .iter()
        .filter(|p| lower.contains(*p))
        .count();
    let certainty_count = certainty_phrases
        .iter()
        .filter(|p| lower.contains(*p))
        .count();

    if uncertainty_count > 0 && uncertainty_count > certainty_count {
        debug!(
            uncertainty_count,
            certainty_count, "Inferred low confidence from verbal uncertainty signals"
        );
        return Some(AgentConfidence {
            score: 0.3, // Low confidence from verbal signals
            explicit: false,
        });
    }

    None
}

/// Whether a worker response signals that the current approach is fundamentally
/// stuck and a pivot to a different strategy would be productive.
///
/// A pivot signal is distinct from low confidence — it means the worker
/// explicitly indicates a dead end, not just uncertainty. The orchestrator
/// uses this to create a `PivotDecision` in the `MutationRecord`.
#[derive(Debug, Clone)]
pub struct PivotSignal {
    /// Which iteration triggered the signal.
    pub iteration: usize,
    /// The specific phrase that triggered the signal.
    pub trigger_phrase: String,
}

/// Detect whether a response signals that the current approach is a dead end
/// requiring a strategic pivot rather than incremental refinement.
///
/// Returns `Some(PivotSignal)` if the response contains strong dead-end
/// indicators. Returns `None` if the approach appears to be making progress
/// or if confidence is just low (escalate rather than pivot in that case).
pub fn detect_pivot_needed(response: &str, iteration: usize) -> Option<PivotSignal> {
    let lower = response.to_lowercase();

    // Strong dead-end indicators — these suggest the fundamental approach
    // is wrong, not just that a specific fix didn't work.
    let pivot_phrases = [
        "fundamentally flawed",
        "wrong approach",
        "need to rethink",
        "completely different approach",
        "start over",
        "this approach won't work",
        "cannot be fixed this way",
        "architectural issue",
        "design problem",
        "the entire design needs",
        "we need to redesign",
        "the abstraction is wrong",
    ];

    for phrase in &pivot_phrases {
        if lower.contains(phrase) {
            debug!(trigger = phrase, "Detected pivot signal in response");
            return Some(PivotSignal {
                iteration,
                trigger_phrase: phrase.to_string(),
            });
        }
    }

    None
}

/// Detect "context anxiety" — premature wrap-up signals in agent responses.
///
/// Context anxiety occurs when the context window is filling up and the agent
/// starts rushing to finish and summarize rather than solving the actual problem.
/// This is a quality signal: detect it early and trigger a structured handoff
/// + context reset rather than letting the agent produce incomplete work.
///
/// Returns `true` if the response shows anxiety patterns without also showing
/// natural completion signals (like verifier-passing confirmation).
pub fn detect_context_anxiety(response: &str) -> bool {
    let lower = response.to_lowercase();

    // Signals that suggest the agent is wrapping up prematurely
    let anxiety_patterns = [
        "i think that covers",
        "this should be sufficient",
        "i'll wrap up here",
        "i believe this is complete",
        "let me summarize what i've done",
        "in summary, i've done",
        "i've addressed the main points",
        "that should handle most",
        "i'll leave the rest",
        "the remaining issues",
        "due to context limitations",
        "i'm running out of space",
        "given the length of this response",
        "to keep this response focused",
    ];

    // Natural completion signals — these counteract anxiety flags
    let completion_signals = [
        "all tests pass",
        "cargo check succeeds",
        "the implementation is complete",
        "fully implemented",
        "all requirements are met",
        "verified working",
    ];

    let anxiety_count = anxiety_patterns
        .iter()
        .filter(|p| lower.contains(*p))
        .count();

    let completion_count = completion_signals
        .iter()
        .filter(|p| lower.contains(*p))
        .count();

    // Flag anxiety only when there are more anxiety signals than completion signals
    if anxiety_count > 0 && anxiety_count > completion_count {
        debug!(
            anxiety_count,
            completion_count, "Context anxiety detected in agent response"
        );
        return true;
    }

    false
}

fn parse_confidence_value(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(pct) = s.strip_suffix('%') {
        pct.parse::<f64>().ok().map(|v| v / 100.0)
    } else {
        let v = s.parse::<f64>().ok()?;
        // If > 1.0, assume percentage
        if v > 1.0 {
            Some(v / 100.0)
        } else {
            Some(v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_explicit_decimal() {
        let r = extract_confidence("I'll try this approach. Confidence: 0.8").unwrap();
        assert!((r.score - 0.8).abs() < 0.01);
        assert!(r.explicit);
    }

    #[test]
    fn test_explicit_percentage() {
        let r = extract_confidence("CONFIDENCE_SCORE: 45%").unwrap();
        assert!((r.score - 0.45).abs() < 0.01);
        assert!(r.explicit);
    }

    #[test]
    fn test_verbal_uncertainty() {
        let r = extract_confidence("I'm not sure how to fix this borrow checker issue").unwrap();
        assert!(r.score < 0.5);
        assert!(!r.explicit);
    }

    #[test]
    fn test_no_signal() {
        assert!(extract_confidence("Here's the fix for the clippy warning").is_none());
    }

    #[test]
    fn test_confident_overrides_uncertain() {
        // "straightforward" certainty signal should suppress escalation
        assert!(
            extract_confidence("This is a straightforward fix, I'm confident").is_none()
                || extract_confidence("This is a straightforward fix, I'm confident")
                    .unwrap()
                    .score
                    > 0.5
        );
    }

    #[test]
    fn test_parse_value_edge_cases() {
        assert_eq!(parse_confidence_value("85%"), Some(0.85));
        assert_eq!(parse_confidence_value("0.7"), Some(0.7));
        assert_eq!(parse_confidence_value("70"), Some(0.7)); // >1.0 treated as percentage
        assert_eq!(parse_confidence_value(""), None);
    }

    #[test]
    fn test_detect_pivot_needed_fundamental_flaw() {
        let signal = detect_pivot_needed(
            "The current approach is fundamentally flawed. We need a completely different approach.",
            3,
        );
        assert!(signal.is_some());
        assert_eq!(signal.unwrap().iteration, 3);
    }

    #[test]
    fn test_detect_pivot_needed_none_for_minor_issues() {
        let signal = detect_pivot_needed(
            "There's a borrow checker error on line 42. I'll fix the lifetime annotation.",
            2,
        );
        assert!(signal.is_none());
    }

    #[test]
    fn test_detect_pivot_needed_architectural() {
        let signal = detect_pivot_needed(
            "This is an architectural issue. The entire design needs to change.",
            4,
        );
        assert!(signal.is_some());
    }

    #[test]
    fn test_context_anxiety_detected() {
        assert!(detect_context_anxiety(
            "I think that covers the main requirements. The remaining issues can be addressed later."
        ));
    }

    #[test]
    fn test_context_anxiety_suppressed_by_completion() {
        // Completion signal should suppress anxiety flag
        assert!(!detect_context_anxiety(
            "All tests pass and the implementation is complete. Cargo check succeeds."
        ));
    }

    #[test]
    fn test_context_anxiety_no_signal() {
        assert!(!detect_context_anxiety(
            "Fixed the borrow checker error in parser.rs by adding an explicit lifetime."
        ));
    }
}
