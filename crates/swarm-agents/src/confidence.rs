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
}
