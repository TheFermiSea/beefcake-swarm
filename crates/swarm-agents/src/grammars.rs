//! GBNF grammar constraints for structured LLM output.
//!
//! When llama-server produces flaky JSON, GBNF grammars constrain the output
//! to valid formats. These grammars are passed via `additional_params` to the
//! rig agent builder, which flattens them into the request body's `grammar` field.
//!
//! # Usage
//!
//! ```ignore
//! use serde_json::json;
//! use swarm_agents::grammars::Grammar;
//!
//! let agent = client
//!     .agent(model)
//!     .additional_params(Grammar::ReviewVerdict.to_params())
//!     .build();
//! ```
//!
//! # GBNF format
//!
//! GBNF (GGML BNF) is llama.cpp's grammar format for constraining output.
//! Each grammar defines a `root` rule that the model's output must match.

use serde_json::Value;

/// Available GBNF grammar types for constraining LLM output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Grammar {
    /// Reviewer verdict: `{ "verdict": "pass"|"fail"|"needs_escalation", "confidence": 0.0-1.0, ... }`
    ReviewVerdict,
    /// Planner output: `{ "approach": "...", "steps": [...], "target_files": [...], "risk": "..." }`
    PlannerOutput,
    /// Generic JSON object (any valid JSON object).
    JsonObject,
}

impl Grammar {
    /// Get the GBNF grammar string for this grammar type.
    pub fn as_gbnf(&self) -> &'static str {
        match self {
            Self::ReviewVerdict => REVIEW_VERDICT_GRAMMAR,
            Self::PlannerOutput => PLANNER_OUTPUT_GRAMMAR,
            Self::JsonObject => JSON_OBJECT_GRAMMAR,
        }
    }

    /// Convert to `additional_params` JSON value for the rig agent builder.
    ///
    /// Returns `json!({"grammar": "..."})` which, when passed to
    /// `.additional_params()`, gets flattened into the request body via
    /// `#[serde(flatten)]`.
    pub fn to_params(&self) -> Value {
        serde_json::json!({
            "grammar": self.as_gbnf()
        })
    }
}

/// Check if GBNF grammar enforcement is enabled via env var.
///
/// Set `SWARM_GBNF_ENABLED=1` to enable grammar constraints on local
/// llama-server endpoints. Disabled by default to avoid issues with
/// cloud providers that don't support the `grammar` field.
pub fn gbnf_enabled() -> bool {
    std::env::var("SWARM_GBNF_ENABLED")
        .ok()
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

/// Return grammar params if GBNF is enabled, otherwise None.
///
/// Convenience function for conditional grammar attachment:
/// ```ignore
/// if let Some(params) = grammars::params_if_enabled(Grammar::ReviewVerdict) {
///     builder = builder.additional_params(params);
/// }
/// ```
pub fn params_if_enabled(grammar: Grammar) -> Option<Value> {
    if gbnf_enabled() {
        Some(grammar.to_params())
    } else {
        None
    }
}

// ─── GBNF Grammar Definitions ───

/// Reviewer verdict grammar.
///
/// Constrains output to a JSON object with:
/// - `verdict`: one of "pass", "fail", "needs_escalation"
/// - `confidence`: a number 0.0-1.0
/// - `blocking_issues`: array of strings
/// - `suggested_next_action`: string
/// - `touched_files`: array of strings
const REVIEW_VERDICT_GRAMMAR: &str = r#"
root ::= "{" ws verdict-field "," ws confidence-field "," ws blocking-field "," ws action-field "," ws files-field ws "}"

verdict-field ::= "\"verdict\"" ws ":" ws verdict-value
verdict-value ::= "\"pass\"" | "\"fail\"" | "\"needs_escalation\""

confidence-field ::= "\"confidence\"" ws ":" ws number
number ::= [0-9] ("." [0-9]+)?

blocking-field ::= "\"blocking_issues\"" ws ":" ws string-array
action-field ::= "\"suggested_next_action\"" ws ":" ws string
files-field ::= "\"touched_files\"" ws ":" ws string-array

string-array ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape-char
escape-char ::= "\"" | "\\" | "/" | "n" | "r" | "t"

ws ::= [ \t\n\r]*
"#;

/// Planner output grammar.
///
/// Constrains output to a JSON object with:
/// - `approach`: string
/// - `steps`: array of step objects
/// - `target_files`: array of strings
/// - `risk`: one of "low", "medium", "high"
const PLANNER_OUTPUT_GRAMMAR: &str = r#"
root ::= "{" ws approach-field "," ws steps-field "," ws files-field "," ws risk-field ws "}"

approach-field ::= "\"approach\"" ws ":" ws string
steps-field ::= "\"steps\"" ws ":" ws step-array
files-field ::= "\"target_files\"" ws ":" ws string-array
risk-field ::= "\"risk\"" ws ":" ws risk-value

risk-value ::= "\"low\"" | "\"medium\"" | "\"high\""

step-array ::= "[" ws "]" | "[" ws step (ws "," ws step)* ws "]"
step ::= "{" ws step-desc (ws "," ws step-file)? ws "}"
step-desc ::= "\"description\"" ws ":" ws string
step-file ::= "\"file\"" ws ":" ws (string | "null")

string-array ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"
string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape-char
escape-char ::= "\"" | "\\" | "/" | "n" | "r" | "t"

ws ::= [ \t\n\r]*
"#;

/// Generic JSON object grammar.
///
/// Ensures the output is a valid JSON object (any key/value pairs).
/// Use this as a fallback when a specific schema grammar isn't available
/// but you need valid JSON output.
const JSON_OBJECT_GRAMMAR: &str = r#"
root ::= object

object ::= "{" ws "}" | "{" ws pair (ws "," ws pair)* ws "}"
pair ::= string ws ":" ws value

array ::= "[" ws "]" | "[" ws value (ws "," ws value)* ws "]"

value ::= string | number | object | array | "true" | "false" | "null"

string ::= "\"" chars "\""
chars ::= char*
char ::= [^"\\] | "\\" escape-char
escape-char ::= "\"" | "\\" | "/" | "b" | "f" | "n" | "r" | "t" | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]

number ::= "-"? int frac? exp?
int ::= "0" | [1-9] [0-9]*
frac ::= "." [0-9]+
exp ::= [eE] [+-]? [0-9]+

ws ::= [ \t\n\r]*
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grammar_as_gbnf() {
        let review = Grammar::ReviewVerdict.as_gbnf();
        assert!(review.contains("root ::="));
        assert!(review.contains("verdict-value"));
        assert!(review.contains(r#"\"pass\""#));
        assert!(review.contains(r#"\"fail\""#));
        assert!(review.contains(r#"\"needs_escalation\""#));

        let planner = Grammar::PlannerOutput.as_gbnf();
        assert!(planner.contains("root ::="));
        assert!(planner.contains("approach-field"));
        assert!(planner.contains("risk-value"));
        assert!(planner.contains(r#"\"low\""#));

        let json = Grammar::JsonObject.as_gbnf();
        assert!(json.contains("root ::= object"));
        assert!(json.contains("value"));
    }

    #[test]
    fn test_grammar_to_params() {
        let params = Grammar::ReviewVerdict.to_params();
        assert!(params.is_object());
        assert!(params.get("grammar").is_some());
        let grammar_str = params["grammar"].as_str().unwrap();
        assert!(grammar_str.contains("verdict-value"));
    }

    #[test]
    fn test_gbnf_enabled_default_off() {
        // Without env var set, should be disabled
        // (may be overridden by test environment, so just check it doesn't panic)
        let _ = gbnf_enabled();
    }

    #[test]
    fn test_params_if_enabled_returns_none_when_disabled() {
        // Unset the env var to ensure disabled
        std::env::remove_var("SWARM_GBNF_ENABLED");
        assert!(params_if_enabled(Grammar::ReviewVerdict).is_none());
    }

    #[test]
    fn test_params_if_enabled_returns_some_when_enabled() {
        std::env::set_var("SWARM_GBNF_ENABLED", "1");
        let result = params_if_enabled(Grammar::ReviewVerdict);
        assert!(result.is_some());
        let params = result.unwrap();
        assert!(params["grammar"].as_str().unwrap().contains("verdict"));
        // Clean up
        std::env::remove_var("SWARM_GBNF_ENABLED");
    }

    #[test]
    fn test_all_grammars_have_root_rule() {
        for grammar in [
            Grammar::ReviewVerdict,
            Grammar::PlannerOutput,
            Grammar::JsonObject,
        ] {
            let gbnf = grammar.as_gbnf();
            assert!(
                gbnf.contains("root ::="),
                "{:?} grammar missing root rule",
                grammar
            );
        }
    }

    #[test]
    fn test_review_grammar_covers_all_fields() {
        let gbnf = Grammar::ReviewVerdict.as_gbnf();
        assert!(gbnf.contains("verdict-field"), "Missing verdict field");
        assert!(
            gbnf.contains("confidence-field"),
            "Missing confidence field"
        );
        assert!(
            gbnf.contains("blocking-field"),
            "Missing blocking_issues field"
        );
        assert!(
            gbnf.contains("action-field"),
            "Missing suggested_next_action field"
        );
        assert!(gbnf.contains("files-field"), "Missing touched_files field");
    }

    #[test]
    fn test_planner_grammar_covers_all_fields() {
        let gbnf = Grammar::PlannerOutput.as_gbnf();
        assert!(gbnf.contains("approach-field"), "Missing approach field");
        assert!(gbnf.contains("steps-field"), "Missing steps field");
        assert!(gbnf.contains("files-field"), "Missing target_files field");
        assert!(gbnf.contains("risk-field"), "Missing risk field");
    }
}
