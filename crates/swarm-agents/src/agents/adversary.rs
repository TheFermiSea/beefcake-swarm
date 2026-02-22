//! Adversarial Breaker agent for pre-merge red-teaming.
//!
//! After the verifier passes (all compilation and test gates green), the
//! Breaker agent receives ONLY the diff and public API signatures — no
//! implementation context — and attempts to break the implementation by
//! generating edge-case tests, boundary conditions, and invalid-state checks.
//!
//! If any generated test fails, the implementation is rejected and the
//! failing tests are fed back to the implementer.

use std::path::Path;

use rig::client::CompletionClient;
use rig::providers::openai;
use serde::Deserialize;

use crate::prompts;
use crate::tools::bundles::{self, WorkerRole};

use super::coder::OaiAgent;

const DEFAULT_BREAKER_MAX_TURNS: usize = 8;

fn breaker_max_turns() -> usize {
    std::env::var("SWARM_BREAKER_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_BREAKER_MAX_TURNS)
}

/// Build the adversarial breaker agent.
///
/// Tools: read_file, write_file, edit_file, list_files, run_command.
/// The breaker writes adversarial test files and runs them to find
/// implementation flaws that compilation gates miss.
pub fn build_breaker(client: &openai::CompletionsClient, model: &str, wt_path: &Path) -> OaiAgent {
    build_breaker_named(client, model, wt_path, "breaker", false)
}

/// Build the adversarial breaker with a custom agent name.
pub fn build_breaker_named(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    name: &str,
    proxy_tools: bool,
) -> OaiAgent {
    client
        .agent(model)
        .name(name)
        .description(
            "Adversarial red-team agent. Generates edge-case tests to break implementations.",
        )
        .preamble(prompts::BREAKER_PREAMBLE)
        .temperature(0.4)
        .tools(bundles::worker_tools(
            wt_path,
            WorkerRole::General,
            proxy_tools,
        ))
        .default_max_turns(breaker_max_turns())
        .build()
}

/// Verdict from the adversarial breaker's test run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BreakerVerdict {
    /// All adversarial tests passed — implementation is robust.
    Clean,
    /// One or more adversarial tests failed — implementation has flaws.
    Broken,
    /// Breaker could not generate meaningful tests for this diff.
    Inconclusive,
}

/// Structured result from the adversarial breaker agent.
#[derive(Debug, Clone, Deserialize)]
pub struct BreakerReport {
    /// Overall verdict.
    pub verdict: BreakerVerdict,
    /// Number of adversarial tests generated.
    pub tests_generated: u32,
    /// Number of tests that passed.
    pub tests_passed: u32,
    /// Number of tests that failed.
    pub tests_failed: u32,
    /// Descriptions of failing test cases (for feedback to implementer).
    pub failing_tests: Vec<FailingTest>,
    /// Attack strategies attempted.
    pub strategies_used: Vec<String>,
}

/// A single failing adversarial test case.
#[derive(Debug, Clone, Deserialize)]
pub struct FailingTest {
    /// Name of the test function.
    pub test_name: String,
    /// What the test was trying to break.
    pub attack_vector: String,
    /// The error/assertion message.
    pub failure_message: String,
    /// File path where the test was written.
    pub test_file: String,
}

/// Parse a breaker agent response into a structured report.
///
/// Falls back to a conservative "broken" verdict if the response
/// doesn't parse as valid JSON, since we can't trust an unstructured
/// adversarial result.
pub fn parse_breaker_response(response: &str) -> BreakerReport {
    match serde_json::from_str::<BreakerReport>(response) {
        Ok(report) => report,
        Err(_) => {
            // Check if response mentions any test failures
            let lower = response.to_lowercase();
            let has_failures = lower.contains("fail")
                || lower.contains("error")
                || lower.contains("panic")
                || lower.contains("assertion");

            BreakerReport {
                verdict: if has_failures {
                    BreakerVerdict::Broken
                } else {
                    BreakerVerdict::Inconclusive
                },
                tests_generated: 0,
                tests_passed: 0,
                tests_failed: if has_failures { 1 } else { 0 },
                failing_tests: if has_failures {
                    vec![FailingTest {
                        test_name: "unparsed_response".to_string(),
                        attack_vector: "unknown".to_string(),
                        failure_message: response.chars().take(500).collect::<String>(),
                        test_file: "unknown".to_string(),
                    }]
                } else {
                    vec![]
                },
                strategies_used: vec!["response_not_parseable".to_string()],
            }
        }
    }
}

/// Format a breaker report as feedback for the implementer.
///
/// When the breaker finds failures, this produces a concise summary
/// that the implementer can use to fix the issues.
pub fn format_feedback(report: &BreakerReport) -> String {
    if report.verdict == BreakerVerdict::Clean {
        return "Adversarial testing passed. No issues found.".to_string();
    }

    let mut lines = Vec::new();
    lines.push(format!(
        "ADVERSARIAL TESTING: {} of {} tests failed.",
        report.tests_failed, report.tests_generated
    ));

    for (i, test) in report.failing_tests.iter().enumerate() {
        lines.push(format!(
            "\n{}. {} (in {})\n   Attack: {}\n   Failure: {}",
            i + 1,
            test.test_name,
            test.test_file,
            test.attack_vector,
            test.failure_message,
        ));
    }

    if !report.strategies_used.is_empty() {
        lines.push(format!(
            "\nStrategies attempted: {}",
            report.strategies_used.join(", ")
        ));
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_clean_report() {
        let json = r#"{
            "verdict": "clean",
            "tests_generated": 5,
            "tests_passed": 5,
            "tests_failed": 0,
            "failing_tests": [],
            "strategies_used": ["boundary_values", "empty_inputs", "overflow"]
        }"#;

        let report = parse_breaker_response(json);
        assert_eq!(report.verdict, BreakerVerdict::Clean);
        assert_eq!(report.tests_generated, 5);
        assert_eq!(report.tests_passed, 5);
        assert_eq!(report.tests_failed, 0);
        assert!(report.failing_tests.is_empty());
        assert_eq!(report.strategies_used.len(), 3);
    }

    #[test]
    fn test_parse_valid_broken_report() {
        let json = r#"{
            "verdict": "broken",
            "tests_generated": 4,
            "tests_passed": 2,
            "tests_failed": 2,
            "failing_tests": [
                {
                    "test_name": "test_empty_input_panics",
                    "attack_vector": "empty input to parser",
                    "failure_message": "thread panicked at 'index out of bounds'",
                    "test_file": "tests/adversarial_parser.rs"
                },
                {
                    "test_name": "test_max_u64_overflow",
                    "attack_vector": "integer overflow on counter",
                    "failure_message": "assertion failed: counter <= MAX",
                    "test_file": "tests/adversarial_counter.rs"
                }
            ],
            "strategies_used": ["empty_inputs", "overflow", "boundary_values"]
        }"#;

        let report = parse_breaker_response(json);
        assert_eq!(report.verdict, BreakerVerdict::Broken);
        assert_eq!(report.tests_failed, 2);
        assert_eq!(report.failing_tests.len(), 2);
        assert_eq!(report.failing_tests[0].test_name, "test_empty_input_panics");
    }

    #[test]
    fn test_parse_invalid_response_with_failures() {
        let response = "I found several issues. The test_boundary failed with a panic.";
        let report = parse_breaker_response(response);
        assert_eq!(report.verdict, BreakerVerdict::Broken);
        assert_eq!(report.tests_failed, 1);
        assert!(!report.failing_tests.is_empty());
    }

    #[test]
    fn test_parse_invalid_response_no_failures() {
        let response = "All looks good, couldn't generate meaningful tests for this change.";
        let report = parse_breaker_response(response);
        assert_eq!(report.verdict, BreakerVerdict::Inconclusive);
        assert_eq!(report.tests_generated, 0);
        assert!(report.failing_tests.is_empty());
    }

    #[test]
    fn test_format_feedback_clean() {
        let report = BreakerReport {
            verdict: BreakerVerdict::Clean,
            tests_generated: 3,
            tests_passed: 3,
            tests_failed: 0,
            failing_tests: vec![],
            strategies_used: vec![],
        };
        let feedback = format_feedback(&report);
        assert!(feedback.contains("passed"));
        assert!(feedback.contains("No issues"));
    }

    #[test]
    fn test_format_feedback_broken() {
        let report = BreakerReport {
            verdict: BreakerVerdict::Broken,
            tests_generated: 3,
            tests_passed: 1,
            tests_failed: 2,
            failing_tests: vec![
                FailingTest {
                    test_name: "test_overflow".to_string(),
                    attack_vector: "integer overflow".to_string(),
                    failure_message: "assertion failed".to_string(),
                    test_file: "tests/adv.rs".to_string(),
                },
                FailingTest {
                    test_name: "test_empty".to_string(),
                    attack_vector: "empty input".to_string(),
                    failure_message: "panic: index out of bounds".to_string(),
                    test_file: "tests/adv.rs".to_string(),
                },
            ],
            strategies_used: vec!["overflow".to_string(), "empty_inputs".to_string()],
        };
        let feedback = format_feedback(&report);
        assert!(feedback.contains("2 of 3 tests failed"));
        assert!(feedback.contains("test_overflow"));
        assert!(feedback.contains("test_empty"));
        assert!(feedback.contains("integer overflow"));
        assert!(feedback.contains("overflow, empty_inputs"));
    }

    #[test]
    fn test_breaker_max_turns_default() {
        assert!(breaker_max_turns() > 0);
    }

    #[test]
    fn test_parse_inconclusive_report() {
        let json = r#"{
            "verdict": "inconclusive",
            "tests_generated": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "failing_tests": [],
            "strategies_used": ["diff_too_small"]
        }"#;

        let report = parse_breaker_response(json);
        assert_eq!(report.verdict, BreakerVerdict::Inconclusive);
        assert_eq!(report.tests_generated, 0);
    }
}
