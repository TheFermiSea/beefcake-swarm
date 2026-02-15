//! Blind code reviewer agent.

use rig::client::CompletionClient;
use rig::providers::openai;

use crate::prompts;

use super::coder::OaiAgent;

/// Build the blind reviewer agent.
///
/// NO tools — the reviewer only sees a diff passed via prompt.
/// Returns PASS or FAIL on the first line.
pub fn build_reviewer(client: &openai::CompletionsClient, model: &str) -> OaiAgent {
    client
        .agent(model)
        .name("reviewer")
        .description("Blind code reviewer. Returns PASS/FAIL with structured feedback.")
        .preamble(prompts::REVIEWER_PREAMBLE)
        .temperature(0.1)
        .build()
}

/// Parse a reviewer response into pass/fail + feedback.
pub struct ReviewResult {
    pub passed: bool,
    pub feedback: String,
}

impl ReviewResult {
    pub fn parse(response: &str) -> Self {
        let passed = response
            .lines()
            .next()
            .map(|line| line.trim().to_uppercase().starts_with("PASS"))
            .unwrap_or(false);
        Self {
            passed,
            feedback: response.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_pass() {
        let result = ReviewResult::parse("PASS\n- Looks good\n- Minor style nit");
        assert!(result.passed);
        assert!(result.feedback.contains("Looks good"));
    }

    #[test]
    fn test_parse_fail() {
        let result = ReviewResult::parse("FAIL\n- Missing error handling\n- Unsafe unwrap");
        assert!(!result.passed);
        assert!(result.feedback.contains("Missing error handling"));
    }

    #[test]
    fn test_parse_empty() {
        let result = ReviewResult::parse("");
        assert!(!result.passed);
    }

    #[test]
    fn test_parse_lowercase_pass() {
        let result = ReviewResult::parse("pass\n- All good");
        assert!(result.passed);
    }

    #[test]
    fn test_parse_mixed_case() {
        let result = ReviewResult::parse("  Pass  \n- Approved");
        assert!(result.passed);
    }

    #[test]
    fn test_parse_fail_without_prefix() {
        let result = ReviewResult::parse("This code has issues\n- Bug on line 5");
        assert!(!result.passed);
    }
}
