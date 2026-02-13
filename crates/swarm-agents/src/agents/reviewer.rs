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
