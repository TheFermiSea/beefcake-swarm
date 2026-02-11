use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openai;

use crate::config::SwarmConfig;

/// The Validator agent: blind code review using the 14B fast tier.
///
/// "Blind" means the Validator does NOT see the Implementer's conversation
/// context — only the resulting diff. This prevents rubber-stamping.
///
/// Validation flow:
/// 1. Receive diff from the Implementer's changes
/// 2. Review for correctness, style, and potential issues
/// 3. Return pass/fail with feedback
pub struct Validator {
    client: openai::CompletionsClient,
    model: String,
}

/// Result of a validation pass.
pub struct ValidationResult {
    pub passed: bool,
    pub feedback: String,
}

impl Validator {
    pub fn new(config: &SwarmConfig) -> Result<Self> {
        let client = openai::CompletionsClient::builder()
            .api_key("not-needed")
            .base_url(&config.fast_endpoint.url)
            .build()?;

        Ok(Self {
            model: config.fast_endpoint.model.clone(),
            client,
        })
    }

    /// Validate a diff produced by the Implementer.
    /// Returns pass/fail with feedback.
    pub async fn validate(&self, diff: &str) -> Result<ValidationResult> {
        let agent = self
            .client
            .agent(&self.model)
            .preamble(
                "You are a code reviewer. Review the following diff for correctness, \
                 style issues, and potential bugs. Respond with PASS or FAIL followed \
                 by your reasoning.",
            )
            .build();

        let response: String = agent.prompt(diff).await?;

        let passed = response.to_uppercase().starts_with("PASS");
        Ok(ValidationResult {
            passed,
            feedback: response,
        })
    }
}
