use anyhow::Result;
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openai;

use crate::config::SwarmConfig;

/// The Implementer agent: writes code using the 72B reasoning tier.
///
/// In the 2-agent loop, the Implementer:
/// 1. Receives a task from the orchestrator
/// 2. Reads relevant files from the Gastown worktree
/// 3. Generates code changes via the 72B model
/// 4. Writes changes to disk
/// 5. Hands off to the Verifier (deterministic) then Validator (14B blind review)
pub struct Implementer {
    client: openai::CompletionsClient,
    model: String,
}

impl Implementer {
    pub fn new(config: &SwarmConfig) -> Result<Self> {
        let client = openai::CompletionsClient::builder()
            .api_key("not-needed")
            .base_url(&config.reasoning_endpoint.url)
            .build()?;

        Ok(Self {
            model: config.reasoning_endpoint.model.clone(),
            client,
        })
    }

    /// Implement a task described by the given prompt.
    /// Returns the generated code/diff as a string.
    pub async fn implement(&self, task_description: &str) -> Result<String> {
        let agent = self
            .client
            .agent(&self.model)
            .preamble("You are an expert Rust developer. Write clean, idiomatic code.")
            .build();

        let response: String = agent.prompt(task_description).await?;
        Ok(response)
    }
}
