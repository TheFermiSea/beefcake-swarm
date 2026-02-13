//! Cloud escalation agent (CLIAPIProxy on ai-proxy:8317).
//!
//! Used when local models are stuck after multiple failed attempts.
//! Text-only — no tools. Provides architectural guidance.

use anyhow::{Context, Result};
use rig::client::CompletionClient;
use rig::providers::openai;

use crate::config::CloudEndpoint;
use crate::prompts;

use super::coder::OaiAgent;

/// Build the cloud escalation agent.
///
/// Creates its own CompletionsClient with the CLIAPIProxy API key.
/// NO tools — text-only architectural guidance.
pub fn build_cloud_agent(endpoint: &CloudEndpoint) -> Result<OaiAgent> {
    let client = openai::CompletionsClient::builder()
        .api_key(&endpoint.api_key)
        .base_url(&endpoint.url)
        .build()
        .context("Failed to build cloud CompletionsClient")?;

    Ok(client
        .agent(&endpoint.model)
        .name("cloud_escalation")
        .description("Cloud AI for architectural decisions when local models are stuck")
        .preamble(prompts::CLOUD_PREAMBLE)
        .temperature(0.3)
        .build())
}
