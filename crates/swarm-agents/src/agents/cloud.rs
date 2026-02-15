//! Cloud client builder (CLIAPIProxy on ai-proxy:8317).
//!
//! The cloud endpoint is now used as the Manager tier (see `manager.rs`).
//! This module is retained for building standalone cloud clients if needed.

use anyhow::{Context, Result};
use rig::client::CompletionClient;
use rig::providers::openai;

use crate::config::CloudEndpoint;

use super::coder::OaiAgent;

/// Build a standalone cloud agent (text-only, no tools).
///
/// Primarily used for testing. In production, the cloud model serves as
/// the Manager with full tool access (see `manager::build_cloud_manager`).
pub fn build_cloud_agent(endpoint: &CloudEndpoint) -> Result<OaiAgent> {
    let client = openai::CompletionsClient::builder()
        .api_key(&endpoint.api_key)
        .base_url(&endpoint.url)
        .build()
        .context("Failed to build cloud CompletionsClient")?;

    Ok(client
        .agent(&endpoint.model)
        .name("cloud_standalone")
        .description("Standalone cloud agent for testing")
        .temperature(0.3)
        .build())
}
