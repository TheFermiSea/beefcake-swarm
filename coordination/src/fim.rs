//! FIM (Fill-in-the-Middle) endpoint support for localized single-point edits.
//!
//! This module provides an async function to call the llama-server /v1/completions
//! endpoint with a suffix parameter for FIM-style code completion.

use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Request body for FIM completion endpoint.
#[derive(Debug, Serialize, Deserialize)]
pub struct FimRequest {
    /// The prefix part (code before the insertion point).
    pub prompt: String,
    /// The suffix part (code after the insertion point).
    pub suffix: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
}

/// Response from the FIM completion endpoint.
#[derive(Debug, Deserialize)]
pub struct FimResponse {
    /// The generated completion text.
    pub choices: Vec<FimChoice>,
}

/// A single choice in the FIM response.
#[derive(Debug, Deserialize)]
pub struct FimChoice {
    /// The generated text.
    pub text: String,
}

/// Calls the llama-server /v1/completions endpoint with FIM parameters.
///
/// # Arguments
/// * `endpoint` - The base URL of the llama-server (e.g., "http://localhost:8080")
/// * `prefix` - The code before the insertion point
/// * `suffix` - The code after the insertion point
/// * `max_tokens` - Maximum number of tokens to generate
///
/// # Returns
/// The generated completion text, or an error if the request fails.
pub async fn complete_fim(
    endpoint: &str,
    prefix: &str,
    suffix: &str,
    max_tokens: u32,
) -> Result<String, anyhow::Error> {
    let client = Client::new();
    let url = format!("{}/v1/completions", endpoint.trim_end_matches('/'));

    let request_body = FimRequest {
        prompt: prefix.to_string(),
        suffix: suffix.to_string(),
        max_tokens,
    };

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await?
        .error_for_status()?;

    let fim_response: FimResponse = response.json().await?;

    fim_response
        .choices
        .into_iter()
        .next()
        .map(|c| c.text)
        .ok_or_else(|| anyhow::anyhow!("No completion choices returned"))
}
