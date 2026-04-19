//! FIM (Fill-in-the-Middle) endpoint support for localized single-point edits.
//!
//! This module provides an async function to call the llama-server /v1/completions
//! endpoint with a suffix parameter for FIM-style code completion.

use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Request body for FIM completion endpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FimRequest {
    /// The prefix part (code before the insertion point).
    pub prompt: String,
    /// The suffix part (code after the insertion point).
    pub suffix: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: u32,
}

/// Response from the FIM completion endpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct FimResponse {
    /// The generated completion text.
    pub choices: Vec<FimChoice>,
}

/// A single choice in the FIM response.
#[derive(Debug, Clone, Deserialize)]
pub struct FimChoice {
    /// The generated text.
    pub text: String,
}
