//! Bounded summarizer — strict prompt/response contract for context compaction.
//!
//! Defines the summarizer interface and a deterministic mock for testing.
//! The real implementation delegates to an LLM via the agent harness.

use serde::{Deserialize, Serialize};

use super::errors::{CompactionError, CompactionErrorKind, SummarizationError};
use super::store::MemoryEntry;

/// Contract for summarizer input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryRequest {
    /// Entries to summarize.
    pub entries: Vec<SummaryInputEntry>,
    /// Maximum tokens for the summary output.
    pub max_output_tokens: u32,
    /// Required sections in the summary.
    pub required_sections: Vec<String>,
    /// Context about what the debate/session is about.
    pub session_context: String,
}

/// Simplified entry for summarizer input (no internal metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryInputEntry {
    /// Source of the entry (agent name, tool, etc.).
    pub source: String,
    /// The content text.
    pub content: String,
    /// Estimated tokens.
    pub tokens: u32,
}

impl SummaryInputEntry {
    /// Create from a MemoryEntry.
    pub fn from_memory_entry(entry: &MemoryEntry) -> Self {
        Self {
            source: entry.source.clone(),
            content: entry.content.clone(),
            tokens: entry.estimated_tokens,
        }
    }
}

/// Contract for summarizer output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryResponse {
    /// The summary text.
    pub summary: String,
    /// Estimated tokens in the summary.
    pub summary_tokens: u32,
    /// Sections that were included.
    pub sections_included: Vec<String>,
    /// Number of input entries that were summarized.
    pub entries_summarized: usize,
    /// Total input tokens that were compressed.
    pub input_tokens_compressed: u64,
    /// Compression ratio (input / output).
    pub compression_ratio: f64,
}

impl SummaryResponse {
    /// Validate the response against the request contract.
    pub fn validate(&self, request: &SummaryRequest) -> Result<(), CompactionError> {
        // Check token budget
        if self.summary_tokens > request.max_output_tokens {
            return Err(CompactionError::new(
                CompactionErrorKind::SummaryTooLarge,
                &format!(
                    "summary {} tokens exceeds budget {} tokens",
                    self.summary_tokens, request.max_output_tokens
                ),
            ));
        }

        // Check required sections
        for section in &request.required_sections {
            if !self.sections_included.contains(section) {
                return Err(CompactionError::new(
                    CompactionErrorKind::IntegrityViolation,
                    &format!("required section missing: {}", section),
                ));
            }
        }

        // Check summary is non-empty
        if self.summary.trim().is_empty() {
            return Err(CompactionError::new(
                CompactionErrorKind::SummarizationFailed,
                "summary is empty",
            ));
        }

        Ok(())
    }
}

/// Trait for summarizers.
pub trait Summarizer {
    /// Summarize the given entries according to the request contract.
    fn summarize(&self, request: &SummaryRequest) -> Result<SummaryResponse, SummarizationError>;
}

/// Deterministic mock summarizer for testing.
///
/// Produces a structured summary by concatenating source names
/// and truncating to the token budget.
pub struct MockSummarizer {
    /// Simulated model name.
    pub model_name: String,
    /// Whether to simulate failure.
    pub should_fail: bool,
    /// Whether to produce oversize output.
    pub produce_oversize: bool,
}

impl MockSummarizer {
    /// Create a working mock summarizer.
    pub fn new() -> Self {
        Self {
            model_name: "mock-summarizer".to_string(),
            should_fail: false,
            produce_oversize: false,
        }
    }

    /// Create a mock that always fails.
    pub fn failing() -> Self {
        Self {
            model_name: "mock-summarizer-fail".to_string(),
            should_fail: true,
            produce_oversize: false,
        }
    }

    /// Create a mock that produces oversize output.
    pub fn oversize() -> Self {
        Self {
            model_name: "mock-summarizer-oversize".to_string(),
            should_fail: false,
            produce_oversize: true,
        }
    }
}

impl Default for MockSummarizer {
    fn default() -> Self {
        Self::new()
    }
}

impl Summarizer for MockSummarizer {
    fn summarize(&self, request: &SummaryRequest) -> Result<SummaryResponse, SummarizationError> {
        if self.should_fail {
            return Err(SummarizationError::new(
                &self.model_name,
                "simulated failure",
                request.entries.len(),
                request.entries.iter().map(|e| e.tokens as u64).sum::<u64>(),
            ));
        }

        let input_tokens: u64 = request.entries.iter().map(|e| e.tokens as u64).sum();

        // Build summary from entry sources
        let mut summary_parts: Vec<String> = Vec::new();
        summary_parts.push(format!(
            "## Session Summary\n\nContext: {}",
            request.session_context
        ));

        for section in &request.required_sections {
            summary_parts.push(format!("\n### {}", section));
        }

        summary_parts.push(format!("\n### Entries ({})\n", request.entries.len()));

        for entry in &request.entries {
            summary_parts.push(format!(
                "- [{}]: {}",
                entry.source,
                truncate(&entry.content, 80)
            ));
        }

        let summary = summary_parts.join("\n");

        let summary_tokens = if self.produce_oversize {
            request.max_output_tokens + 100
        } else {
            // Rough estimate: 1 token per 4 chars
            (summary.len() as u32 / 4).min(request.max_output_tokens)
        };

        Ok(SummaryResponse {
            summary,
            summary_tokens,
            sections_included: request.required_sections.clone(),
            entries_summarized: request.entries.len(),
            input_tokens_compressed: input_tokens,
            compression_ratio: if summary_tokens > 0 {
                input_tokens as f64 / summary_tokens as f64
            } else {
                0.0
            },
        })
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        format!("{}...", &s[..max])
    }
}

/// Build a SummaryRequest from memory entries.
pub fn build_summary_request(
    entries: &[&MemoryEntry],
    max_output_tokens: u32,
    session_context: &str,
) -> SummaryRequest {
    SummaryRequest {
        entries: entries
            .iter()
            .map(|e| SummaryInputEntry::from_memory_entry(e))
            .collect(),
        max_output_tokens,
        required_sections: vec![
            "Key Decisions".to_string(),
            "Current State".to_string(),
            "Open Issues".to_string(),
        ],
        session_context: session_context.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::store::MemoryEntryKind;
    use super::*;

    fn make_entries() -> Vec<MemoryEntry> {
        vec![
            MemoryEntry::new(
                MemoryEntryKind::AgentTurn,
                "Implemented feature X",
                "coder",
                100,
            ),
            MemoryEntry::new(
                MemoryEntryKind::AgentTurn,
                "Found borrow checker issue",
                "reviewer",
                80,
            ),
            MemoryEntry::new(
                MemoryEntryKind::ToolResult,
                "cargo test: 5 passed",
                "verifier",
                30,
            ),
        ]
    }

    #[test]
    fn test_mock_summarizer_success() {
        let summarizer = MockSummarizer::new();
        let entries = make_entries();
        let request = build_summary_request(
            &entries.iter().collect::<Vec<_>>(),
            500,
            "debugging session",
        );
        let response = summarizer.summarize(&request).unwrap();

        assert!(!response.summary.is_empty());
        assert_eq!(response.entries_summarized, 3);
        assert!(response.input_tokens_compressed > 0);
        assert!(response.summary_tokens <= 500);
        assert!(response.validate(&request).is_ok());
    }

    #[test]
    fn test_mock_summarizer_failure() {
        let summarizer = MockSummarizer::failing();
        let entries = make_entries();
        let request = build_summary_request(&entries.iter().collect::<Vec<_>>(), 500, "test");
        let err = summarizer.summarize(&request).unwrap_err();
        assert_eq!(err.model, "mock-summarizer-fail");
        assert!(err.retryable);
    }

    #[test]
    fn test_mock_summarizer_oversize() {
        let summarizer = MockSummarizer::oversize();
        let entries = make_entries();
        let request = build_summary_request(&entries.iter().collect::<Vec<_>>(), 50, "test");
        let response = summarizer.summarize(&request).unwrap();

        // Validate should fail because summary is oversize
        let err = response.validate(&request).unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::SummaryTooLarge);
    }

    #[test]
    fn test_validate_missing_section() {
        let response = SummaryResponse {
            summary: "Some content".to_string(),
            summary_tokens: 10,
            sections_included: vec!["Key Decisions".to_string()],
            entries_summarized: 1,
            input_tokens_compressed: 100,
            compression_ratio: 10.0,
        };
        let request = SummaryRequest {
            entries: vec![],
            max_output_tokens: 100,
            required_sections: vec!["Key Decisions".to_string(), "Current State".to_string()],
            session_context: "test".to_string(),
        };
        let err = response.validate(&request).unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::IntegrityViolation);
    }

    #[test]
    fn test_validate_empty_summary() {
        let response = SummaryResponse {
            summary: "   ".to_string(),
            summary_tokens: 1,
            sections_included: vec![],
            entries_summarized: 0,
            input_tokens_compressed: 0,
            compression_ratio: 0.0,
        };
        let request = SummaryRequest {
            entries: vec![],
            max_output_tokens: 100,
            required_sections: vec![],
            session_context: "test".to_string(),
        };
        let err = response.validate(&request).unwrap_err();
        assert_eq!(err.kind, CompactionErrorKind::SummarizationFailed);
    }

    #[test]
    fn test_summary_request_serde() {
        let entries = make_entries();
        let request = build_summary_request(&entries.iter().collect::<Vec<_>>(), 500, "test");
        let json = serde_json::to_string(&request).unwrap();
        let parsed: SummaryRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.entries.len(), 3);
        assert_eq!(parsed.max_output_tokens, 500);
    }

    #[test]
    fn test_summary_response_serde() {
        let summarizer = MockSummarizer::new();
        let entries = make_entries();
        let request = build_summary_request(&entries.iter().collect::<Vec<_>>(), 500, "test");
        let response = summarizer.summarize(&request).unwrap();
        let json = serde_json::to_string(&response).unwrap();
        let parsed: SummaryResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.entries_summarized, 3);
    }

    #[test]
    fn test_truncate_utility() {
        assert_eq!(truncate("hello", 10), "hello");
        assert_eq!(truncate("hello world", 5), "hello...");
    }
}
