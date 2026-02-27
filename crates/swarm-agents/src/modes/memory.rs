//! NS-5: Memory manager and token-aware context compaction.
//!
//! Provides `MemoryManager` — a shared utility consumed by all three modes.
//! It tracks token load, triggers compaction, and manages the history window.
//!
//! ## Design
//!
//! - Token estimation: character count ÷ 4 (fast, provider-agnostic).
//! - Compaction trigger: when `estimated_tokens ≥ threshold_tokens` AND
//!   `history.len() ≥ min_messages_before_compaction`.
//! - Compaction strategy: summarise the oldest half of the window, keep the
//!   newest N messages intact to preserve immediate context.
//! - Reconstruction invariant: summary message always appears first after
//!   compaction, followed by the retained tail.
//!
//! ## Token estimator pluggability
//!
//! `TokenEstimator` is a trait so callers can supply a tiktoken-backed or
//! model-specific implementation without changing the manager.

use std::collections::VecDeque;

use async_trait::async_trait;
use rig::client::CompletionClient;
use rig::completion::{Message, Prompt};
use tracing::{debug, info};

use crate::modes::{
    errors::OrchestrationError,
    provider_config::{CompactionConfig, ModeRunnerConfig},
    types::CompactionSummary,
};

// ── TokenEstimator ────────────────────────────────────────────────────────────

/// Pluggable token counting strategy.
pub trait TokenEstimator: Send + Sync {
    /// Estimate the number of tokens in `text`.
    fn estimate(&self, text: &str) -> u64;
}

/// Default estimator: 1 token ≈ 4 characters.
pub struct CharCountEstimator;

impl TokenEstimator for CharCountEstimator {
    fn estimate(&self, text: &str) -> u64 {
        (text.len() as u64) / 4
    }
}

// ── SummarizerAgent ───────────────────────────────────────────────────────────

/// Drives the compaction LLM call.
///
/// Trait exists so tests can inject a stub without needing an inference endpoint.
#[async_trait]
pub trait SummarizerAgent: Send + Sync {
    async fn summarise(&self, history_text: &str) -> Result<String, OrchestrationError>;
}

/// Production implementation backed by a Rig agent.
pub struct RigSummarizer {
    config: ModeRunnerConfig,
}

impl RigSummarizer {
    pub fn new(config: ModeRunnerConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl SummarizerAgent for RigSummarizer {
    async fn summarise(&self, history_text: &str) -> Result<String, OrchestrationError> {
        let client = self.config.local_client().map_err(|e| {
            OrchestrationError::Configuration(format!("client build failed: {e}"))
        })?;

        let agent = client
            .agent(&self.config.models.compactor)
            .preamble(
                "You are a context compressor for an autonomous coding agent. \
                Summarize the conversation into dense bullet points. \
                Preserve: task objectives, constraints, error patterns, data structures, \
                key architectural decisions. \
                Discard: pleasantries, redundant re-statements, formatting discussions.",
            )
            .temperature(0.2)
            .build();

        agent
            .prompt(&format!("Summarize:\n\n{history_text}"))
            .await
            .map_err(|e| OrchestrationError::InferenceFailure(e.to_string()))
    }
}

// ── MemoryManager ─────────────────────────────────────────────────────────────

/// Token-budgeted context window manager.
///
/// Wraps the conversation history and handles compaction transparently.
pub struct MemoryManager {
    config: CompactionConfig,
    estimator: Box<dyn TokenEstimator>,
    summarizer: Box<dyn SummarizerAgent>,
    history: VecDeque<Message>,
    /// Running estimate of total tokens in `history`.
    estimated_tokens: u64,
    /// Total tokens compacted across all compaction runs (for telemetry).
    total_compacted: u64,
    /// Number of compaction runs performed.
    compaction_count: u32,
    /// Number of tail messages to keep after compaction.
    tail_keep: usize,
}

impl MemoryManager {
    /// Create with production summarizer backed by `config`.
    pub fn new(config: &ModeRunnerConfig) -> Self {
        Self {
            config: config.compaction.clone(),
            estimator: Box::new(CharCountEstimator),
            summarizer: Box::new(RigSummarizer::new(config.clone())),
            history: VecDeque::new(),
            estimated_tokens: 0,
            total_compacted: 0,
            compaction_count: 0,
            tail_keep: 4,
        }
    }

    /// Create with a custom token estimator and summarizer (for testing).
    pub fn with_components(
        config: CompactionConfig,
        estimator: Box<dyn TokenEstimator>,
        summarizer: Box<dyn SummarizerAgent>,
    ) -> Self {
        Self {
            config,
            estimator,
            summarizer,
            history: VecDeque::new(),
            estimated_tokens: 0,
            total_compacted: 0,
            compaction_count: 0,
            tail_keep: 4,
        }
    }

    /// Set how many tail messages are preserved intact during compaction.
    pub fn with_tail_keep(mut self, n: usize) -> Self {
        self.tail_keep = n;
        self
    }

    // ── Public API ────────────────────────────────────────────────────────

    /// Add a message to the history, updating the token estimate.
    pub fn push(&mut self, msg: Message) {
        let tokens = self.estimate_message(&msg);
        self.estimated_tokens += tokens;
        self.history.push_back(msg);
    }

    /// Return the full history as a `Vec<Message>` for use with Rig `.chat()`.
    pub fn history(&self) -> Vec<Message> {
        self.history.iter().cloned().collect()
    }

    /// Current estimated token count.
    pub fn estimated_tokens(&self) -> u64 {
        self.estimated_tokens
    }

    /// Whether the manager believes compaction should be triggered.
    pub fn should_compact(&self) -> bool {
        self.estimated_tokens >= self.config.trigger_threshold_tokens()
            && self.history.len() >= self.config.min_messages_before_compaction
    }

    /// Run compaction if needed; returns `Some(CompactionSummary)` if it ran.
    ///
    /// Callers should call this at the start of every iteration.
    pub async fn maybe_compact(&mut self) -> Result<Option<CompactionSummary>, OrchestrationError> {
        if !self.should_compact() {
            return Ok(None);
        }
        let summary = self.compact().await?;
        Ok(Some(summary))
    }

    /// Force a compaction regardless of threshold.
    pub async fn compact(&mut self) -> Result<CompactionSummary, OrchestrationError> {
        let tokens_before = self.estimated_tokens;
        let messages_before = self.history.len();

        // Split: compact the head, keep the tail intact.
        let keep = self.tail_keep.min(self.history.len());
        let compact_count = self.history.len().saturating_sub(keep);

        if compact_count == 0 {
            debug!("nothing to compact — too few messages");
            return Ok(CompactionSummary {
                summary: String::new(),
                tokens_compacted: 0,
                tokens_summary: 0,
                messages_compacted: 0,
            });
        }

        // Build text for the head segment.
        let head: Vec<&Message> = self.history.iter().take(compact_count).collect();
        let history_text = serialise_messages(&head);

        // Call the summarizer.
        let summary_text = self.summarizer.summarise(&history_text).await?;

        let tokens_summary = self.estimator.estimate(&summary_text);

        // Rebuild history: summary prefix + tail.
        let tail: Vec<Message> = self
            .history
            .iter()
            .skip(compact_count)
            .cloned()
            .collect();

        self.history.clear();
        self.estimated_tokens = 0;

        // Insert the summary as a user message so the LLM sees it as context.
        self.push(Message::user(format!(
            "[CONTEXT SUMMARY — {} messages, ~{} tokens compacted]\n{}",
            compact_count, tokens_before, summary_text
        )));

        for msg in tail {
            self.push(msg);
        }

        self.total_compacted += tokens_before;
        self.compaction_count += 1;

        let summary = CompactionSummary {
            summary: summary_text,
            tokens_compacted: tokens_before,
            tokens_summary,
            messages_compacted: messages_before,
        };

        info!(
            compaction = self.compaction_count,
            tokens_before,
            tokens_after = self.estimated_tokens,
            ratio = format!("{:.2}", summary.compression_ratio()),
            "compaction complete"
        );

        Ok(summary)
    }

    // ── Stats ─────────────────────────────────────────────────────────────

    pub fn total_compacted_tokens(&self) -> u64 {
        self.total_compacted
    }

    pub fn compaction_count(&self) -> u32 {
        self.compaction_count
    }

    // ── Internal helpers ──────────────────────────────────────────────────

    fn estimate_message(&self, msg: &Message) -> u64 {
        let text = message_text(msg);
        self.estimator.estimate(&text)
    }
}

fn message_text(msg: &Message) -> String {
    use rig::completion::AssistantContent;
    use rig::message::UserContent;
    match msg {
        Message::User { content } => content
            .iter()
            .filter_map(|c| {
                if let UserContent::Text(t) = c {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        Message::Assistant { content, .. } => content
            .iter()
            .filter_map(|c| {
                if let AssistantContent::Text(t) = c {
                    Some(t.text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

fn serialise_messages(msgs: &[&Message]) -> String {
    msgs.iter()
        .map(|m| {
            let text = message_text(m);
            if text.is_empty() {
                String::new()
            } else {
                match m {
                    Message::User { .. } => format!("User: {text}"),
                    Message::Assistant { .. } => format!("Assistant: {text}"),
                    _ => String::new(),
                }
            }
        })
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FixedEstimator(u64);
    impl TokenEstimator for FixedEstimator {
        fn estimate(&self, _: &str) -> u64 {
            self.0
        }
    }

    struct EchoSummarizer;
    #[async_trait]
    impl SummarizerAgent for EchoSummarizer {
        async fn summarise(&self, history_text: &str) -> Result<String, OrchestrationError> {
            Ok(format!("SUMMARY: {}", &history_text[..history_text.len().min(40)]))
        }
    }

    fn make_manager(token_per_msg: u64, threshold: u64) -> MemoryManager {
        let config = CompactionConfig {
            context_window_tokens: threshold * 2,
            compaction_threshold: 0.5,
            min_messages_before_compaction: 2,
        };
        MemoryManager::with_components(
            config,
            Box::new(FixedEstimator(token_per_msg)),
            Box::new(EchoSummarizer),
        )
        .with_tail_keep(2)
    }

    #[test]
    fn should_compact_below_threshold() {
        let mgr = make_manager(10, 1000);
        assert!(!mgr.should_compact());
    }

    #[test]
    fn should_compact_above_threshold() {
        let mut mgr = make_manager(100, 400);
        // Need >= min_messages_before_compaction AND above threshold
        for i in 0..6 {
            mgr.push(Message::user(format!("msg {i}")));
        }
        // 6 messages × 100 tokens = 600 > threshold (400 * 0.5 = 200)
        assert!(mgr.should_compact());
    }

    #[tokio::test]
    async fn compact_reduces_message_count() {
        let mut mgr = make_manager(100, 400);
        for i in 0..8 {
            mgr.push(Message::user(format!("message number {i}")));
        }
        let before = mgr.history.len();
        let summary = mgr.compact().await.unwrap();
        let after = mgr.history.len();
        // Summary + 2 tail + 0 or more = at most tail_keep + 1
        assert!(after < before);
        assert!(summary.tokens_compacted > 0);
        assert_eq!(mgr.compaction_count(), 1);
    }

    #[tokio::test]
    async fn maybe_compact_returns_none_when_not_needed() {
        let mut mgr = make_manager(1, 10000);
        mgr.push(Message::user("hi"));
        let result = mgr.maybe_compact().await.unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn history_returns_pushed_messages() {
        let config = CompactionConfig::default();
        let mut mgr = MemoryManager::with_components(
            config,
            Box::new(CharCountEstimator),
            Box::new(EchoSummarizer),
        );
        mgr.push(Message::user("hello"));
        mgr.push(Message::assistant("world"));
        let h = mgr.history();
        assert_eq!(h.len(), 2);
    }
}
