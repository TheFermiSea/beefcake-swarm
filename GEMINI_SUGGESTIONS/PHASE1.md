Phase 1: Deep-Dive Implementation of Context CompactionTarget Files:coordination/src/state/compaction.rs (NEW)coordination/src/events/history.rs (MODIFIED)coordination/src/state/mod.rs (MODIFIED)1. The Core SwarmMemory Struct (coordination/src/state/compaction.rs)To comply with your rules/async-safety.yml and rules/error-handling.yml, we cannot simply use .unwrap() or block the async runtime. We must use tokio::sync::RwLock if sharing across threads, and we must implement a custom Error type.```rustuse rig::agent::Agent;use rig::message::{Message, MessageContent};use rig::providers::anthropic::CompletionModel; // Or openai, depending on your primary configuse thiserror::Error;use tracing::{debug, error, info, instrument, warn};#[derive(Error, Debug)]pub enum CompactionError {#[error("Failed to generate summary: {0}")]PromptError(#[from] rig::completion::CompletionError),#[error("History is empty, cannot compact")]EmptyHistory,}pub struct SwarmMemory {/// The actual sliding window of messagespub history: Vec<Message>,/// The hard limit before compaction is forcedmax_tokens: usize,/// The specialized agent strictly used for summarizing old contextsummarizer: Agent<CompletionModel>,}impl SwarmMemory {pub fn new(summarizer: Agent<CompletionModel>, max_tokens: usize) -> Self {Self {history: Vec::new(),max_tokens,summarizer,}}/// Appends a message and immediately checks if compaction is required.
#[instrument(skip(self, msg), fields(msg_type = ?msg))]
pub async fn push_and_compact(&mut self, msg: Message) -> Result<(), CompactionError> {
    self.history.push(msg);
    
    let current_estimate = self.estimate_tokens();
    let threshold = (self.max_tokens as f64 * 0.8) as usize;

    if current_estimate >= threshold {
        info!(
            current_tokens = current_estimate,
            threshold = threshold,
            "Context limit approaching. Triggering synchronous compaction."
        );
        self.execute_compaction().await?;
    }

    Ok(())
}

async fn execute_compaction(&mut self) -> Result<(), CompactionError> {
    if self.history.is_empty() {
        return Err(CompactionError::EmptyHistory);
    }

    // We want to summarize the oldest 70% of the conversation.
    let split_idx = (self.history.len() as f64 * 0.7).floor() as usize;
    if split_idx == 0 {
        warn!("History too short to split effectively despite token count.");
        return Ok(());
    }

    let (old_context, recent_context) = self.history.split_at(split_idx);
    
    // Serialize old context for the LLM
    let old_text: String = old_context.iter().map(|m| {
        match m {
            Message::User { content } => format!("USER: {:?}", content),
            Message::Assistant { content, .. } => format!("ASSISTANT: {:?}", content),
            Message::System { content } => format!("SYSTEM: {}", content),
        }
    }).collect::<Vec<_>>().join("\n");

    debug!("Prompting summarizer agent with {} bytes of old context", old_text.len());
    
    let summary = self.summarizer.prompt(&old_text).await.map_err(CompactionError::PromptError)?;

    // Reconstruct history: [Summary] + [Recent 30%]
    let mut new_history = Vec::with_capacity(recent_context.len() + 1);
    new_history.push(Message::System { 
        content: format!("PREVIOUS CONTEXT SUMMARY:\n{}", summary)
    });
    new_history.extend_from_slice(recent_context);

    let old_len = self.history.len();
    self.history = new_history;
    
    info!(
        old_msg_count = old_len,
        new_msg_count = self.history.len(),
        "Compaction completed successfully."
    );

    Ok(())
}

/// Heuristic token estimation (roughly 4 chars per token for standard English/Code)
/// Consider integrating `tiktoken-rs` here for extreme accuracy if using OpenAI.
fn estimate_tokens(&self) -> usize {
    self.history.iter().map(|m| {
        match m {
            Message::User { content } => content.len() * 4,
            Message::Assistant { content, .. } => content.len() * 4,
            Message::System { content } => content.len() * 4,
        }
    }).sum::<usize>() / 4
}
}```2. Wire into the Event Bus (coordination/src/events/history.rs)Your current history.rs likely manages Event traits. You need to map these events to Rig Message types and push them to SwarmMemory.```rust// Inside coordination/src/events/history.rsuse crate::state::compaction::SwarmMemory;use rig::message::Message;use std::sync::Arc;use tokio::sync::RwLock;pub struct GlobalHistory {memory: Arc<RwLock<SwarmMemory>>,}impl GlobalHistory {pub async fn record_event(&self, event_content: String, is_agent: bool) -> Result<(), String> {let message = if is_agent {Message::Assistant { content: vec![rig::message::MessageContent::Text(event_content)] }} else {Message::User { content: vec![rig::message::MessageContent::Text(event_content)] }};    let mut mem_lock = self.memory.write().await;
    
    // Complying with rules/no-unwrap-in-prod.yml
    mem_lock.push_and_compact(message).await.map_err(|e| {
        tracing::error!("Failed to record event and compact: {}", e);
        e.to_string()
    })?;
    
    Ok(())
}
}```
