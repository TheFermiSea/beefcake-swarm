use std::collections::VecDeque;
use rig::completion::{Message, AssistantContent};
use rig::message::UserContent;

/// Shared chat history management with token estimation and compaction support.
pub struct ChatHistory {
    messages: VecDeque<Message>,
    max_tokens: usize,
    estimated_tokens: usize,
}

impl ChatHistory {
    /// Create a new ChatHistory with a given token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            messages: VecDeque::new(),
            max_tokens,
            estimated_tokens: 0,
        }
    }

    /// Push a user message to history.
    pub fn push_user(&mut self, text: impl Into<String>) {
        let text_str = text.into();
        self.estimated_tokens += Self::estimate_tokens_for_text(&text_str);
        self.messages.push_back(Message::user(text_str));
    }

    /// Push an assistant message to history.
    pub fn push_assistant(&mut self, text: impl Into<String>) {
        let text_str = text.into();
        self.estimated_tokens += Self::estimate_tokens_for_text(&text_str);
        self.messages.push_back(Message::assistant(text_str));
    }

    /// Push a raw Message to history.
    pub fn push_message(&mut self, msg: Message) {
        let text = extract_message_text(&msg);
        self.estimated_tokens += Self::estimate_tokens_for_text(&text);
        self.messages.push_back(msg);
    }

    /// Rough token estimation: ~1 token per 4 chars.
    fn estimate_tokens_for_text(s: &str) -> usize {
        s.len() / 4
    }

    /// Get total estimated tokens in history.
    pub fn estimate_tokens(&self) -> usize {
        self.estimated_tokens
    }

    /// Get history as a Vec<Message> for passing to completion tools.
    pub fn to_vec(&self) -> Vec<Message> {
        self.messages.iter().cloned().collect()
    }

    /// Check if history exceeds the token budget.
    pub fn needs_compaction(&self) -> bool {
        self.estimated_tokens > self.max_tokens
    }

    /// Number of messages in history.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Clear history.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.estimated_tokens = 0;
    }

    /// Compact old messages by replacing early history with a summary.
    ///
    /// Takes a summarizer function to avoid coupling to specific agent types.
    /// The summarizer should take a conversation history as text and return a summary.
    pub async fn compact<F, Fut>(&mut self, summarizer: F) -> Result<(), anyhow::Error>
    where
        F: FnOnce(String) -> Fut,
        Fut: std::future::Future<Output = Result<String, anyhow::Error>>,
    {
        let tokens_before = self.estimated_tokens;

        let history_text: String = self
            .messages
            .iter()
            .map(|m| {
                let text = extract_message_text(m);
                match m {
                    Message::User { .. } => format!("User: {text}"),
                    Message::Assistant { .. } => format!("Assistant: {text}"),
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        // The summarizer is expected to handle the "Summarize this conversation" instruction internally
        // or we can wrap it here. Let's wrap it for consistency.
        let prompt = format!("Summarize this conversation:\n\n{history_text}");
        let summary_text = summarizer(prompt).await?;

        // Keep the most recent 2 messages (latest exchange) intact.
        let keep = 2.min(self.messages.len());
        let tail: Vec<Message> = self.messages.iter().rev().take(keep).cloned().collect();
        
        self.messages.clear();
        self.estimated_tokens = 0;

        // Insert summary as a system-style user message.
        let summary_msg_text = format!(
            "[CONTEXT SUMMARY — {} tokens compacted]\n{}",
            tokens_before, summary_text
        );
        self.push_user(summary_msg_text);

        // Re-add the tail (reversed back to chronological).
        for msg in tail.into_iter().rev() {
            self.push_message(msg);
        }

        Ok(())
    }
}

/// Extract plain text from a `Message` (user or assistant).
pub fn extract_message_text(msg: &Message) -> String {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_estimate() {
        let mut history = ChatHistory::new(100);
        history.push_user("Hello");
        assert_eq!(history.estimate_tokens(), 5 / 4);
        history.push_assistant("Hi there!");
        assert_eq!(history.estimate_tokens(), (5 / 4) + (9 / 4));
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_needs_compaction() {
        let mut history = ChatHistory::new(10);
        history.push_user("This is a long message that should trigger compaction.");
        assert!(history.needs_compaction());
    }

    #[tokio::test]
    async fn test_compaction() {
        let mut history = ChatHistory::new(100);
        history.push_user("Message 1");
        history.push_assistant("Response 1");
        history.push_user("Message 2");
        history.push_assistant("Response 2");

        let summarizer = |_| async { Ok("Summarized history".to_string()) };
        
        history.compact(summarizer).await.unwrap();

        let messages = history.to_vec();
        // Should have: Summary, Message 2, Response 2
        assert_eq!(messages.len(), 3);
        
        let first_text = extract_message_text(&messages[0]);
        assert!(first_text.contains("Summarized history"));
        assert!(first_text.contains("tokens compacted"));
        
        assert_eq!(extract_message_text(&messages[1]), "Message 2");
        assert_eq!(extract_message_text(&messages[2]), "Response 2");
    }
}
