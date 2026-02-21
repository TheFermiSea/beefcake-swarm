//! Token budgeting — threshold-based compaction triggers with pluggable estimator.
//!
//! Determines when compaction should occur based on token counts,
//! with configurable thresholds and a pluggable token estimation strategy.

use serde::{Deserialize, Serialize};

/// Trait for estimating token counts from text.
pub trait TokenEstimator {
    /// Estimate the number of tokens in the given text.
    fn estimate(&self, text: &str) -> u32;

    /// Estimator name for logging.
    fn name(&self) -> &str;
}

/// Simple word-count based estimator (words × factor).
///
/// Uses the approximation that ~1.3 tokens per word for English text.
#[derive(Debug, Clone)]
pub struct WordCountEstimator {
    /// Tokens per word multiplier.
    pub factor: f64,
}

impl Default for WordCountEstimator {
    fn default() -> Self {
        Self { factor: 1.3 }
    }
}

impl TokenEstimator for WordCountEstimator {
    fn estimate(&self, text: &str) -> u32 {
        let word_count = text.split_whitespace().count();
        (word_count as f64 * self.factor).ceil() as u32
    }

    fn name(&self) -> &str {
        "word_count"
    }
}

/// Character-count based estimator (chars / divisor).
///
/// Uses the approximation that ~4 characters per token for English text.
#[derive(Debug, Clone)]
pub struct CharCountEstimator {
    /// Characters per token.
    pub chars_per_token: f64,
}

impl Default for CharCountEstimator {
    fn default() -> Self {
        Self {
            chars_per_token: 4.0,
        }
    }
}

impl TokenEstimator for CharCountEstimator {
    fn estimate(&self, text: &str) -> u32 {
        (text.len() as f64 / self.chars_per_token).ceil() as u32
    }

    fn name(&self) -> &str {
        "char_count"
    }
}

/// Token budget configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    /// Maximum tokens before compaction is triggered.
    pub max_tokens: u64,
    /// Target tokens after compaction (must be < max_tokens).
    pub target_tokens: u64,
    /// Minimum entries to keep even if over budget (recent context).
    pub min_retained_entries: usize,
    /// Reserve tokens for system prompt (never compacted).
    pub system_reserve: u64,
}

impl TokenBudget {
    /// Available budget (max - system reserve).
    pub fn available(&self) -> u64 {
        self.max_tokens.saturating_sub(self.system_reserve)
    }

    /// Validate budget configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.target_tokens >= self.max_tokens {
            return Err(format!(
                "target_tokens ({}) must be less than max_tokens ({})",
                self.target_tokens, self.max_tokens
            ));
        }
        if self.system_reserve >= self.max_tokens {
            return Err(format!(
                "system_reserve ({}) must be less than max_tokens ({})",
                self.system_reserve, self.max_tokens
            ));
        }
        Ok(())
    }
}

impl Default for TokenBudget {
    fn default() -> Self {
        Self {
            max_tokens: 128_000,
            target_tokens: 64_000,
            min_retained_entries: 5,
            system_reserve: 4_000,
        }
    }
}

/// Decision from the compaction trigger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BudgetDecision {
    /// Within budget — no compaction needed.
    WithinBudget,
    /// Approaching budget — compaction recommended.
    CompactionRecommended {
        /// Current token usage.
        current_tokens: u64,
        /// Tokens to free.
        tokens_to_free: u64,
    },
    /// Over budget — compaction required.
    CompactionRequired {
        /// Current token usage.
        current_tokens: u64,
        /// How much over budget.
        overage: u64,
    },
}

impl BudgetDecision {
    /// Whether compaction should run.
    pub fn should_compact(&self) -> bool {
        !matches!(self, Self::WithinBudget)
    }

    /// Whether compaction is urgent (over max).
    pub fn is_urgent(&self) -> bool {
        matches!(self, Self::CompactionRequired { .. })
    }
}

impl std::fmt::Display for BudgetDecision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WithinBudget => write!(f, "within_budget"),
            Self::CompactionRecommended {
                current_tokens,
                tokens_to_free,
            } => write!(
                f,
                "compaction_recommended ({} tokens, free {})",
                current_tokens, tokens_to_free
            ),
            Self::CompactionRequired {
                current_tokens,
                overage,
            } => write!(
                f,
                "compaction_required ({} tokens, {} over)",
                current_tokens, overage
            ),
        }
    }
}

/// Evaluates whether compaction should trigger based on token budget.
pub struct CompactionTrigger {
    budget: TokenBudget,
}

impl CompactionTrigger {
    /// Create a new trigger with the given budget.
    pub fn new(budget: TokenBudget) -> Self {
        Self { budget }
    }

    /// Evaluate the current token usage against the budget.
    pub fn evaluate(&self, current_tokens: u64) -> BudgetDecision {
        let available = self.budget.available();

        if current_tokens > available {
            return BudgetDecision::CompactionRequired {
                current_tokens,
                overage: current_tokens - available,
            };
        }

        // Trigger at 80% of available budget
        let threshold = (available as f64 * 0.8) as u64;
        if current_tokens >= threshold {
            let tokens_to_free = current_tokens.saturating_sub(self.budget.target_tokens);
            return BudgetDecision::CompactionRecommended {
                current_tokens,
                tokens_to_free,
            };
        }

        BudgetDecision::WithinBudget
    }

    /// Calculate how many entries should be compacted to meet target.
    ///
    /// Returns the number of oldest entries to compact, respecting
    /// the minimum retained entries constraint.
    pub fn entries_to_compact(&self, entry_tokens: &[u64], current_total: u64) -> usize {
        if current_total <= self.budget.target_tokens {
            return 0;
        }

        let tokens_to_free = current_total - self.budget.target_tokens;
        let mut freed = 0u64;
        let mut count = 0usize;

        // Always keep at least min_retained_entries from the end
        let compactable = entry_tokens
            .len()
            .saturating_sub(self.budget.min_retained_entries);

        for &tokens in entry_tokens.iter().take(compactable) {
            if freed >= tokens_to_free {
                break;
            }
            freed += tokens;
            count += 1;
        }

        count
    }

    /// Get the budget configuration.
    pub fn budget(&self) -> &TokenBudget {
        &self.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_count_estimator() {
        let est = WordCountEstimator::default();
        assert_eq!(est.estimate("hello world"), 3); // 2 * 1.3 = 2.6 → 3
        assert_eq!(est.estimate(""), 0);
        assert_eq!(est.name(), "word_count");
    }

    #[test]
    fn test_char_count_estimator() {
        let est = CharCountEstimator::default();
        assert_eq!(est.estimate("hello world"), 3); // 11 / 4 = 2.75 → 3
        assert_eq!(est.estimate(""), 0);
        assert_eq!(est.name(), "char_count");
    }

    #[test]
    fn test_budget_defaults() {
        let budget = TokenBudget::default();
        assert_eq!(budget.max_tokens, 128_000);
        assert_eq!(budget.target_tokens, 64_000);
        assert_eq!(budget.available(), 124_000);
    }

    #[test]
    fn test_budget_validate() {
        let mut budget = TokenBudget::default();
        assert!(budget.validate().is_ok());

        budget.target_tokens = 200_000;
        assert!(budget.validate().is_err());

        budget.target_tokens = 64_000;
        budget.system_reserve = 200_000;
        assert!(budget.validate().is_err());
    }

    #[test]
    fn test_trigger_within_budget() {
        let trigger = CompactionTrigger::new(TokenBudget::default());
        let decision = trigger.evaluate(50_000);
        assert_eq!(decision, BudgetDecision::WithinBudget);
        assert!(!decision.should_compact());
        assert!(!decision.is_urgent());
    }

    #[test]
    fn test_trigger_recommended() {
        let trigger = CompactionTrigger::new(TokenBudget {
            max_tokens: 100_000,
            target_tokens: 50_000,
            min_retained_entries: 3,
            system_reserve: 0,
        });
        // 80% of 100k = 80k
        let decision = trigger.evaluate(85_000);
        assert!(matches!(
            decision,
            BudgetDecision::CompactionRecommended { .. }
        ));
        assert!(decision.should_compact());
        assert!(!decision.is_urgent());
    }

    #[test]
    fn test_trigger_required() {
        let trigger = CompactionTrigger::new(TokenBudget {
            max_tokens: 100_000,
            target_tokens: 50_000,
            min_retained_entries: 3,
            system_reserve: 10_000,
        });
        // Available = 90k, over that triggers required
        let decision = trigger.evaluate(95_000);
        assert!(matches!(
            decision,
            BudgetDecision::CompactionRequired { .. }
        ));
        assert!(decision.should_compact());
        assert!(decision.is_urgent());
    }

    #[test]
    fn test_entries_to_compact() {
        let trigger = CompactionTrigger::new(TokenBudget {
            max_tokens: 100,
            target_tokens: 50,
            min_retained_entries: 2,
            system_reserve: 0,
        });

        // 5 entries with 20 tokens each = 100 total, target 50
        // Need to free 50 tokens, but keep last 2
        let entry_tokens = vec![20, 20, 20, 20, 20];
        let count = trigger.entries_to_compact(&entry_tokens, 100);
        // Can compact first 3 (compactable = 5-2 = 3), that frees 60 ≥ 50
        assert_eq!(count, 3);
    }

    #[test]
    fn test_entries_to_compact_within_budget() {
        let trigger = CompactionTrigger::new(TokenBudget {
            max_tokens: 100,
            target_tokens: 50,
            min_retained_entries: 2,
            system_reserve: 0,
        });

        let entry_tokens = vec![10, 10, 10];
        let count = trigger.entries_to_compact(&entry_tokens, 30); // under target
        assert_eq!(count, 0);
    }

    #[test]
    fn test_entries_to_compact_respects_min_retained() {
        let trigger = CompactionTrigger::new(TokenBudget {
            max_tokens: 100,
            target_tokens: 10,
            min_retained_entries: 4,
            system_reserve: 0,
        });

        // 5 entries, must keep 4 → can only compact 1
        let entry_tokens = vec![20, 20, 20, 20, 20];
        let count = trigger.entries_to_compact(&entry_tokens, 100);
        assert_eq!(count, 1); // Only first entry is compactable
    }

    #[test]
    fn test_budget_decision_display() {
        assert_eq!(BudgetDecision::WithinBudget.to_string(), "within_budget");
        assert!(BudgetDecision::CompactionRecommended {
            current_tokens: 80000,
            tokens_to_free: 30000
        }
        .to_string()
        .contains("80000"));
        assert!(BudgetDecision::CompactionRequired {
            current_tokens: 130000,
            overage: 6000
        }
        .to_string()
        .contains("6000 over"));
    }

    #[test]
    fn test_budget_serde() {
        let budget = TokenBudget::default();
        let json = serde_json::to_string(&budget).unwrap();
        let parsed: TokenBudget = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.max_tokens, 128_000);
        assert_eq!(parsed.target_tokens, 64_000);
    }

    #[test]
    fn test_decision_serde() {
        let decision = BudgetDecision::CompactionRequired {
            current_tokens: 150_000,
            overage: 26_000,
        };
        let json = serde_json::to_string(&decision).unwrap();
        let parsed: BudgetDecision = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, decision);
    }

    #[test]
    fn test_custom_estimator_factor() {
        let est = WordCountEstimator { factor: 1.0 };
        assert_eq!(est.estimate("one two three"), 3);

        let est2 = WordCountEstimator { factor: 2.0 };
        assert_eq!(est2.estimate("one two three"), 6);
    }

    #[test]
    fn test_trigger_boundary_exact_threshold() {
        let trigger = CompactionTrigger::new(TokenBudget {
            max_tokens: 100,
            target_tokens: 50,
            min_retained_entries: 1,
            system_reserve: 0,
        });
        // Exactly at 80% threshold
        let decision = trigger.evaluate(80);
        assert!(decision.should_compact());
    }
}
