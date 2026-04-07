//! Reformulation engine — classifies agent failures and produces recovery directives.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureClass {
    MissingContext,
    FormulationDefect,
    ContextDeficit,
    DecompositionRequired,
    ImplementationThrash,
    GenuineCodeDefect,
    InfraTransient,
}

impl std::fmt::Display for FailureClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingContext => write!(f, "missing_context"),
            Self::FormulationDefect => write!(f, "formulation_defect"),
            Self::ContextDeficit => write!(f, "context_deficit"),
            Self::DecompositionRequired => write!(f, "decomposition_required"),
            Self::ImplementationThrash => write!(f, "implementation_thrash"),
            Self::GenuineCodeDefect => write!(f, "genuine_code_defect"),
            Self::InfraTransient => write!(f, "infra_transient"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryAction {
    RewriteIssue,
    AppendDirective,
    CompressAndRetry,
    DecomposeIntoSubtasks,
    PinStrategy,
    EscalateToReasoningModel,
    RetryWithBackoff,
}

impl std::fmt::Display for RecoveryAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RewriteIssue => write!(f, "rewrite_issue"),
            Self::AppendDirective => write!(f, "append_directive"),
            Self::CompressAndRetry => write!(f, "compress_and_retry"),
            Self::DecomposeIntoSubtasks => write!(f, "decompose_into_subtasks"),
            Self::PinStrategy => write!(f, "pin_strategy"),
            Self::EscalateToReasoningModel => write!(f, "escalate_to_reasoning_model"),
            Self::RetryWithBackoff => write!(f, "retry_with_backoff"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReformulationSignal {
    pub message: String,
    pub iterations: u32,
    pub tokens_used: u64,
    pub context_exhausted: bool,
    pub infra_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReformulationResult {
    pub class: FailureClass,
    pub action: RecoveryAction,
    pub directive: String,
    pub confidence: f64,
}

pub struct ReformulationEngine;

impl ReformulationEngine {
    pub fn classify(signal: &ReformulationSignal) -> (FailureClass, f64) {
        if signal.infra_error {
            return (FailureClass::InfraTransient, 0.95);
        }
        if signal.context_exhausted || signal.tokens_used > 180_000 {
            return (FailureClass::ContextDeficit, 0.90);
        }
        let msg = signal.message.to_ascii_lowercase();
        if msg.contains("ambiguous") || msg.contains("unclear") || msg.contains("contradictory") {
            return (FailureClass::FormulationDefect, 0.85);
        }
        if msg.contains("too large") || msg.contains("too complex") || msg.contains("split") {
            return (FailureClass::DecompositionRequired, 0.80);
        }
        if signal.iterations >= 4
            && (msg.contains("reverted")
                || msg.contains("oscillat")
                || msg.contains("same approach"))
        {
            return (FailureClass::ImplementationThrash, 0.80);
        }
        if msg.contains("bug") || msg.contains("panic") || msg.contains("undefined behaviour") {
            return (FailureClass::GenuineCodeDefect, 0.75);
        }
        if msg.contains("missing") || msg.contains("not found") || msg.contains("unknown") {
            return (FailureClass::MissingContext, 0.70);
        }
        (FailureClass::FormulationDefect, 0.40)
    }

    pub fn recovery_action(class: FailureClass) -> RecoveryAction {
        match class {
            FailureClass::MissingContext => RecoveryAction::RewriteIssue,
            FailureClass::FormulationDefect => RecoveryAction::AppendDirective,
            FailureClass::ContextDeficit => RecoveryAction::CompressAndRetry,
            FailureClass::DecompositionRequired => RecoveryAction::DecomposeIntoSubtasks,
            FailureClass::ImplementationThrash => RecoveryAction::PinStrategy,
            FailureClass::GenuineCodeDefect => RecoveryAction::EscalateToReasoningModel,
            FailureClass::InfraTransient => RecoveryAction::RetryWithBackoff,
        }
    }

    pub fn directive(class: FailureClass) -> String {
        match class {
            FailureClass::MissingContext => "CONTEXT REQUIRED: The previous attempt lacked necessary background. Restate the full context, relevant file paths, and acceptance criteria before attempting the task.".to_string(),
            FailureClass::FormulationDefect => "CLARIFICATION NEEDED: The task description was ambiguous. Resolve all contradictions and restate the single concrete goal before proceeding.".to_string(),
            FailureClass::ContextDeficit => "CONTEXT COMPRESSED: Prior context was truncated. Work from the summarised state below; do not reference earlier turns.".to_string(),
            FailureClass::DecompositionRequired => "DECOMPOSE TASK: This task exceeds single-pass capacity. Break it into numbered sub-tasks, complete each independently, then integrate the results.".to_string(),
            FailureClass::ImplementationThrash => "STRATEGY LOCKED: Previous attempts oscillated between approaches. Commit to the first viable implementation and do not switch strategies mid-task.".to_string(),
            FailureClass::GenuineCodeDefect => "ROOT CAUSE REQUIRED: A genuine defect is blocking progress. Identify the exact bug, produce a minimal reproducer, then fix it before continuing with the original task.".to_string(),
            FailureClass::InfraTransient => "TRANSIENT ERROR: The previous failure was infrastructure-related. Retry the task unchanged after a short delay.".to_string(),
        }
    }

    pub fn reformulate(signal: &ReformulationSignal) -> ReformulationResult {
        let (class, confidence) = Self::classify(signal);
        let action = Self::recovery_action(class);
        let directive = Self::directive(class);
        ReformulationResult {
            class,
            action,
            directive,
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signal(message: &str) -> ReformulationSignal {
        ReformulationSignal {
            message: message.to_string(),
            iterations: 1,
            tokens_used: 10_000,
            context_exhausted: false,
            infra_error: false,
        }
    }

    #[test]
    fn infra_takes_priority() {
        let mut s = signal("timeout");
        s.infra_error = true;
        let (class, conf) = ReformulationEngine::classify(&s);
        assert_eq!(class, FailureClass::InfraTransient);
        assert!(conf > 0.9);
    }

    #[test]
    fn context_exhausted_flag() {
        let mut s = signal("ran out of space");
        s.context_exhausted = true;
        let (class, _) = ReformulationEngine::classify(&s);
        assert_eq!(class, FailureClass::ContextDeficit);
    }

    #[test]
    fn high_token_count_triggers_context_deficit() {
        let mut s = signal("some error");
        s.tokens_used = 200_000;
        let (class, _) = ReformulationEngine::classify(&s);
        assert_eq!(class, FailureClass::ContextDeficit);
    }

    #[test]
    fn reformulate_returns_consistent_action() {
        let s = signal("task is too large to complete in one pass");
        let result = ReformulationEngine::reformulate(&s);
        assert_eq!(result.class, FailureClass::DecompositionRequired);
        assert_eq!(result.action, RecoveryAction::DecomposeIntoSubtasks);
        assert!(!result.directive.is_empty());
    }
}
