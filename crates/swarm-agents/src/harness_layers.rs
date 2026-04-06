//! Harness layer taxonomy — Filter/Verifier/Policy.
//!
//! Decomposes RuntimeAdapter checks into composable layers per the
//! gyc567/AutoHarness taxonomy (arxiv:2603.03329).
//!
//! - **FilterLayer**: constrain the action space (which tools are available)
//! - **VerifierLayer**: validate tool call outcomes (detect anti-patterns)
//! - **PolicyLayer**: manage execution flow (deadlines, budgets, escalation)
//!
//! Each layer implements [`HarnessLayer`] and returns a [`LayerAction`] to
//! continue, skip, or terminate the agent session. Layers are composed in
//! a pipeline: filters run first, then verifiers, then policies.
//!
//! This module defines the architecture only. The existing [`super::runtime_adapter::RuntimeAdapter`]
//! is not yet migrated to dispatch through these layers — that is a follow-up task.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

/// Action a harness layer can take on a tool call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayerAction {
    /// Allow the tool call to proceed.
    Continue,
    /// Skip the tool call with a message to the LLM.
    Skip(String),
    /// Terminate the agent session with a reason.
    Terminate(String),
}

// Re-export the shared GovernanceTier from config to avoid duplicate enums.
// Mapping: config::Core ↔ "Permissive", config::Standard ↔ "Standard",
// config::Enhanced ↔ "Strict". Using the canonical enum from config.rs.
pub use crate::config::GovernanceTier;

/// Shared state across all harness layers.
///
/// Mirrors the tracking fields in `RuntimeAdapter`'s `AdapterState`, but
/// scoped to the subset relevant to layer decisions.
#[derive(Debug)]
pub struct HarnessState {
    /// Current LLM turn number (incremented on each `on_completion_call`).
    pub turn_count: usize,
    /// Whether edit_file or write_file succeeded at least once.
    pub has_written: bool,
    /// Read-only tool calls since last action tool. Reset by action tools.
    pub consecutive_reads: usize,
    /// Total read-only calls before the first successful write.
    pub total_reads_before_write: usize,
    /// Total tool calls completed so far.
    pub tool_call_count: usize,
    /// Consecutive edit failures per file path.
    pub edit_failure_counts: HashMap<String, u32>,
    /// Consecutive identical tool call tracking: hash -> count.
    pub repeat_call_counts: HashMap<u64, u32>,
    /// The most recent (tool_name, args) hash for repeat detection.
    pub last_call_key: Option<u64>,
    /// Count of successful write operations.
    pub successful_writes: u32,
}

impl HarnessState {
    pub fn new() -> Self {
        Self {
            turn_count: 0,
            has_written: false,
            consecutive_reads: 0,
            total_reads_before_write: 0,
            tool_call_count: 0,
            edit_failure_counts: HashMap::new(),
            repeat_call_counts: HashMap::new(),
            last_call_key: None,
            successful_writes: 0,
        }
    }
}

impl Default for HarnessState {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared configuration across all harness layers.
#[derive(Debug, Clone, Default)]
pub struct HarnessConfig {
    /// Max LLM turns without a write before termination.
    pub max_turns_without_write: Option<usize>,
    /// Turn after which search tools are unlocked (0 = always available).
    pub search_unlock_turn: Option<usize>,
    /// Maximum tool calls before termination.
    pub max_tool_calls: Option<usize>,
    /// Hard deadline for the entire agent session.
    pub deadline: Option<Instant>,
    /// Max read-only calls since last action before termination.
    pub max_reads_without_action: Option<usize>,
    /// Governance tier controlling enforcement strictness.
    pub governance_tier: GovernanceTier,
}

/// A composable harness layer that inspects tool calls.
///
/// Layers are evaluated in order (Filter -> Verifier -> Policy). The first
/// non-Continue action short-circuits the pipeline.
pub trait HarnessLayer: Send + Sync {
    /// Layer name for logging and tracing.
    fn name(&self) -> &str;

    /// Inspect a tool call before execution.
    /// Returns Continue to allow, Skip to block with message, Terminate to end session.
    fn on_tool_call(
        &self,
        tool_name: &str,
        args: &str,
        state: &HarnessState,
        config: &HarnessConfig,
    ) -> LayerAction;

    /// Inspect a tool result after execution (optional).
    /// Layers can update state and return Terminate to stop the agent
    /// (e.g., after too many consecutive edit failures).
    fn on_tool_result(
        &self,
        _tool_name: &str,
        _success: bool,
        _state: &mut HarnessState,
        _config: &HarnessConfig,
    ) -> LayerAction {
        LayerAction::Continue
    }
}

// ---------------------------------------------------------------------------
// Filter layers — constrain the action space
// ---------------------------------------------------------------------------

/// Phase-gated tool access: block search tools in early turns to enforce edit-first behavior.
///
/// Research: ALARA (arxiv:2603.20380) — restricting tools produces "guaranteed behavioral change."
/// When `search_unlock_turn` is set and the agent hasn't written yet, search/exploration
/// tools are blocked until that turn, forcing the agent to use inlined file content.
pub struct PhaseGateFilter;

impl HarnessLayer for PhaseGateFilter {
    fn name(&self) -> &str {
        "phase_gate_filter"
    }

    fn on_tool_call(
        &self,
        tool_name: &str,
        _args: &str,
        state: &HarnessState,
        config: &HarnessConfig,
    ) -> LayerAction {
        let Some(unlock_turn) = config.search_unlock_turn else {
            return LayerAction::Continue;
        };

        // Already wrote or past the unlock turn — allow everything.
        if state.has_written || state.turn_count > unlock_turn {
            return LayerAction::Continue;
        }

        let base = tool_name.strip_prefix("proxy_").unwrap_or(tool_name);
        let is_search = matches!(
            base,
            "search_code" | "colgrep" | "ast_grep" | "list_files" | "file_exists"
        );

        if is_search {
            LayerAction::Skip(format!(
                "Tool '{tool_name}' is not available until turn {unlock_turn} or after your first edit. \
                 The target file content is already in your task prompt. Use edit_file or write_file now."
            ))
        } else {
            LayerAction::Continue
        }
    }
}

/// Anti-loop filter: detect consecutive identical tool calls and skip after threshold.
///
/// Agents sometimes enter stuck loops calling the same tool with identical arguments.
/// This filter hashes (tool_name, args) and skips after `MAX_REPEAT_CALLS` consecutive
/// identical invocations.
pub struct AntiLoopFilter {
    max_repeat_calls: u32,
}

impl AntiLoopFilter {
    pub fn new(max_repeat_calls: u32) -> Self {
        Self { max_repeat_calls }
    }
}

impl Default for AntiLoopFilter {
    fn default() -> Self {
        Self {
            max_repeat_calls: 3,
        }
    }
}

impl HarnessLayer for AntiLoopFilter {
    fn name(&self) -> &str {
        "anti_loop_filter"
    }

    fn on_tool_call(
        &self,
        tool_name: &str,
        args: &str,
        state: &HarnessState,
        _config: &HarnessConfig,
    ) -> LayerAction {
        let call_key = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            tool_name.hash(&mut hasher);
            args.hash(&mut hasher);
            hasher.finish()
        };

        if state.last_call_key == Some(call_key) {
            if let Some(&count) = state.repeat_call_counts.get(&call_key) {
                if count >= self.max_repeat_calls {
                    return LayerAction::Skip(format!(
                        "You have called {tool_name} with identical arguments {} times. \
                         The results will not change. Stop repeating this call and move on: \
                         either run the verifier, delegate to a different worker, or report \
                         completion.",
                        count + 1
                    ));
                }
            }
        }

        LayerAction::Continue
    }
}

// ---------------------------------------------------------------------------
// Verifier layers — validate tool call outcomes
// ---------------------------------------------------------------------------

/// Tracks consecutive edit_file failures on the same file path.
///
/// Research: Graphectory (arxiv:2512.02393) — StrNotFound is the strongest
/// predictor of task failure. One observed case: 183 consecutive failed str_replace calls.
///
/// After `max_failures` consecutive failures on the same file, terminates the session
/// to prevent wasted tokens.
pub struct EditFailureVerifier {
    max_failures: u32,
}

impl EditFailureVerifier {
    pub fn new(max_failures: u32) -> Self {
        Self { max_failures }
    }
}

impl Default for EditFailureVerifier {
    fn default() -> Self {
        Self { max_failures: 3 }
    }
}

impl HarnessLayer for EditFailureVerifier {
    fn name(&self) -> &str {
        "edit_failure_verifier"
    }

    fn on_tool_call(
        &self,
        _tool_name: &str,
        _args: &str,
        _state: &HarnessState,
        _config: &HarnessConfig,
    ) -> LayerAction {
        // Edit failure verification happens in on_tool_result, not on_tool_call.
        LayerAction::Continue
    }

    fn on_tool_result(
        &self,
        tool_name: &str,
        success: bool,
        state: &mut HarnessState,
        _config: &HarnessConfig,
    ) -> LayerAction {
        let base = tool_name.strip_prefix("proxy_").unwrap_or(tool_name);
        if base != "edit_file" {
            return LayerAction::Continue;
        }

        let key = "edit_file_global".to_string();
        if success {
            state.edit_failure_counts.remove(&key);
            state.has_written = true;
            state.successful_writes += 1;
            LayerAction::Continue
        } else {
            let count = state.edit_failure_counts.entry(key.clone()).or_insert(0);
            *count += 1;
            if *count >= self.max_failures {
                LayerAction::Terminate(format!(
                    "Edit anti-pattern: {} consecutive edit_file failures. \
                     Use read_file to verify exact content before retrying.",
                    count
                ))
            } else {
                LayerAction::Continue
            }
        }
    }
}

impl EditFailureVerifier {
    /// Check if any file has exceeded the failure threshold.
    /// Returns a Terminate action if so, Continue otherwise.
    ///
    /// Call this after `on_tool_result` to enforce the failure limit.
    pub fn check_edit_failures(&self, state: &HarnessState) -> LayerAction {
        for (path, &count) in &state.edit_failure_counts {
            if count >= self.max_failures {
                return LayerAction::Terminate(format!(
                    "Edit anti-pattern: {count} consecutive edit_file failures on '{path}'. \
                     The old_content does not match the file. Use read_file to verify \
                     exact current content before retrying."
                ));
            }
        }
        LayerAction::Continue
    }
}

// ---------------------------------------------------------------------------
// Policy layers — manage execution flow
// ---------------------------------------------------------------------------

/// Write deadline + read budget + post-write stall detection.
///
/// Enforces multiple related policies:
/// 1. **Write deadline**: terminates after N turns without a successful write.
/// 2. **Pre-write read budget**: terminates after too many reads without any write.
/// 3. **Post-write read stall**: terminates when agent reads repeatedly after writing
///    (it's likely done but doesn't know it).
/// 4. **Read-without-action budget**: terminates after too many consecutive reads.
///
/// Research: "More with Less" (arxiv:2510.16786) — progressive deadline reminders
/// improve edit rates by inducing focused behavior.
pub struct WriteDeadlinePolicy {
    /// Hard limit on pre-write reads before termination.
    pre_write_read_budget: usize,
    /// Warning threshold for pre-write reads.
    pre_write_read_warn: usize,
    /// Consecutive reads after writing before termination.
    post_write_stall_threshold: usize,
}

impl WriteDeadlinePolicy {
    pub fn new(
        pre_write_read_budget: usize,
        pre_write_read_warn: usize,
        post_write_stall_threshold: usize,
    ) -> Self {
        Self {
            pre_write_read_budget,
            pre_write_read_warn,
            post_write_stall_threshold,
        }
    }
}

impl Default for WriteDeadlinePolicy {
    fn default() -> Self {
        Self {
            pre_write_read_budget: 8,
            pre_write_read_warn: 5,
            post_write_stall_threshold: 3,
        }
    }
}

impl HarnessLayer for WriteDeadlinePolicy {
    fn name(&self) -> &str {
        "write_deadline_policy"
    }

    fn on_tool_call(
        &self,
        tool_name: &str,
        _args: &str,
        state: &HarnessState,
        config: &HarnessConfig,
    ) -> LayerAction {
        let base = tool_name.strip_prefix("proxy_").unwrap_or(tool_name);
        let is_read = matches!(
            base,
            "read_file"
                | "list_files"
                | "get_diff"
                | "list_changed_files"
                | "query_notebook"
                | "team_status"
                | "check_mail"
                | "check_locks"
                | "chat_check"
                | "search_code"
                | "colgrep"
                | "ast_grep"
                | "file_exists"
                | "run_command"
        );

        if !is_read {
            return LayerAction::Continue;
        }

        // Check write-deadline turn limit (if configured).
        if let Some(max_turns) = config.max_turns_without_write {
            if state.turn_count > max_turns && !state.has_written {
                return LayerAction::Terminate(format!(
                    "write deadline exceeded: {} turns with no edit_file/write_file (limit: {}). \
                     Write code now.",
                    state.turn_count, max_turns
                ));
            }
        }

        // Pre-write read budget: terminate after too many reads without writing.
        if !state.has_written
            && config.max_turns_without_write.is_some()
            && state.total_reads_before_write >= self.pre_write_read_budget
        {
            return LayerAction::Terminate(format!(
                "Pre-write read budget exhausted: {} read-only calls with no \
                 edit_file/write_file. Stop exploring and write your edit now.",
                state.total_reads_before_write
            ));
        }

        // Post-write read stall: if the agent wrote files and is now just reading,
        // it's likely done. Terminate to hand control to the verifier.
        if state.has_written && state.consecutive_reads >= self.post_write_stall_threshold {
            return LayerAction::Terminate(format!(
                "Post-write read stall: {} consecutive read-only calls after \
                 successful write. Task appears complete — handing off to verifier.",
                state.consecutive_reads
            ));
        }

        // Read-without-action budget (configurable).
        if let Some(max_reads) = config.max_reads_without_action {
            if state.consecutive_reads > max_reads {
                return LayerAction::Terminate(format!(
                    "read budget exceeded: {} read-only calls since last action (limit: {}). \
                     Delegate to a worker or write code now.",
                    state.consecutive_reads, max_reads
                ));
            }
        }

        LayerAction::Continue
    }
}

impl WriteDeadlinePolicy {
    /// Generate a progressive write-deadline reminder message, if applicable.
    ///
    /// Returns `Some(reminder)` when the agent is approaching the write deadline
    /// and hasn't written yet. The caller should surface this in the next prompt.
    pub fn write_reminder(&self, state: &HarnessState, config: &HarnessConfig) -> Option<String> {
        let max_turns = config.max_turns_without_write?;
        if state.has_written {
            return None;
        }
        let remaining = max_turns.saturating_sub(state.turn_count);
        if remaining <= 2 && remaining > 0 {
            Some(format!(
                "\u{26a0} WRITE DEADLINE IN {} TURN{}. \
                 Your next call MUST be edit_file or write_file.",
                remaining,
                if remaining == 1 { "" } else { "S" }
            ))
        } else if state.turn_count > max_turns / 2 {
            Some(format!(
                "Turn {}/{}: {} turns remaining before write deadline.",
                state.turn_count, max_turns, remaining
            ))
        } else {
            None
        }
    }

    /// Check if the pre-write read count has reached the warning threshold.
    pub fn is_approaching_read_budget(&self, state: &HarnessState) -> bool {
        !state.has_written && state.total_reads_before_write >= self.pre_write_read_warn
    }
}

/// Deadline and tool-budget policy.
///
/// Enforces hard wall-clock deadline and max tool call count.
pub struct BudgetPolicy;

impl HarnessLayer for BudgetPolicy {
    fn name(&self) -> &str {
        "budget_policy"
    }

    fn on_tool_call(
        &self,
        _tool_name: &str,
        _args: &str,
        state: &HarnessState,
        config: &HarnessConfig,
    ) -> LayerAction {
        // Check wall-clock deadline.
        if let Some(deadline) = config.deadline {
            if Instant::now() >= deadline {
                return LayerAction::Terminate("deadline exceeded".to_string());
            }
        }

        // Check tool call budget.
        if let Some(max) = config.max_tool_calls {
            if state.tool_call_count >= max {
                return LayerAction::Terminate(format!("max tool calls ({max}) exceeded"));
            }
        }

        LayerAction::Continue
    }
}

// ---------------------------------------------------------------------------
// Pipeline execution
// ---------------------------------------------------------------------------

/// Execute a pipeline of harness layers on a tool call.
///
/// Layers are evaluated in order. The first non-Continue action short-circuits.
pub fn run_layers(
    layers: &[Box<dyn HarnessLayer>],
    tool_name: &str,
    args: &str,
    state: &HarnessState,
    config: &HarnessConfig,
) -> LayerAction {
    for layer in layers {
        let action = layer.on_tool_call(tool_name, args, state, config);
        if action != LayerAction::Continue {
            return action;
        }
    }
    LayerAction::Continue
}

/// Execute post-tool-result hooks on all layers.
/// Returns `Terminate` if any layer signals termination, `Continue` otherwise.
pub fn run_result_layers(
    layers: &[Box<dyn HarnessLayer>],
    tool_name: &str,
    success: bool,
    state: &mut HarnessState,
    config: &HarnessConfig,
) -> LayerAction {
    for layer in layers {
        let action = layer.on_tool_result(tool_name, success, state, config);
        if matches!(action, LayerAction::Terminate(_)) {
            return action;
        }
    }
    LayerAction::Continue
}

/// Build the default layer pipeline: filters first, then verifiers, then policies.
pub fn default_pipeline() -> Vec<Box<dyn HarnessLayer>> {
    vec![
        // Filters (constrain action space)
        Box::new(PhaseGateFilter),
        Box::new(AntiLoopFilter::default()),
        // Verifiers (validate outcomes)
        Box::new(EditFailureVerifier::default()),
        // Policies (manage execution flow)
        Box::new(WriteDeadlinePolicy::default()),
        Box::new(BudgetPolicy),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> HarnessConfig {
        HarnessConfig::default()
    }

    fn default_state() -> HarnessState {
        HarnessState::new()
    }

    // --- PhaseGateFilter tests ---

    #[test]
    fn phase_gate_allows_non_search_tools() {
        let filter = PhaseGateFilter;
        let config = HarnessConfig {
            search_unlock_turn: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 1;

        let action = filter.on_tool_call("edit_file", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn phase_gate_blocks_search_in_early_turns() {
        let filter = PhaseGateFilter;
        let config = HarnessConfig {
            search_unlock_turn: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 1;

        let action = filter.on_tool_call("colgrep", "{}", &state, &config);
        assert!(
            matches!(action, LayerAction::Skip(_)),
            "Expected Skip for search tool in early turn, got {action:?}"
        );
    }

    #[test]
    fn phase_gate_blocks_proxy_prefixed_search() {
        let filter = PhaseGateFilter;
        let config = HarnessConfig {
            search_unlock_turn: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 2;

        let action = filter.on_tool_call("proxy_search_code", "{}", &state, &config);
        assert!(matches!(action, LayerAction::Skip(_)));
    }

    #[test]
    fn phase_gate_allows_search_after_unlock_turn() {
        let filter = PhaseGateFilter;
        let config = HarnessConfig {
            search_unlock_turn: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 4;

        let action = filter.on_tool_call("colgrep", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn phase_gate_allows_search_after_write() {
        let filter = PhaseGateFilter;
        let config = HarnessConfig {
            search_unlock_turn: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 1;
        state.has_written = true;

        let action = filter.on_tool_call("colgrep", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn phase_gate_noop_when_unconfigured() {
        let filter = PhaseGateFilter;
        let config = default_config(); // search_unlock_turn = None
        let mut state = default_state();
        state.turn_count = 1;

        let action = filter.on_tool_call("colgrep", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    // --- AntiLoopFilter tests ---

    #[test]
    fn anti_loop_allows_first_call() {
        let filter = AntiLoopFilter::default();
        let state = default_state();
        let config = default_config();

        let action = filter.on_tool_call("search_code", r#"{"q":"foo"}"#, &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn anti_loop_skips_after_threshold() {
        let filter = AntiLoopFilter::new(2);
        let config = default_config();

        let args = r#"{"pattern":"DEFAULT_TIMEOUT"}"#;
        let call_key = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "search_code".hash(&mut hasher);
            args.hash(&mut hasher);
            hasher.finish()
        };

        let mut state = default_state();
        state.last_call_key = Some(call_key);
        state.repeat_call_counts.insert(call_key, 2);

        let action = filter.on_tool_call("search_code", args, &state, &config);
        assert!(
            matches!(action, LayerAction::Skip(_)),
            "Expected Skip after 2 repeats, got {action:?}"
        );
    }

    #[test]
    fn anti_loop_allows_different_call() {
        let filter = AntiLoopFilter::new(2);
        let config = default_config();

        let call_key = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "search_code".hash(&mut hasher);
            r#"{"q":"foo"}"#.hash(&mut hasher);
            hasher.finish()
        };

        let mut state = default_state();
        state.last_call_key = Some(call_key);
        state.repeat_call_counts.insert(call_key, 5);

        // Different tool/args — should not trigger
        let action = filter.on_tool_call("read_file", r#"{"path":"main.rs"}"#, &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    // --- EditFailureVerifier tests ---

    #[test]
    fn edit_verifier_tracks_failures() {
        let verifier = EditFailureVerifier::new(3);
        let config = default_config();
        let mut state = default_state();

        // Simulate 3 failures
        for _ in 0..3 {
            verifier.on_tool_result("edit_file", false, &mut state, &config);
        }

        let action = verifier.check_edit_failures(&state);
        assert!(
            matches!(action, LayerAction::Terminate(_)),
            "Expected Terminate after 3 failures, got {action:?}"
        );
    }

    #[test]
    fn edit_verifier_resets_on_success() {
        let verifier = EditFailureVerifier::new(3);
        let config = default_config();
        let mut state = default_state();

        // 2 failures, then success
        verifier.on_tool_result("edit_file", false, &mut state, &config);
        verifier.on_tool_result("edit_file", false, &mut state, &config);
        verifier.on_tool_result("edit_file", true, &mut state, &config);

        let action = verifier.check_edit_failures(&state);
        assert_eq!(action, LayerAction::Continue);
        assert!(state.has_written);
        assert_eq!(state.successful_writes, 1);
    }

    #[test]
    fn edit_verifier_ignores_non_edit_tools() {
        let verifier = EditFailureVerifier::new(3);
        let config = default_config();
        let mut state = default_state();

        // Failure on a different tool should not affect edit tracking.
        verifier.on_tool_result("read_file", false, &mut state, &config);
        verifier.on_tool_result("read_file", false, &mut state, &config);
        verifier.on_tool_result("read_file", false, &mut state, &config);

        let action = verifier.check_edit_failures(&state);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn edit_verifier_handles_proxy_prefix() {
        let verifier = EditFailureVerifier::new(2);
        let config = default_config();
        let mut state = default_state();

        verifier.on_tool_result("proxy_edit_file", false, &mut state, &config);
        verifier.on_tool_result("proxy_edit_file", false, &mut state, &config);

        let action = verifier.check_edit_failures(&state);
        assert!(matches!(action, LayerAction::Terminate(_)));
    }

    #[test]
    fn edit_verifier_on_tool_call_always_continues() {
        let verifier = EditFailureVerifier::default();
        let state = default_state();
        let config = default_config();

        let action = verifier.on_tool_call("edit_file", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    // --- WriteDeadlinePolicy tests ---

    #[test]
    fn write_deadline_allows_non_read_tools() {
        let policy = WriteDeadlinePolicy::default();
        let config = HarnessConfig {
            max_turns_without_write: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 10; // Way past deadline

        // Action tool should not be blocked by write deadline.
        let action = policy.on_tool_call("edit_file", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn write_deadline_terminates_after_turn_limit() {
        let policy = WriteDeadlinePolicy::default();
        let config = HarnessConfig {
            max_turns_without_write: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 4; // > limit of 3

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert!(
            matches!(action, LayerAction::Terminate(ref msg) if msg.contains("write deadline")),
            "Expected write deadline termination, got {action:?}"
        );
    }

    #[test]
    fn write_deadline_not_triggered_after_write() {
        let policy = WriteDeadlinePolicy::default();
        let config = HarnessConfig {
            max_turns_without_write: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 10;
        state.has_written = true;
        state.consecutive_reads = 0;

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn pre_write_read_budget_terminates() {
        let policy = WriteDeadlinePolicy::new(5, 3, 3);
        let config = HarnessConfig {
            max_turns_without_write: Some(20), // high so turn limit doesn't fire
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 2;
        state.total_reads_before_write = 5;

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert!(
            matches!(action, LayerAction::Terminate(ref msg) if msg.contains("Pre-write read budget")),
            "Expected pre-write read budget termination, got {action:?}"
        );
    }

    #[test]
    fn post_write_stall_terminates() {
        let policy = WriteDeadlinePolicy::default(); // stall threshold = 3
        let config = default_config();
        let mut state = default_state();
        state.has_written = true;
        state.consecutive_reads = 3;

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert!(
            matches!(action, LayerAction::Terminate(ref msg) if msg.contains("Post-write read stall")),
            "Expected post-write stall termination, got {action:?}"
        );
    }

    #[test]
    fn post_write_stall_allows_few_reads() {
        let policy = WriteDeadlinePolicy::default();
        let config = default_config();
        let mut state = default_state();
        state.has_written = true;
        state.consecutive_reads = 2;

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn read_without_action_budget_terminates() {
        let policy = WriteDeadlinePolicy::default();
        let config = HarnessConfig {
            max_reads_without_action: Some(5),
            ..default_config()
        };
        let mut state = default_state();
        state.consecutive_reads = 6;
        state.has_written = true; // Avoid post-write stall by setting to false below
        state.has_written = false; // No write, but reads exceed action budget
        state.total_reads_before_write = 0; // Avoid pre-write budget by leaving this low

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert!(
            matches!(action, LayerAction::Terminate(ref msg) if msg.contains("read budget exceeded")),
            "Expected read-without-action termination, got {action:?}"
        );
    }

    // --- WriteDeadlinePolicy reminder tests ---

    #[test]
    fn write_reminder_urgent_when_close() {
        let policy = WriteDeadlinePolicy::default();
        let config = HarnessConfig {
            max_turns_without_write: Some(5),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 4; // 1 turn remaining

        let reminder = policy.write_reminder(&state, &config);
        assert!(reminder.is_some());
        assert!(reminder.unwrap().contains("WRITE DEADLINE"));
    }

    #[test]
    fn write_reminder_progress_past_halfway() {
        let policy = WriteDeadlinePolicy::default();
        let config = HarnessConfig {
            max_turns_without_write: Some(10),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 6;

        let reminder = policy.write_reminder(&state, &config);
        assert!(reminder.is_some());
        assert!(reminder
            .unwrap()
            .contains("remaining before write deadline"));
    }

    #[test]
    fn write_reminder_none_when_written() {
        let policy = WriteDeadlinePolicy::default();
        let config = HarnessConfig {
            max_turns_without_write: Some(5),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 4;
        state.has_written = true;

        assert!(policy.write_reminder(&state, &config).is_none());
    }

    #[test]
    fn write_reminder_none_when_early() {
        let policy = WriteDeadlinePolicy::default();
        let config = HarnessConfig {
            max_turns_without_write: Some(10),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 2;

        assert!(policy.write_reminder(&state, &config).is_none());
    }

    #[test]
    fn approaching_read_budget() {
        let policy = WriteDeadlinePolicy::new(8, 5, 3);
        let mut state = default_state();
        state.total_reads_before_write = 5;

        assert!(policy.is_approaching_read_budget(&state));

        state.has_written = true;
        assert!(!policy.is_approaching_read_budget(&state));
    }

    // --- BudgetPolicy tests ---

    #[test]
    fn budget_policy_allows_within_budget() {
        let policy = BudgetPolicy;
        let config = HarnessConfig {
            max_tool_calls: Some(10),
            deadline: Some(Instant::now() + std::time::Duration::from_secs(3600)),
            ..default_config()
        };
        let mut state = default_state();
        state.tool_call_count = 5;

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn budget_policy_terminates_on_tool_budget() {
        let policy = BudgetPolicy;
        let config = HarnessConfig {
            max_tool_calls: Some(10),
            ..default_config()
        };
        let mut state = default_state();
        state.tool_call_count = 10;

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert!(
            matches!(action, LayerAction::Terminate(ref msg) if msg.contains("max tool calls")),
            "Expected tool budget termination, got {action:?}"
        );
    }

    #[test]
    fn budget_policy_terminates_on_deadline() {
        let policy = BudgetPolicy;
        let config = HarnessConfig {
            deadline: Some(Instant::now() - std::time::Duration::from_secs(1)),
            ..default_config()
        };
        let state = default_state();

        let action = policy.on_tool_call("read_file", "{}", &state, &config);
        assert!(
            matches!(action, LayerAction::Terminate(ref msg) if msg.contains("deadline")),
            "Expected deadline termination, got {action:?}"
        );
    }

    // --- Pipeline tests ---

    #[test]
    fn pipeline_short_circuits_on_skip() {
        let layers: Vec<Box<dyn HarnessLayer>> = vec![
            Box::new(PhaseGateFilter),
            Box::new(BudgetPolicy), // should not run
        ];

        let config = HarnessConfig {
            search_unlock_turn: Some(3),
            ..default_config()
        };
        let mut state = default_state();
        state.turn_count = 1;

        let action = run_layers(&layers, "colgrep", "{}", &state, &config);
        assert!(matches!(action, LayerAction::Skip(_)));
    }

    #[test]
    fn pipeline_runs_all_when_continue() {
        let layers = default_pipeline();
        let config = default_config();
        let state = default_state();

        let action = run_layers(&layers, "edit_file", "{}", &state, &config);
        assert_eq!(action, LayerAction::Continue);
    }

    #[test]
    fn pipeline_result_layers_update_state() {
        let layers: Vec<Box<dyn HarnessLayer>> = vec![Box::new(EditFailureVerifier::new(3))];
        let config = default_config();
        let mut state = default_state();

        run_result_layers(&layers, "edit_file", true, &mut state, &config);
        assert!(state.has_written);
        assert_eq!(state.successful_writes, 1);
    }

    // --- GovernanceTier tests ---

    #[test]
    fn governance_tier_default_is_standard() {
        assert_eq!(GovernanceTier::default(), GovernanceTier::Standard);
    }

    // --- HarnessState tests ---

    #[test]
    fn harness_state_default() {
        let state = HarnessState::default();
        assert_eq!(state.turn_count, 0);
        assert!(!state.has_written);
        assert_eq!(state.consecutive_reads, 0);
        assert_eq!(state.total_reads_before_write, 0);
        assert_eq!(state.tool_call_count, 0);
        assert!(state.edit_failure_counts.is_empty());
        assert!(state.repeat_call_counts.is_empty());
        assert!(state.last_call_key.is_none());
        assert_eq!(state.successful_writes, 0);
    }

    // --- Layer trait object safety ---

    #[test]
    fn layers_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<PhaseGateFilter>();
        assert_send_sync::<AntiLoopFilter>();
        assert_send_sync::<EditFailureVerifier>();
        assert_send_sync::<WriteDeadlinePolicy>();
        assert_send_sync::<BudgetPolicy>();
    }

    #[test]
    fn default_pipeline_has_all_layers() {
        let pipeline = default_pipeline();
        let names: Vec<&str> = pipeline.iter().map(|l| l.name()).collect();
        assert!(names.contains(&"phase_gate_filter"));
        assert!(names.contains(&"anti_loop_filter"));
        assert!(names.contains(&"edit_failure_verifier"));
        assert!(names.contains(&"write_deadline_policy"));
        assert!(names.contains(&"budget_policy"));
        assert_eq!(names.len(), 5);
    }
}
