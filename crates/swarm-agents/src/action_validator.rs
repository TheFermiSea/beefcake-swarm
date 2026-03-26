//! Pre-tool-call validators that catch common agent mistakes before they waste a turn.
//!
//! Each validator implements [`ActionValidator`] and runs in the
//! [`RuntimeAdapter::on_tool_call`] hook. Validators return `Ok(())` to pass
//! or `Err(message)` with an actionable hint that is returned to the LLM as
//! a tool error (the agent loop continues — the call is rejected, not terminated).

use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Shared state tracked across tool calls within a single agent session.
pub struct ValidatorState {
    /// Files that have been read during this session.
    pub files_read: HashSet<String>,
    /// Files that have been written (edit_file or write_file) during this session.
    pub files_written: HashSet<String>,
    /// Consecutive read count per path (resets on write to that path).
    pub read_counts: HashMap<String, usize>,
    /// Current LLM turn number.
    pub turn: usize,
}

impl ValidatorState {
    pub fn new() -> Self {
        Self {
            files_read: HashSet::new(),
            files_written: HashSet::new(),
            read_counts: HashMap::new(),
            turn: 0,
        }
    }

    /// Record that a file was read.
    pub fn record_read(&mut self, path: &str) {
        self.files_read.insert(path.to_string());
        *self.read_counts.entry(path.to_string()).or_insert(0) += 1;
    }

    /// Record that a file was written.
    pub fn record_write(&mut self, path: &str) {
        self.files_written.insert(path.to_string());
        // Reset consecutive read counter for this path.
        self.read_counts.remove(path);
    }
}

impl Default for ValidatorState {
    fn default() -> Self {
        Self::new()
    }
}

/// A pre-tool-call validator. Implementations inspect the tool name, args JSON,
/// and accumulated state to decide whether to allow or reject the call.
pub trait ActionValidator: Send + Sync {
    fn name(&self) -> &str;
    /// Return `Ok(())` to allow, `Err(message)` to reject with an actionable hint.
    fn validate(
        &self,
        tool_name: &str,
        args_json: &str,
        state: &ValidatorState,
    ) -> Result<(), String>;
}

// ---------------------------------------------------------------------------
// Concrete validators
// ---------------------------------------------------------------------------

/// Reject `edit_file` / `write_file` on a path the agent hasn't read yet.
pub struct ReadBeforeWriteValidator;

impl ActionValidator for ReadBeforeWriteValidator {
    fn name(&self) -> &str {
        "read_before_write"
    }

    fn validate(
        &self,
        tool_name: &str,
        args_json: &str,
        state: &ValidatorState,
    ) -> Result<(), String> {
        let base = tool_name.strip_prefix("proxy_").unwrap_or(tool_name);
        if base != "edit_file" && base != "write_file" {
            return Ok(());
        }
        // Extract path from args JSON (best-effort parse).
        let path = extract_path(args_json);
        if let Some(ref p) = path {
            if !state.files_read.contains(p.as_str()) {
                return Err(format!(
                    "You must read_file '{p}' before editing it. \
                     Call read_file first to see the current content."
                ));
            }
        }
        Ok(())
    }
}

/// Reject `edit_file` when `old_content` has fewer than 2 non-empty lines.
/// This catches extremely vague search strings that are likely to match wrong.
pub struct MinContextValidator;

impl ActionValidator for MinContextValidator {
    fn name(&self) -> &str {
        "min_context"
    }

    fn validate(
        &self,
        tool_name: &str,
        args_json: &str,
        _state: &ValidatorState,
    ) -> Result<(), String> {
        let base = tool_name.strip_prefix("proxy_").unwrap_or(tool_name);
        if base != "edit_file" {
            return Ok(());
        }
        // Skip validation when anchors are present — anchor mode doesn't need old_content
        let has_anchors = extract_field(args_json, "anchor_start").is_some()
            && extract_field(args_json, "anchor_end").is_some();
        if has_anchors {
            return Ok(());
        }
        let old = extract_field(args_json, "old_content");
        if let Some(ref content) = old {
            let non_empty_lines = content.lines().filter(|l| !l.trim().is_empty()).count();
            if non_empty_lines < 2 {
                return Err("old_content has fewer than 2 non-empty lines. \
                     Include 3-5 lines of surrounding context to ensure a unique match."
                    .to_string());
            }
        }
        Ok(())
    }
}

/// Detect truncated JSON in `new_content` (unmatched braces/brackets).
pub struct TruncatedJsonValidator;

impl ActionValidator for TruncatedJsonValidator {
    fn name(&self) -> &str {
        "truncated_json"
    }

    fn validate(
        &self,
        tool_name: &str,
        args_json: &str,
        _state: &ValidatorState,
    ) -> Result<(), String> {
        let base = tool_name.strip_prefix("proxy_").unwrap_or(tool_name);
        if base != "edit_file" && base != "write_file" {
            return Ok(());
        }
        let content =
            extract_field(args_json, "new_content").or_else(|| extract_field(args_json, "content"));
        if let Some(ref text) = content {
            if looks_truncated(text) {
                return Err("Content appears truncated (unmatched braces/brackets). \
                     Verify the content is complete before writing."
                    .to_string());
            }
        }
        Ok(())
    }
}

/// After 3 consecutive reads of the same file without writing, hint the agent.
pub struct ConsecutiveReadDetector;

impl ActionValidator for ConsecutiveReadDetector {
    fn name(&self) -> &str {
        "consecutive_read"
    }

    fn validate(
        &self,
        tool_name: &str,
        args_json: &str,
        state: &ValidatorState,
    ) -> Result<(), String> {
        let base = tool_name.strip_prefix("proxy_").unwrap_or(tool_name);
        if base != "read_file" {
            return Ok(());
        }
        let path = extract_path(args_json);
        if let Some(ref p) = path {
            if let Some(&count) = state.read_counts.get(p.as_str()) {
                if count >= 3 {
                    return Err(format!(
                        "You have read '{p}' {count} times without editing it. \
                         Either edit the file now or move on to a different file."
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Validate anchor hashes against current file content.
pub struct AnchorFreshnessValidator {
    working_dir: std::path::PathBuf,
}

impl AnchorFreshnessValidator {
    pub fn new(working_dir: &Path) -> Self {
        Self {
            working_dir: working_dir.to_path_buf(),
        }
    }
}

impl ActionValidator for AnchorFreshnessValidator {
    fn name(&self) -> &str {
        "anchor_freshness"
    }

    fn validate(
        &self,
        tool_name: &str,
        args_json: &str,
        _state: &ValidatorState,
    ) -> Result<(), String> {
        let base = tool_name.strip_prefix("proxy_").unwrap_or(tool_name);
        if base != "edit_file" {
            return Ok(());
        }
        let anchor_start = extract_field(args_json, "anchor_start");
        let anchor_end = extract_field(args_json, "anchor_end");
        if anchor_start.is_none() || anchor_end.is_none() {
            return Ok(()); // No anchors — skip validation
        }

        let path = match extract_path(args_json) {
            Some(p) => p,
            None => return Ok(()),
        };

        let full_path = self.working_dir.join(&path);
        let content = match std::fs::read_to_string(&full_path) {
            Ok(c) => c,
            Err(_) => return Ok(()), // Can't read — let the tool itself handle the error
        };

        let lines: Vec<&str> = content.lines().collect();

        for anchor_str in [&anchor_start, &anchor_end].into_iter().flatten() {
            if let Some((line_num, expected_hash)) = parse_anchor(anchor_str) {
                if line_num == 0 || line_num > lines.len() {
                    return Err(format!(
                        "Anchor '{anchor_str}' references line {line_num} but file has {} lines. \
                         Re-read the file to get fresh anchors.",
                        lines.len()
                    ));
                }
                let actual_hash = blake3_short(lines[line_num - 1].trim());
                if actual_hash != expected_hash {
                    return Err(format!(
                        "Stale anchor '{anchor_str}': expected hash '{expected_hash}' \
                         but line {line_num} now hashes to '{actual_hash}'. \
                         Re-read the file to get fresh anchors."
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Reject manager→worker delegations that stuff too much text into the `prompt` arg.
pub struct DelegationPromptLengthValidator {
    max_chars: usize,
}

impl DelegationPromptLengthValidator {
    pub fn new(max_chars: usize) -> Self {
        Self { max_chars }
    }
}

impl ActionValidator for DelegationPromptLengthValidator {
    fn name(&self) -> &str {
        "delegation_prompt_length"
    }

    fn validate(
        &self,
        tool_name: &str,
        args_json: &str,
        _state: &ValidatorState,
    ) -> Result<(), String> {
        if !is_delegation_tool(tool_name) {
            return Ok(());
        }
        let Some(prompt) = extract_prompt(args_json) else {
            return Ok(());
        };
        if prompt.chars().count() <= self.max_chars {
            return Ok(());
        }

        Err(format!(
            "Delegation prompt is too long ({} chars > {}). Shorten it to the objective, \
             target files, and the smallest relevant error excerpts. Do NOT paste repo maps, \
             full verifier output, or large file dumps into worker prompts.",
            prompt.chars().count(),
            self.max_chars
        ))
    }
}

/// Reject delegations that explicitly point workers at volatile runtime metadata.
pub struct ForbiddenDelegationPathValidator;

impl ActionValidator for ForbiddenDelegationPathValidator {
    fn name(&self) -> &str {
        "forbidden_delegation_paths"
    }

    fn validate(
        &self,
        tool_name: &str,
        args_json: &str,
        _state: &ValidatorState,
    ) -> Result<(), String> {
        if !is_delegation_tool(tool_name) {
            return Ok(());
        }
        let Some(prompt) = extract_prompt(args_json) else {
            return Ok(());
        };

        let forbidden_hits: Vec<&str> = [
            ".beads/backup",
            ".beads/metadata.json",
            ".beads/push-state.json",
            ".git/",
            ".dolt/",
        ]
        .into_iter()
        .filter(|needle| prompt.contains(needle))
        .collect();

        if forbidden_hits.is_empty() {
            return Ok(());
        }

        Err(format!(
            "Delegation prompt references volatile runtime metadata ({}) which workers must not \
             target. Focus the worker on source files or stable config/docs paths instead.",
            forbidden_hits.join(", ")
        ))
    }
}

/// Create the default set of validators for a worker agent session.
pub fn default_validators(working_dir: &Path) -> Vec<Box<dyn ActionValidator>> {
    vec![
        Box::new(ReadBeforeWriteValidator),
        Box::new(MinContextValidator),
        Box::new(TruncatedJsonValidator),
        Box::new(ConsecutiveReadDetector),
        Box::new(AnchorFreshnessValidator::new(working_dir)),
    ]
}

/// Create validators for manager sessions that delegate to worker agents.
pub fn manager_validators() -> Vec<Box<dyn ActionValidator>> {
    vec![
        Box::new(DelegationPromptLengthValidator::new(
            std::env::var("SWARM_MAX_DELEGATION_PROMPT_CHARS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0)
                .unwrap_or(12_000),
        )),
        Box::new(ForbiddenDelegationPathValidator),
    ]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Parse an anchor string like "42:a3" into (line_number, hash).
fn parse_anchor(s: &str) -> Option<(usize, String)> {
    let parts: Vec<&str> = s.splitn(2, ':').collect();
    if parts.len() != 2 {
        return None;
    }
    let line_num: usize = parts[0].parse().ok()?;
    Some((line_num, parts[1].to_string()))
}

/// Compute a 2-hex-char content hash for a line (trimmed).
pub fn blake3_short(s: &str) -> String {
    let hash = blake3::hash(s.as_bytes());
    format!("{:02x}", hash.as_bytes()[0])
}

/// Best-effort extraction of "path" field from JSON args.
fn extract_path(json: &str) -> Option<String> {
    extract_field(json, "path")
}

/// Best-effort extraction of "prompt" field from JSON args.
fn extract_prompt(json: &str) -> Option<String> {
    extract_field(json, "prompt")
}

/// Best-effort extraction of a string field from JSON.
fn extract_field(json: &str, field: &str) -> Option<String> {
    // Use serde_json for reliable parsing
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    v.get(field)?.as_str().map(|s| s.to_string())
}

fn is_delegation_tool(tool_name: &str) -> bool {
    matches!(
        tool_name.strip_prefix("proxy_").unwrap_or(tool_name),
        "rust_coder"
            | "general_coder"
            | "reasoning_worker"
            | "planner"
            | "fixer"
            | "reviewer"
            | "architect"
            | "editor"
    )
}

/// Check if content looks truncated (unmatched delimiters).
fn looks_truncated(content: &str) -> bool {
    let mut braces = 0i32;
    let mut brackets = 0i32;
    for ch in content.chars() {
        match ch {
            '{' => braces += 1,
            '}' => braces -= 1,
            '[' => brackets += 1,
            ']' => brackets -= 1,
            _ => {}
        }
    }
    // Only flag as truncated if delimiters are opened but never closed
    // (positive imbalance). Negative imbalance is a different kind of error.
    braces > 2 || brackets > 2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blake3_short_deterministic() {
        let h1 = blake3_short("fn main() {");
        let h2 = blake3_short("fn main() {");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 2); // 2 hex chars
    }

    #[test]
    fn test_blake3_short_different_inputs() {
        let h1 = blake3_short("fn main() {");
        let h2 = blake3_short("fn foo() {");
        // Not guaranteed to differ (1/256 chance of collision) but very likely
        // This test just ensures both produce valid 2-char hex strings
        assert_eq!(h1.len(), 2);
        assert_eq!(h2.len(), 2);
    }

    #[test]
    fn test_parse_anchor() {
        assert_eq!(parse_anchor("42:a3"), Some((42, "a3".to_string())));
        assert_eq!(parse_anchor("1:ff"), Some((1, "ff".to_string())));
        assert_eq!(parse_anchor("invalid"), None);
        assert_eq!(parse_anchor(""), None);
    }

    #[test]
    fn test_extract_path() {
        let json = r#"{"path": "src/main.rs", "content": "hello"}"#;
        assert_eq!(extract_path(json), Some("src/main.rs".to_string()));
    }

    #[test]
    fn test_extract_path_missing() {
        let json = r#"{"content": "hello"}"#;
        assert_eq!(extract_path(json), None);
    }

    #[test]
    fn test_looks_truncated_balanced() {
        assert!(!looks_truncated("fn main() { if true { x } }"));
    }

    #[test]
    fn test_looks_truncated_unmatched() {
        assert!(looks_truncated("fn main() { if true { if false { x"));
    }

    #[test]
    fn test_looks_truncated_minor_imbalance() {
        // Single unmatched brace is below threshold (>2)
        assert!(!looks_truncated("fn main() { if true {"));
    }

    #[test]
    fn test_read_before_write_rejects_unread() {
        let v = ReadBeforeWriteValidator;
        let state = ValidatorState::new();
        let args = r#"{"path": "src/main.rs", "old_content": "x", "new_content": "y"}"#;
        assert!(v.validate("edit_file", args, &state).is_err());
    }

    #[test]
    fn test_read_before_write_allows_read_file() {
        let v = ReadBeforeWriteValidator;
        let mut state = ValidatorState::new();
        state.record_read("src/main.rs");
        let args = r#"{"path": "src/main.rs", "old_content": "x", "new_content": "y"}"#;
        assert!(v.validate("edit_file", args, &state).is_ok());
    }

    #[test]
    fn test_read_before_write_ignores_non_write() {
        let v = ReadBeforeWriteValidator;
        let state = ValidatorState::new();
        let args = r#"{"path": "src/main.rs"}"#;
        assert!(v.validate("read_file", args, &state).is_ok());
    }

    #[test]
    fn test_min_context_rejects_single_line() {
        let v = MinContextValidator;
        let state = ValidatorState::new();
        let args = r#"{"path": "x.rs", "old_content": "fn main()", "new_content": "fn foo()"}"#;
        assert!(v.validate("edit_file", args, &state).is_err());
    }

    #[test]
    fn test_min_context_allows_multi_line() {
        let v = MinContextValidator;
        let state = ValidatorState::new();
        let args = r#"{"path": "x.rs", "old_content": "fn main() {\n    hello();\n}", "new_content": "fn foo()"}"#;
        assert!(v.validate("edit_file", args, &state).is_ok());
    }

    #[test]
    fn test_min_context_skipped_for_anchor_mode() {
        let v = MinContextValidator;
        let state = ValidatorState::new();
        // Single-line old_content would normally fail, but anchors are present
        let args = r#"{"path": "x.rs", "old_content": "fn main()", "new_content": "fn foo()", "anchor_start": "1:ab", "anchor_end": "3:cd"}"#;
        assert!(v.validate("edit_file", args, &state).is_ok());
    }

    #[test]
    fn test_min_context_skipped_for_anchor_no_old_content() {
        let v = MinContextValidator;
        let state = ValidatorState::new();
        let args = r#"{"path": "x.rs", "new_content": "fn foo()", "anchor_start": "1:ab", "anchor_end": "3:cd"}"#;
        assert!(v.validate("edit_file", args, &state).is_ok());
    }

    #[test]
    fn test_consecutive_read_warns_after_3() {
        let v = ConsecutiveReadDetector;
        let mut state = ValidatorState::new();
        state.record_read("src/main.rs");
        state.record_read("src/main.rs");
        state.record_read("src/main.rs");
        let args = r#"{"path": "src/main.rs"}"#;
        assert!(v.validate("read_file", args, &state).is_err());
    }

    #[test]
    fn test_consecutive_read_ok_under_3() {
        let v = ConsecutiveReadDetector;
        let mut state = ValidatorState::new();
        state.record_read("src/main.rs");
        state.record_read("src/main.rs");
        let args = r#"{"path": "src/main.rs"}"#;
        assert!(v.validate("read_file", args, &state).is_ok());
    }

    #[test]
    fn test_consecutive_read_resets_on_write() {
        let v = ConsecutiveReadDetector;
        let mut state = ValidatorState::new();
        state.record_read("src/main.rs");
        state.record_read("src/main.rs");
        state.record_read("src/main.rs");
        state.record_write("src/main.rs"); // resets
        let args = r#"{"path": "src/main.rs"}"#;
        assert!(v.validate("read_file", args, &state).is_ok());
    }

    #[test]
    fn test_truncated_json_detects_missing_braces() {
        let v = TruncatedJsonValidator;
        let state = ValidatorState::new();
        let args = r#"{"path": "x.rs", "new_content": "fn main() { if true { if false { x"}"#;
        assert!(v.validate("write_file", args, &state).is_err());
    }

    #[test]
    fn test_truncated_json_allows_balanced() {
        let v = TruncatedJsonValidator;
        let state = ValidatorState::new();
        let args = r#"{"path": "x.rs", "new_content": "fn main() { println!(\"hi\"); }"}"#;
        assert!(v.validate("write_file", args, &state).is_ok());
    }

    #[test]
    fn test_validator_state_default() {
        let state = ValidatorState::default();
        assert!(state.files_read.is_empty());
        assert!(state.files_written.is_empty());
        assert!(state.read_counts.is_empty());
        assert_eq!(state.turn, 0);
    }

    #[test]
    fn test_proxy_prefix_stripping() {
        let v = ReadBeforeWriteValidator;
        let mut state = ValidatorState::new();
        state.record_read("src/main.rs");
        let args = r#"{"path": "src/main.rs", "old_content": "x\ny\n", "new_content": "z"}"#;
        // proxy_edit_file should be treated the same as edit_file
        assert!(v.validate("proxy_edit_file", args, &state).is_ok());
    }

    #[test]
    fn test_delegation_prompt_length_validator_rejects_long_prompt() {
        let v = DelegationPromptLengthValidator::new(20);
        let state = ValidatorState::new();
        let args = r#"{"prompt":"This prompt is definitely longer than twenty characters"}"#;
        assert!(v.validate("proxy_rust_coder", args, &state).is_err());
    }

    #[test]
    fn test_delegation_prompt_length_validator_ignores_non_worker_tools() {
        let v = DelegationPromptLengthValidator::new(5);
        let state = ValidatorState::new();
        let args = r#"{"prompt":"too long"}"#;
        assert!(v.validate("proxy_run_verifier", args, &state).is_ok());
    }

    #[test]
    fn test_forbidden_delegation_path_validator_rejects_runtime_metadata() {
        let v = ForbiddenDelegationPathValidator;
        let state = ValidatorState::new();
        let args = r#"{"prompt":"Only modify these files:\n- `.beads/backup/backup_state.json`"}"#;
        assert!(v.validate("proxy_general_coder", args, &state).is_err());
    }

    #[test]
    fn test_forbidden_delegation_path_validator_allows_stable_beads_docs() {
        let v = ForbiddenDelegationPathValidator;
        let state = ValidatorState::new();
        let args = r#"{"prompt":"Update `.beads/PRIME.md` and `scripts/bd-safe.sh`"}"#;
        assert!(v.validate("proxy_general_coder", args, &state).is_ok());
    }
}
