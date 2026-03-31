//! Self-Correcting Task Reformulation Engine
//!
//! When an issue fails repeatedly, this module classifies WHY it failed and
//! reformulates the task to be solvable within actual constraints, while
//! guarding against shortcuts that weaken the original intent.
//!
//! # Architecture
//!
//! ```text
//! Session fails → Classify failure → Choose recovery action
//!     → Intent Guard (does rewrite preserve goal?) → Update bead → Retry
//! ```
//!
//! # Storage
//!
//! - `.swarm/intent-contracts.jsonl` — Original task goals (append-only, first attempt)
//! - `.swarm/reformulations.jsonl` — Audit trail of all rewrites

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::telemetry::FailureLedgerEntry;

// ── Phase 1: Immutable Intent Contract ──────────────────────────────

/// Captures the original task goal on first pickup so reformulations
/// can't silently weaken it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentContract {
    pub issue_id: String,
    pub original_title: String,
    pub original_description: Option<String>,
    /// Key outcomes extracted from the description (e.g., "add import", "replace types").
    pub required_outcomes: Vec<String>,
    /// Verifier gates that must pass for this issue to be considered resolved.
    pub acceptance_signals: Vec<String>,
    /// BLAKE3 digest of title + description for change detection.
    pub intent_digest: String,
    pub created_at: DateTime<Utc>,
}

impl IntentContract {
    /// Create a new intent contract from an issue.
    pub fn from_issue(issue_id: &str, title: &str, description: Option<&str>) -> Self {
        let required_outcomes = extract_required_outcomes(title, description);
        let acceptance_signals = extract_acceptance_signals(description);
        let intent_digest = content_digest(title, description);

        Self {
            issue_id: issue_id.to_string(),
            original_title: title.to_string(),
            original_description: description.map(|s| s.to_string()),
            required_outcomes,
            acceptance_signals,
            intent_digest,
            created_at: Utc::now(),
        }
    }
}

// ── Phase 2: Failure Classification ─────────────────────────────────

/// Why a session failed — derived from failure ledger + mutation archive data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum FailureClassification {
    /// Issue instructs a command blocked by the exec_tool allowlist.
    ToolConstraintMismatch { blocked_command: String },
    /// Issue says "run X to verify" but X is unavailable.
    VerificationInstructionMismatch { instruction: String },
    /// Ambiguous/wrong file names, missing targets.
    TaskFormulationDefect { detail: String },
    /// Agent reads many turns without locating target files.
    ContextDeficit { read_turns: usize },
    /// Multi-file coordination failure, cross-file regressions.
    DecompositionRequired { affected_files: Vec<String> },
    /// Same file edited/reverted, no net progress.
    ImplementationThrash { thrash_file: String },
    /// Clear compiler/test failure with adequate task wording.
    GenuineCodeDefect,
    /// Timeout, rate limit, endpoint down.
    InfraTransient { detail: String },
}

impl FailureClassification {
    /// Returns a stable fingerprint for grouping identical failure types.
    pub fn fingerprint(&self) -> String {
        match self {
            Self::ToolConstraintMismatch { blocked_command } => {
                format!("tool_constraint:{blocked_command}")
            }
            Self::VerificationInstructionMismatch { instruction } => {
                format!("verify_mismatch:{}", truncate(instruction, 60))
            }
            Self::TaskFormulationDefect { detail } => {
                format!("formulation_defect:{}", truncate(detail, 60))
            }
            Self::ContextDeficit { read_turns } => {
                format!("context_deficit:{read_turns}")
            }
            Self::DecompositionRequired { affected_files } => {
                format!("decomp_needed:{}", affected_files.len())
            }
            Self::ImplementationThrash { thrash_file } => {
                format!("impl_thrash:{thrash_file}")
            }
            Self::GenuineCodeDefect => "genuine_defect".to_string(),
            Self::InfraTransient { detail } => {
                format!("infra_transient:{}", truncate(detail, 40))
            }
        }
    }
}

/// What to do about a classified failure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RecoveryAction {
    /// Rewrite the issue description (next loop iteration picks it up).
    RewriteIssue,
    /// Rewrite AND signal immediate retry (used for ToolConstraintMismatch).
    RewriteAndRetryNow,
    /// Add a directive note without changing the description.
    AppendDirective { directive: String },
    /// Retry with more context injected (e.g., file paths, prior fix patterns).
    RetryWithMoreContext,
    /// Create child subtask issues via `create_molecule`.
    DecomposeIntoSubtasks,
    /// Create escalation bead for human review.
    EscalateToHuman { reason: String },
    /// Transient failure — retry after cooldown (no rewrite needed).
    CooldownRetry,
    /// No rewrite needed — normal retry (genuine code defect).
    NormalRetry,
}

// ── Phase 3: Reformulation Engine ───────────────────────────────────

/// Input data for failure classification.
pub struct FailureReviewInput {
    pub issue_id: String,
    pub issue_title: String,
    pub issue_description: Option<String>,
    pub failure_ledger: Vec<FailureLedgerEntry>,
    pub iterations_used: u32,
    pub max_iterations: u32,
    pub files_changed: Vec<String>,
    pub error_categories: Vec<String>,
    pub failure_reason: Option<String>,
}

/// Audit trail record for each reformulation attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReformulationRecord {
    pub issue_id: String,
    pub attempt_index: u32,
    pub classification: FailureClassification,
    pub action: RecoveryAction,
    pub prior_description_digest: String,
    pub new_description_digest: Option<String>,
    pub failure_fingerprint: String,
    pub intent_guard_passed: bool,
    pub timestamp: DateTime<Utc>,
}

// ── Phase 4: Intent Guard ───────────────────────────────────────────

/// Result of checking a proposed rewrite against the original intent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntentGuardVerdict {
    pub allowed: bool,
    pub violations: Vec<String>,
    pub preserved_requirements: Vec<String>,
    pub removed_requirements: Vec<String>,
}

// ── Reformulation Store ─────────────────────────────────────────────

/// Manages append-only JSONL storage for intent contracts and reformulations.
pub struct ReformulationStore {
    contracts_path: PathBuf,
    reformulations_path: PathBuf,
}

impl ReformulationStore {
    pub fn new(repo_root: &Path) -> Self {
        let swarm_dir = repo_root.join(".swarm");
        Self {
            contracts_path: swarm_dir.join("intent-contracts.jsonl"),
            reformulations_path: swarm_dir.join("reformulations.jsonl"),
        }
    }

    /// Save an intent contract (only if one doesn't already exist for this issue).
    pub fn save_contract(&self, contract: &IntentContract) {
        if self.load_contract(&contract.issue_id).is_some() {
            debug!(issue = %contract.issue_id, "Intent contract already exists — skipping");
            return;
        }

        append_jsonl(&self.contracts_path, contract);
        info!(
            issue = %contract.issue_id,
            outcomes = contract.required_outcomes.len(),
            "Saved intent contract"
        );
    }

    /// Load the intent contract for an issue (if one exists).
    pub fn load_contract(&self, issue_id: &str) -> Option<IntentContract> {
        load_jsonl::<IntentContract>(&self.contracts_path)
            .into_iter()
            .find(|c| c.issue_id == issue_id)
    }

    /// Record a reformulation attempt.
    pub fn record_reformulation(&self, record: &ReformulationRecord) {
        append_jsonl(&self.reformulations_path, record);
        info!(
            issue = %record.issue_id,
            attempt = record.attempt_index,
            classification = ?record.classification,
            action = ?record.action,
            guard_passed = record.intent_guard_passed,
            "Recorded reformulation"
        );
    }

    /// Count reformulation attempts for an issue with the same failure fingerprint.
    pub fn count_same_fingerprint(&self, issue_id: &str, fingerprint: &str) -> u32 {
        load_jsonl::<ReformulationRecord>(&self.reformulations_path)
            .into_iter()
            .filter(|r| r.issue_id == issue_id && r.failure_fingerprint == fingerprint)
            .count() as u32
    }

    /// Get the next attempt index for an issue.
    pub fn next_attempt_index(&self, issue_id: &str) -> u32 {
        load_jsonl::<ReformulationRecord>(&self.reformulations_path)
            .into_iter()
            .filter(|r| r.issue_id == issue_id)
            .count() as u32
    }

    /// Load all reformulation records for an issue.
    pub fn records_for_issue(&self, issue_id: &str) -> Vec<ReformulationRecord> {
        load_jsonl::<ReformulationRecord>(&self.reformulations_path)
            .into_iter()
            .filter(|r| r.issue_id == issue_id)
            .collect()
    }
}

// ── Classification Engine ───────────────────────────────────────────

/// Classify why a session failed based on collected failure data.
///
/// This is purely rule-based — no LLM call needed.
pub fn classify_failure(input: &FailureReviewInput) -> FailureClassification {
    let ledger = &input.failure_ledger;
    let desc = input.issue_description.as_deref().unwrap_or("");

    // Rule 1: Tool constraint mismatch — exec_tool / command_not_allowed
    if let Some(blocked) = detect_tool_constraint_mismatch(ledger, desc) {
        return FailureClassification::ToolConstraintMismatch {
            blocked_command: blocked,
        };
    }

    // Rule 2: Verification instruction mismatch
    if let Some(instruction) = detect_verification_mismatch(ledger, desc) {
        return FailureClassification::VerificationInstructionMismatch { instruction };
    }

    // Rule 3: Infra transient — timeout, rate limit (check early, before context deficit)
    if let Some(detail) = detect_infra_transient(ledger, &input.failure_reason) {
        return FailureClassification::InfraTransient { detail };
    }

    // Rule 4: Implementation thrash — same file edited and reverted
    if let Some(thrash_file) = detect_implementation_thrash(ledger) {
        return FailureClassification::ImplementationThrash { thrash_file };
    }

    // Rule 5: Decomposition needed — many files changed with regressions
    if input.files_changed.len() > 8 {
        return FailureClassification::DecompositionRequired {
            affected_files: input.files_changed.clone(),
        };
    }

    // Rule 6: Context deficit — agent spent most turns reading without writing
    let read_turns = count_read_only_turns(ledger);
    let total_turns = input.iterations_used;
    if total_turns > 3 && read_turns as u32 > total_turns * 2 / 3 {
        return FailureClassification::ContextDeficit { read_turns };
    }

    // Rule 7: Task formulation defect — exhausted iterations with no changes
    if total_turns >= input.max_iterations && input.files_changed.is_empty() {
        return FailureClassification::TaskFormulationDefect {
            detail: "Exhausted all iterations without making changes".to_string(),
        };
    }

    // Default: genuine code defect (the task wording is fine)
    FailureClassification::GenuineCodeDefect
}

/// Choose the recovery action for a classified failure.
pub fn choose_recovery_action(
    classification: &FailureClassification,
    prior_reformulations: u32,
) -> RecoveryAction {
    // Exhaustion guard: after 3 reformulations with same fingerprint, escalate
    if prior_reformulations >= 3 {
        return RecoveryAction::EscalateToHuman {
            reason: format!(
                "Reformulation exhausted: {prior_reformulations} attempts with same failure pattern"
            ),
        };
    }

    match classification {
        FailureClassification::ToolConstraintMismatch { .. } => RecoveryAction::RewriteAndRetryNow,
        FailureClassification::VerificationInstructionMismatch { instruction } => {
            RecoveryAction::AppendDirective {
                directive: format!(
                    "Do NOT run `{instruction}` — it is not available in this environment. \
                     Make the changes directly; the verifier gates will validate."
                ),
            }
        }
        FailureClassification::TaskFormulationDefect { .. } => RecoveryAction::RewriteIssue,
        FailureClassification::ContextDeficit { .. } => RecoveryAction::RetryWithMoreContext,
        FailureClassification::DecompositionRequired { .. } => {
            RecoveryAction::DecomposeIntoSubtasks
        }
        FailureClassification::ImplementationThrash { .. } => RecoveryAction::DecomposeIntoSubtasks,
        FailureClassification::GenuineCodeDefect => RecoveryAction::NormalRetry,
        FailureClassification::InfraTransient { .. } => RecoveryAction::CooldownRetry,
    }
}

// ── Reformulation Application ───────────────────────────────────────

/// Result of applying a reformulation.
pub struct ReformulationResult {
    pub classification: FailureClassification,
    pub action: RecoveryAction,
    pub new_description: Option<String>,
    pub notes_appended: Option<String>,
    pub intent_guard_passed: bool,
    /// If true, the caller should retry immediately (not wait for next loop).
    pub retry_now: bool,
    /// If true, the issue is exhausted and needs human review.
    pub escalated: bool,
}

/// Run the full reformulation pipeline on a failed issue.
///
/// 1. Classify the failure
/// 2. Check prior reformulations for exhaustion
/// 3. Generate rewrite (template-based for common cases)
/// 4. Run intent guard against the contract
/// 5. Record the reformulation
/// 6. Return the result for the caller to apply
pub fn reformulate(store: &ReformulationStore, input: &FailureReviewInput) -> ReformulationResult {
    // Step 1: Classify
    let classification = classify_failure(input);
    let fingerprint = classification.fingerprint();

    // Step 2: Check exhaustion
    let prior_count = store.count_same_fingerprint(&input.issue_id, &fingerprint);
    let attempt_index = store.next_attempt_index(&input.issue_id);
    let action = choose_recovery_action(&classification, prior_count);

    info!(
        issue = %input.issue_id,
        classification = ?classification,
        fingerprint = %fingerprint,
        prior_same = prior_count,
        action = ?action,
        "Failure classified"
    );

    // Step 3: Generate rewrite based on action
    let (new_description, notes_appended) = match &action {
        RecoveryAction::RewriteAndRetryNow | RecoveryAction::RewriteIssue => {
            let rewrite = generate_rewrite(&classification, input);
            (Some(rewrite), None)
        }
        RecoveryAction::AppendDirective { directive } => (None, Some(directive.clone())),
        RecoveryAction::RetryWithMoreContext => {
            let note = format!(
                "## Context Enrichment (reformulation attempt {})\n\n\
                 The agent struggled to locate target files. Provide explicit file paths \
                 and function names in the description.",
                attempt_index + 1
            );
            (None, Some(note))
        }
        _ => (None, None),
    };

    // Step 4: Intent guard
    let contract = store.load_contract(&input.issue_id);
    let intent_guard_passed =
        if let (Some(ref new_desc), Some(ref contract)) = (&new_description, &contract) {
            let verdict = check_intent_guard(contract, new_desc);
            if !verdict.allowed {
                warn!(
                    issue = %input.issue_id,
                    violations = ?verdict.violations,
                    "Intent guard REJECTED rewrite — falling back to directive append"
                );
            }
            verdict.allowed
        } else {
            true // No rewrite or no contract → guard trivially passes
        };

    // Step 5: Record
    let record = ReformulationRecord {
        issue_id: input.issue_id.clone(),
        attempt_index,
        classification: classification.clone(),
        action: action.clone(),
        prior_description_digest: content_digest(
            &input.issue_title,
            input.issue_description.as_deref(),
        ),
        new_description_digest: new_description
            .as_deref()
            .map(|d| content_digest(&input.issue_title, Some(d))),
        failure_fingerprint: fingerprint,
        intent_guard_passed,
        timestamp: Utc::now(),
    };
    store.record_reformulation(&record);

    // Step 6: Build result
    let retry_now = matches!(action, RecoveryAction::RewriteAndRetryNow);
    let escalated = matches!(action, RecoveryAction::EscalateToHuman { .. });

    // If intent guard failed, don't apply the rewrite
    let final_description = if intent_guard_passed {
        new_description
    } else {
        None
    };

    ReformulationResult {
        classification,
        action,
        new_description: final_description,
        notes_appended,
        intent_guard_passed,
        retry_now,
        escalated,
    }
}

// ── Template-Based Rewrites ─────────────────────────────────────────

/// Generate a rewritten description for the given failure classification.
///
/// Uses template-based rewrites for common cases (no LLM needed).
fn generate_rewrite(classification: &FailureClassification, input: &FailureReviewInput) -> String {
    let desc = input
        .issue_description
        .as_deref()
        .unwrap_or(&input.issue_title);

    match classification {
        FailureClassification::ToolConstraintMismatch { blocked_command } => {
            rewrite_remove_blocked_command(desc, blocked_command)
        }
        FailureClassification::VerificationInstructionMismatch { instruction } => {
            rewrite_remove_verification_instruction(desc, instruction)
        }
        FailureClassification::TaskFormulationDefect { .. } => rewrite_clarify_task(desc),
        _ => desc.to_string(),
    }
}

/// Remove references to a blocked command from the description.
///
/// Replaces "Run: X" / "Verify: X" / "Check: X" / backtick-quoted commands
/// with a directive to let the verifier handle validation.
fn rewrite_remove_blocked_command(desc: &str, blocked_cmd: &str) -> String {
    let blocked_lower = blocked_cmd.to_lowercase();
    let mut result = String::with_capacity(desc.len());
    let mut in_code_block = false;

    for line in desc.lines() {
        let trimmed = line.trim().to_lowercase();

        // Track code block boundaries
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            // Skip entire code block line if it references the blocked command
            if trimmed.contains(&blocked_lower) {
                continue;
            }
        }

        // Skip lines that instruct running the blocked command
        if trimmed.contains(&blocked_lower)
            && (trimmed.starts_with("run:")
                || trimmed.starts_with("run ")
                || trimmed.starts_with("verify:")
                || trimmed.starts_with("check:")
                || trimmed.starts_with("- run")
                || (in_code_block && trimmed.contains(&blocked_lower)))
        {
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }

    result.push_str(
        "\n**Note:** Make the changes directly. \
         The verifier's lint and format gates will validate.\n",
    );

    result.trim().to_string()
}

/// Remove verification instructions from the description.
fn rewrite_remove_verification_instruction(desc: &str, instruction: &str) -> String {
    let inst_lower = instruction.to_lowercase();
    let mut result = String::with_capacity(desc.len());

    for line in desc.lines() {
        if line.trim().to_lowercase().contains(&inst_lower) {
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }

    result.push_str("\n**Note:** The verifier gates will validate your changes automatically.\n");

    result.trim().to_string()
}

/// Clarify an ambiguous task description.
fn rewrite_clarify_task(desc: &str) -> String {
    format!(
        "{desc}\n\n**Note:** Previous attempt exhausted all iterations without making changes. \
         Focus on making targeted edits early rather than extensive reading."
    )
}

// ── Intent Guard ────────────────────────────────────────────────────

/// Check whether a proposed rewrite preserves the original intent.
///
/// Compares extracted outcomes from the new description against the contract.
/// The title is never rewritten (only the description is), so outcomes matching
/// the original title are always considered preserved.
fn check_intent_guard(contract: &IntentContract, new_description: &str) -> IntentGuardVerdict {
    let new_outcomes = extract_required_outcomes_from_text(new_description);
    let title_lower = contract.original_title.to_lowercase();
    // Normalize: strip backticks/quotes for fuzzy matching
    let new_desc_normalized = new_description.to_lowercase().replace(['`', '\'', '"'], "");

    let mut preserved = Vec::new();
    let mut removed = Vec::new();
    let mut violations = Vec::new();

    for req in &contract.required_outcomes {
        let req_lower = req.to_lowercase();
        let req_normalized = req_lower.replace(['`', '\'', '"'], "");

        // The title is never rewritten — if this outcome IS the title, it's preserved.
        // Use strict matching (exact or substring) to avoid false positives.
        if req_lower == title_lower
            || title_lower.contains(&req_lower)
            || req_lower.contains(&title_lower)
        {
            preserved.push(req.clone());
            continue;
        }

        // Check if any new outcome covers this requirement
        let found = new_outcomes.iter().any(|o| {
            let o_lower = o.to_lowercase();
            let o_normalized = o_lower.replace(['`', '\'', '"'], "");
            o_normalized.contains(&req_normalized)
                || req_normalized.contains(&o_normalized)
                || key_terms_overlap(&req_lower, &o_lower)
        }) || new_desc_normalized.contains(&req_normalized);

        if found {
            preserved.push(req.clone());
        } else {
            removed.push(req.clone());
            violations.push(format!("Required outcome missing: {req}"));
        }
    }

    // Allow removal of verification-only requirements (Run:/Verify:/Check:)
    let non_verification_removed: Vec<_> = removed
        .iter()
        .filter(|r| {
            let rl = r.to_lowercase();
            !rl.starts_with("run:")
                && !rl.starts_with("verify:")
                && !rl.starts_with("check:")
                && !rl.starts_with("- run")
        })
        .collect();

    let allowed = non_verification_removed.is_empty();

    IntentGuardVerdict {
        allowed,
        violations,
        preserved_requirements: preserved,
        removed_requirements: removed,
    }
}

// ── Phase 5: Issue Selection ────────────────────────────────────────

/// Check if an issue should be skipped based on reformulation state.
///
/// Returns `Some(reason)` if the issue should be skipped, `None` if it's workable.
pub fn should_skip_issue(store: &ReformulationStore, issue_id: &str) -> Option<String> {
    let records = store.records_for_issue(issue_id);
    if records.is_empty() {
        return None;
    }

    // Check for escalation (any record with EscalateToHuman action)
    if records
        .iter()
        .any(|r| matches!(r.action, RecoveryAction::EscalateToHuman { .. }))
    {
        return Some("Reformulation exhausted — needs human review".to_string());
    }

    // Check if the most recent reformulation had a new description applied
    // If yes, allow retry (the description was rewritten)
    if let Some(last) = records.last() {
        if last.new_description_digest.is_some() && last.intent_guard_passed {
            return None; // Rewrite was applied, allow retry
        }
    }

    None
}

// ── Helper Functions ────────────────────────────────────────────────

/// Compute BLAKE3 digest of title + description (truncated to 16 hex chars).
fn content_digest(title: &str, description: Option<&str>) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(title.as_bytes());
    if let Some(desc) = description {
        hasher.update(desc.as_bytes());
    }
    hasher.finalize().to_hex()[..16].to_string()
}

/// Extract required outcomes from an issue title + description.
fn extract_required_outcomes(title: &str, description: Option<&str>) -> Vec<String> {
    let mut outcomes = vec![title.to_string()];
    if let Some(desc) = description {
        outcomes.extend(extract_required_outcomes_from_text(desc));
    }
    outcomes
}

/// Extract action items from description text.
fn extract_required_outcomes_from_text(text: &str) -> Vec<String> {
    let mut outcomes = Vec::new();

    for line in text.lines() {
        let trimmed = line.trim();

        // Capture bullet-point items (common in issue descriptions)
        if let Some(item) = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "))
        {
            if item.len() > 5 {
                outcomes.push(item.to_string());
            }
        }

        // Capture "Then X" patterns
        if trimmed.to_lowercase().starts_with("then ") {
            outcomes.push(trimmed.to_string());
        }

        // Capture "Run: X" / "Verify: X" patterns (so we can track their removal)
        let lower = trimmed.to_lowercase();
        if lower.starts_with("run:") || lower.starts_with("verify:") || lower.starts_with("check:")
        {
            outcomes.push(trimmed.to_string());
        }
    }

    outcomes
}

/// Extract acceptance signals (verification commands) from description.
fn extract_acceptance_signals(description: Option<&str>) -> Vec<String> {
    let Some(desc) = description else {
        return vec!["verifier_all_green".to_string()];
    };

    let mut signals = Vec::new();
    for line in desc.lines() {
        let lower = line.trim().to_lowercase();
        if lower.starts_with("run:") || lower.starts_with("verify:") || lower.starts_with("check:")
        {
            signals.push(line.trim().to_string());
        }
    }

    if signals.is_empty() {
        signals.push("verifier_all_green".to_string());
    }
    signals
}

/// Check for key term overlap between two strings.
fn key_terms_overlap(a: &str, b: &str) -> bool {
    let a_terms: HashSet<&str> = a.split_whitespace().filter(|w| w.len() > 3).collect();
    let b_terms: HashSet<&str> = b.split_whitespace().filter(|w| w.len() > 3).collect();

    let overlap = a_terms.intersection(&b_terms).count();
    let min_terms = a_terms.len().min(b_terms.len());

    min_terms > 0 && overlap as f64 / min_terms as f64 > 0.3
}

/// Detect if the failure was caused by a blocked tool/command.
fn detect_tool_constraint_mismatch(
    ledger: &[FailureLedgerEntry],
    description: &str,
) -> Option<String> {
    // Look for failure entries with command-not-allowed patterns
    for entry in ledger {
        if !entry.success {
            let signal = entry.signal_traced.to_lowercase();
            if signal.contains("not in allowlist")
                || signal.contains("command not allowed")
                || signal.contains("not in the command allowlist")
            {
                let cmd = extract_blocked_command(&entry.signal_traced);
                if !cmd.is_empty() {
                    return Some(cmd);
                }
            }
        }
    }

    // Check if description mentions commands known to be blocked by typical
    // allowlists, AND the ledger shows exec failures (suggesting the agent tried)
    let blocked_commands = [
        "ruff", "mypy", "pylint", "flake8", "black", "isort", "pytest", "tox",
    ];
    let desc_lower = description.to_lowercase();
    let exec_failures = ledger
        .iter()
        .filter(|e| !e.success && (e.tool.contains("exec") || e.tool.contains("bash")))
        .count();

    if exec_failures > 0 {
        for cmd in &blocked_commands {
            if desc_lower.contains(&format!("run: {cmd}"))
                || desc_lower.contains(&format!("run: `{cmd}"))
                || desc_lower.contains(&format!("run {cmd}"))
                || desc_lower.contains(&format!("`{cmd} "))
            {
                return Some(cmd.to_string());
            }
        }
    }

    None
}

/// Detect verification instruction mismatches.
fn detect_verification_mismatch(
    ledger: &[FailureLedgerEntry],
    description: &str,
) -> Option<String> {
    let has_exec_failures = ledger
        .iter()
        .any(|e| !e.success && (e.tool.contains("exec") || e.tool.contains("bash")));

    if !has_exec_failures {
        return None;
    }

    for line in description.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_lowercase();
        if lower.starts_with("run:") || lower.starts_with("verify:") || lower.starts_with("check:")
        {
            return Some(trimmed.to_string());
        }
    }

    None
}

/// Count iterations where the agent only read files (no writes).
fn count_read_only_turns(ledger: &[FailureLedgerEntry]) -> usize {
    let mut iterations_with_write: HashSet<usize> = HashSet::new();
    let mut all_iterations: HashSet<usize> = HashSet::new();

    for entry in ledger {
        all_iterations.insert(entry.iteration);
        if entry.tool.contains("edit")
            || entry.tool.contains("write")
            || entry.tool.contains("create")
        {
            iterations_with_write.insert(entry.iteration);
        }
    }

    all_iterations
        .len()
        .saturating_sub(iterations_with_write.len())
}

/// Detect implementation thrash (same file edited/reverted multiple times).
fn detect_implementation_thrash(ledger: &[FailureLedgerEntry]) -> Option<String> {
    let mut edit_counts: HashMap<String, usize> = HashMap::new();

    for entry in ledger {
        if (entry.tool.contains("edit") || entry.tool.contains("write")) && !entry.success {
            if let Some(ref path) = entry.file_path {
                *edit_counts.entry(path.clone()).or_insert(0) += 1;
            }
        }
    }

    edit_counts
        .into_iter()
        .filter(|(_, count)| *count >= 4)
        .max_by_key(|(_, count)| *count)
        .map(|(file, _)| file)
}

/// Detect infrastructure transient failures.
fn detect_infra_transient(
    ledger: &[FailureLedgerEntry],
    failure_reason: &Option<String>,
) -> Option<String> {
    if let Some(reason) = failure_reason {
        let reason_lower = reason.to_lowercase();
        if reason_lower.contains("timeout")
            || reason_lower.contains("rate limit")
            || reason_lower.contains("503")
            || reason_lower.contains("502")
            || reason_lower.contains("connection refused")
        {
            return Some(reason.clone());
        }
    }

    let timeout_count = ledger
        .iter()
        .filter(|e| {
            let s = e.signal_traced.to_lowercase();
            s.contains("timeout") || s.contains("connection refused") || s.contains("rate limit")
        })
        .count();

    if timeout_count >= 3 {
        return Some(format!("{timeout_count} timeout/connection failures"));
    }

    None
}

/// Extract a blocked command name from an error message.
fn extract_blocked_command(signal: &str) -> String {
    // Patterns like "command 'ruff' not in allowlist"
    // or "'ruff check' is not allowed"
    for delim in ['\'', '`', '"'] {
        if let Some(start) = signal.find(delim) {
            if let Some(end) = signal[start + 1..].find(delim) {
                let cmd = &signal[start + 1..start + 1 + end];
                // Return just the command name (first word)
                return cmd.split_whitespace().next().unwrap_or(cmd).to_string();
            }
        }
    }
    String::new()
}

/// Truncate a string to max_len chars.
fn truncate(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len]
    }
}

/// Load all records from a JSONL file.
fn load_jsonl<T: serde::de::DeserializeOwned>(path: &Path) -> Vec<T> {
    match std::fs::read_to_string(path) {
        Ok(content) => content
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// Append a single record to a JSONL file.
fn append_jsonl<T: Serialize>(path: &Path, record: &T) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    match serde_json::to_string(record) {
        Ok(json) => match std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            Ok(mut file) => {
                if let Err(e) = writeln!(file, "{json}") {
                    warn!(error = %e, path = %path.display(), "Failed to write JSONL");
                }
            }
            Err(e) => {
                warn!(error = %e, path = %path.display(), "Failed to open JSONL file");
            }
        },
        Err(e) => {
            warn!(error = %e, "Failed to serialize JSONL record");
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_entry(
        tool: &str,
        error: &str,
        signal: &str,
        success: bool,
        iteration: usize,
    ) -> FailureLedgerEntry {
        FailureLedgerEntry {
            tool: tool.to_string(),
            error_class: error.to_string(),
            signal_traced: signal.to_string(),
            file_path: None,
            iteration,
            timestamp: "2026-03-22T00:00:00Z".to_string(),
            success,
        }
    }

    #[test]
    fn classify_tool_constraint_mismatch() {
        let input = FailureReviewInput {
            issue_id: "test-001".into(),
            issue_title: "Add annotations".into(),
            issue_description: Some("Run: ruff check --select UP006".into()),
            failure_ledger: vec![make_entry(
                "exec_tool",
                "CommandBlocked",
                "command 'ruff' not in allowlist",
                false,
                1,
            )],
            iterations_used: 10,
            max_iterations: 10,
            files_changed: vec![],
            error_categories: vec![],
            failure_reason: None,
        };

        let classification = classify_failure(&input);
        assert!(matches!(
            classification,
            FailureClassification::ToolConstraintMismatch {
                blocked_command
            } if blocked_command == "ruff"
        ));
    }

    #[test]
    fn classify_infra_transient() {
        let input = FailureReviewInput {
            issue_id: "test-003".into(),
            issue_title: "Fix bug".into(),
            issue_description: Some("Fix the bug".into()),
            failure_ledger: vec![],
            iterations_used: 2,
            max_iterations: 10,
            files_changed: vec!["src/main.rs".into()],
            error_categories: vec![],
            failure_reason: Some("timeout after 300s".into()),
        };

        let classification = classify_failure(&input);
        assert!(matches!(
            classification,
            FailureClassification::InfraTransient { .. }
        ));
    }

    #[test]
    fn classify_genuine_defect() {
        let input = FailureReviewInput {
            issue_id: "test-004".into(),
            issue_title: "Fix borrow checker error".into(),
            issue_description: Some("Fix the lifetime issue in parser.rs".into()),
            failure_ledger: vec![
                make_entry("edit_file", "Ok", "", true, 1),
                make_entry("edit_file", "CompileError", "borrow checker", false, 2),
            ],
            iterations_used: 5,
            max_iterations: 10,
            files_changed: vec!["src/parser.rs".into()],
            error_categories: vec!["BorrowChecker".into()],
            failure_reason: None,
        };

        let classification = classify_failure(&input);
        assert!(matches!(
            classification,
            FailureClassification::GenuineCodeDefect
        ));
    }

    #[test]
    fn classify_task_formulation_defect() {
        let ledger: Vec<_> = (0..10)
            .map(|i| make_entry("read_file", "Ok", "", true, i))
            .collect();

        let input = FailureReviewInput {
            issue_id: "test-005".into(),
            issue_title: "Fix something".into(),
            issue_description: Some("Fix the thing".into()),
            failure_ledger: ledger,
            iterations_used: 10,
            max_iterations: 10,
            files_changed: vec![],
            error_categories: vec![],
            failure_reason: None,
        };

        let classification = classify_failure(&input);
        // Either ContextDeficit or TaskFormulationDefect — both are valid
        // since all iterations were read-only AND no files changed
        assert!(
            matches!(
                classification,
                FailureClassification::ContextDeficit { .. }
                    | FailureClassification::TaskFormulationDefect { .. }
            ),
            "Expected ContextDeficit or TaskFormulationDefect, got {classification:?}"
        );
    }

    #[test]
    fn recovery_action_for_tool_mismatch() {
        let classification = FailureClassification::ToolConstraintMismatch {
            blocked_command: "ruff".into(),
        };
        let action = choose_recovery_action(&classification, 0);
        assert!(matches!(action, RecoveryAction::RewriteAndRetryNow));
    }

    #[test]
    fn recovery_action_exhausted() {
        let classification = FailureClassification::ToolConstraintMismatch {
            blocked_command: "ruff".into(),
        };
        let action = choose_recovery_action(&classification, 3);
        assert!(matches!(action, RecoveryAction::EscalateToHuman { .. }));
    }

    #[test]
    fn intent_guard_rejects_requirement_removal() {
        let contract = IntentContract {
            issue_id: "test".into(),
            original_title: "Add annotations and fix imports".into(),
            original_description: Some(
                "- Add `from __future__ import annotations`\n\
                 - Fix circular imports in utils.py"
                    .into(),
            ),
            required_outcomes: vec![
                "Add annotations and fix imports".into(),
                "Add `from __future__ import annotations`".into(),
                "Fix circular imports in utils.py".into(),
            ],
            acceptance_signals: vec!["verifier_all_green".into()],
            intent_digest: "abc123".into(),
            created_at: Utc::now(),
        };

        // Rewrite that drops the circular imports requirement
        let new_desc = "- Add `from __future__ import annotations`\n\
                         Make the changes directly.";

        let verdict = check_intent_guard(&contract, new_desc);
        assert!(
            !verdict.allowed,
            "Should reject removing a non-verification requirement"
        );
    }

    #[test]
    fn rewrite_removes_blocked_command() {
        let desc = "Add `from __future__ import annotations` to boltzmann.py.\n\
                     Then replace `typing.List` with `list`.\n\
                     Run: ruff check cflibs/ --select UP006,UP007";

        let result = rewrite_remove_blocked_command(desc, "ruff");
        assert!(
            !result.to_lowercase().contains("ruff"),
            "Should remove ruff reference: {result}"
        );
        assert!(
            result.contains("annotations"),
            "Should preserve the actual task"
        );
        assert!(result.contains("verifier"), "Should mention verifier");
    }

    #[test]
    fn store_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReformulationStore::new(dir.path());

        let contract =
            IntentContract::from_issue("test-001", "Fix the bug", Some("- Fix the null pointer"));
        store.save_contract(&contract);

        let loaded = store.load_contract("test-001");
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().issue_id, "test-001");

        // Second save should be no-op
        store.save_contract(&contract);
        let all: Vec<IntentContract> = load_jsonl(&store.contracts_path);
        assert_eq!(all.len(), 1, "Should not duplicate contract");
    }

    #[test]
    fn reformulation_record_tracking() {
        let dir = tempfile::tempdir().unwrap();
        let store = ReformulationStore::new(dir.path());

        let record = ReformulationRecord {
            issue_id: "test-001".into(),
            attempt_index: 0,
            classification: FailureClassification::ToolConstraintMismatch {
                blocked_command: "ruff".into(),
            },
            action: RecoveryAction::RewriteAndRetryNow,
            prior_description_digest: "aabb".into(),
            new_description_digest: Some("ccdd".into()),
            failure_fingerprint: "tool_constraint:ruff".into(),
            intent_guard_passed: true,
            timestamp: Utc::now(),
        };
        store.record_reformulation(&record);

        assert_eq!(
            store.count_same_fingerprint("test-001", "tool_constraint:ruff"),
            1
        );
        assert_eq!(store.next_attempt_index("test-001"), 1);
    }
}
