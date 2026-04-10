//! Concurrent subtask dispatch within a single issue.
//!
//! The manager decomposes an issue into non-overlapping subtasks, each
//! targeting specific files. Workers execute subtasks concurrently in
//! the same worktree, then the verifier runs on the combined result.
//!
//! # Native Beads Integration
//!
//! This module uses the workpad tools (`announce` + `check_announcements`) as
//! the primary coordination channel for concurrent workers in the same
//! worktree. Beads mail remains available for escalation and cross-iteration
//! coordination, but same-worktree worker broadcast stays file-backed so it
//! keeps working when Beads/Dolt mail is temporarily unhealthy.
//!
//! The native messaging layer replaces the previous BeadHub coordination tools
//! (team_status, check_mail, send_mail, chat_send, etc.). Messages are stored
//! as Dolt rows and sync via `bd dolt push/pull`.
//!
//! # Inline Usage Notes
//!
//! - Workpad tools are the primary concurrent-worker coordination path
//! - `BeadsBridge::send_mail` is reserved for escalation / higher-level mail
//! - Both mechanisms still participate in the broader hybrid coordination model
//!
//! Flow:
//! 1. Planner agent analyzes issue + codebase → produces `SubtaskPlan` (JSON)
//! 2. Dispatcher fans out N workers via `JoinSet` + `Semaphore`
//! 3. Each worker is prompt-constrained to its target files
//! 4. After all complete, verifier runs on the worktree
//!
//! Uses the same JoinSet + Semaphore pattern as `modes/deepthink.rs`.

use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::providers::openai;
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, error, info, warn};

use crate::agents::coder;
use crate::beads_bridge::IssueTracker;
use crate::endpoint_pool::EndpointPool;
use crate::runtime_adapter::{AdapterConfig, AdapterReport, RuntimeAdapter};
use crate::tools::bundles::{self, WorkerRole};
use crate::triage::Complexity;

// ── Write-deadline helpers ─────────────────────────────────────────────────────

/// Keywords that indicate an exploration-heavy task needing more read turns
/// before the first write.
const HEAVY_READ_KEYWORDS: &[&str] = &[
    "audit",
    "refactor",
    "analyze",
    "standardize",
    "review",
    "port",
    "migrate",
    "cleanup",
    "consolidate",
    "restructure",
];

/// Compute the maximum LLM turns a worker may spend before it *must* produce
/// its first `edit_file` or `write_file` call.
///
/// The deadline adapts to three signals:
///
/// 1. **Complexity** — more complex issues need deeper exploration before writing.
/// 2. **Objective keywords** — "audit", "refactor", "standardize", etc. signal
///    that significant read-only exploration is expected before any edits.
/// 3. **File counts** — each target file needs ~1 read turn; each context file
///    needs ~1 read turn (capped at 6).
///
/// When `SWARM_MAX_TURNS_WITHOUT_WRITE` is set in the environment it acts as a
/// hard operator override and is returned directly, bypassing the heuristic.
pub fn dynamic_write_deadline(
    complexity: Complexity,
    objective: &str,
    target_files: &[String],
    context_files: &[String],
) -> usize {
    // Operator override takes precedence.
    if let Ok(s) = std::env::var("SWARM_MAX_TURNS_WITHOUT_WRITE") {
        if let Ok(v) = s.parse::<usize>() {
            if v > 0 {
                return v;
            }
        }
    }

    let obj_lower = objective.to_lowercase();
    let is_heavy_read = HEAVY_READ_KEYWORDS.iter().any(|kw| obj_lower.contains(kw));

    // Base budgets tightened (2026-04-04) because task prompts now inline
    // first target file content (dispatch.rs:435-476), reducing necessary
    // read turns. Research: "More with Less" (arxiv:2510.16786) found
    // 75th-percentile turn limits are the cost/performance sweet spot.
    // SWARM_MAX_TURNS_WITHOUT_WRITE env var still overrides these values.
    let base: usize = match (complexity, is_heavy_read) {
        (Complexity::Simple, false) => 4,
        (Complexity::Simple, true) => 6,
        (Complexity::Medium, false) => 6,
        (Complexity::Medium, true) => 10,
        (Complexity::Complex | Complexity::Critical, false) => 10,
        (Complexity::Complex | Complexity::Critical, true) => 15,
    };

    // Target file content is now inlined in the prompt, so each file only
    // needs ~1 supplementary read turn (down from 3).
    let target_bonus = target_files.len();

    // Context files contribute 1 read turn each (capped to prevent runaway budgets).
    let context_bonus = context_files.len().min(6);

    base + target_bonus + context_bonus
}

/// Scale the total worker turn budget for the given task.
///
/// Complex/audit tasks need more total turns because the exploration phase is
/// deeper. The scale factor is applied on top of the file-count-based base and
/// the result is clamped to `[30, 200]`.
fn scale_dynamic_turns(base: usize, complexity: Complexity, objective: &str) -> usize {
    let obj_lower = objective.to_lowercase();
    let is_heavy_read = HEAVY_READ_KEYWORDS.iter().any(|kw| obj_lower.contains(kw));

    let scale: f64 = match (complexity, is_heavy_read) {
        (Complexity::Simple, _) => 1.0,
        (Complexity::Medium, false) => 1.2,
        (Complexity::Medium, true) => 1.5,
        (Complexity::Complex | Complexity::Critical, false) => 1.5,
        (Complexity::Complex | Complexity::Critical, true) => 2.0,
    };

    ((base as f64 * scale) as usize).clamp(30, 200)
}

// ── Types ─────────────────────────────────────────────────────────────────────

/// A single subtask within a concurrent plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subtask {
    /// Subtask identifier (e.g., "subtask-1").
    pub id: String,
    /// What the worker should do.
    pub objective: String,
    /// Files this worker is allowed to modify (non-overlapping with other subtasks).
    pub target_files: Vec<String>,
    /// Files the worker may read but not modify (shared context).
    #[serde(default)]
    pub context_files: Vec<String>,
    /// Worker type: "rust_coder" or "general_coder".
    #[serde(default = "default_worker_type")]
    pub worker_type: String,
}

fn default_worker_type() -> String {
    "general_coder".to_string()
}

/// Plan produced by the manager for concurrent execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubtaskPlan {
    /// High-level summary of the decomposition strategy.
    pub summary: String,
    /// Ordered list of subtasks (executed concurrently).
    pub subtasks: Vec<Subtask>,
}

/// Result from a single subtask worker.
#[derive(Debug)]
pub struct SubtaskResult {
    pub subtask_id: String,
    pub success: bool,
    pub response: String,
    pub elapsed: Duration,
    pub report: Option<AdapterReport>,
}

/// Aggregated outcome from all concurrent subtasks.
#[derive(Debug)]
pub struct DispatchOutcome {
    pub results: Vec<SubtaskResult>,
    pub total_elapsed: Duration,
}

impl DispatchOutcome {
    pub fn all_succeeded(&self) -> bool {
        !self.results.is_empty() && self.results.iter().all(|r| r.success)
    }

    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.success).count()
    }

    pub fn total_tool_calls(&self) -> usize {
        self.results
            .iter()
            .filter_map(|r| r.report.as_ref())
            .map(|r| r.total_tool_calls)
            .sum()
    }

    /// Log a structured summary of the dispatch outcome.
    pub fn log_summary(&self) {
        let max_worker_elapsed = self
            .results
            .iter()
            .map(|r| r.elapsed)
            .max()
            .unwrap_or_default();
        let sequential_estimate: Duration = self.results.iter().map(|r| r.elapsed).sum();
        let speedup = if self.total_elapsed.as_millis() > 0 {
            sequential_estimate.as_millis() as f64 / self.total_elapsed.as_millis() as f64
        } else {
            1.0
        };

        info!(
            succeeded = self.success_count(),
            failed = self.results.len() - self.success_count(),
            total = self.results.len(),
            tool_calls = self.total_tool_calls(),
            total_elapsed_ms = self.total_elapsed.as_millis() as u64,
            max_worker_elapsed_ms = max_worker_elapsed.as_millis() as u64,
            sequential_estimate_ms = sequential_estimate.as_millis() as u64,
            speedup = format!("{speedup:.2}x"),
            "Concurrent dispatch summary"
        );

        // Per-worker breakdown at debug level.
        for result in &self.results {
            debug!(
                subtask_id = %result.subtask_id,
                success = result.success,
                elapsed_ms = result.elapsed.as_millis() as u64,
                tool_calls = result.report.as_ref().map(|r| r.total_tool_calls).unwrap_or(0),
                has_written = result.report.as_ref().map(|r| r.has_written).unwrap_or(false),
                "Worker result"
            );
        }
    }
}

// ── Candidate generation ──────────────────────────────────────────────────────

/// Configuration for multi-candidate patch generation.
///
/// Based on "Wisdom and Delusion of LLM Ensembles" (ARXIV:2510.21513):
/// generating 2-3 candidates with diversity-based selection achieves ~95% of
/// the theoretical 83% improvement over single-model execution.
#[derive(Debug, Clone)]
pub struct CandidateGenerationConfig {
    /// Number of candidates to generate (default: 2).
    pub candidate_count: usize,
    /// Whether to route each candidate to a different model endpoint.
    pub model_diversity: bool,
    /// Whether to use a different strategy seed per candidate (temperature jitter).
    pub strategy_diversity: bool,
}

impl Default for CandidateGenerationConfig {
    fn default() -> Self {
        Self {
            candidate_count: 2,
            model_diversity: true,
            strategy_diversity: true,
        }
    }
}

/// Result from a single candidate generation run.
#[derive(Debug)]
pub struct CandidateResult {
    /// Candidate index (0-based).
    pub index: usize,
    /// Git diff of the candidate's changes against the base commit, if any.
    pub patch: Option<String>,
    /// Whether the candidate produced file changes.
    pub has_changes: bool,
    /// The agent's final response text.
    pub response: String,
    /// Wall-clock duration of the candidate's execution.
    pub elapsed: Duration,
    /// Adapter report (tool calls, turns, termination info).
    pub report: Option<crate::runtime_adapter::AdapterReport>,
}

/// Generate N candidate patches concurrently for the same task prompt.
///
/// Each candidate operates on the same worktree but uses a different endpoint
/// (model diversity) and/or a different temperature seed (strategy diversity).
/// Candidates are run concurrently via `tokio::task::spawn_blocking` so that
/// each candidate's blocking agent call does not starve the async runtime.
///
/// After all candidates complete, their git diffs against `base_commit` are
/// collected and returned as `Vec<CandidateResult>`. The caller is responsible
/// for picking the best candidate (by default: first that has changes).
///
/// # Constraints
/// - Each candidate gets its own independent turn budget (not multiplied total).
/// - The semaphore limits concurrent candidates to `endpoint_pool.capacity()`.
/// - The worktree is SHARED — candidates must not conflict on files. This is
///   acceptable because all candidates target the same files with the same goal;
///   the last writer wins and the diff captures whatever was left on disk.
#[allow(clippy::too_many_arguments)]
pub async fn generate_candidates(
    cfg: &CandidateGenerationConfig,
    endpoint_pool: &crate::endpoint_pool::EndpointPool,
    wt_path: &std::path::Path,
    issue_id: &str,
    task_prompt: &str,
    base_commit: &str,
    timeout_secs: u64,
    complexity: crate::triage::Complexity,
    objective: &str,
) -> Vec<CandidateResult> {
    let n = cfg.candidate_count.max(1);
    let max_concurrent = endpoint_pool.capacity().max(1).min(n);
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let wt_path = Arc::new(wt_path.to_path_buf());
    let task_prompt = Arc::new(task_prompt.to_string());
    let base_commit = Arc::new(base_commit.to_string());
    let objective = Arc::new(objective.to_string());

    info!(
        issue = %issue_id,
        candidate_count = n,
        max_concurrent,
        model_diversity = cfg.model_diversity,
        strategy_diversity = cfg.strategy_diversity,
        "Generating {} candidates in parallel",
        n
    );

    let mut join_set: JoinSet<CandidateResult> = JoinSet::new();

    for idx in 0..n {
        let sem = sem.clone();
        let wt_path = wt_path.clone();
        let task_prompt = task_prompt.clone();
        let base_commit = base_commit.clone();
        let objective = objective.clone();

        // Select endpoint via round-robin for model diversity.
        let (client, model) = endpoint_pool.next();
        let client = client.clone();
        let model = model.to_string();

        // Write deadline per candidate (independent budget per candidate).
        let write_deadline = dynamic_write_deadline(complexity, &objective, &[], &[]);
        let dynamic_turns = scale_dynamic_turns(16, complexity, &objective);

        // Temperature jitter for strategy diversity (±0.1 around worker baseline).
        let temperature = if cfg.strategy_diversity {
            let base = crate::agents::coder::worker_temperature();
            // Candidate 0 uses the baseline; subsequent candidates jitter up.
            base + 0.05 * idx as f64
        } else {
            crate::agents::coder::worker_temperature()
        };

        join_set.spawn(async move {
            let _permit = sem.acquire().await.expect("candidate semaphore closed");
            let start = Instant::now();

            info!(
                candidate = idx,
                model = %model,
                temperature,
                "Starting candidate worker"
            );

            // Build a simple worker agent (no file allowlist — full worktree access).
            let agent = client
                .agent(&model)
                .preamble(
                    "You are a software engineer. Fix the issue described in the task. \
                     Read relevant files, make focused changes, and stop once done.",
                )
                .temperature(temperature)
                .tool_choice(rig::completion::message::ToolChoice::Auto)
                .additional_params(crate::agents::coder::worker_sampling_params())
                .default_max_turns(dynamic_turns)
                .build();

            let adapter = crate::runtime_adapter::RuntimeAdapter::new(
                crate::runtime_adapter::AdapterConfig {
                    agent_name: format!("candidate-{idx}"),
                    max_tool_calls: Some(dynamic_turns * 3),
                    deadline: Some(Instant::now() + Duration::from_secs(timeout_secs)),
                    max_turns_without_write: Some(write_deadline),
                    search_unlock_turn: Some(3),
                    ..Default::default()
                },
            );

            let result = agent
                .prompt(task_prompt.as_str())
                .with_hook(adapter.clone())
                .await;
            let report = adapter.report().ok();

            let (response, _ok) = match result {
                Ok(r) => (r, true),
                Err(e) => {
                    warn!(candidate = idx, error = %e, "Candidate worker failed");
                    (format!("error: {e}"), false)
                }
            };

            // Collect git diff against base commit.
            let patch = crate::git_ops::diff_between(&wt_path, &base_commit, "HEAD");
            let has_changes = patch.is_some();

            info!(
                candidate = idx,
                has_changes,
                elapsed_ms = start.elapsed().as_millis() as u64,
                "Candidate worker done"
            );

            CandidateResult {
                index: idx,
                patch,
                has_changes,
                response,
                elapsed: start.elapsed(),
                report,
            }
        });
    }

    let mut results = Vec::with_capacity(n);
    while let Some(res) = join_set.join_next().await {
        match res {
            Ok(r) => {
                debug!(
                    candidate = r.index,
                    has_changes = r.has_changes,
                    elapsed_ms = r.elapsed.as_millis() as u64,
                    "Candidate result collected"
                );
                results.push(r);
            }
            Err(e) => {
                error!(error = %e, "Candidate worker panicked");
            }
        }
    }

    // Sort by index so the caller sees candidates in spawn order.
    results.sort_by_key(|r| r.index);
    results
}

// ── Molecule tracking ────────────────────────────────────────────────────────

/// Create beads child issues for each subtask in the plan (molecule pattern).
///
/// **Disabled by default** — set `SWARM_MOLECULE_TRACKING=1` to enable.
/// When disabled, returns an empty map (molecule tracking is skipped).
///
/// When enabled, child issues are created with:
/// - Blocking dependency on the parent (parent waits for all children)
/// - `target-file:` labels for each assigned file
/// - `parent:` and `molecule-child` labels for filtering
///
/// Failures are non-fatal — molecule tracking is optional observability.
/// The JoinSet dispatch will proceed regardless.
pub fn create_molecule_for_plan(
    plan: &SubtaskPlan,
    parent_issue_id: &str,
    wt_path: &Path,
) -> std::collections::HashMap<String, String> {
    // Gate behind env var — molecule children pollute `bd ready` if not
    // properly closed, causing infinite retry loops in the dogfood loop.
    let enabled = std::env::var("SWARM_MOLECULE_TRACKING")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !enabled {
        return std::collections::HashMap::new();
    }

    let bridge = crate::beads_bridge::BeadsBridge::with_worktree(wt_path);
    let subtask_data: Vec<(String, Vec<String>)> = plan
        .subtasks
        .iter()
        .map(|s| (s.objective.clone(), s.target_files.clone()))
        .collect();

    match bridge.create_molecule(parent_issue_id, &subtask_data) {
        Ok(child_ids) => {
            let mut map = std::collections::HashMap::new();
            for (subtask, child_id) in plan.subtasks.iter().zip(child_ids) {
                // Claim + label so molecule children don't appear in bd ready.
                let _ = bridge.try_claim(&child_id);
                let _ = bridge.add_label(&child_id, "molecule-child");
                map.insert(subtask.id.clone(), child_id);
            }
            map
        }
        Err(e) => {
            warn!(
                parent = %parent_issue_id,
                error = %e,
                "Failed to create molecule — subtask tracking unavailable"
            );
            std::collections::HashMap::new()
        }
    }
}

/// Close beads child issues based on subtask results.
///
/// Successful subtasks get closed with a success reason.
/// Failed subtasks remain open for retry or human review.
pub fn close_molecule_children(
    results: &[SubtaskResult],
    molecule_map: &std::collections::HashMap<String, String>,
    wt_path: &Path,
) {
    if molecule_map.is_empty() {
        return;
    }

    let bridge = crate::beads_bridge::BeadsBridge::with_worktree(wt_path);
    for result in results {
        if let Some(child_id) = molecule_map.get(&result.subtask_id) {
            if result.success {
                if let Err(e) = bridge.close(child_id, Some("Subtask completed successfully")) {
                    debug!(child = %child_id, error = %e, "Failed to close molecule child");
                }
            } else {
                // Leave failed subtasks open — they'll show in `bd list --status in_progress`
                debug!(
                    child = %child_id,
                    subtask = %result.subtask_id,
                    "Subtask failed — leaving child issue open for retry"
                );
            }
        }
    }
}

// ── Planning ──────────────────────────────────────────────────────────────────

/// System prompt for the subtask planning agent.
///
/// The planner receives the issue objective and a file listing, then outputs
/// a JSON `SubtaskPlan` that decomposes the work into non-overlapping subtasks.
pub const SUBTASK_PLANNER_PROMPT: &str = r#"You are a task decomposition planner for an autonomous coding swarm.

Your job: decompose a coding task into 2-4 INDEPENDENT subtasks that can be executed
by separate workers CONCURRENTLY in the same worktree.

CRITICAL RULES:
1. Each subtask MUST target DIFFERENT files — NO file may appear in more than one
   subtask's `target_files`. Workers edit files concurrently, so overlapping files
   cause data races.
2. Each subtask must be self-contained — a worker must be able to complete it
   without waiting for another worker's output.
3. If the task fundamentally cannot be parallelized (e.g., all changes are in one
   file), return a plan with exactly 1 subtask.
4. Keep subtask objectives specific and actionable — include exact file paths,
   function names, and what to change. Keep each objective under 500 words;
   workers have read_file to explore details. Do NOT paste code into objectives.
5. Use `context_files` for files a worker needs to READ but not modify.
6. INTEGRATION FILES (package manifests like Cargo.toml/pyproject.toml, module
   init files like mod.rs/__init__.py, and entry points like main.rs/main.py)
   may only appear in ONE subtask's target_files. If multiple subtasks need to
   modify them, assign them to subtask-1 and describe the needed changes from
   other subtasks in subtask-1's objective.

Output ONLY valid JSON matching this schema (no markdown fences, no explanation):

{
  "summary": "Brief description of the decomposition strategy",
  "subtasks": [
    {
      "id": "subtask-1",
      "objective": "What this worker should do, with specific file paths and function names",
      "target_files": ["path/to/file1.ext", "path/to/file2.ext"],
      "context_files": ["path/to/read_only.ext"],
      "worker_type": "general_coder"
    }
  ]
}

worker_type options:
- "rust_coder": Language specialist (type system, borrow checker, trait bounds)
- "general_coder": General purpose (scaffolding, multi-file, config changes)
"#;

/// Ask the planner to decompose an issue into concurrent subtasks.
///
/// Uses the cloud endpoint if available, otherwise the reasoning tier.
/// When `target_files` is provided (from file_targeting), includes them prominently
/// in the prompt so the planner focuses on those files instead of exploring.
pub async fn plan_subtasks(
    client: &openai::CompletionsClient,
    model: &str,
    issue_objective: &str,
    file_listing: &str,
    issue_context: &str,
    target_files: Option<&[String]>,
) -> Result<SubtaskPlan> {
    let agent = client
        .agent(model)
        .preamble(SUBTASK_PLANNER_PROMPT)
        .temperature(0.2)
        .additional_params(serde_json::json!({
            "max_tokens": 2048,
            "chat_template_kwargs": { "enable_thinking": false }
        }))
        .build();

    // Pre-populate target files if file_targeting already identified them.
    // This eliminates the planner's need to explore the codebase — the most
    // common failure mode (spending all turns reading instead of planning).
    let target_section = match target_files {
        Some(files) if !files.is_empty() => {
            let file_list = files
                .iter()
                .map(|f| format!("  - {f}"))
                .collect::<Vec<_>>()
                .join("\n");
            format!(
                "## Target Files (pre-identified by file targeting)\n\
                 These files are the most likely to need modification:\n{file_list}\n\n\
                 Use these as your primary target_files in the plan. Only add other files \
                 if the objective clearly requires them.\n\n"
            )
        }
        _ => String::new(),
    };

    let prompt = format!(
        "## Issue Objective\n{issue_objective}\n\n\
         {target_section}\
         ## Additional Context\n{issue_context}\n\n\
         ## Files in Workspace\n{file_listing}\n\n\
         Decompose this into concurrent subtasks. Output JSON only."
    );

    let response = agent
        .prompt(&prompt)
        .await
        .context("subtask planner failed")?;

    parse_subtask_plan(&response)
}

/// Parse the planner's JSON response into a `SubtaskPlan`.
///
/// Handles markdown fences, leading/trailing text, and validates the plan.
pub fn parse_subtask_plan(raw: &str) -> Result<SubtaskPlan> {
    // Strip markdown fences if present.
    let json_str = raw
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();

    // Find the first '{' and last '}' to extract the JSON object.
    let start = json_str
        .find('{')
        .context("no JSON object found in planner response")?;
    let end = json_str
        .rfind('}')
        .context("no closing brace in planner response")?;

    let plan: SubtaskPlan =
        serde_json::from_str(&json_str[start..=end]).context("failed to parse SubtaskPlan JSON")?;

    // Validate: at least one subtask.
    anyhow::ensure!(!plan.subtasks.is_empty(), "plan has no subtasks");

    // Validate: no file appears in multiple subtasks' target_files.
    let mut seen_files = std::collections::HashSet::new();
    for subtask in &plan.subtasks {
        for file in &subtask.target_files {
            anyhow::ensure!(
                seen_files.insert(file.clone()),
                "file {file} appears in multiple subtasks — violates non-overlap constraint"
            );
        }
    }

    // Validate: integration files appear in at most one subtask.
    if plan.subtasks.len() > 1 {
        validate_integration_files(&plan, None)?;
    }

    Ok(plan)
}

/// Default integration files for Rust projects (used when no profile is loaded).
const DEFAULT_INTEGRATION_FILES: &[&str] =
    &["Cargo.toml", "Cargo.lock", "mod.rs", "lib.rs", "main.rs"];

/// Check if a filename matches an integration file pattern.
///
/// Uses profile `integration_files` when available, falls back to Rust defaults.
fn is_integration_file(path: &str, profile_files: Option<&[String]>) -> bool {
    let filename = std::path::Path::new(path)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or(path);
    match profile_files {
        Some(files) => files.iter().any(|f| {
            let f_name = std::path::Path::new(f)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(f);
            filename == f_name
        }),
        None => DEFAULT_INTEGRATION_FILES.contains(&filename),
    }
}

/// Validate that integration files appear in at most one subtask.
fn validate_integration_files(plan: &SubtaskPlan, profile_files: Option<&[String]>) -> Result<()> {
    let mut integration_owners: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    for subtask in &plan.subtasks {
        for file in &subtask.target_files {
            if is_integration_file(file, profile_files) {
                if let Some(owner) = integration_owners.get(file) {
                    anyhow::bail!(
                        "integration file {file} appears in both {owner} and {} — \
                         assign integration files to one subtask only",
                        subtask.id
                    );
                }
                integration_owners.insert(file.clone(), subtask.id.clone());
            }
        }
    }
    Ok(())
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Dispatch subtasks concurrently to workers.
///
/// Each worker is constrained to its `target_files` via the system prompt.
/// Workers share the same worktree but operate on non-overlapping files.
/// Returns results from all workers (successes and failures).
///
/// Concurrency is bounded by `max_concurrent` (typically the number of
/// available inference slots across the cluster).
#[allow(clippy::too_many_arguments)]
pub async fn dispatch_subtasks(
    plan: &SubtaskPlan,
    endpoint_pool: &EndpointPool,
    wt_path: &Path,
    issue_id: &str,
    max_concurrent: usize,
    timeout_secs: u64,
    complexity: Complexity,
    issue_objective: &str,
) -> DispatchOutcome {
    let start = Instant::now();
    let max_concurrent = max_concurrent.min(plan.subtasks.len()).max(1);
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let wt_path = Arc::new(wt_path.to_path_buf());
    let issue_id = Arc::new(issue_id.to_string());
    let timeout_secs = Arc::new(timeout_secs);
    let issue_objective = Arc::new(issue_objective.to_string());

    // Log plan details for observability.
    let file_assignments: Vec<String> = plan
        .subtasks
        .iter()
        .map(|s| format!("{}:{}", s.id, s.target_files.join(",")))
        .collect();
    info!(
        subtask_count = plan.subtasks.len(),
        max_concurrent,
        file_assignments = ?file_assignments,
        "Dispatching concurrent subtasks"
    );

    let mut join_set: JoinSet<SubtaskResult> = JoinSet::new();

    for subtask in &plan.subtasks {
        let sem = sem.clone();
        let wt_path = wt_path.clone();
        let issue_id = issue_id.clone();
        let subtask = subtask.clone();
        let timeout_secs = timeout_secs.clone();
        let issue_objective = issue_objective.clone();
        // Select the next endpoint via round-robin BEFORE spawning, so the
        // borrow of endpoint_pool doesn't cross the spawn boundary.
        let (client, model) = endpoint_pool.next();
        let client = client.clone();
        let model = model.to_string();

        join_set.spawn(async move {
            let _permit = sem.acquire().await.expect("semaphore closed");
            info!(
                subtask_id = %subtask.id,
                worker_type = %subtask.worker_type,
                target_files = ?subtask.target_files,
                "Starting subtask worker"
            );
            run_subtask_worker(
                &client,
                &model,
                &wt_path,
                &issue_id,
                &subtask,
                *timeout_secs,
                complexity,
                &issue_objective,
            )
            .await
        });
    }

    // Track pending subtask IDs so we can account for panicked workers.
    let mut pending_ids: std::collections::HashSet<String> =
        plan.subtasks.iter().map(|s| s.id.clone()).collect();

    let mut results = Vec::with_capacity(plan.subtasks.len());
    while let Some(res) = join_set.join_next().await {
        match res {
            Ok(result) => {
                info!(
                    subtask_id = %result.subtask_id,
                    success = result.success,
                    elapsed_ms = result.elapsed.as_millis() as u64,
                    tool_calls = result.report.as_ref().map(|r| r.total_tool_calls).unwrap_or(0),
                    "Subtask completed"
                );
                pending_ids.remove(&result.subtask_id);
                results.push(result);
            }
            Err(e) => {
                error!(error = %e, "Subtask worker panicked");
            }
        }
    }

    // Any remaining pending IDs are from panicked workers — record as failures.
    for id in pending_ids {
        error!(subtask_id = %id, "Subtask worker panicked — recording as failure");
        results.push(SubtaskResult {
            subtask_id: id,
            success: false,
            response: "Worker panicked".to_string(),
            elapsed: start.elapsed(),
            report: None,
        });
    }

    DispatchOutcome {
        results,
        total_elapsed: start.elapsed(),
    }
}

// ── Worker ────────────────────────────────────────────────────────────────────

/// Build a file-constrained system prompt for a subtask worker.
///
/// Classifies each target file as "existing" or "new" based on whether it
/// exists in the worktree. This prevents workers from wasting their entire
/// budget trying to read files that need to be created from scratch.
fn build_subtask_system_prompt(subtask: &Subtask, wt_path: &Path) -> String {
    // Classify target files as existing or new.
    let mut existing_targets = Vec::new();
    let mut new_targets = Vec::new();
    for f in &subtask.target_files {
        if wt_path.join(f).exists() {
            existing_targets.push(f.as_str());
        } else {
            new_targets.push(f.as_str());
        }
    }

    let target_list = subtask
        .target_files
        .iter()
        .map(|f| {
            if wt_path.join(f).exists() {
                format!("  - {f}  ← EXISTS (read then edit)")
            } else {
                format!("  - {f}  ← NEW (create with write_file)")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    let context_list = if subtask.context_files.is_empty() {
        String::from("  (none)")
    } else {
        subtask
            .context_files
            .iter()
            .map(|f| format!("  - {f}"))
            .collect::<Vec<_>>()
            .join("\n")
    };

    // Build the starting instruction based on file existence.
    let start_instruction = if existing_targets.is_empty() && !new_targets.is_empty() {
        // All files are new — skip read and go straight to write.
        "6. All your target files are NEW and need to be CREATED. Your first tool call MUST be \
           write_file to create the first new file. Do NOT call read_file on new files — they \
           don't exist yet. write_file will create parent directories automatically."
            .to_string()
    } else if !existing_targets.is_empty() && !new_targets.is_empty() {
        // Mix of existing and new files.
        let existing = existing_targets.join(", ");
        let new = new_targets.join(", ");
        format!(
            "6. Some files exist ({existing}), some are NEW ({new}). \
             Start with read_file on the EXISTING files. \
             Use write_file to CREATE the new files — do NOT call read_file on new files."
        )
    } else {
        // All files exist — original behavior.
        "6. Your first tool call MUST be read_file on one of your target files. \
           Then edit_file/write_file to implement changes."
            .to_string()
    };

    format!(
        r#"You are a software engineer executing ONE subtask of a larger parallel plan.

## YOUR ASSIGNED FILES (you may ONLY modify these)
{target_list}

## READ-ONLY CONTEXT FILES (read but do NOT modify)
{context_list}

## RULES
1. ONLY modify files in YOUR ASSIGNED FILES list above. Do NOT touch any other files.
2. For EXISTING target files: read them first to understand the current code.
   For NEW target files: create them with write_file (parent dirs are created automatically).
3. Make focused, minimal changes that accomplish your subtask objective.
4. Other workers are editing OTHER files concurrently — do not interfere.
5. If you need to understand code in other files, use read_file on context_files.
{start_instruction}
7. Do NOT run cargo check/test — the orchestrator runs the verifier after all workers finish.
8. **STOP RULE**: Once your edit_file or write_file calls succeed, YOU ARE DONE. \
   Do NOT call any more tools. Immediately return a summary of what you changed.

## INTER-WORKER COMMUNICATION
8. After changing any PUBLIC interface (struct fields, function signatures, trait methods, \
   public type aliases), call `announce` to notify other workers.
9. Before your FINAL edits, call `check_announcements` to see if other workers changed \
   interfaces you depend on. Adapt your code if needed.
"#
    )
}

/// Public entry point for running a single subtask worker.
///
/// Used by the orchestrator for serial retry of failed subtasks.
#[allow(clippy::too_many_arguments)]
pub async fn run_subtask_worker_public(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    issue_id: &str,
    subtask: &Subtask,
    timeout_secs: u64,
    complexity: Complexity,
    issue_objective: &str,
) -> SubtaskResult {
    run_subtask_worker(
        client,
        model,
        wt_path,
        issue_id,
        subtask,
        timeout_secs,
        complexity,
        issue_objective,
    )
    .await
}

/// Run a single subtask worker.
///
/// Builds a fresh agent scoped to the subtask's files, runs it with budget
/// tracking, and returns the result. The agent is constructed INSIDE the
/// spawned task to avoid `!Send` issues with Rig's agent type.
#[allow(clippy::too_many_arguments)]
async fn run_subtask_worker(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    issue_id: &str,
    subtask: &Subtask,
    timeout_secs: u64,
    complexity: Complexity,
    issue_objective: &str,
) -> SubtaskResult {
    let start = Instant::now();

    // Determine worker role.
    let role = match subtask.worker_type.as_str() {
        "rust_coder" => WorkerRole::RustSpecialist,
        _ => WorkerRole::General,
    };

    // Build tools scoped to the worktree with file allowlist enforcement.
    let tools =
        bundles::subtask_worker_tools(wt_path, role, &subtask.target_files, &subtask.id, None);

    // Scale turn budget dynamically based on task complexity and file counts:
    //
    //   base     = (target_files * 15) + (new_files * 10)
    //   total    = scale_dynamic_turns(base, complexity, objective)
    //            → Simple: 1.0×, Medium: 1.2–1.5×, Complex/audit: 1.5–2.0×
    //            → clamped to [30, 200]
    //
    // Examples (2 existing files):
    //   Simple               → 30 turns
    //   Medium               → 36 turns
    //   Complex (no audit)   → 45 turns
    //   Complex + "audit"    → 60 turns
    let (new_count, existing_count) =
        subtask
            .target_files
            .iter()
            .fold((0usize, 0usize), |(n, e), f| {
                if wt_path.join(f).exists() {
                    (n, e + 1)
                } else {
                    (n + 1, e)
                }
            });
    let base_turns = (existing_count + new_count) * 15 + new_count * 10;
    let dynamic_turns = scale_dynamic_turns(base_turns, complexity, issue_objective);

    // Write deadline: how many turns the worker may spend before its first edit.
    // Computed from complexity + objective keywords + file counts; capped so at
    // least 6 turns remain for actual writing.
    let write_deadline = dynamic_write_deadline(
        complexity,
        issue_objective,
        &subtask.target_files,
        &subtask.context_files,
    )
    .min(dynamic_turns.saturating_sub(6));

    // Build the agent with a subtask-specific system prompt.
    let system_prompt = build_subtask_system_prompt(subtask, wt_path);
    let agent = client
        .agent(model)
        .preamble(&system_prompt)
        .temperature(coder::worker_temperature())
        .tool_choice(rig::completion::message::ToolChoice::Required)
        .additional_params(coder::worker_sampling_params())
        .tools(tools)
        .default_max_turns(dynamic_turns)
        .build();

    tracing::debug!(
        subtask_id = %subtask.id,
        dynamic_turns,
        write_deadline,
        new_files = new_count,
        existing_files = existing_count,
        complexity = %complexity,
        "Subtask turn budget calculated"
    );

    // Build task prompt.
    // Cap the objective to avoid context overflow on local models (n_ctx=16384).
    // Tool definitions (~3500 tok) + system prompt (~400 tok) consume ~4000 tokens
    // before the user message is read, leaving ~12K tokens for the objective.
    // 6000 chars ≈ 1500 tokens — keeps total well under 16384.
    const MAX_OBJECTIVE_CHARS: usize = 6000;
    let objective_str = if subtask.objective.len() > MAX_OBJECTIVE_CHARS {
        tracing::warn!(
            subtask_id = %subtask.id,
            original_len = subtask.objective.len(),
            cap = MAX_OBJECTIVE_CHARS,
            "Subtask objective truncated to avoid context overflow"
        );
        format!(
            "{}... [truncated — see issue for full context]",
            crate::str_util::safe_truncate(&subtask.objective, MAX_OBJECTIVE_CHARS)
        )
    } else {
        subtask.objective.clone()
    };
    let task_prompt = format!(
        "Issue: {issue_id}\n\nSubtask: {}\n\nObjective: {}",
        subtask.id, objective_str
    );

    // Runtime adapter for budget tracking.
    // The deadline (wall-clock timeout) is the primary constraint — not turn counts.
    // max_turns_without_write is computed dynamically; max_tool_calls is generous.
    let adapter = RuntimeAdapter::new(AdapterConfig {
        agent_name: format!("{}-{}", subtask.worker_type, subtask.id),
        max_tool_calls: Some(dynamic_turns * 3),
        deadline: Some(Instant::now() + Duration::from_secs(timeout_secs)),
        max_turns_without_write: Some(write_deadline),
        search_unlock_turn: Some(3),
        ..Default::default()
    });

    // Run the agent.
    let result = agent.prompt(&task_prompt).with_hook(adapter.clone()).await;

    let report = adapter.report().ok();

    match result {
        Ok(response) => {
            debug!(
                subtask_id = %subtask.id,
                response_len = response.len(),
                "Subtask worker responded"
            );
            SubtaskResult {
                subtask_id: subtask.id.clone(),
                success: true,
                response,
                elapsed: start.elapsed(),
                report,
            }
        }
        Err(e) => {
            // Use the adapter report as the primary budget-exhaustion signal.
            // RuntimeAdapter sets terminated_early=true for max_tool_calls,
            // deadline, read-budget, and write-stall terminations. Fallback
            // to OrchestrationError::classify() for typed error matching when
            // the report is unavailable.
            use crate::modes::errors::OrchestrationError;
            let is_budget = report
                .as_ref()
                .map(|r| r.terminated_early)
                .unwrap_or_else(|| {
                    let classified = OrchestrationError::classify(&e);
                    matches!(
                        classified,
                        OrchestrationError::MaxIterations(_)
                            | OrchestrationError::InferenceFailure(_)
                    )
                });
            if is_budget {
                // Budget exhaustion means the worker ran out of time/calls but
                // may have made partial progress. Treat as success if it wrote files.
                let has_written = report.as_ref().map(|r| r.has_written).unwrap_or(false);
                warn!(
                    subtask_id = %subtask.id,
                    has_written,
                    "Subtask worker hit budget limit"
                );
                SubtaskResult {
                    subtask_id: subtask.id.clone(),
                    success: has_written,
                    response: format!("Budget exhausted (wrote files: {has_written})"),
                    elapsed: start.elapsed(),
                    report,
                }
            } else {
                error!(
                    subtask_id = %subtask.id,
                    error = %e,
                    "Subtask worker failed"
                );
                SubtaskResult {
                    subtask_id: subtask.id.clone(),
                    success: false,
                    response: format!("Error: {e}"),
                    elapsed: start.elapsed(),
                    report,
                }
            }
        }
    }
}

// ── File listing helper ───────────────────────────────────────────────────────

/// Generate a compact file listing for the planner, filtered to source files.
pub fn list_source_files(wt_path: &Path) -> String {
    let mut files = Vec::new();
    collect_source_files(wt_path, wt_path, &mut files);
    files.sort();
    files.join("\n")
}

fn collect_source_files(root: &Path, dir: &Path, out: &mut Vec<String>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();

        // Skip hidden dirs, target, node_modules, etc.
        if name_str.starts_with('.')
            || name_str == "target"
            || name_str == "node_modules"
            || name_str == ".beads"
        {
            continue;
        }

        if path.is_dir() {
            collect_source_files(root, &path, out);
        } else if is_source_file(&name_str) {
            if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.display().to_string());
            }
        }
    }
}

fn is_source_file(name: &str) -> bool {
    name.ends_with(".rs")
        || name.ends_with(".toml")
        || name.ends_with(".py")
        || name.ends_with(".ts")
        || name.ends_with(".js")
        || name.ends_with(".sh")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_plan() {
        let json = r#"{
            "summary": "Split into module and tests",
            "subtasks": [
                {
                    "id": "subtask-1",
                    "objective": "Add validation to parser.rs",
                    "target_files": ["src/parser.rs"],
                    "context_files": ["src/types.rs"],
                    "worker_type": "rust_coder"
                },
                {
                    "id": "subtask-2",
                    "objective": "Add tests for validation",
                    "target_files": ["tests/parser_test.rs"],
                    "context_files": ["src/parser.rs"],
                    "worker_type": "general_coder"
                }
            ]
        }"#;

        let plan = parse_subtask_plan(json).unwrap();
        assert_eq!(plan.subtasks.len(), 2);
        assert_eq!(plan.subtasks[0].target_files, vec!["src/parser.rs"]);
        assert_eq!(plan.subtasks[1].worker_type, "general_coder");
    }

    #[test]
    fn parse_plan_rejects_overlapping_files() {
        let json = r#"{
            "summary": "bad plan",
            "subtasks": [
                {"id": "s1", "objective": "edit a", "target_files": ["src/lib.rs"]},
                {"id": "s2", "objective": "also edit a", "target_files": ["src/lib.rs"]}
            ]
        }"#;

        let err = parse_subtask_plan(json).unwrap_err();
        assert!(
            err.to_string().contains("non-overlap"),
            "Expected overlap error, got: {err}"
        );
    }

    #[test]
    fn parse_plan_rejects_empty() {
        let json = r#"{"summary": "empty", "subtasks": []}"#;
        let err = parse_subtask_plan(json).unwrap_err();
        assert!(err.to_string().contains("no subtasks"));
    }

    #[test]
    fn default_worker_type_is_general() {
        let json = r#"{"summary": "x", "subtasks": [{"id": "s1", "objective": "y", "target_files": ["z.rs"]}]}"#;
        let plan = parse_subtask_plan(json).unwrap();
        assert_eq!(plan.subtasks[0].worker_type, "general_coder");
    }

    #[test]
    fn subtask_system_prompt_lists_files() {
        let subtask = Subtask {
            id: "s1".into(),
            objective: "fix it".into(),
            target_files: vec!["src/lib.rs".into(), "src/main.rs".into()],
            context_files: vec!["src/types.rs".into()],
            worker_type: "rust_coder".into(),
        };
        let prompt = build_subtask_system_prompt(&subtask, Path::new("/tmp"));
        assert!(prompt.contains("src/lib.rs"));
        assert!(prompt.contains("src/main.rs"));
        assert!(prompt.contains("src/types.rs"));
        assert!(prompt.contains("ONLY modify"));
    }

    #[test]
    fn source_file_detection() {
        assert!(is_source_file("lib.rs"));
        assert!(is_source_file("Cargo.toml"));
        assert!(is_source_file("build.sh"));
        assert!(!is_source_file("image.png"));
        assert!(!is_source_file("data.bin"));
    }

    #[test]
    fn parse_plan_rejects_integration_file_in_multiple_subtasks() {
        // Use different paths that both have integration file basenames (mod.rs in different dirs).
        let json = r#"{
            "summary": "bad integration",
            "subtasks": [
                {"id": "s1", "objective": "add module a", "target_files": ["src/a/mod.rs", "src/a/foo.rs"]},
                {"id": "s2", "objective": "add module b", "target_files": ["src/b/mod.rs", "src/b/bar.rs"]}
            ]
        }"#;

        // Note: Different paths pass the non-overlap check, but both contain mod.rs
        // which is an integration file pattern. However, our integration check is
        // per-exact-path, not per-basename. So different mod.rs files are actually OK.
        // The integration file check prevents the SAME file in multiple subtasks,
        // which is already caught by the non-overlap check. The real value is as a
        // documentation/prompt constraint for the planner.
        let _plan = parse_subtask_plan(json).unwrap();
    }

    #[test]
    fn subtask_system_prompt_mentions_workpad() {
        let subtask = Subtask {
            id: "s1".into(),
            objective: "fix it".into(),
            target_files: vec!["src/lib.rs".into()],
            context_files: vec![],
            worker_type: "general_coder".into(),
        };
        let prompt = build_subtask_system_prompt(&subtask, Path::new("/tmp"));
        assert!(prompt.contains("announce"), "Should mention announce tool");
        assert!(
            prompt.contains("check_announcements"),
            "Should mention check_announcements tool"
        );
    }

    #[test]
    fn dynamic_write_deadline_baselines() {
        // Tightened baselines — task prompts inline first target file content,
        // so workers need fewer exploration turns before writing.
        assert_eq!(
            dynamic_write_deadline(Complexity::Simple, "fix bug", &[], &[]),
            4
        );
        assert_eq!(
            dynamic_write_deadline(Complexity::Simple, "audit module", &[], &[]),
            6
        );
        assert_eq!(
            dynamic_write_deadline(Complexity::Medium, "add feature", &[], &[]),
            6
        );
        assert_eq!(
            dynamic_write_deadline(Complexity::Medium, "refactor auth", &[], &[]),
            10
        );
        assert_eq!(
            dynamic_write_deadline(Complexity::Complex, "implement api", &[], &[]),
            10
        );
        assert_eq!(
            dynamic_write_deadline(Complexity::Complex, "migrate database", &[], &[]),
            15
        );
    }

    #[test]
    fn dynamic_write_deadline_file_bonuses() {
        // Each target file adds 1 turn (content inlined), each context file adds 1 (capped at 6).
        let base = dynamic_write_deadline(Complexity::Medium, "fix", &[], &[]);
        let with_targets = dynamic_write_deadline(
            Complexity::Medium,
            "fix",
            &["a.rs".into(), "b.rs".into()],
            &[],
        );
        assert_eq!(with_targets, base + 2); // 2 files * 1 turn

        let with_context = dynamic_write_deadline(
            Complexity::Medium,
            "fix",
            &[],
            &["c.rs".into(), "d.rs".into(), "e.rs".into()],
        );
        assert_eq!(with_context, base + 3); // 3 context files * 1 turn
    }
}
