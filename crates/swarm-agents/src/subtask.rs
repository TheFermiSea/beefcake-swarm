//! Concurrent subtask dispatch within a single issue.
//!
//! The manager decomposes an issue into non-overlapping subtasks, each
//! targeting specific files. Workers execute subtasks concurrently in
//! the same worktree, then the verifier runs on the combined result.
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
use crate::endpoint_pool::EndpointPool;
use crate::runtime_adapter::{AdapterConfig, AdapterReport, RuntimeAdapter};
use crate::tools::bundles::{self, WorkerRole};

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
}

// ── Planning ──────────────────────────────────────────────────────────────────

/// System prompt for the subtask planning agent.
///
/// The planner receives the issue objective and a file listing, then outputs
/// a JSON `SubtaskPlan` that decomposes the work into non-overlapping subtasks.
pub const SUBTASK_PLANNER_PROMPT: &str = r#"You are a task decomposition planner for a Rust coding swarm.

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
   function names, and what to change.
5. Use `context_files` for files a worker needs to READ but not modify.

Output ONLY valid JSON matching this schema (no markdown fences, no explanation):

{
  "summary": "Brief description of the decomposition strategy",
  "subtasks": [
    {
      "id": "subtask-1",
      "objective": "What this worker should do, with specific file paths and function names",
      "target_files": ["path/to/file1.rs", "path/to/file2.rs"],
      "context_files": ["path/to/read_only.rs"],
      "worker_type": "general_coder"
    }
  ]
}

worker_type options:
- "rust_coder": Rust specialist (borrow checker, lifetimes, trait bounds)
- "general_coder": General purpose (scaffolding, multi-file, config changes)
"#;

/// Ask the planner to decompose an issue into concurrent subtasks.
///
/// Uses the cloud endpoint if available, otherwise the reasoning tier.
pub async fn plan_subtasks(
    client: &openai::CompletionsClient,
    model: &str,
    issue_objective: &str,
    file_listing: &str,
    issue_context: &str,
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

    let prompt = format!(
        "## Issue Objective\n{issue_objective}\n\n\
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

    Ok(plan)
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
pub async fn dispatch_subtasks(
    plan: &SubtaskPlan,
    endpoint_pool: &EndpointPool,
    wt_path: &Path,
    issue_id: &str,
    max_concurrent: usize,
    timeout_secs: u64,
) -> DispatchOutcome {
    let start = Instant::now();
    let max_concurrent = max_concurrent.min(plan.subtasks.len()).max(1);
    let sem = Arc::new(Semaphore::new(max_concurrent));
    let wt_path = Arc::new(wt_path.to_path_buf());
    let issue_id = Arc::new(issue_id.to_string());
    let timeout_secs = Arc::new(timeout_secs);

    info!(
        subtask_count = plan.subtasks.len(),
        max_concurrent, "Dispatching concurrent subtasks"
    );

    let mut join_set: JoinSet<SubtaskResult> = JoinSet::new();

    for subtask in &plan.subtasks {
        let sem = sem.clone();
        let wt_path = wt_path.clone();
        let issue_id = issue_id.clone();
        let subtask = subtask.clone();
        let timeout_secs = timeout_secs.clone();
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
fn build_subtask_system_prompt(subtask: &Subtask) -> String {
    let target_list = subtask
        .target_files
        .iter()
        .map(|f| format!("  - {f}"))
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

    format!(
        r#"You are a Rust engineer executing ONE subtask of a larger parallel plan.

## YOUR ASSIGNED FILES (you may ONLY modify these)
{target_list}

## READ-ONLY CONTEXT FILES (read but do NOT modify)
{context_list}

## RULES
1. ONLY modify files in YOUR ASSIGNED FILES list above. Do NOT touch any other files.
2. Start by reading your target files to understand the current code.
3. Make focused, minimal changes that accomplish your subtask objective.
4. Other workers are editing OTHER files concurrently — do not interfere.
5. If you need to understand code in other files, use read_file on context_files.
6. Your first tool call MUST be read_file. Then edit_file/write_file to implement.
7. Do NOT run cargo check/test — the orchestrator runs the verifier after all workers finish.
"#
    )
}

/// Run a single subtask worker.
///
/// Builds a fresh agent scoped to the subtask's files, runs it with budget
/// tracking, and returns the result. The agent is constructed INSIDE the
/// spawned task to avoid `!Send` issues with Rig's agent type.
async fn run_subtask_worker(
    client: &openai::CompletionsClient,
    model: &str,
    wt_path: &Path,
    issue_id: &str,
    subtask: &Subtask,
    timeout_secs: u64,
) -> SubtaskResult {
    let start = Instant::now();

    // Determine worker role.
    let role = match subtask.worker_type.as_str() {
        "rust_coder" => WorkerRole::RustSpecialist,
        _ => WorkerRole::General,
    };

    // Build tools scoped to the worktree with file allowlist enforcement.
    let tools = bundles::subtask_worker_tools(wt_path, role, &subtask.target_files);

    // Build the agent with a subtask-specific system prompt.
    let system_prompt = build_subtask_system_prompt(subtask);
    let agent = client
        .agent(model)
        .preamble(&system_prompt)
        .temperature(coder::worker_temperature())
        .tool_choice(rig::completion::message::ToolChoice::Required)
        .additional_params(coder::worker_sampling_params())
        .tools(tools)
        .default_max_turns(10) // Shorter than normal workers — subtasks are scoped
        .build();

    // Build task prompt.
    let task_prompt = format!(
        "Issue: {issue_id}\n\nSubtask: {}\n\nObjective: {}",
        subtask.id, subtask.objective
    );

    // Runtime adapter for budget tracking.
    let adapter = RuntimeAdapter::new(AdapterConfig {
        agent_name: format!("{}-{}", subtask.worker_type, subtask.id),
        max_tool_calls: Some(30),
        deadline: Some(Instant::now() + Duration::from_secs(timeout_secs)),
        max_turns_without_write: Some(5),
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
            // to error string matching for cases where the report is unavailable.
            let is_budget = report
                .as_ref()
                .map(|r| r.terminated_early)
                .unwrap_or_else(|| {
                    let msg = e.to_string();
                    msg.contains("Budget exhausted")
                        || msg.contains("deadline")
                        || msg.contains("max tool calls")
                        || msg.contains("exceeded")
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
    fn parse_plan_with_markdown_fences() {
        let json = "```json\n{\"summary\": \"one task\", \"subtasks\": [{\"id\": \"s1\", \"objective\": \"do it\", \"target_files\": [\"a.rs\"]}]}\n```";
        let plan = parse_subtask_plan(json).unwrap();
        assert_eq!(plan.subtasks.len(), 1);
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
        let prompt = build_subtask_system_prompt(&subtask);
        assert!(prompt.contains("src/lib.rs"));
        assert!(prompt.contains("src/main.rs"));
        assert!(prompt.contains("src/types.rs"));
        assert!(prompt.contains("ONLY modify"));
    }

    #[test]
    fn dispatch_outcome_aggregation() {
        let outcome = DispatchOutcome {
            results: vec![
                SubtaskResult {
                    subtask_id: "s1".into(),
                    success: true,
                    response: "done".into(),
                    elapsed: Duration::from_secs(10),
                    report: Some(AdapterReport {
                        agent_name: "test".into(),
                        tool_events: vec![],
                        turn_count: 3,
                        total_tool_calls: 5,
                        total_tool_time_ms: 100,
                        wall_time_ms: 10000,
                        terminated_early: false,
                        termination_reason: None,
                        has_written: true,
                    }),
                },
                SubtaskResult {
                    subtask_id: "s2".into(),
                    success: true,
                    response: "done".into(),
                    elapsed: Duration::from_secs(8),
                    report: Some(AdapterReport {
                        agent_name: "test".into(),
                        tool_events: vec![],
                        turn_count: 2,
                        total_tool_calls: 3,
                        total_tool_time_ms: 50,
                        wall_time_ms: 8000,
                        terminated_early: false,
                        termination_reason: None,
                        has_written: true,
                    }),
                },
            ],
            total_elapsed: Duration::from_secs(12),
        };

        assert!(outcome.all_succeeded());
        assert_eq!(outcome.success_count(), 2);
        assert_eq!(outcome.total_tool_calls(), 8);
    }

    #[test]
    fn source_file_detection() {
        assert!(is_source_file("lib.rs"));
        assert!(is_source_file("Cargo.toml"));
        assert!(is_source_file("build.sh"));
        assert!(!is_source_file("image.png"));
        assert!(!is_source_file("data.bin"));
    }
}
