//! Manager/orchestrator agent (Kernel in the Slate LLM-as-OS pattern).
//!
//! When cloud is available: backed by Opus 4.6 / G3-Pro with local workers as tools.
//! Fallback: backed by Qwen3.5-Architect (local reasoning) with coders as tools.
//!
//! The Manager gets worker agents as tools (agent-as-tool pattern) plus
//! **strategy-only** deterministic tools (verifier, get_diff, list_changed_files).
//! No read/write/edit/list — the kernel orchestrates, workers execute.

use std::path::Path;
use std::sync::Arc;

use rig::client::CompletionClient;
use rig::providers::openai;

use crate::context_firewall::CondensedAgentTool;
use crate::notebook_bridge::KnowledgeBase;
use crate::prompts;
use crate::tools::bundles;
use crate::tools::plan_parallel_tool::{PlanParallelWorkTool, PlanSlot};
use crate::tools::submit_plan_tool::{SubmitPlanTool, WorkPlanSlot};

use super::coder::OaiAgent;

const DEFAULT_MANAGER_MAX_TURNS: usize = 60;

fn manager_max_turns() -> usize {
    std::env::var("SWARM_MANAGER_MAX_TURNS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_MANAGER_MAX_TURNS)
}

/// Bundled workers and tools for building a Manager agent.
///
/// Avoids passing 8+ individual arguments to the builder functions.
pub struct ManagerWorkers {
    pub rust_coder: OaiAgent,
    pub general_coder: OaiAgent,
    pub reviewer: OaiAgent,
    /// Planning specialist — produces structured repair plans (read-only tools).
    pub planner: OaiAgent,
    /// Implementation specialist — follows plans with targeted edits.
    pub fixer: OaiAgent,
    /// Architect specialist — reads codebase, produces ArchitectPlan with exact
    /// SEARCH/REPLACE edit blocks. Cloud model (Opus/Gemini). Read-only.
    pub architect: Option<OaiAgent>,
    /// Editor specialist — applies ArchitectPlan edits mechanically. Local 27B.
    pub editor: Option<OaiAgent>,
    /// Qwen3.5-Architect reasoning worker (cloud manager only).
    pub reasoning_worker: Option<OaiAgent>,
    /// Qwen3.5-397B-A17B strategist advisor (strategist profile only).
    pub strategist: Option<OaiAgent>,
    /// Optional knowledge base for the query_notebook tool.
    pub notebook_bridge: Option<Arc<dyn KnowledgeBase>>,
    /// Shared slot where the manager deposits a parallel work plan.
    /// When the manager calls `plan_parallel_work`, the validated plan
    /// is stored here for the orchestrator to pick up and dispatch.
    pub plan_slot: Option<PlanSlot>,
    /// Shared slot for the plan-before-execute gate (ClawTeam pattern).
    /// When the manager calls `submit_plan`, its approach is captured here
    /// and injected into subsequent iteration prompts.
    pub work_plan_slot: Option<WorkPlanSlot>,
}

/// Build the cloud-backed Manager with reasoning_worker and coders as tools.
///
/// Cloud model (Opus 4.6 / G3-Pro) manages local workers:
/// - reasoning_worker (Qwen3.5-Architect): deep analysis, repair plans
/// - strategist (Qwen3.5-397B-A17B): advisor for architectural review
/// - rust_coder (Qwen3.5-Implementer): fast Rust fixes
/// - general_coder (Qwen3.5-Implementer): multi-file scaffolding
/// - reviewer: blind code review
pub fn build_cloud_manager(
    client: &openai::CompletionsClient,
    model: &str,
    workers: ManagerWorkers,
    wt_path: &Path,
    verifier_packages: &[String],
) -> OaiAgent {
    build_cloud_manager_for_language(client, model, workers, wt_path, verifier_packages, None)
}

/// Build the cloud-backed Manager with language-aware prompts.
pub fn build_cloud_manager_for_language(
    client: &openai::CompletionsClient,
    model: &str,
    workers: ManagerWorkers,
    wt_path: &Path,
    verifier_packages: &[String],
    language: Option<&str>,
) -> OaiAgent {
    let preamble = prompts::build_full_manager_preamble(
        "manager",
        wt_path,
        prompts::CLOUD_MANAGER_PREAMBLE,
        language,
    );
    // Context firewalls: wrap worker agents so their raw tool call logs
    // are condensed before the manager sees them. The manager receives only
    // a compact summary (files modified, write count, termination reason).
    // Research: NLAH (arxiv:2603.25723) — sub-agent isolation is the
    // "most impactful structural decision" for multi-agent harnesses.
    let mut builder = client
        .agent(model)
        .name("manager")
        .description("Cloud-backed orchestrator that delegates to local HPC model workers")
        .preamble(&preamble)
        .temperature(0.3)
        // Agent-as-Tool: specialists (wrapped with context firewall)
        .tool(CondensedAgentTool::new(workers.planner))
        .tool(CondensedAgentTool::new(workers.fixer))
        // Agent-as-Tool: workers (wrapped with context firewall)
        .tool(CondensedAgentTool::new(workers.rust_coder))
        .tool(CondensedAgentTool::new(workers.general_coder))
        .tool(CondensedAgentTool::new(workers.reviewer));

    // Architect/Editor pattern (cloud manager only — Aider-inspired split)
    if let Some(architect) = workers.architect {
        builder = builder.tool(CondensedAgentTool::new(architect));
    }
    if let Some(editor) = workers.editor {
        builder = builder.tool(CondensedAgentTool::new(editor));
    }

    // Direct plan application — instant alternative to proxy_editor.
    // The manager can call apply_plan with the Architect's JSON output
    // to apply edits deterministically (~0.1s) instead of routing through
    // the Editor agent (15 turns, ~10 min on local models).
    builder = builder.tool(crate::tools::apply_plan_tool::ApplyPlanTool::new(wt_path));

    // Reasoning worker only present in cloud manager (context firewall applied)
    if let Some(rw) = workers.reasoning_worker {
        builder = builder.tool(CondensedAgentTool::new(rw));
    }

    // Strategist advisor (read-only, context firewall applied)
    if let Some(st) = workers.strategist {
        builder = builder.tool(CondensedAgentTool::new(st));
    }

    // Strategy tools — proxy-prefixed for CLIAPIProxy compatibility.
    // Kernel sees verifier, diff, changed_files only. No read/write/edit.
    builder = builder.tools(bundles::kernel_strategy_tools(
        wt_path,
        verifier_packages,
        true,
    ));

    // Parallel work planning tool (optional — enables manager-guided decomposition).
    if let Some(plan_slot) = workers.plan_slot {
        builder = builder.tool(PlanParallelWorkTool::new(plan_slot));
    }

    // Work plan submission tool (ClawTeam pattern — plan-before-execute gate).
    if let Some(work_plan_slot) = workers.work_plan_slot.clone() {
        builder = builder.tool(SubmitPlanTool::new(work_plan_slot, 1));
    }

    // Knowledge base tool (optional — gracefully absent if not configured)
    let kb_tools = bundles::notebook_tool(workers.notebook_bridge, true);
    if !kb_tools.is_empty() {
        builder = builder.tools(kb_tools);
    }

    // Coordination tools (placeholder — Phase 2 will add `bd mail` tools)
    let coord_tools = bundles::coordination_tools(wt_path);
    if !coord_tools.is_empty() {
        builder = builder.tools(coord_tools);
    }

    builder.default_max_turns(manager_max_turns()).build()
}

/// Build the local-only Manager (Qwen3.5-Architect fallback when cloud unavailable).
///
/// Workers are coders only (no reasoning_worker — Qwen3.5-Architect IS the manager).
pub fn build_local_manager(
    client: &openai::CompletionsClient,
    model: &str,
    workers: ManagerWorkers,
    wt_path: &Path,
    verifier_packages: &[String],
) -> OaiAgent {
    build_local_manager_for_language(client, model, workers, wt_path, verifier_packages, None)
}

/// Build the local-only Manager with language-aware prompts.
pub fn build_local_manager_for_language(
    client: &openai::CompletionsClient,
    model: &str,
    workers: ManagerWorkers,
    wt_path: &Path,
    verifier_packages: &[String],
    language: Option<&str>,
) -> OaiAgent {
    let preamble = prompts::build_full_manager_preamble(
        "local_manager",
        wt_path,
        prompts::LOCAL_MANAGER_PREAMBLE,
        language,
    );
    // Context firewalls: wrap worker agents so their raw tool call logs
    // are condensed before the local manager sees them.
    let mut builder = client
        .agent(model)
        .name("manager")
        .description("Orchestrator that decomposes tasks and delegates to specialized workers")
        .preamble(&preamble)
        .temperature(crate::agents::coder::worker_temperature())
        // Agent-as-Tool: specialists (wrapped with context firewall)
        .tool(CondensedAgentTool::new(workers.planner))
        .tool(CondensedAgentTool::new(workers.fixer))
        // Agent-as-Tool: workers (wrapped with context firewall)
        .tool(CondensedAgentTool::new(workers.rust_coder))
        .tool(CondensedAgentTool::new(workers.general_coder))
        .tool(CondensedAgentTool::new(workers.reviewer));

    // Strategist advisor (read-only, context firewall applied)
    if let Some(st) = workers.strategist {
        builder = builder.tool(CondensedAgentTool::new(st));
    }

    // Strategy tools — no proxy prefix for local models.
    // Same segregation as cloud: verifier, diff, changed_files only.
    builder = builder.tools(bundles::kernel_strategy_tools(
        wt_path,
        verifier_packages,
        false,
    ));

    // Parallel work planning tool (optional).
    if let Some(plan_slot) = workers.plan_slot {
        builder = builder.tool(PlanParallelWorkTool::new(plan_slot));
    }

    // Work plan submission tool (ClawTeam pattern — plan-before-execute gate).
    if let Some(work_plan_slot) = workers.work_plan_slot.clone() {
        builder = builder.tool(SubmitPlanTool::new(work_plan_slot, 1));
    }

    // Knowledge base tool (optional)
    let kb_tools = bundles::notebook_tool(workers.notebook_bridge, false);
    if !kb_tools.is_empty() {
        builder = builder.tools(kb_tools);
    }

    // Coordination tools (placeholder — Phase 2 will add `bd mail` tools)
    let coord_tools = bundles::coordination_tools(wt_path);
    if !coord_tools.is_empty() {
        builder = builder.tools(coord_tools);
    }

    builder.default_max_turns(manager_max_turns()).build()
}
