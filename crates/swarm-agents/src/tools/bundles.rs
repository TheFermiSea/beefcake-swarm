//! Tool bundle constructors for role-based agent wiring.
//!
//! Eliminates duplicated `.tool(...)` chains by building `Vec<Box<dyn ToolDyn>>`
//! bundles per role. Handles proxy/non-proxy wrapping internally.
//!
//! # Roles
//!
//! - **Worker (Rust specialist)**: read, write, edit, run_command (no list_files)
//! - **Worker (General/Reasoning)**: read, write, edit, list_files, run_command
//! - **Manager (deterministic tools)**: verifier, read, list_files
//! - **Manager (knowledge base)**: query_notebook (optional addon)

use std::path::Path;
use std::sync::Arc;

use rig::tool::ToolDyn;

use super::astgrep_tool::AstGrepTool;
use super::bdh_tools::{
    ChatCheckTool, ChatSendTool, CheckLocksTool, CheckMailTool, SendMailTool, TeamStatusTool,
};
use super::colgrep_tool::ColGrepTool;
use super::exec_tool::RunCommandTool;
use super::fs_tools::{ListFilesTool, ReadFileTool, WriteFileTool};
use super::git_tools::{GetDiffTool, ListChangedFilesTool};
use super::notebook_tool::QueryNotebookTool;
use super::patch_tool::EditFileTool;
use super::proxy_wrappers::{
    ProxyAstGrep, ProxyColGrep, ProxyEditFile, ProxyGetDiff, ProxyListChangedFiles, ProxyListFiles,
    ProxyQueryNotebook, ProxyReadFile, ProxyRunCommand, ProxyRunVerifier, ProxySearchCode,
    ProxyWriteFile,
};
use super::search_code_tool::SearchCodeTool;
use super::verifier_tool::RunVerifierTool;
use super::workpad_tool::{AnnounceTool, CheckAnnouncementsTool};
use crate::notebook_bridge::KnowledgeBase;

/// Which set of tools a worker agent receives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerRole {
    /// Rust specialist: read, write, edit, run_command (no list_files).
    RustSpecialist,
    /// General coder / reasoning worker: read, write, edit, list_files, run_command.
    General,
    /// Planner: read-only tools for analysis (read_file, list_files, run_command).
    Planner,
    /// Strategist: advisor tier (read-only tools for analysis).
    Strategist,
}

/// Build the tool bundle for a worker agent.
///
/// Delegates to [`process_tactical_tools`] for the core tactical tools,
/// then appends worker chat tools for bdh coordination.
///
/// When `proxy` is true, tools are wrapped with `proxy_` prefixed names
/// for CLIAPIProxy compatibility.
pub fn worker_tools(wt_path: &Path, role: WorkerRole, proxy: bool) -> Vec<Box<dyn ToolDyn>> {
    let mut tools = process_tactical_tools(wt_path, role, proxy);

    // Workers get chat_send when bdh coordination is active.
    // This lets workers signal the manager when stuck or need clarification.
    // (Planners and Strategists are read-only — no chat needed.)
    if role != WorkerRole::Planner && role != WorkerRole::Strategist {
        tools.extend(worker_chat_tools(wt_path));
    }

    tools
}

/// Build a worker tool bundle with a file allowlist for subtask dispatch.
///
/// Like `worker_tools`, but `edit_file` and `write_file` will reject writes
/// to paths outside `target_files`, enforcing the non-overlap constraint at
/// the tool layer (not just the prompt).
///
/// Includes workpad tools (`announce` + `check_announcements`) for inter-worker
/// communication during concurrent execution. Workers announce interface changes
/// so other workers can adapt before their final edits.
pub fn subtask_worker_tools(
    wt_path: &Path,
    role: WorkerRole,
    target_files: &[String],
    worker_id: &str,
) -> Vec<Box<dyn ToolDyn>> {
    let allowlist: std::collections::HashSet<String> = target_files.iter().cloned().collect();

    let mut tools: Vec<Box<dyn ToolDyn>> = vec![
        Box::new(ReadFileTool::new(wt_path)),
        Box::new(WriteFileTool::new_with_allowlist(
            wt_path,
            allowlist.clone(),
        )),
        Box::new(EditFileTool::new_with_allowlist(wt_path, allowlist)),
        Box::new(RunCommandTool::new(wt_path)),
        Box::new(SearchCodeTool::new(wt_path)),
        Box::new(ColGrepTool::new(wt_path)),
        Box::new(AstGrepTool::new(wt_path)),
    ];

    if role == WorkerRole::General {
        tools.push(Box::new(ListFilesTool::new(wt_path)));
    }

    // Workpad tools for inter-worker communication during concurrent dispatch.
    tools.push(Box::new(AnnounceTool::new(wt_path, worker_id)));
    tools.push(Box::new(CheckAnnouncementsTool::new(wt_path, worker_id)));

    tools
}

/// Build the deterministic tool bundle for a manager agent.
///
/// Both cloud and local managers get strategy-only tools via
/// [`kernel_strategy_tools`]: verifier, diff, changed_files.
/// NO read_file, write_file, or list_files — forces the manager to
/// delegate exploration and code changes to workers.
///
/// Local managers previously retained read/list, but the Slate
/// architecture enforces strict segregation: managers orchestrate,
/// workers execute. Local managers have workers as agent-tools and
/// should delegate through them.
///
/// When `proxy` is true, tools are wrapped with `proxy_` prefix.
pub fn manager_tools(
    wt_path: &Path,
    verifier_packages: &[String],
    proxy: bool,
) -> Vec<Box<dyn ToolDyn>> {
    kernel_strategy_tools(wt_path, verifier_packages, proxy)
}

/// Build bdh coordination tools for the manager agent.
///
/// Returns coordination tools when `SWARM_USE_BDH=1`, empty vec otherwise.
/// These give the manager team awareness: who's working on what, file locks,
/// and inter-agent messaging (including chat).
pub fn coordination_tools(wt_path: &Path) -> Vec<Box<dyn ToolDyn>> {
    let use_bdh = std::env::var("SWARM_USE_BDH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !use_bdh {
        return vec![];
    }

    vec![
        Box::new(TeamStatusTool::new(wt_path)),
        Box::new(CheckMailTool::new(wt_path)),
        Box::new(SendMailTool::new(wt_path)),
        Box::new(CheckLocksTool::new(wt_path)),
        Box::new(ChatSendTool::new(wt_path)),
        Box::new(ChatCheckTool::new(wt_path)),
    ]
}

/// Build bdh chat tools for worker agents.
///
/// Returns chat_send when `SWARM_USE_BDH=1`, empty vec otherwise.
/// Workers can signal the manager when stuck or when they need clarification.
pub fn worker_chat_tools(wt_path: &Path) -> Vec<Box<dyn ToolDyn>> {
    let use_bdh = std::env::var("SWARM_USE_BDH")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if !use_bdh {
        return vec![];
    }

    vec![Box::new(ChatSendTool::new(wt_path))]
}

/// Build the **strategy-only** tool bundle for the kernel (cloud manager).
///
/// Strategy tools let the kernel observe and orchestrate without touching code:
/// - `run_verifier`: validate worker output (cargo fmt, clippy, check, test)
/// - `get_diff` / `list_changed_files`: inspect what workers changed
///
/// The kernel delegates all code-level work to worker processes. This enforces
/// the LLM-as-OS-Kernel pattern: the kernel sees the world through summaries
/// and diffs, never through raw file reads or writes.
///
/// When `proxy` is true, tools get `proxy_` prefix for CLIAPIProxy compatibility.
pub fn kernel_strategy_tools(
    wt_path: &Path,
    verifier_packages: &[String],
    proxy: bool,
) -> Vec<Box<dyn ToolDyn>> {
    // Manager gets ONLY coordination/observation tools — NO search tools.
    // Search tools (search_code, colgrep, ast_grep) belong exclusively in
    // process_tactical_tools() for workers. The manager MUST delegate all
    // code exploration to workers via proxy_rust_coder/proxy_general_coder.
    if proxy {
        vec![
            Box::new(ProxyRunVerifier(
                RunVerifierTool::new(wt_path).with_packages(verifier_packages.to_vec()),
            )),
            Box::new(ProxyGetDiff(GetDiffTool::new(wt_path))),
            Box::new(ProxyListChangedFiles(ListChangedFilesTool::new(wt_path))),
        ]
    } else {
        vec![
            Box::new(RunVerifierTool::new(wt_path).with_packages(verifier_packages.to_vec())),
            Box::new(GetDiffTool::new(wt_path)),
            Box::new(ListChangedFilesTool::new(wt_path)),
        ]
    }
}

/// Build the **tactics-only** tool bundle for worker processes.
///
/// Tactical tools let workers interact with code and the filesystem:
/// - `read_file`, `write_file`, `edit_file`: code manipulation
/// - `run_command`: execute cargo, grep, etc.
/// - `list_files`: directory exploration (General/Reasoning roles only)
///
/// Workers never get dispatch, verifier, or coordination tools — they execute
/// subtasks assigned by the kernel and report results back.
///
/// When `proxy` is true, tools get `proxy_` prefix for CLIAPIProxy compatibility.
pub fn process_tactical_tools(
    wt_path: &Path,
    role: WorkerRole,
    proxy: bool,
) -> Vec<Box<dyn ToolDyn>> {
    match role {
        WorkerRole::Planner | WorkerRole::Strategist => {
            // Read-only tools for analysis: read_file, list_files, run_command, plus search tools.
            if proxy {
                vec![
                    Box::new(ProxyReadFile(ReadFileTool::new(wt_path))),
                    Box::new(ProxyListFiles(ListFilesTool::new(wt_path))),
                    Box::new(ProxyRunCommand(RunCommandTool::new(wt_path))),
                    Box::new(ProxySearchCode(SearchCodeTool::new(wt_path))),
                    Box::new(ProxyColGrep(ColGrepTool::new(wt_path))),
                    Box::new(ProxyAstGrep(AstGrepTool::new(wt_path))),
                ]
            } else {
                vec![
                    Box::new(ReadFileTool::new(wt_path)),
                    Box::new(ListFilesTool::new(wt_path)),
                    Box::new(RunCommandTool::new(wt_path)),
                    Box::new(SearchCodeTool::new(wt_path)),
                    Box::new(ColGrepTool::new(wt_path)),
                    Box::new(AstGrepTool::new(wt_path)),
                ]
            }
        }
        _ => {
            let mut tools: Vec<Box<dyn ToolDyn>> = if proxy {
                vec![
                    Box::new(ProxyReadFile(ReadFileTool::new(wt_path))),
                    Box::new(ProxyWriteFile(WriteFileTool::new(wt_path))),
                    Box::new(ProxyEditFile(EditFileTool::new(wt_path))),
                    Box::new(ProxyRunCommand(RunCommandTool::new(wt_path))),
                    Box::new(ProxySearchCode(SearchCodeTool::new(wt_path))),
                    Box::new(ProxyColGrep(ColGrepTool::new(wt_path))),
                    Box::new(ProxyAstGrep(AstGrepTool::new(wt_path))),
                ]
            } else {
                vec![
                    Box::new(ReadFileTool::new(wt_path)),
                    Box::new(WriteFileTool::new(wt_path)),
                    Box::new(EditFileTool::new(wt_path)),
                    Box::new(RunCommandTool::new(wt_path)),
                    Box::new(SearchCodeTool::new(wt_path)),
                    Box::new(ColGrepTool::new(wt_path)),
                    Box::new(AstGrepTool::new(wt_path)),
                ]
            };

            // General/reasoning workers also get list_files for directory exploration.
            if role == WorkerRole::General {
                if proxy {
                    tools.push(Box::new(ProxyListFiles(ListFilesTool::new(wt_path))));
                } else {
                    tools.push(Box::new(ListFilesTool::new(wt_path)));
                }
            }

            tools
        }
    }
}

/// Build the optional knowledge base tool for a manager agent.
///
/// Returns an empty vec if `kb` is None.
pub fn notebook_tool(kb: Option<Arc<dyn KnowledgeBase>>, proxy: bool) -> Vec<Box<dyn ToolDyn>> {
    match kb {
        Some(kb) => {
            if proxy {
                vec![Box::new(ProxyQueryNotebook(QueryNotebookTool::new(kb)))]
            } else {
                vec![Box::new(QueryNotebookTool::new(kb))]
            }
        }
        None => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_rust_specialist_has_expected_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::RustSpecialist, false);
        // read, write, edit, run + search_code, colgrep, astgrep = 7
        assert_eq!(tools.len(), 7, "Rust specialist should have 7 tools");
    }

    #[test]
    fn test_worker_general_has_expected_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::General, false);
        // read, write, edit, run, list + search_code, colgrep, astgrep = 8
        assert_eq!(tools.len(), 8, "General worker should have 8 tools");
    }

    #[test]
    fn test_worker_proxy_same_count() {
        let dir = tempfile::tempdir().unwrap();
        let local = worker_tools(dir.path(), WorkerRole::General, false);
        let proxy = worker_tools(dir.path(), WorkerRole::General, true);
        assert_eq!(local.len(), proxy.len());
    }

    #[test]
    fn test_worker_proxy_names_have_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::General, true);
        for tool in &tools {
            assert!(
                tool.name().starts_with("proxy_"),
                "Expected proxy_ prefix on tool name: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn test_worker_local_names_no_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::General, false);
        for tool in &tools {
            assert!(
                !tool.name().starts_with("proxy_"),
                "Local tool should not have proxy_ prefix: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn test_manager_local_has_strategy_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = manager_tools(dir.path(), &["test-pkg".to_string()], false);
        // verifier, get_diff, list_changed_files = 3 (no search tools — manager must delegate)
        assert_eq!(
            tools.len(),
            3,
            "Local manager should have 3 strategy tools (verifier, get_diff, list_changed_files)"
        );
    }

    #[test]
    fn test_manager_cloud_has_strategy_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = manager_tools(dir.path(), &["test-pkg".to_string()], true);
        assert_eq!(tools.len(), 3, "Cloud manager should have 3 strategy tools (verifier, diff, changed_files)");
    }

    #[test]
    fn test_manager_no_read_list_tools() {
        let dir = tempfile::tempdir().unwrap();
        // Both cloud and local managers should lack read_file/list_files.
        for proxy in [false, true] {
            let tools = manager_tools(dir.path(), &[], proxy);
            let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
            assert!(
                !names
                    .iter()
                    .any(|n| n.contains("read_file") || n.contains("list_files")),
                "Manager (proxy={proxy}) should not have read_file/list_files, got: {names:?}"
            );
        }
    }

    #[test]
    fn test_manager_no_write_tools() {
        let dir = tempfile::tempdir().unwrap();
        for proxy in [false, true] {
            let tools = manager_tools(dir.path(), &[], proxy);
            let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
            assert!(
                !names
                    .iter()
                    .any(|n| n.contains("write") || n.contains("edit")),
                "Manager (proxy={proxy}) should not have write/edit tools, got: {names:?}"
            );
        }
    }

    #[test]
    fn test_manager_no_search_tools() {
        let dir = tempfile::tempdir().unwrap();
        // Manager must delegate ALL code exploration to workers.
        for proxy in [false, true] {
            let tools = manager_tools(dir.path(), &[], proxy);
            let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
            assert!(
                !names
                    .iter()
                    .any(|n| n.contains("search_code") || n.contains("colgrep") || n.contains("ast_grep")),
                "Manager (proxy={proxy}) must not have search tools — delegate to workers, got: {names:?}"
            );
        }
    }

    #[test]
    fn test_manager_proxy_names_have_prefix() {
        let dir = tempfile::tempdir().unwrap();
        let tools = manager_tools(dir.path(), &[], true);
        for tool in &tools {
            assert!(
                tool.name().starts_with("proxy_"),
                "Expected proxy_ prefix: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn test_worker_planner_has_expected_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::Planner, false);
        // read, list, run + search_code, colgrep, astgrep = 6
        assert_eq!(
            tools.len(),
            6,
            "Planner should have 6 read-only tools (read, list, run, search_code, colgrep, astgrep)"
        );
    }

    #[test]
    fn test_worker_planner_no_write_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::Planner, false);
        let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
        assert!(
            !names
                .iter()
                .any(|n| n.contains("write") || n.contains("edit")),
            "Planner should not have write/edit tools, got: {names:?}"
        );
    }

    #[test]
    fn test_worker_planner_proxy() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::Planner, true);
        assert_eq!(tools.len(), 6);
        for tool in &tools {
            assert!(
                tool.name().starts_with("proxy_"),
                "Expected proxy_ prefix: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn test_notebook_tool_none_returns_empty() {
        let tools = notebook_tool(None, false);
        assert!(tools.is_empty());
    }

    #[test]
    fn test_notebook_tool_with_kb() {
        use crate::notebook_bridge::NoOpKnowledgeBase;
        let kb: Arc<dyn KnowledgeBase> = Arc::new(NoOpKnowledgeBase);
        let tools = notebook_tool(Some(kb), false);
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "query_notebook");
    }

    #[test]
    fn test_notebook_tool_proxy() {
        use crate::notebook_bridge::NoOpKnowledgeBase;
        let kb: Arc<dyn KnowledgeBase> = Arc::new(NoOpKnowledgeBase);
        let tools = notebook_tool(Some(kb), true);
        assert_eq!(tools.len(), 1);
        assert!(tools[0].name().starts_with("proxy_"));
    }

    // ── Strategy/Tactics segregation tests ──────────────────────────

    #[test]
    fn test_kernel_strategy_tools_local() {
        let dir = tempfile::tempdir().unwrap();
        let tools = kernel_strategy_tools(dir.path(), &[], false);
        // verifier, get_diff, list_changed_files = 3 (no search tools — manager must delegate)
        assert_eq!(
            tools.len(),
            3,
            "Strategy bundle: verifier, get_diff, list_changed_files (no search tools)"
        );
        let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
        assert!(names.iter().any(|n| n.contains("verifier")));
        assert!(names.iter().any(|n| n.contains("get_diff")));
        assert!(names.iter().any(|n| n.contains("list_changed_files")));
    }

    #[test]
    fn test_kernel_strategy_tools_proxy() {
        let dir = tempfile::tempdir().unwrap();
        let tools = kernel_strategy_tools(dir.path(), &["pkg".to_string()], true);
        assert_eq!(tools.len(), 3);
        for tool in &tools {
            assert!(
                tool.name().starts_with("proxy_"),
                "Strategy tool should have proxy_ prefix: {}",
                tool.name()
            );
        }
    }

    #[test]
    fn test_kernel_strategy_has_no_tactical_tools() {
        let dir = tempfile::tempdir().unwrap();
        for proxy in [false, true] {
            let tools = kernel_strategy_tools(dir.path(), &[], proxy);
            let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
            let tactical_names = [
                "read_file",
                "write_file",
                "edit_file",
                "run_command",
                "list_files",
            ];
            for bad in &tactical_names {
                assert!(
                    !names.iter().any(|n| n.contains(bad)),
                    "Strategy bundle (proxy={proxy}) should not contain tactical tool '{bad}', got: {names:?}"
                );
            }
        }
    }

    #[test]
    fn test_process_tactical_tools_general() {
        let dir = tempfile::tempdir().unwrap();
        let tools = process_tactical_tools(dir.path(), WorkerRole::General, false);
        // read, write, edit, run, list + search_code, colgrep, astgrep = 8
        assert_eq!(
            tools.len(),
            8,
            "General tactical: read, write, edit, run, list, search_code, colgrep, astgrep"
        );
        let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
        assert!(names.iter().any(|n| n.contains("read_file")));
        assert!(names.iter().any(|n| n.contains("write_file")));
        assert!(names.iter().any(|n| n.contains("edit_file")));
        assert!(names.iter().any(|n| n.contains("run_command")));
        assert!(names.iter().any(|n| n.contains("list_files")));
    }

    #[test]
    fn test_process_tactical_tools_rust_specialist() {
        let dir = tempfile::tempdir().unwrap();
        let tools = process_tactical_tools(dir.path(), WorkerRole::RustSpecialist, false);
        // read, write, edit, run + search_code, colgrep, astgrep = 7
        assert_eq!(
            tools.len(),
            7,
            "Rust specialist tactical: read, write, edit, run, search_code, colgrep, astgrep (no list)"
        );
    }

    #[test]
    fn test_process_tactical_has_no_strategy_tools() {
        let dir = tempfile::tempdir().unwrap();
        for role in [
            WorkerRole::RustSpecialist,
            WorkerRole::General,
            WorkerRole::Planner,
        ] {
            for proxy in [false, true] {
                let tools = process_tactical_tools(dir.path(), role, proxy);
                let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
                let strategy_names = ["verifier", "get_diff", "list_changed_files"];
                for bad in &strategy_names {
                    assert!(
                        !names.iter().any(|n| n.contains(bad)),
                        "Tactical bundle ({role:?}, proxy={proxy}) should not contain strategy tool '{bad}', got: {names:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_strategy_and_tactical_are_disjoint_except_search() {
        let dir = tempfile::tempdir().unwrap();
        let strategy: Vec<String> = kernel_strategy_tools(dir.path(), &[], false)
            .iter()
            .map(|t| t.name())
            .collect();
        let tactical: Vec<String> = process_tactical_tools(dir.path(), WorkerRole::General, false)
            .iter()
            .map(|t| t.name())
            .collect();
        // Search tools (search_code, colgrep, ast_grep) are intentionally shared
        // between strategy and tactical bundles — both managers and workers need
        // code search. Only the core strategy tools must be exclusive.
        let shared_ok = ["search_code", "colgrep", "ast_grep"];
        for s in &strategy {
            if shared_ok.iter().any(|ok| s.contains(ok)) {
                continue;
            }
            assert!(
                !tactical.contains(s),
                "Tool '{s}' appears in both strategy and tactical bundles — must be disjoint (except search tools)"
            );
        }
    }
}
