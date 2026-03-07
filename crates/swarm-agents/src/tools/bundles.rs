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

use super::bdh_tools::{CheckLocksTool, CheckMailTool, SendMailTool, TeamStatusTool};
use super::exec_tool::RunCommandTool;
use super::fs_tools::{ListFilesTool, ReadFileTool, WriteFileTool};
use super::git_tools::{GetDiffTool, ListChangedFilesTool};
use super::notebook_tool::QueryNotebookTool;
use super::patch_tool::EditFileTool;
use super::proxy_wrappers::{
    ProxyEditFile, ProxyGetDiff, ProxyListChangedFiles, ProxyListFiles, ProxyQueryNotebook,
    ProxyReadFile, ProxyRunCommand, ProxyRunVerifier, ProxyWriteFile,
};
use super::verifier_tool::RunVerifierTool;
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
}

/// Build the tool bundle for a worker agent.
///
/// When `proxy` is true, tools are wrapped with `proxy_` prefixed names
/// for CLIAPIProxy compatibility.
pub fn worker_tools(wt_path: &Path, role: WorkerRole, proxy: bool) -> Vec<Box<dyn ToolDyn>> {
    match role {
        WorkerRole::Planner => {
            // Read-only tools for analysis: read_file, list_files, run_command.
            if proxy {
                vec![
                    Box::new(ProxyReadFile(ReadFileTool::new(wt_path))),
                    Box::new(ProxyListFiles(ListFilesTool::new(wt_path))),
                    Box::new(ProxyRunCommand(RunCommandTool::new(wt_path))),
                ]
            } else {
                vec![
                    Box::new(ReadFileTool::new(wt_path)),
                    Box::new(ListFilesTool::new(wt_path)),
                    Box::new(RunCommandTool::new(wt_path)),
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
                ]
            } else {
                vec![
                    Box::new(ReadFileTool::new(wt_path)),
                    Box::new(WriteFileTool::new(wt_path)),
                    Box::new(EditFileTool::new(wt_path)),
                    Box::new(RunCommandTool::new(wt_path)),
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

/// Build a worker tool bundle with a file allowlist for subtask dispatch.
///
/// Like `worker_tools`, but `edit_file` and `write_file` will reject writes
/// to paths outside `target_files`, enforcing the non-overlap constraint at
/// the tool layer (not just the prompt).
pub fn subtask_worker_tools(
    wt_path: &Path,
    role: WorkerRole,
    target_files: &[String],
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
    ];

    if role == WorkerRole::General {
        tools.push(Box::new(ListFilesTool::new(wt_path)));
    }

    tools
}

/// Build the deterministic tool bundle for a manager agent.
///
/// Cloud managers get delegate-only tools: verifier, diff, changed_files.
/// NO read_file or list_files — forces the manager to delegate exploration
/// to workers instead of absorbing read-work itself.
///
/// Local managers retain read/list since they may need to explore without
/// cloud-quality planning ability.
///
/// When `proxy` is true, tools are wrapped with `proxy_` prefix.
pub fn manager_tools(
    wt_path: &Path,
    verifier_packages: &[String],
    proxy: bool,
) -> Vec<Box<dyn ToolDyn>> {
    if proxy {
        // Cloud manager: delegate-only (no read_file, no list_files).
        // Workers have these tools — the manager must delegate to them.
        vec![
            Box::new(ProxyRunVerifier(
                RunVerifierTool::new(wt_path).with_packages(verifier_packages.to_vec()),
            )),
            Box::new(ProxyGetDiff(GetDiffTool::new(wt_path))),
            Box::new(ProxyListChangedFiles(ListChangedFilesTool::new(wt_path))),
        ]
    } else {
        // Local manager: retains read/list for direct exploration.
        vec![
            Box::new(RunVerifierTool::new(wt_path).with_packages(verifier_packages.to_vec())),
            Box::new(ReadFileTool::new(wt_path)),
            Box::new(ListFilesTool::new(wt_path)),
            Box::new(GetDiffTool::new(wt_path)),
            Box::new(ListChangedFilesTool::new(wt_path)),
        ]
    }
}

/// Build bdh coordination tools for the manager agent.
///
/// Returns coordination tools when `SWARM_USE_BDH=1`, empty vec otherwise.
/// These give the manager team awareness: who's working on what, file locks,
/// and inter-agent messaging.
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
    ]
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
    fn test_worker_rust_specialist_has_4_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::RustSpecialist, false);
        assert_eq!(tools.len(), 4, "Rust specialist should have 4 tools");
    }

    #[test]
    fn test_worker_general_has_5_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::General, false);
        assert_eq!(tools.len(), 5, "General worker should have 5 tools");
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
    fn test_manager_local_has_5_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = manager_tools(dir.path(), &["test-pkg".to_string()], false);
        assert_eq!(
            tools.len(),
            5,
            "Local manager should have 5 tools (verifier, read, list, get_diff, list_changed_files)"
        );
    }

    #[test]
    fn test_manager_cloud_has_3_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = manager_tools(dir.path(), &["test-pkg".to_string()], true);
        assert_eq!(
            tools.len(),
            3,
            "Cloud manager should have 3 delegate-only tools (verifier, get_diff, list_changed_files)"
        );
    }

    #[test]
    fn test_manager_cloud_no_read_list_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = manager_tools(dir.path(), &[], true);
        let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
        assert!(
            !names
                .iter()
                .any(|n| n.contains("read_file") || n.contains("list_files")),
            "Cloud manager should not have read_file/list_files, got: {names:?}"
        );
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
    fn test_worker_planner_has_3_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = worker_tools(dir.path(), WorkerRole::Planner, false);
        assert_eq!(
            tools.len(),
            3,
            "Planner should have 3 read-only tools (read, list, run)"
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
        assert_eq!(tools.len(), 3);
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
}
