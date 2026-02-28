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

use super::exec_tool::RunCommandTool;
use super::fs_tools::{ListFilesTool, ReadFileTool, WriteFileTool};
use super::notebook_tool::QueryNotebookTool;
use super::patch_tool::EditFileTool;
use super::proxy_wrappers::{
    ProxyEditFile, ProxyListFiles, ProxyQueryNotebook, ProxyReadFile, ProxyRunCommand,
    ProxyRunVerifier, ProxyWriteFile,
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

/// Build the deterministic tool bundle for a manager agent.
///
/// Includes verifier, read, list_files. Managers delegate code changes
/// to workers — they should never write/edit files directly.
/// When `proxy` is true, tools are wrapped with `proxy_` prefix.
pub fn manager_tools(
    wt_path: &Path,
    verifier_packages: &[String],
    proxy: bool,
) -> Vec<Box<dyn ToolDyn>> {
    if proxy {
        vec![
            Box::new(ProxyRunVerifier(
                RunVerifierTool::new(wt_path).with_packages(verifier_packages.to_vec()),
            )),
            Box::new(ProxyReadFile(ReadFileTool::new(wt_path))),
            Box::new(ProxyListFiles(ListFilesTool::new(wt_path))),
        ]
    } else {
        vec![
            Box::new(RunVerifierTool::new(wt_path).with_packages(verifier_packages.to_vec())),
            Box::new(ReadFileTool::new(wt_path)),
            Box::new(ListFilesTool::new(wt_path)),
        ]
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
    fn test_manager_tools_has_3_tools() {
        let dir = tempfile::tempdir().unwrap();
        let tools = manager_tools(dir.path(), &["test-pkg".to_string()], false);
        assert_eq!(
            tools.len(),
            3,
            "Manager should have 3 tools (verifier, read, list)"
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
