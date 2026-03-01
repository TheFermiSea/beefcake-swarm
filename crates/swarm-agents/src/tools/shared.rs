//! Shared tool factory for concurrent agent sessions.
//!
//! Wraps tool construction so multiple agents can share the same tool
//! factory without rebuilding configuration per-agent. Each agent still
//! gets its own `Vec<Box<dyn ToolDyn>>` (tools are rebuilt from the factory),
//! but construction parameters are centralized and the factory is `Clone`-able.
//!
//! # Thread Safety
//!
//! `ToolFactory` is `Clone + Send + Sync`, so it can be shared across
//! concurrent agent sessions (e.g., behind an `Arc` or just cloned).
//!
//! # Example
//!
//! ```ignore
//! let wt_path = Path::new("/tmp/worktree");
//! let factory = ToolFactory::new(wt_path, false, vec![], None);
//!
//! // Each agent gets its own tool set from the same factory
//! let coder_tools = factory.worker_tools(WorkerRole::RustSpecialist);
//! let planner_tools = factory.worker_tools(WorkerRole::Planner);
//! let mgr_tools = factory.manager_tools();
//! ```

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig::tool::ToolDyn;

use super::bundles::{self, WorkerRole};
use crate::notebook_bridge::KnowledgeBase;

/// Factory for creating tool sets scoped to a worktree path.
///
/// Clone-able: each clone produces tools for the same worktree.
/// Thread-safe: can be shared across concurrent agent sessions.
///
/// This centralizes tool construction parameters (worktree path, proxy mode,
/// verifier packages, knowledge base) so callers don't need to thread them
/// through every agent builder call.
#[derive(Clone)]
pub struct ToolFactory {
    wt_path: PathBuf,
    proxy: bool,
    verifier_packages: Vec<String>,
    kb: Option<Arc<dyn KnowledgeBase>>,
}

impl ToolFactory {
    /// Create a new tool factory scoped to a worktree.
    ///
    /// # Arguments
    ///
    /// * `wt_path` - Root directory of the agent's worktree (sandbox boundary).
    /// * `proxy` - If true, tools get `proxy_` prefix for CLIAPIProxy compatibility.
    /// * `verifier_packages` - Cargo packages to scope the verifier to (empty = whole workspace).
    /// * `kb` - Optional knowledge base for the notebook query tool.
    pub fn new(
        wt_path: &Path,
        proxy: bool,
        verifier_packages: Vec<String>,
        kb: Option<Arc<dyn KnowledgeBase>>,
    ) -> Self {
        Self {
            wt_path: wt_path.to_path_buf(),
            proxy,
            verifier_packages,
            kb,
        }
    }

    /// The worktree path this factory is scoped to.
    pub fn wt_path(&self) -> &Path {
        &self.wt_path
    }

    /// Whether tools are proxy-prefixed.
    pub fn is_proxy(&self) -> bool {
        self.proxy
    }

    /// Build the tool bundle for a worker agent of the given role.
    ///
    /// Delegates to [`bundles::worker_tools`] with the factory's stored parameters.
    pub fn worker_tools(&self, role: WorkerRole) -> Vec<Box<dyn ToolDyn>> {
        bundles::worker_tools(&self.wt_path, role, self.proxy)
    }

    /// Build the deterministic tool bundle for a manager agent.
    ///
    /// Includes verifier, read_file, list_files.
    /// Delegates to [`bundles::manager_tools`] with the factory's stored parameters.
    pub fn manager_tools(&self) -> Vec<Box<dyn ToolDyn>> {
        bundles::manager_tools(&self.wt_path, &self.verifier_packages, self.proxy)
    }

    /// Build the optional knowledge base (notebook) tool bundle.
    ///
    /// Returns an empty vec if no knowledge base was configured.
    /// Delegates to [`bundles::notebook_tool`] with the factory's stored parameters.
    pub fn notebook_tools(&self) -> Vec<Box<dyn ToolDyn>> {
        bundles::notebook_tool(self.kb.clone(), self.proxy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type TestResult = Result<(), Box<dyn std::error::Error>>;

    #[test]
    fn test_factory_worker_rust_specialist() -> TestResult {
        // Rust specialist: read_file, write_file, edit_file, run_command (no list_files).
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), false, vec![], None);
        let tools = factory.worker_tools(WorkerRole::RustSpecialist);
        assert_eq!(tools.len(), 4, "Rust specialist should have 4 tools");
        Ok(())
    }

    #[test]
    fn test_factory_worker_general() -> TestResult {
        // General: read_file, write_file, edit_file, run_command, list_files.
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), false, vec![], None);
        let tools = factory.worker_tools(WorkerRole::General);
        assert_eq!(tools.len(), 5, "General worker should have 5 tools");
        Ok(())
    }

    #[test]
    fn test_factory_worker_planner() -> TestResult {
        // Planner (read-only): read_file, list_files, run_command.
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), false, vec![], None);
        let tools = factory.worker_tools(WorkerRole::Planner);
        assert_eq!(tools.len(), 3, "Planner should have 3 tools");
        Ok(())
    }

    #[test]
    fn test_factory_proxy_mode() -> TestResult {
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), true, vec![], None);
        assert!(factory.is_proxy());
        let tools = factory.worker_tools(WorkerRole::General);
        for tool in &tools {
            assert!(
                tool.name().starts_with("proxy_"),
                "Expected proxy_ prefix on tool name: {}",
                tool.name()
            );
        }
        Ok(())
    }

    #[test]
    fn test_factory_manager_tools() -> TestResult {
        // Manager (deterministic): run_verifier, read_file, list_files.
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), false, vec!["test-pkg".to_string()], None);
        let tools = factory.manager_tools();
        assert_eq!(tools.len(), 3, "Manager should have 3 tools");
        Ok(())
    }

    #[test]
    fn test_factory_notebook_tools_none() -> TestResult {
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), false, vec![], None);
        let tools = factory.notebook_tools();
        assert!(tools.is_empty(), "No KB should produce empty tool vec");
        Ok(())
    }

    #[test]
    fn test_factory_notebook_tools_with_kb() -> TestResult {
        use crate::notebook_bridge::NoOpKnowledgeBase;
        let dir = tempfile::tempdir()?;
        let kb: Arc<dyn KnowledgeBase> = Arc::new(NoOpKnowledgeBase);
        let factory = ToolFactory::new(dir.path(), false, vec![], Some(kb));
        let tools = factory.notebook_tools();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name(), "query_notebook");
        Ok(())
    }

    #[test]
    fn test_factory_is_clone() -> TestResult {
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), false, vec![], None);
        let factory2 = factory.clone();
        assert_eq!(
            factory.worker_tools(WorkerRole::General).len(),
            factory2.worker_tools(WorkerRole::General).len()
        );
        Ok(())
    }

    #[test]
    fn test_factory_is_send_sync() {
        use rig::wasm_compat::{WasmCompatSend, WasmCompatSync};
        fn assert_wasm_compat<T: WasmCompatSend + WasmCompatSync>() {}
        assert_wasm_compat::<ToolFactory>();
    }

    #[test]
    fn test_factory_wt_path() -> TestResult {
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), false, vec![], None);
        assert_eq!(factory.wt_path(), dir.path());
        Ok(())
    }

    #[test]
    fn test_factory_matches_direct_bundles() -> TestResult {
        // Verify factory produces identical tool sets to direct bundle calls.
        let dir = tempfile::tempdir()?;
        let factory = ToolFactory::new(dir.path(), false, vec!["pkg".to_string()], None);

        let mut direct_names: Vec<String> =
            bundles::worker_tools(dir.path(), WorkerRole::RustSpecialist, false)
                .iter()
                .map(|t| t.name())
                .collect();
        let mut factory_names: Vec<String> = factory
            .worker_tools(WorkerRole::RustSpecialist)
            .iter()
            .map(|t| t.name())
            .collect();
        direct_names.sort();
        factory_names.sort();
        assert_eq!(direct_names, factory_names, "Worker tool names must match");

        let mut direct_mgr_names: Vec<String> =
            bundles::manager_tools(dir.path(), &["pkg".to_string()], false)
                .iter()
                .map(|t| t.name())
                .collect();
        let mut factory_mgr_names: Vec<String> =
            factory.manager_tools().iter().map(|t| t.name()).collect();
        direct_mgr_names.sort();
        factory_mgr_names.sort();
        assert_eq!(
            direct_mgr_names, factory_mgr_names,
            "Manager tool names must match"
        );
        Ok(())
    }
}
