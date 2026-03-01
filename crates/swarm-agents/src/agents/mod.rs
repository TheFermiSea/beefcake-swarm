//! Agent builders for the Manager-Worker swarm.
//!
//! Hierarchy (when cloud available):
//!   Cloud Manager (Opus 4.6) → Local Workers (Qwen3.5-Architect on vasp-01, Qwen3.5-Implementer on vasp-02)
//!
//! Fallback (no cloud):
//!   Local Manager (Qwen3.5-Architect on vasp-01) → Local Workers (Qwen3.5-Implementer on vasp-02)

pub mod adversary;
pub mod cloud;
pub mod coder;
pub mod manager;
pub mod reviewer;
pub mod specialists;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use tracing::info;

use crate::config::{ClientSet, SwarmConfig};
use crate::notebook_bridge::KnowledgeBase;
use crate::tools::shared::ToolFactory;
use coder::OaiAgent;

/// Factory that builds all agents from a `SwarmConfig`.
///
/// Holds pre-built `ClientSet` and config references needed to construct
/// agents scoped to a particular worktree path.
pub struct AgentFactory {
    pub clients: ClientSet,
    pub config: SwarmConfig,
    /// Shared knowledge base for the notebook tool (None if unavailable).
    pub notebook_bridge: Option<Arc<dyn KnowledgeBase>>,
    /// Centralized tool construction factory, scoped to a worktree.
    ///
    /// When set, agent builders can use this instead of calling
    /// `bundles::worker_tools()` / `bundles::manager_tools()` directly.
    /// Initialize via [`AgentFactory::with_worktree`].
    pub tool_factory: Option<ToolFactory>,
}

impl AgentFactory {
    pub fn new(config: &SwarmConfig) -> Result<Self> {
        let clients = ClientSet::from_config(config)?;
        Ok(Self {
            clients,
            config: config.clone(),
            notebook_bridge: None,
            tool_factory: None,
        })
    }

    /// Set the knowledge base for the notebook tool.
    ///
    /// If a [`ToolFactory`] was already initialized via [`with_worktree`],
    /// it is rebuilt to include the new knowledge base.
    pub fn with_notebook_bridge(mut self, kb: Arc<dyn KnowledgeBase>) -> Self {
        self.notebook_bridge = Some(kb);
        // Rebuild tool factory with the updated KB if one already exists.
        if let Some(ref existing) = self.tool_factory {
            self.tool_factory = Some(ToolFactory::new(
                existing.wt_path(),
                existing.is_proxy(),
                self.config.verifier_packages.clone(),
                self.notebook_bridge.clone(),
            ));
        }
        self
    }

    /// Initialize the centralized tool factory for a given worktree path.
    ///
    /// Once set, the `tool_factory` field is available for callers that want
    /// centralized, Clone-able tool construction instead of calling
    /// `bundles::worker_tools()` / `bundles::manager_tools()` directly.
    ///
    /// Existing agent builder methods (e.g., `build_rust_coder`) continue to
    /// work unchanged via `bundles` -- this is an additive capability.
    ///
    /// # Proxy mode
    ///
    /// Uses `cloud_only` to set the proxy flag, matching all public agent
    /// builder methods (`build_rust_coder`, `build_general_coder`, etc.).
    /// The cloud manager's sub-workers are built separately with explicit
    /// `proxy=true` inside [`build_manager`] — `tool_factory` is not used
    /// for those.
    pub fn with_worktree(mut self, wt_path: &Path) -> Self {
        self.tool_factory = Some(ToolFactory::new(
            wt_path,
            self.config.cloud_only,
            self.config.verifier_packages.clone(),
            self.notebook_bridge.clone(),
        ));
        self
    }

    /// Build the Rust specialist coder (Qwen3.5-397B on vasp-02, Rust system prompt).
    ///
    /// In `cloud_only` mode, registers proxy-prefixed tools since all clients
    /// route through CLIAPIProxy which mangles tool names.
    pub fn build_rust_coder(&self, wt_path: &Path) -> OaiAgent {
        coder::build_rust_coder_named(
            &self.clients.local,
            &self.config.fast_endpoint.model,
            wt_path,
            "rust_coder",
            self.config.cloud_only,
        )
    }

    /// Build the general coder (Qwen3-Coder-Next on vasp-01, general coding system prompt).
    ///
    /// In `cloud_only` mode, registers proxy-prefixed tools since all clients
    /// route through CLIAPIProxy which mangles tool names.
    pub fn build_general_coder(&self, wt_path: &Path) -> OaiAgent {
        coder::build_general_coder_named(
            &self.clients.coder,
            &self.config.coder_endpoint.model,
            wt_path,
            "general_coder",
            self.config.cloud_only,
        )
    }

    /// Build the reasoning worker (Qwen3.5-397B on vasp-01, Architect node).
    ///
    /// Tool-equipped agent for deep analysis and complex fixes.
    /// Used as a worker tool by the cloud manager.
    /// In `cloud_only` mode, registers proxy-prefixed tools.
    pub fn build_reasoning_worker(&self, wt_path: &Path) -> OaiAgent {
        coder::build_reasoning_worker_named(
            &self.clients.reasoning,
            &self.config.reasoning_endpoint.model,
            wt_path,
            "reasoning_worker",
            self.config.cloud_only,
        )
    }

    /// Build the blind reviewer (Qwen3.5-397B on vasp-02).
    pub fn build_reviewer(&self) -> OaiAgent {
        reviewer::build_reviewer(&self.clients.local, &self.config.fast_endpoint.model)
    }

    /// Build the planner specialist.
    ///
    /// Read-only tools for analysis. Produces structured JSON repair plans.
    pub fn build_planner(&self, wt_path: &Path) -> OaiAgent {
        specialists::build_planner_named(
            &self.clients.reasoning,
            &self.config.reasoning_endpoint.model,
            wt_path,
            "planner",
            self.config.cloud_only,
        )
    }

    /// Build the fixer specialist.
    ///
    /// Full editing tools. Takes a plan and implements it step by step.
    pub fn build_fixer(&self, wt_path: &Path) -> OaiAgent {
        specialists::build_fixer_named(
            &self.clients.local,
            &self.config.fast_endpoint.model,
            wt_path,
            "fixer",
            self.config.cloud_only,
        )
    }

    /// Build the adversarial breaker agent.
    ///
    /// Red-teams the implementation after verifier passes.
    /// Writes adversarial test files and runs them to find correctness bugs.
    pub fn build_breaker(&self, wt_path: &Path) -> OaiAgent {
        adversary::build_breaker_named(
            &self.clients.local,
            &self.config.fast_endpoint.model,
            wt_path,
            "breaker",
            self.config.cloud_only,
        )
    }

    /// Build the Manager agent.
    ///
    /// When cloud is available: Cloud model (Opus 4.6) manages local workers
    /// including Qwen3.5-Architect as a reasoning tool and planner/fixer specialists.
    /// Worker agents are registered with `proxy_` prefixed names to work around
    /// the CLIAPIProxy tool name prefixing behavior.
    ///
    /// Fallback: Qwen3.5-Architect (vasp-01) manages coders directly (no reasoning worker).
    /// No proxy prefix needed — local models don't mangle tool names.
    pub fn build_manager(&self, wt_path: &Path) -> OaiAgent {
        if let Some(ref cloud_client) = self.clients.cloud {
            let cloud_ep = self.config.cloud_endpoint.as_ref().unwrap();
            info!(
                model = %cloud_ep.model,
                "Building cloud-backed manager with proxy-prefixed workers"
            );
            // Workers get proxy_ prefix for CLIAPIProxy compatibility.
            // proxy_tools=true so tool names match after CLIAPIProxy prefixing.
            let planner = specialists::build_planner_named(
                &self.clients.reasoning,
                &self.config.reasoning_endpoint.model,
                wt_path,
                "proxy_planner",
                true,
            );
            let fixer = specialists::build_fixer_named(
                &self.clients.local,
                &self.config.fast_endpoint.model,
                wt_path,
                "proxy_fixer",
                true,
            );
            let rust_coder = coder::build_rust_coder_named(
                &self.clients.local,
                &self.config.fast_endpoint.model,
                wt_path,
                "proxy_rust_coder",
                true,
            );
            let general_coder = coder::build_general_coder_named(
                &self.clients.coder,
                &self.config.coder_endpoint.model,
                wt_path,
                "proxy_general_coder",
                true,
            );
            let reviewer = reviewer::build_reviewer_named(
                &self.clients.local,
                &self.config.fast_endpoint.model,
                "proxy_reviewer",
            );
            let reasoning_worker = coder::build_reasoning_worker_named(
                &self.clients.reasoning,
                &self.config.reasoning_endpoint.model,
                wt_path,
                "proxy_reasoning_worker",
                true,
            );
            let workers = manager::ManagerWorkers {
                rust_coder,
                general_coder,
                reviewer,
                planner,
                fixer,
                reasoning_worker: Some(reasoning_worker),
                notebook_bridge: self.notebook_bridge.clone(),
            };
            manager::build_cloud_manager(
                cloud_client,
                &cloud_ep.model,
                workers,
                wt_path,
                &self.config.verifier_packages,
            )
        } else {
            info!(
                model = %self.config.reasoning_endpoint.model,
                "No cloud endpoint — building local manager (Qwen3.5-Architect)"
            );
            // Local manager doesn't need proxy prefix
            let planner = self.build_planner(wt_path);
            let fixer = self.build_fixer(wt_path);
            let rust_coder = self.build_rust_coder(wt_path);
            let general_coder = self.build_general_coder(wt_path);
            let reviewer = self.build_reviewer();
            let workers = manager::ManagerWorkers {
                rust_coder,
                general_coder,
                reviewer,
                planner,
                fixer,
                reasoning_worker: None,
                notebook_bridge: self.notebook_bridge.clone(),
            };
            manager::build_local_manager(
                &self.clients.reasoning,
                &self.config.reasoning_endpoint.model,
                workers,
                wt_path,
                &self.config.verifier_packages,
            )
        }
    }
}
