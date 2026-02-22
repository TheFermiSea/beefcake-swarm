//! Agent builders for the Manager-Worker swarm.
//!
//! Hierarchy (when cloud available):
//!   Cloud Manager (Opus 4.6) → Local Workers (OR1-Behemoth, strand-14B, Qwen3-Coder-Next)
//!
//! Fallback (no cloud):
//!   Local Manager (OR1-Behemoth) → Local Workers (strand-14B, Qwen3-Coder-Next)

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
}

impl AgentFactory {
    pub fn new(config: &SwarmConfig) -> Result<Self> {
        let clients = ClientSet::from_config(config)?;
        Ok(Self {
            clients,
            config: config.clone(),
            notebook_bridge: None,
        })
    }

    /// Set the knowledge base for the notebook tool.
    pub fn with_notebook_bridge(mut self, kb: Arc<dyn KnowledgeBase>) -> Self {
        self.notebook_bridge = Some(kb);
        self
    }

    /// Build the Rust specialist coder (strand-14B on vasp-02).
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

    /// Build the general coder (Qwen3-Coder-Next on vasp-02).
    ///
    /// In `cloud_only` mode, registers proxy-prefixed tools since all clients
    /// route through CLIAPIProxy which mangles tool names.
    pub fn build_general_coder(&self, wt_path: &Path) -> OaiAgent {
        coder::build_general_coder_named(
            &self.clients.local,
            &self.config.coder_endpoint.model,
            wt_path,
            "general_coder",
            self.config.cloud_only,
        )
    }

    /// Build the reasoning worker (OR1-Behemoth on vasp-01).
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

    /// Build the blind reviewer (strand-14B on vasp-02).
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
    /// including OR1-Behemoth as a reasoning tool and planner/fixer specialists.
    /// Worker agents are registered with `proxy_` prefixed names to work around
    /// the CLIAPIProxy tool name prefixing behavior.
    ///
    /// Fallback: OR1-Behemoth manages coders directly (no reasoning worker).
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
                &self.clients.local,
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
                "No cloud endpoint — building local manager (OR1-Behemoth)"
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
