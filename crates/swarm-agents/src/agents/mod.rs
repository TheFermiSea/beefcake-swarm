//! Agent builders for the Manager-Worker swarm.
//!
//! Hierarchy (when cloud available):
//!   Cloud Manager (Opus 4.6) → Local Workers (OR1-Behemoth, strand-14B, Qwen3-Coder-Next)
//!
//! Fallback (no cloud):
//!   Local Manager (OR1-Behemoth) → Local Workers (strand-14B, Qwen3-Coder-Next)

pub mod cloud;
pub mod coder;
pub mod manager;
pub mod reviewer;

use std::path::Path;

use anyhow::Result;
use tracing::info;

use crate::config::{ClientSet, SwarmConfig};
use coder::OaiAgent;

/// Factory that builds all agents from a `SwarmConfig`.
///
/// Holds pre-built `ClientSet` and config references needed to construct
/// agents scoped to a particular worktree path.
pub struct AgentFactory {
    pub clients: ClientSet,
    pub config: SwarmConfig,
}

impl AgentFactory {
    pub fn new(config: &SwarmConfig) -> Result<Self> {
        let clients = ClientSet::from_config(config)?;
        Ok(Self {
            clients,
            config: config.clone(),
        })
    }

    /// Build the Rust specialist coder (strand-14B on vasp-02).
    pub fn build_rust_coder(&self, wt_path: &Path) -> OaiAgent {
        coder::build_rust_coder(
            &self.clients.local,
            &self.config.fast_endpoint.model,
            wt_path,
        )
    }

    /// Build the general coder (Qwen3-Coder-Next on vasp-02).
    pub fn build_general_coder(&self, wt_path: &Path) -> OaiAgent {
        coder::build_general_coder(
            &self.clients.local,
            &self.config.coder_endpoint.model,
            wt_path,
        )
    }

    /// Build the reasoning worker (OR1-Behemoth on vasp-01).
    ///
    /// Tool-equipped agent for deep analysis and complex fixes.
    /// Used as a worker tool by the cloud manager.
    pub fn build_reasoning_worker(&self, wt_path: &Path) -> OaiAgent {
        coder::build_reasoning_worker(
            &self.clients.reasoning,
            &self.config.reasoning_endpoint.model,
            wt_path,
        )
    }

    /// Build the blind reviewer (strand-14B on vasp-02).
    pub fn build_reviewer(&self) -> OaiAgent {
        reviewer::build_reviewer(&self.clients.local, &self.config.fast_endpoint.model)
    }

    /// Build the Manager agent.
    ///
    /// When cloud is available: Cloud model (Opus 4.6) manages local workers
    /// including OR1-Behemoth as a reasoning tool.
    ///
    /// Fallback: OR1-Behemoth manages coders directly (no reasoning worker).
    pub fn build_manager(&self, wt_path: &Path) -> OaiAgent {
        let rust_coder = self.build_rust_coder(wt_path);
        let general_coder = self.build_general_coder(wt_path);
        let reviewer = self.build_reviewer();

        if let Some(ref cloud_client) = self.clients.cloud {
            let cloud_ep = self.config.cloud_endpoint.as_ref().unwrap();
            info!(
                model = %cloud_ep.model,
                "Building cloud-backed manager with local workers"
            );
            let reasoning_worker = self.build_reasoning_worker(wt_path);
            manager::build_cloud_manager(
                cloud_client,
                &cloud_ep.model,
                reasoning_worker,
                rust_coder,
                general_coder,
                reviewer,
                wt_path,
            )
        } else {
            info!(
                model = %self.config.reasoning_endpoint.model,
                "No cloud endpoint — building local manager (OR1-Behemoth)"
            );
            manager::build_local_manager(
                &self.clients.reasoning,
                &self.config.reasoning_endpoint.model,
                rust_coder,
                general_coder,
                reviewer,
                wt_path,
            )
        }
    }
}
