//! Agent builders for the Manager-Worker swarm.
//!
//! Each agent is built via a free function that returns `Agent<openai::completion::CompletionModel>`.
//! The `AgentFactory` ties them together using `ClientSet` and `SwarmConfig`.

pub mod cloud;
pub mod coder;
pub mod manager;
pub mod reviewer;

use std::path::Path;

use anyhow::Result;

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

    /// Build the blind reviewer (strand-14B on vasp-02).
    pub fn build_reviewer(&self) -> OaiAgent {
        reviewer::build_reviewer(&self.clients.local, &self.config.fast_endpoint.model)
    }

    /// Build the Manager orchestrator (OR1-Behemoth on vasp-01).
    ///
    /// The Manager receives worker agents as tools. Workers are scoped to
    /// the given worktree path.
    pub fn build_manager(&self, wt_path: &Path) -> OaiAgent {
        let rust_coder = self.build_rust_coder(wt_path);
        let general_coder = self.build_general_coder(wt_path);
        let reviewer = self.build_reviewer();

        manager::build_manager(
            &self.clients.reasoning,
            &self.config.reasoning_endpoint.model,
            rust_coder,
            general_coder,
            reviewer,
            wt_path,
        )
    }

    /// Build the cloud escalation agent (CLIAPIProxy, optional).
    pub fn build_cloud_agent(&self) -> Option<OaiAgent> {
        self.config
            .cloud_endpoint
            .as_ref()
            .and_then(|ep| cloud::build_cloud_agent(ep).ok())
    }
}
