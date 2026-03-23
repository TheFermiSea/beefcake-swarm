//! Agent builders for the Manager-Worker swarm.
//!
//! Specialized ensemble topology:
//!   vasp-03:8081 — Scout/Reviewer (Qwen3-Coder-Next, 80B/3B MoE expert-offload, 65K context)
//!   vasp-01:8081 — Integrator/RPC head (Qwen3.5-122B-A10B MoE, layer-split with vasp-02, 128K context)
//!   vasp-02     — RPC worker shard for vasp-01 (NO independent HTTP endpoint)
//!
//! Hierarchy (when cloud available):
//!   Cloud Manager (Opus 4.6) → Local Workers (Qwen3.5-122B-A10B on vasp-01, Qwen3.5-27B on vasp-03)
//!
//! Fallback (no cloud):
//!   Local Manager (Qwen3.5-122B-A10B on vasp-01) → Local Workers (Qwen3.5-27B on vasp-03)

pub mod adversary;
pub mod cloud;
pub mod coder;
pub mod manager;
pub mod reviewer;
pub mod specialists;

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use rig::providers::openai;
use tracing::info;

use crate::config::{ClientSet, SwarmConfig, SwarmRole};
use crate::endpoint_pool::EndpointPool;
use crate::notebook_bridge::KnowledgeBase;
use crate::tools::plan_parallel_tool::PlanSlot;
use crate::tools::shared::ToolFactory;
use coder::OaiAgent;

/// Factory that builds all agents from a `SwarmConfig`.
///
/// Holds pre-built `ClientSet` and config references needed to construct
/// agents scoped to a particular worktree path.
///
/// Cloning `AgentFactory` is cheap: config is Arc-wrapped, tools are reference-counted.
/// Parallel tasks should each hold a clone — the shared `EndpointPool` counter
/// ensures each clone naturally selects the next node in round-robin order.
#[derive(Clone)]
pub struct AgentFactory {
    pub clients: ClientSet,
    pub config: Arc<SwarmConfig>,
    /// Shared knowledge base for the notebook tool (None if unavailable).
    pub notebook_bridge: Option<Arc<dyn KnowledgeBase>>,
    /// Centralized tool construction factory, scoped to a worktree.
    ///
    /// When set, agent builders can use this instead of calling
    /// `bundles::worker_tools()` / `bundles::kernel_strategy_tools()` directly.
    /// Initialize via [`AgentFactory::with_worktree`].
    pub tool_factory: Option<ToolFactory>,
    /// Round-robin pool for selecting the next worker node.
    ///
    /// Cloning `AgentFactory` shares this pool's `Arc<AtomicUsize>` counter,
    /// so parallel issue tasks automatically cycle across different nodes.
    pub endpoint_pool: EndpointPool,
    /// Shared slot where the manager deposits a parallel work plan.
    /// Set via [`AgentFactory::with_plan_slot`] before building the manager.
    /// The orchestrator checks this after each manager invocation.
    pub plan_slot: Option<PlanSlot>,
    /// Shared slot where the manager deposits a work plan before delegating.
    /// Set via [`AgentFactory::with_work_plan_slot`] before building the manager.
    /// Enables the ClawTeam-style plan-before-execute gate.
    pub work_plan_slot: Option<crate::tools::submit_plan_tool::WorkPlanSlot>,
}

impl AgentFactory {
    pub fn new(config: &SwarmConfig) -> Result<Self> {
        let clients = ClientSet::from_config(config)?;
        let endpoint_pool = EndpointPool::new(&clients, config);
        Ok(Self {
            clients,
            config: Arc::new(config.clone()),
            notebook_bridge: None,
            tool_factory: None,
            endpoint_pool,
            plan_slot: None,
            work_plan_slot: None,
        })
    }

    /// Attach a health monitor to the endpoint pool for health-aware routing.
    ///
    /// When set, `endpoint_pool.next()` will skip endpoints that are marked
    /// as `Down` by the health monitor, falling back to round-robin when all
    /// endpoints are down or health data is unavailable.
    pub fn with_health(mut self, health: crate::cluster_health::ClusterHealth) -> Self {
        self.endpoint_pool = self.endpoint_pool.with_health(health);
        self
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

    /// Set a shared plan slot for manager-guided parallel work planning.
    ///
    /// When set, `build_manager()` includes the `plan_parallel_work` tool.
    /// The orchestrator checks this slot after each manager invocation for
    /// a submitted subtask plan.
    pub fn with_plan_slot(mut self, slot: PlanSlot) -> Self {
        self.plan_slot = Some(slot);
        self
    }

    /// Set a shared work plan slot for the plan-before-execute gate.
    ///
    /// When set, `build_manager()` includes the `submit_plan` tool.
    /// The orchestrator injects the captured plan into subsequent iteration prompts.
    pub fn with_work_plan_slot(
        mut self,
        slot: crate::tools::submit_plan_tool::WorkPlanSlot,
    ) -> Self {
        self.work_plan_slot = Some(slot);
        self
    }

    /// Return the cluster health monitor attached to the endpoint pool, if any.
    pub fn cluster_health(&self) -> Option<&crate::cluster_health::ClusterHealth> {
        self.endpoint_pool.cluster_health()
    }

    /// Initialize the centralized tool factory for a given worktree path.
    ///
    /// Once set, the `tool_factory` field is available for callers that want
    /// centralized, Clone-able tool construction instead of calling
    /// `bundles::worker_tools()` / `bundles::kernel_strategy_tools()` directly.
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

    /// Resolve the appropriate client and model for a given swarm role.
    ///
    /// Respects the active [`SwarmStackProfile`] and utilizes the [`EndpointPool`]
    /// for roles that should be distributed across the cluster (GeneralWorker, etc.).
    pub fn resolve_role_endpoint(&self, role: SwarmRole) -> (openai::CompletionsClient, String) {
        use crate::config::SwarmStackProfile;

        // In cloud-only mode, bypass local routing and pool logic entirely.
        if self.config.cloud_only {
            if let Some(ref cloud_ep) = self.config.cloud_endpoint {
                return (
                    self.clients.cloud.clone().expect("cloud client missing"),
                    cloud_ep.model.clone(),
                );
            }
        }

        // Determine if this role should use the round-robin worker pool.
        let use_pool = match self.config.stack_profile {
            SwarmStackProfile::HybridBalancedV1 => matches!(
                role,
                SwarmRole::GeneralWorker
                    | SwarmRole::Planner
                    | SwarmRole::ReasoningWorker
                    | SwarmRole::LocalManagerFallback
            ),
            SwarmStackProfile::SmallSpecialistV1 => matches!(
                role,
                SwarmRole::GeneralWorker
                    | SwarmRole::ReasoningWorker
                    | SwarmRole::LocalManagerFallback
            ),
            SwarmStackProfile::StrategistHybridV1 => matches!(
                role,
                SwarmRole::GeneralWorker
                    | SwarmRole::Planner
                    | SwarmRole::ReasoningWorker
                    | SwarmRole::LocalManagerFallback
            ),
        };

        if use_pool {
            let (client, model) = self.endpoint_pool.next();
            (client.clone(), model.to_string())
        } else {
            // resolve_role_client/model are guaranteed to return valid results
            // if the config was validated at startup.
            let client = self
                .config
                .resolve_role_client(role, &self.clients)
                .expect("Failed to resolve client for role");
            let model = self.config.resolve_role_model(role).to_string();
            (client, model)
        }
    }

    /// Build the Rust specialist coder (Qwen3.5-27B-Distilled on vasp-03, Rust system prompt).
    ///
    /// In `cloud_only` mode, registers proxy-prefixed tools since all clients
    /// route through CLIAPIProxy which mangles tool names.
    pub fn build_rust_coder(&self, wt_path: &Path) -> OaiAgent {
        let (client, model) = self.resolve_role_endpoint(SwarmRole::RustWorker);
        coder::build_rust_coder_named(
            &client,
            &model,
            wt_path,
            "rust_coder",
            self.config.cloud_only,
        )
    }

    /// Build the general coder (Qwen3.5-122B-A10B on the integrator tier, general coding system prompt).
    ///
    /// In `cloud_only` mode, registers proxy-prefixed tools since all clients
    /// route through CLIAPIProxy which mangles tool names.
    pub fn build_general_coder(&self, wt_path: &Path) -> OaiAgent {
        let (client, model) = self.resolve_role_endpoint(SwarmRole::GeneralWorker);
        coder::build_general_coder_named(
            &client,
            &model,
            wt_path,
            "general_coder",
            self.config.cloud_only,
        )
    }

    /// Build the specialized worker pair for a single issue.
    ///
    /// The Rust specialist stays pinned to the fast 27B scout tier for focused
    /// Rust repairs, while the general coder uses the worker endpoint pool so
    /// concurrent issues still spread across the integrator tier.
    pub fn build_worker_pair(&self, wt_path: &Path) -> (OaiAgent, OaiAgent) {
        let rust_coder = self.build_rust_coder(wt_path);
        let general_coder = self.build_general_coder(wt_path);
        (rust_coder, general_coder)
    }

    /// Build the reasoning worker (Qwen3.5-122B-A10B on vasp-01, Architect node).
    ///
    /// Tool-equipped agent for deep analysis and complex fixes.
    /// Used as a worker tool by the cloud manager.
    /// In `cloud_only` mode, registers proxy-prefixed tools.
    pub fn build_reasoning_worker(&self, wt_path: &Path) -> OaiAgent {
        let (client, model) = self.resolve_role_endpoint(SwarmRole::ReasoningWorker);
        coder::build_reasoning_worker_named(
            &client,
            &model,
            wt_path,
            "reasoning_worker",
            self.config.cloud_only,
        )
    }

    /// Build the blind reviewer (Qwen3.5-27B-Distilled on vasp-03).
    pub fn build_reviewer(&self) -> OaiAgent {
        let (client, model) = self.resolve_role_endpoint(SwarmRole::Reviewer);
        reviewer::build_reviewer(&client, &model)
    }

    /// Build the strategist advisor (Qwen3.5-397B-A17B).
    pub fn build_strategist(&self, wt_path: &Path) -> OaiAgent {
        let (client, model) = self.resolve_role_endpoint(SwarmRole::Strategist);
        coder::build_strategist_named(
            &client,
            &model,
            wt_path,
            "strategist",
            self.config.cloud_only,
        )
    }

    /// Build the planner specialist.
    ///
    /// Read-only tools for analysis. Produces structured JSON repair plans.
    pub fn build_planner(&self, wt_path: &Path) -> OaiAgent {
        let (client, model) = self.resolve_role_endpoint(SwarmRole::Planner);
        specialists::build_planner_named(
            &client,
            &model,
            wt_path,
            "planner",
            self.config.cloud_only,
        )
    }

    /// Build the fixer specialist.
    ///
    /// Full editing tools. Takes a plan and implements it step by step.
    pub fn build_fixer(&self, wt_path: &Path) -> OaiAgent {
        let (client, model) = self.resolve_role_endpoint(SwarmRole::Fixer);
        specialists::build_fixer_named(&client, &model, wt_path, "fixer", self.config.cloud_only)
    }

    /// Build the adversarial breaker agent.
    ///
    /// Red-teams the implementation after verifier passes.
    /// Writes adversarial test files and runs them to find correctness bugs.
    pub fn build_breaker(&self, wt_path: &Path) -> OaiAgent {
        let (client, model) = self.resolve_role_endpoint(SwarmRole::Reviewer);
        adversary::build_breaker_named(&client, &model, wt_path, "breaker", self.config.cloud_only)
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
        // Resolve strategist if available for this profile
        let strategist = self.clients.strategist.as_ref().map(|client| {
            let model = self.config.resolve_role_model(SwarmRole::Strategist);
            coder::build_strategist_named(
                client,
                &model,
                wt_path,
                if self.clients.cloud.is_some() {
                    "proxy_strategist"
                } else {
                    "strategist"
                },
                self.clients.cloud.is_some(),
            )
        });

        // Prefer TZ gateway for the manager; fall back to direct CLIAPIProxy.
        // cloud_tz is only set when SWARM_TENSORZERO_URL is configured.
        let active_cloud = self
            .clients
            .cloud_tz
            .as_ref()
            .or(self.clients.cloud.as_ref());
        if let Some(cloud_client) = active_cloud {
            let cloud_ep = self.config.cloud_endpoint.as_ref().unwrap();
            let manager_model = if self.config.tensorzero_url.is_some() {
                "tensorzero::function_name::cloud_manager_delegation"
            } else {
                cloud_ep.model.as_str()
            };
            info!(
                model = %manager_model,
                tensorzero = self.config.tensorzero_url.is_some(),
                "Building cloud-backed manager with proxy-prefixed workers"
            );
            // Workers get proxy_ prefix for CLIAPIProxy compatibility.
            // proxy_tools=true so tool names match after CLIAPIProxy prefixing.
            let (p_client, p_model) = self.resolve_role_endpoint(SwarmRole::Planner);
            let planner = specialists::build_planner_named(
                &p_client,
                &p_model,
                wt_path,
                "proxy_planner",
                true,
            );

            let (f_client, f_model) = self.resolve_role_endpoint(SwarmRole::Fixer);
            let fixer =
                specialists::build_fixer_named(&f_client, &f_model, wt_path, "proxy_fixer", true);

            let (r_client, r_model) = self.resolve_role_endpoint(SwarmRole::RustWorker);
            let rust_coder = coder::build_rust_coder_named(
                &r_client,
                &r_model,
                wt_path,
                "proxy_rust_coder",
                true,
            );

            let (g_client, g_model) = self.resolve_role_endpoint(SwarmRole::GeneralWorker);
            let general_coder = coder::build_general_coder_named(
                &g_client,
                &g_model,
                wt_path,
                "proxy_general_coder",
                true,
            );

            let (rev_client, rev_model) = self.resolve_role_endpoint(SwarmRole::Reviewer);
            let reviewer = reviewer::build_reviewer_named(
                &rev_client,
                &rev_model,
                "proxy_reviewer",
                Some(wt_path),
            );

            let (reas_client, reas_model) = self.resolve_role_endpoint(SwarmRole::ReasoningWorker);
            let reasoning_worker = coder::build_reasoning_worker_named(
                &reas_client,
                &reas_model,
                wt_path,
                "proxy_reasoning_worker",
                true,
            );
            // Architect/Editor pattern: Architect runs on cloud (deep understanding),
            // Editor runs on local 27B (fast mechanical edits).
            let architect = {
                let (a_client, a_model) = self.resolve_role_endpoint(SwarmRole::Council);
                specialists::build_architect_named(
                    &a_client,
                    &a_model,
                    wt_path,
                    "proxy_architect",
                    true,
                )
            };
            let editor = {
                let (e_client, e_model) = self.resolve_role_endpoint(SwarmRole::Scout);
                specialists::build_editor_named(&e_client, &e_model, wt_path, "proxy_editor", true)
            };

            let workers = manager::ManagerWorkers {
                rust_coder,
                general_coder,
                reviewer,
                planner,
                fixer,
                architect: Some(architect),
                editor: Some(editor),
                reasoning_worker: Some(reasoning_worker),
                strategist,
                notebook_bridge: self.notebook_bridge.clone(),
                plan_slot: self.plan_slot.clone(),
                work_plan_slot: self.work_plan_slot.clone(),
            };
            manager::build_cloud_manager(
                cloud_client,
                manager_model,
                workers,
                wt_path,
                &self.config.verifier_packages,
            )
        } else {
            let (m_client, m_model) = self.resolve_role_endpoint(SwarmRole::LocalManagerFallback);
            info!(
                model = %m_model,
                "No cloud endpoint — building local manager"
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
                architect: None, // Architect needs cloud model
                editor: None,    // Editor only useful with Architect
                reasoning_worker: None,
                strategist,
                notebook_bridge: self.notebook_bridge.clone(),
                plan_slot: self.plan_slot.clone(),
                work_plan_slot: self.work_plan_slot.clone(),
            };
            manager::build_local_manager(
                &m_client,
                &m_model,
                workers,
                wt_path,
                &self.config.verifier_packages,
            )
        }
    }

    /// Build a cloud manager using a specific fallback model name.
    ///
    /// Used by the orchestrator when the primary cloud model fails (429/5xx/quota)
    /// and we need to try the next model in the CloudFallbackMatrix.
    /// Returns `None` if no cloud client is available.
    pub fn build_manager_for_model(&self, wt_path: &Path, model: &str) -> Option<OaiAgent> {
        let cloud_client = self.clients.cloud.as_ref()?;
        info!(
            model = %model,
            "Building cloud manager with fallback model"
        );

        let strategist = self.clients.strategist.as_ref().map(|client| {
            let model = self.config.resolve_role_model(SwarmRole::Strategist);
            coder::build_strategist_named(client, &model, wt_path, "proxy_strategist", true)
        });

        let (p_client, p_model) = self.resolve_role_endpoint(SwarmRole::Planner);
        let planner =
            specialists::build_planner_named(&p_client, &p_model, wt_path, "proxy_planner", true);

        let (f_client, f_model) = self.resolve_role_endpoint(SwarmRole::Fixer);
        let fixer =
            specialists::build_fixer_named(&f_client, &f_model, wt_path, "proxy_fixer", true);

        let (r_client, r_model) = self.resolve_role_endpoint(SwarmRole::RustWorker);
        let rust_coder =
            coder::build_rust_coder_named(&r_client, &r_model, wt_path, "proxy_rust_coder", true);

        let (g_client, g_model) = self.resolve_role_endpoint(SwarmRole::GeneralWorker);
        let general_coder = coder::build_general_coder_named(
            &g_client,
            &g_model,
            wt_path,
            "proxy_general_coder",
            true,
        );

        let (rev_client, rev_model) = self.resolve_role_endpoint(SwarmRole::Reviewer);
        let reviewer = reviewer::build_reviewer_named(
            &rev_client,
            &rev_model,
            "proxy_reviewer",
            Some(wt_path),
        );

        let (reas_client, reas_model) = self.resolve_role_endpoint(SwarmRole::ReasoningWorker);
        let reasoning_worker = coder::build_reasoning_worker_named(
            &reas_client,
            &reas_model,
            wt_path,
            "proxy_reasoning_worker",
            true,
        );
        // Architect/Editor for this cloud manager path too
        let architect = specialists::build_architect_named(
            cloud_client,
            model,
            wt_path,
            "proxy_architect",
            true,
        );
        let editor = {
            let (e_client, e_model) = self.resolve_role_endpoint(SwarmRole::Scout);
            specialists::build_editor_named(&e_client, &e_model, wt_path, "proxy_editor", true)
        };

        let workers = manager::ManagerWorkers {
            rust_coder,
            general_coder,
            reviewer,
            planner,
            fixer,
            architect: Some(architect),
            editor: Some(editor),
            reasoning_worker: Some(reasoning_worker),
            strategist,
            notebook_bridge: self.notebook_bridge.clone(),
            plan_slot: self.plan_slot.clone(),
            work_plan_slot: self.work_plan_slot.clone(),
        };
        Some(manager::build_cloud_manager(
            cloud_client,
            model,
            workers,
            wt_path,
            &self.config.verifier_packages,
        ))
    }
}
