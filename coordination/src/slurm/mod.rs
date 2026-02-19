//! SLURM Lifecycle Management for AI Inference Jobs
//!
//! Manages llama.cpp inference server jobs on the beefcake2 cluster.
//! Handles job submission, health checking, and preemption recovery.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use thiserror::Error;

/// Inference tier for model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InferenceTier {
    /// HydraCoder worker on vasp-02
    Worker,
    /// Qwen3.5 distributed manager on vasp-01+vasp-03
    ManagerLocal,
}

impl InferenceTier {
    pub fn job_script(&self) -> &'static str {
        match self {
            Self::Worker => "run-worker.slurm",
            Self::ManagerLocal => "run-qwen35-distributed.slurm",
        }
    }

    pub fn model_name(&self) -> &'static str {
        match self {
            Self::Worker => "HydraCoder",
            Self::ManagerLocal => "Qwen3.5",
        }
    }

    pub fn expected_tok_per_sec(&self) -> (u32, u32) {
        match self {
            Self::Worker => (30, 50),
            Self::ManagerLocal => (5, 12),
        }
    }
}

impl std::fmt::Display for InferenceTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Worker => write!(f, "worker"),
            Self::ManagerLocal => write!(f, "manager-local"),
        }
    }
}

/// Error types for SLURM operations
#[derive(Debug, Error)]
pub enum SlurmError {
    #[error("SLURM command failed: {0}")]
    CommandFailed(String),

    #[error("Job submission failed: {0}")]
    SubmitFailed(String),

    #[error("Job not found: {0}")]
    JobNotFound(u32),

    #[error("Health check failed: {0}")]
    HealthCheckFailed(String),

    #[error("Endpoint not ready after {0:?}")]
    EndpointTimeout(Duration),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Parse error: {0}")]
    Parse(String),
}

/// SLURM job state
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum JobState {
    Pending,
    Running,
    Completing,
    Completed,
    Failed,
    Cancelled,
    Timeout,
    NodeFail,
    Preempted,
    Suspended,
    Unknown,
}

impl From<&str> for JobState {
    fn from(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "PENDING" | "PD" => Self::Pending,
            "RUNNING" | "R" => Self::Running,
            "COMPLETING" | "CG" => Self::Completing,
            "COMPLETED" | "CD" => Self::Completed,
            "FAILED" | "F" => Self::Failed,
            "CANCELLED" | "CA" => Self::Cancelled,
            "TIMEOUT" | "TO" => Self::Timeout,
            "NODE_FAIL" | "NF" => Self::NodeFail,
            "PREEMPTED" | "PR" => Self::Preempted,
            "SUSPENDED" | "S" => Self::Suspended,
            _ => Self::Unknown,
        }
    }
}

/// Information about a running inference endpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointInfo {
    pub job_id: u32,
    pub model: String,
    pub tier: String,
    pub node: String,
    pub host: String,
    pub port: u16,
    pub endpoint: String,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head_node: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rpc_workers: Option<Vec<String>>,
}

/// Health state of an inference endpoint
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointHealth {
    Healthy,
    Degraded,
    Unhealthy,
    Stale,
}

/// Detailed endpoint health report
#[derive(Debug, Clone)]
pub struct EndpointHealthDetails {
    pub state: EndpointHealth,
    pub job_state: Option<JobState>,
    pub attempts: u32,
    pub healthy_workers: Vec<String>,
    pub unhealthy_workers: Vec<String>,
    pub last_error: Option<String>,
    pub timed_out: bool,
}

/// SLURM job information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobInfo {
    pub job_id: u32,
    pub name: String,
    pub state: JobState,
    pub node_list: Option<String>,
    pub partition: String,
    pub time_used: String,
    pub time_limit: String,
}

/// Configuration for the SLURM inference manager
#[derive(Debug, Clone)]
pub struct SlurmConfig {
    /// Path to SLURM job scripts
    pub scripts_path: PathBuf,
    /// Path to endpoint discovery directory
    pub endpoints_path: PathBuf,
    /// SSH host for SLURM controller (if remote)
    pub slurm_host: Option<String>,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Maximum wait time for endpoint to be ready
    pub endpoint_timeout: Duration,
    /// Health check configuration
    pub health_check: HealthCheckConfig,
}

impl Default for SlurmConfig {
    fn default() -> Self {
        Self {
            // Use environment variables with sensible defaults
            // SLURM_SCRIPTS_PATH: Path to llama.cpp job scripts
            scripts_path: std::env::var("SLURM_SCRIPTS_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/cluster/shared/scripts/llama-cpp")),
            // SLURM_ENDPOINTS_PATH: Path to endpoint discovery directory (must be on shared filesystem)
            endpoints_path: std::env::var("SLURM_ENDPOINTS_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|_| PathBuf::from("/cluster/shared/ai/endpoints")),
            // SLURM_HOST: SSH host for SLURM controller (empty or "local" for direct execution)
            slurm_host: std::env::var("SLURM_HOST")
                .ok()
                .filter(|s| !s.is_empty() && s != "local")
                .or_else(|| Some("slurm-ctl".to_string())),
            // SLURM_HEALTH_CHECK_INTERVAL: Seconds between health checks
            health_check_interval: std::env::var("SLURM_HEALTH_CHECK_INTERVAL")
                .ok()
                .and_then(|s| s.parse().ok())
                .map(Duration::from_secs)
                .unwrap_or_else(|| Duration::from_secs(10)),
            // SLURM_ENDPOINT_TIMEOUT: Seconds to wait for endpoint to be ready
            endpoint_timeout: std::env::var("SLURM_ENDPOINT_TIMEOUT")
                .ok()
                .and_then(|s| s.parse().ok())
                .map(Duration::from_secs)
                .unwrap_or_else(|| Duration::from_secs(120)),
            health_check: HealthCheckConfig::default(),
        }
    }
}

/// Health check timeouts and retry configuration
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// TCP connect timeout
    pub connect_timeout: Duration,
    /// Response timeout for /health
    pub response_timeout: Duration,
    /// Retry attempts for health checks
    pub max_retries: u32,
    /// Delay between retries
    pub retry_backoff: Duration,
    /// Minimum number of RPC workers that must be healthy
    pub min_rpc_workers_healthy: usize,
    /// Consecutive degraded checks before recycling
    pub max_consecutive_degraded: u32,
    /// Maximum recovery attempts before forcing job restart
    pub max_recovery_attempts: u32,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        let connect_timeout = std::env::var("SLURM_HEALTH_CONNECT_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(2));

        let response_timeout = std::env::var("SLURM_HEALTH_RESPONSE_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(4));

        let max_retries = std::env::var("SLURM_HEALTH_MAX_RETRIES")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);

        let retry_backoff = std::env::var("SLURM_HEALTH_RETRY_BACKOFF_MS")
            .ok()
            .and_then(|s| s.parse().ok())
            .map(Duration::from_millis)
            .unwrap_or_else(|| Duration::from_millis(500));

        let min_rpc_workers_healthy = std::env::var("SLURM_HEALTH_MIN_RPC_WORKERS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);

        let max_consecutive_degraded = std::env::var("SLURM_HEALTH_MAX_DEGRADED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(3);

        let max_recovery_attempts = std::env::var("SLURM_HEALTH_MAX_RECOVERY_ATTEMPTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(5);

        Self {
            connect_timeout,
            response_timeout,
            max_retries,
            retry_backoff,
            min_rpc_workers_healthy,
            max_consecutive_degraded,
            max_recovery_attempts,
        }
    }
}

/// Health check metrics counters
#[derive(Debug, Default)]
pub struct HealthCheckMetrics {
    healthy: AtomicU64,
    degraded: AtomicU64,
    unhealthy: AtomicU64,
    stale: AtomicU64,
    timeouts: AtomicU64,
    errors: AtomicU64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckMetricsSnapshot {
    pub healthy: u64,
    pub degraded: u64,
    pub unhealthy: u64,
    pub stale: u64,
    pub timeouts: u64,
    pub errors: u64,
}

impl HealthCheckMetrics {
    fn record(&self, state: EndpointHealth) {
        match state {
            EndpointHealth::Healthy => self.healthy.fetch_add(1, Ordering::Relaxed),
            EndpointHealth::Degraded => self.degraded.fetch_add(1, Ordering::Relaxed),
            EndpointHealth::Unhealthy => self.unhealthy.fetch_add(1, Ordering::Relaxed),
            EndpointHealth::Stale => self.stale.fetch_add(1, Ordering::Relaxed),
        };
    }

    fn record_timeout(&self) {
        self.timeouts.fetch_add(1, Ordering::Relaxed);
    }

    fn record_error(&self) {
        self.errors.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot(&self) -> HealthCheckMetricsSnapshot {
        HealthCheckMetricsSnapshot {
            healthy: self.healthy.load(Ordering::Relaxed),
            degraded: self.degraded.load(Ordering::Relaxed),
            unhealthy: self.unhealthy.load(Ordering::Relaxed),
            stale: self.stale.load(Ordering::Relaxed),
            timeouts: self.timeouts.load(Ordering::Relaxed),
            errors: self.errors.load(Ordering::Relaxed),
        }
    }
}

/// Recovery state for partial failures
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoveryState {
    Stable,
    Degraded { consecutive: u32 },
    Recovering { attempts: u32 },
}

impl SlurmConfig {
    /// Create config from environment with validation
    pub fn from_env() -> Result<Self, SlurmError> {
        let config = Self::default();

        // Validate scripts path exists (if we can check locally)
        if config.slurm_host.is_none() && !config.scripts_path.exists() {
            tracing::warn!(
                "Scripts path does not exist locally: {:?}. \
                 Set SLURM_SCRIPTS_PATH or ensure path exists on SLURM host.",
                config.scripts_path
            );
        }

        Ok(config)
    }
}

/// Manages SLURM jobs for AI inference
pub struct SlurmInferenceManager {
    config: SlurmConfig,
    /// Currently tracked jobs by tier
    active_jobs: HashMap<InferenceTier, u32>,
    /// Cached endpoint info
    endpoint_cache: HashMap<InferenceTier, EndpointInfo>,
    /// HTTP client for health checks
    http_client: reqwest::Client,
    /// Recovery state per tier
    recovery_state: HashMap<InferenceTier, RecoveryState>,
    /// Health check metrics
    health_metrics: Arc<HealthCheckMetrics>,
}

impl SlurmInferenceManager {
    /// Create a new SLURM inference manager
    ///
    /// Returns an error if the HTTP client cannot be created.
    /// Performs startup job reconciliation to prevent duplicate submissions.
    pub fn new(config: SlurmConfig) -> Result<Self, SlurmError> {
        let http_client = reqwest::Client::builder()
            .connect_timeout(config.health_check.connect_timeout)
            .build()
            .map_err(|e| {
                SlurmError::CommandFailed(format!("Failed to create HTTP client: {}", e))
            })?;

        let mut manager = Self {
            config,
            active_jobs: HashMap::new(),
            endpoint_cache: HashMap::new(),
            http_client,
            recovery_state: HashMap::new(),
            health_metrics: Arc::new(HealthCheckMetrics::default()),
        };

        // Startup job reconciliation: discover existing inference jobs to prevent duplicates
        // This is critical for cold start scenarios where we're re-created while jobs are running
        if let Err(e) = manager.reconcile_active_jobs() {
            tracing::warn!("Failed to reconcile active jobs at startup: {}", e);
            // Continue anyway - we'll discover jobs via endpoints
        }

        Ok(manager)
    }

    /// Reconcile active jobs from SLURM queue state
    /// Called at startup and can be called manually to sync state
    pub fn reconcile_active_jobs(&mut self) -> Result<(), SlurmError> {
        let jobs = self.list_active_jobs()?;

        for job in jobs {
            let tier = if job.name.contains("worker") || job.name.contains("hydra") {
                InferenceTier::Worker
            } else if job.name.contains("qwen35") || job.name.contains("manager") {
                InferenceTier::ManagerLocal
            } else {
                continue; // Unknown job type
            };

            // Only track if running or pending (active)
            if matches!(
                job.state,
                JobState::Running | JobState::Pending | JobState::Completing
            ) {
                tracing::info!(
                    "Reconciled existing {} job {} (state: {:?})",
                    tier,
                    job.job_id,
                    job.state
                );
                self.active_jobs.insert(tier, job.job_id);
            }
        }

        tracing::info!(
            "Startup reconciliation complete: {} active jobs tracked",
            self.active_jobs.len()
        );
        Ok(())
    }

    /// Create with default configuration
    pub fn with_defaults() -> Result<Self, SlurmError> {
        Self::new(SlurmConfig::default())
    }

    /// Run a SLURM command, optionally via SSH to slurm-ctl
    fn run_slurm_cmd(&self, cmd: &str, args: &[&str]) -> Result<String, SlurmError> {
        let output = if let Some(ref host) = self.config.slurm_host {
            let full_cmd = format!("{} {}", cmd, args.join(" "));
            Command::new("ssh")
                .args([host.as_str(), &full_cmd])
                .output()?
        } else {
            Command::new(cmd).args(args).output()?
        };

        if output.status.success() {
            Ok(String::from_utf8_lossy(&output.stdout).to_string())
        } else {
            Err(SlurmError::CommandFailed(
                String::from_utf8_lossy(&output.stderr).to_string(),
            ))
        }
    }

    /// Submit a job for the given inference tier
    pub fn submit_job(&mut self, tier: InferenceTier) -> Result<u32, SlurmError> {
        let script_path = self.config.scripts_path.join(tier.job_script());

        let output = self.run_slurm_cmd(
            "sbatch",
            &["--parsable", script_path.to_str().unwrap_or("")],
        )?;

        let job_id = output
            .trim()
            .split(';')
            .next()
            .and_then(|s| s.parse::<u32>().ok())
            .ok_or_else(|| SlurmError::Parse(format!("Failed to parse job ID from: {}", output)))?;

        self.active_jobs.insert(tier, job_id);
        tracing::info!("Submitted {} job: {}", tier, job_id);

        Ok(job_id)
    }

    /// Get job information
    pub fn get_job_info(&self, job_id: u32) -> Result<JobInfo, SlurmError> {
        let output = self.run_slurm_cmd(
            "squeue",
            &[
                "-j",
                &job_id.to_string(),
                "-o",
                "%i|%j|%T|%N|%P|%M|%l",
                "--noheader",
            ],
        )?;

        let line = output.trim();
        if line.is_empty() {
            // Job not in queue, check sacct
            return self.get_completed_job_info(job_id);
        }

        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 7 {
            return Err(SlurmError::Parse(format!(
                "Unexpected squeue output: {}",
                line
            )));
        }

        Ok(JobInfo {
            job_id,
            name: parts[1].to_string(),
            state: JobState::from(parts[2]),
            node_list: if parts[3].is_empty() {
                None
            } else {
                Some(parts[3].to_string())
            },
            partition: parts[4].to_string(),
            time_used: parts[5].to_string(),
            time_limit: parts[6].to_string(),
        })
    }

    /// Get completed job info from sacct
    fn get_completed_job_info(&self, job_id: u32) -> Result<JobInfo, SlurmError> {
        let output = self.run_slurm_cmd(
            "sacct",
            &[
                "-j",
                &job_id.to_string(),
                "-o",
                "JobID,JobName,State,NodeList,Partition,Elapsed,Timelimit",
                "--noheader",
                "-P",
            ],
        )?;

        let line = output
            .lines()
            .find(|l| !l.contains('.'))
            .ok_or(SlurmError::JobNotFound(job_id))?;

        let parts: Vec<&str> = line.split('|').collect();
        if parts.len() < 7 {
            return Err(SlurmError::Parse(format!(
                "Unexpected sacct output: {}",
                line
            )));
        }

        Ok(JobInfo {
            job_id,
            name: parts[1].to_string(),
            state: JobState::from(parts[2]),
            node_list: if parts[3].is_empty() {
                None
            } else {
                Some(parts[3].to_string())
            },
            partition: parts[4].to_string(),
            time_used: parts[5].to_string(),
            time_limit: parts[6].to_string(),
        })
    }

    /// Check if a job is running and healthy
    pub fn is_job_running(&self, job_id: u32) -> Result<bool, SlurmError> {
        let info = self.get_job_info(job_id)?;
        Ok(info.state == JobState::Running)
    }

    /// Check if a job is active (running, pending, or completing)
    /// This prevents the "pending job explosion" bug where we submit duplicate jobs
    /// for jobs that are queued but not yet running.
    pub fn is_job_active(&self, job_id: u32) -> Result<bool, SlurmError> {
        let info = self.get_job_info(job_id)?;
        Ok(matches!(
            info.state,
            JobState::Running | JobState::Pending | JobState::Completing
        ))
    }

    /// Get detailed job status for smarter lifecycle decisions
    pub fn get_job_status(&self, job_id: u32) -> Result<JobState, SlurmError> {
        let info = self.get_job_info(job_id)?;
        Ok(info.state)
    }

    /// Discover endpoint for a tier by reading endpoint files
    pub fn discover_endpoint(
        &mut self,
        tier: InferenceTier,
    ) -> Result<Option<EndpointInfo>, SlurmError> {
        let pattern = match tier {
            InferenceTier::Worker => "*-worker.json",
            InferenceTier::ManagerLocal => "*-qwen35.json",
        };

        // List endpoint files
        // Use -1t to sort by modification time (newest first)
        let glob_path = self.config.endpoints_path.join(pattern);
        let output = self.run_slurm_cmd("ls", &["-1t", &glob_path.to_string_lossy()]);

        let files = match output {
            Ok(o) if !o.trim().is_empty() => o,
            Ok(_) => {
                tracing::debug!(
                    "No {} endpoints found at {:?}",
                    tier,
                    self.config.endpoints_path
                );
                return Ok(None);
            }
            // Expected "no files" errors - return Ok(None)
            Err(SlurmError::CommandFailed(ref msg)) if msg.contains("No such file") => {
                tracing::debug!("Endpoint directory or pattern not found: {:?}", glob_path);
                return Ok(None);
            }
            Err(SlurmError::CommandFailed(ref msg)) if msg.contains("cannot access") => {
                tracing::debug!("No matching endpoint files: {}", msg);
                return Ok(None);
            }
            // Unexpected errors - propagate them instead of silently swallowing
            // This catches permission errors, NFS mount issues, SSH failures, etc.
            Err(SlurmError::CommandFailed(ref msg))
                if msg.contains("Permission denied")
                    || msg.contains("Input/output error")
                    || msg.contains("Stale file handle")
                    || msg.contains("Connection refused")
                    || msg.contains("ssh:") =>
            {
                tracing::error!(
                    "Critical error discovering endpoints (not swallowing): {}",
                    msg
                );
                return Err(SlurmError::CommandFailed(format!(
                    "Endpoint discovery failed: {}",
                    msg
                )));
            }
            Err(e) => {
                // Unknown error - log as warning but propagate for safety
                tracing::warn!("Unexpected error listing endpoints: {}", e);
                return Err(e);
            }
        };

        // Get the most recent endpoint file (first line since ls -1t sorts by mtime desc)
        let endpoint_file = match files.lines().next() {
            Some(f) if !f.is_empty() => f,
            _ => return Ok(None),
        };

        // Read endpoint info
        let content = match self.run_slurm_cmd("cat", &[endpoint_file]) {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!("Failed to read endpoint file {}: {}", endpoint_file, e);
                return Ok(None);
            }
        };

        let endpoint: EndpointInfo = match serde_json::from_str(&content) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Invalid endpoint JSON in {}: {}", endpoint_file, e);
                return Ok(None);
            }
        };

        // Endpoint files can outlive jobs; avoid caching stale endpoints.
        match self.get_job_status(endpoint.job_id) {
            Ok(state) if Self::job_state_stale_for_discovery(&state) => {
                tracing::info!(
                    "Ignoring stale endpoint {} with job {} in state {:?}",
                    endpoint.endpoint,
                    endpoint.job_id,
                    state
                );
                return Ok(None);
            }
            Ok(_) => {}
            Err(SlurmError::JobNotFound(_)) => {
                tracing::info!(
                    "Ignoring endpoint {} because job {} is missing",
                    endpoint.endpoint,
                    endpoint.job_id
                );
                return Ok(None);
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to verify job state for endpoint {}: {}",
                    endpoint.endpoint,
                    e
                );
            }
        }

        // Cache it
        self.endpoint_cache.insert(tier, endpoint.clone());
        tracing::debug!("Discovered {} endpoint: {}", tier, endpoint.endpoint);

        Ok(Some(endpoint))
    }

    /// Check endpoint health via HTTP with retry logic
    pub async fn check_endpoint_health(&self, endpoint: &EndpointInfo) -> bool {
        let details = self.check_endpoint_health_detailed(endpoint).await;
        matches!(
            details.state,
            EndpointHealth::Healthy | EndpointHealth::Degraded
        )
    }

    /// Check endpoint health with detailed state for recovery logic
    pub async fn check_endpoint_health_detailed(
        &self,
        endpoint: &EndpointInfo,
    ) -> EndpointHealthDetails {
        let mut healthy_workers = Vec::new();
        let mut unhealthy_workers = Vec::new();
        let job_state = match self.get_job_status(endpoint.job_id) {
            Ok(state) => Some(state),
            Err(SlurmError::JobNotFound(_)) => Some(JobState::Completed),
            Err(e) => {
                tracing::warn!(
                    "Failed to query job state for endpoint {}: {}",
                    endpoint.endpoint,
                    e
                );
                None
            }
        };

        if let Some(state) = job_state.as_ref() {
            if Self::job_state_stale_for_discovery(state) {
                let details = EndpointHealthDetails {
                    state: EndpointHealth::Stale,
                    job_state: job_state.clone(),
                    attempts: 0,
                    healthy_workers,
                    unhealthy_workers,
                    last_error: Some(format!("stale job state {:?}", state)),
                    timed_out: false,
                };
                self.health_metrics.record(EndpointHealth::Stale);
                tracing::info!(
                    "Endpoint {} marked stale (job {} state {:?})",
                    endpoint.endpoint,
                    endpoint.job_id,
                    state
                );
                return details;
            }
        }

        let health_url = format!("http://{}:{}/health", endpoint.host, endpoint.port);
        let (head_ok, attempts_used, mut timed_out, mut last_error) =
            self.probe_health(&health_url).await;

        if !head_ok {
            let details = EndpointHealthDetails {
                state: EndpointHealth::Unhealthy,
                job_state,
                attempts: attempts_used,
                healthy_workers,
                unhealthy_workers,
                last_error,
                timed_out,
            };
            self.health_metrics.record(EndpointHealth::Unhealthy);
            if timed_out {
                self.health_metrics.record_timeout();
            } else if details.last_error.is_some() {
                self.health_metrics.record_error();
            }
            tracing::warn!(
                "Endpoint {} health check failed (head unhealthy)",
                endpoint.endpoint
            );
            return details;
        }

        let rpc_workers = endpoint.rpc_workers.clone().unwrap_or_default();
        if rpc_workers.is_empty() {
            let details = EndpointHealthDetails {
                state: EndpointHealth::Healthy,
                job_state,
                attempts: attempts_used,
                healthy_workers,
                unhealthy_workers,
                last_error: None,
                timed_out,
            };
            self.health_metrics.record(EndpointHealth::Healthy);
            return details;
        }

        for worker in rpc_workers {
            let worker_url = format!("http://{}:{}/health", worker, endpoint.port);
            let (ok, _attempts, worker_timeout, worker_error) =
                self.probe_health(&worker_url).await;
            if ok {
                healthy_workers.push(worker);
            } else {
                if worker_timeout {
                    timed_out = true;
                }
                if worker_error.is_some() {
                    last_error = worker_error;
                }
                unhealthy_workers.push(worker);
            }
        }

        let healthy_count = healthy_workers.len();
        let min_required = self.config.health_check.min_rpc_workers_healthy;
        let total_workers = healthy_count + unhealthy_workers.len();

        // For Reasoning tier (distributed 72B), tensors are split across ALL GPUs.
        // If ANY worker is unhealthy, the model cannot function - it's Unhealthy, not Degraded.
        let is_distributed = endpoint.tier == "manager-local" || endpoint.rpc_workers.is_some();
        let state = if healthy_count == 0 {
            EndpointHealth::Unhealthy
        } else if is_distributed && !unhealthy_workers.is_empty() {
            // Distributed inference requires ALL workers - partial = broken
            tracing::warn!(
                "Distributed endpoint {} has {}/{} workers unhealthy - marking Unhealthy",
                endpoint.endpoint,
                unhealthy_workers.len(),
                total_workers
            );
            EndpointHealth::Unhealthy
        } else if healthy_count < min_required || !unhealthy_workers.is_empty() {
            EndpointHealth::Degraded
        } else {
            EndpointHealth::Healthy
        };

        let details = EndpointHealthDetails {
            state,
            job_state,
            attempts: attempts_used,
            healthy_workers,
            unhealthy_workers,
            last_error,
            timed_out,
        };

        self.health_metrics.record(state);
        if timed_out {
            self.health_metrics.record_timeout();
        } else if details.last_error.is_some() {
            self.health_metrics.record_error();
        }

        if state == EndpointHealth::Degraded {
            tracing::warn!(
                "Endpoint {} degraded: {} healthy, {} unhealthy workers",
                endpoint.endpoint,
                details.healthy_workers.len(),
                details.unhealthy_workers.len()
            );
        }

        details
    }

    /// Snapshot health check counters for observability
    pub fn health_metrics_snapshot(&self) -> HealthCheckMetricsSnapshot {
        self.health_metrics.snapshot()
    }

    /// Update recovery state machine and return effective health state.
    /// Returns `EndpointHealth::Stale` when max recovery attempts exceeded (signals forced restart).
    fn update_recovery_state(
        &mut self,
        tier: InferenceTier,
        state: EndpointHealth,
    ) -> EndpointHealth {
        let max_degraded = self.config.health_check.max_consecutive_degraded;
        let max_recovery = self.config.health_check.max_recovery_attempts;
        let current = self
            .recovery_state
            .get(&tier)
            .copied()
            .unwrap_or(RecoveryState::Stable);

        let next = match (current, state) {
            // Healthy always resets to Stable - recovery succeeded
            (RecoveryState::Recovering { attempts }, EndpointHealth::Healthy) if attempts > 0 => {
                tracing::info!("{} endpoint recovered after {} attempts", tier, attempts);
                RecoveryState::Stable
            }
            (_, EndpointHealth::Healthy) => RecoveryState::Stable,

            // Track consecutive degraded states
            (RecoveryState::Degraded { consecutive }, EndpointHealth::Degraded) => {
                RecoveryState::Degraded {
                    consecutive: consecutive.saturating_add(1),
                }
            }
            (_, EndpointHealth::Degraded) => RecoveryState::Degraded { consecutive: 1 },

            // Track recovery attempts for unhealthy
            (RecoveryState::Recovering { attempts }, EndpointHealth::Unhealthy) => {
                RecoveryState::Recovering {
                    attempts: attempts.saturating_add(1),
                }
            }
            (_, EndpointHealth::Unhealthy | EndpointHealth::Stale) => {
                RecoveryState::Recovering { attempts: 1 }
            }
        };

        self.recovery_state.insert(tier, next);

        // Check thresholds and escalate if needed
        match next {
            RecoveryState::Degraded { consecutive } if consecutive >= max_degraded => {
                tracing::warn!(
                    "{} endpoint degraded {} times; escalating to recovery",
                    tier,
                    consecutive
                );
                self.recovery_state
                    .insert(tier, RecoveryState::Recovering { attempts: 0 });
                return EndpointHealth::Unhealthy;
            }
            RecoveryState::Recovering { attempts } if attempts >= max_recovery => {
                tracing::error!(
                    "{} endpoint failed {} recovery attempts; forcing restart",
                    tier,
                    attempts
                );
                // Return Stale to signal that caller should cancel job and resubmit
                // Reset state to avoid infinite restart loop
                self.recovery_state.insert(tier, RecoveryState::Stable);
                return EndpointHealth::Stale;
            }
            _ => {}
        }

        state
    }

    /// Get the current recovery state for a tier (for observability)
    pub fn get_recovery_state(&self, tier: InferenceTier) -> RecoveryState {
        self.recovery_state
            .get(&tier)
            .copied()
            .unwrap_or(RecoveryState::Stable)
    }

    /// Check if job state is stale for DISCOVERY purposes.
    /// A job is "stale for discovery" if its endpoint file shouldn't be trusted.
    /// Completing jobs may have stale endpoints as the server is shutting down.
    fn job_state_stale_for_discovery(state: &JobState) -> bool {
        // Only Running jobs have trustworthy endpoints
        // Completing = server shutting down, endpoint may not respond
        // Pending = no endpoint exists yet
        !matches!(state, JobState::Running)
    }

    /// Check if job state is stale for SUBMISSION purposes.
    /// A job is "stale for submission" if we should submit a new job.
    /// This is different from discovery - we don't want to submit duplicates
    /// for Pending or Completing jobs.
    fn job_state_stale_for_submission(state: &JobState) -> bool {
        !matches!(
            state,
            JobState::Running | JobState::Pending | JobState::Completing
        )
    }

    /// Probe health endpoint with timeout covering BOTH headers AND body.
    /// This ensures we detect slow/hung servers, not just connection failures.
    async fn probe_health(&self, url: &str) -> (bool, u32, bool, Option<String>) {
        let mut last_error = None;
        let mut timed_out = false;
        let retries = self.config.health_check.max_retries;

        for attempt in 0..=retries {
            // Wrap ENTIRE request (headers + body) in timeout to detect hung servers
            let health_check = async {
                let resp = self
                    .http_client
                    .get(url)
                    .send()
                    .await
                    .map_err(|e| e.to_string())?;
                if !resp.status().is_success() {
                    return Err(format!("non-success status {}", resp.status()));
                }
                // Read body to ensure server is fully responsive, not just accepting connections
                let _body = resp.bytes().await.map_err(|e| e.to_string())?;
                Ok::<(), String>(())
            };

            match tokio::time::timeout(self.config.health_check.response_timeout, health_check)
                .await
            {
                Ok(Ok(())) => {
                    if attempt > 0 {
                        tracing::debug!(
                            "Health check succeeded on attempt {} for {}",
                            attempt + 1,
                            url
                        );
                    }
                    return (true, attempt + 1, timed_out, None);
                }
                Ok(Err(e)) => {
                    last_error = Some(e);
                }
                Err(_) => {
                    timed_out = true;
                    last_error = Some("response timeout".to_string());
                }
            }

            if attempt < retries {
                tokio::time::sleep(self.config.health_check.retry_backoff).await;
            }
        }

        (false, retries + 1, timed_out, last_error)
    }

    /// Ensure an inference server is running for the given tier
    pub async fn ensure_running(
        &mut self,
        tier: InferenceTier,
    ) -> Result<EndpointInfo, SlurmError> {
        // Check if we have a cached endpoint
        if let Some(endpoint) = self.endpoint_cache.get(&tier).cloned() {
            let details = self.check_endpoint_health_detailed(&endpoint).await;
            let state = self.update_recovery_state(tier, details.state);
            match state {
                EndpointHealth::Healthy | EndpointHealth::Degraded => return Ok(endpoint),
                EndpointHealth::Stale => {
                    // Stale from update_recovery_state means max recovery attempts exceeded
                    // Force cancel the job and resubmit
                    if let Some(job_id) = self.active_jobs.remove(&tier) {
                        tracing::warn!(
                            "Forcing restart of {} job {} due to max recovery attempts",
                            tier,
                            job_id
                        );
                        if let Err(e) = self.cancel_job(job_id) {
                            tracing::warn!("Failed to cancel job {}: {}", job_id, e);
                        }
                    }
                    self.endpoint_cache.remove(&tier);
                }
                EndpointHealth::Unhealthy => {
                    self.endpoint_cache.remove(&tier);
                }
            }
        }

        // Check if we have an active job (running, pending, or completing)
        // CRITICAL: Use is_job_active, not is_job_running, to prevent "pending job explosion"
        // where we submit duplicate jobs for jobs that are queued but not yet running.
        if let Some(&job_id) = self.active_jobs.get(&tier) {
            let job_state = self.get_job_status(job_id)?;

            match job_state {
                JobState::Running => {
                    // Job is running, try to discover endpoint
                    if let Some(endpoint) = self.discover_endpoint(tier)? {
                        let details = self.check_endpoint_health_detailed(&endpoint).await;
                        let state = self.update_recovery_state(tier, details.state);
                        if matches!(state, EndpointHealth::Healthy | EndpointHealth::Degraded) {
                            return Ok(endpoint);
                        }
                        if state == EndpointHealth::Stale {
                            // Max recovery attempts exceeded - cancel and resubmit
                            tracing::warn!(
                                "Forcing restart of {} job {} due to max recovery attempts",
                                tier,
                                job_id
                            );
                            self.active_jobs.remove(&tier);
                            if let Err(e) = self.cancel_job(job_id) {
                                tracing::warn!("Failed to cancel job {}: {}", job_id, e);
                            }
                        }
                        self.endpoint_cache.remove(&tier);
                    }
                    // Job running but endpoint not ready yet - wait, don't resubmit
                    tracing::debug!("Job {} running but endpoint not ready, waiting...", job_id);
                    return Err(SlurmError::EndpointTimeout(Duration::from_secs(0)));
                }
                JobState::Pending => {
                    // Job is queued, waiting for resources - do NOT submit another job
                    tracing::info!(
                        "Job {} is pending in queue, waiting for resources...",
                        job_id
                    );
                    return Err(SlurmError::EndpointTimeout(Duration::from_secs(0)));
                }
                JobState::Completing => {
                    // Job is finishing up - wait for it
                    tracing::debug!("Job {} is completing, waiting...", job_id);
                    return Err(SlurmError::EndpointTimeout(Duration::from_secs(0)));
                }
                JobState::Preempted => {
                    // Job was preempted by higher priority work - it will be requeued by SLURM
                    // Remove from our tracking so we can discover the requeued job
                    tracing::info!("Job {} was preempted, will be requeued by SLURM", job_id);
                    self.active_jobs.remove(&tier);
                }
                _ => {
                    // Job completed, failed, cancelled, etc. - remove from tracking
                    tracing::debug!("Job {} ended with state {:?}", job_id, job_state);
                    self.active_jobs.remove(&tier);
                }
            }
        }

        // Try to discover an existing endpoint (from another job)
        if let Some(endpoint) = self.discover_endpoint(tier)? {
            let details = self.check_endpoint_health_detailed(&endpoint).await;
            let state = self.update_recovery_state(tier, details.state);
            if matches!(state, EndpointHealth::Healthy | EndpointHealth::Degraded) {
                return Ok(endpoint);
            }
            if state == EndpointHealth::Stale {
                self.active_jobs.remove(&tier);
            }
            self.endpoint_cache.remove(&tier);
        }

        // No running endpoint, submit new job
        let _job_id = self.submit_job(tier)?;

        // Wait for endpoint to be ready
        self.wait_for_ready(tier, self.config.endpoint_timeout)
            .await
    }

    /// Wait for endpoint to be ready
    pub async fn wait_for_ready(
        &mut self,
        tier: InferenceTier,
        timeout: Duration,
    ) -> Result<EndpointInfo, SlurmError> {
        let start = std::time::Instant::now();
        let check_interval = Duration::from_secs(5);

        while start.elapsed() < timeout {
            // Check for endpoint
            if let Some(endpoint) = self.discover_endpoint(tier)? {
                let details = self.check_endpoint_health_detailed(&endpoint).await;
                let state = self.update_recovery_state(tier, details.state);
                if matches!(state, EndpointHealth::Healthy | EndpointHealth::Degraded) {
                    tracing::info!(
                        "{} endpoint ready: {} ({:?})",
                        tier,
                        endpoint.endpoint,
                        state
                    );
                    return Ok(endpoint);
                }
                if state == EndpointHealth::Stale {
                    self.active_jobs.remove(&tier);
                }
                self.endpoint_cache.remove(&tier);
            }

            tokio::time::sleep(check_interval).await;
        }

        Err(SlurmError::EndpointTimeout(timeout))
    }

    /// Cancel a running job
    pub fn cancel_job(&mut self, job_id: u32) -> Result<(), SlurmError> {
        self.run_slurm_cmd("scancel", &[&job_id.to_string()])?;

        // Remove from active jobs
        self.active_jobs.retain(|_, &mut id| id != job_id);

        tracing::info!("Cancelled job: {}", job_id);
        Ok(())
    }

    /// Get all active inference jobs
    pub fn list_active_jobs(&self) -> Result<Vec<JobInfo>, SlurmError> {
        // Query by job name pattern, not user (jobs run as submitting user)
        let output = self.run_slurm_cmd(
            "squeue",
            &[
                "-n",
                "llama-worker,llama-qwen35",
                "-o",
                "%i|%j|%T|%N|%P|%M|%l",
                "--noheader",
            ],
        )?;

        let mut jobs = Vec::new();
        for line in output.lines() {
            let parts: Vec<&str> = line.split('|').collect();
            if parts.len() >= 7 {
                if let Ok(job_id) = parts[0].parse::<u32>() {
                    jobs.push(JobInfo {
                        job_id,
                        name: parts[1].to_string(),
                        state: JobState::from(parts[2]),
                        node_list: if parts[3].is_empty() {
                            None
                        } else {
                            Some(parts[3].to_string())
                        },
                        partition: parts[4].to_string(),
                        time_used: parts[5].to_string(),
                        time_limit: parts[6].to_string(),
                    });
                }
            }
        }

        Ok(jobs)
    }

    /// Get the cached endpoint for a tier
    pub fn get_cached_endpoint(&self, tier: InferenceTier) -> Option<&EndpointInfo> {
        self.endpoint_cache.get(&tier)
    }

    /// Clear cached endpoints (useful after preemption)
    pub fn clear_cache(&mut self) {
        self.endpoint_cache.clear();
        self.active_jobs.clear();
        self.recovery_state.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_tier_display() {
        assert_eq!(InferenceTier::Worker.to_string(), "worker");
        assert_eq!(InferenceTier::ManagerLocal.to_string(), "manager-local");
    }

    #[test]
    fn test_job_state_from_str() {
        assert_eq!(JobState::from("RUNNING"), JobState::Running);
        assert_eq!(JobState::from("R"), JobState::Running);
        assert_eq!(JobState::from("PENDING"), JobState::Pending);
        assert_eq!(JobState::from("PREEMPTED"), JobState::Preempted);
    }

    #[test]
    fn test_tier_job_script() {
        assert_eq!(InferenceTier::Worker.job_script(), "run-worker.slurm");
        assert_eq!(
            InferenceTier::ManagerLocal.job_script(),
            "run-qwen35-distributed.slurm"
        );
    }

    #[test]
    fn test_job_state_is_active() {
        // These states should be considered "active" (don't submit duplicate jobs)
        assert!(matches!(
            JobState::Running,
            JobState::Running | JobState::Pending | JobState::Completing
        ));
        assert!(matches!(
            JobState::Pending,
            JobState::Running | JobState::Pending | JobState::Completing
        ));
        assert!(matches!(
            JobState::Completing,
            JobState::Running | JobState::Pending | JobState::Completing
        ));

        // These states should NOT be considered active
        assert!(!matches!(
            JobState::Completed,
            JobState::Running | JobState::Pending | JobState::Completing
        ));
        assert!(!matches!(
            JobState::Failed,
            JobState::Running | JobState::Pending | JobState::Completing
        ));
        assert!(!matches!(
            JobState::Cancelled,
            JobState::Running | JobState::Pending | JobState::Completing
        ));
        assert!(!matches!(
            JobState::Preempted,
            JobState::Running | JobState::Pending | JobState::Completing
        ));
    }

    #[test]
    fn test_config_from_env() {
        // Test that default config can be created
        let config = SlurmConfig::default();
        assert!(!config.scripts_path.as_os_str().is_empty());
        assert!(!config.endpoints_path.as_os_str().is_empty());
    }
}
