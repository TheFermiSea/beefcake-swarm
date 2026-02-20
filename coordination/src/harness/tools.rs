//! MCP Tools for the Agent Harness
//!
//! Provides MCP tool interfaces for harness operations:
//! - harness_start: Initialize a harness session
//! - harness_status: Get current session and feature status
//! - harness_complete_feature: Mark a feature as complete
//! - harness_checkpoint: Create a git checkpoint
//! - harness_rollback: Rollback to a previous checkpoint

use crate::harness::error::HarnessResult;
use crate::harness::feature_registry::FeatureRegistry;
use crate::harness::git_manager::GitManager;
use crate::harness::progress::ProgressTracker;
use crate::harness::session::{SessionManager, SessionSummary};
use crate::harness::startup::{format_startup_context, perform_startup_ritual};
use crate::harness::types::{
    FeatureSummary, HarnessConfig, ProgressMarker, SessionStatus, StartupContext,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

/// Shared harness state for MCP tools
pub struct HarnessState {
    pub config: HarnessConfig,
    pub session: Option<SessionManager>,
    pub registry: Option<FeatureRegistry>,
    pub progress: ProgressTracker,
    pub git: GitManager,
    /// Startup checklist for tracking whether agent acknowledged startup ritual
    pub startup_checklist: Option<crate::harness::types::StartupChecklist>,
    /// Timestamp of last status read (for startup ritual enforcement)
    pub last_status_read: Option<chrono::DateTime<chrono::Utc>>,
    // Note: pending_interventions now stored in SessionState for persistence
}

impl HarnessState {
    /// Create new harness state with config
    pub fn new(config: HarnessConfig) -> Self {
        let progress = ProgressTracker::new(&config.progress_path);
        let git = GitManager::new(&config.working_directory, &config.commit_prefix);

        Self {
            config,
            session: None,
            registry: None,
            progress,
            git,
            startup_checklist: None,
            last_status_read: None,
        }
    }

    /// Create from environment
    pub fn from_env() -> Self {
        Self::new(HarnessConfig::from_env())
    }

    /// Get pending interventions from session (convenience method)
    pub fn pending_interventions(&self) -> Vec<crate::harness::types::PendingIntervention> {
        self.session
            .as_ref()
            .map(|s| s.state().pending_interventions.clone())
            .unwrap_or_default()
    }
}

/// Thread-safe harness state wrapper
pub type SharedHarnessState = Arc<Mutex<HarnessState>>;

/// Create shared harness state
pub fn create_shared_state(config: HarnessConfig) -> SharedHarnessState {
    Arc::new(Mutex::new(HarnessState::new(config)))
}

// ============================================================================
// Request/Response Types for MCP Tools
// ============================================================================

/// Request for harness_start tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessStartRequest {
    /// Maximum iterations for this session (default: 20)
    #[schemars(description = "Maximum iterations before session auto-stops")]
    pub max_iterations: Option<u32>,

    /// Whether to require clean git state (default: false)
    #[schemars(description = "Require no uncommitted changes before starting")]
    pub require_clean_git: Option<bool>,

    /// Resume from previous session if found (default: true)
    #[schemars(description = "Resume interrupted session if found")]
    pub auto_resume: Option<bool>,
}

/// Response for harness_start tool
#[derive(Debug, Serialize)]
pub struct HarnessStartResponse {
    pub success: bool,
    pub session_id: String,
    pub is_resume: bool,
    pub context_summary: String,
    pub startup_context: StartupContext,
    /// Startup checklist for agent to acknowledge (Phase 3: Startup Ritual)
    pub startup_checklist: crate::harness::types::StartupChecklist,
}

/// Request for harness_status tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessStatusRequest {
    /// Include detailed feature list (default: false)
    #[schemars(description = "Include full feature list in response")]
    pub include_features: Option<bool>,

    /// Include recent progress entries (default: true)
    #[schemars(description = "Include recent progress log entries")]
    pub include_progress: Option<bool>,

    /// Include structured session summary (anchored iterative) (default: false)
    #[schemars(description = "Include structured session summary")]
    pub include_structured_summary: Option<bool>,

    /// Maximum features to return (default: 20, to manage token budget)
    #[schemars(description = "Maximum features to include in response")]
    pub max_features: Option<u32>,

    /// Maximum progress entries to return (default: 10)
    #[schemars(description = "Maximum progress entries to include")]
    pub max_progress_entries: Option<u32>,
}

/// Response for harness_status tool
#[derive(Debug, Serialize)]
pub struct HarnessStatusResponse {
    pub session: Option<SessionSummary>,
    pub features: FeatureSummary,
    pub next_feature: Option<String>,
    pub git_status: GitStatus,
    pub recent_progress: Option<Vec<String>>,

    /// Structured session summary (anchored iterative)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub structured_summary: Option<crate::harness::types::StructuredSessionSummary>,

    /// Truncation notice for features (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub features_truncation: Option<crate::harness::types::TruncationNotice>,

    /// Truncation notice for progress entries (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub progress_truncation: Option<crate::harness::types::TruncationNotice>,

    /// Pending human interventions (Phase 5)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub pending_interventions: Vec<crate::harness::types::PendingIntervention>,
}

/// Git status info
#[derive(Debug, Serialize)]
pub struct GitStatus {
    pub branch: String,
    pub commit: String,
    pub has_uncommitted_changes: bool,
}

/// Request for harness_complete_feature tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessCompleteFeatureRequest {
    /// Feature ID to mark as complete
    #[schemars(description = "ID of the feature to mark as complete")]
    pub feature_id: String,

    /// Summary of what was accomplished
    #[schemars(description = "Brief summary of how the feature was implemented")]
    pub summary: String,

    /// Auto-checkpoint after completion (default: true)
    #[schemars(description = "Create git checkpoint after marking complete")]
    pub checkpoint: Option<bool>,
}

/// Response for harness_complete_feature tool
#[derive(Debug, Serialize)]
pub struct HarnessCompleteFeatureResponse {
    pub success: bool,
    pub feature_id: String,
    pub checkpoint_commit: Option<String>,
    pub remaining_features: usize,
    pub completion_percent: f32,
}

/// Request for harness_checkpoint tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessCheckpointRequest {
    /// Description for the checkpoint
    #[schemars(description = "Description of what this checkpoint captures")]
    pub description: String,

    /// Associated feature (optional)
    #[schemars(description = "Feature ID this checkpoint is for")]
    pub feature_id: Option<String>,
}

/// Response for harness_checkpoint tool
#[derive(Debug, Serialize)]
pub struct HarnessCheckpointResponse {
    pub success: bool,
    pub commit_hash: String,
    pub commit_message: String,
}

/// Request for harness_rollback tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessRollbackRequest {
    /// Commit hash to rollback to
    #[schemars(description = "Git commit hash to rollback to")]
    pub commit_hash: String,

    /// Hard rollback discards all changes (default: false for soft)
    #[schemars(description = "Hard rollback discards changes, soft preserves them")]
    pub hard: Option<bool>,
}

/// Response for harness_rollback tool
#[derive(Debug, Serialize)]
pub struct HarnessRollbackResponse {
    pub success: bool,
    pub rolled_back_to: String,
    pub was_hard_rollback: bool,
}

/// Request for harness_iterate tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessIterateRequest {
    /// Summary of work done this iteration
    #[schemars(description = "Brief summary of what was accomplished")]
    pub summary: String,

    /// Feature being worked on (optional)
    #[schemars(description = "Feature ID currently being worked on")]
    pub feature_id: Option<String>,
}

/// Response for harness_iterate tool
#[derive(Debug, Serialize)]
pub struct HarnessIterateResponse {
    pub success: bool,
    pub iteration: u32,
    pub max_iterations: u32,
    pub can_continue: bool,
    pub session_status: SessionStatus,
}

// ============================================================================
// Phase 2: Token Budget Management - Compact Progress Tool
// ============================================================================

/// Request for harness_compact_progress tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessCompactProgressRequest {
    /// Number of recent entries to keep uncompacted (default: 10)
    #[schemars(description = "Keep this many recent entries without summarization")]
    pub keep_recent: Option<u32>,

    /// Whether to actually perform compaction or just preview (default: false = preview)
    #[schemars(description = "Set to true to perform compaction, false to preview")]
    pub execute: Option<bool>,
}

/// Response for harness_compact_progress tool
#[derive(Debug, Serialize)]
pub struct HarnessCompactProgressResponse {
    pub success: bool,
    /// Number of entries before compaction
    pub entries_before: usize,
    /// Number of entries after compaction
    pub entries_after: usize,
    /// Number of entries that were compacted
    pub entries_compacted: usize,
    /// Summary of compacted entries
    pub compaction_summary: String,
    /// Whether compaction was actually executed (vs preview)
    pub executed: bool,
}

// ============================================================================
// Phase 3: Startup Ritual - Acknowledge Tool
// ============================================================================

/// Request for harness_acknowledge tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessAcknowledgeRequest {
    /// Checklist items that were reviewed (optional, for tracking)
    #[schemars(description = "IDs of checklist items that were reviewed")]
    pub reviewed_items: Option<Vec<String>>,
}

/// Response for harness_acknowledge tool
#[derive(Debug, Serialize)]
pub struct HarnessAcknowledgeResponse {
    pub success: bool,
    pub acknowledged: bool,
    pub session_id: String,
    /// Warning if startup ritual was not fully completed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub warning: Option<String>,
}

// ============================================================================
// Phase 4: Workflow-First Tools
// ============================================================================

/// Request for harness_work_on_feature tool (combines iterate + set_current_feature + log)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessWorkOnFeatureRequest {
    /// Feature ID to work on
    #[schemars(description = "ID of the feature to start working on")]
    pub feature_id: String,

    /// Summary of what you're doing
    #[schemars(description = "Brief summary of the work being done")]
    pub summary: String,
}

/// Response for harness_work_on_feature tool
#[derive(Debug, Serialize)]
pub struct HarnessWorkOnFeatureResponse {
    pub success: bool,
    pub feature_id: String,
    pub iteration: u32,
    pub can_continue: bool,
    /// Feature details (description, steps, etc.)
    pub feature_description: String,
    pub feature_steps: Vec<String>,
}

/// Request for harness_complete_and_next tool (combines complete + checkpoint + get_next)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessCompleteAndNextRequest {
    /// Feature ID to mark as complete
    #[schemars(description = "ID of the feature to mark as complete")]
    pub feature_id: String,

    /// Summary of what was accomplished
    #[schemars(description = "Brief summary of how the feature was implemented")]
    pub summary: String,

    /// Whether to create a checkpoint (default: true)
    #[schemars(description = "Create git checkpoint after completion")]
    pub checkpoint: Option<bool>,
}

/// Response for harness_complete_and_next tool
#[derive(Debug, Serialize)]
pub struct HarnessCompleteAndNextResponse {
    pub success: bool,
    pub completed_feature_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub checkpoint_hash: Option<String>,
    /// Next feature to work on (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_feature_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_feature_description: Option<String>,
    pub completion_percent: f32,
    pub remaining_features: usize,
}

/// Request for harness_quick_status tool (minimal status for rapid polling)
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessQuickStatusRequest {
    // No parameters needed - returns minimal info
}

/// Response for harness_quick_status tool
#[derive(Debug, Serialize)]
pub struct HarnessQuickStatusResponse {
    pub iteration: u32,
    pub max_iterations: u32,
    pub can_continue: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current_feature: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub next_feature: Option<String>,
    pub session_status: String,
}

// ============================================================================
// Phase 5: Human Intervention Points
// ============================================================================

/// Request for harness_request_intervention tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessRequestInterventionRequest {
    /// Type of intervention: review_required, approval_needed, decision_point, clarification_needed
    #[schemars(description = "Type of intervention required")]
    pub intervention_type: String,

    /// Question or description for the human
    #[schemars(description = "What decision or review is needed")]
    pub question: String,

    /// Associated feature (optional)
    #[schemars(description = "Feature ID this intervention relates to")]
    pub feature_id: Option<String>,

    /// Available options for decision points
    #[schemars(description = "Options for the human to choose from (for decision points)")]
    pub options: Option<Vec<String>>,
}

/// Response for harness_request_intervention tool
#[derive(Debug, Serialize)]
pub struct HarnessRequestInterventionResponse {
    pub success: bool,
    pub intervention_id: String,
    pub intervention_type: String,
    pub question: String,
}

/// Request for harness_resolve_intervention tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessResolveInterventionRequest {
    /// ID of the intervention to resolve
    #[schemars(description = "Intervention ID to resolve")]
    pub intervention_id: String,

    /// Resolution/decision made
    #[schemars(description = "The resolution or decision")]
    pub resolution: String,
}

/// Response for harness_resolve_intervention tool
#[derive(Debug, Serialize)]
pub struct HarnessResolveInterventionResponse {
    pub success: bool,
    pub intervention_id: String,
    pub resolved: bool,
    /// Feature ID that can now continue (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unblocked_feature: Option<String>,
}

// ============================================================================
// Phase 6: Sub-Agent Delegation
// ============================================================================

/// Request for harness_delegate tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessDelegateRequest {
    /// Feature ID to delegate
    #[schemars(description = "Feature ID for the delegated work")]
    pub feature_id: String,

    /// Task description for the sub-agent
    #[schemars(description = "Detailed description of the task for the sub-agent")]
    pub task_description: String,

    /// Maximum iterations for the sub-session (default: 10)
    #[schemars(description = "Maximum iterations for the sub-agent")]
    pub max_iterations: Option<u32>,
}

/// Response for harness_delegate tool
#[derive(Debug, Serialize)]
pub struct HarnessDelegateResponse {
    pub success: bool,
    pub sub_session_id: String,
    pub feature_id: String,
    /// Path to context file for the sub-session
    pub context_path: String,
}

/// Request for harness_sub_session_status tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessSubSessionStatusRequest {
    /// Sub-session ID to check
    #[schemars(description = "Sub-session ID to check status of")]
    pub sub_session_id: String,
}

/// Response for harness_sub_session_status tool
#[derive(Debug, Serialize)]
pub struct HarnessSubSessionStatusResponse {
    pub sub_session_id: String,
    pub status: String,
    pub iteration: u32,
    pub max_iterations: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,
}

/// Request for harness_claim_sub_session_result tool
#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct HarnessClaimSubSessionResultRequest {
    /// Sub-session ID to claim results from
    #[schemars(description = "Sub-session ID to claim results from")]
    pub sub_session_id: String,

    /// Summary of the sub-session work (optional, uses sub-session summary if not provided)
    #[schemars(description = "Optional summary to use instead of sub-session's own summary")]
    pub summary: Option<String>,
}

/// Response for harness_claim_sub_session_result tool
#[derive(Debug, Serialize)]
pub struct HarnessClaimSubSessionResultResponse {
    pub success: bool,
    pub sub_session_id: String,
    pub feature_id: String,
    pub summary: String,
    /// Progress entry added to main session
    pub progress_logged: bool,
}

// ============================================================================
// Tool Implementation Functions
// ============================================================================

/// Start or resume a harness session
pub fn harness_start(
    state: &mut HarnessState,
    req: HarnessStartRequest,
) -> HarnessResult<HarnessStartResponse> {
    // Update config from request
    if let Some(max) = req.max_iterations {
        state.config.max_iterations = max;
    }
    if let Some(clean) = req.require_clean_git {
        state.config.require_clean_git = clean;
    }

    // Perform startup ritual
    let context = perform_startup_ritual(&state.config)?;
    let context_summary = format_startup_context(&context);

    // Check for resumable session - first from persisted state file, then from progress
    let (session, is_resume) = if req.auto_resume.unwrap_or(true) {
        // Try to load persisted session state first (higher priority)
        if let Some(persisted) =
            crate::harness::session::load_session_state(&state.config.session_state_path)?
        {
            // Resume from persisted state
            let session = SessionManager::from_state(persisted);
            (session, true)
        } else if let Some(ref last) = context.last_session {
            // Fall back to progress file
            let session = SessionManager::from_state(last.clone());
            (session, true)
        } else {
            // Create new session
            let mut session = SessionManager::new(
                context.working_directory.clone(),
                state.config.max_iterations,
            );
            session.set_initial_commit(context.current_commit.clone());
            (session, false)
        }
    } else {
        // Force new session - clear any persisted state
        let _ = crate::harness::session::clear_session_state(&state.config.session_state_path);
        let mut session = SessionManager::new(
            context.working_directory.clone(),
            state.config.max_iterations,
        );
        session.set_initial_commit(context.current_commit.clone());
        (session, false)
    };

    let session_id = session.session_id().to_string();

    // Load or create registry
    let registry = if state.config.features_path.exists() {
        FeatureRegistry::load(&state.config.features_path)?
    } else {
        FeatureRegistry::empty(&state.config.features_path)
    };

    // Log session start
    if !is_resume {
        state.progress.log_session_start(
            &session_id,
            format!(
                "Started new session with {} features",
                context.feature_summary.total
            ),
        )?;
    }

    // Create startup checklist for agent to acknowledge
    let startup_checklist = crate::harness::types::StartupChecklist::new();

    // Store state
    state.session = Some(session);
    state.registry = Some(registry);
    state.startup_checklist = Some(startup_checklist.clone());

    Ok(HarnessStartResponse {
        success: true,
        session_id,
        is_resume,
        context_summary,
        startup_context: context,
        startup_checklist,
    })
}

/// Get current harness status
pub fn harness_status(
    state: &mut HarnessState,
    req: HarnessStatusRequest,
) -> HarnessResult<HarnessStatusResponse> {
    use crate::harness::types::TruncationNotice;

    // Track that status was read (for startup ritual enforcement)
    state.last_status_read = Some(chrono::Utc::now());

    let session = state.session.as_ref().map(|s| s.summary());

    let features = state
        .registry
        .as_ref()
        .map(|r| r.summary())
        .unwrap_or_default();

    let next_feature = state
        .registry
        .as_ref()
        .and_then(|r| r.next_incomplete())
        .map(|f| f.id.clone());

    let git_status = GitStatus {
        branch: state
            .git
            .current_branch()
            .unwrap_or_else(|_| "unknown".into()),
        commit: state
            .git
            .current_commit()
            .unwrap_or_else(|_| "unknown".into()),
        has_uncommitted_changes: state.git.has_uncommitted_changes().unwrap_or(false),
    };

    // Handle progress with pagination
    let max_progress = req.max_progress_entries.unwrap_or(10) as usize;
    let (recent_progress, progress_truncation) = if req.include_progress.unwrap_or(true) {
        let all_entries = state.progress.read_last(100).unwrap_or_default();
        let total_count = all_entries.len();

        if total_count > max_progress {
            let truncated: Vec<String> = all_entries
                .into_iter()
                .take(max_progress)
                .map(|e| e.to_log_line())
                .collect();

            (
                Some(truncated),
                Some(TruncationNotice::truncated(
                    total_count,
                    max_progress,
                    "harness_get_all_progress",
                )),
            )
        } else {
            let entries: Vec<String> = all_entries.into_iter().map(|e| e.to_log_line()).collect();
            (Some(entries), None)
        }
    } else {
        (None, None)
    };

    // Handle structured summary
    let structured_summary = if req.include_structured_summary.unwrap_or(false) {
        if let Some(ref session) = state.session {
            let all_entries = state.progress.read_all().unwrap_or_default();
            Some(session.structured_summary(&all_entries))
        } else {
            None
        }
    } else {
        None
    };

    // Features truncation notice (summary already limits data, but note if large)
    let features_truncation = if features.total > req.max_features.unwrap_or(20) as usize {
        Some(TruncationNotice::truncated(
            features.total,
            req.max_features.unwrap_or(20) as usize,
            "harness_get_all_features",
        ))
    } else {
        None
    };

    // Get pending interventions from session (persisted)
    let pending_interventions: Vec<crate::harness::types::PendingIntervention> = state
        .session
        .as_ref()
        .map(|s| {
            s.state()
                .unresolved_interventions()
                .into_iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default();

    Ok(HarnessStatusResponse {
        session,
        features,
        next_feature,
        git_status,
        recent_progress,
        structured_summary,
        features_truncation,
        progress_truncation,
        pending_interventions,
    })
}

/// Mark a feature as complete
pub fn harness_complete_feature(
    state: &mut HarnessState,
    req: HarnessCompleteFeatureRequest,
) -> HarnessResult<HarnessCompleteFeatureResponse> {
    let registry = state
        .registry
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let session = state
        .session
        .as_ref()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    // Check for blocking interventions (Phase 5: Human Intervention Points)
    if session.state().has_blocking_interventions() {
        let blockers: Vec<String> = session
            .state()
            .unresolved_interventions()
            .iter()
            .filter(|i| {
                matches!(
                    i.intervention_type,
                    crate::harness::types::InterventionType::ApprovalNeeded
                        | crate::harness::types::InterventionType::DecisionPoint
                )
            })
            .map(|i| format!("[{}] {}", i.id, i.question))
            .collect();

        return Err(crate::harness::error::HarnessError::session(format!(
            "Cannot complete feature while blocking interventions are pending. \
             Resolve interventions first using harness_resolve_intervention. \
             Blocking: {}",
            blockers.join(", ")
        )));
    }

    // Mark feature as passing
    registry.mark_passing(&req.feature_id)?;
    registry.save()?;

    // Log completion
    state.progress.log_feature_complete(
        session.session_id(),
        session.iteration(),
        &req.feature_id,
        &req.summary,
    )?;

    // Optional checkpoint
    let checkpoint_commit = if req.checkpoint.unwrap_or(state.config.auto_checkpoint) {
        match state.git.create_checkpoint(&req.feature_id, &req.summary) {
            Ok(hash) => {
                state
                    .progress
                    .log_checkpoint(session.session_id(), session.iteration(), &hash)?;
                Some(hash)
            }
            Err(_) => None, // No changes to commit
        }
    } else {
        None
    };

    let summary = registry.summary();

    Ok(HarnessCompleteFeatureResponse {
        success: true,
        feature_id: req.feature_id,
        checkpoint_commit,
        remaining_features: summary.total - summary.passing,
        completion_percent: summary.completion_percent,
    })
}

/// Create a checkpoint
pub fn harness_checkpoint(
    state: &mut HarnessState,
    req: HarnessCheckpointRequest,
) -> HarnessResult<HarnessCheckpointResponse> {
    let session = state
        .session
        .as_ref()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let feature = req.feature_id.as_deref().unwrap_or("checkpoint");
    let commit_hash = state.git.create_checkpoint(feature, &req.description)?;

    state
        .progress
        .log_checkpoint(session.session_id(), session.iteration(), &commit_hash)?;

    // Persist session state for resume capability
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    let commit_message = format!(
        "{} {}: {}",
        state.config.commit_prefix, feature, req.description
    );

    Ok(HarnessCheckpointResponse {
        success: true,
        commit_hash,
        commit_message,
    })
}

/// Rollback to a previous checkpoint
pub fn harness_rollback(
    state: &mut HarnessState,
    req: HarnessRollbackRequest,
) -> HarnessResult<HarnessRollbackResponse> {
    let session = state
        .session
        .as_ref()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let was_hard = req.hard.unwrap_or(false);

    if was_hard {
        state.git.hard_rollback(&req.commit_hash)?;
    } else {
        state.git.rollback(&req.commit_hash)?;
    }

    // Log rollback
    let entry = crate::harness::types::ProgressEntry::new(
        session.session_id(),
        session.iteration(),
        ProgressMarker::Rollback,
        format!(
            "Rolled back to {} ({})",
            &req.commit_hash,
            if was_hard { "hard" } else { "soft" }
        ),
    );
    state.progress.append(&entry)?;

    Ok(HarnessRollbackResponse {
        success: true,
        rolled_back_to: req.commit_hash,
        was_hard_rollback: was_hard,
    })
}

/// Increment iteration counter
pub fn harness_iterate(
    state: &mut HarnessState,
    req: HarnessIterateRequest,
) -> HarnessResult<HarnessIterateResponse> {
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    // Check for blocking interventions (Phase 5: Human Intervention Points)
    if session.state().has_blocking_interventions() {
        let blockers: Vec<String> = session
            .state()
            .unresolved_interventions()
            .iter()
            .filter(|i| {
                matches!(
                    i.intervention_type,
                    crate::harness::types::InterventionType::ApprovalNeeded
                        | crate::harness::types::InterventionType::DecisionPoint
                )
            })
            .map(|i| format!("[{}] {}", i.id, i.question))
            .collect();

        return Err(
            crate::harness::error::HarnessError::blocked_by_intervention(blockers.join("; ")),
        );
    }

    // Start session if still initializing
    if session.status() == SessionStatus::Initializing {
        session.start()?;
    }

    // Increment iteration
    let iteration = session.next_iteration()?;

    // Update current feature
    if let Some(ref feature) = req.feature_id {
        session.set_current_feature(feature);
    }

    // Log progress
    let mut entry = crate::harness::types::ProgressEntry::new(
        session.session_id(),
        iteration,
        ProgressMarker::Progress,
        &req.summary,
    );
    if let Some(ref feature) = req.feature_id {
        entry = entry.with_feature(feature);
    }
    state.progress.append(&entry)?;

    Ok(HarnessIterateResponse {
        success: true,
        iteration,
        max_iterations: session.state().max_iterations,
        can_continue: session.can_continue(),
        session_status: session.status(),
    })
}

/// End the session
pub fn harness_end(
    state: &mut HarnessState,
    success: bool,
    summary: &str,
) -> HarnessResult<SessionSummary> {
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    if success {
        session.complete();
    } else {
        session.fail();
    }

    state
        .progress
        .log_session_end(session.session_id(), session.iteration(), summary)?;

    Ok(session.summary())
}

// ============================================================================
// Phase 2: Compact Progress Implementation
// ============================================================================

/// Compact old progress entries to manage token budget
pub fn harness_compact_progress(
    state: &mut HarnessState,
    req: HarnessCompactProgressRequest,
) -> HarnessResult<HarnessCompactProgressResponse> {
    let keep_recent = req.keep_recent.unwrap_or(10) as usize;
    let execute = req.execute.unwrap_or(false);

    // Read all progress entries
    let all_entries = state.progress.read_last(1000).unwrap_or_default();
    let entries_before = all_entries.len();

    if entries_before <= keep_recent {
        return Ok(HarnessCompactProgressResponse {
            success: true,
            entries_before,
            entries_after: entries_before,
            entries_compacted: 0,
            compaction_summary: "No compaction needed - entry count within threshold".to_string(),
            executed: false,
        });
    }

    // Split into entries to compact and entries to keep
    let split_point = entries_before.saturating_sub(keep_recent);
    let entries_to_compact = &all_entries[..split_point];
    let entries_to_keep = &all_entries[split_point..];

    // Generate summary of compacted entries
    let mut summary_parts = Vec::new();

    // Count markers
    let mut marker_counts: std::collections::HashMap<String, u32> =
        std::collections::HashMap::new();
    let mut features_worked: std::collections::HashSet<String> = std::collections::HashSet::new();

    for entry in entries_to_compact {
        *marker_counts.entry(entry.marker.to_string()).or_insert(0) += 1;
        if let Some(ref feature) = entry.feature_id {
            features_worked.insert(feature.clone());
        }
    }

    summary_parts.push(format!(
        "Compacted {} entries from iterations {}-{}",
        entries_to_compact.len(),
        entries_to_compact.first().map(|e| e.iteration).unwrap_or(0),
        entries_to_compact.last().map(|e| e.iteration).unwrap_or(0),
    ));

    if !features_worked.is_empty() {
        summary_parts.push(format!(
            "Features worked on: {}",
            features_worked.into_iter().collect::<Vec<_>>().join(", ")
        ));
    }

    let marker_summary: Vec<String> = marker_counts
        .iter()
        .map(|(k, v)| format!("{}:{}", k, v))
        .collect();
    summary_parts.push(format!("Markers: {}", marker_summary.join(", ")));

    let compaction_summary = summary_parts.join(". ");

    if execute {
        // Actually write the compacted progress file
        // First, clear the file and write a compaction marker
        let session_id = state
            .session
            .as_ref()
            .map(|s| s.session_id().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        let iteration = state.session.as_ref().map(|s| s.iteration()).unwrap_or(0);

        // Create compaction marker entry
        let compaction_entry = crate::harness::types::ProgressEntry::new(
            &session_id,
            iteration,
            ProgressMarker::Progress,
            format!("[COMPACTED] {}", compaction_summary),
        );

        // Rewrite progress file with compaction marker + kept entries
        state.progress.clear()?;
        state.progress.append(&compaction_entry)?;
        for entry in entries_to_keep {
            state.progress.append(entry)?;
        }
    }

    Ok(HarnessCompactProgressResponse {
        success: true,
        entries_before,
        entries_after: if execute {
            keep_recent + 1
        } else {
            entries_before
        },
        entries_compacted: split_point,
        compaction_summary,
        executed: execute,
    })
}

// ============================================================================
// Phase 4: Workflow-First Tool Implementations
// ============================================================================

/// Work on a feature - combines iterate, set current feature, and log
pub fn harness_work_on_feature(
    state: &mut HarnessState,
    req: HarnessWorkOnFeatureRequest,
) -> HarnessResult<HarnessWorkOnFeatureResponse> {
    // Get feature details first
    let registry = state
        .registry
        .as_ref()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let feature = registry
        .find(&req.feature_id)
        .ok_or_else(|| crate::harness::error::HarnessError::feature_not_found(&req.feature_id))?;
    let feature_description = feature.description.clone();
    let feature_steps = feature.steps.clone();

    // Now perform the iteration
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    // Check for blocking interventions (Phase 5: Human Intervention Points)
    if session.state().has_blocking_interventions() {
        let blockers: Vec<String> = session
            .state()
            .unresolved_interventions()
            .iter()
            .filter(|i| {
                matches!(
                    i.intervention_type,
                    crate::harness::types::InterventionType::ApprovalNeeded
                        | crate::harness::types::InterventionType::DecisionPoint
                )
            })
            .map(|i| format!("[{}] {}", i.id, i.question))
            .collect();

        return Err(
            crate::harness::error::HarnessError::blocked_by_intervention(blockers.join("; ")),
        );
    }

    // Start session if still initializing
    if session.status() == SessionStatus::Initializing {
        session.start()?;
    }

    // Increment iteration
    let iteration = session.next_iteration()?;

    // Set current feature
    session.set_current_feature(&req.feature_id);

    // Log progress
    let entry = crate::harness::types::ProgressEntry::new(
        session.session_id(),
        iteration,
        ProgressMarker::FeatureStart,
        &req.summary,
    )
    .with_feature(&req.feature_id);
    state.progress.append(&entry)?;

    // Persist session state
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    Ok(HarnessWorkOnFeatureResponse {
        success: true,
        feature_id: req.feature_id,
        iteration,
        can_continue: session.can_continue(),
        feature_description,
        feature_steps,
    })
}

/// Complete a feature and get the next one - combines complete, checkpoint, and get_next
pub fn harness_complete_and_next(
    state: &mut HarnessState,
    req: HarnessCompleteAndNextRequest,
) -> HarnessResult<HarnessCompleteAndNextResponse> {
    // Check for blocking interventions first (Phase 5: Human Intervention Points)
    {
        let session = state
            .session
            .as_ref()
            .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

        if session.state().has_blocking_interventions() {
            let blockers: Vec<String> = session
                .state()
                .unresolved_interventions()
                .iter()
                .filter(|i| {
                    matches!(
                        i.intervention_type,
                        crate::harness::types::InterventionType::ApprovalNeeded
                            | crate::harness::types::InterventionType::DecisionPoint
                    )
                })
                .map(|i| format!("[{}] {}", i.id, i.question))
                .collect();

            return Err(
                crate::harness::error::HarnessError::blocked_by_intervention(blockers.join("; ")),
            );
        }
    }

    // Mark feature as passing
    let registry = state
        .registry
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    registry.mark_passing(&req.feature_id)?;
    registry.save()?;

    let session = state
        .session
        .as_ref()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    // Log completion
    state.progress.log_feature_complete(
        session.session_id(),
        session.iteration(),
        &req.feature_id,
        &req.summary,
    )?;

    // Optional checkpoint
    let checkpoint_hash = if req.checkpoint.unwrap_or(state.config.auto_checkpoint) {
        match state.git.create_checkpoint(&req.feature_id, &req.summary) {
            Ok(hash) => {
                state
                    .progress
                    .log_checkpoint(session.session_id(), session.iteration(), &hash)?;
                Some(hash)
            }
            Err(_) => None, // No changes to commit
        }
    } else {
        None
    };

    // Get next feature
    let registry = state.registry.as_ref().unwrap();
    let next = registry.next_incomplete();
    let summary = registry.summary();

    // Persist session state
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    Ok(HarnessCompleteAndNextResponse {
        success: true,
        completed_feature_id: req.feature_id,
        checkpoint_hash,
        next_feature_id: next.as_ref().map(|f| f.id.clone()),
        next_feature_description: next.as_ref().map(|f| f.description.clone()),
        completion_percent: summary.completion_percent,
        remaining_features: summary.total - summary.passing,
    })
}

/// Quick status - minimal info for rapid polling
pub fn harness_quick_status(
    state: &HarnessState,
    _req: HarnessQuickStatusRequest,
) -> HarnessResult<HarnessQuickStatusResponse> {
    let session = state
        .session
        .as_ref()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let next_feature = state
        .registry
        .as_ref()
        .and_then(|r| r.next_incomplete())
        .map(|f| f.id.clone());

    Ok(HarnessQuickStatusResponse {
        iteration: session.iteration(),
        max_iterations: session.state().max_iterations,
        can_continue: session.can_continue(),
        current_feature: session.current_feature().map(|s| s.to_string()),
        next_feature,
        session_status: session.status().to_string(),
    })
}

// ============================================================================
// Phase 3: Startup Ritual - Acknowledge Implementation
// ============================================================================

/// Acknowledge that the agent has completed the startup ritual
pub fn harness_acknowledge(
    state: &mut HarnessState,
    req: HarnessAcknowledgeRequest,
) -> HarnessResult<HarnessAcknowledgeResponse> {
    let session = state
        .session
        .as_ref()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let session_id = session.session_id().to_string();

    // Check if there's a startup checklist to acknowledge
    let checklist = state.startup_checklist.as_mut().ok_or_else(|| {
        crate::harness::error::HarnessError::session(
            "No startup checklist found - call harness_start first",
        )
    })?;

    // Generate warning if agent didn't read status
    let warning = if state.last_status_read.is_none() {
        Some("Warning: You acknowledged the startup ritual without reading status. Call harness_status first to understand session context.".to_string())
    } else {
        None
    };

    // Mark checklist as acknowledged
    checklist.acknowledge();

    // Mark specific items as reviewed if provided
    if let Some(ref items) = req.reviewed_items {
        for item_id in items {
            if let Some(item) = checklist.items.iter_mut().find(|i| i.id == *item_id) {
                item.completed = true;
            }
        }
    }

    Ok(HarnessAcknowledgeResponse {
        success: true,
        acknowledged: true,
        session_id,
        warning,
    })
}

// ============================================================================
// Phase 5: Human Intervention Points - Tool Implementations
// ============================================================================

/// Request a human intervention
pub fn harness_request_intervention(
    state: &mut HarnessState,
    req: HarnessRequestInterventionRequest,
) -> HarnessResult<HarnessRequestInterventionResponse> {
    use crate::harness::types::{InterventionType, PendingIntervention};

    // Parse intervention type from string
    let intervention_type = match req.intervention_type.to_lowercase().as_str() {
        "review_required" | "review" => InterventionType::ReviewRequired,
        "approval_needed" | "approval" => InterventionType::ApprovalNeeded,
        "decision_point" | "decision" => InterventionType::DecisionPoint,
        "clarification_needed" | "clarification" => InterventionType::ClarificationNeeded,
        _ => {
            return Err(crate::harness::error::HarnessError::validation(format!(
                "Invalid intervention type '{}'. Must be one of: review_required, approval_needed, decision_point, clarification_needed",
                req.intervention_type
            )));
        }
    };

    // Create the pending intervention
    let mut intervention = PendingIntervention::new(intervention_type, &req.question);

    // Associate with feature if provided
    if let Some(ref feature_id) = req.feature_id {
        intervention = intervention.with_feature(feature_id);
    }

    // Add options for decision points
    if let Some(options) = req.options {
        intervention = intervention.with_options(options);
    }

    let intervention_id = intervention.id.clone();
    let intervention_type_str = intervention.intervention_type.to_string();
    let question = intervention.question.clone();

    // Get session for logging and state update
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let session_id = session.session_id().to_string();
    let iteration = session.iteration();

    // Add to session state (persisted)
    session.state_mut().add_intervention(intervention);

    // Persist session state
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    // Log the intervention request
    let entry = crate::harness::types::ProgressEntry::new(
        &session_id,
        iteration,
        crate::harness::types::ProgressMarker::Progress,
        format!(
            "[INTERVENTION REQUESTED] {}: {}",
            intervention_type_str, question
        ),
    );
    state.progress.append(&entry)?;

    Ok(HarnessRequestInterventionResponse {
        success: true,
        intervention_id,
        intervention_type: intervention_type_str,
        question,
    })
}

/// Resolve a pending intervention
pub fn harness_resolve_intervention(
    state: &mut HarnessState,
    req: HarnessResolveInterventionRequest,
) -> HarnessResult<HarnessResolveInterventionResponse> {
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let session_id = session.session_id().to_string();
    let iteration = session.iteration();

    // Resolve the intervention in session state (returns the resolved intervention)
    let resolved = session
        .state_mut()
        .resolve_intervention(&req.intervention_id, &req.resolution);

    // Check if intervention was found
    let unblocked_feature = match resolved {
        Some(intervention) => intervention.feature_id.clone(),
        None => {
            return Err(crate::harness::error::HarnessError::validation(format!(
                "Intervention '{}' not found or already resolved",
                req.intervention_id
            )));
        }
    };

    // Persist session state
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    // Log the resolution
    let entry = crate::harness::types::ProgressEntry::new(
        &session_id,
        iteration,
        crate::harness::types::ProgressMarker::Progress,
        format!(
            "[INTERVENTION RESOLVED] {}: {}",
            req.intervention_id, req.resolution
        ),
    );
    state.progress.append(&entry)?;

    Ok(HarnessResolveInterventionResponse {
        success: true,
        intervention_id: req.intervention_id,
        resolved: true,
        unblocked_feature,
    })
}

/// Get pending interventions (utility function)
pub fn harness_get_pending_interventions(
    state: &HarnessState,
) -> Vec<crate::harness::types::PendingIntervention> {
    state
        .session
        .as_ref()
        .map(|s| {
            s.state()
                .unresolved_interventions()
                .into_iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

// ============================================================================
// Phase 6: Sub-Agent Delegation Tools
// ============================================================================

/// Delegate work to an isolated sub-session
///
/// Creates a sub-session with isolated context for token-heavy subtasks.
/// The sub-session inherits feature context but has isolated progress.
pub fn harness_delegate(
    state: &mut HarnessState,
    req: HarnessDelegateRequest,
) -> HarnessResult<HarnessDelegateResponse> {
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    // Check for blocking interventions
    if session.state().has_blocking_interventions() {
        let blockers: Vec<String> = session
            .state()
            .unresolved_interventions()
            .iter()
            .filter(|i| {
                matches!(
                    i.intervention_type,
                    crate::harness::types::InterventionType::ApprovalNeeded
                        | crate::harness::types::InterventionType::DecisionPoint
                )
            })
            .map(|i| format!("[{}] {}", i.id, i.question))
            .collect();

        return Err(
            crate::harness::error::HarnessError::blocked_by_intervention(blockers.join("; ")),
        );
    }

    // Verify feature exists
    if state
        .registry
        .as_ref()
        .is_none_or(|r| r.features().iter().all(|f| f.id != req.feature_id))
    {
        return Err(crate::harness::error::HarnessError::feature_not_found(
            &req.feature_id,
        ));
    }

    // Create sub-session
    let max_iterations = req.max_iterations.unwrap_or(10);
    let sub_session = crate::harness::types::SubSession::new(
        session.state().id.clone(),
        req.feature_id.clone(),
        req.task_description.clone(),
        max_iterations,
    );

    let sub_session_id = sub_session.id.clone();

    // Generate context file path
    let context_path = state
        .config
        .working_directory
        .join(format!(".harness-subsession-{}.md", &sub_session_id[..8]));
    let context_path_str = context_path.to_string_lossy().to_string();

    // Write context file for sub-agent
    let feature_info = state
        .registry
        .as_ref()
        .and_then(|r| r.features().iter().find(|f| f.id == req.feature_id))
        .map(|f| {
            format!(
                "## Feature: {}\n\n{}\n\n### Steps:\n{}",
                f.id,
                f.description,
                f.steps
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("{}. {}", i + 1, s))
                    .collect::<Vec<_>>()
                    .join("\n")
            )
        })
        .unwrap_or_default();

    let context_content = format!(
        "# Sub-Session Context\n\n\
        **Sub-Session ID:** {}\n\
        **Parent Session:** {}\n\
        **Feature:** {}\n\
        **Max Iterations:** {}\n\n\
        ## Task\n\n{}\n\n\
        {}\n\n\
        ## Instructions\n\n\
        This is an isolated sub-session. Complete the task and call `harness_complete_sub_session` \
        with a summary when done. The parent session will claim your results.\n",
        sub_session_id,
        session.state().id,
        req.feature_id,
        max_iterations,
        req.task_description,
        feature_info,
    );
    std::fs::write(&context_path, &context_content)?;

    // Add sub-session with context path
    let sub_session = sub_session.with_context_path(&context_path_str);
    session.state_mut().add_sub_session(sub_session);

    // Persist session state
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    // Log progress
    let entry = crate::harness::types::ProgressEntry::new(
        session.state().id.clone(),
        session.state().iteration,
        crate::harness::types::ProgressMarker::Progress,
        format!(
            "Delegated task to sub-session {}: {}",
            &sub_session_id[..8],
            req.task_description
        ),
    )
    .with_feature(&req.feature_id);

    state.progress.append(&entry)?;

    Ok(HarnessDelegateResponse {
        success: true,
        sub_session_id,
        feature_id: req.feature_id,
        context_path: context_path_str,
    })
}

/// Check status of a sub-session
pub fn harness_sub_session_status(
    state: &HarnessState,
    req: HarnessSubSessionStatusRequest,
) -> HarnessResult<HarnessSubSessionStatusResponse> {
    let session = state
        .session
        .as_ref()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    let sub_session = session
        .state()
        .get_sub_session(&req.sub_session_id)
        .ok_or_else(|| {
            crate::harness::error::HarnessError::session(format!(
                "Sub-session {} not found",
                req.sub_session_id
            ))
        })?;

    Ok(HarnessSubSessionStatusResponse {
        sub_session_id: sub_session.id.clone(),
        status: sub_session.status.to_string(),
        iteration: sub_session.iteration,
        max_iterations: sub_session.max_iterations,
        summary: sub_session.summary.clone(),
    })
}

/// Claim results from a completed sub-session
///
/// Incorporates sub-session work into the main session and compacts
/// the sub-session progress into a summary.
pub fn harness_claim_sub_session_result(
    state: &mut HarnessState,
    req: HarnessClaimSubSessionResultRequest,
) -> HarnessResult<HarnessClaimSubSessionResultResponse> {
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    // Get sub-session info (before mutating)
    let sub_session = session
        .state()
        .get_sub_session(&req.sub_session_id)
        .ok_or_else(|| {
            crate::harness::error::HarnessError::session(format!(
                "Sub-session {} not found",
                req.sub_session_id
            ))
        })?;

    // Check if completed or failed (both are claimable for cleanup)
    let is_completed = matches!(
        sub_session.status,
        crate::harness::types::SubSessionStatus::Completed
    );
    let is_failed = matches!(
        sub_session.status,
        crate::harness::types::SubSessionStatus::Failed
    );

    if !is_completed && !is_failed {
        return Err(crate::harness::error::HarnessError::session(format!(
            "Sub-session {} is not finished (status: {}). Only completed or failed sub-sessions can be claimed.",
            req.sub_session_id,
            sub_session.status
        )));
    }

    let feature_id = sub_session.feature_id.clone();
    let status_word = if is_failed { "failed" } else { "completed" };
    let sub_session_summary = req
        .summary
        .clone()
        .or_else(|| sub_session.summary.clone())
        .unwrap_or_else(|| format!("Sub-session {} {}", &req.sub_session_id[..8], status_word));

    // Clean up context file if it exists
    if let Some(context_path) = &sub_session.context_path {
        let _ = std::fs::remove_file(context_path);
    }

    // Log claim to progress
    let entry = crate::harness::types::ProgressEntry::new(
        session.state().id.clone(),
        session.state().iteration,
        crate::harness::types::ProgressMarker::Progress,
        format!(
            "Claimed sub-session {}: {}",
            &req.sub_session_id[..8],
            sub_session_summary
        ),
    )
    .with_feature(&feature_id);

    state.progress.append(&entry)?;

    // Persist session state
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    Ok(HarnessClaimSubSessionResultResponse {
        success: true,
        sub_session_id: req.sub_session_id,
        feature_id,
        summary: sub_session_summary,
        progress_logged: true,
    })
}

/// Complete a sub-session (called by the sub-agent)
pub fn harness_complete_sub_session(
    state: &mut HarnessState,
    sub_session_id: &str,
    summary: &str,
) -> HarnessResult<()> {
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    if !session
        .state_mut()
        .complete_sub_session(sub_session_id, summary)
    {
        return Err(crate::harness::error::HarnessError::session(format!(
            "Sub-session {} not found",
            sub_session_id
        )));
    }

    // Persist session state
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    Ok(())
}

/// Fail a sub-session (called by the sub-agent on error)
pub fn harness_fail_sub_session(
    state: &mut HarnessState,
    sub_session_id: &str,
    reason: &str,
) -> HarnessResult<()> {
    let session = state
        .session
        .as_mut()
        .ok_or_else(|| crate::harness::error::HarnessError::session("No active session"))?;

    if !session.state_mut().fail_sub_session(sub_session_id, reason) {
        return Err(crate::harness::error::HarnessError::session(format!(
            "Sub-session {} not found",
            sub_session_id
        )));
    }

    // Persist session state
    crate::harness::session::save_session_state(session.state(), &state.config.session_state_path)?;

    Ok(())
}

/// Get all active sub-sessions
pub fn harness_list_sub_sessions(state: &HarnessState) -> Vec<crate::harness::types::SubSession> {
    state
        .session
        .as_ref()
        .map(|s| {
            s.state()
                .active_sub_sessions()
                .into_iter()
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use tempfile::tempdir;

    fn setup_test_harness() -> (tempfile::TempDir, HarnessState) {
        let dir = tempdir().unwrap();

        // Initialize git repo
        Command::new("git")
            .args(["init"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.email", "test@test.com"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["config", "user.name", "Test"])
            .current_dir(dir.path())
            .output()
            .unwrap();
        std::fs::write(dir.path().join("README.md"), "# Test").unwrap();
        Command::new("git")
            .args(["add", "."])
            .current_dir(dir.path())
            .output()
            .unwrap();
        Command::new("git")
            .args(["commit", "-m", "Initial commit"])
            .current_dir(dir.path())
            .output()
            .unwrap();

        let config = HarnessConfig {
            features_path: dir.path().join("features.json"),
            progress_path: dir.path().join("claude-progress.txt"),
            session_state_path: dir.path().join(".harness-session.json"),
            working_directory: dir.path().to_path_buf(),
            max_iterations: 10,
            auto_checkpoint: true,
            require_clean_git: false,
            commit_prefix: "[harness]".to_string(),
        };

        let state = HarnessState::new(config);
        (dir, state)
    }

    #[test]
    fn test_harness_start() {
        let (_dir, mut state) = setup_test_harness();

        let response = harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(5),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();

        assert!(response.success);
        assert!(!response.is_resume);
        assert!(!response.session_id.is_empty());
    }

    #[test]
    fn test_harness_iterate() {
        let (_dir, mut state) = setup_test_harness();

        // Start session
        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(3),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();

        // First iteration
        let resp = harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Did some work".to_string(),
                feature_id: None,
            },
        )
        .unwrap();

        assert_eq!(resp.iteration, 1);
        assert!(resp.can_continue);

        // Second iteration
        let resp = harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "More work".to_string(),
                feature_id: Some("feature-1".to_string()),
            },
        )
        .unwrap();

        assert_eq!(resp.iteration, 2);
        assert!(resp.can_continue);

        // Third iteration (max)
        let resp = harness_iterate(
            &mut state,
            HarnessIterateRequest {
                summary: "Final work".to_string(),
                feature_id: None,
            },
        )
        .unwrap();

        assert_eq!(resp.iteration, 3);
        assert!(!resp.can_continue);
    }

    #[test]
    fn test_harness_status() {
        let (_dir, mut state) = setup_test_harness();

        harness_start(
            &mut state,
            HarnessStartRequest {
                max_iterations: Some(10),
                require_clean_git: None,
                auto_resume: Some(false),
            },
        )
        .unwrap();

        let status = harness_status(
            &mut state,
            HarnessStatusRequest {
                include_features: Some(true),
                include_progress: Some(true),
                max_features: None,
                max_progress_entries: None,
            },
        )
        .unwrap();

        assert!(status.session.is_some());
        assert!(!status.git_status.branch.is_empty());
    }
}
