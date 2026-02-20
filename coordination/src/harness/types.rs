//! Core types for the agent harness
//!
//! These types implement the patterns from Anthropic's engineering guide
//! for effective long-running agents.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use uuid::Uuid;

/// Session state tracking across context windows
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionState {
    /// Unique session identifier (UUID v4)
    pub id: String,

    /// Session start timestamp
    pub started_at: DateTime<Utc>,

    /// Current iteration within this session
    pub iteration: u32,

    /// Maximum allowed iterations
    pub max_iterations: u32,

    /// Currently active feature (if any)
    pub current_feature: Option<String>,

    /// Session status
    pub status: SessionStatus,

    /// Working directory for this session
    pub working_directory: PathBuf,

    /// Git commit hash at session start (for rollback)
    pub initial_commit: Option<String>,

    /// Pending human interventions (Phase 5: persisted for restart recovery)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pending_interventions: Vec<PendingIntervention>,

    /// Active sub-sessions (Phase 6: Sub-Agent Delegation)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sub_sessions: Vec<SubSession>,
}

// ============================================================================
// Structured Session Summary (Anchored Iterative)
// ============================================================================

/// Structured session summary using the anchored iterative pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuredSessionSummary {
    /// Session identifier
    pub session_id: String,
    
    /// Current session status
    pub status: SessionStatus,
    
    /// Total iterations used
    pub total_iterations: u32,
    
    /// Features worked on during this session (the anchors)
    pub features: Vec<FeatureProgressSummary>,
    
    /// Checkpoints created during this session
    pub checkpoints: Vec<CheckpointSummary>,
    
    /// Any errors encountered
    pub errors: Vec<String>,
}

/// Summary of progress on a specific feature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureProgressSummary {
    /// Feature ID
    pub feature_id: String,
    
    /// Iteration when work started
    pub start_iteration: u32,
    
    /// Iteration when work completed/failed (if applicable)
    pub end_iteration: Option<u32>,
    
    /// Current status of the feature work
    pub status: FeatureWorkStatus,
    
    /// Key iterative steps taken (summaries from progress entries)
    pub iterative_steps: Vec<String>,
}

/// Status of feature work
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureWorkStatus {
    /// Feature is currently being worked on
    InProgress,
    /// Feature was successfully completed
    Completed,
    /// Feature failed verification
    Failed,
}

/// Summary of a checkpoint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointSummary {
    /// Iteration when checkpoint was created
    pub iteration: u32,
    
    /// Commit hash
    pub commit_hash: String,
    
    /// Associated feature (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_id: Option<String>,
}

impl SessionState {
    /// Create a new session with default settings
    pub fn new(working_directory: PathBuf, max_iterations: u32) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            started_at: Utc::now(),
            iteration: 0,
            max_iterations,
            current_feature: None,
            status: SessionStatus::Initializing,
            working_directory,
            initial_commit: None,
            pending_interventions: Vec::new(),
            sub_sessions: Vec::new(),
        }
    }

    /// Increment iteration counter, returning error if max reached
    pub fn next_iteration(&mut self) -> Result<u32, u32> {
        if self.iteration >= self.max_iterations {
            Err(self.max_iterations)
        } else {
            self.iteration += 1;
            Ok(self.iteration)
        }
    }

    /// Check if session can continue
    pub fn can_continue(&self) -> bool {
        self.iteration < self.max_iterations
            && matches!(
                self.status,
                SessionStatus::Active | SessionStatus::Initializing
            )
    }

    /// Calculate elapsed time
    pub fn elapsed(&self) -> chrono::Duration {
        Utc::now().signed_duration_since(self.started_at)
    }

    /// Check if there are unresolved blocking interventions
    pub fn has_blocking_interventions(&self) -> bool {
        self.pending_interventions.iter().any(|i| {
            !i.resolved
                && matches!(
                    i.intervention_type,
                    InterventionType::ApprovalNeeded | InterventionType::DecisionPoint
                )
        })
    }

    /// Get all unresolved interventions
    pub fn unresolved_interventions(&self) -> Vec<&PendingIntervention> {
        self.pending_interventions
            .iter()
            .filter(|i| !i.resolved)
            .collect()
    }

    /// Add a pending intervention
    pub fn add_intervention(&mut self, intervention: PendingIntervention) {
        self.pending_interventions.push(intervention);
    }

    /// Resolve an intervention by ID
    pub fn resolve_intervention(
        &mut self,
        id: &str,
        resolution: &str,
    ) -> Option<&PendingIntervention> {
        if let Some(intervention) = self.pending_interventions.iter_mut().find(|i| i.id == id) {
            intervention.resolve(resolution);
            // Return reference to the resolved intervention
            self.pending_interventions.iter().find(|i| i.id == id)
        } else {
            None
        }
    }

    // --- Sub-session methods (Phase 6: Sub-Agent Delegation) ---

    /// Add a new sub-session
    pub fn add_sub_session(&mut self, sub_session: SubSession) {
        self.sub_sessions.push(sub_session);
    }

    /// Get a sub-session by ID
    pub fn get_sub_session(&self, id: &str) -> Option<&SubSession> {
        self.sub_sessions.iter().find(|s| s.id == id)
    }

    /// Get a mutable sub-session by ID
    pub fn get_sub_session_mut(&mut self, id: &str) -> Option<&mut SubSession> {
        self.sub_sessions.iter_mut().find(|s| s.id == id)
    }

    /// Get all active sub-sessions
    pub fn active_sub_sessions(&self) -> Vec<&SubSession> {
        self.sub_sessions
            .iter()
            .filter(|s| matches!(s.status, SubSessionStatus::Active))
            .collect()
    }

    /// Complete a sub-session by ID
    pub fn complete_sub_session(&mut self, id: &str, summary: impl Into<String>) -> bool {
        if let Some(sub_session) = self.get_sub_session_mut(id) {
            sub_session.complete(summary);
            true
        } else {
            false
        }
    }

    /// Fail a sub-session by ID
    pub fn fail_sub_session(&mut self, id: &str, reason: impl Into<String>) -> bool {
        if let Some(sub_session) = self.get_sub_session_mut(id) {
            sub_session.fail(reason);
            true
        } else {
            false
        }
    }

    /// Cancel a sub-session by ID
    pub fn cancel_sub_session(&mut self, id: &str) -> bool {
        if let Some(sub_session) = self.get_sub_session_mut(id) {
            sub_session.cancel();
            true
        } else {
            false
        }
    }
}

/// Session lifecycle status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    /// Session is being initialized (startup ritual)
    Initializing,
    /// Session is actively working
    Active,
    /// Session completed successfully
    Completed,
    /// Session aborted due to max iterations
    MaxIterationsReached,
    /// Session failed with error
    Failed,
    /// Session paused for manual intervention
    Paused,
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Initializing => write!(f, "initializing"),
            Self::Active => write!(f, "active"),
            Self::Completed => write!(f, "completed"),
            Self::MaxIterationsReached => write!(f, "max_iterations_reached"),
            Self::Failed => write!(f, "failed"),
            Self::Paused => write!(f, "paused"),
        }
    }
}

/// Feature specification matching Anthropic's JSON format
///
/// Example:
/// ```json
/// {
///   "id": "new-chat-button",
///   "category": "functional",
///   "description": "New chat button creates fresh conversation",
///   "steps": [
///     "Navigate to main interface",
///     "Click 'New Chat' button",
///     "Verify conversation created"
///   ],
///   "passes": false,
///   "priority": 1
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FeatureSpec {
    /// Unique feature identifier
    pub id: String,

    /// Feature category (e.g., "functional", "ui", "integration")
    pub category: FeatureCategory,

    /// Human-readable description
    pub description: String,

    /// Verification steps
    pub steps: Vec<String>,

    /// Whether the feature passes verification
    pub passes: bool,

    /// Priority level (lower = higher priority)
    #[serde(default = "default_priority")]
    pub priority: u8,

    /// When the feature was last verified
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_verified: Option<DateTime<Utc>>,

    /// Notes from verification attempts
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,

    /// Dependencies on other features
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub depends_on: Vec<String>,
}

fn default_priority() -> u8 {
    5
}

impl FeatureSpec {
    /// Create a new feature specification
    pub fn new(
        id: impl Into<String>,
        category: FeatureCategory,
        description: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            category,
            description: description.into(),
            steps: Vec::new(),
            passes: false,
            priority: default_priority(),
            last_verified: None,
            notes: Vec::new(),
            depends_on: Vec::new(),
        }
    }

    /// Add a verification step
    pub fn with_step(mut self, step: impl Into<String>) -> Self {
        self.steps.push(step.into());
        self
    }

    /// Set priority
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority;
        self
    }

    /// Mark as passing
    pub fn mark_passing(&mut self) {
        self.passes = true;
        self.last_verified = Some(Utc::now());
    }

    /// Mark as failing with note
    pub fn mark_failing(&mut self, note: impl Into<String>) {
        self.passes = false;
        self.last_verified = Some(Utc::now());
        self.notes.push(note.into());
    }
}

/// Feature categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FeatureCategory {
    /// Core functionality
    Functional,
    /// User interface
    Ui,
    /// API endpoints
    Api,
    /// Integration with external systems
    Integration,
    /// Performance requirements
    Performance,
    /// Security requirements
    Security,
    /// Documentation
    Documentation,
    /// Testing infrastructure
    Testing,
}

impl std::fmt::Display for FeatureCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Functional => write!(f, "functional"),
            Self::Ui => write!(f, "ui"),
            Self::Api => write!(f, "api"),
            Self::Integration => write!(f, "integration"),
            Self::Performance => write!(f, "performance"),
            Self::Security => write!(f, "security"),
            Self::Documentation => write!(f, "documentation"),
            Self::Testing => write!(f, "testing"),
        }
    }
}

/// Progress entry for claude-progress.txt
///
/// Format: `[TIMESTAMP] [SESSION_ID] [ITER:N] [MARKER] summary`
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ProgressEntry {
    /// Entry timestamp
    pub timestamp: DateTime<Utc>,

    /// Session ID this entry belongs to
    pub session_id: String,

    /// Iteration number within session
    pub iteration: u32,

    /// Entry marker/type
    pub marker: ProgressMarker,

    /// Human-readable summary
    pub summary: String,

    /// Associated feature (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_id: Option<String>,

    /// Additional structured data
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl ProgressEntry {
    /// Create a new progress entry
    pub fn new(
        session_id: impl Into<String>,
        iteration: u32,
        marker: ProgressMarker,
        summary: impl Into<String>,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            session_id: session_id.into(),
            iteration,
            marker,
            summary: summary.into(),
            feature_id: None,
            metadata: serde_json::Map::new(),
        }
    }

    /// Add feature association
    pub fn with_feature(mut self, feature_id: impl Into<String>) -> Self {
        self.feature_id = Some(feature_id.into());
        self
    }

    /// Add metadata key-value pair
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Format as log line for claude-progress.txt
    pub fn to_log_line(&self) -> String {
        let feature_part = self
            .feature_id
            .as_ref()
            .map(|f| format!(" [{}]", f))
            .unwrap_or_default();

        let short_id = if self.session_id.len() >= 8 {
            &self.session_id[..8]
        } else {
            &self.session_id
        };

        format!(
            "[{}] [{}] [ITER:{}] [{}]{} {}",
            self.timestamp.format("%Y-%m-%d %H:%M:%S UTC"),
            short_id,
            self.iteration,
            self.marker,
            feature_part,
            self.summary
        )
    }

    /// Parse from log line
    pub fn from_log_line(line: &str) -> Option<Self> {
        // Basic parsing - production would use regex or nom
        // Format: [TIMESTAMP] [SESSION] [ITER:N] [MARKER] [FEATURE?] summary
        let parts: Vec<&str> = line.splitn(6, "] ").collect();
        if parts.len() < 5 {
            return None;
        }

        // Extract timestamp
        let timestamp_str = parts[0].trim_start_matches('[');
        let timestamp = DateTime::parse_from_str(
            &format!("{} +0000", timestamp_str),
            "%Y-%m-%d %H:%M:%S UTC %z",
        )
        .ok()?
        .with_timezone(&Utc);

        // Extract session ID
        let session_id = parts[1].trim_start_matches('[').to_string();

        // Extract iteration
        let iter_part = parts[2].trim_start_matches("[ITER:");
        let iteration: u32 = iter_part.parse().ok()?;

        // Extract marker
        let marker_str = parts[3].trim_start_matches('[');
        let marker = ProgressMarker::from_str(marker_str)?;

        // Remaining is summary (possibly with feature)
        let remaining = parts.get(4).unwrap_or(&"");
        let (feature_id, summary) = if remaining.starts_with('[') {
            if let Some(end) = remaining.find(']') {
                let feature = remaining[1..end].to_string();
                let sum = remaining[end + 1..].trim().to_string();
                (Some(feature), sum)
            } else {
                (None, remaining.to_string())
            }
        } else {
            (None, remaining.to_string())
        };

        Some(Self {
            timestamp,
            session_id,
            iteration,
            marker,
            summary,
            feature_id,
            metadata: serde_json::Map::new(),
        })
    }
}

/// Progress entry markers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum ProgressMarker {
    /// Session started
    SessionStart,
    /// Feature work started
    FeatureStart,
    /// Feature completed successfully
    FeatureComplete,
    /// Feature failed verification
    FeatureFailed,
    /// Checkpoint created
    Checkpoint,
    /// Rollback performed
    Rollback,
    /// Session ended normally
    SessionEnd,
    /// Session aborted
    SessionAbort,
    /// General progress note
    Progress,
    /// Error occurred
    Error,
}

impl ProgressMarker {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "SESSION_START" => Some(Self::SessionStart),
            "FEATURE_START" => Some(Self::FeatureStart),
            "FEATURE_COMPLETE" => Some(Self::FeatureComplete),
            "FEATURE_FAILED" => Some(Self::FeatureFailed),
            "CHECKPOINT" => Some(Self::Checkpoint),
            "ROLLBACK" => Some(Self::Rollback),
            "SESSION_END" => Some(Self::SessionEnd),
            "SESSION_ABORT" => Some(Self::SessionAbort),
            "PROGRESS" => Some(Self::Progress),
            "ERROR" => Some(Self::Error),
            _ => None,
        }
    }
}

impl std::fmt::Display for ProgressMarker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SessionStart => write!(f, "SESSION_START"),
            Self::FeatureStart => write!(f, "FEATURE_START"),
            Self::FeatureComplete => write!(f, "FEATURE_COMPLETE"),
            Self::FeatureFailed => write!(f, "FEATURE_FAILED"),
            Self::Checkpoint => write!(f, "CHECKPOINT"),
            Self::Rollback => write!(f, "ROLLBACK"),
            Self::SessionEnd => write!(f, "SESSION_END"),
            Self::SessionAbort => write!(f, "SESSION_ABORT"),
            Self::Progress => write!(f, "PROGRESS"),
            Self::Error => write!(f, "ERROR"),
        }
    }
}

/// Harness configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HarnessConfig {
    /// Path to features.json registry
    pub features_path: PathBuf,

    /// Path to claude-progress.txt
    pub progress_path: PathBuf,

    /// Path to session state file for persistence
    pub session_state_path: PathBuf,

    /// Working directory (project root)
    pub working_directory: PathBuf,

    /// Maximum iterations per session
    pub max_iterations: u32,

    /// Whether to auto-checkpoint after each feature
    pub auto_checkpoint: bool,

    /// Whether to require clean git state before starting
    pub require_clean_git: bool,

    /// Commit message prefix for harness commits
    pub commit_prefix: String,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            features_path: PathBuf::from("features.json"),
            progress_path: PathBuf::from("claude-progress.txt"),
            session_state_path: PathBuf::from(".harness-session.json"),
            working_directory: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            max_iterations: 20,
            auto_checkpoint: true,
            require_clean_git: false,
            commit_prefix: "[harness]".to_string(),
        }
    }
}

impl HarnessConfig {
    /// Create config from environment variables
    pub fn from_env() -> Self {
        let mut config = Self::default();

        if let Ok(path) = std::env::var("HARNESS_FEATURES_PATH") {
            config.features_path = PathBuf::from(path);
        }
        if let Ok(path) = std::env::var("HARNESS_PROGRESS_PATH") {
            config.progress_path = PathBuf::from(path);
        }
        if let Ok(path) = std::env::var("HARNESS_SESSION_STATE_PATH") {
            config.session_state_path = PathBuf::from(path);
        }
        if let Ok(dir) = std::env::var("HARNESS_WORKING_DIR") {
            config.working_directory = PathBuf::from(dir);
        }
        if let Ok(max) = std::env::var("HARNESS_MAX_ITERATIONS") {
            if let Ok(n) = max.parse() {
                config.max_iterations = n;
            }
        }
        if let Ok(val) = std::env::var("HARNESS_AUTO_CHECKPOINT") {
            config.auto_checkpoint = val.to_lowercase() == "true" || val == "1";
        }
        if let Ok(val) = std::env::var("HARNESS_REQUIRE_CLEAN_GIT") {
            config.require_clean_git = val.to_lowercase() == "true" || val == "1";
        }
        if let Ok(prefix) = std::env::var("HARNESS_COMMIT_PREFIX") {
            config.commit_prefix = prefix;
        }

        config
    }

    /// Resolve paths relative to working directory
    pub fn resolve_paths(&mut self) {
        if self.features_path.is_relative() {
            self.features_path = self.working_directory.join(&self.features_path);
        }
        if self.progress_path.is_relative() {
            self.progress_path = self.working_directory.join(&self.progress_path);
        }
        if self.session_state_path.is_relative() {
            self.session_state_path = self.working_directory.join(&self.session_state_path);
        }
    }
}

/// Startup context gathered during initialization ritual
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupContext {
    /// Current working directory
    pub working_directory: PathBuf,

    /// Recent git commits
    pub recent_commits: Vec<GitCommitInfo>,

    /// Current branch name
    pub current_branch: String,

    /// Current commit hash
    pub current_commit: String,

    /// Whether there are uncommitted changes
    pub has_uncommitted_changes: bool,

    /// Last session state (if resuming)
    pub last_session: Option<SessionState>,

    /// Feature registry summary
    pub feature_summary: FeatureSummary,

    /// Next feature to work on
    pub next_feature: Option<String>,

    /// Recent progress entries
    pub recent_progress: Vec<ProgressEntry>,
}

/// Summary of feature registry state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FeatureSummary {
    /// Total features in registry
    pub total: usize,

    /// Features passing verification
    pub passing: usize,

    /// Features failing verification
    pub failing: usize,

    /// Features not yet verified
    pub pending: usize,

    /// Completion percentage
    pub completion_percent: f32,
}

impl FeatureSummary {
    /// Calculate summary from feature list
    pub fn from_features(features: &[FeatureSpec]) -> Self {
        let total = features.len();
        let passing = features.iter().filter(|f| f.passes).count();
        let failing = features
            .iter()
            .filter(|f| !f.passes && f.last_verified.is_some())
            .count();
        let pending = total - passing - failing;
        let completion_percent = if total > 0 {
            (passing as f32 / total as f32) * 100.0
        } else {
            0.0
        };

        Self {
            total,
            passing,
            failing,
            pending,
            completion_percent,
        }
    }
}

/// Git commit information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitCommitInfo {
    /// Commit hash (short)
    pub hash: String,

    /// Commit message (first line)
    pub message: String,

    /// Commit timestamp
    pub timestamp: Option<DateTime<Utc>>,

    /// Whether this is a harness checkpoint
    pub is_harness_checkpoint: bool,
}

// ============================================================================
// Pagination Types (Phase 2: Token Budget Management)
// ============================================================================

/// Pagination/truncation notice for large responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TruncationNotice {
    /// Whether the response was truncated
    pub truncated: bool,

    /// Total count of items before truncation
    pub total_count: usize,

    /// Number of items returned
    pub returned_count: usize,

    /// Tool to call for full data (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_data_tool: Option<String>,
}

impl TruncationNotice {
    /// Create a notice indicating no truncation
    pub fn none(count: usize) -> Self {
        Self {
            truncated: false,
            total_count: count,
            returned_count: count,
            full_data_tool: None,
        }
    }

    /// Create a notice for truncated data
    pub fn truncated(total: usize, returned: usize, tool: impl Into<String>) -> Self {
        Self {
            truncated: true,
            total_count: total,
            returned_count: returned,
            full_data_tool: Some(tool.into()),
        }
    }
}

/// Compact progress entry for summaries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactProgressEntry {
    /// Iteration number
    pub iteration: u32,

    /// Marker type
    pub marker: String,

    /// Summary text (may be truncated)
    pub summary: String,

    /// Associated feature (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_id: Option<String>,
}

impl From<&ProgressEntry> for CompactProgressEntry {
    fn from(entry: &ProgressEntry) -> Self {
        let summary = if entry.summary.len() > 100 {
            format!("{}...", &entry.summary[..97])
        } else {
            entry.summary.clone()
        };

        Self {
            iteration: entry.iteration,
            marker: entry.marker.to_string(),
            summary,
            feature_id: entry.feature_id.clone(),
        }
    }
}

// ============================================================================
// Startup Checklist (Phase 3: Startup Ritual Enforcement)
// ============================================================================

/// Startup checklist item that agent should verify
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChecklistItem {
    /// Item identifier
    pub id: String,

    /// Human-readable description
    pub description: String,

    /// Whether this item has been completed
    pub completed: bool,
}

impl ChecklistItem {
    /// Create a new checklist item
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            completed: false,
        }
    }

    /// Mark as completed
    pub fn complete(mut self) -> Self {
        self.completed = true;
        self
    }
}

/// Startup checklist for agent to acknowledge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StartupChecklist {
    /// Whether the agent has acknowledged the checklist
    pub acknowledged: bool,

    /// Checklist items to verify
    pub items: Vec<ChecklistItem>,

    /// Timestamp when checklist was created
    pub created_at: DateTime<Utc>,

    /// Timestamp when acknowledged (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acknowledged_at: Option<DateTime<Utc>>,
}

impl StartupChecklist {
    /// Create a new startup checklist with default items
    pub fn new() -> Self {
        Self {
            acknowledged: false,
            items: vec![
                ChecklistItem::new(
                    "read_progress",
                    "Review recent progress entries to understand session history",
                ),
                ChecklistItem::new(
                    "check_features",
                    "Check feature summary for completion status",
                ),
                ChecklistItem::new(
                    "verify_git",
                    "Verify git state (branch, uncommitted changes)",
                ),
                ChecklistItem::new("identify_next", "Identify next feature to work on"),
            ],
            created_at: Utc::now(),
            acknowledged_at: None,
        }
    }

    /// Mark checklist as acknowledged
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
        self.acknowledged_at = Some(Utc::now());
        for item in &mut self.items {
            item.completed = true;
        }
    }
}

impl Default for StartupChecklist {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Human Intervention (Phase 5: Human Intervention Points)
// ============================================================================

/// Type of intervention required
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InterventionType {
    /// Human should review before proceeding
    ReviewRequired,

    /// Explicit approval needed for destructive action
    ApprovalNeeded,

    /// Multiple valid paths, human chooses
    DecisionPoint,

    /// Need clarification on requirements
    ClarificationNeeded,
}

impl std::fmt::Display for InterventionType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ReviewRequired => write!(f, "review_required"),
            Self::ApprovalNeeded => write!(f, "approval_needed"),
            Self::DecisionPoint => write!(f, "decision_point"),
            Self::ClarificationNeeded => write!(f, "clarification_needed"),
        }
    }
}

/// A pending intervention request
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PendingIntervention {
    /// Unique intervention ID
    pub id: String,

    /// Type of intervention
    pub intervention_type: InterventionType,

    /// Associated feature (if any)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature_id: Option<String>,

    /// Question or description for the human
    pub question: String,

    /// Available options (if decision point)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub options: Vec<String>,

    /// When the intervention was created
    pub created_at: DateTime<Utc>,

    /// Whether the intervention has been resolved
    pub resolved: bool,

    /// Resolution (if resolved)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,

    /// When resolved (if resolved)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolved_at: Option<DateTime<Utc>>,
}

impl PendingIntervention {
    /// Create a new pending intervention
    pub fn new(intervention_type: InterventionType, question: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            intervention_type,
            feature_id: None,
            question: question.into(),
            options: Vec::new(),
            created_at: Utc::now(),
            resolved: false,
            resolution: None,
            resolved_at: None,
        }
    }

    /// Associate with a feature
    pub fn with_feature(mut self, feature_id: impl Into<String>) -> Self {
        self.feature_id = Some(feature_id.into());
        self
    }

    /// Add options for decision points
    pub fn with_options(mut self, options: Vec<String>) -> Self {
        self.options = options;
        self
    }

    /// Resolve the intervention
    pub fn resolve(&mut self, resolution: impl Into<String>) {
        self.resolved = true;
        self.resolution = Some(resolution.into());
        self.resolved_at = Some(Utc::now());
    }
}

// ============================================================================
// Sub-Session (Phase 6: Sub-Agent Isolation)
// ============================================================================

/// Status of a sub-session
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SubSessionStatus {
    /// Sub-session is active
    Active,

    /// Sub-session completed successfully
    Completed,

    /// Sub-session failed
    Failed,

    /// Sub-session was cancelled
    Cancelled,
}

impl std::fmt::Display for SubSessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Completed => write!(f, "completed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Sub-session for isolated sub-agent work
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SubSession {
    /// Unique sub-session ID
    pub id: String,

    /// Parent session ID
    pub parent_session_id: String,

    /// Feature being worked on
    pub feature_id: String,

    /// Task description
    pub task_description: String,

    /// Sub-session status
    pub status: SubSessionStatus,

    /// Summary of work done (populated on completion)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<String>,

    /// Current iteration within sub-session
    pub iteration: u32,

    /// Maximum iterations for sub-session
    pub max_iterations: u32,

    /// When the sub-session started
    pub started_at: DateTime<Utc>,

    /// When the sub-session completed (if complete)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,

    /// Path to context file for the sub-session
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_path: Option<String>,
}

impl SubSession {
    /// Create a new sub-session
    pub fn new(
        parent_session_id: impl Into<String>,
        feature_id: impl Into<String>,
        task_description: impl Into<String>,
        max_iterations: u32,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            parent_session_id: parent_session_id.into(),
            feature_id: feature_id.into(),
            task_description: task_description.into(),
            status: SubSessionStatus::Active,
            summary: None,
            iteration: 0,
            max_iterations,
            started_at: Utc::now(),
            completed_at: None,
            context_path: None,
        }
    }

    /// Set context path
    pub fn with_context_path(mut self, path: impl Into<String>) -> Self {
        self.context_path = Some(path.into());
        self
    }

    /// Complete the sub-session
    pub fn complete(&mut self, summary: impl Into<String>) {
        self.status = SubSessionStatus::Completed;
        self.summary = Some(summary.into());
        self.completed_at = Some(Utc::now());
    }

    /// Fail the sub-session
    pub fn fail(&mut self, reason: impl Into<String>) {
        self.status = SubSessionStatus::Failed;
        self.summary = Some(format!("FAILED: {}", reason.into()));
        self.completed_at = Some(Utc::now());
    }

    /// Cancel the sub-session
    pub fn cancel(&mut self) {
        self.status = SubSessionStatus::Cancelled;
        self.completed_at = Some(Utc::now());
    }

    /// Increment iteration
    pub fn next_iteration(&mut self) -> Result<u32, u32> {
        if self.iteration >= self.max_iterations {
            Err(self.max_iterations)
        } else {
            self.iteration += 1;
            Ok(self.iteration)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_state_new() {
        let session = SessionState::new(PathBuf::from("/tmp/test"), 20);
        assert_eq!(session.iteration, 0);
        assert_eq!(session.max_iterations, 20);
        assert!(session.can_continue());
        assert_eq!(session.status, SessionStatus::Initializing);
    }

    #[test]
    fn test_session_next_iteration() {
        let mut session = SessionState::new(PathBuf::from("/tmp/test"), 3);
        assert_eq!(session.next_iteration(), Ok(1));
        assert_eq!(session.next_iteration(), Ok(2));
        assert_eq!(session.next_iteration(), Ok(3));
        assert_eq!(session.next_iteration(), Err(3));
    }

    #[test]
    fn test_feature_spec_builder() {
        let feature = FeatureSpec::new("test-feature", FeatureCategory::Functional, "Test feature")
            .with_step("Step 1")
            .with_step("Step 2")
            .with_priority(1);

        assert_eq!(feature.id, "test-feature");
        assert_eq!(feature.steps.len(), 2);
        assert_eq!(feature.priority, 1);
        assert!(!feature.passes);
    }

    #[test]
    fn test_feature_mark_passing() {
        let mut feature = FeatureSpec::new("test", FeatureCategory::Functional, "Test");
        assert!(!feature.passes);
        assert!(feature.last_verified.is_none());

        feature.mark_passing();
        assert!(feature.passes);
        assert!(feature.last_verified.is_some());
    }

    #[test]
    fn test_progress_entry_to_log_line() {
        let entry = ProgressEntry::new(
            "abc12345-6789",
            1,
            ProgressMarker::SessionStart,
            "Started work",
        )
        .with_feature("my-feature");

        let line = entry.to_log_line();
        assert!(line.contains("[abc12345]"));
        assert!(line.contains("[ITER:1]"));
        assert!(line.contains("[SESSION_START]"));
        assert!(line.contains("[my-feature]"));
        assert!(line.contains("Started work"));
    }

    #[test]
    fn test_harness_config_default() {
        let config = HarnessConfig::default();
        assert_eq!(config.max_iterations, 20);
        assert!(config.auto_checkpoint);
        assert!(!config.require_clean_git);
    }

    #[test]
    fn test_feature_summary() {
        let features = vec![
            {
                let mut f = FeatureSpec::new("f1", FeatureCategory::Functional, "Feature 1");
                f.passes = true;
                f
            },
            {
                let mut f = FeatureSpec::new("f2", FeatureCategory::Functional, "Feature 2");
                f.passes = true;
                f
            },
            FeatureSpec::new("f3", FeatureCategory::Functional, "Feature 3"),
        ];

        let summary = FeatureSummary::from_features(&features);
        assert_eq!(summary.total, 3);
        assert_eq!(summary.passing, 2);
        assert_eq!(summary.pending, 1);
        assert!((summary.completion_percent - 66.67).abs() < 1.0);
    }

    #[test]
    fn test_session_serialization_roundtrip() {
        let session = SessionState::new(PathBuf::from("/tmp/test"), 20);
        let json = serde_json::to_string(&session).unwrap();
        let restored: SessionState = serde_json::from_str(&json).unwrap();
        assert_eq!(session.id, restored.id);
        assert_eq!(session.max_iterations, restored.max_iterations);
    }

    #[test]
    fn test_feature_serialization_roundtrip() {
        let feature = FeatureSpec::new("test", FeatureCategory::Api, "Test API")
            .with_step("Call endpoint")
            .with_priority(2);

        let json = serde_json::to_string(&feature).unwrap();
        let restored: FeatureSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(feature.id, restored.id);
        assert_eq!(feature.category, restored.category);
        assert_eq!(feature.steps, restored.steps);
    }
}
