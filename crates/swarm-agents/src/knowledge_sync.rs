//! Automated knowledge sync: captures resolutions, error patterns, and
//! session retrospectives into NotebookLM for institutional memory.
//!
//! # Architecture
//!
//! Low-level functions (`capture_resolution`, `capture_error_pattern`) are called
//! directly from the orchestrator's outcome handler. The higher-level
//! [`KnowledgeSyncService`] wraps these with routing rules, structured content
//! formatting, and title-based deduplication.

use anyhow::{Context, Result};
use coordination::SessionRetrospective;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use crate::notebook_bridge::KnowledgeBase;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Where the capture originated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptureSource {
    /// Generated from a post-session retrospective.
    Retrospective,
    /// Generated from a session summary (iteration-level).
    Summary,
}

/// Which notebook to route the capture to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotebookTarget {
    /// Architecture decisions, successful patterns, resolution summaries.
    ProjectBrain,
    /// Error patterns, debugging playbooks, tricky-bug resolutions.
    DebuggingKb,
}

impl NotebookTarget {
    /// The notebook role string used by [`KnowledgeBase`] methods.
    pub fn role(&self) -> &'static str {
        match self {
            NotebookTarget::ProjectBrain => "project_brain",
            NotebookTarget::DebuggingKb => "debugging_kb",
        }
    }
}

/// A structured knowledge capture ready for upload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeCapture {
    /// How this capture was produced.
    pub source: CaptureSource,
    /// Which notebook it should be uploaded to.
    pub target: NotebookTarget,
    /// Formatted markdown content.
    pub content: String,
    /// Title used for deduplication and source identification.
    pub title: String,
    /// Searchable tags for categorization.
    pub tags: Vec<String>,
}

// ---------------------------------------------------------------------------
// KnowledgeSyncService
// ---------------------------------------------------------------------------

/// Minimum iterations that qualify a resolution as a "tricky bug"
/// worth capturing to the Debugging KB.
const TRICKY_BUG_THRESHOLD: u32 = 3;

/// High-level service for routing structured knowledge captures to the
/// appropriate NotebookLM notebooks.
///
/// Wraps a [`KnowledgeBase`] backend and applies routing rules:
/// - All successful resolutions → Project Brain
/// - Tricky bugs (3+ iterations) → also Debugging KB
/// - Observations and recommendations → Project Brain
pub struct KnowledgeSyncService<'a> {
    kb: &'a dyn KnowledgeBase,
    dedup_enabled: bool,
}

impl<'a> KnowledgeSyncService<'a> {
    /// Create a new service wrapping the given knowledge base.
    pub fn new(kb: &'a dyn KnowledgeBase) -> Self {
        Self {
            kb,
            dedup_enabled: true,
        }
    }

    /// Disable deduplication (useful for testing or forced re-uploads).
    pub fn without_dedup(mut self) -> Self {
        self.dedup_enabled = false;
        self
    }

    /// Process a session retrospective and route captures to the appropriate notebooks.
    ///
    /// Routing rules:
    /// 1. Successful sessions → resolution summary to Project Brain
    /// 2. Sessions with 3+ iterations → error pattern to Debugging KB
    /// 3. Observations and recommendations → Project Brain
    ///
    /// Returns the list of captures that were successfully uploaded.
    pub fn capture_from_retrospective(
        &self,
        retro: &SessionRetrospective,
        issue_id: &str,
        issue_title: &str,
    ) -> Vec<KnowledgeCapture> {
        let mut uploaded = Vec::new();

        // Rule 1: Successful resolutions → Project Brain
        if retro.status == coordination::SessionStatus::Completed {
            let capture = KnowledgeCapture {
                source: CaptureSource::Retrospective,
                target: NotebookTarget::ProjectBrain,
                title: format!("Retrospective: {issue_id}"),
                content: format_resolution_retrospective(retro, issue_id, issue_title),
                tags: vec![
                    "resolution".into(),
                    issue_id.into(),
                    format!("iterations:{}", retro.iterations_used),
                ],
            };

            if self.try_upload(&capture) {
                uploaded.push(capture);
            }
        }

        // Rule 2: Tricky bugs (3+ iterations) → Debugging KB
        if retro.iterations_used >= TRICKY_BUG_THRESHOLD {
            let capture = KnowledgeCapture {
                source: CaptureSource::Retrospective,
                target: NotebookTarget::DebuggingKb,
                title: format!("Debug Pattern: {issue_id}"),
                content: format_debug_retrospective(retro, issue_id, issue_title),
                tags: vec![
                    "error-pattern".into(),
                    issue_id.into(),
                    format!("iterations:{}", retro.iterations_used),
                ],
            };

            if self.try_upload(&capture) {
                uploaded.push(capture);
            }
        }

        // Rule 3: Non-empty observations/recommendations → Project Brain
        if !retro.observations.is_empty() || !retro.recommendations.is_empty() {
            let capture = KnowledgeCapture {
                source: CaptureSource::Retrospective,
                target: NotebookTarget::ProjectBrain,
                title: format!("Session Insights: {issue_id}"),
                content: format_insights(retro, issue_id),
                tags: vec!["insights".into(), issue_id.into()],
            };

            // Skip if identical to the resolution capture (both go to project_brain)
            let dominated = retro.status == coordination::SessionStatus::Completed
                && retro.observations.is_empty();
            if !dominated && self.try_upload(&capture) {
                uploaded.push(capture);
            }
        }

        info!(
            issue_id,
            captures = uploaded.len(),
            "Knowledge sync: captured retrospective"
        );

        uploaded
    }

    /// Attempt to upload a capture, returning true on success.
    ///
    /// Checks for duplicates first (if enabled), then uploads.
    /// Failures are logged but never propagated — knowledge capture is non-fatal.
    fn try_upload(&self, capture: &KnowledgeCapture) -> bool {
        let role = capture.target.role();

        // Deduplication: query for existing content with same title
        if self.dedup_enabled {
            if let Some(true) = self.title_exists(role, &capture.title) {
                debug!(
                    title = %capture.title,
                    target = role,
                    "Skipping duplicate capture"
                );
                return false;
            }
        }

        match self
            .kb
            .add_source_text(role, &capture.title, &capture.content)
        {
            Ok(()) => {
                info!(title = %capture.title, target = role, "Captured to notebook");
                true
            }
            Err(e) => {
                warn!(
                    title = %capture.title,
                    target = role,
                    "Failed to capture (non-fatal): {e}"
                );
                false
            }
        }
    }

    /// Check if a source with the given title already exists in the notebook.
    ///
    /// Uses a title-prefix query. Returns `Some(true)` if found, `Some(false)` if not,
    /// `None` if the query itself failed (treated as "not found" for upload purposes).
    fn title_exists(&self, role: &str, title: &str) -> Option<bool> {
        match self.kb.query(role, &format!("source titled \"{title}\"")) {
            Ok(response) => {
                // If the response mentions the exact title, consider it a duplicate
                let found = !response.is_empty() && response.contains(title);
                Some(found)
            }
            Err(e) => {
                debug!(
                    role,
                    title, "Dedup query failed (proceeding with upload): {e}"
                );
                None
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Content formatters
// ---------------------------------------------------------------------------

/// Format a successful resolution retrospective for Project Brain.
fn format_resolution_retrospective(
    retro: &SessionRetrospective,
    issue_id: &str,
    issue_title: &str,
) -> String {
    let mut md = format!(
        "## Retrospective: {issue_id} — {issue_title}\n\n\
         - **Status:** {:?}\n\
         - **Iterations:** {} / {} ({:.0}% efficiency)\n\
         - **Features:** {} attempted, {} completed, {} failed\n\
         - **Checkpoints:** {}, Rollbacks: {}\n\
         - **Errors encountered:** {}\n",
        retro.status,
        retro.iterations_used,
        retro.max_iterations,
        retro.iteration_efficiency_pct,
        retro.features_attempted,
        retro.features_completed,
        retro.features_failed,
        retro.checkpoints_created,
        retro.rollbacks_performed,
        retro.errors_encountered,
    );

    if !retro.observations.is_empty() {
        md.push_str("\n### Observations\n");
        for obs in &retro.observations {
            md.push_str(&format!("- {obs}\n"));
        }
    }

    if !retro.recommendations.is_empty() {
        md.push_str("\n### Recommendations\n");
        for rec in &retro.recommendations {
            md.push_str(&format!("- {rec}\n"));
        }
    }

    md
}

/// Format a debug pattern retrospective for Debugging KB.
fn format_debug_retrospective(
    retro: &SessionRetrospective,
    issue_id: &str,
    issue_title: &str,
) -> String {
    let mut md = format!(
        "## Debug Pattern: {issue_id} — {issue_title}\n\n\
         - **Iterations to resolve:** {} / {}\n\
         - **Errors encountered:** {}\n\
         - **Rollbacks:** {}\n\
         - **Final status:** {:?}\n",
        retro.iterations_used,
        retro.max_iterations,
        retro.errors_encountered,
        retro.rollbacks_performed,
        retro.status,
    );

    if !retro.observations.is_empty() {
        md.push_str("\n### What happened\n");
        for obs in &retro.observations {
            md.push_str(&format!("- {obs}\n"));
        }
    }

    if !retro.recommendations.is_empty() {
        md.push_str("\n### Lessons learned\n");
        for rec in &retro.recommendations {
            md.push_str(&format!("- {rec}\n"));
        }
    }

    md
}

/// Format session insights (observations + recommendations) for Project Brain.
fn format_insights(retro: &SessionRetrospective, issue_id: &str) -> String {
    let mut md = format!("## Session Insights: {issue_id}\n\n");

    if !retro.observations.is_empty() {
        md.push_str("### Observations\n");
        for obs in &retro.observations {
            md.push_str(&format!("- {obs}\n"));
        }
    }

    if !retro.recommendations.is_empty() {
        md.push_str("\n### Recommendations\n");
        for rec in &retro.recommendations {
            md.push_str(&format!("- {rec}\n"));
        }
    }

    md
}

// ---------------------------------------------------------------------------
// Low-level capture functions (called directly by orchestrator)
// ---------------------------------------------------------------------------

/// Generate and upload a resolution summary to the Project Brain notebook.
///
/// Called after a successful issue resolution. Includes issue metadata
/// and implementation details for future context.
pub fn capture_resolution(
    kb: &dyn KnowledgeBase,
    issue_id: &str,
    issue_title: &str,
    iterations: u32,
    tier: &str,
    files_touched: &[String],
) -> Result<()> {
    let summary = format!(
        "## Resolution: {issue_id} — {issue_title}\n\n\
         - **Issue:** {issue_id}\n\
         - **Iterations:** {iterations}\n\
         - **Tier:** {tier}\n\
         - **Files:** {files}\n\
         - **Status:** Resolved by swarm orchestrator\n",
        files = if files_touched.is_empty() {
            "(none recorded)".to_string()
        } else {
            files_touched.join(", ")
        },
    );

    let title = format!("Resolution: {issue_id}");
    match kb.add_source_text("project_brain", &title, &summary) {
        Ok(()) => {
            info!(issue_id, "Captured resolution to Project Brain");
            Ok(())
        }
        Err(e) => {
            warn!(issue_id, "Failed to capture resolution (non-fatal): {e}");
            Ok(()) // Non-fatal — don't break the pipeline
        }
    }
}

/// Capture an error pattern + solution to the Debugging KB.
///
/// Called when a tricky bug took 3+ iterations to resolve.
/// This builds institutional memory about recurring error patterns.
pub fn capture_error_pattern(
    kb: &dyn KnowledgeBase,
    issue_id: &str,
    error_categories: &[String],
    iterations: u32,
    resolution_summary: &str,
) -> Result<()> {
    let pattern = format!(
        "## Error Pattern from {issue_id}\n\n\
         - **Categories:** {cats}\n\
         - **Iterations to resolve:** {iterations}\n\n\
         ### Resolution\n\
         {resolution_summary}\n",
        cats = if error_categories.is_empty() {
            "unknown".to_string()
        } else {
            error_categories.join(", ")
        },
    );

    let title = format!("Error Pattern: {issue_id}");
    match kb.add_source_text("debugging_kb", &title, &pattern) {
        Ok(()) => {
            info!(issue_id, "Captured error pattern to Debugging KB");
            Ok(())
        }
        Err(e) => {
            warn!(issue_id, "Failed to capture error pattern (non-fatal): {e}");
            Ok(()) // Non-fatal
        }
    }
}

/// Sync the codebase to the Codebase notebook via repomix.
///
/// Runs `repomix --style markdown` to pack the repo, then uploads the result.
/// This is expensive and should be run sparingly (e.g., via CLI flag).
pub fn sync_codebase_notebook(kb: &dyn KnowledgeBase, repo_root: &std::path::Path) -> Result<()> {
    let output_path = std::env::temp_dir().join("beefcake-swarm-repomix.md");

    info!("Running repomix to pack codebase...");
    let repomix = std::process::Command::new("repomix")
        .args([
            "--style",
            "markdown",
            "--output",
            &output_path.to_string_lossy(),
        ])
        .current_dir(repo_root)
        .output()
        .context("Failed to run repomix. Is it installed? (npm i -g repomix)")?;

    if !repomix.status.success() {
        let stderr = String::from_utf8_lossy(&repomix.stderr);
        anyhow::bail!("repomix failed: {stderr}");
    }

    info!(path = %output_path.display(), "Uploading packed codebase to Codebase notebook");
    kb.add_source_file("codebase", &output_path.to_string_lossy())?;

    // Clean up
    let _ = std::fs::remove_file(&output_path);

    info!("Codebase sync complete");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notebook_bridge::tests::MockKnowledgeBase;
    use coordination::SessionStatus;

    /// Build a minimal retrospective for testing with the given overrides.
    fn test_retrospective(
        status: SessionStatus,
        iterations: u32,
        observations: Vec<String>,
        recommendations: Vec<String>,
    ) -> SessionRetrospective {
        SessionRetrospective {
            session_id: "test-session-001".into(),
            status,
            iterations_used: iterations,
            max_iterations: 10,
            iteration_efficiency_pct: (iterations as f32 / 10.0) * 100.0,
            features_attempted: 1,
            features_completed: if status == SessionStatus::Completed {
                1
            } else {
                0
            },
            features_failed: if status == SessionStatus::Failed {
                1
            } else {
                0
            },
            feature_completion_rate: if status == SessionStatus::Completed {
                1.0
            } else {
                0.0
            },
            checkpoints_created: iterations as usize,
            rollbacks_performed: 0,
            errors_encountered: if iterations > 1 {
                iterations as usize - 1
            } else {
                0
            },
            avg_iterations_per_feature: Some(iterations as f32),
            observations,
            recommendations,
        }
    }

    // --- Low-level function tests (preserved from before) ---

    #[test]
    fn test_capture_resolution() {
        let mock = MockKnowledgeBase::new();

        capture_resolution(
            &mock,
            "beads-abc123",
            "Fix borrow checker error",
            3,
            "Integrator",
            &["src/lib.rs".to_string(), "src/main.rs".to_string()],
        )
        .unwrap();

        let uploads = mock.captured_uploads.lock().unwrap();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].0, "project_brain");
        assert!(uploads[0].2.contains("beads-abc123"));
        assert!(uploads[0].2.contains("3"));
        assert!(uploads[0].2.contains("src/lib.rs"));
    }

    #[test]
    fn test_capture_error_pattern() {
        let mock = MockKnowledgeBase::new();

        capture_error_pattern(
            &mock,
            "beads-def456",
            &["BorrowChecker".to_string(), "Lifetime".to_string()],
            5,
            "Used Arc<Mutex<>> to wrap the shared state across async boundaries.",
        )
        .unwrap();

        let uploads = mock.captured_uploads.lock().unwrap();
        assert_eq!(uploads.len(), 1);
        assert_eq!(uploads[0].0, "debugging_kb");
        assert!(uploads[0].2.contains("BorrowChecker"));
        assert!(uploads[0].2.contains("Arc<Mutex<>>"));
    }

    #[test]
    fn test_capture_with_empty_files() {
        let mock = MockKnowledgeBase::new();

        capture_resolution(&mock, "beads-xyz", "Test issue", 1, "Implementer", &[]).unwrap();

        let uploads = mock.captured_uploads.lock().unwrap();
        assert!(uploads[0].2.contains("(none recorded)"));
    }

    // --- KnowledgeSyncService tests ---

    #[test]
    fn test_service_successful_session_routes_to_project_brain() {
        let mock = MockKnowledgeBase::new();
        let svc = KnowledgeSyncService::new(&mock).without_dedup();

        let retro = test_retrospective(SessionStatus::Completed, 2, vec![], vec![]);

        let captures = svc.capture_from_retrospective(&retro, "beads-001", "Simple fix");

        // 1 iteration < 3, so only project_brain capture
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].target, NotebookTarget::ProjectBrain);
        assert_eq!(captures[0].source, CaptureSource::Retrospective);
        assert!(captures[0].content.contains("beads-001"));
        assert!(captures[0].content.contains("Simple fix"));
    }

    #[test]
    fn test_service_tricky_bug_routes_to_both_notebooks() {
        let mock = MockKnowledgeBase::new();
        let svc = KnowledgeSyncService::new(&mock).without_dedup();

        let retro = test_retrospective(
            SessionStatus::Completed,
            5,
            vec!["Borrow checker cascade".into()],
            vec!["Use Arc for shared state".into()],
        );

        let captures = svc.capture_from_retrospective(&retro, "beads-002", "Tricky bug");

        // Should get: project_brain (resolution), debugging_kb (pattern), project_brain (insights)
        assert_eq!(captures.len(), 3);

        let targets: Vec<_> = captures.iter().map(|c| c.target.clone()).collect();
        assert!(targets.contains(&NotebookTarget::ProjectBrain));
        assert!(targets.contains(&NotebookTarget::DebuggingKb));

        // Debugging KB capture should mention iterations
        let debug_capture = captures
            .iter()
            .find(|c| c.target == NotebookTarget::DebuggingKb)
            .unwrap();
        assert!(debug_capture.content.contains("5 / 10"));
    }

    #[test]
    fn test_service_failed_session_no_resolution_capture() {
        let mock = MockKnowledgeBase::new();
        let svc = KnowledgeSyncService::new(&mock).without_dedup();

        let retro = test_retrospective(
            SessionStatus::Failed,
            8,
            vec!["Model kept generating invalid code".into()],
            vec!["Try different model tier".into()],
        );

        let captures = svc.capture_from_retrospective(&retro, "beads-003", "Failed task");

        // No resolution (failed), but should get debugging_kb (8 >= 3) + insights
        assert_eq!(captures.len(), 2);

        let targets: Vec<_> = captures.iter().map(|c| c.target.clone()).collect();
        assert!(targets.contains(&NotebookTarget::DebuggingKb));
        assert!(targets.contains(&NotebookTarget::ProjectBrain)); // insights
    }

    #[test]
    fn test_service_low_iteration_failure_only_insights() {
        let mock = MockKnowledgeBase::new();
        let svc = KnowledgeSyncService::new(&mock).without_dedup();

        let retro = test_retrospective(
            SessionStatus::Failed,
            1,
            vec!["Immediate failure".into()],
            vec![],
        );

        let captures = svc.capture_from_retrospective(&retro, "beads-004", "Quick fail");

        // No resolution (failed), no tricky bug (1 < 3), but has observations
        assert_eq!(captures.len(), 1);
        assert_eq!(captures[0].target, NotebookTarget::ProjectBrain);
        assert!(captures[0].title.contains("Insights"));
    }

    #[test]
    fn test_service_no_observations_skips_insights_for_success() {
        let mock = MockKnowledgeBase::new();
        let svc = KnowledgeSyncService::new(&mock).without_dedup();

        let retro = test_retrospective(SessionStatus::Completed, 1, vec![], vec![]);

        let captures = svc.capture_from_retrospective(&retro, "beads-005", "Clean win");

        // Only resolution capture, insights skipped (dominated by resolution + empty)
        assert_eq!(captures.len(), 1);
        assert!(captures[0].title.starts_with("Retrospective:"));
    }

    #[test]
    fn test_dedup_skips_existing_title() {
        let mock = MockKnowledgeBase::new()
            .with_response("project_brain", "Found source: Retrospective: beads-006");
        let svc = KnowledgeSyncService::new(&mock);

        let retro = test_retrospective(SessionStatus::Completed, 1, vec![], vec![]);

        let captures = svc.capture_from_retrospective(&retro, "beads-006", "Already captured");

        // Dedup should skip the upload
        assert_eq!(captures.len(), 0);

        // Should have queried but not uploaded
        let queries = mock.captured_queries.lock().unwrap();
        assert!(!queries.is_empty());
        let uploads = mock.captured_uploads.lock().unwrap();
        assert_eq!(uploads.len(), 0);
    }

    #[test]
    fn test_dedup_allows_new_title() {
        let mock = MockKnowledgeBase::new().with_response("project_brain", "No matching sources.");
        let svc = KnowledgeSyncService::new(&mock);

        let retro = test_retrospective(SessionStatus::Completed, 1, vec![], vec![]);

        let captures = svc.capture_from_retrospective(&retro, "beads-007", "New issue");

        // Not a duplicate — should upload
        assert_eq!(captures.len(), 1);
    }

    #[test]
    fn test_dedup_disabled_always_uploads() {
        let mock = MockKnowledgeBase::new()
            .with_response("project_brain", "Found source: Retrospective: beads-008");
        let svc = KnowledgeSyncService::new(&mock).without_dedup();

        let retro = test_retrospective(SessionStatus::Completed, 1, vec![], vec![]);

        let captures = svc.capture_from_retrospective(&retro, "beads-008", "Force upload");

        // Dedup disabled — should upload even though title matches
        assert_eq!(captures.len(), 1);

        // Should NOT have queried
        let queries = mock.captured_queries.lock().unwrap();
        assert_eq!(queries.len(), 0);
    }

    #[test]
    fn test_notebook_target_roles() {
        assert_eq!(NotebookTarget::ProjectBrain.role(), "project_brain");
        assert_eq!(NotebookTarget::DebuggingKb.role(), "debugging_kb");
    }

    #[test]
    fn test_capture_tags_include_issue_id() {
        let mock = MockKnowledgeBase::new();
        let svc = KnowledgeSyncService::new(&mock).without_dedup();

        let retro = test_retrospective(SessionStatus::Completed, 5, vec!["obs".into()], vec![]);

        let captures = svc.capture_from_retrospective(&retro, "beads-009", "Tagged");

        for capture in &captures {
            assert!(
                capture.tags.contains(&"beads-009".to_string()),
                "Missing issue_id tag in {:?}",
                capture.title
            );
        }
    }

    #[test]
    fn test_format_resolution_includes_efficiency() {
        let retro = test_retrospective(
            SessionStatus::Completed,
            3,
            vec!["Needed multiple passes".into()],
            vec![],
        );

        let content = format_resolution_retrospective(&retro, "beads-010", "Efficiency check");
        assert!(content.contains("3 / 10"));
        assert!(content.contains("30%")); // 3/10 = 30%
        assert!(content.contains("Needed multiple passes"));
    }

    #[test]
    fn test_format_debug_includes_lessons() {
        let retro = test_retrospective(
            SessionStatus::Completed,
            5,
            vec!["Borrow cascade".into()],
            vec!["Use RefCell".into()],
        );

        let content = format_debug_retrospective(&retro, "beads-011", "Debug format");
        assert!(content.contains("5 / 10"));
        assert!(content.contains("Borrow cascade"));
        assert!(content.contains("Use RefCell"));
        assert!(content.contains("Lessons learned"));
    }

    #[test]
    fn test_format_insights_both_sections() {
        let retro = test_retrospective(
            SessionStatus::Completed,
            1,
            vec!["Fast resolution".into()],
            vec!["Good pattern to reuse".into()],
        );

        let content = format_insights(&retro, "beads-012");
        assert!(content.contains("Fast resolution"));
        assert!(content.contains("Good pattern to reuse"));
        assert!(content.contains("Observations"));
        assert!(content.contains("Recommendations"));
    }
}
