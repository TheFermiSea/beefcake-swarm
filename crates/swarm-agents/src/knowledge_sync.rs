//! Automated knowledge sync: captures resolutions and error patterns
//! into NotebookLM for institutional memory.

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::notebook_bridge::KnowledgeBase;

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
            warn!(
                issue_id,
                "Failed to capture error pattern (non-fatal): {e}"
            );
            Ok(()) // Non-fatal
        }
    }
}

/// Sync the codebase to the Codebase notebook via repomix.
///
/// Runs `repomix --style xml` to pack the repo, then uploads the result.
/// This is expensive and should be run sparingly (e.g., via CLI flag).
pub fn sync_codebase_notebook(kb: &dyn KnowledgeBase, repo_root: &std::path::Path) -> Result<()> {
    let output_path = std::env::temp_dir().join("beefcake-swarm-repomix.xml");

    info!("Running repomix to pack codebase...");
    let repomix = std::process::Command::new("repomix")
        .args([
            "--style",
            "xml",
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
}
