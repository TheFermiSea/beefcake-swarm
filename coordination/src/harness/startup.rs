//! Startup ritual implementation
//!
//! Implements the initialization sequence for harness mode:
//! 1. Verify working directory
//! 2. Load git state
//! 3. Load progress file
//! 4. Load feature registry
//! 5. Identify next task

use crate::harness::error::{HarnessError, HarnessResult};
use crate::harness::feature_registry::FeatureRegistry;
use crate::harness::git_manager::GitManager;
use crate::harness::progress::ProgressTracker;
use crate::harness::types::{HarnessConfig, ProgressMarker, StartupContext};
use std::path::Path;

/// Perform startup ritual and gather context
pub fn perform_startup_ritual(config: &HarnessConfig) -> HarnessResult<StartupContext> {
    // Step 1: Verify working directory
    let working_directory = verify_working_directory(&config.working_directory)?;

    // Step 2: Load git state
    let git_manager = GitManager::new(&working_directory, &config.commit_prefix);

    let current_branch = git_manager
        .current_branch()
        .map_err(|e| HarnessError::startup_failed("git_branch", e.to_string()))?;

    let current_commit = git_manager
        .current_commit()
        .map_err(|e| HarnessError::startup_failed("git_commit", e.to_string()))?;

    let has_uncommitted_changes = git_manager
        .has_uncommitted_changes()
        .map_err(|e| HarnessError::startup_failed("git_status", e.to_string()))?;

    if config.require_clean_git && has_uncommitted_changes {
        return Err(HarnessError::UncommittedChanges);
    }

    let recent_commits = git_manager.recent_commits(20).unwrap_or_default();

    // Step 3: Load progress file
    let progress_tracker = ProgressTracker::new(&config.progress_path);
    let recent_progress = progress_tracker.read_last(10).unwrap_or_default();

    // Try to find last session state
    let last_session = if let Some(last_start) = progress_tracker.last_session_start()? {
        // Check if session ended
        let session_entries = progress_tracker.session_entries(&last_start.session_id)?;
        let ended = session_entries.iter().any(|e| {
            matches!(
                e.marker,
                ProgressMarker::SessionEnd | ProgressMarker::SessionAbort
            )
        });

        if !ended {
            // Session didn't end cleanly - could resume
            // Reconstruct partial state
            let max_iteration = session_entries
                .iter()
                .map(|e| e.iteration)
                .max()
                .unwrap_or(0);
            let current_feature = session_entries
                .iter()
                .rev()
                .find(|e| matches!(e.marker, ProgressMarker::FeatureStart))
                .and_then(|e| e.feature_id.clone());

            Some(crate::harness::types::SessionState {
                id: last_start.session_id,
                started_at: last_start.timestamp,
                iteration: max_iteration,
                max_iterations: 20, // Default, should be from config
                current_feature,
                status: crate::harness::types::SessionStatus::Paused,
                working_directory: working_directory.clone(),
                initial_commit: None,
                pending_interventions: Vec::new(), // Will be loaded from session state file
                sub_sessions: Vec::new(),          // Will be loaded from session state file
            })
        } else {
            None
        }
    } else {
        None
    };

    // Step 4: Load feature registry
    let registry = if config.features_path.exists() {
        FeatureRegistry::load(&config.features_path)?
    } else {
        FeatureRegistry::empty(&config.features_path)
    };

    let feature_summary = registry.summary();
    let next_feature = registry.next_incomplete().map(|f| f.id.clone());

    Ok(StartupContext {
        working_directory,
        recent_commits,
        current_branch,
        current_commit,
        has_uncommitted_changes,
        last_session,
        feature_summary,
        next_feature,
        recent_progress,
    })
}

/// Verify working directory exists and is accessible
fn verify_working_directory(path: &Path) -> HarnessResult<std::path::PathBuf> {
    let canonical = path
        .canonicalize()
        .map_err(|e| HarnessError::startup_failed("working_directory", e.to_string()))?;

    if !canonical.is_dir() {
        return Err(HarnessError::startup_failed(
            "working_directory",
            format!("{} is not a directory", canonical.display()),
        ));
    }

    // Verify it's a git repository
    let git_dir = canonical.join(".git");
    if !git_dir.exists() {
        return Err(HarnessError::startup_failed(
            "working_directory",
            format!("{} is not a git repository", canonical.display()),
        ));
    }

    Ok(canonical)
}

/// Format startup context as human-readable summary
pub fn format_startup_context(ctx: &StartupContext) -> String {
    let mut output = String::new();

    output.push_str("=== Harness Startup Context ===\n\n");

    // Working directory
    output.push_str(&format!(
        "Working Directory: {}\n",
        ctx.working_directory.display()
    ));
    output.push_str(&format!(
        "Branch: {} @ {}\n",
        ctx.current_branch, ctx.current_commit
    ));

    if ctx.has_uncommitted_changes {
        output.push_str("⚠️  Uncommitted changes detected\n");
    }

    output.push('\n');

    // Feature summary
    output.push_str("--- Feature Status ---\n");
    output.push_str(&format!(
        "Total: {} | Passing: {} | Failing: {} | Pending: {}\n",
        ctx.feature_summary.total,
        ctx.feature_summary.passing,
        ctx.feature_summary.failing,
        ctx.feature_summary.pending
    ));
    output.push_str(&format!(
        "Completion: {:.1}%\n",
        ctx.feature_summary.completion_percent
    ));

    if let Some(ref next) = ctx.next_feature {
        output.push_str(&format!("Next feature: {}\n", next));
    }

    output.push('\n');

    // Recent commits
    if !ctx.recent_commits.is_empty() {
        output.push_str("--- Recent Commits ---\n");
        for commit in ctx.recent_commits.iter().take(5) {
            let checkpoint = if commit.is_harness_checkpoint {
                " [checkpoint]"
            } else {
                ""
            };
            output.push_str(&format!(
                "{} {}{}\n",
                commit.hash, commit.message, checkpoint
            ));
        }
    }

    output.push('\n');

    // Last session
    if let Some(ref session) = ctx.last_session {
        output.push_str("--- Resumable Session ---\n");
        output.push_str(&format!(
            "Session {} was interrupted at iteration {}\n",
            &session.id[..8],
            session.iteration
        ));
        if let Some(ref feature) = session.current_feature {
            output.push_str(&format!("Was working on: {}\n", feature));
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;
    use tempfile::tempdir;

    fn setup_test_repo() -> (tempfile::TempDir, HarnessConfig) {
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

        // Create initial commit
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
            max_iterations: 20,
            auto_checkpoint: true,
            require_clean_git: false,
            commit_prefix: "[harness]".to_string(),
        };

        (dir, config)
    }

    #[test]
    fn test_startup_ritual_basic() {
        let (_dir, config) = setup_test_repo();

        let context = perform_startup_ritual(&config).unwrap();

        assert!(!context.current_branch.is_empty());
        assert!(!context.current_commit.is_empty());
        assert!(!context.has_uncommitted_changes);
        assert!(context.last_session.is_none());
        assert_eq!(context.feature_summary.total, 0);
    }

    #[test]
    fn test_startup_with_features() {
        let (dir, config) = setup_test_repo();

        // Create features.json
        let features = r#"[
            {"id": "f1", "category": "functional", "description": "Feature 1", "steps": [], "passes": true},
            {"id": "f2", "category": "functional", "description": "Feature 2", "steps": [], "passes": false}
        ]"#;
        std::fs::write(dir.path().join("features.json"), features).unwrap();

        let context = perform_startup_ritual(&config).unwrap();

        assert_eq!(context.feature_summary.total, 2);
        assert_eq!(context.feature_summary.passing, 1);
        assert_eq!(context.next_feature, Some("f2".to_string()));
    }

    #[test]
    fn test_startup_requires_clean_git() {
        let (dir, mut config) = setup_test_repo();
        config.require_clean_git = true;

        // Create uncommitted change
        std::fs::write(dir.path().join("dirty.txt"), "dirty").unwrap();

        let result = perform_startup_ritual(&config);
        assert!(matches!(result, Err(HarnessError::UncommittedChanges)));
    }
}
