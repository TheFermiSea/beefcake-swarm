use super::*;

use std::path::Path;
use tracing::{info, warn};

/// Write session metrics to `.swarm-metrics.json` in the worktree.
pub fn write_session_metrics(metrics: &SessionMetrics, wt_path: &Path) {
    let path = wt_path.join(".swarm-metrics.json");
    match serde_json::to_string_pretty(metrics) {
        Ok(json) => match std::fs::write(&path, json) {
            Ok(()) => info!(path = %path.display(), "Wrote session metrics"),
            Err(e) => warn!("Failed to write session metrics: {e}"),
        },
        Err(e) => warn!("Failed to serialize session metrics: {e}"),
    }
}

/// Append session metrics to `.swarm-telemetry.jsonl` in the repo root.
///
/// Each line is a complete JSON object (JSONL format) for easy streaming analysis.
pub fn append_telemetry(metrics: &SessionMetrics, repo_root: &Path) {
    let path = repo_root.join(".swarm-telemetry.jsonl");
    match serde_json::to_string(metrics) {
        Ok(json) => {
            use std::io::Write;
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(mut file) => {
                    if let Err(e) = writeln!(file, "{json}") {
                        warn!("Failed to append telemetry: {e}");
                    } else {
                        info!(path = %path.display(), "Appended session telemetry");
                    }
                }
                Err(e) => warn!("Failed to open telemetry file: {e}"),
            }
        }
        Err(e) => warn!("Failed to serialize telemetry: {e}"),
    }
}

/// Append a row to `experiments.tsv` in the worktree.
///
/// Each row captures a single iteration decision point for trajectory analysis.
/// Header is written on first call. The TSV format enables easy `sort | uniq -c`
/// analysis without JSON parsing.
pub fn append_experiment_tsv(
    worktree_path: &Path,
    commit: &str,
    error_count: usize,
    gates_passed: &[&str],
    status: &str,
    description: &str,
) {
    use std::io::Write;
    let tsv_path = worktree_path.join("experiments.tsv");
    let needs_header = !tsv_path.exists();

    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&tsv_path)
    {
        Ok(mut file) => {
            if needs_header {
                let _ = writeln!(
                    file,
                    "timestamp\tcommit\terror_count\tgates_passed\tstatus\tdescription"
                );
            }
            let ts = chrono::Utc::now().to_rfc3339();
            let gates = gates_passed.join(",");
            let _ = writeln!(
                file,
                "{ts}\t{commit}\t{error_count}\t{gates}\t{status}\t{description}"
            );
        }
        Err(e) => warn!("Failed to write experiments.tsv: {e}"),
    }
}

/// Append an entry to `.swarm-failure-ledger.jsonl` in the worktree.
pub fn append_failure_ledger(worktree_path: &Path, entry: &FailureLedgerEntry) {
    use std::io::Write;
    let path = worktree_path.join(".swarm-failure-ledger.jsonl");
    match serde_json::to_string(entry) {
        Ok(json) => {
            match std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
            {
                Ok(mut file) => {
                    if let Err(e) = writeln!(file, "{json}") {
                        warn!("Failed to append failure ledger: {e}");
                    }
                }
                Err(e) => warn!("Failed to open failure ledger: {e}"),
            }
        }
        Err(e) => warn!("Failed to serialize failure ledger entry: {e}"),
    }
}

/// Write execution artifacts from session metrics to `.swarm-artifacts/` directory.
///
/// Creates one JSON file per iteration: `iteration-001.json`, `iteration-002.json`, etc.
/// Only writes files for iterations that have execution artifacts attached.
/// Supports retention: if `max_sessions` is set, prunes oldest session directories.
pub fn write_execution_artifacts(
    metrics: &SessionMetrics,
    wt_path: &Path,
    max_sessions: Option<usize>,
) {
    let artifacts_dir = wt_path.join(".swarm-artifacts").join(&metrics.session_id);

    // Create the session directory
    if let Err(e) = std::fs::create_dir_all(&artifacts_dir) {
        warn!("Failed to create artifacts directory: {e}");
        return;
    }

    let mut written = 0usize;
    for iter_metrics in &metrics.iterations {
        if let Some(ref artifact) = iter_metrics.execution_artifact {
            let filename = format!("iteration-{:03}.json", iter_metrics.iteration);
            let path = artifacts_dir.join(&filename);
            match serde_json::to_string_pretty(artifact) {
                Ok(json) => match std::fs::write(&path, json) {
                    Ok(()) => written += 1,
                    Err(e) => warn!("Failed to write artifact {filename}: {e}"),
                },
                Err(e) => warn!("Failed to serialize artifact {filename}: {e}"),
            }
        }
    }

    if written > 0 {
        info!(
            path = %artifacts_dir.display(),
            count = written,
            "Wrote execution artifacts"
        );
    }

    // Retention: prune old session directories if max_sessions is set
    if let Some(max) = max_sessions {
        let parent = wt_path.join(".swarm-artifacts");
        prune_artifact_sessions(&parent, max);
    }
}

/// Remove oldest session artifact directories to stay within the retention limit.
fn prune_artifact_sessions(artifacts_root: &Path, max_sessions: usize) {
    let entries: Vec<_> = match std::fs::read_dir(artifacts_root) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .collect(),
        Err(_) => return,
    };

    if entries.len() <= max_sessions {
        return;
    }

    // Sort by modification time (oldest first)
    let mut sorted: Vec<_> = entries
        .into_iter()
        .filter_map(|e| {
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((mtime, e.path()))
        })
        .collect();
    sorted.sort_by_key(|(mtime, _)| *mtime);

    let to_remove = sorted.len() - max_sessions;
    for (_, path) in sorted.into_iter().take(to_remove) {
        if let Err(e) = std::fs::remove_dir_all(&path) {
            warn!("Failed to prune artifact session {}: {e}", path.display());
        } else {
            info!(path = %path.display(), "Pruned old artifact session");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;



    #[test]
    fn test_write_session_metrics_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = SessionMetrics {
            session_id: "test-session".into(),
            issue_id: "test-issue".into(),
            issue_title: "Test".into(),
            success: true,
            total_iterations: 1,
            final_tier: "Integrator".into(),
            elapsed_ms: 5000,
            total_no_change_iterations: 0,
            no_change_rate: 0.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2024-01-01T00:00:00Z".into(),
            stack_profile: "hybrid_balanced_v1".into(),
            repo_id: None,
            adapter_id: None,
            turns_until_first_write: None,
            write_by_turn_2: false,
            role_map_version: "v1".into(),
            tensorzero_episode_id: None,
            harness_trace: HarnessComponentTrace::default(),
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
        };

        write_session_metrics(&metrics, dir.path());

        let path = dir.path().join(".swarm-metrics.json");
        assert!(path.exists());

        let contents = std::fs::read_to_string(&path).unwrap();
        let loaded: SessionMetrics = serde_json::from_str(&contents).unwrap();
        assert_eq!(loaded.session_id, "test-session");
        assert!(loaded.success);
    }

    #[test]
    fn test_append_telemetry_jsonl() {
        let dir = tempfile::tempdir().unwrap();

        let metrics1 = SessionMetrics {
            session_id: "sess-1".into(),
            issue_id: "issue-1".into(),
            issue_title: "First".into(),
            success: true,
            total_iterations: 1,
            final_tier: "Integrator".into(),
            elapsed_ms: 3000,
            total_no_change_iterations: 0,
            no_change_rate: 0.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2024-01-01T00:00:00Z".into(),
            stack_profile: "hybrid_balanced_v1".into(),
            repo_id: None,
            adapter_id: None,
            turns_until_first_write: None,
            write_by_turn_2: false,
            role_map_version: "v1".into(),
            tensorzero_episode_id: None,
            harness_trace: HarnessComponentTrace::default(),
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
        };
        let metrics2 = SessionMetrics {
            session_id: "sess-2".into(),
            issue_id: "issue-2".into(),
            issue_title: "Second".into(),
            success: false,
            total_iterations: 3,
            final_tier: "Cloud".into(),
            elapsed_ms: 15000,
            total_no_change_iterations: 1,
            no_change_rate: 1.0 / 3.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![],
            timestamp: "2024-01-01T01:00:00Z".into(),
            stack_profile: "hybrid_balanced_v1".into(),
            repo_id: None,
            adapter_id: None,
            turns_until_first_write: None,
            write_by_turn_2: false,
            role_map_version: "v1".into(),
            tensorzero_episode_id: None,
            harness_trace: HarnessComponentTrace::default(),
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
        };

        append_telemetry(&metrics1, dir.path());
        append_telemetry(&metrics2, dir.path());

        let path = dir.path().join(".swarm-telemetry.jsonl");
        let contents = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2);

        let loaded1: SessionMetrics = serde_json::from_str(lines[0]).unwrap();
        let loaded2: SessionMetrics = serde_json::from_str(lines[1]).unwrap();
        assert_eq!(loaded1.session_id, "sess-1");
        assert_eq!(loaded2.session_id, "sess-2");
        assert!(loaded1.success);
        assert!(!loaded2.success);
    }

    #[test]
    fn test_write_execution_artifacts_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let metrics = SessionMetrics {
            session_id: "test-session-art".into(),
            issue_id: "test-issue".into(),
            issue_title: "Test".into(),
            success: true,
            total_iterations: 2,
            final_tier: "Integrator".into(),
            elapsed_ms: 5000,
            total_no_change_iterations: 0,
            no_change_rate: 0.0,
            cloud_validations: vec![],
            local_validations: vec![],
            iterations: vec![
                IterationMetrics {
                    iteration: 1,
                    tier: "Integrator".into(),
                    agent_model: "m1".into(),
                    agent_prompt_tokens: 0,
                    agent_completion_tokens: 0,
                    agent_response_ms: 0,
                    verifier_ms: 0,
                    error_count: 0,
                    error_categories: vec![],
                    no_change: false,
                    auto_fix_applied: false,
                    regression_detected: false,
                    rollback_performed: false,
                    escalated: false,
                    coder_route: None,
                    artifacts: vec![],
                    execution_artifact: Some(ExecutionArtifact {
                        schema_version: ARTIFACT_SCHEMA_VERSION,
                        route_decision: Some(RouteDecision {
                            coder: "RustCoder".into(),
                            input_error_categories: vec![],
                            tier: "Integrator".into(),
                            rationale: None,
                        }),
                        verifier_snapshot: None,
                        evaluator_snapshot: None,
                        retry_rationale: None,
                    }),
                    progress_score: None,
                    best_error_count: None,
                },
                IterationMetrics {
                    iteration: 2,
                    tier: "Integrator".into(),
                    agent_model: "m1".into(),
                    agent_prompt_tokens: 0,
                    agent_completion_tokens: 0,
                    agent_response_ms: 0,
                    verifier_ms: 0,
                    error_count: 0,
                    error_categories: vec![],
                    no_change: false,
                    auto_fix_applied: false,
                    regression_detected: false,
                    rollback_performed: false,
                    escalated: false,
                    coder_route: None,
                    artifacts: vec![],
                    // No artifact for this iteration
                    execution_artifact: None,
                    progress_score: None,
                    best_error_count: None,
                },
            ],
            timestamp: "2026-01-01T00:00:00Z".into(),
            stack_profile: "hybrid_balanced_v1".into(),
            repo_id: None,
            adapter_id: None,
            turns_until_first_write: None,
            write_by_turn_2: false,
            role_map_version: "v1".into(),
            tensorzero_episode_id: None,
            harness_trace: HarnessComponentTrace::default(),
            input_tokens: 0,
            output_tokens: 0,
            estimated_cost_usd: 0.0,
        };

        write_execution_artifacts(&metrics, dir.path(), None);

        let art_dir = dir.path().join(".swarm-artifacts").join("test-session-art");
        assert!(art_dir.exists());

        // Only iteration 1 has an artifact
        let iter1 = art_dir.join("iteration-001.json");
        assert!(iter1.exists());
        let iter2 = art_dir.join("iteration-002.json");
        assert!(!iter2.exists());

        // Verify content is valid JSON
        let content = std::fs::read_to_string(&iter1).unwrap();
        let loaded: ExecutionArtifact = serde_json::from_str(&content).unwrap();
        assert_eq!(loaded.schema_version, ARTIFACT_SCHEMA_VERSION);
        assert_eq!(loaded.route_decision.unwrap().coder, "RustCoder");
    }

    #[test]
    fn test_artifact_retention_pruning() {
        let dir = tempfile::tempdir().unwrap();
        let artifacts_root = dir.path().join(".swarm-artifacts");

        // Create 5 session directories with staggered modification times
        for i in 1..=5 {
            let session_dir = artifacts_root.join(format!("session-{i}"));
            std::fs::create_dir_all(&session_dir).unwrap();
            std::fs::write(session_dir.join("iteration-001.json"), "{}").unwrap();
            // Small delay to ensure different modification times
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        // Prune to keep only 3
        prune_artifact_sessions(&artifacts_root, 3);

        let remaining: Vec<_> = std::fs::read_dir(&artifacts_root)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(remaining.len(), 3);
    }
}
