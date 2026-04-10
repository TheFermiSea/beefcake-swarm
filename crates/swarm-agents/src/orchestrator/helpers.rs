//! Environment parsing utilities, directive management, scaffolding fallback,
//! and knowledge base helpers.

use std::path::Path;
use std::process::Command;
use std::time::Duration;

use tracing::{info, warn};

use crate::notebook_bridge::KnowledgeBase;
use coordination::{
    InterventionType, PendingIntervention, ProgressTracker, SessionManager, SwarmTier,
};

// ── Environment parsing ─────────────────────────────────────────────

pub(crate) fn timeout_from_env(var: &str, default_secs: u64) -> Duration {
    let secs = std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default_secs);
    Duration::from_secs(secs)
}

pub(crate) fn u32_from_env(var: &str, default: u32) -> u32 {
    std::env::var(var)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|v| *v > 0)
        .unwrap_or(default)
}

pub(crate) fn bool_from_env(var: &str, default: bool) -> bool {
    std::env::var(var)
        .ok()
        .map(|v| {
            matches!(
                v.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(default)
}

pub(crate) fn tier_from_env(var: &str, default: SwarmTier) -> SwarmTier {
    match std::env::var(var)
        .ok()
        .map(|v| v.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("worker") => SwarmTier::Worker,
        Some("human") => SwarmTier::Human,
        Some("council") => SwarmTier::Council,
        _ => default,
    }
}

pub(crate) fn default_initial_tier(
    worker_first_enabled: bool,
    worker_first_tier: SwarmTier,
) -> SwarmTier {
    if worker_first_enabled {
        worker_first_tier
    } else {
        SwarmTier::Council
    }
}

// ── Knowledge base ──────────────────────────────────────────────────

/// Query the knowledge base with graceful degradation on failure.
///
/// Wraps `KnowledgeBase::query` with error handling so that any KB failure
/// (connection error, auth failure, or a hanging `nlm` CLI subprocess) returns
/// an empty string instead of propagating an error. This ensures KB
/// unavailability never blocks the orchestration loop.
pub(crate) fn query_kb_with_failsafe(kb: &dyn KnowledgeBase, role: &str, question: &str) -> String {
    match kb.query(role, question) {
        Ok(response) => response,
        Err(e) => {
            warn!(role, error = %e, "KB query failed — proceeding without context");
            String::new()
        }
    }
}

// ── Test Gate ───────────────────────────────────────────────────────

/// Runs `cargo test --workspace` to verify the health of the entire project.
/// Returns `Ok(())` if all tests pass, or an error describing the failure.
pub fn check_baseline_tests(wt_path: &Path) -> Result<(), String> {
    let status = Command::new("cargo")
        .args(["test", "--workspace"])
        .current_dir(wt_path)
        .status()
        .map_err(|e| format!("Failed to execute cargo test: {}", e))?;

    if status.success() {
        Ok(())
    } else {
        Err(format!(
            "Baseline tests failed with exit code: {:?}",
            status.code()
        ))
    }
}

// ── Doc task scaffolding ────────────────────────────────────────────

/// Detect whether an issue is doc-oriented based on title/description keywords.
fn is_doc_task(title: &str, description: &str) -> bool {
    let combined = format!("{} {}", title, description).to_ascii_lowercase();
    let doc_keywords = [
        ".md",
        "rfc",
        "doc",
        "architecture",
        "planning",
        "readme",
        "design doc",
    ];
    doc_keywords.iter().any(|kw| combined.contains(kw))
}

/// Generate a minimal markdown scaffold for a doc-oriented task.
///
/// When doc tasks hit the no-change circuit breaker, this creates a template
/// file so at least a skeleton exists for human completion. Returns `true`
/// if a scaffold was committed.
pub fn try_scaffold_fallback(
    wt_path: &Path,
    issue_id: &str,
    issue_title: &str,
    issue_description: &str,
    iteration: u32,
) -> bool {
    if !is_doc_task(issue_title, issue_description) {
        return false;
    }

    // Generate a safe filename from the issue title
    let safe_name: String = issue_title
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '-'
            }
        })
        .collect::<String>()
        .to_ascii_lowercase();
    let filename = format!("docs/{}.md", safe_name.trim_matches('-'));

    let scaffold = format!(
        "# {title}\n\n\
         > Auto-generated scaffold by swarm orchestrator.\n\
         > Issue: `{id}` | Generated at iteration {iter}\n\n\
         ## Overview\n\n\
         <!-- Purpose and scope of the issue being tracked in the swarm orchestrator -->

         Add a brief description of what this issue is about and its scope. What problem does it solve? What are the boundaries of this work?\n\n\
         ## Details\n\n\
         <!-- Implementation details and technical context for the tracked issue -->\n\n\
        ## Open Questions\n\n\n         <!-- Questions that need answers before implementation can proceed -->\n\n         - What are the key requirements and constraints?\n\n         - What are the potential edge cases or failure modes?\n\n         - What alternatives were considered and why were they rejected?\n\n         - Who are the stakeholders and what are their concerns?\n\n         \n\n\n         
         ",
        title = issue_title,
        id = issue_id,
        iter = iteration,
    );

    // Ensure docs/ directory exists
    let docs_dir = wt_path.join("docs");
    if let Err(e) = std::fs::create_dir_all(&docs_dir) {
        warn!("Failed to create docs dir for scaffold: {e}");
        return false;
    }

    let file_path = wt_path.join(&filename);
    if let Err(e) = std::fs::write(&file_path, &scaffold) {
        warn!("Failed to write scaffold file: {e}");
        return false;
    }

    // Stage and commit the scaffold
    let add = std::process::Command::new("git")
        .args(["add", &filename])
        .current_dir(wt_path)
        .output();
    if !matches!(add, Ok(ref out) if out.status.success()) {
        warn!("Failed to git add scaffold");
        return false;
    }

    let msg = format!("swarm: scaffold fallback for {issue_id} (iteration {iteration})");
    let commit = std::process::Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(wt_path)
        .output();
    if !matches!(commit, Ok(ref out) if out.status.success()) {
        warn!("Failed to commit scaffold");
        return false;
    }

    info!(
        issue_id,
        filename, "Scaffold fallback committed for doc task"
    );
    true
}

/// Create a human intervention request when the escalation engine reports stuck.
///
/// Surfaces the intervention through 4 mechanisms:
/// 1. Records in session state (in-memory)
/// 2. Writes `.swarm-interventions.json` in the worktree root
/// 3. POSTs to `SWARM_WEBHOOK_URL` if configured
/// 4. Sends escalation mail via `bd mail send lead` (non-blocking)
pub(crate) fn create_stuck_intervention(
    session: &mut SessionManager,
    progress: &ProgressTracker,
    wt_path: &Path,
    iteration: u32,
    reason: &str,
) {
    let feature_id = session.current_feature().unwrap_or("unknown").to_string();

    let intervention = PendingIntervention::new(
        InterventionType::ReviewRequired,
        format!("Stuck after iteration {iteration}: {reason}. Manual review needed."),
    )
    .with_feature(&feature_id);

    session.state_mut().add_intervention(intervention);

    let _ = progress.log_error(
        session.session_id(),
        iteration,
        format!("Stuck — human intervention requested: {reason}"),
    );

    // --- Mechanism 4: Native beads mail escalation ---
    let mail_health = crate::beads_bridge::escalate_via_mail(wt_path, &feature_id, reason);

    // --- Mechanism 2: Write intervention JSON to worktree ---
    let mut intervention_data = serde_json::json!({
        "session_id": session.session_id(),
        "feature_id": feature_id,
        "iteration": iteration,
        "reason": reason,
        "type": "review_required",
        "timestamp": chrono::Utc::now().to_rfc3339(),
    });
    if let Some(record) = mail_health {
        intervention_data["mail_status"] = serde_json::json!({
            "operation": record.operation,
            "error_class": record.error_class,
            "error": record.error,
        });
    } else if let Some(record) = crate::beads_bridge::latest_mail_health(wt_path) {
        intervention_data["mail_status"] = serde_json::json!({
            "operation": record.operation,
            "error_class": record.error_class,
            "error": record.error,
        });
    }
    let intervention_path = wt_path.join(".swarm-interventions.json");
    match std::fs::write(
        &intervention_path,
        serde_json::to_string_pretty(&intervention_data).unwrap_or_default(),
    ) {
        Ok(()) => info!(path = %intervention_path.display(), "Wrote intervention file"),
        Err(e) => warn!("Failed to write intervention file: {e}"),
    }

    // --- Mechanism 3: Webhook notification ---
    if let Ok(webhook_url) = std::env::var("SWARM_WEBHOOK_URL") {
        // Fire-and-forget — don't block the orchestrator on webhook delivery
        let payload = intervention_data.clone();
        tokio::spawn(async move {
            let client = reqwest::Client::new();
            match client
                .post(&webhook_url)
                .json(&payload)
                .timeout(Duration::from_secs(10))
                .send()
                .await
            {
                Ok(resp) => info!(status = %resp.status(), "Webhook notification sent"),
                Err(e) => warn!("Webhook notification failed: {e}"),
            }
        });
    }
}

// ── Pattern detection + directive injection ─────────────────────────

/// Detect repeated failure patterns from the failure ledger.
///
/// Groups entries by `(tool, error_class)` and generates directives for
/// patterns that occurred 3+ times. Returns at most 5 directives.
#[allow(dead_code)] // Used by directive pipeline (not yet wired into driver)
pub(crate) fn detect_failure_patterns(worktree_path: &Path) -> Vec<String> {
    let path = worktree_path.join(".swarm-failure-ledger.jsonl");
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut counts: std::collections::HashMap<(String, String), usize> =
        std::collections::HashMap::new();
    for line in content.lines() {
        if let Ok(entry) = serde_json::from_str::<crate::telemetry::FailureLedgerEntry>(line) {
            if !entry.success {
                *counts
                    .entry((entry.tool.clone(), entry.error_class.clone()))
                    .or_insert(0) += 1;
            }
        }
    }

    let mut directives: Vec<String> = counts
        .into_iter()
        .filter(|(_, count)| *count >= 3)
        .map(|((tool, class), count)| {
            format!("Avoid: {tool} failures due to '{class}' (seen {count}x). Use anchor-based editing or re-read files before editing.")
        })
        .collect();

    directives.sort();
    directives.truncate(5);
    directives
}

/// Save generated directives to `.swarm-directives.jsonl` in the repo root.
#[allow(dead_code)] // Used by directive pipeline (not yet wired into driver)
pub(crate) fn save_directives(repo_root: &Path, directives: &[String]) {
    use std::io::Write;
    let path = repo_root.join(".swarm-directives.jsonl");
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(mut file) => {
            let ts = chrono::Utc::now().to_rfc3339();
            for d in directives {
                let entry = serde_json::json!({"timestamp": ts, "directive": d});
                let _ = writeln!(file, "{}", entry);
            }
        }
        Err(e) => warn!("Failed to write directives: {e}"),
    }
}

/// Load recent directives from `.swarm-directives.jsonl`.
///
/// Returns at most 5 recent directives (deduped).
#[allow(dead_code)] // Used by directive pipeline (not yet wired into driver)
pub(crate) fn load_directives(repo_root: &Path) -> Vec<String> {
    let path = repo_root.join(".swarm-directives.jsonl");
    let content = match std::fs::read_to_string(&path) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut directives: Vec<String> = content
        .lines()
        .rev() // most recent first
        .filter_map(|line| {
            let v: serde_json::Value = serde_json::from_str(line).ok()?;
            v.get("directive")?.as_str().map(|s| s.to_string())
        })
        .collect();

    // Deduplicate while preserving order
    let mut seen = std::collections::HashSet::new();
    directives.retain(|d| seen.insert(d.clone()));
    directives.truncate(5);
    directives
}

// ── Reformulation helpers ────────────────────────────────────────────

/// Load failure ledger entries from a worktree's `.swarm-failure-ledger.jsonl`.
#[allow(dead_code)] // Used by reformulation pipeline (not yet wired into driver)
pub(crate) fn load_failure_ledger(
    worktree_path: &Path,
) -> Vec<crate::telemetry::FailureLedgerEntry> {
    let path = worktree_path.join(".swarm-failure-ledger.jsonl");
    match std::fs::read_to_string(&path) {
        Ok(content) => content
            .lines()
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect(),
        Err(_) => Vec::new(),
    }
}

/// List files changed in the worktree relative to main (via `git diff --name-only`).
#[allow(dead_code)] // Used by reformulation pipeline (not yet wired into driver)
pub(crate) fn list_changed_files(worktree_path: &Path) -> Vec<String> {
    match std::process::Command::new("git")
        .args(["diff", "--name-only", "main"])
        .current_dir(worktree_path)
        .output()
    {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter(|l| !l.is_empty())
            .map(|l| l.to_string())
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::sync::{Mutex, OnceLock};

    fn beads_env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn test_query_kb_with_failsafe_on_failure() {
        use crate::notebook_bridge::KnowledgeBase;
        use anyhow::Result;

        struct FailingKb;
        impl KnowledgeBase for FailingKb {
            fn query(&self, _role: &str, _question: &str) -> Result<String> {
                anyhow::bail!("simulated nlm connection failure")
            }
            fn add_source_text(&self, _role: &str, _title: &str, _content: &str) -> Result<()> {
                Ok(())
            }
            fn add_source_file(&self, _role: &str, _file_path: &str) -> Result<()> {
                Ok(())
            }
            fn source_count(&self, _role: &str) -> Option<usize> {
                None
            }
            fn is_available(&self) -> bool {
                false
            }
        }

        let kb = FailingKb;
        // Must not panic or propagate error — returns empty string
        let result = query_kb_with_failsafe(&kb, "project_brain", "What is the architecture?");
        assert_eq!(result, "", "failsafe must return empty string on KB error");
    }

    #[test]
    fn test_query_kb_with_failsafe_on_success() {
        use crate::notebook_bridge::KnowledgeBase;
        use anyhow::Result;

        struct SucceedingKb;
        impl KnowledgeBase for SucceedingKb {
            fn query(&self, _role: &str, _question: &str) -> Result<String> {
                Ok("The architecture uses a 4-tier escalation ladder.".to_string())
            }
            fn add_source_text(&self, _role: &str, _title: &str, _content: &str) -> Result<()> {
                Ok(())
            }
            fn add_source_file(&self, _role: &str, _file_path: &str) -> Result<()> {
                Ok(())
            }
            fn source_count(&self, _role: &str) -> Option<usize> {
                Some(10)
            }
            fn is_available(&self) -> bool {
                true
            }
        }

        let kb = SucceedingKb;
        let result = query_kb_with_failsafe(&kb, "project_brain", "What is the architecture?");
        assert_eq!(result, "The architecture uses a 4-tier escalation ladder.");
    }

    #[test]
    fn test_default_initial_tier_uses_worker_first_only_when_enabled() {
        assert_eq!(
            default_initial_tier(true, SwarmTier::Worker),
            SwarmTier::Worker
        );
        assert_eq!(
            default_initial_tier(true, SwarmTier::Council),
            SwarmTier::Council
        );
        assert_eq!(
            default_initial_tier(false, SwarmTier::Worker),
            SwarmTier::Council
        );
    }

    #[test]
    fn test_create_stuck_intervention_adds_to_session() {
        let dir = tempfile::tempdir().unwrap();
        let mut session = SessionManager::new(dir.path().to_path_buf(), 10);
        session.start().unwrap();
        session.set_current_feature("test-issue-001");

        let progress = ProgressTracker::new(dir.path().join("progress.txt"));

        create_stuck_intervention(&mut session, &progress, dir.path(), 3, "repeated errors");

        // Intervention should be recorded in session state
        let interventions = session.state().unresolved_interventions();
        assert_eq!(interventions.len(), 1);
        assert!(interventions[0].question.contains("iteration 3"));
        assert!(interventions[0].question.contains("repeated errors"));
        assert_eq!(
            interventions[0].feature_id.as_deref(),
            Some("test-issue-001")
        );

        // Progress file should have the error logged
        let entries = progress.read_all().unwrap();
        assert_eq!(entries.len(), 1);
        assert!(entries[0].summary.contains("human intervention"));
    }

    #[test]
    fn test_create_stuck_intervention_records_mail_status_when_escalation_fails() {
        let _guard = beads_env_lock().lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("bd_mock");
        let script_content = "#!/bin/bash\nif [ \"$1\" = \"mail\" ] && [ \"$2\" = \"send\" ]; then\n  echo \"failed to stage dolt_ignore\" >&2\n  exit 1\nfi\nexit 0\n";
        {
            let mut file = fs::File::create(&script_path).unwrap();
            file.write_all(script_content.as_bytes()).unwrap();
        }
        fs::set_permissions(
            &script_path,
            std::os::unix::fs::PermissionsExt::from_mode(0o755),
        )
        .unwrap();

        let old_bin = std::env::var("SWARM_BEADS_BIN").ok();
        std::env::set_var("SWARM_BEADS_BIN", &script_path);

        let mut session = SessionManager::new(dir.path().to_path_buf(), 10);
        session.start().unwrap();
        session.set_current_feature("test-issue-002");
        let progress = ProgressTracker::new(dir.path().join("progress.txt"));

        create_stuck_intervention(&mut session, &progress, dir.path(), 2, "mail send failed");

        let intervention_path = dir.path().join(".swarm-interventions.json");
        let intervention_json: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(intervention_path).unwrap()).unwrap();
        assert_eq!(
            intervention_json["mail_status"]["error_class"],
            "dolt_ignore_stage_failed"
        );

        if let Some(bin) = old_bin {
            std::env::set_var("SWARM_BEADS_BIN", bin);
        } else {
            std::env::remove_var("SWARM_BEADS_BIN");
        }
    }
}
