# Exploration Notes: swarm-agents/src/orchestrator/mod.rs

## Git Diff (HEAD~1)
```
diff --git a/crates/swarm-agents/src/orchestrator/mod.rs b/crates/swarm-agents/src/orchestrator/mod.rs
index a59eeb8..5f54e78 100644
--- a/crates/swarm-agents/src/orchestrator/mod.rs
+++ b/crates/swarm-agents/src/orchestrator/mod.rs
@@ -2954,7 +2954,10 @@ async fn process_issue_core(
             let inf_ids = crate::tensorzero::resolve_recent_inference_ids(pg_url, since, 1).await;
             if let Some(inf_id) = inf_ids.first() {
                 // Segmentation tags — must use Display (snake_case) to match episode-level convention.
-                let primary_err = report.unique_error_categories().into_iter().next()
+                let primary_err = report
+                    .unique_error_categories()
+                    .into_iter()
+                    .next()
                     .map(|c| c.to_string());
                 let iter_tags = crate::tensorzero::FeedbackTags {
                     issue_id: Some(issue.id.clone()),
```

## SwarmResumeFile References
```
3767:        let resume = SwarmResumeFile {
4485:pub struct SwarmResumeFile {
4503:pub fn check_for_resume(repo_root: &Path) -> Option<SwarmResumeFile> {
4507:            Ok(contents) => match serde_json::from_str::<SwarmResumeFile>(&contents) {
```

## Key Function/Pattern References
```
32:    cloud_validate, extract_local_validator_feedback, extract_validator_feedback, local_validate,
80:use coordination::feedback::ErrorCategory;
214:pub async fn process_issue(
261:async fn process_issue_core(
737:    let mut last_validator_feedback: Vec<ValidatorFeedback> = Vec::new();
1258:    // Track cognition items retrieved across iterations for TZ feedback.
1367:        // worked into the objective. This closes the feedback loop:
1417:        // Inject structured validator feedback from prior iteration (TextGrad pattern)
1418:        if !last_validator_feedback.is_empty() {
1419:            packet.validator_feedback = std::mem::take(&mut last_validator_feedback);
1422:                feedback_count = packet.validator_feedback.len(),
1423:                \"Injected validator feedback into work packet\"
1481:                let typed_cats: Vec<coordination::feedback::ErrorCategory> =
1679:        // --- Active feedback injection (Robin pattern — adoption #2) ---
1695:                let feedback = archive.format_feedback_context(&error_cats, &files);
1696:                if !feedback.is_empty() {
1697:                    task_prompt.push_str(&feedback);
1706:        // items, archive feedback, etc.). This unbounded growth causes:
2860:                // Inject critiques as validator feedback for the next iteration's prompt
2875:                    last_validator_feedback.push(ValidatorFeedback {
2942:        // --- TZ inference-level feedback: verifier_pass per iteration ---
2968:                crate::tensorzero::post_inference_feedback(
2976:                crate::tensorzero::post_inference_feedback(
3099:            // prior validator feedback is stale.
3283:                        // Extract feedback for next iteration
3284:                        let feedback = extract_local_validator_feedback(&local_result);
3285:                        last_validator_feedback = feedback;
3298:                                feedback_count = last_validator_feedback.len(),
3299:                                \"Local validation rejected — looping with feedback\"
3349:            // don't block acceptance — avoids subjective LLM feedback loops.
3354:                    // Collect structured feedback for next iteration (TextGrad pattern)
3355:                    last_validator_feedback.clear();
3365:                                v.feedback.lines().take(5).collect::<Vec<_>>().join(\" | \")
3367:                            let feedback = extract_validator_feedback(v);
3368:                            last_validator_feedback.extend(feedback);
3371:                    if !last_validator_feedback.is_empty() {
3373:                            feedback_count = last_validator_feedback.len(),
3374:                            \"Collected structured validator feedback for next iteration\"
3530:        // --- TZ demonstration feedback: capture the successful diff ---
3903:    // --- TensorZero feedback ---
3905:    // TZ assigned (not our self-generated ones) and post feedback to those.
3912:        // TZ can slice code_fixing feedback by error type (e.g. borrow_checker vs type_mismatch).
3968:            // Resolve episode IDs once — reused for both feedback calls below.
3974:                crate::tensorzero::post_episode_feedback(
3987:                    success, \"Posted TZ feedback to all resolved episodes\"
4007:            // doesn't recognize it, but harmless — feedback is best-effort)
4008:            crate::tensorzero::post_episode_feedback(
4193:                    let triggers: Vec<coordination::feedback::ErrorCategory> = candidate
4503:pub fn check_for_resume(repo_root: &Path) -> Option<SwarmResumeFile> {
4750:                category: coordination::feedback::ErrorCategory::Other,
```

## SwarmResumeFile Struct Definition (lines 4480-4540)
```rust
/// Saved state for session resume after SLURM preemption or crash.
///
/// Written to `.swarm-resume.json` in the repo root on failure.
/// Checked on startup to restore worktree, iteration count, and escalation state.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SwarmResumeFile {
    /// Issue being worked on
    pub issue: BeadsIssue,
    /// Worktree path for the in-progress work
    pub worktree_path: String,
    /// Current iteration count
    pub iteration: u32,
    /// Escalation state summary
    pub escalation_summary: String,
    /// Current tier
    pub current_tier: String,
    /// Total iterations across all tiers
    pub total_iterations: u32,
    /// Timestamp when saved
    pub saved_at: String,
}

/// Check for a resume file and return the data if found.
pub fn check_for_resume(repo_root: &Path) -> Option<SwarmResumeFile> {
    let resume_path = repo_root.join(".swarm-resume.json");
    if resume_path.exists() {
        match std::fs::read_to_string(&resume_path) {
            Ok(contents) => match serde_json::from_str::<SwarmResumeFile>(&contents) {
                Ok(resume) => {
                    info!(
                        issue = %resume.issue.id,
                        worktree = %resume.worktree_path,
                        iteration = resume.iteration,
                        \"Found resume file — previous session can be continued\"
                    );
                    Some(resume)
                }
                Err(e) => {
                    warn!(\"Failed to parse resume file: {e}\");
                    None
                }
            },
            Err(e) => {
                warn!(\"Failed to read resume file: {e}\");
                None
            }
        }
    } else {
        None
    }
}

/// Clear the resume file after successful completion.
pub(crate) fn clear_resume_file(repo_root: &Path) {
    let resume_path = repo_root.join(".swarm-resume.json");
    if resume_path.exists() {
        let _ = std::fs::remove_file(&resume_path);
    }
}

/// Prompt an agent with exponential backoff retry for transient errors.
```

## TODO/FIXME References
```
(no matches found)
```
