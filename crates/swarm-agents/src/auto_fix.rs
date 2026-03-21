//! Automated fix logic for trivial verifier failures.
//!
//! The "Janitor" layer: handle mechanical fixes before involving expensive models.
//! For Rust: `cargo clippy --fix` + `cargo fmt`.
//! For non-Rust: runs auto-fix commands from `.swarm/profile.toml`.

use std::path::Path;

use tracing::{info, warn};

use coordination::{LanguageProfile, ScriptVerifier, Verifier, VerifierConfig, VerifierReport};

use crate::acceptance::AcceptancePolicy;

/// Try to auto-fix trivial verifier failures without LLM delegation.
///
/// For Rust targets: runs `cargo clippy --fix` + `cargo fmt`.
/// For non-Rust targets: runs `[[auto_fix]]` commands from the language profile.
///
/// If fixes are applied, re-runs the verifier and returns the updated report.
pub async fn try_auto_fix(
    wt_path: &Path,
    verifier_config: &VerifierConfig,
    iteration: u32,
    language_profile: &Option<LanguageProfile>,
) -> Option<VerifierReport> {
    // Dispatch to profile-based auto-fix for non-Rust targets
    if let Some(profile) = language_profile {
        if !profile.is_rust() {
            return try_auto_fix_script(wt_path, profile, iteration).await;
        }
    }

    // --- Rust auto-fix path (unchanged) ---
    try_auto_fix_rust(wt_path, verifier_config, iteration).await
}

/// Auto-fix for non-Rust targets using profile commands.
async fn try_auto_fix_script(
    wt_path: &Path,
    profile: &LanguageProfile,
    iteration: u32,
) -> Option<VerifierReport> {
    let script_verifier = ScriptVerifier::new(wt_path, profile.clone());
    let attempted = script_verifier.run_auto_fix().await;

    if !attempted {
        return None;
    }

    // Check if there are actual changes
    let status = tokio::process::Command::new("git")
        .args(["diff", "--quiet"])
        .current_dir(wt_path)
        .output()
        .await;

    let has_changes = matches!(status, Ok(ref out) if !out.status.success());
    if !has_changes {
        info!(iteration, "auto-fix (script): no changes produced");
        return None;
    }

    // Commit auto-fix changes
    let _ = tokio::process::Command::new("git")
        .args(["add", "."])
        .current_dir(wt_path)
        .output()
        .await;

    let msg = format!(
        "swarm: auto-fix iteration {iteration} ({})",
        profile.language
    );
    let _ = tokio::process::Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(wt_path)
        .output()
        .await;

    info!(
        iteration,
        language = %profile.language,
        "auto-fix (script): committed changes, re-running verifier"
    );

    let report = script_verifier.run_pipeline().await;

    if report.all_green {
        info!(
            iteration,
            summary = %report.summary(),
            "auto-fix (script): verifier now passes!"
        );
    } else {
        info!(
            iteration,
            summary = %report.summary(),
            "auto-fix (script): verifier still failing"
        );
    }
    Some(report)
}

/// Auto-fix for Rust targets (cargo clippy --fix + cargo fmt).
async fn try_auto_fix_rust(
    wt_path: &Path,
    verifier_config: &VerifierConfig,
    iteration: u32,
) -> Option<VerifierReport> {
    // Build package args for scoped commands
    let mut pkg_args: Vec<&str> = Vec::new();
    for pkg in &verifier_config.packages {
        pkg_args.push("-p");
        pkg_args.push(pkg);
    }

    // Step 1: Try cargo clippy --fix for MachineApplicable suggestions
    let mut clippy_args = vec!["clippy", "--fix", "--allow-dirty", "--allow-staged"];
    clippy_args.extend_from_slice(&pkg_args);
    clippy_args.extend_from_slice(&["--", "-D", "warnings"]);

    let clippy_fix = tokio::process::Command::new("cargo")
        .args(&clippy_args)
        .current_dir(wt_path)
        .output()
        .await;

    let clippy_fixed = match clippy_fix {
        Ok(ref out) if out.status.success() => {
            info!(iteration, "auto-fix: cargo clippy --fix succeeded");
            true
        }
        Ok(ref out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            warn!(
                iteration,
                "auto-fix: cargo clippy --fix partial: {}",
                &stderr[..stderr.len().min(200)]
            );
            true // Still worth re-checking
        }
        Err(e) => {
            warn!(iteration, "auto-fix: cargo clippy --fix failed to run: {e}");
            false
        }
    };

    // Step 2: Run cargo fmt to fix formatting
    let mut fmt_args = vec!["fmt"];
    for pkg in &verifier_config.packages {
        fmt_args.push("--package");
        fmt_args.push(pkg);
    }

    let fmt_fix = tokio::process::Command::new("cargo")
        .args(&fmt_args)
        .current_dir(wt_path)
        .output()
        .await;

    let fmt_fixed = match fmt_fix {
        Ok(ref out) if out.status.success() => {
            info!(iteration, "auto-fix: cargo fmt succeeded");
            true
        }
        Ok(ref out) => {
            let stderr = String::from_utf8_lossy(&out.stderr);
            warn!(
                iteration,
                "auto-fix: cargo fmt failed (syntax error?): {}",
                &stderr[..stderr.len().min(200)]
            );
            false
        }
        Err(e) => {
            warn!(iteration, "auto-fix: cargo fmt failed to run: {e}");
            false
        }
    };

    if !clippy_fixed && !fmt_fixed {
        return None; // Nothing was attempted
    }

    // Check if there are actual changes to commit
    let status = tokio::process::Command::new("git")
        .args(["diff", "--quiet"])
        .current_dir(wt_path)
        .output()
        .await;

    let has_changes = matches!(status, Ok(ref out) if !out.status.success());
    if !has_changes {
        info!(iteration, "auto-fix: no changes produced");
        return None;
    }

    // Commit auto-fix changes
    let _ = tokio::process::Command::new("git")
        .args(["add", "."])
        .current_dir(wt_path)
        .output()
        .await;

    let msg = format!("swarm: auto-fix iteration {iteration} (clippy --fix + fmt)");
    let _ = tokio::process::Command::new("git")
        .args(["commit", "-m", &msg])
        .current_dir(wt_path)
        .output()
        .await;

    info!(
        iteration,
        "auto-fix: committed changes, re-running verifier"
    );

    // Re-run the full verifier pipeline
    let verifier = Verifier::new(wt_path, verifier_config.clone());
    let report = verifier.run_pipeline().await;

    if report.all_green {
        info!(
            iteration,
            summary = %report.summary(),
            "auto-fix: verifier now passes! Skipping LLM delegation"
        );
        Some(report)
    } else {
        info!(
            iteration,
            summary = %report.summary(),
            "auto-fix: verifier still failing after auto-fix"
        );
        // Return the updated report so the next iteration uses it
        Some(report)
    }
}

/// Returns `true` when the auto-fix false-positive guard should apply.
///
/// The guard fires only when auto-fix actually ran this iteration AND a minimum
/// agent diff size is configured. This prevents rejecting legitimate small fixes
/// that pass the verifier on their own merit (i.e. without auto-fix).
pub(crate) fn should_reject_auto_fix(auto_fix_applied: bool, policy: &AcceptancePolicy) -> bool {
    auto_fix_applied && policy.min_diff_lines > 0
}
