//! Verifier Pipeline — Sequential execution of quality gates
//!
//! Runs cargo fmt, clippy, check, and test in order, stopping at the first
//! failure by default (fail-fast) or running all gates (comprehensive mode).
//!
//! All gates are run async with `tokio::process::Command` and enforced
//! timeout via `tokio::time::timeout(gate_timeout_secs)`.

use crate::feedback::compiler::CargoMessage;
use crate::feedback::error_parser::RustcErrorParser;
use crate::verifier::report::{GateOutcome, GateResult, VerifierReport};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

/// Configuration for the Verifier pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifierConfig {
    /// Run all gates even if earlier ones fail
    pub comprehensive: bool,
    /// Include `cargo fmt --check` gate
    pub check_fmt: bool,
    /// Include `cargo clippy -D warnings` gate
    pub check_clippy: bool,
    /// Include `cargo check` gate
    pub check_compile: bool,
    /// Include `cargo test` gate
    pub check_test: bool,
    /// Maximum time per gate (seconds)
    pub gate_timeout_secs: u64,
    /// Truncate stderr output to this many bytes
    pub stderr_max_bytes: usize,
    /// Additional cargo flags (e.g., --features, --manifest-path)
    pub extra_cargo_args: Vec<String>,
    /// Scope cargo commands to specific packages (e.g., ["swarm-agents"])
    /// When empty, commands target the entire workspace.
    pub packages: Vec<String>,
    /// Include `sg scan` (ast-grep) warning gate
    pub check_sg: bool,
    /// Include `cargo deny check` gate (security advisories, license compliance).
    /// Requires `cargo-deny` to be installed; degrades gracefully if missing.
    pub check_deny: bool,
    /// Include `cargo doc --no-deps` gate (doc tests, rustdoc lints).
    pub check_doc: bool,
    /// Use `cargo nextest run` instead of `cargo test` when available.
    /// Falls back to `cargo test` if nextest is not installed.
    pub use_nextest: bool,
    /// Enable adaptive gate selection based on diff risk profile.
    /// When true, deny/doc/nextest are auto-enabled based on what changed.
    pub adaptive: bool,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            comprehensive: false,
            check_fmt: true,
            check_clippy: true,
            check_compile: true,
            check_test: true,
            gate_timeout_secs: 300,
            stderr_max_bytes: 4096,
            extra_cargo_args: Vec::new(),
            packages: Vec::new(),
            check_sg: false,
            check_deny: false,
            check_doc: false,
            use_nextest: false,
            adaptive: false,
        }
    }
}

impl VerifierConfig {
    /// Quick check — only fmt and cargo check (no clippy, no tests)
    pub fn quick() -> Self {
        Self {
            check_clippy: false,
            check_test: false,
            ..Default::default()
        }
    }

    /// Full pipeline with all gates
    pub fn full() -> Self {
        Self::default()
    }

    /// Compilation-only (check + clippy, no fmt or tests)
    pub fn compile_only() -> Self {
        Self {
            check_fmt: false,
            check_test: false,
            ..Default::default()
        }
    }

    /// Documentation-oriented profile (fmt + compile only, no clippy or tests).
    ///
    /// Suitable for doc/RFC tasks that only need basic formatting and syntax checking.
    pub fn docs() -> Self {
        Self {
            check_fmt: true,
            check_clippy: false,
            check_compile: true,
            check_test: false,
            ..Default::default()
        }
    }

    /// All gates disabled — no verification at all.
    ///
    /// Useful for tasks where deterministic quality gates don't apply
    /// (e.g., pure documentation, planning, or external scaffolding).
    pub fn none() -> Self {
        Self {
            check_fmt: false,
            check_clippy: false,
            check_compile: false,
            check_test: false,
            ..Default::default()
        }
    }

    /// Adaptive mode — core gates on, extras auto-selected by diff risk profile.
    ///
    /// Runs fmt/clippy/check/test as baseline, then enables deny/doc/nextest
    /// based on what the diff actually changed (Cargo.toml → deny, pub API → doc, etc.).
    pub fn adaptive() -> Self {
        Self {
            adaptive: true,
            ..Default::default()
        }
    }

    /// Apply a diff risk profile to auto-enable adaptive gates.
    ///
    /// Only modifies config when `self.adaptive` is true. Mutates in place
    /// so callers can inspect which gates were auto-enabled.
    pub fn apply_risk_profile(&mut self, profile: &super::risk_profile::DiffRiskProfile) {
        if !self.adaptive {
            return;
        }

        if profile.should_run_deny() {
            self.check_deny = true;
        }
        if profile.should_run_doc() {
            self.check_doc = true;
        }
        if profile.should_prefer_nextest() {
            self.use_nextest = true;
        }
    }
}

/// The Verifier — runs the deterministic quality gate pipeline
pub struct Verifier {
    /// Working directory (crate root)
    working_dir: PathBuf,
    /// Configuration
    config: VerifierConfig,
}

impl Verifier {
    /// Create a new Verifier for the given crate directory
    pub fn new(working_dir: impl AsRef<Path>, config: VerifierConfig) -> Self {
        Self {
            working_dir: working_dir.as_ref().to_path_buf(),
            config,
        }
    }

    /// Run the full verification pipeline
    ///
    /// Returns a structured VerifierReport with classified errors and gate results.
    pub async fn run_pipeline(&self) -> VerifierReport {
        let start = Instant::now();
        let mut report = VerifierReport::new(self.working_dir.display().to_string());

        // Populate git info
        report.branch = self.git_branch();
        report.commit = self.git_commit();

        // Adaptive gate selection: analyze diff risk profile and auto-enable gates
        let mut config = self.config.clone();
        if config.adaptive {
            let profile = super::risk_profile::DiffRiskProfile::from_working_dir(&self.working_dir);
            tracing::info!(
                files_changed = profile.files_changed,
                lines_added = profile.lines_added,
                has_unsafe = profile.has_unsafe,
                has_cargo_toml = profile.has_cargo_toml_change,
                has_pub_api = profile.has_public_api_change,
                "Diff risk profile"
            );
            config.apply_risk_profile(&profile);
            report.risk_profile = Some(profile);
        }

        // Gate 0: Pre-gate safety scan for dangerous code patterns
        let safety_warnings = super::safety_scan::scan_diff(&self.working_dir);
        if !safety_warnings.is_empty() {
            tracing::warn!(
                count = safety_warnings.len(),
                "Safety scan detected dangerous patterns in agent diff"
            );
            for w in &safety_warnings {
                tracing::warn!(
                    category = %w.category,
                    file = %w.file,
                    reason = %w.reason,
                    "Safety warning"
                );
            }
        }
        report.safety_warnings = safety_warnings;

        // Gate 1: cargo fmt --check
        if config.check_fmt {
            let result = self.run_fmt_gate().await;
            let failed = result.outcome == GateOutcome::Failed;
            report.add_gate(result);
            if failed && !config.comprehensive {
                self.skip_remaining_with(
                    &mut report,
                    &["clippy", "sg", "check", "test", "deny", "doc"],
                    &config,
                );
                report.finalize(start.elapsed());
                return report;
            }
        }

        // Gate 2: cargo clippy -D warnings
        if config.check_clippy {
            let result = self.run_clippy_gate().await;
            let failed = result.outcome == GateOutcome::Failed;
            report.add_gate(result);
            if failed && !config.comprehensive {
                self.skip_remaining_with(
                    &mut report,
                    &["sg", "check", "test", "deny", "doc"],
                    &config,
                );
                report.finalize(start.elapsed());
                return report;
            }
        }

        // Gate 2.5: sg scan (ast-grep) — warning-only, never blocks pipeline
        if config.check_sg {
            let result = self.run_sg_gate().await;
            report.add_gate(result);
            // Never fail-fast on warnings — always continue to check gate
        }

        // Gate 3: cargo check --message-format=json
        if config.check_compile {
            let result = self.run_check_gate().await;
            let failed = result.outcome == GateOutcome::Failed;
            report.add_gate(result);
            if failed && !config.comprehensive {
                self.skip_remaining_with(&mut report, &["test", "deny", "doc"], &config);
                report.finalize(start.elapsed());
                return report;
            }
        }

        // Gate 4: cargo test (or nextest if configured)
        if config.check_test {
            let result = if config.use_nextest {
                self.run_nextest_gate().await
            } else {
                self.run_test_gate().await
            };
            let failed = result.outcome == GateOutcome::Failed;
            report.add_gate(result);
            if failed && !config.comprehensive {
                self.skip_remaining_with(&mut report, &["deny", "doc"], &config);
                report.finalize(start.elapsed());
                return report;
            }
        }

        // Gate 5: cargo deny check (warning-only — security advisories, license compliance)
        if config.check_deny {
            let result = self.run_deny_gate().await;
            report.add_gate(result);
            // deny is warning-only: advisory findings don't block the pipeline
        }

        // Gate 6: cargo doc --no-deps (doc tests + rustdoc lints)
        if config.check_doc {
            let result = self.run_doc_gate().await;
            report.add_gate(result);
            // doc gate is warning-only for now
        }

        report.finalize(start.elapsed());
        report
    }

    /// Run a tokio command with the configured gate timeout.
    ///
    /// On Unix, creates a new process group so that on timeout the entire
    /// process tree (including descendants like test binaries) is killed.
    /// Returns `Ok(output)` on success, `Err(message)` on timeout or spawn failure.
    async fn run_with_timeout(
        &self,
        cmd: &mut tokio::process::Command,
    ) -> Result<std::process::Output, String> {
        cmd.current_dir(&self.working_dir).kill_on_drop(true);

        // Create a new process group so we can kill the entire tree on timeout.
        // process_group(0) calls setpgid(0, 0) which puts the child in its own
        // group. When kill_on_drop fires, tokio kills the child; the group
        // ensures all descendants (e.g. cargo-spawned test binaries) also die.
        #[cfg(unix)]
        cmd.process_group(0);

        let timeout_dur = Duration::from_secs(self.config.gate_timeout_secs);
        match tokio::time::timeout(timeout_dur, cmd.output()).await {
            Ok(Ok(output)) => Ok(output),
            Ok(Err(e)) => Err(format!("Failed to execute: {e}")),
            Err(_) => Err(format!(
                "Gate timed out after {}s",
                self.config.gate_timeout_secs
            )),
        }
    }

    /// Run `cargo fmt --check`
    async fn run_fmt_gate(&self) -> GateResult {
        let start = Instant::now();

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.arg("fmt");
        if self.config.packages.is_empty() {
            cmd.arg("--all");
        } else {
            for pkg in &self.config.packages {
                cmd.args(["--package", pkg]);
            }
        }
        cmd.arg("--check");
        for arg in &self.config.extra_cargo_args {
            cmd.arg(arg);
        }

        match self.run_with_timeout(&mut cmd).await {
            Ok(output) => {
                let stderr = self.truncate_stderr(&output.stderr);
                let exit_code = output.status.code();
                let passed = output.status.success();

                GateResult {
                    gate: "fmt".to_string(),
                    outcome: if passed {
                        GateOutcome::Passed
                    } else {
                        GateOutcome::Failed
                    },
                    duration_ms: start.elapsed().as_millis() as u64,
                    exit_code,
                    error_count: if passed { 0 } else { 1 },
                    warning_count: 0,
                    errors: vec![],
                    stderr_excerpt: if passed { None } else { Some(stderr) },
                }
            }
            Err(e) => GateResult {
                gate: "fmt".to_string(),
                outcome: GateOutcome::Failed,
                duration_ms: start.elapsed().as_millis() as u64,
                exit_code: None,
                error_count: 1,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: Some(e),
            },
        }
    }

    /// Run `cargo clippy -D warnings` with JSON output
    async fn run_clippy_gate(&self) -> GateResult {
        let start = Instant::now();

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.arg("clippy");
        for pkg in &self.config.packages {
            cmd.args(["-p", pkg]);
        }
        cmd.args(["--message-format=json", "--", "-D", "warnings"]);

        match self.run_with_timeout(&mut cmd).await {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let messages = Self::parse_json_messages(&stdout);
                let result = Self::output_to_compile_result(&output, messages);
                self.compile_result_to_gate("clippy", result, start.elapsed())
            }
            Err(e) => GateResult {
                gate: "clippy".to_string(),
                outcome: GateOutcome::Failed,
                duration_ms: start.elapsed().as_millis() as u64,
                exit_code: None,
                error_count: 1,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: Some(e),
            },
        }
    }

    /// Run `sg scan` (ast-grep) as a warning-only gate.
    ///
    /// Scans the working directory against project rules in `rules/`.
    /// Produces `GateOutcome::Warning` when diagnostics are found — never blocks the pipeline.
    async fn run_sg_gate(&self) -> GateResult {
        let start = Instant::now();

        // Locate rules directory: try working_dir/rules/ first, then repo root
        let rules_dir = if self.working_dir.join("rules").is_dir() {
            self.working_dir.join("rules")
        } else {
            // Walk up to find rules/ (worktrees may be nested)
            let mut candidate = self.working_dir.as_path();
            loop {
                let rules_path = candidate.join("rules");
                if rules_path.is_dir() {
                    break rules_path;
                }
                match candidate.parent() {
                    Some(parent) => candidate = parent,
                    None => break self.working_dir.join("rules"),
                }
            }
        };

        let mut cmd = tokio::process::Command::new("sg");
        cmd.args(["scan", "--rule"]);
        cmd.arg(&rules_dir);
        cmd.arg(&self.working_dir);

        match self.run_with_timeout(&mut cmd).await {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = self.truncate_stderr(&output.stderr);

                // Count diagnostic lines (non-empty lines from stdout that contain file paths)
                let diagnostic_count = stdout.lines().filter(|l| !l.trim().is_empty()).count();

                let has_diagnostics = !output.status.success() || diagnostic_count > 0;

                GateResult {
                    gate: "sg".to_string(),
                    outcome: if has_diagnostics {
                        GateOutcome::Warning
                    } else {
                        GateOutcome::Passed
                    },
                    duration_ms: start.elapsed().as_millis() as u64,
                    exit_code: output.status.code(),
                    error_count: 0,
                    warning_count: diagnostic_count,
                    errors: vec![],
                    stderr_excerpt: if has_diagnostics {
                        let details = if stdout.is_empty() {
                            stderr
                        } else {
                            self.truncate_str(&stdout)
                        };
                        Some(details)
                    } else {
                        None
                    },
                }
            }
            Err(e) => {
                // sg not installed or timed out — degrade gracefully to Warning
                GateResult {
                    gate: "sg".to_string(),
                    outcome: GateOutcome::Warning,
                    duration_ms: start.elapsed().as_millis() as u64,
                    exit_code: None,
                    error_count: 0,
                    warning_count: 0,
                    errors: vec![],
                    stderr_excerpt: Some(format!("sg scan unavailable: {e}")),
                }
            }
        }
    }

    /// Run `cargo check --message-format=json`
    async fn run_check_gate(&self) -> GateResult {
        let start = Instant::now();

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.arg("check");
        for pkg in &self.config.packages {
            cmd.args(["-p", pkg]);
        }
        cmd.arg("--message-format=json");

        match self.run_with_timeout(&mut cmd).await {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let messages = Self::parse_json_messages(&stdout);
                let result = Self::output_to_compile_result(&output, messages);
                self.compile_result_to_gate("check", result, start.elapsed())
            }
            Err(e) => GateResult {
                gate: "check".to_string(),
                outcome: GateOutcome::Failed,
                duration_ms: start.elapsed().as_millis() as u64,
                exit_code: None,
                error_count: 1,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: Some(e),
            },
        }
    }

    /// Run `cargo test`
    async fn run_test_gate(&self) -> GateResult {
        let start = Instant::now();

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.arg("test");
        for pkg in &self.config.packages {
            cmd.args(["-p", pkg]);
        }
        for arg in &self.config.extra_cargo_args {
            cmd.arg(arg);
        }

        match self.run_with_timeout(&mut cmd).await {
            Ok(output) => {
                let stderr = self.truncate_stderr(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let exit_code = output.status.code();
                let passed = output.status.success();

                // Count test failures from stdout
                let (test_count, fail_count) = self.parse_test_summary(&stdout);

                GateResult {
                    gate: "test".to_string(),
                    outcome: if passed {
                        GateOutcome::Passed
                    } else {
                        GateOutcome::Failed
                    },
                    duration_ms: start.elapsed().as_millis() as u64,
                    exit_code,
                    error_count: fail_count,
                    warning_count: 0,
                    errors: vec![],
                    stderr_excerpt: if passed {
                        None
                    } else {
                        Some(format!(
                            "{} tests, {} failures\n\n{}",
                            test_count, fail_count, stderr
                        ))
                    },
                }
            }
            Err(e) => GateResult {
                gate: "test".to_string(),
                outcome: GateOutcome::Failed,
                duration_ms: start.elapsed().as_millis() as u64,
                exit_code: None,
                error_count: 1,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: Some(e),
            },
        }
    }

    /// Run `cargo nextest run` as an alternative to `cargo test`.
    ///
    /// Falls back to `cargo test` if nextest is not installed.
    async fn run_nextest_gate(&self) -> GateResult {
        let start = Instant::now();

        // Probe for nextest availability
        let probe = tokio::process::Command::new("cargo")
            .args(["nextest", "--version"])
            .output()
            .await;

        let nextest_available = probe.map(|o| o.status.success()).unwrap_or(false);

        if !nextest_available {
            tracing::info!("cargo-nextest not installed, falling back to cargo test");
            return self.run_test_gate().await;
        }

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.args(["nextest", "run"]);
        for pkg in &self.config.packages {
            cmd.args(["-p", pkg]);
        }
        for arg in &self.config.extra_cargo_args {
            cmd.arg(arg);
        }

        match self.run_with_timeout(&mut cmd).await {
            Ok(output) => {
                let stderr = self.truncate_stderr(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let exit_code = output.status.code();
                let passed = output.status.success();

                // nextest outputs its own summary format; parse test counts
                let (test_count, fail_count) = self.parse_nextest_summary(&stdout, &stderr);

                GateResult {
                    gate: "nextest".to_string(),
                    outcome: if passed {
                        GateOutcome::Passed
                    } else {
                        GateOutcome::Failed
                    },
                    duration_ms: start.elapsed().as_millis() as u64,
                    exit_code,
                    error_count: fail_count,
                    warning_count: 0,
                    errors: vec![],
                    stderr_excerpt: if passed {
                        None
                    } else {
                        Some(format!(
                            "{} tests, {} failures\n\n{}",
                            test_count, fail_count, stderr
                        ))
                    },
                }
            }
            Err(e) => GateResult {
                gate: "nextest".to_string(),
                outcome: GateOutcome::Failed,
                duration_ms: start.elapsed().as_millis() as u64,
                exit_code: None,
                error_count: 1,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: Some(e),
            },
        }
    }

    /// Run `cargo deny check` for security advisories and license compliance.
    ///
    /// Warning-only: findings are reported but don't block the pipeline.
    /// Degrades gracefully if `cargo-deny` is not installed.
    async fn run_deny_gate(&self) -> GateResult {
        let start = Instant::now();

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.args(["deny", "check"]);

        match self.run_with_timeout(&mut cmd).await {
            Ok(output) => {
                let stderr = self.truncate_stderr(&output.stderr);
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let exit_code = output.status.code();
                let passed = output.status.success();

                // Count advisory/warning lines
                let warning_count = stdout
                    .lines()
                    .filter(|l| l.contains("warning") || l.contains("WARN"))
                    .count();
                let error_count = stdout
                    .lines()
                    .filter(|l| l.contains("error") || l.contains("ERROR"))
                    .count();

                GateResult {
                    gate: "deny".to_string(),
                    outcome: if passed {
                        GateOutcome::Passed
                    } else {
                        GateOutcome::Warning // deny is advisory — never blocks
                    },
                    duration_ms: start.elapsed().as_millis() as u64,
                    exit_code,
                    error_count,
                    warning_count,
                    errors: vec![],
                    stderr_excerpt: if passed {
                        None
                    } else {
                        let details = if stdout.is_empty() {
                            stderr
                        } else {
                            self.truncate_str(&stdout)
                        };
                        Some(details)
                    },
                }
            }
            Err(e) => {
                // cargo-deny not installed or timed out — degrade gracefully
                GateResult {
                    gate: "deny".to_string(),
                    outcome: GateOutcome::Warning,
                    duration_ms: start.elapsed().as_millis() as u64,
                    exit_code: None,
                    error_count: 0,
                    warning_count: 0,
                    errors: vec![],
                    stderr_excerpt: Some(format!("cargo deny unavailable: {e}")),
                }
            }
        }
    }

    /// Run `cargo doc --no-deps` for doc test validation and rustdoc lints.
    ///
    /// Warning-only: broken doc links are reported but don't block the pipeline.
    async fn run_doc_gate(&self) -> GateResult {
        let start = Instant::now();

        let mut cmd = tokio::process::Command::new("cargo");
        cmd.arg("doc").arg("--no-deps");
        for pkg in &self.config.packages {
            cmd.args(["-p", pkg]);
        }
        // Enable rustdoc lint warnings
        cmd.env("RUSTDOCFLAGS", "-D warnings");

        match self.run_with_timeout(&mut cmd).await {
            Ok(output) => {
                let stderr = self.truncate_stderr(&output.stderr);
                let exit_code = output.status.code();
                let passed = output.status.success();

                // Count rustdoc warnings/errors from stderr
                let stderr_str = String::from_utf8_lossy(&output.stderr).to_string();
                let warning_count = stderr_str
                    .lines()
                    .filter(|l| l.contains("warning:"))
                    .count();
                let error_count = stderr_str.lines().filter(|l| l.contains("error:")).count();

                GateResult {
                    gate: "doc".to_string(),
                    outcome: if passed {
                        GateOutcome::Passed
                    } else {
                        GateOutcome::Warning // doc is advisory — never blocks
                    },
                    duration_ms: start.elapsed().as_millis() as u64,
                    exit_code,
                    error_count,
                    warning_count,
                    errors: vec![],
                    stderr_excerpt: if passed { None } else { Some(stderr) },
                }
            }
            Err(e) => GateResult {
                gate: "doc".to_string(),
                outcome: GateOutcome::Warning,
                duration_ms: start.elapsed().as_millis() as u64,
                exit_code: None,
                error_count: 0,
                warning_count: 0,
                errors: vec![],
                stderr_excerpt: Some(format!("cargo doc failed: {e}")),
            },
        }
    }

    /// Parse JSON lines from cargo output (same as Compiler::parse_json_messages)
    fn parse_json_messages(output: &str) -> Vec<CargoMessage> {
        output
            .lines()
            .filter_map(|line| serde_json::from_str::<CargoMessage>(line).ok())
            .collect()
    }

    /// Convert a process output + parsed messages into a lightweight CompileResult-like tuple.
    fn output_to_compile_result(
        output: &std::process::Output,
        messages: Vec<CargoMessage>,
    ) -> CompileResultLite {
        CompileResultLite {
            success: output.status.success(),
            exit_code: output.status.code(),
            messages,
            raw_stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        }
    }

    /// Convert a CompileResultLite to a GateResult with parsed errors
    fn compile_result_to_gate(
        &self,
        gate_name: &str,
        result: CompileResultLite,
        duration: Duration,
    ) -> GateResult {
        let errors = RustcErrorParser::parse_cargo_messages(&result.messages);
        let warning_count = result.messages.iter().filter(|m| m.is_warning()).count();

        GateResult {
            gate: gate_name.to_string(),
            outcome: if result.success {
                GateOutcome::Passed
            } else {
                GateOutcome::Failed
            },
            duration_ms: duration.as_millis() as u64,
            exit_code: result.exit_code,
            error_count: errors.len(),
            warning_count,
            errors,
            stderr_excerpt: if result.success {
                None
            } else {
                Some(self.truncate_str(&result.raw_stderr))
            },
        }
    }

    /// Add skipped gates to the report
    fn skip_remaining(&self, report: &mut VerifierReport, gates: &[&str]) {
        self.skip_remaining_with(report, gates, &self.config);
    }

    /// Add skipped gates to the report using an external config (for adaptive mode).
    fn skip_remaining_with(
        &self,
        report: &mut VerifierReport,
        gates: &[&str],
        config: &VerifierConfig,
    ) {
        for gate in gates {
            // Only skip gates that are enabled in config
            let should_skip = match *gate {
                "fmt" => config.check_fmt,
                "clippy" => config.check_clippy,
                "sg" => config.check_sg,
                "check" => config.check_compile,
                "test" => config.check_test,
                "deny" => config.check_deny,
                "doc" => config.check_doc,
                _ => false,
            };

            if should_skip {
                report.add_gate(GateResult {
                    gate: gate.to_string(),
                    outcome: GateOutcome::Skipped,
                    duration_ms: 0,
                    exit_code: None,
                    error_count: 0,
                    warning_count: 0,
                    errors: vec![],
                    stderr_excerpt: None,
                });
            }
        }
    }

    /// Get current git branch
    fn git_branch(&self) -> Option<String> {
        Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .current_dir(&self.working_dir)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Get current git commit SHA
    fn git_commit(&self) -> Option<String> {
        Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .current_dir(&self.working_dir)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
    }

    /// Parse test summary from cargo test stdout
    ///
    /// Looks for lines like: "test result: FAILED. 5 passed; 2 failed; 0 ignored"
    fn parse_test_summary(&self, stdout: &str) -> (usize, usize) {
        let mut total_tests = 0;
        let mut total_failures = 0;

        for line in stdout.lines() {
            if line.starts_with("test result:") {
                // Extract the number before each keyword
                for part in line.split(';') {
                    let part = part.trim();
                    // Find the number immediately before "passed", "failed", or "ignored"
                    let words: Vec<&str> = part.split_whitespace().collect();
                    for (i, word) in words.iter().enumerate() {
                        if i > 0 {
                            if let Ok(n) = words[i - 1].parse::<usize>() {
                                if *word == "passed" || word.starts_with("passed") {
                                    total_tests += n;
                                } else if *word == "failed" || word.starts_with("failed") {
                                    total_tests += n;
                                    total_failures += n;
                                } else if *word == "ignored" || word.starts_with("ignored") {
                                    total_tests += n;
                                }
                            }
                        }
                    }
                }
            }
        }

        (total_tests, total_failures)
    }

    /// Parse nextest summary output.
    ///
    /// Nextest outputs to stderr with lines like:
    /// "Summary [  0.123s] 10 tests run: 9 passed, 1 failed, 0 skipped"
    fn parse_nextest_summary(&self, stdout: &str, stderr: &str) -> (usize, usize) {
        // Try stderr first (nextest's primary output), then stdout
        for text in [stderr, stdout] {
            for line in text.lines() {
                if line.contains("tests run:") {
                    // Parse "N tests run: M passed, K failed"
                    let words: Vec<&str> = line.split_whitespace().collect();
                    for (i, word) in words.iter().enumerate() {
                        if *word == "tests" && i > 0 {
                            if let Ok(n) = words[i - 1].parse::<usize>() {
                                // Found total tests — now find failures
                                let fail_count = words
                                    .iter()
                                    .enumerate()
                                    .find(|(_, w)| w.starts_with("failed"))
                                    .and_then(|(j, _)| {
                                        if j > 0 {
                                            words[j - 1].trim_end_matches(',').parse::<usize>().ok()
                                        } else {
                                            None
                                        }
                                    })
                                    .unwrap_or(0);
                                return (n, fail_count);
                            }
                        }
                    }
                }
            }
        }

        // Fallback: try standard cargo test format
        self.parse_test_summary(stdout)
    }

    /// Truncate stderr bytes to configured limit
    fn truncate_stderr(&self, stderr: &[u8]) -> String {
        let s = String::from_utf8_lossy(stderr);
        self.truncate_str(&s)
    }

    /// Truncate string to configured limit
    fn truncate_str(&self, s: &str) -> String {
        if s.len() <= self.config.stderr_max_bytes {
            s.to_string()
        } else {
            let truncated = &s[..self.config.stderr_max_bytes];
            format!("{}...\n[truncated at {} bytes]", truncated, s.len())
        }
    }
}

/// Lightweight compile result for async pipeline (avoids depending on sync Compiler).
struct CompileResultLite {
    success: bool,
    exit_code: Option<i32>,
    messages: Vec<CargoMessage>,
    raw_stderr: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verifier_config_defaults() {
        let config = VerifierConfig::default();
        assert!(config.check_fmt);
        assert!(config.check_clippy);
        assert!(config.check_compile);
        assert!(config.check_test);
        assert!(!config.check_sg);
        assert!(!config.comprehensive);
    }

    #[test]
    fn test_verifier_config_quick() {
        let config = VerifierConfig::quick();
        assert!(config.check_fmt);
        assert!(!config.check_clippy);
        assert!(config.check_compile);
        assert!(!config.check_test);
    }

    #[test]
    fn test_verifier_config_compile_only() {
        let config = VerifierConfig::compile_only();
        assert!(!config.check_fmt);
        assert!(config.check_clippy);
        assert!(config.check_compile);
        assert!(!config.check_test);
    }

    #[test]
    fn test_verifier_config_docs() {
        let config = VerifierConfig::docs();
        assert!(config.check_fmt);
        assert!(!config.check_clippy);
        assert!(config.check_compile);
        assert!(!config.check_test);
    }

    #[test]
    fn test_verifier_config_none() {
        let config = VerifierConfig::none();
        assert!(!config.check_fmt);
        assert!(!config.check_clippy);
        assert!(!config.check_compile);
        assert!(!config.check_test);
    }

    #[test]
    fn test_parse_test_summary() {
        let verifier = Verifier::new("/tmp", VerifierConfig::default());

        let stdout = "\
running 10 tests
test foo ... ok
test bar ... FAILED
test result: FAILED. 9 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out";

        let (total, failures) = verifier.parse_test_summary(stdout);
        assert_eq!(total, 10);
        assert_eq!(failures, 1);
    }

    #[test]
    fn test_parse_test_summary_all_pass() {
        let verifier = Verifier::new("/tmp", VerifierConfig::default());

        let stdout = "test result: ok. 15 passed; 0 failed; 2 ignored; 0 measured";

        let (total, failures) = verifier.parse_test_summary(stdout);
        assert_eq!(total, 17);
        assert_eq!(failures, 0);
    }

    #[tokio::test]
    async fn test_sg_gate_returns_warning_on_missing_sg() {
        // sg is unlikely to be installed in CI — verify graceful degradation
        let config = VerifierConfig {
            check_sg: true,
            check_fmt: false,
            check_clippy: false,
            check_compile: false,
            check_test: false,
            ..Default::default()
        };
        let verifier = Verifier::new("/tmp", config);
        let result = verifier.run_sg_gate().await;
        // Should produce Warning (not Failed) whether sg is installed or not
        assert!(
            result.outcome == GateOutcome::Warning || result.outcome == GateOutcome::Passed,
            "sg gate should never produce Failed, got: {:?}",
            result.outcome
        );
        assert_eq!(result.gate, "sg");
    }

    #[tokio::test]
    async fn test_sg_gate_does_not_block_pipeline() {
        // Run pipeline with only sg enabled — all_green should still be true
        // (Warning counts as passed for all_green)
        let config = VerifierConfig {
            check_sg: true,
            check_fmt: false,
            check_clippy: false,
            check_compile: false,
            check_test: false,
            ..Default::default()
        };
        let verifier = Verifier::new("/tmp", config);
        let report = verifier.run_pipeline().await;
        // sg gate is Warning or Passed — either way all_green should be true
        assert!(
            report.all_green,
            "Pipeline with only sg gate should be all_green"
        );
        assert_eq!(report.gates.len(), 1);
        assert_eq!(report.gates[0].gate, "sg");
    }

    #[test]
    fn test_sg_config_off_by_default_in_named_constructors() {
        assert!(!VerifierConfig::quick().check_sg);
        assert!(!VerifierConfig::full().check_sg);
        assert!(!VerifierConfig::compile_only().check_sg);
        assert!(!VerifierConfig::docs().check_sg);
        assert!(!VerifierConfig::none().check_sg);
    }

    #[test]
    fn test_truncate_stderr() {
        let config = VerifierConfig {
            stderr_max_bytes: 20,
            ..Default::default()
        };
        let verifier = Verifier::new("/tmp", config);

        let long_msg = "this is a very long error message that should be truncated";
        let truncated = verifier.truncate_str(long_msg);
        assert!(truncated.contains("truncated"));
        assert!(truncated.starts_with("this is a very long "));
    }

    #[test]
    fn test_verifier_config_adaptive() {
        let config = VerifierConfig::adaptive();
        assert!(config.adaptive);
        // Core gates on by default
        assert!(config.check_fmt);
        assert!(config.check_clippy);
        assert!(config.check_compile);
        assert!(config.check_test);
        // Extra gates off until risk profile applied
        assert!(!config.check_deny);
        assert!(!config.check_doc);
        assert!(!config.use_nextest);
    }

    #[test]
    fn test_adaptive_config_applies_risk_profile() {
        let mut config = VerifierConfig::adaptive();

        // Simulate Cargo.toml change
        let profile = super::super::risk_profile::DiffRiskProfile {
            has_cargo_toml_change: true,
            has_public_api_change: true,
            has_doc_change: false,
            files_changed: 5,
            lines_added: 200,
            ..Default::default()
        };

        config.apply_risk_profile(&profile);
        assert!(config.check_deny, "Cargo.toml change should enable deny");
        assert!(config.check_doc, "Public API change should enable doc");
        assert!(config.use_nextest, "Large changeset should enable nextest");
    }

    #[test]
    fn test_adaptive_noop_when_disabled() {
        let mut config = VerifierConfig::default();
        assert!(!config.adaptive);

        let profile = super::super::risk_profile::DiffRiskProfile {
            has_cargo_toml_change: true,
            ..Default::default()
        };

        config.apply_risk_profile(&profile);
        assert!(
            !config.check_deny,
            "Should not enable deny when adaptive is off"
        );
    }

    #[test]
    fn test_new_gates_off_by_default() {
        let config = VerifierConfig::default();
        assert!(!config.check_deny);
        assert!(!config.check_doc);
        assert!(!config.use_nextest);
        assert!(!config.adaptive);
    }

    #[test]
    fn test_new_gates_off_in_named_constructors() {
        for config in [
            VerifierConfig::quick(),
            VerifierConfig::full(),
            VerifierConfig::compile_only(),
            VerifierConfig::docs(),
            VerifierConfig::none(),
        ] {
            assert!(!config.check_deny, "deny should be off by default");
            assert!(!config.check_doc, "doc should be off by default");
            assert!(!config.use_nextest, "nextest should be off by default");
        }
    }

    #[test]
    fn test_parse_nextest_summary() {
        let verifier = Verifier::new("/tmp", VerifierConfig::default());

        let stderr = "Summary [  0.123s] 10 tests run: 9 passed, 1 failed, 0 skipped";
        let (total, failures) = verifier.parse_nextest_summary("", stderr);
        assert_eq!(total, 10);
        assert_eq!(failures, 1);
    }

    #[test]
    fn test_parse_nextest_summary_all_pass() {
        let verifier = Verifier::new("/tmp", VerifierConfig::default());

        let stderr = "Summary [  1.234s] 25 tests run: 25 passed, 0 failed, 0 skipped";
        let (total, failures) = verifier.parse_nextest_summary("", stderr);
        assert_eq!(total, 25);
        assert_eq!(failures, 0);
    }

    #[test]
    fn test_parse_nextest_summary_fallback_to_cargo() {
        let verifier = Verifier::new("/tmp", VerifierConfig::default());

        // No nextest format → should fall back to cargo test parser
        let stdout = "test result: ok. 5 passed; 0 failed; 0 ignored; 0 measured";
        let (total, failures) = verifier.parse_nextest_summary(stdout, "");
        assert_eq!(total, 5);
        assert_eq!(failures, 0);
    }

    #[tokio::test]
    async fn test_deny_gate_degrades_gracefully() {
        // cargo-deny is unlikely to be installed in all environments
        let verifier = Verifier::new("/tmp", VerifierConfig::default());
        let result = verifier.run_deny_gate().await;
        // Should produce Warning (not Failed) if deny is not installed
        assert!(
            result.outcome == GateOutcome::Warning || result.outcome == GateOutcome::Passed,
            "deny gate should never produce Failed, got: {:?}",
            result.outcome
        );
        assert_eq!(result.gate, "deny");
    }

    #[tokio::test]
    async fn test_doc_gate_on_invalid_dir() {
        let verifier = Verifier::new("/tmp/nonexistent-crate-dir", VerifierConfig::default());
        let result = verifier.run_doc_gate().await;
        // Should produce Warning — doc gate never blocks
        assert!(
            result.outcome == GateOutcome::Warning,
            "doc gate on invalid dir should be Warning, got: {:?}",
            result.outcome
        );
        assert_eq!(result.gate, "doc");
    }

    #[test]
    fn test_skip_remaining_with_includes_new_gates() {
        let config = VerifierConfig {
            check_deny: true,
            check_doc: true,
            ..Default::default()
        };
        let verifier = Verifier::new("/tmp", config.clone());
        let mut report = VerifierReport::new("/tmp".to_string());

        verifier.skip_remaining_with(&mut report, &["deny", "doc"], &config);
        assert_eq!(report.gates.len(), 2);
        assert_eq!(report.gates[0].gate, "deny");
        assert_eq!(report.gates[0].outcome, GateOutcome::Skipped);
        assert_eq!(report.gates[1].gate, "doc");
        assert_eq!(report.gates[1].outcome, GateOutcome::Skipped);
    }
}
