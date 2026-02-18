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

        // Gate 1: cargo fmt --check
        if self.config.check_fmt {
            let result = self.run_fmt_gate().await;
            let failed = result.outcome == GateOutcome::Failed;
            report.add_gate(result);
            if failed && !self.config.comprehensive {
                self.skip_remaining(&mut report, &["clippy", "check", "test"]);
                report.finalize(start.elapsed());
                return report;
            }
        }

        // Gate 2: cargo clippy -D warnings
        if self.config.check_clippy {
            let result = self.run_clippy_gate().await;
            let failed = result.outcome == GateOutcome::Failed;
            report.add_gate(result);
            if failed && !self.config.comprehensive {
                self.skip_remaining(&mut report, &["check", "test"]);
                report.finalize(start.elapsed());
                return report;
            }
        }

        // Gate 3: cargo check --message-format=json
        if self.config.check_compile {
            let result = self.run_check_gate().await;
            let failed = result.outcome == GateOutcome::Failed;
            report.add_gate(result);
            if failed && !self.config.comprehensive {
                self.skip_remaining(&mut report, &["test"]);
                report.finalize(start.elapsed());
                return report;
            }
        }

        // Gate 4: cargo test
        if self.config.check_test {
            let result = self.run_test_gate().await;
            report.add_gate(result);
        }

        report.finalize(start.elapsed());
        report
    }

    /// Run a tokio command with the configured gate timeout.
    ///
    /// Returns `Ok(output)` on success, `Err(message)` on timeout or spawn failure.
    async fn run_with_timeout(
        &self,
        cmd: &mut tokio::process::Command,
    ) -> Result<std::process::Output, String> {
        cmd.current_dir(&self.working_dir).kill_on_drop(true);
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
        for gate in gates {
            // Only skip gates that are enabled in config
            let should_skip = match *gate {
                "fmt" => self.config.check_fmt,
                "clippy" => self.config.check_clippy,
                "check" => self.config.check_compile,
                "test" => self.config.check_test,
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
}
