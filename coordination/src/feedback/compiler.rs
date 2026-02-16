//! Cargo check/clippy wrapper with JSON output parsing
//!
//! Runs cargo commands and captures structured output for analysis.

use serde::{Deserialize, Serialize};
use std::path::Path;
use std::process::Command;

/// Compiler wrapper for running cargo check/clippy
pub struct Compiler {
    /// Working directory (crate root)
    working_dir: std::path::PathBuf,
}

impl Compiler {
    /// Create a new compiler for the given directory
    pub fn new(working_dir: impl AsRef<Path>) -> Self {
        Self {
            working_dir: working_dir.as_ref().to_path_buf(),
        }
    }

    /// Run `cargo check` with JSON message format
    pub fn check(&self) -> CompileResult {
        self.run_cargo(&["check", "--message-format=json"], &[])
    }

    /// Run `cargo check` scoped to specific packages
    pub fn check_packages(&self, packages: &[String]) -> CompileResult {
        let pkg_args: Vec<String> = packages.iter().flat_map(|p| vec!["-p".to_string(), p.clone()]).collect();
        let pkg_refs: Vec<&str> = pkg_args.iter().map(|s| s.as_str()).collect();
        self.run_cargo(&["check", "--message-format=json"], &pkg_refs)
    }

    /// Run `cargo clippy` with JSON message format
    pub fn clippy(&self) -> CompileResult {
        self.run_cargo(&["clippy", "--message-format=json", "--", "-D", "warnings"], &[])
    }

    /// Run `cargo clippy` scoped to specific packages
    pub fn clippy_packages(&self, packages: &[String]) -> CompileResult {
        let pkg_args: Vec<String> = packages.iter().flat_map(|p| vec!["-p".to_string(), p.clone()]).collect();
        let pkg_refs: Vec<&str> = pkg_args.iter().map(|s| s.as_str()).collect();
        let mut all_args = vec!["clippy"];
        all_args.extend_from_slice(&pkg_refs);
        all_args.extend_from_slice(&["--message-format=json", "--", "-D", "warnings"]);
        self.run_cargo(&all_args, &[])
    }

    /// Run `cargo build` with JSON message format
    pub fn build(&self) -> CompileResult {
        self.run_cargo(&["build", "--message-format=json"], &[])
    }

    /// Run cargo command and parse output
    fn run_cargo(&self, args: &[&str], extra_args: &[&str]) -> CompileResult {
        let output = Command::new("cargo")
            .args(args)
            .args(extra_args)
            .current_dir(&self.working_dir)
            .output();

        match output {
            Ok(output) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                let messages = self.parse_json_messages(&stdout);

                CompileResult {
                    success: output.status.success(),
                    exit_code: output.status.code(),
                    messages,
                    raw_stdout: stdout,
                    raw_stderr: stderr,
                }
            }
            Err(e) => CompileResult {
                success: false,
                exit_code: None,
                messages: vec![],
                raw_stdout: String::new(),
                raw_stderr: format!("Failed to run cargo: {}", e),
            },
        }
    }

    /// Parse JSON lines from cargo output
    fn parse_json_messages(&self, output: &str) -> Vec<CargoMessage> {
        output
            .lines()
            .filter_map(|line| serde_json::from_str::<CargoMessage>(line).ok())
            .collect()
    }
}

/// Result of a cargo compilation attempt
#[derive(Debug, Clone)]
pub struct CompileResult {
    /// Whether compilation succeeded
    pub success: bool,
    /// Exit code if available
    pub exit_code: Option<i32>,
    /// Parsed cargo messages
    pub messages: Vec<CargoMessage>,
    /// Raw stdout
    pub raw_stdout: String,
    /// Raw stderr
    pub raw_stderr: String,
}

impl CompileResult {
    /// Get all error messages
    pub fn errors(&self) -> Vec<&CargoMessage> {
        self.messages.iter().filter(|m| m.is_error()).collect()
    }

    /// Get all warning messages
    pub fn warnings(&self) -> Vec<&CargoMessage> {
        self.messages.iter().filter(|m| m.is_warning()).collect()
    }

    /// Get the first error message (for single-error focus)
    pub fn first_error(&self) -> Option<&CargoMessage> {
        self.messages.iter().find(|m| m.is_error())
    }

    /// Count total errors
    pub fn error_count(&self) -> usize {
        self.messages.iter().filter(|m| m.is_error()).count()
    }

    /// Format errors for LLM consumption
    pub fn format_for_llm(&self) -> String {
        let errors: Vec<String> = self
            .errors()
            .iter()
            .filter_map(|m| m.format_diagnostic())
            .collect();

        if errors.is_empty() {
            if self.success {
                "Compilation successful - no errors.".to_string()
            } else {
                format!(
                    "Compilation failed but no structured errors parsed.\nStderr:\n{}",
                    self.raw_stderr
                )
            }
        } else {
            format!(
                "Compilation errors ({} total):\n\n{}",
                errors.len(),
                errors.join("\n\n---\n\n")
            )
        }
    }
}

/// Cargo JSON message format
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "reason")]
#[allow(clippy::large_enum_variant)]
pub enum CargoMessage {
    /// Compiler diagnostic message
    #[serde(rename = "compiler-message")]
    CompilerMessage {
        message: DiagnosticMessage,
        target: Option<Target>,
    },

    /// Compiler artifact produced
    #[serde(rename = "compiler-artifact")]
    CompilerArtifact { target: Target },

    /// Build script output
    #[serde(rename = "build-script-executed")]
    BuildScriptExecuted { package_id: String },

    /// Build finished
    #[serde(rename = "build-finished")]
    BuildFinished { success: bool },

    /// Unknown message type (catch-all)
    #[serde(other)]
    Other,
}

impl CargoMessage {
    /// Check if this is an error message
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            CargoMessage::CompilerMessage { message, .. } if message.level == "error"
        )
    }

    /// Check if this is a warning message
    pub fn is_warning(&self) -> bool {
        matches!(
            self,
            CargoMessage::CompilerMessage { message, .. } if message.level == "warning"
        )
    }

    /// Get the diagnostic message if this is a compiler message
    pub fn as_diagnostic(&self) -> Option<&DiagnosticMessage> {
        match self {
            CargoMessage::CompilerMessage { message, .. } => Some(message),
            _ => None,
        }
    }

    /// Format diagnostic for display
    pub fn format_diagnostic(&self) -> Option<String> {
        let diag = self.as_diagnostic()?;
        Some(diag.format())
    }
}

/// Compiler diagnostic message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticMessage {
    /// Message text
    pub message: String,
    /// Error code (e.g., "E0308")
    pub code: Option<ErrorCode>,
    /// Severity level ("error", "warning", "note")
    pub level: String,
    /// Source spans where the error occurred
    #[serde(default)]
    pub spans: Vec<Span>,
    /// Child diagnostics (notes, helps, suggestions)
    #[serde(default)]
    pub children: Vec<DiagnosticMessage>,
    /// Rendered message (human-readable format)
    pub rendered: Option<String>,
}

impl DiagnosticMessage {
    /// Format for LLM consumption
    pub fn format(&self) -> String {
        // Prefer rendered format if available
        if let Some(rendered) = &self.rendered {
            return rendered.clone();
        }

        // Otherwise build from parts
        let mut result = format!("[{}]", self.level.to_uppercase());

        if let Some(code) = &self.code {
            result.push_str(&format!(" {}", code.code));
        }

        result.push_str(&format!(": {}", self.message));

        // Add primary span location
        if let Some(span) = self.spans.iter().find(|s| s.is_primary) {
            result.push_str(&format!(
                "\n  --> {}:{}:{}",
                span.file_name, span.line_start, span.column_start
            ));

            // Add source text if available
            if let Some(text) = &span.text.first() {
                result.push_str(&format!("\n  |\n  | {}", text.text));
            }
        }

        // Add suggestions from children
        for child in &self.children {
            if child.level == "help" || child.level == "suggestion" {
                result.push_str(&format!("\n  = {}: {}", child.level, child.message));
            }
        }

        result
    }

    /// Get the error code string
    pub fn error_code(&self) -> Option<&str> {
        self.code.as_ref().map(|c| c.code.as_str())
    }

    /// Get the primary span
    pub fn primary_span(&self) -> Option<&Span> {
        self.spans.iter().find(|s| s.is_primary)
    }

    /// Extract suggested replacement if available
    pub fn suggested_replacement(&self) -> Option<&str> {
        for span in &self.spans {
            if let Some(replacement) = &span.suggested_replacement {
                return Some(replacement);
            }
        }
        for child in &self.children {
            if let Some(replacement) = child.suggested_replacement() {
                return Some(replacement);
            }
        }
        None
    }
}

/// Error code with explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorCode {
    /// Error code (e.g., "E0308")
    pub code: String,
    /// Explanation URL or text
    pub explanation: Option<String>,
}

/// Source span indicating where in the code the error occurred
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// File path
    pub file_name: String,
    /// Starting byte offset
    pub byte_start: usize,
    /// Ending byte offset
    pub byte_end: usize,
    /// Starting line number (1-indexed)
    pub line_start: usize,
    /// Ending line number
    pub line_end: usize,
    /// Starting column (1-indexed)
    pub column_start: usize,
    /// Ending column
    pub column_end: usize,
    /// Whether this is the primary span
    #[serde(default)]
    pub is_primary: bool,
    /// Text content at this span
    #[serde(default)]
    pub text: Vec<SpanText>,
    /// Label for this span
    pub label: Option<String>,
    /// Suggested replacement text
    pub suggested_replacement: Option<String>,
    /// Suggestion applicability
    pub suggestion_applicability: Option<String>,
}

/// Text content within a span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanText {
    /// The source text
    pub text: String,
    /// Highlight start within text
    pub highlight_start: usize,
    /// Highlight end within text
    pub highlight_end: usize,
}

/// Cargo target (crate/binary/test)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Target {
    /// Crate name
    pub name: String,
    /// Kind (lib, bin, test, etc.)
    #[serde(default)]
    pub kind: Vec<String>,
    /// Source path
    pub src_path: Option<String>,
}

/// Wrapper for structured cargo output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CargoOutput {
    /// All messages from the compilation
    pub messages: Vec<CargoMessage>,
    /// Whether the build succeeded
    pub success: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_message() {
        let json = r#"{
            "reason": "compiler-message",
            "message": {
                "message": "mismatched types",
                "code": {"code": "E0308", "explanation": null},
                "level": "error",
                "spans": [{
                    "file_name": "src/main.rs",
                    "byte_start": 100,
                    "byte_end": 110,
                    "line_start": 5,
                    "line_end": 5,
                    "column_start": 10,
                    "column_end": 20,
                    "is_primary": true,
                    "text": [{"text": "let x: i32 = \"hello\";", "highlight_start": 10, "highlight_end": 17}],
                    "label": "expected `i32`, found `&str`",
                    "suggested_replacement": null,
                    "suggestion_applicability": null
                }],
                "children": [],
                "rendered": "error[E0308]: mismatched types\n --> src/main.rs:5:10"
            },
            "target": {"name": "test", "kind": ["lib"], "src_path": "src/lib.rs"}
        }"#;

        let msg: CargoMessage = serde_json::from_str(json).unwrap();
        assert!(msg.is_error());
        assert!(!msg.is_warning());

        let diag = msg.as_diagnostic().unwrap();
        assert_eq!(diag.error_code(), Some("E0308"));
    }

    #[test]
    fn test_compile_result_format() {
        let result = CompileResult {
            success: true,
            exit_code: Some(0),
            messages: vec![],
            raw_stdout: String::new(),
            raw_stderr: String::new(),
        };

        assert!(result.format_for_llm().contains("successful"));
    }
}
