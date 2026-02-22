//! Diff Risk Profile — Adaptive gate selection based on code change analysis.
//!
//! Analyzes the git diff to build a risk profile that determines which
//! additional verification gates should run. This avoids running expensive
//! gates (deny, doc, nextest) on every change — only when the diff warrants it.

use serde::{Deserialize, Serialize};
use std::path::Path;

/// Risk signals extracted from the git diff.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DiffRiskProfile {
    /// Diff touches unsafe blocks (new or modified)
    pub has_unsafe: bool,
    /// Diff modifies Cargo.toml (dependency changes)
    pub has_cargo_toml_change: bool,
    /// Diff modifies public API (pub fn, pub struct, pub trait, pub enum)
    pub has_public_api_change: bool,
    /// Diff adds or modifies doc comments (///, //!, #[doc])
    pub has_doc_change: bool,
    /// Number of files changed
    pub files_changed: usize,
    /// Number of lines added
    pub lines_added: usize,
    /// Number of lines removed
    pub lines_removed: usize,
}

impl DiffRiskProfile {
    /// Build a risk profile from the git diff in the given working directory.
    pub fn from_working_dir(working_dir: &Path) -> Self {
        let diff_text = get_diff_text(working_dir);
        Self::from_diff_text(&diff_text)
    }

    /// Build a risk profile from raw diff text (testable without git).
    pub fn from_diff_text(diff_text: &str) -> Self {
        let mut profile = Self::default();
        let mut current_file = String::new();
        let mut seen_files = std::collections::HashSet::new();

        for line in diff_text.lines() {
            if let Some(rest) = line.strip_prefix("+++ b/") {
                current_file = rest.to_string();
                seen_files.insert(current_file.clone());
            } else if let Some(added) = line.strip_prefix('+') {
                // Skip diff header lines
                if added.starts_with("++") {
                    continue;
                }

                profile.lines_added += 1;

                // Check for unsafe
                if added.contains("unsafe {") || added.contains("unsafe{") {
                    profile.has_unsafe = true;
                }

                // Check for Cargo.toml changes
                if current_file.ends_with("Cargo.toml") {
                    profile.has_cargo_toml_change = true;
                }

                // Check for public API changes
                if added.contains("pub fn ")
                    || added.contains("pub struct ")
                    || added.contains("pub trait ")
                    || added.contains("pub enum ")
                    || added.contains("pub type ")
                    || added.contains("pub const ")
                    || added.contains("pub static ")
                    || added.contains("pub async fn ")
                {
                    profile.has_public_api_change = true;
                }

                // Check for doc comment changes
                let trimmed = added.trim_start();
                if trimmed.starts_with("///")
                    || trimmed.starts_with("//!")
                    || trimmed.starts_with("#[doc")
                {
                    profile.has_doc_change = true;
                }
            } else if line.starts_with('-') && !line.starts_with("---") {
                profile.lines_removed += 1;
            }
        }

        profile.files_changed = seen_files.len();
        profile
    }

    /// Whether `cargo deny check` should run (Cargo.toml changed).
    pub fn should_run_deny(&self) -> bool {
        self.has_cargo_toml_change
    }

    /// Whether `cargo doc` gate should run (doc comments or public API changed).
    pub fn should_run_doc(&self) -> bool {
        self.has_doc_change || self.has_public_api_change
    }

    /// Whether nextest should be preferred over cargo test (large changesets).
    pub fn should_prefer_nextest(&self) -> bool {
        self.files_changed >= 3 || self.lines_added >= 100
    }
}

/// Get the git diff text from a working directory.
fn get_diff_text(working_dir: &Path) -> String {
    // Try staged + unstaged diff first
    let output = std::process::Command::new("git")
        .args(["diff", "HEAD"])
        .current_dir(working_dir)
        .output();

    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).to_string(),
        _ => {
            // Fallback: unstaged diff only
            let fallback = std::process::Command::new("git")
                .args(["diff"])
                .current_dir(working_dir)
                .output();
            match fallback {
                Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
                Err(_) => String::new(),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CARGO_TOML_DIFF: &str = r#"diff --git a/Cargo.toml b/Cargo.toml
--- a/Cargo.toml
+++ b/Cargo.toml
@@ -10,6 +10,7 @@
 [dependencies]
 serde = "1"
+reqwest = "0.12"
"#;

    const UNSAFE_DIFF: &str = r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,7 @@
 fn helper() {
+    unsafe {
+        std::ptr::null::<u8>().read();
+    }
 }
"#;

    const PUBLIC_API_DIFF: &str = r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,8 @@
+/// A new public function.
+pub fn new_feature(x: u32) -> u32 {
+    x + 1
+}
+
 fn private_helper() {
 }
"#;

    const LARGE_DIFF: &str = r#"diff --git a/src/a.rs b/src/a.rs
--- a/src/a.rs
+++ b/src/a.rs
@@ -1,3 +1,50 @@
+fn a1() {}
+fn a2() {}
+fn a3() {}
+fn a4() {}
+fn a5() {}
+fn a6() {}
+fn a7() {}
+fn a8() {}
+fn a9() {}
+fn a10() {}
diff --git a/src/b.rs b/src/b.rs
--- a/src/b.rs
+++ b/src/b.rs
@@ -1,3 +1,50 @@
+fn b1() {}
+fn b2() {}
+fn b3() {}
+fn b4() {}
+fn b5() {}
+fn b6() {}
+fn b7() {}
+fn b8() {}
+fn b9() {}
+fn b10() {}
diff --git a/src/c.rs b/src/c.rs
--- a/src/c.rs
+++ b/src/c.rs
@@ -1,3 +1,50 @@
+fn c1() {}
+fn c2() {}
+fn c3() {}
+fn c4() {}
+fn c5() {}
+fn c6() {}
+fn c7() {}
+fn c8() {}
+fn c9() {}
+fn c10() {}
"#;

    const CLEAN_DIFF: &str = r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,3 +1,5 @@
 fn helper() {
+    let x = 42;
+    println!("{x}");
 }
"#;

    #[test]
    fn test_cargo_toml_change_triggers_deny() {
        let profile = DiffRiskProfile::from_diff_text(CARGO_TOML_DIFF);
        assert!(profile.has_cargo_toml_change);
        assert!(profile.should_run_deny());
    }

    #[test]
    fn test_unsafe_detected() {
        let profile = DiffRiskProfile::from_diff_text(UNSAFE_DIFF);
        assert!(profile.has_unsafe);
    }

    #[test]
    fn test_public_api_change_triggers_doc() {
        let profile = DiffRiskProfile::from_diff_text(PUBLIC_API_DIFF);
        assert!(profile.has_public_api_change);
        assert!(profile.has_doc_change);
        assert!(profile.should_run_doc());
    }

    #[test]
    fn test_large_diff_prefers_nextest() {
        let profile = DiffRiskProfile::from_diff_text(LARGE_DIFF);
        assert_eq!(profile.files_changed, 3);
        assert!(profile.should_prefer_nextest());
    }

    #[test]
    fn test_clean_diff_no_extra_gates() {
        let profile = DiffRiskProfile::from_diff_text(CLEAN_DIFF);
        assert!(!profile.has_unsafe);
        assert!(!profile.has_cargo_toml_change);
        assert!(!profile.has_public_api_change);
        assert!(!profile.has_doc_change);
        assert!(!profile.should_run_deny());
        assert!(!profile.should_run_doc());
        assert!(!profile.should_prefer_nextest());
        assert_eq!(profile.files_changed, 1);
        assert_eq!(profile.lines_added, 2);
    }

    #[test]
    fn test_lines_removed_counted() {
        let diff = r#"diff --git a/src/lib.rs b/src/lib.rs
--- a/src/lib.rs
+++ b/src/lib.rs
@@ -1,5 +1,3 @@
 fn helper() {
-    let old = 1;
-    let stale = 2;
+    let new = 3;
 }
"#;
        let profile = DiffRiskProfile::from_diff_text(diff);
        assert_eq!(profile.lines_added, 1);
        assert_eq!(profile.lines_removed, 2);
    }

    #[test]
    fn test_empty_diff() {
        let profile = DiffRiskProfile::from_diff_text("");
        assert_eq!(profile.files_changed, 0);
        assert_eq!(profile.lines_added, 0);
        assert!(!profile.should_run_deny());
        assert!(!profile.should_run_doc());
        assert!(!profile.should_prefer_nextest());
    }

    #[test]
    fn test_profile_serialization() {
        let profile = DiffRiskProfile {
            has_unsafe: true,
            has_cargo_toml_change: true,
            has_public_api_change: false,
            has_doc_change: false,
            files_changed: 5,
            lines_added: 200,
            lines_removed: 50,
        };
        let json = serde_json::to_string(&profile).unwrap();
        let parsed: DiffRiskProfile = serde_json::from_str(&json).unwrap();
        assert!(parsed.has_unsafe);
        assert!(parsed.has_cargo_toml_change);
        assert_eq!(parsed.files_changed, 5);
    }
}
