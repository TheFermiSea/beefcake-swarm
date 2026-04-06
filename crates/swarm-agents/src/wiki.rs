//! Incremental wiki compilation — Karpathy LLM Wiki pattern.
//!
//! After each successful resolution, update the structured wiki
//! pages in `docs/wiki/` with new information. Knowledge compounds
//! over time instead of being dumped as flat NotebookLM sources.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::Utc;

/// Path to the wiki directory relative to repo root.
const WIKI_DIR: &str = "docs/wiki";

/// Append an entry to the wiki activity log.
pub fn append_log(repo_root: &Path, entry: &str) -> Result<()> {
    let log_path = repo_root.join(WIKI_DIR).join("log.md");
    let existing = fs::read_to_string(&log_path)
        .with_context(|| format!("reading wiki log at {}", log_path.display()))?;
    let timestamp = Utc::now().format("%Y-%m-%d %H:%M");
    let line = format!("\n## [{timestamp}] {entry}\n");
    fs::write(&log_path, format!("{existing}{line}"))
        .with_context(|| format!("writing wiki log at {}", log_path.display()))?;
    Ok(())
}

/// Record a successful resolution in the wiki.
///
/// Updates the activity log and, if an error category is provided,
/// appends to the error patterns page.
pub fn record_resolution(
    repo_root: &Path,
    issue_id: &str,
    title: &str,
    error_category: Option<&str>,
    _model_used: Option<&str>,
    _iterations: u32,
) -> Result<()> {
    // Append to activity log
    append_log(repo_root, &format!("resolution | {issue_id} — {title}"))?;

    // If we know the error category, append to error-patterns.md
    if let Some(category) = error_category {
        append_error_pattern(repo_root, category, issue_id, title)?;
    }

    Ok(())
}

fn append_error_pattern(
    repo_root: &Path,
    category: &str,
    issue_id: &str,
    title: &str,
) -> Result<()> {
    let path = repo_root.join(WIKI_DIR).join("error-patterns.md");
    let mut content = fs::read_to_string(&path)
        .with_context(|| format!("reading error patterns at {}", path.display()))?;
    let entry = format!("- **{category}** ({issue_id}): {title}\n");

    // Append under the "Recent Resolutions" section if it exists
    if let Some(pos) = content.find("## Recent Resolutions") {
        let insert_pos = content[pos..]
            .find('\n')
            .map(|p| pos + p + 1)
            .unwrap_or(content.len());
        content.insert_str(insert_pos, &entry);
    } else {
        content.push_str(&format!("\n## Recent Resolutions\n\n{entry}"));
    }

    fs::write(&path, content)
        .with_context(|| format!("writing error patterns at {}", path.display()))?;
    Ok(())
}

/// Check whether the wiki directory exists at the given repo root.
pub fn wiki_exists(repo_root: &Path) -> bool {
    repo_root.join(WIKI_DIR).join("index.md").exists()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Create a minimal wiki directory for testing.
    fn setup_wiki(tmp: &TempDir) -> std::path::PathBuf {
        let root = tmp.path().to_path_buf();
        let wiki = root.join(WIKI_DIR);
        fs::create_dir_all(&wiki).unwrap();
        fs::write(
            wiki.join("log.md"),
            "# Activity Log\n\nAppend-only record of wiki updates.\n",
        )
        .unwrap();
        fs::write(
            wiki.join("error-patterns.md"),
            "# Error Patterns\n\nCommon compiler errors.\n",
        )
        .unwrap();
        fs::write(wiki.join("index.md"), "# Wiki\n").unwrap();
        root
    }

    #[test]
    fn test_append_log() {
        let tmp = TempDir::new().unwrap();
        let root = setup_wiki(&tmp);

        append_log(&root, "test entry").unwrap();

        let content = fs::read_to_string(root.join(WIKI_DIR).join("log.md")).unwrap();
        assert!(content.contains("test entry"));
        assert!(content.contains("## ["));
    }

    #[test]
    fn test_append_log_preserves_existing() {
        let tmp = TempDir::new().unwrap();
        let root = setup_wiki(&tmp);

        append_log(&root, "first").unwrap();
        append_log(&root, "second").unwrap();

        let content = fs::read_to_string(root.join(WIKI_DIR).join("log.md")).unwrap();
        assert!(content.contains("first"));
        assert!(content.contains("second"));
        assert!(content.contains("# Activity Log"));
    }

    #[test]
    fn test_record_resolution_without_error_category() {
        let tmp = TempDir::new().unwrap();
        let root = setup_wiki(&tmp);

        record_resolution(&root, "abc-123", "fix the thing", None, None, 1).unwrap();

        let log = fs::read_to_string(root.join(WIKI_DIR).join("log.md")).unwrap();
        assert!(log.contains("abc-123"));
        assert!(log.contains("fix the thing"));

        // error-patterns.md should not be modified
        let errors = fs::read_to_string(root.join(WIKI_DIR).join("error-patterns.md")).unwrap();
        assert!(!errors.contains("abc-123"));
    }

    #[test]
    fn test_record_resolution_with_error_category() {
        let tmp = TempDir::new().unwrap();
        let root = setup_wiki(&tmp);

        record_resolution(
            &root,
            "def-456",
            "borrow checker fix",
            Some("BorrowChecker"),
            Some("Qwen3.5-27B"),
            3,
        )
        .unwrap();

        let errors = fs::read_to_string(root.join(WIKI_DIR).join("error-patterns.md")).unwrap();
        assert!(errors.contains("BorrowChecker"));
        assert!(errors.contains("def-456"));
        assert!(errors.contains("## Recent Resolutions"));
    }

    #[test]
    fn test_error_pattern_appends_under_existing_section() {
        let tmp = TempDir::new().unwrap();
        let root = setup_wiki(&tmp);

        // First resolution creates the section
        record_resolution(&root, "id-1", "first fix", Some("TypeMismatch"), None, 1).unwrap();

        // Second resolution appends under existing section
        record_resolution(&root, "id-2", "second fix", Some("Lifetime"), None, 2).unwrap();

        let errors = fs::read_to_string(root.join(WIKI_DIR).join("error-patterns.md")).unwrap();
        assert!(errors.contains("TypeMismatch"));
        assert!(errors.contains("Lifetime"));
        // Should have exactly one "Recent Resolutions" header
        assert_eq!(errors.matches("## Recent Resolutions").count(), 1);
    }

    #[test]
    fn test_wiki_exists() {
        let tmp = TempDir::new().unwrap();
        let root = setup_wiki(&tmp);
        assert!(wiki_exists(&root));

        let empty_tmp = TempDir::new().unwrap();
        assert!(!wiki_exists(empty_tmp.path()));
    }
}
