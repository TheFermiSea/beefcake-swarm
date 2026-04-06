//! Persistent wiki for swarm-produced analyses and retrospectives.
//!
//! When the swarm produces a valuable synthesis (retrospective recommendations,
//! TZ performance comparisons, non-obvious resolution patterns), this module
//! files it as a markdown wiki page under `docs/wiki/` and updates the
//! auto-generated index.
//!
//! Karpathy: "good answers file back into the wiki."

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use tracing::info;

/// File a synthesis/analysis as a wiki page.
///
/// Called when the swarm produces a valuable analysis that should persist
/// beyond the current session.
pub fn capture_analysis(
    repo_root: &Path,
    slug: &str,
    title: &str,
    category: &str,
    content: &str,
    source_issue: Option<&str>,
) -> Result<()> {
    let wiki_dir = repo_root.join("docs/wiki");
    if !wiki_dir.exists() {
        anyhow::bail!("wiki directory not found at {}", wiki_dir.display());
    }

    let page_path = wiki_dir.join(format!("{slug}.md"));
    let timestamp = chrono::Utc::now().format("%Y-%m-%d");
    let source = source_issue.unwrap_or("manual");
    let frontmatter = format!(
        "# {title}\n\n\
         > Category: {category} | Date: {timestamp} | Source: {source}\n\n",
    );
    fs::write(&page_path, format!("{frontmatter}{content}"))
        .with_context(|| format!("writing wiki page {}", page_path.display()))?;

    update_index(repo_root, slug, title, category)?;

    append_log(repo_root, &format!("{category} | {title}"))?;

    info!(slug, title, category, source, "Filed wiki analysis page");

    Ok(())
}

/// Add a row to `docs/wiki/index.md` for the given slug, unless it already exists.
fn update_index(repo_root: &Path, slug: &str, title: &str, category: &str) -> Result<()> {
    let index_path = repo_root.join("docs/wiki/index.md");
    let existing = fs::read_to_string(&index_path)
        .with_context(|| format!("reading wiki index at {}", index_path.display()))?;

    // Skip if slug already has a row
    if existing.contains(&format!("| {slug} |")) {
        return Ok(());
    }

    let date = chrono::Utc::now().format("%Y-%m-%d");
    let new_row = format!("| {slug} | [{title}]({slug}.md) | {category} | {date} |\n");

    // Append after the last line (which should be the table header separator or a previous row)
    let updated = format!("{}{new_row}", existing);
    fs::write(&index_path, updated)
        .with_context(|| format!("writing wiki index at {}", index_path.display()))?;

    Ok(())
}

/// Append an entry to the wiki activity log (`docs/wiki/activity.log`).
pub fn append_log(repo_root: &Path, entry: &str) -> Result<()> {
    use std::io::Write;

    let log_path = repo_root.join("docs/wiki/activity.log");
    let timestamp = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ");
    let line = format!("[{timestamp}] {entry}\n");

    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .with_context(|| format!("opening wiki activity log at {}", log_path.display()))?;

    file.write_all(line.as_bytes())
        .with_context(|| "writing to wiki activity log")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    /// Set up a temp directory with the expected `docs/wiki/index.md` structure.
    fn setup_wiki_dir() -> TempDir {
        let tmp = TempDir::new().unwrap();
        let wiki_dir = tmp.path().join("docs/wiki");
        fs::create_dir_all(&wiki_dir).unwrap();
        fs::write(
            wiki_dir.join("index.md"),
            "# Wiki Index\n\n\
             | Slug | Title | Category | Date |\n\
             |------|-------|----------|------|\n",
        )
        .unwrap();
        tmp
    }

    #[test]
    fn test_capture_analysis_creates_page_and_updates_index() {
        let tmp = setup_wiki_dir();
        let repo = tmp.path();

        capture_analysis(
            repo,
            "retro-beads-001",
            "Retrospective: beads-001",
            "retrospective",
            "Some useful recommendations here.",
            Some("beads-001"),
        )
        .unwrap();

        // Page should exist with content
        let page = fs::read_to_string(repo.join("docs/wiki/retro-beads-001.md")).unwrap();
        assert!(page.contains("# Retrospective: beads-001"));
        assert!(page.contains("Category: retrospective"));
        assert!(page.contains("Source: beads-001"));
        assert!(page.contains("Some useful recommendations here."));

        // Index should have a new row
        let index = fs::read_to_string(repo.join("docs/wiki/index.md")).unwrap();
        assert!(index.contains("| retro-beads-001 |"));
        assert!(index.contains("[Retrospective: beads-001](retro-beads-001.md)"));

        // Activity log should have an entry
        let log = fs::read_to_string(repo.join("docs/wiki/activity.log")).unwrap();
        assert!(log.contains("retrospective | Retrospective: beads-001"));
    }

    #[test]
    fn test_capture_analysis_without_source_issue() {
        let tmp = setup_wiki_dir();
        let repo = tmp.path();

        capture_analysis(
            repo,
            "comparison-tz-2026-04",
            "TZ Performance Comparison",
            "comparison",
            "Model A beats Model B on code-fixing.",
            None,
        )
        .unwrap();

        let page = fs::read_to_string(repo.join("docs/wiki/comparison-tz-2026-04.md")).unwrap();
        assert!(page.contains("Source: manual"));
    }

    #[test]
    fn test_capture_analysis_fails_without_wiki_dir() {
        let tmp = TempDir::new().unwrap();
        let result = capture_analysis(tmp.path(), "slug", "Title", "analysis", "content", None);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("wiki directory not found"));
    }

    #[test]
    fn test_update_index_deduplicates() {
        let tmp = setup_wiki_dir();
        let repo = tmp.path();

        // Capture twice with the same slug
        capture_analysis(repo, "dup-slug", "First", "analysis", "c1", None).unwrap();
        capture_analysis(repo, "dup-slug", "Second", "analysis", "c2", None).unwrap();

        let index = fs::read_to_string(repo.join("docs/wiki/index.md")).unwrap();
        // Should only have one row for dup-slug (the first title)
        let count = index.matches("| dup-slug |").count();
        assert_eq!(count, 1, "expected exactly one row for dup-slug");
    }

    #[test]
    fn test_append_log_creates_file() {
        let tmp = setup_wiki_dir();
        let repo = tmp.path();

        append_log(repo, "test entry 1").unwrap();
        append_log(repo, "test entry 2").unwrap();

        let log = fs::read_to_string(repo.join("docs/wiki/activity.log")).unwrap();
        assert!(log.contains("test entry 1"));
        assert!(log.contains("test entry 2"));
        // Should have two lines
        assert_eq!(log.lines().count(), 2);
    }

    #[test]
    fn test_page_overwrites_on_recapture() {
        let tmp = setup_wiki_dir();
        let repo = tmp.path();

        capture_analysis(repo, "evolving", "V1", "analysis", "old content", None).unwrap();
        capture_analysis(repo, "evolving", "V2", "analysis", "new content", None).unwrap();

        let page = fs::read_to_string(repo.join("docs/wiki/evolving.md")).unwrap();
        assert!(page.contains("new content"));
        assert!(!page.contains("old content"));
    }
}
