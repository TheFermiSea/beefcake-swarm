//! Source File Provider — caching file reader for context assembly.
//!
//! Deduplicates file I/O between `WorkPacketGenerator` and `ContextPacker`,
//! which both independently read source files from disk.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Caching source file reader.
///
/// Reads files from disk on first access and caches the content for
/// subsequent reads. Both `WorkPacketGenerator` and `ContextPacker` can
/// share a single provider to avoid redundant I/O.
pub struct SourceFileProvider {
    root: PathBuf,
    cache: HashMap<String, Option<String>>,
}

impl SourceFileProvider {
    /// Create a new provider rooted at the given directory.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            cache: HashMap::new(),
        }
    }

    /// Read a file's content, caching the result.
    ///
    /// The `relative_path` is joined to the root directory.
    /// Returns `None` if the file cannot be read.
    pub fn read(&mut self, relative_path: &str) -> Option<&str> {
        if !self.cache.contains_key(relative_path) {
            let full_path = self.root.join(relative_path);
            let content = std::fs::read_to_string(&full_path).ok();
            self.cache.insert(relative_path.to_string(), content);
        }
        self.cache.get(relative_path).and_then(|opt| opt.as_deref())
    }

    /// Read a file and split into lines (cached).
    pub fn read_lines(&mut self, relative_path: &str) -> Option<Vec<&str>> {
        self.read(relative_path).map(|c| c.lines().collect())
    }

    /// Read a file and return a numbered context string (first N lines).
    ///
    /// Each line is formatted as `{line_num:4} | {content}`.
    pub fn read_numbered_header(
        &mut self,
        relative_path: &str,
        max_lines: usize,
    ) -> Option<(String, usize)> {
        let content = self.read(relative_path)?;
        let lines: Vec<&str> = content.lines().collect();
        let end = lines.len().min(max_lines);

        let numbered: String = lines[..end]
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:4} | {}", i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        Some((numbered, end))
    }

    /// Read a file and return a full numbered content string.
    pub fn read_numbered_full(&mut self, relative_path: &str) -> Option<(String, usize)> {
        let content = self.read(relative_path)?;
        let lines: Vec<&str> = content.lines().collect();
        let line_count = lines.len();

        let numbered: String = lines
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{:4} | {}", i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        Some((numbered, line_count))
    }

    /// Estimate the character cost of including a file's numbered content.
    pub fn estimate_chars(&mut self, relative_path: &str) -> usize {
        match self.read(relative_path) {
            Some(content) => {
                // Content + line number formatting overhead (~8 chars per line)
                content.len() + content.lines().count() * 8 + relative_path.len() + 100
            }
            None => 0,
        }
    }

    /// Get the root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Number of cached files.
    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Clear the cache.
    pub fn clear_cache(&mut self) {
        self.cache.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_read_caches_content() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.rs"), "fn main() {}\n").unwrap();

        let mut provider = SourceFileProvider::new(dir.path());
        assert_eq!(provider.cache_size(), 0);

        // First read loads from disk
        assert!(provider.read("test.rs").unwrap().contains("fn main"));
        assert_eq!(provider.cache_size(), 1);

        // Second read uses cache (still returns same content)
        assert!(provider.read("test.rs").unwrap().contains("fn main"));
        assert_eq!(provider.cache_size(), 1);
    }

    #[test]
    fn test_read_missing_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut provider = SourceFileProvider::new(dir.path());
        assert!(provider.read("nonexistent.rs").is_none());
        // Should cache the None result too
        assert_eq!(provider.cache_size(), 1);
    }

    #[test]
    fn test_read_lines() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.rs"), "line1\nline2\nline3\n").unwrap();

        let mut provider = SourceFileProvider::new(dir.path());
        let lines = provider.read_lines("test.rs").unwrap();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0], "line1");
    }

    #[test]
    fn test_read_numbered_header() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(
            dir.path().join("test.rs"),
            "line1\nline2\nline3\nline4\nline5\n",
        )
        .unwrap();

        let mut provider = SourceFileProvider::new(dir.path());
        let (numbered, end) = provider.read_numbered_header("test.rs", 3).unwrap();
        assert_eq!(end, 3);
        assert!(numbered.contains("   1 | line1"));
        assert!(numbered.contains("   3 | line3"));
        assert!(!numbered.contains("line4"));
    }

    #[test]
    fn test_read_numbered_full() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.rs"), "a\nb\nc\n").unwrap();

        let mut provider = SourceFileProvider::new(dir.path());
        let (numbered, count) = provider.read_numbered_full("test.rs").unwrap();
        assert_eq!(count, 3);
        assert!(numbered.contains("   1 | a"));
        assert!(numbered.contains("   3 | c"));
    }

    #[test]
    fn test_estimate_chars() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.rs"), "hello world\n").unwrap();

        let mut provider = SourceFileProvider::new(dir.path());
        let estimate = provider.estimate_chars("test.rs");
        assert!(estimate > 0);
        assert!(estimate > "hello world\n".len());
    }

    #[test]
    fn test_clear_cache() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("test.rs"), "content\n").unwrap();

        let mut provider = SourceFileProvider::new(dir.path());
        provider.read("test.rs");
        assert_eq!(provider.cache_size(), 1);

        provider.clear_cache();
        assert_eq!(provider.cache_size(), 0);
    }
}
