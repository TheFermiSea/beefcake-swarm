//! File Walker — .gitignore-respecting file discovery using the `ignore` crate

use ignore::WalkBuilder;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Walks a worktree for source files, respecting .gitignore rules.
pub struct FileWalker {
    root: PathBuf,
}

impl FileWalker {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
        }
    }

    /// Return all .rs files under root, respecting .gitignore.
    pub fn rust_files(&self) -> Vec<PathBuf> {
        let mut files = Vec::new();
        let walker = WalkBuilder::new(&self.root)
            .hidden(true) // skip hidden dirs
            .git_ignore(true)
            .build();

        for entry in walker.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().is_some_and(|ext| ext == "rs") {
                files.push(path.to_path_buf());
            }
        }

        files.sort();
        files
    }

    /// Return files reported by `git status --porcelain` (modified/added).
    pub fn modified_files(&self) -> Vec<String> {
        let output = Command::new("git")
            .args(["status", "--porcelain", "-s"])
            .current_dir(&self.root)
            .output();

        match output {
            Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter_map(|line| {
                    let trimmed = line.trim();
                    if trimmed.len() > 3 {
                        Some(trimmed[3..].to_string())
                    } else {
                        None
                    }
                })
                .collect(),
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_rust_files_finds_rs_files() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("src");
        fs::create_dir_all(&src).unwrap();
        fs::write(src.join("main.rs"), "fn main() {}").unwrap();
        fs::write(src.join("lib.rs"), "pub fn hello() {}").unwrap();
        fs::write(src.join("readme.txt"), "not rust").unwrap();

        let walker = FileWalker::new(dir.path());
        let files = walker.rust_files();

        assert_eq!(files.len(), 2);
        assert!(files.iter().all(|f| f.extension().unwrap() == "rs"));
    }

    #[test]
    fn test_rust_files_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let walker = FileWalker::new(dir.path());
        assert!(walker.rust_files().is_empty());
    }
}
