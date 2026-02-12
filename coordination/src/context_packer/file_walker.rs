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
            if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("rs") {
                files.push(path.to_path_buf());
            }
        }

        files.sort();
        files
    }

    /// Return files reported by `git status` (modified/added/renamed).
    /// Uses `--porcelain=v1 -z` for NUL-delimited output to handle renames and
    /// filenames with spaces correctly.
    pub fn modified_files(&self) -> Vec<String> {
        let output = Command::new("git")
            .args(["status", "--porcelain=v1", "-z"])
            .current_dir(&self.root)
            .output();

        match output {
            Ok(o) if o.status.success() => {
                let mut files = Vec::new();
                let mut iter = o.stdout.split(|b| *b == b'\0').peekable();

                while let Some(entry) = iter.next() {
                    if entry.is_empty() {
                        continue;
                    }

                    let line = String::from_utf8_lossy(entry);
                    // Porcelain v1: two-char status, space, then path
                    if line.len() > 3 {
                        let status = &line[..2];
                        let path = &line[3..];
                        if !path.is_empty() {
                            files.push(path.to_string());
                        }

                        // Renames in -z mode have the new path as a separate NUL entry
                        if status.starts_with('R') || status.starts_with('C') {
                            if let Some(new_path_bytes) = iter.next() {
                                if !new_path_bytes.is_empty() {
                                    let new_path = String::from_utf8_lossy(new_path_bytes);
                                    files.push(new_path.to_string());
                                }
                            }
                        }
                    }
                }

                files
            }
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
        assert!(files
            .iter()
            .all(|f| f.extension().and_then(|e| e.to_str()) == Some("rs")));
    }

    #[test]
    fn test_rust_files_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let walker = FileWalker::new(dir.path());
        assert!(walker.rust_files().is_empty());
    }
}
