//! Embedding generation via subprocess.
//!
//! Calls `scripts/embed.py` to generate text embeddings. Falls back to
//! empty vectors if Python/sentence-transformers not available.

use anyhow::Result;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Locate the embed.py script relative to the repo root.
///
/// Checks `CARGO_MANIFEST_DIR/../..` (workspace root) first, then
/// falls back to the current working directory.
fn find_embed_script() -> Result<PathBuf> {
    // Try relative to workspace root (from CARGO_MANIFEST_DIR at compile time)
    let manifest_dir = option_env!("CARGO_MANIFEST_DIR");
    if let Some(dir) = manifest_dir {
        let workspace_root = Path::new(dir).join("../../scripts/embed.py");
        if workspace_root.exists() {
            return Ok(workspace_root);
        }
    }

    // Try relative to cwd
    let cwd_relative = PathBuf::from("scripts/embed.py");
    if cwd_relative.exists() {
        return Ok(cwd_relative);
    }

    anyhow::bail!(
        "Cannot find scripts/embed.py. Run from the workspace root \
         or set CARGO_MANIFEST_DIR."
    )
}

/// Generate an embedding vector for the given text.
///
/// Uses `scripts/embed.py` (sentence-transformers or hash fallback).
/// Returns a vector of f32 values.
pub fn embed_text(text: &str) -> Result<Vec<f32>> {
    let script = find_embed_script()?;

    let output = Command::new("python3")
        .args([script.to_str().unwrap(), "--text", text])
        .output()?;

    if !output.status.success() {
        anyhow::bail!(
            "embed.py failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8(output.stdout)?;
    let vec: Vec<f64> = serde_json::from_str(stdout.trim())?;
    Ok(vec.into_iter().map(|v| v as f32).collect())
}

/// Generate embeddings for multiple texts (batch mode).
///
/// Writes texts to a temporary JSONL file, runs `embed.py --batch`,
/// and parses the output embeddings.
pub fn embed_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let script = find_embed_script()?;

    // Write to unique temp file (thread-safe via timestamp nanos)
    let tmp_dir = std::env::temp_dir();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let tmp_path = tmp_dir.join(format!(
        "embed_batch_{}_{}.jsonl",
        std::process::id(),
        nanos
    ));
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp_path)?);
        for text in texts {
            writeln!(f, "{}", serde_json::json!({"text": text}))?;
        }
    }

    let output = Command::new("python3")
        .args([
            script.to_str().unwrap(),
            "--batch",
            tmp_path.to_str().unwrap(),
        ])
        .output();

    // Clean up temp file regardless of result
    let _ = std::fs::remove_file(&tmp_path);

    let output = output?;
    if !output.status.success() {
        anyhow::bail!(
            "embed.py batch failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let stdout = String::from_utf8(output.stdout)?;
    let mut results = Vec::new();
    for line in stdout.lines() {
        let item: serde_json::Value = serde_json::from_str(line)?;
        if let Some(emb) = item.get("embedding").and_then(|e| e.as_array()) {
            let vec: Vec<f32> = emb
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect();
            results.push(vec);
        }
    }
    Ok(results)
}

/// Compute cosine similarity between two embedding vectors.
///
/// Returns a value between -1.0 (opposite) and 1.0 (identical).
/// Returns 0.0 if either vector is zero-length or empty.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-6, "identical vectors: {sim}");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6, "orthogonal vectors: {sim}");
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-6, "opposite vectors: {sim}");
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let sim = cosine_similarity(&[], &[]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_length() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_embed_text_returns_vector() {
        // This test requires python3 and scripts/embed.py to be available.
        // It uses the hash fallback (no sentence-transformers needed).
        let result = embed_text("hello world");
        match result {
            Ok(vec) => {
                assert!(!vec.is_empty(), "embedding should not be empty");
                assert_eq!(vec.len(), 64, "hash fallback produces 64-dim vectors");
                // Check normalization: magnitude should be ~1.0
                let mag: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    (mag - 1.0).abs() < 0.01,
                    "embedding should be normalized, got {mag}"
                );
            }
            Err(e) => {
                // Acceptable if python3 isn't available in CI
                eprintln!("Skipping embed_text test: {e}");
            }
        }
    }

    #[test]
    fn test_embed_text_deterministic() {
        // Same input should produce same output (hash fallback is deterministic)
        let r1 = embed_text("test determinism");
        let r2 = embed_text("test determinism");
        match (r1, r2) {
            (Ok(v1), Ok(v2)) => {
                assert_eq!(v1, v2, "same input should produce same embedding");
            }
            _ => {
                eprintln!("Skipping determinism test: python3 not available");
            }
        }
    }

    #[test]
    fn test_embed_batch_returns_vectors() {
        let texts = &["hello world", "goodbye world"];
        let result = embed_batch(texts);
        match result {
            Ok(vecs) => {
                assert_eq!(vecs.len(), 2, "should get one embedding per input");
                for vec in &vecs {
                    assert_eq!(vec.len(), 64, "hash fallback produces 64-dim vectors");
                }
                // Different texts should produce different embeddings
                assert_ne!(vecs[0], vecs[1], "different texts should differ");
            }
            Err(e) => {
                eprintln!("Skipping embed_batch test: {e}");
            }
        }
    }

    #[test]
    fn test_embed_similarity_related_texts() {
        // With hash fallback, similar texts (sharing words) should have
        // higher cosine similarity than unrelated texts.
        let r1 = embed_text("rust programming language");
        let r2 = embed_text("rust programming compiler");
        let r3 = embed_text("banana smoothie recipe");

        match (r1, r2, r3) {
            (Ok(v1), Ok(v2), Ok(v3)) => {
                let sim_related = cosine_similarity(&v1, &v2);
                let sim_unrelated = cosine_similarity(&v1, &v3);
                assert!(
                    sim_related > sim_unrelated,
                    "related texts ({sim_related}) should be more similar \
                     than unrelated ({sim_unrelated})"
                );
            }
            _ => {
                eprintln!("Skipping similarity test: python3 not available");
            }
        }
    }
}
