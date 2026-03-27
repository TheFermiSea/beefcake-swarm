//! Semantic relevance scoring for context packing.
//!
//! Calls a local embedding service (nomic-embed-text-v1.5 on port 8082)
//! to compute cosine similarity between the task objective and each file
//! context. File contexts are re-ranked by semantic relevance before
//! `trim_to_budget()` drops the least relevant ones.
//!
//! Degrades gracefully: if the embedding service is unavailable, the
//! original priority-based ordering is preserved.

use crate::work_packet::types::FileContext;
use tracing::debug;

/// Default embedding service URL. Each compute node runs this on port 8082.
/// Override with `SWARM_EMBEDDING_URL` env var.
pub const EMBEDDING_SERVICE_URL: &str = "http://localhost:8082/v1/embeddings";

/// Timeout for embedding service calls (ms).
const EMBEDDING_TIMEOUT_MS: u64 = 2000;

/// Minimum priority to rerank. Priority 0 (compiler error) and 1 (modified file)
/// are too important to demote — only priority 2+ contexts get reranked.
const MIN_RERANK_PRIORITY: u8 = 2;

/// Model name sent to the embedding endpoint.
const EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";

/// Score file contexts by semantic similarity to the objective and reorder them.
///
/// Priority 0-1 contexts (error signals, modified files) are never demoted.
/// Priority 2+ contexts (structural/reference) are reranked by cosine similarity
/// to the objective, with the most relevant ones getting lower (better) priority
/// numbers so they survive `trim_to_budget()`.
pub(crate) fn score_and_rerank(objective: &str, file_contexts: &mut [FileContext]) {
    if file_contexts.is_empty() || objective.is_empty() {
        return;
    }

    // Collect the indices and text of rerank-eligible contexts
    let eligible: Vec<(usize, String)> = file_contexts
        .iter()
        .enumerate()
        .filter(|(_, ctx)| ctx.priority >= MIN_RERANK_PRIORITY)
        .map(|(i, ctx)| {
            // Use file path + first 200 chars of content as the embedding input
            let text = format!(
                "{}: {}",
                ctx.file,
                &ctx.content[..ctx.content.len().min(200)]
            );
            (i, text)
        })
        .collect();

    if eligible.is_empty() {
        return;
    }

    // Build the embedding request: objective + all eligible context texts
    let mut texts: Vec<String> = Vec::with_capacity(eligible.len() + 1);
    texts.push(objective.to_string());
    for (_, text) in &eligible {
        texts.push(text.clone());
    }

    let scores = match compute_similarities(&texts) {
        Ok(s) => s,
        Err(e) => {
            debug!(error = %e, "Semantic scoring unavailable — keeping priority-based order");
            return;
        }
    };

    // scores[i] is the cosine similarity between texts[0] (objective) and texts[i+1]
    // Map scores back to file_context indices and sort by descending similarity
    let mut scored: Vec<(usize, f32)> = eligible
        .iter()
        .zip(scores.iter())
        .map(|((idx, _), score)| (*idx, *score))
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Assign new priorities: most relevant gets priority MIN_RERANK_PRIORITY,
    // least relevant gets MIN_RERANK_PRIORITY + count. This preserves the
    // invariant that priority 0-1 contexts are always more important.
    for (rank, (idx, score)) in scored.iter().enumerate() {
        let new_priority = MIN_RERANK_PRIORITY + rank as u8;
        let old_priority = file_contexts[*idx].priority;
        if new_priority != old_priority {
            debug!(
                file = %file_contexts[*idx].file,
                score = format!("{:.3}", score),
                old_priority,
                new_priority,
                "Semantic rerank"
            );
        }
        file_contexts[*idx].priority = new_priority;
    }
}

/// Call the embedding service and compute cosine similarities.
///
/// `texts[0]` is the query (objective). `texts[1..]` are the documents.
/// Returns a Vec of cosine similarities between texts[0] and each of texts[1..].
fn compute_similarities(texts: &[String]) -> Result<Vec<f32>, String> {
    let url =
        std::env::var("SWARM_EMBEDDING_URL").unwrap_or_else(|_| EMBEDDING_SERVICE_URL.to_string());

    let client = reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_millis(EMBEDDING_TIMEOUT_MS))
        .build()
        .map_err(|e| format!("HTTP client error: {e}"))?;

    let body = serde_json::json!({
        "model": EMBEDDING_MODEL,
        "input": texts,
    });

    let response = client
        .post(&url)
        .json(&body)
        .send()
        .map_err(|e| format!("Embedding request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Embedding service returned {}", response.status()));
    }

    let json: serde_json::Value = response
        .json()
        .map_err(|e| format!("Failed to parse embedding response: {e}"))?;

    // Parse embeddings from OpenAI-compatible response
    let data = json["data"]
        .as_array()
        .ok_or("Missing 'data' array in embedding response")?;

    let embeddings: Vec<Vec<f32>> = data
        .iter()
        .map(|item| {
            item["embedding"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect::<Vec<f32>>()
        })
        .collect();

    if embeddings.len() != texts.len() {
        return Err(format!(
            "Expected {} embeddings, got {}",
            texts.len(),
            embeddings.len()
        ));
    }

    // Compute cosine similarity between embeddings[0] (objective) and each other
    let query = &embeddings[0];
    let similarities: Vec<f32> = embeddings[1..]
        .iter()
        .map(|doc| cosine_similarity(query, doc))
        .collect();

    Ok(similarities)
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
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
    use crate::work_packet::types::ContextProvenance;

    fn make_context(file: &str, content: &str, priority: u8) -> FileContext {
        FileContext {
            file: file.to_string(),
            start_line: 1,
            end_line: 10,
            content: content.to_string(),
            relevance: "test".to_string(),
            priority,
            provenance: ContextProvenance::Header,
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_score_and_rerank_preserves_high_priority() {
        let mut contexts = vec![
            make_context("error.rs", "compiler error", 0),
            make_context("modified.rs", "changed code", 1),
            make_context("reference.rs", "some reference", 2),
        ];
        // Without embedding service, priorities should be unchanged
        score_and_rerank("fix the bug", &mut contexts);
        assert_eq!(contexts[0].priority, 0);
        assert_eq!(contexts[1].priority, 1);
        // priority 2 context may or may not change depending on service
    }

    #[test]
    fn test_score_and_rerank_empty_contexts() {
        let mut contexts: Vec<FileContext> = vec![];
        score_and_rerank("anything", &mut contexts);
        assert!(contexts.is_empty());
    }

    #[test]
    fn test_score_and_rerank_empty_objective() {
        let mut contexts = vec![make_context("a.rs", "code", 2)];
        score_and_rerank("", &mut contexts);
        assert_eq!(contexts[0].priority, 2); // unchanged
    }
}
