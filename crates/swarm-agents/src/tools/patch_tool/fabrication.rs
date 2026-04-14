// ---------------------------------------------------------------------------
// Data fabrication guard
// ---------------------------------------------------------------------------

/// File patterns that are likely data/results files (not source code).
const DATA_FILE_EXTENSIONS: &[&str] = &[
    ".tsv", ".csv", ".jsonl", ".ndjson", ".dat", ".out", ".results",
];

const DATA_FILE_PATTERNS: &[&str] = &[
    "experiments",
    "results",
    "benchmark",
    "metrics",
    "output",
    "quality-trend",
    "summary",
    "scores",
    "measurements",
];

/// Check if a path looks like a data/results file rather than source code.
pub fn is_data_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    // Check extension
    if DATA_FILE_EXTENSIONS.iter().any(|ext| lower.ends_with(ext)) {
        return true;
    }
    // Check path components
    DATA_FILE_PATTERNS.iter().any(|pat| lower.contains(pat))
}

/// Heuristic: does this content look like fabricated benchmark/experimental data?
///
/// Detects structured numeric data that a worker model would generate when
/// it can't actually run benchmarks. Looks for patterns like:
/// - Multiple lines with floating point numbers and metric-like labels
/// - Tabular data with consistent column structure
/// - Benchmark result patterns (F1, RMSE, MAE, latency, wall_time, etc.)
///
/// This is conservative — it only triggers on content with BOTH metric keywords
/// AND dense numeric data, to avoid false positives on legitimate code edits.
pub fn looks_like_fabricated_data(content: &str) -> bool {
    // Must have metric-like keywords
    let metric_keywords = [
        "F1",
        "RMSE",
        "MAE",
        "precision",
        "recall",
        "accuracy",
        "latency",
        "wall_time",
        "throughput",
        "Jaccard",
        "Aitchison",
        "score",
        "baseline",
        "benchmark",
        "elapsed",
        "tok/s",
        "PASS",
        "FAIL",
    ];
    let keyword_count = metric_keywords
        .iter()
        .filter(|kw| content.contains(*kw))
        .count();
    if keyword_count < 2 {
        return false;
    }

    // Must have dense numeric content (>30% of lines have floats)
    let lines: Vec<&str> = content.lines().collect();
    if lines.len() < 3 {
        return false;
    }
    let numeric_lines = lines
        .iter()
        .filter(|line| {
            // Line contains at least one float-like pattern (e.g., 0.847, 2.34)
            line.split_whitespace().any(|word| {
                let cleaned = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '.');
                cleaned.contains('.') && cleaned.parse::<f64>().is_ok()
            })
        })
        .count();

    let ratio = numeric_lines as f64 / lines.len() as f64;
    if ratio < 0.3 {
        return false;
    }

    tracing::warn!(
        keyword_count,
        numeric_lines,
        total_lines = lines.len(),
        numeric_ratio = format!("{:.1}%", ratio * 100.0),
        "Data fabrication guard: content looks like fabricated benchmark results"
    );
    true
}
