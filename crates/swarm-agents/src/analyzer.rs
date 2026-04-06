//! Analyzer agent — distills resolution outcomes into reusable insights.
//!
//! After each successful resolution, asks the fast-tier LLM to extract
//! what worked, what principle it demonstrates, and guidance for future workers.
//!
//! Source: ASI-Evolve (arxiv:2603.29640) pipeline/analyzer.py

/// Input to the analyzer — everything we know about the resolution.
#[derive(Debug)]
pub struct AnalyzerInput {
    pub issue_id: String,
    pub issue_title: String,
    pub error_category: Option<String>,
    pub model_used: Option<String>,
    pub iterations: u32,
    pub diff_summary: String,
    pub verifier_gates: String,
}

/// Output from the analyzer — a structured insight.
#[derive(Debug, Clone)]
pub struct AnalyzerInsight {
    /// What fix pattern was used.
    pub pattern: String,
    /// What general principle this demonstrates.
    pub principle: String,
    /// What future workers should know.
    pub guidance: String,
    /// What error category this applies to.
    pub error_type: Option<String>,
}

/// Build the analyzer prompt for the LLM.
pub fn build_analyzer_prompt(input: &AnalyzerInput) -> String {
    format!(
        "Analyze this successful code resolution and distill reusable insights.\n\n\
         Issue: {} ({})\n\
         Error category: {}\n\
         Model: {}\n\
         Iterations to resolve: {}\n\
         Verifier: {}\n\n\
         Diff summary:\n{}\n\n\
         Distill in 2-3 sentences:\n\
         1. What fix pattern was applied?\n\
         2. What general principle does this demonstrate?\n\
         3. What should future workers know about this error type?\n\n\
         Be concise and actionable — this will be stored as context for future attempts.",
        input.issue_title,
        input.issue_id,
        input.error_category.as_deref().unwrap_or("unknown"),
        input.model_used.as_deref().unwrap_or("unknown"),
        input.iterations,
        input.verifier_gates,
        input.diff_summary,
    )
}

/// Parse the LLM's analysis response into a structured insight.
pub fn parse_analysis(response: &str) -> AnalyzerInsight {
    let lines: Vec<&str> = response.lines().collect();

    AnalyzerInsight {
        pattern: lines.first().unwrap_or(&"").to_string(),
        principle: lines.get(1).unwrap_or(&"").to_string(),
        guidance: lines.get(2..).map(|l| l.join(" ")).unwrap_or_default(),
        error_type: None, // Set by caller from AnalyzerInput
    }
}

/// Format an insight for storage in the Cognition Base.
pub fn insight_to_cognition_content(input: &AnalyzerInput, insight: &AnalyzerInsight) -> String {
    format!(
        "Resolution insight for {} errors (from {}):\n\
         Pattern: {}\n\
         Principle: {}\n\
         Guidance: {}",
        input.error_category.as_deref().unwrap_or("general"),
        input.issue_id,
        insight.pattern,
        insight.principle,
        insight.guidance,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> AnalyzerInput {
        AnalyzerInput {
            issue_id: "beefcake-001".into(),
            issue_title: "Fix borrow checker error in router".into(),
            error_category: Some("borrow_checker".into()),
            model_used: Some("Qwen3.5-27B".into()),
            iterations: 2,
            diff_summary: "3 files changed, 12 insertions(+), 4 deletions(-)".into(),
            verifier_gates: "fmt=pass, clippy=pass, check=pass, test=pass".into(),
        }
    }

    #[test]
    fn test_build_analyzer_prompt_includes_all_fields() {
        let input = sample_input();
        let prompt = build_analyzer_prompt(&input);

        assert!(prompt.contains("Fix borrow checker error in router"));
        assert!(prompt.contains("beefcake-001"));
        assert!(prompt.contains("borrow_checker"));
        assert!(prompt.contains("Qwen3.5-27B"));
        assert!(prompt.contains("Iterations to resolve: 2"));
        assert!(prompt.contains("3 files changed"));
        assert!(prompt.contains("fmt=pass"));
    }

    #[test]
    fn test_build_analyzer_prompt_handles_none_fields() {
        let input = AnalyzerInput {
            issue_id: "test-002".into(),
            issue_title: "Some issue".into(),
            error_category: None,
            model_used: None,
            iterations: 1,
            diff_summary: String::new(),
            verifier_gates: String::new(),
        };
        let prompt = build_analyzer_prompt(&input);

        assert!(prompt.contains("Error category: unknown"));
        assert!(prompt.contains("Model: unknown"));
    }

    #[test]
    fn test_parse_analysis_basic() {
        let response = "Clone the value before moving into the closure.\n\
                        Ownership transfer across closures requires explicit cloning.\n\
                        Always check if a value is moved into a closure and clone if needed.";

        let insight = parse_analysis(response);

        assert!(insight.pattern.contains("Clone the value"));
        assert!(insight.principle.contains("Ownership transfer"));
        assert!(insight.guidance.contains("Always check"));
        assert!(insight.error_type.is_none());
    }

    #[test]
    fn test_parse_analysis_empty_response() {
        let insight = parse_analysis("");

        assert!(insight.pattern.is_empty());
        assert!(insight.principle.is_empty());
        assert!(insight.guidance.is_empty());
    }

    #[test]
    fn test_parse_analysis_single_line() {
        let insight = parse_analysis("Just one line of analysis.");

        assert_eq!(insight.pattern, "Just one line of analysis.");
        assert!(insight.principle.is_empty());
        assert!(insight.guidance.is_empty());
    }

    #[test]
    fn test_insight_to_cognition_content() {
        let input = sample_input();
        let insight = AnalyzerInsight {
            pattern: "Clone before closure capture".into(),
            principle: "Rust ownership requires explicit cloning".into(),
            guidance: "Check for move semantics in closures".into(),
            error_type: Some("borrow_checker".into()),
        };

        let content = insight_to_cognition_content(&input, &insight);

        assert!(content.contains("borrow_checker errors"));
        assert!(content.contains("beefcake-001"));
        assert!(content.contains("Pattern: Clone before closure capture"));
        assert!(content.contains("Principle: Rust ownership requires explicit cloning"));
        assert!(content.contains("Guidance: Check for move semantics"));
    }

    #[test]
    fn test_insight_to_cognition_content_no_error_category() {
        let input = AnalyzerInput {
            issue_id: "test-003".into(),
            issue_title: "Add new feature".into(),
            error_category: None,
            model_used: None,
            iterations: 1,
            diff_summary: String::new(),
            verifier_gates: String::new(),
        };
        let insight = AnalyzerInsight {
            pattern: "Simple addition".into(),
            principle: "Feature gating".into(),
            guidance: "Use config flags".into(),
            error_type: None,
        };

        let content = insight_to_cognition_content(&input, &insight);
        assert!(content.contains("general errors"));
    }
}
