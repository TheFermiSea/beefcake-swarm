//! Patch critique feedback — structured reviewer→coder repair plumbing.
//!
//! Transforms raw reviewer feedback into actionable repair instructions
//! that the coder can consume for targeted fixes.

use serde::{Deserialize, Serialize};

/// Severity of a critique issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueSeverity {
    /// Must fix before approval.
    Blocking,
    /// Should fix but won't block.
    Warning,
    /// Nice to have.
    Suggestion,
}

impl CritiqueSeverity {
    /// Whether this severity blocks approval.
    pub fn is_blocking(self) -> bool {
        matches!(self, Self::Blocking)
    }
}

impl std::fmt::Display for CritiqueSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Blocking => write!(f, "blocking"),
            Self::Warning => write!(f, "warning"),
            Self::Suggestion => write!(f, "suggestion"),
        }
    }
}

/// Category of the critique issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CritiqueCategory {
    /// Correctness bug.
    Correctness,
    /// Borrow checker or lifetime issue.
    BorrowChecker,
    /// Type mismatch.
    TypeMismatch,
    /// Missing error handling.
    ErrorHandling,
    /// Missing test coverage.
    TestCoverage,
    /// API design issue.
    ApiDesign,
    /// Performance concern.
    Performance,
    /// Style / idiomatic Rust.
    Style,
    /// Security vulnerability.
    Security,
    /// Other / uncategorized.
    Other,
}

impl std::fmt::Display for CritiqueCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Correctness => write!(f, "correctness"),
            Self::BorrowChecker => write!(f, "borrow_checker"),
            Self::TypeMismatch => write!(f, "type_mismatch"),
            Self::ErrorHandling => write!(f, "error_handling"),
            Self::TestCoverage => write!(f, "test_coverage"),
            Self::ApiDesign => write!(f, "api_design"),
            Self::Performance => write!(f, "performance"),
            Self::Style => write!(f, "style"),
            Self::Security => write!(f, "security"),
            Self::Other => write!(f, "other"),
        }
    }
}

/// A single critique item from the reviewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueItem {
    /// Severity of this issue.
    pub severity: CritiqueSeverity,
    /// Category of this issue.
    pub category: CritiqueCategory,
    /// File path where the issue was found.
    pub file: Option<String>,
    /// Line range (start, end) if applicable.
    pub line_range: Option<(u32, u32)>,
    /// Description of the issue.
    pub description: String,
    /// Suggested fix (if reviewer has one).
    pub suggested_fix: Option<String>,
}

impl CritiqueItem {
    /// Create a new blocking critique.
    pub fn blocking(category: CritiqueCategory, description: &str) -> Self {
        Self {
            severity: CritiqueSeverity::Blocking,
            category,
            file: None,
            line_range: None,
            description: description.to_string(),
            suggested_fix: None,
        }
    }

    /// Create a new warning critique.
    pub fn warning(category: CritiqueCategory, description: &str) -> Self {
        Self {
            severity: CritiqueSeverity::Warning,
            category,
            file: None,
            line_range: None,
            description: description.to_string(),
            suggested_fix: None,
        }
    }

    /// Set the file location.
    pub fn in_file(mut self, file: &str) -> Self {
        self.file = Some(file.to_string());
        self
    }

    /// Set the line range.
    pub fn at_lines(mut self, start: u32, end: u32) -> Self {
        self.line_range = Some((start, end));
        self
    }

    /// Set a suggested fix.
    pub fn with_fix(mut self, fix: &str) -> Self {
        self.suggested_fix = Some(fix.to_string());
        self
    }

    /// Location string for display.
    pub fn location(&self) -> String {
        match (&self.file, self.line_range) {
            (Some(f), Some((start, end))) => format!("{}:{}-{}", f, start, end),
            (Some(f), None) => f.clone(),
            (None, _) => "(no location)".to_string(),
        }
    }
}

impl std::fmt::Display for CritiqueItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}][{}] {} @ {}",
            self.severity,
            self.category,
            self.description,
            self.location()
        )
    }
}

/// A complete critique from a reviewer for one round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchCritique {
    /// Round this critique applies to.
    pub round: u32,
    /// Individual critique items.
    pub items: Vec<CritiqueItem>,
    /// Overall reviewer assessment.
    pub overall_assessment: String,
    /// Whether the reviewer considers the approach sound.
    pub approach_sound: bool,
}

impl PatchCritique {
    /// Create a new patch critique.
    pub fn new(round: u32, overall_assessment: &str) -> Self {
        Self {
            round,
            items: Vec::new(),
            overall_assessment: overall_assessment.to_string(),
            approach_sound: true,
        }
    }

    /// Add a critique item.
    pub fn add_item(&mut self, item: CritiqueItem) {
        self.items.push(item);
    }

    /// Number of blocking issues.
    pub fn blocking_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| i.severity.is_blocking())
            .count()
    }

    /// Number of non-blocking issues (warnings + suggestions).
    pub fn non_blocking_count(&self) -> usize {
        self.items
            .iter()
            .filter(|i| !i.severity.is_blocking())
            .count()
    }

    /// Whether there are any blocking issues.
    pub fn has_blockers(&self) -> bool {
        self.blocking_count() > 0
    }

    /// Get all blocking items.
    pub fn blockers(&self) -> Vec<&CritiqueItem> {
        self.items
            .iter()
            .filter(|i| i.severity.is_blocking())
            .collect()
    }

    /// Get items by category.
    pub fn by_category(&self, category: CritiqueCategory) -> Vec<&CritiqueItem> {
        self.items
            .iter()
            .filter(|i| i.category == category)
            .collect()
    }
}

/// Structured repair instruction derived from a critique.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairInstruction {
    /// Priority (lower = more important).
    pub priority: u32,
    /// Category of the repair.
    pub category: CritiqueCategory,
    /// File to modify.
    pub target_file: Option<String>,
    /// What to fix.
    pub instruction: String,
    /// Suggested approach.
    pub approach: Option<String>,
}

impl std::fmt::Display for RepairInstruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[P{}][{}] {}",
            self.priority, self.category, self.instruction
        )
    }
}

/// Generates repair instructions from a patch critique.
///
/// Transforms reviewer feedback into prioritized, actionable repair
/// instructions that a coder agent can process sequentially.
pub fn generate_repair_instructions(critique: &PatchCritique) -> Vec<RepairInstruction> {
    let mut instructions: Vec<RepairInstruction> = Vec::new();
    let mut priority = 0u32;

    // Blocking items first, in order
    for item in critique.items.iter().filter(|i| i.severity.is_blocking()) {
        priority += 1;
        instructions.push(RepairInstruction {
            priority,
            category: item.category,
            target_file: item.file.clone(),
            instruction: item.description.clone(),
            approach: item.suggested_fix.clone(),
        });
    }

    // Then warnings
    for item in &critique.items {
        if item.severity == CritiqueSeverity::Warning {
            priority += 1;
            instructions.push(RepairInstruction {
                priority,
                category: item.category,
                target_file: item.file.clone(),
                instruction: item.description.clone(),
                approach: item.suggested_fix.clone(),
            });
        }
    }

    // Suggestions last
    for item in &critique.items {
        if item.severity == CritiqueSeverity::Suggestion {
            priority += 1;
            instructions.push(RepairInstruction {
                priority,
                category: item.category,
                target_file: item.file.clone(),
                instruction: item.description.clone(),
                approach: item.suggested_fix.clone(),
            });
        }
    }

    instructions
}

/// Format a critique as a compact prompt fragment for the coder.
pub fn format_critique_for_coder(critique: &PatchCritique) -> String {
    let mut out = String::new();

    out.push_str(&format!(
        "## Review Feedback (Round {})\n\n",
        critique.round
    ));

    if !critique.approach_sound {
        out.push_str("**WARNING: Approach is fundamentally flawed. Consider redesign.**\n\n");
    }

    out.push_str(&format!("Assessment: {}\n\n", critique.overall_assessment));

    let blockers: Vec<_> = critique.blockers();
    if !blockers.is_empty() {
        out.push_str(&format!("### Blocking Issues ({}):\n", blockers.len()));
        for (i, item) in blockers.iter().enumerate() {
            out.push_str(&format!(
                "{}. [{}] {} @ {}\n",
                i + 1,
                item.category,
                item.description,
                item.location()
            ));
            if let Some(fix) = &item.suggested_fix {
                out.push_str(&format!("   Fix: {}\n", fix));
            }
        }
        out.push('\n');
    }

    let warnings: Vec<_> = critique
        .items
        .iter()
        .filter(|i| i.severity == CritiqueSeverity::Warning)
        .collect();
    if !warnings.is_empty() {
        out.push_str(&format!("### Warnings ({}):\n", warnings.len()));
        for (i, item) in warnings.iter().enumerate() {
            out.push_str(&format!(
                "{}. [{}] {} @ {}\n",
                i + 1,
                item.category,
                item.description,
                item.location()
            ));
        }
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_critique() -> PatchCritique {
        let mut critique = PatchCritique::new(1, "Needs error handling improvements");
        critique.add_item(
            CritiqueItem::blocking(
                CritiqueCategory::ErrorHandling,
                "Missing Result return type",
            )
            .in_file("src/lib.rs")
            .at_lines(42, 50)
            .with_fix("Change return type to Result<(), Error>"),
        );
        critique.add_item(
            CritiqueItem::blocking(
                CritiqueCategory::BorrowChecker,
                "Dangling reference in loop",
            )
            .in_file("src/lib.rs")
            .at_lines(60, 65),
        );
        critique.add_item(
            CritiqueItem::warning(CritiqueCategory::Performance, "Unnecessary clone")
                .in_file("src/lib.rs")
                .at_lines(70, 70),
        );
        critique.add_item(CritiqueItem {
            severity: CritiqueSeverity::Suggestion,
            category: CritiqueCategory::Style,
            file: None,
            line_range: None,
            description: "Consider using iterators instead of for loop".to_string(),
            suggested_fix: None,
        });
        critique
    }

    #[test]
    fn test_critique_counts() {
        let critique = sample_critique();
        assert_eq!(critique.blocking_count(), 2);
        assert_eq!(critique.non_blocking_count(), 2);
        assert!(critique.has_blockers());
    }

    #[test]
    fn test_blockers() {
        let critique = sample_critique();
        let blockers = critique.blockers();
        assert_eq!(blockers.len(), 2);
        assert!(blockers
            .iter()
            .all(|b| b.severity == CritiqueSeverity::Blocking));
    }

    #[test]
    fn test_by_category() {
        let critique = sample_critique();
        let eh = critique.by_category(CritiqueCategory::ErrorHandling);
        assert_eq!(eh.len(), 1);
        assert_eq!(eh[0].category, CritiqueCategory::ErrorHandling);
    }

    #[test]
    fn test_generate_repair_instructions() {
        let critique = sample_critique();
        let instructions = generate_repair_instructions(&critique);

        assert_eq!(instructions.len(), 4);

        // Blocking items first (priority 1, 2)
        assert_eq!(instructions[0].priority, 1);
        assert!(instructions[0].category == CritiqueCategory::ErrorHandling);
        assert!(instructions[0].approach.is_some());

        assert_eq!(instructions[1].priority, 2);
        assert!(instructions[1].category == CritiqueCategory::BorrowChecker);

        // Warning next
        assert_eq!(instructions[2].priority, 3);
        assert!(instructions[2].category == CritiqueCategory::Performance);

        // Suggestion last
        assert_eq!(instructions[3].priority, 4);
        assert!(instructions[3].category == CritiqueCategory::Style);
    }

    #[test]
    fn test_format_critique_for_coder() {
        let critique = sample_critique();
        let formatted = format_critique_for_coder(&critique);

        assert!(formatted.contains("Round 1"));
        assert!(formatted.contains("Blocking Issues (2)"));
        assert!(formatted.contains("Warnings (1)"));
        assert!(formatted.contains("Missing Result return type"));
        assert!(formatted.contains("src/lib.rs:42-50"));
        assert!(formatted.contains("Fix: Change return type"));
    }

    #[test]
    fn test_format_critique_unsound_approach() {
        let mut critique = PatchCritique::new(2, "Wrong approach entirely");
        critique.approach_sound = false;
        let formatted = format_critique_for_coder(&critique);
        assert!(formatted.contains("fundamentally flawed"));
    }

    #[test]
    fn test_critique_item_location() {
        let item = CritiqueItem::blocking(CritiqueCategory::Correctness, "bug")
            .in_file("src/main.rs")
            .at_lines(10, 20);
        assert_eq!(item.location(), "src/main.rs:10-20");

        let item2 =
            CritiqueItem::blocking(CritiqueCategory::Correctness, "bug").in_file("src/main.rs");
        assert_eq!(item2.location(), "src/main.rs");

        let item3 = CritiqueItem::blocking(CritiqueCategory::Correctness, "bug");
        assert_eq!(item3.location(), "(no location)");
    }

    #[test]
    fn test_critique_item_display() {
        let item = CritiqueItem::blocking(CritiqueCategory::Security, "SQL injection")
            .in_file("src/db.rs")
            .at_lines(5, 10);
        let display = item.to_string();
        assert!(display.contains("[blocking]"));
        assert!(display.contains("[security]"));
        assert!(display.contains("src/db.rs:5-10"));
    }

    #[test]
    fn test_severity_display_and_blocking() {
        assert_eq!(CritiqueSeverity::Blocking.to_string(), "blocking");
        assert!(CritiqueSeverity::Blocking.is_blocking());
        assert!(!CritiqueSeverity::Warning.is_blocking());
        assert!(!CritiqueSeverity::Suggestion.is_blocking());
    }

    #[test]
    fn test_category_display() {
        assert_eq!(CritiqueCategory::Correctness.to_string(), "correctness");
        assert_eq!(
            CritiqueCategory::BorrowChecker.to_string(),
            "borrow_checker"
        );
        assert_eq!(CritiqueCategory::TypeMismatch.to_string(), "type_mismatch");
    }

    #[test]
    fn test_repair_instruction_display() {
        let instr = RepairInstruction {
            priority: 1,
            category: CritiqueCategory::ErrorHandling,
            target_file: Some("src/lib.rs".to_string()),
            instruction: "Add error handling".to_string(),
            approach: None,
        };
        let display = instr.to_string();
        assert!(display.contains("[P1]"));
        assert!(display.contains("[error_handling]"));
    }

    #[test]
    fn test_empty_critique() {
        let critique = PatchCritique::new(1, "Looks good");
        assert_eq!(critique.blocking_count(), 0);
        assert_eq!(critique.non_blocking_count(), 0);
        assert!(!critique.has_blockers());

        let instructions = generate_repair_instructions(&critique);
        assert!(instructions.is_empty());
    }

    #[test]
    fn test_critique_serde_roundtrip() {
        let critique = sample_critique();
        let json = serde_json::to_string(&critique).unwrap();
        let parsed: PatchCritique = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.round, 1);
        assert_eq!(parsed.items.len(), 4);
        assert_eq!(parsed.blocking_count(), 2);
    }

    #[test]
    fn test_severity_serde() {
        let json = serde_json::to_string(&CritiqueSeverity::Blocking).unwrap();
        assert_eq!(json, "\"blocking\"");
        let parsed: CritiqueSeverity = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, CritiqueSeverity::Blocking);
    }

    #[test]
    fn test_category_serde() {
        let json = serde_json::to_string(&CritiqueCategory::BorrowChecker).unwrap();
        assert_eq!(json, "\"borrow_checker\"");
        let parsed: CritiqueCategory = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, CritiqueCategory::BorrowChecker);
    }
}
