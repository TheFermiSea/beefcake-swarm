//! Agent profile hardening — role-based tool access control.
//!
//! Defines the capability matrix for swarm agent roles (coder, reviewer,
//! manager, reasoner) and enforces tool access boundaries.
//!
//! # Roles
//!
//! - **Coder**: Mutation + required read tools only (write files, run commands)
//! - **Reviewer**: Verifier/search/analysis tools only (no mutation)
//! - **Manager**: Orchestration + delegation tools (no direct mutation)
//! - **Reasoner**: Analysis + knowledge tools (read-only)

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Agent role in the swarm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentRole {
    /// Code producer — writes/modifies files and runs build commands.
    Coder,
    /// Code evaluator — runs verifier, search, analysis tools.
    Reviewer,
    /// Orchestration coordinator — delegates tasks, manages flow.
    Manager,
    /// Deep analysis/reasoning — read-only knowledge queries.
    Reasoner,
}

impl AgentRole {
    /// All defined roles.
    pub fn all() -> &'static [AgentRole] {
        &[Self::Coder, Self::Reviewer, Self::Manager, Self::Reasoner]
    }
}

impl std::fmt::Display for AgentRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Coder => write!(f, "coder"),
            Self::Reviewer => write!(f, "reviewer"),
            Self::Manager => write!(f, "manager"),
            Self::Reasoner => write!(f, "reasoner"),
        }
    }
}

/// Tool category for access control classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolCategory {
    /// File mutation (write, edit, delete).
    FileMutation,
    /// File read (read, search, glob).
    FileRead,
    /// Shell command execution.
    ShellExec,
    /// Verifier/quality gate execution.
    Verifier,
    /// AST analysis (ast-grep).
    AstAnalysis,
    /// Dependency/graph analysis.
    DependencyAnalysis,
    /// Knowledge query (NotebookLM, RAG).
    KnowledgeQuery,
    /// Task delegation (assign work to agents).
    Delegation,
    /// Git operations (commit, branch, merge).
    GitOps,
    /// Beads issue tracking.
    IssueTracking,
}

impl std::fmt::Display for ToolCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileMutation => write!(f, "file_mutation"),
            Self::FileRead => write!(f, "file_read"),
            Self::ShellExec => write!(f, "shell_exec"),
            Self::Verifier => write!(f, "verifier"),
            Self::AstAnalysis => write!(f, "ast_analysis"),
            Self::DependencyAnalysis => write!(f, "dependency_analysis"),
            Self::KnowledgeQuery => write!(f, "knowledge_query"),
            Self::Delegation => write!(f, "delegation"),
            Self::GitOps => write!(f, "git_ops"),
            Self::IssueTracking => write!(f, "issue_tracking"),
        }
    }
}

/// Access level for a tool category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AccessLevel {
    /// Full access — tool is available and expected to be used.
    Allowed,
    /// Read-only access — can observe but not mutate.
    ReadOnly,
    /// Denied — tool must not be provided to this role.
    Denied,
}

impl AccessLevel {
    /// Whether this access level permits the tool to be used.
    pub fn is_permitted(self) -> bool {
        matches!(self, Self::Allowed | Self::ReadOnly)
    }

    /// Whether this is a mutation-capable access level.
    pub fn allows_mutation(self) -> bool {
        matches!(self, Self::Allowed)
    }
}

impl std::fmt::Display for AccessLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Allowed => write!(f, "allowed"),
            Self::ReadOnly => write!(f, "read_only"),
            Self::Denied => write!(f, "denied"),
        }
    }
}

/// A single entry in the capability matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityEntry {
    /// The role.
    pub role: AgentRole,
    /// The tool category.
    pub category: ToolCategory,
    /// Access level.
    pub access: AccessLevel,
    /// Rationale for this access decision.
    pub rationale: String,
}

/// The complete agent capability matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityMatrix {
    entries: Vec<CapabilityEntry>,
}

impl CapabilityMatrix {
    /// Build the default capability matrix.
    pub fn default_matrix() -> Self {
        let mut entries = Vec::new();

        // Coder: mutation + required reads
        let coder_rules = [
            (
                ToolCategory::FileMutation,
                AccessLevel::Allowed,
                "Primary coder capability",
            ),
            (
                ToolCategory::FileRead,
                AccessLevel::Allowed,
                "Must read files to modify them",
            ),
            (
                ToolCategory::ShellExec,
                AccessLevel::Allowed,
                "Runs build/test commands",
            ),
            (
                ToolCategory::Verifier,
                AccessLevel::Denied,
                "Reviewer responsibility",
            ),
            (
                ToolCategory::AstAnalysis,
                AccessLevel::Denied,
                "Reviewer responsibility",
            ),
            (
                ToolCategory::DependencyAnalysis,
                AccessLevel::Denied,
                "Reviewer responsibility",
            ),
            (
                ToolCategory::KnowledgeQuery,
                AccessLevel::ReadOnly,
                "Can query for guidance",
            ),
            (
                ToolCategory::Delegation,
                AccessLevel::Denied,
                "Manager responsibility",
            ),
            (
                ToolCategory::GitOps,
                AccessLevel::Allowed,
                "Commits own changes",
            ),
            (
                ToolCategory::IssueTracking,
                AccessLevel::ReadOnly,
                "Reads issue details",
            ),
        ];

        for (cat, access, rationale) in coder_rules {
            entries.push(CapabilityEntry {
                role: AgentRole::Coder,
                category: cat,
                access,
                rationale: rationale.to_string(),
            });
        }

        // Reviewer: verifier/search/analysis, no mutation
        let reviewer_rules = [
            (
                ToolCategory::FileMutation,
                AccessLevel::Denied,
                "Reviewer must not mutate code",
            ),
            (
                ToolCategory::FileRead,
                AccessLevel::Allowed,
                "Must read code for review",
            ),
            (
                ToolCategory::ShellExec,
                AccessLevel::Denied,
                "No arbitrary shell access",
            ),
            (
                ToolCategory::Verifier,
                AccessLevel::Allowed,
                "Primary reviewer capability",
            ),
            (
                ToolCategory::AstAnalysis,
                AccessLevel::Allowed,
                "Structural code analysis",
            ),
            (
                ToolCategory::DependencyAnalysis,
                AccessLevel::Allowed,
                "Impact analysis",
            ),
            (
                ToolCategory::KnowledgeQuery,
                AccessLevel::Allowed,
                "Reference documentation",
            ),
            (
                ToolCategory::Delegation,
                AccessLevel::Denied,
                "Manager responsibility",
            ),
            (
                ToolCategory::GitOps,
                AccessLevel::Denied,
                "No git mutations",
            ),
            (
                ToolCategory::IssueTracking,
                AccessLevel::ReadOnly,
                "Reads issue context",
            ),
        ];

        for (cat, access, rationale) in reviewer_rules {
            entries.push(CapabilityEntry {
                role: AgentRole::Reviewer,
                category: cat,
                access,
                rationale: rationale.to_string(),
            });
        }

        // Manager: orchestration + delegation
        let manager_rules = [
            (
                ToolCategory::FileMutation,
                AccessLevel::Denied,
                "Delegates to coder",
            ),
            (
                ToolCategory::FileRead,
                AccessLevel::Allowed,
                "Reads for context",
            ),
            (
                ToolCategory::ShellExec,
                AccessLevel::Denied,
                "Delegates to coder",
            ),
            (
                ToolCategory::Verifier,
                AccessLevel::ReadOnly,
                "Reads verifier results",
            ),
            (
                ToolCategory::AstAnalysis,
                AccessLevel::ReadOnly,
                "Reads analysis results",
            ),
            (
                ToolCategory::DependencyAnalysis,
                AccessLevel::ReadOnly,
                "Reads analysis results",
            ),
            (
                ToolCategory::KnowledgeQuery,
                AccessLevel::Allowed,
                "Architecture decisions",
            ),
            (
                ToolCategory::Delegation,
                AccessLevel::Allowed,
                "Primary manager capability",
            ),
            (
                ToolCategory::GitOps,
                AccessLevel::Allowed,
                "Merge orchestration",
            ),
            (
                ToolCategory::IssueTracking,
                AccessLevel::Allowed,
                "Full issue management",
            ),
        ];

        for (cat, access, rationale) in manager_rules {
            entries.push(CapabilityEntry {
                role: AgentRole::Manager,
                category: cat,
                access,
                rationale: rationale.to_string(),
            });
        }

        // Reasoner: read-only analysis + knowledge
        let reasoner_rules = [
            (
                ToolCategory::FileMutation,
                AccessLevel::Denied,
                "Read-only role",
            ),
            (
                ToolCategory::FileRead,
                AccessLevel::Allowed,
                "Deep code analysis",
            ),
            (
                ToolCategory::ShellExec,
                AccessLevel::Denied,
                "Read-only role",
            ),
            (
                ToolCategory::Verifier,
                AccessLevel::ReadOnly,
                "Can read results",
            ),
            (
                ToolCategory::AstAnalysis,
                AccessLevel::Allowed,
                "Deep structural analysis",
            ),
            (
                ToolCategory::DependencyAnalysis,
                AccessLevel::Allowed,
                "Deep impact analysis",
            ),
            (
                ToolCategory::KnowledgeQuery,
                AccessLevel::Allowed,
                "Primary reasoner capability",
            ),
            (
                ToolCategory::Delegation,
                AccessLevel::Denied,
                "Manager responsibility",
            ),
            (ToolCategory::GitOps, AccessLevel::Denied, "Read-only role"),
            (
                ToolCategory::IssueTracking,
                AccessLevel::ReadOnly,
                "Reads issue context",
            ),
        ];

        for (cat, access, rationale) in reasoner_rules {
            entries.push(CapabilityEntry {
                role: AgentRole::Reasoner,
                category: cat,
                access,
                rationale: rationale.to_string(),
            });
        }

        Self { entries }
    }

    /// Get the access level for a role + category combination.
    pub fn access_level(&self, role: AgentRole, category: ToolCategory) -> AccessLevel {
        self.entries
            .iter()
            .find(|e| e.role == role && e.category == category)
            .map(|e| e.access)
            .unwrap_or(AccessLevel::Denied) // default deny
    }

    /// Check whether a role is allowed to use a tool category.
    pub fn is_permitted(&self, role: AgentRole, category: ToolCategory) -> bool {
        self.access_level(role, category).is_permitted()
    }

    /// Get all permitted categories for a role.
    pub fn permitted_categories(&self, role: AgentRole) -> BTreeSet<ToolCategory> {
        self.entries
            .iter()
            .filter(|e| e.role == role && e.access.is_permitted())
            .map(|e| e.category)
            .collect()
    }

    /// Get all denied categories for a role.
    pub fn denied_categories(&self, role: AgentRole) -> BTreeSet<ToolCategory> {
        self.entries
            .iter()
            .filter(|e| e.role == role && !e.access.is_permitted())
            .map(|e| e.category)
            .collect()
    }

    /// Validate a tool request against the matrix.
    pub fn validate_access(
        &self,
        role: AgentRole,
        category: ToolCategory,
    ) -> Result<AccessLevel, AccessViolation> {
        let level = self.access_level(role, category);
        if level.is_permitted() {
            Ok(level)
        } else {
            Err(AccessViolation {
                role,
                category,
                attempted: level,
                rationale: self
                    .entries
                    .iter()
                    .find(|e| e.role == role && e.category == category)
                    .map(|e| e.rationale.clone())
                    .unwrap_or_default(),
            })
        }
    }

    /// Generate a summary table of the matrix.
    pub fn summary_table(&self) -> BTreeMap<AgentRole, BTreeMap<ToolCategory, AccessLevel>> {
        let mut table = BTreeMap::new();
        for entry in &self.entries {
            table
                .entry(entry.role)
                .or_insert_with(BTreeMap::new)
                .insert(entry.category, entry.access);
        }
        table
    }
}

/// Access violation error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AccessViolation {
    /// Role that attempted the access.
    pub role: AgentRole,
    /// Tool category that was requested.
    pub category: ToolCategory,
    /// The access level (Denied).
    pub attempted: AccessLevel,
    /// Rationale for the denial.
    pub rationale: String,
}

impl std::fmt::Display for AccessViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "access denied: {} cannot use {} ({})",
            self.role, self.category, self.rationale
        )
    }
}

impl std::error::Error for AccessViolation {}

/// Prompt contract for a role — defines expected behavior constraints.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptContract {
    /// The role this contract applies to.
    pub role: AgentRole,
    /// Required output format (e.g., "json", "structured_review").
    pub output_format: String,
    /// Whether structured output is required (vs. freeform).
    pub structured_output_required: bool,
    /// Maximum output tokens.
    pub max_output_tokens: u32,
    /// Constraints that the prompt must enforce.
    pub constraints: Vec<String>,
    /// Tools that must be mentioned in the prompt.
    pub required_tool_mentions: Vec<String>,
    /// Tools that must NOT be mentioned in the prompt.
    pub forbidden_tool_mentions: Vec<String>,
}

impl PromptContract {
    /// Create the default coder contract.
    pub fn coder() -> Self {
        Self {
            role: AgentRole::Coder,
            output_format: "code_with_explanation".to_string(),
            structured_output_required: false,
            max_output_tokens: 8192,
            constraints: vec![
                "Must only modify files within the assigned worktree".to_string(),
                "Must not run destructive git commands (reset --hard, clean -f)".to_string(),
                "Must explain all changes made".to_string(),
            ],
            required_tool_mentions: vec!["file_write".to_string(), "shell_exec".to_string()],
            forbidden_tool_mentions: vec![
                "verifier_gate".to_string(),
                "ast_grep".to_string(),
                "delegate_task".to_string(),
            ],
        }
    }

    /// Create the default reviewer contract.
    pub fn reviewer() -> Self {
        Self {
            role: AgentRole::Reviewer,
            output_format: "structured_review".to_string(),
            structured_output_required: true,
            max_output_tokens: 4096,
            constraints: vec![
                "Must not suggest file modifications directly".to_string(),
                "Must produce structured verdict (approve/request_changes/abstain)".to_string(),
                "Must list all blocking issues with file locations".to_string(),
            ],
            required_tool_mentions: vec!["verifier_gate".to_string(), "file_read".to_string()],
            forbidden_tool_mentions: vec![
                "file_write".to_string(),
                "shell_exec".to_string(),
                "git_commit".to_string(),
            ],
        }
    }

    /// Create the default manager contract.
    pub fn manager() -> Self {
        Self {
            role: AgentRole::Manager,
            output_format: "delegation_plan".to_string(),
            structured_output_required: true,
            max_output_tokens: 4096,
            constraints: vec![
                "Must not write code directly".to_string(),
                "Must delegate implementation to coder role".to_string(),
                "Must check verifier results before approving merge".to_string(),
            ],
            required_tool_mentions: vec!["delegate_task".to_string(), "issue_tracking".to_string()],
            forbidden_tool_mentions: vec!["file_write".to_string(), "shell_exec".to_string()],
        }
    }

    /// Create the default reasoner contract.
    pub fn reasoner() -> Self {
        Self {
            role: AgentRole::Reasoner,
            output_format: "analysis_report".to_string(),
            structured_output_required: true,
            max_output_tokens: 8192,
            constraints: vec![
                "Must not suggest specific code changes".to_string(),
                "Must provide architectural rationale".to_string(),
                "Must reference existing patterns in codebase".to_string(),
            ],
            required_tool_mentions: vec!["file_read".to_string(), "knowledge_query".to_string()],
            forbidden_tool_mentions: vec![
                "file_write".to_string(),
                "shell_exec".to_string(),
                "git_commit".to_string(),
                "delegate_task".to_string(),
            ],
        }
    }

    /// Validate that a prompt text respects the contract.
    pub fn validate_prompt(&self, prompt: &str) -> Vec<ContractViolation> {
        let mut violations = Vec::new();

        for tool in &self.forbidden_tool_mentions {
            if prompt.contains(tool.as_str()) {
                violations.push(ContractViolation {
                    role: self.role,
                    kind: ViolationKind::ForbiddenToolMentioned,
                    detail: format!("prompt mentions forbidden tool: {}", tool),
                });
            }
        }

        for tool in &self.required_tool_mentions {
            if !prompt.contains(tool.as_str()) {
                violations.push(ContractViolation {
                    role: self.role,
                    kind: ViolationKind::RequiredToolMissing,
                    detail: format!("prompt does not mention required tool: {}", tool),
                });
            }
        }

        violations
    }
}

/// Kind of contract violation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationKind {
    /// A forbidden tool was mentioned in the prompt.
    ForbiddenToolMentioned,
    /// A required tool was not mentioned in the prompt.
    RequiredToolMissing,
}

impl std::fmt::Display for ViolationKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ForbiddenToolMentioned => write!(f, "forbidden_tool_mentioned"),
            Self::RequiredToolMissing => write!(f, "required_tool_missing"),
        }
    }
}

/// A specific contract violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractViolation {
    /// Role the contract belongs to.
    pub role: AgentRole,
    /// Kind of violation.
    pub kind: ViolationKind,
    /// Detail about what went wrong.
    pub detail: String,
}

impl std::fmt::Display for ContractViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.role, self.kind, self.detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_matrix_completeness() {
        let matrix = CapabilityMatrix::default_matrix();
        // Every role should have an entry for every tool category
        for role in AgentRole::all() {
            let permitted = matrix.permitted_categories(*role);
            let denied = matrix.denied_categories(*role);
            // Sum should cover all categories
            assert_eq!(
                permitted.len() + denied.len(),
                10,
                "Role {} has {} permitted + {} denied != 10",
                role,
                permitted.len(),
                denied.len()
            );
        }
    }

    // --- Coder boundary tests (hx0.3.2) ---

    #[test]
    fn test_coder_can_mutate_files() {
        let matrix = CapabilityMatrix::default_matrix();
        assert!(matrix.is_permitted(AgentRole::Coder, ToolCategory::FileMutation));
        assert!(matrix.is_permitted(AgentRole::Coder, ToolCategory::FileRead));
        assert!(matrix.is_permitted(AgentRole::Coder, ToolCategory::ShellExec));
        assert!(matrix.is_permitted(AgentRole::Coder, ToolCategory::GitOps));
    }

    #[test]
    fn test_coder_denied_reviewer_tools() {
        let matrix = CapabilityMatrix::default_matrix();
        assert!(!matrix.is_permitted(AgentRole::Coder, ToolCategory::Verifier));
        assert!(!matrix.is_permitted(AgentRole::Coder, ToolCategory::AstAnalysis));
        assert!(!matrix.is_permitted(AgentRole::Coder, ToolCategory::DependencyAnalysis));
        assert!(!matrix.is_permitted(AgentRole::Coder, ToolCategory::Delegation));
    }

    // --- Reviewer boundary tests (hx0.3.3) ---

    #[test]
    fn test_reviewer_can_analyze() {
        let matrix = CapabilityMatrix::default_matrix();
        assert!(matrix.is_permitted(AgentRole::Reviewer, ToolCategory::FileRead));
        assert!(matrix.is_permitted(AgentRole::Reviewer, ToolCategory::Verifier));
        assert!(matrix.is_permitted(AgentRole::Reviewer, ToolCategory::AstAnalysis));
        assert!(matrix.is_permitted(AgentRole::Reviewer, ToolCategory::DependencyAnalysis));
        assert!(matrix.is_permitted(AgentRole::Reviewer, ToolCategory::KnowledgeQuery));
    }

    #[test]
    fn test_reviewer_denied_mutation() {
        let matrix = CapabilityMatrix::default_matrix();
        assert!(!matrix.is_permitted(AgentRole::Reviewer, ToolCategory::FileMutation));
        assert!(!matrix.is_permitted(AgentRole::Reviewer, ToolCategory::ShellExec));
        assert!(!matrix.is_permitted(AgentRole::Reviewer, ToolCategory::GitOps));
        assert!(!matrix.is_permitted(AgentRole::Reviewer, ToolCategory::Delegation));
    }

    // --- Manager boundary tests ---

    #[test]
    fn test_manager_can_delegate() {
        let matrix = CapabilityMatrix::default_matrix();
        assert!(matrix.is_permitted(AgentRole::Manager, ToolCategory::Delegation));
        assert!(matrix.is_permitted(AgentRole::Manager, ToolCategory::IssueTracking));
        assert!(matrix.is_permitted(AgentRole::Manager, ToolCategory::GitOps));
        assert!(matrix.is_permitted(AgentRole::Manager, ToolCategory::KnowledgeQuery));
    }

    #[test]
    fn test_manager_denied_direct_mutation() {
        let matrix = CapabilityMatrix::default_matrix();
        assert!(!matrix.is_permitted(AgentRole::Manager, ToolCategory::FileMutation));
        assert!(!matrix.is_permitted(AgentRole::Manager, ToolCategory::ShellExec));
    }

    // --- Reasoner boundary tests ---

    #[test]
    fn test_reasoner_read_only() {
        let matrix = CapabilityMatrix::default_matrix();
        assert!(matrix.is_permitted(AgentRole::Reasoner, ToolCategory::FileRead));
        assert!(matrix.is_permitted(AgentRole::Reasoner, ToolCategory::AstAnalysis));
        assert!(matrix.is_permitted(AgentRole::Reasoner, ToolCategory::DependencyAnalysis));
        assert!(matrix.is_permitted(AgentRole::Reasoner, ToolCategory::KnowledgeQuery));
    }

    #[test]
    fn test_reasoner_denied_mutation() {
        let matrix = CapabilityMatrix::default_matrix();
        assert!(!matrix.is_permitted(AgentRole::Reasoner, ToolCategory::FileMutation));
        assert!(!matrix.is_permitted(AgentRole::Reasoner, ToolCategory::ShellExec));
        assert!(!matrix.is_permitted(AgentRole::Reasoner, ToolCategory::GitOps));
        assert!(!matrix.is_permitted(AgentRole::Reasoner, ToolCategory::Delegation));
    }

    // --- Boundary assertion tests (hx0.3.4) ---

    #[test]
    fn test_no_role_gets_all_tools() {
        let matrix = CapabilityMatrix::default_matrix();
        for role in AgentRole::all() {
            let denied = matrix.denied_categories(*role);
            assert!(
                !denied.is_empty(),
                "Role {} has no denied categories — security boundary failure",
                role
            );
        }
    }

    #[test]
    fn test_mutation_tools_not_shared_with_reviewers() {
        let matrix = CapabilityMatrix::default_matrix();
        let mutation_tools = [ToolCategory::FileMutation, ToolCategory::ShellExec];
        for tool in mutation_tools {
            let coder_access = matrix.access_level(AgentRole::Coder, tool);
            let reviewer_access = matrix.access_level(AgentRole::Reviewer, tool);
            assert!(
                coder_access.allows_mutation(),
                "Coder should have mutation access to {}",
                tool
            );
            assert!(
                !reviewer_access.is_permitted(),
                "Reviewer should not have access to {}",
                tool
            );
        }
    }

    #[test]
    fn test_validate_access_permitted() {
        let matrix = CapabilityMatrix::default_matrix();
        let result = matrix.validate_access(AgentRole::Coder, ToolCategory::FileMutation);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), AccessLevel::Allowed);
    }

    #[test]
    fn test_validate_access_denied() {
        let matrix = CapabilityMatrix::default_matrix();
        let result = matrix.validate_access(AgentRole::Reviewer, ToolCategory::FileMutation);
        assert!(result.is_err());
        let violation = result.unwrap_err();
        assert_eq!(violation.role, AgentRole::Reviewer);
        assert_eq!(violation.category, ToolCategory::FileMutation);
    }

    // --- Prompt contract tests (hx0.3.5) ---

    #[test]
    fn test_coder_prompt_contract_valid() {
        let contract = PromptContract::coder();
        let prompt = "You are a coder. Use file_write and shell_exec to implement changes.";
        let violations = contract.validate_prompt(prompt);
        assert!(violations.is_empty(), "violations: {:?}", violations);
    }

    #[test]
    fn test_coder_prompt_forbidden_tool() {
        let contract = PromptContract::coder();
        let prompt = "You are a coder. Use file_write, shell_exec, and verifier_gate.";
        let violations = contract.validate_prompt(prompt);
        assert!(!violations.is_empty());
        assert!(violations
            .iter()
            .any(|v| v.kind == ViolationKind::ForbiddenToolMentioned));
    }

    #[test]
    fn test_reviewer_prompt_contract_valid() {
        let contract = PromptContract::reviewer();
        let prompt = "You are a reviewer. Use verifier_gate and file_read to evaluate code.";
        let violations = contract.validate_prompt(prompt);
        assert!(violations.is_empty(), "violations: {:?}", violations);
    }

    #[test]
    fn test_reviewer_prompt_forbidden_mutation() {
        let contract = PromptContract::reviewer();
        let prompt = "You are a reviewer. Use verifier_gate, file_read, and file_write.";
        let violations = contract.validate_prompt(prompt);
        assert!(violations.iter().any(|v| v.detail.contains("file_write")));
    }

    #[test]
    fn test_prompt_missing_required_tool() {
        let contract = PromptContract::coder();
        let prompt = "You are a coder. Use file_write to implement.";
        // Missing shell_exec
        let violations = contract.validate_prompt(prompt);
        assert!(violations
            .iter()
            .any(|v| v.kind == ViolationKind::RequiredToolMissing));
    }

    // --- Display and serde tests ---

    #[test]
    fn test_role_display() {
        assert_eq!(AgentRole::Coder.to_string(), "coder");
        assert_eq!(AgentRole::Reviewer.to_string(), "reviewer");
        assert_eq!(AgentRole::Manager.to_string(), "manager");
        assert_eq!(AgentRole::Reasoner.to_string(), "reasoner");
    }

    #[test]
    fn test_tool_category_display() {
        assert_eq!(ToolCategory::FileMutation.to_string(), "file_mutation");
        assert_eq!(ToolCategory::Verifier.to_string(), "verifier");
        assert_eq!(ToolCategory::Delegation.to_string(), "delegation");
    }

    #[test]
    fn test_access_level_display() {
        assert_eq!(AccessLevel::Allowed.to_string(), "allowed");
        assert_eq!(AccessLevel::ReadOnly.to_string(), "read_only");
        assert_eq!(AccessLevel::Denied.to_string(), "denied");
    }

    #[test]
    fn test_access_violation_display() {
        let violation = AccessViolation {
            role: AgentRole::Reviewer,
            category: ToolCategory::FileMutation,
            attempted: AccessLevel::Denied,
            rationale: "no mutation".to_string(),
        };
        let display = violation.to_string();
        assert!(display.contains("reviewer"));
        assert!(display.contains("file_mutation"));
    }

    #[test]
    fn test_matrix_serde_roundtrip() {
        let matrix = CapabilityMatrix::default_matrix();
        let json = serde_json::to_string(&matrix).unwrap();
        let parsed: CapabilityMatrix = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.entries.len(), 40); // 4 roles × 10 categories
    }

    #[test]
    fn test_summary_table() {
        let matrix = CapabilityMatrix::default_matrix();
        let table = matrix.summary_table();
        assert_eq!(table.len(), 4); // 4 roles
        for role in AgentRole::all() {
            assert_eq!(table[role].len(), 10); // 10 categories each
        }
    }

    #[test]
    fn test_contract_serde() {
        let contract = PromptContract::coder();
        let json = serde_json::to_string(&contract).unwrap();
        let parsed: PromptContract = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, AgentRole::Coder);
        assert_eq!(parsed.max_output_tokens, 8192);
    }

    #[test]
    fn test_contract_violation_display() {
        let v = ContractViolation {
            role: AgentRole::Coder,
            kind: ViolationKind::ForbiddenToolMentioned,
            detail: "verifier_gate".to_string(),
        };
        assert!(v.to_string().contains("coder"));
        assert!(v.to_string().contains("forbidden_tool_mentioned"));
    }

    #[test]
    fn test_role_serde() {
        let json = serde_json::to_string(&AgentRole::Coder).unwrap();
        assert_eq!(json, "\"coder\"");
        let parsed: AgentRole = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, AgentRole::Coder);
    }
}
