//! Tool bundle assembly — role-based tool sets for swarm agents.
//!
//! Uses the [`CapabilityMatrix`] to assemble tool bundles per agent role,
//! ensuring tools are available where intended and nowhere else.
//!
//! # Usage
//!
//! ```text
//! let bundle = ToolBundle::for_role(AgentRole::Reviewer);
//! assert!(bundle.has_tool("ast_grep"));
//! assert!(!bundle.has_tool("file_write"));
//! ```

use crate::agent_profile::{AccessLevel, AgentRole, CapabilityMatrix, ToolCategory};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// A tool descriptor within a bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDescriptor {
    /// Unique tool name (e.g., "ast_grep", "verifier_gate").
    pub name: String,
    /// Tool category for access control.
    pub category: ToolCategory,
    /// Human-readable description.
    pub description: String,
    /// Whether the tool is read-only.
    pub read_only: bool,
}

/// A bundle of tools assembled for a specific agent role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolBundle {
    /// The role this bundle is for.
    pub role: AgentRole,
    /// Available tools (name → descriptor).
    tools: BTreeMap<String, ToolDescriptor>,
    /// Tool categories that are permitted (for quick lookup).
    permitted_categories: BTreeSet<ToolCategory>,
}

impl ToolBundle {
    /// Build a tool bundle for the given role using the default matrix.
    pub fn for_role(role: AgentRole) -> Self {
        Self::from_matrix(role, &CapabilityMatrix::default_matrix())
    }

    /// Build a tool bundle from a specific capability matrix.
    pub fn from_matrix(role: AgentRole, matrix: &CapabilityMatrix) -> Self {
        let permitted_categories = matrix.permitted_categories(role);
        let mut tools = BTreeMap::new();

        // Register all known tools with their categories
        for &(name, cat, desc, read_only) in ALL_TOOLS {
            let access = matrix.access_level(role, *cat);
            if access.is_permitted() {
                tools.insert(
                    name.to_string(),
                    ToolDescriptor {
                        name: name.to_string(),
                        category: *cat,
                        description: desc.to_string(),
                        read_only: *read_only || access == AccessLevel::ReadOnly,
                    },
                );
            }
        }

        Self {
            role,
            tools,
            permitted_categories,
        }
    }

    /// Check if a tool is available in this bundle.
    pub fn has_tool(&self, name: &str) -> bool {
        self.tools.contains_key(name)
    }

    /// Get a tool descriptor by name.
    pub fn get_tool(&self, name: &str) -> Option<&ToolDescriptor> {
        self.tools.get(name)
    }

    /// Check if a tool category is permitted.
    pub fn category_permitted(&self, category: ToolCategory) -> bool {
        self.permitted_categories.contains(&category)
    }

    /// Get all tool names in this bundle.
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.keys().map(|s| s.as_str()).collect()
    }

    /// Get all tools in this bundle.
    pub fn tools(&self) -> &BTreeMap<String, ToolDescriptor> {
        &self.tools
    }

    /// Count of tools in this bundle.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the bundle is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Filter tools by category.
    pub fn tools_in_category(&self, category: ToolCategory) -> Vec<&ToolDescriptor> {
        self.tools
            .values()
            .filter(|t| t.category == category)
            .collect()
    }

    /// Get read-only tools only.
    pub fn read_only_tools(&self) -> Vec<&ToolDescriptor> {
        self.tools.values().filter(|t| t.read_only).collect()
    }

    /// Get mutation-capable tools.
    pub fn mutation_tools(&self) -> Vec<&ToolDescriptor> {
        self.tools.values().filter(|t| !t.read_only).collect()
    }

    /// Validate that a tool request is permitted.
    pub fn validate_request(&self, tool_name: &str) -> Result<&ToolDescriptor, BundleViolation> {
        self.tools.get(tool_name).ok_or(BundleViolation {
            role: self.role,
            tool_name: tool_name.to_string(),
            reason: format!(
                "tool '{}' is not available for role '{}'",
                tool_name, self.role
            ),
        })
    }

    /// Format as a tool list for prompt injection.
    pub fn format_for_prompt(&self) -> String {
        let mut out = format!("Available tools for {} role:\n\n", self.role);
        for tool in self.tools.values() {
            let mode = if tool.read_only { " (read-only)" } else { "" };
            out.push_str(&format!("- {}{}: {}\n", tool.name, mode, tool.description));
        }
        out
    }
}

/// A violation when a tool is not available in a bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BundleViolation {
    /// Role that attempted the access.
    pub role: AgentRole,
    /// Tool that was requested.
    pub tool_name: String,
    /// Reason for denial.
    pub reason: String,
}

impl std::fmt::Display for BundleViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "bundle violation: {}", self.reason)
    }
}

impl std::error::Error for BundleViolation {}

/// Registry of all known tools and their categories.
///
/// Format: (name, category, description, inherently_read_only)
const ALL_TOOLS: &[(&str, &ToolCategory, &str, &bool)] = &[
    // File mutation tools
    (
        "file_write",
        &ToolCategory::FileMutation,
        "Write or edit files",
        &false,
    ),
    (
        "file_delete",
        &ToolCategory::FileMutation,
        "Delete files",
        &false,
    ),
    // File read tools
    (
        "file_read",
        &ToolCategory::FileRead,
        "Read file contents",
        &true,
    ),
    (
        "file_search",
        &ToolCategory::FileRead,
        "Search file contents with patterns",
        &true,
    ),
    (
        "file_glob",
        &ToolCategory::FileRead,
        "Find files by glob pattern",
        &true,
    ),
    // Shell execution
    (
        "shell_exec",
        &ToolCategory::ShellExec,
        "Execute shell commands",
        &false,
    ),
    // Verifier tools
    (
        "verifier_gate",
        &ToolCategory::Verifier,
        "Run quality gate pipeline (fmt, clippy, check, test)",
        &true,
    ),
    // AST analysis tools
    (
        "ast_grep",
        &ToolCategory::AstAnalysis,
        "Structural code search via ast-grep patterns",
        &true,
    ),
    (
        "rule_pack_scan",
        &ToolCategory::AstAnalysis,
        "Scan code against rule packs (safety, performance, style)",
        &true,
    ),
    // Dependency analysis tools
    (
        "graph_rag_query",
        &ToolCategory::DependencyAnalysis,
        "Query dependency graph for callers, callees, impact",
        &true,
    ),
    (
        "dependency_check",
        &ToolCategory::DependencyAnalysis,
        "Analyze dependency impact of file changes",
        &true,
    ),
    // Knowledge query tools
    (
        "knowledge_query",
        &ToolCategory::KnowledgeQuery,
        "Query NotebookLM knowledge base",
        &true,
    ),
    // Delegation tools
    (
        "delegate_task",
        &ToolCategory::Delegation,
        "Delegate work to another agent",
        &false,
    ),
    // Git operations
    (
        "git_commit",
        &ToolCategory::GitOps,
        "Create git commit",
        &false,
    ),
    (
        "git_branch",
        &ToolCategory::GitOps,
        "Create or switch git branch",
        &false,
    ),
    ("git_diff", &ToolCategory::GitOps, "View git diff", &true),
    // Issue tracking
    (
        "issue_read",
        &ToolCategory::IssueTracking,
        "Read beads issue details",
        &true,
    ),
    (
        "issue_update",
        &ToolCategory::IssueTracking,
        "Update beads issue status",
        &false,
    ),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coder_bundle_has_mutation() {
        let bundle = ToolBundle::for_role(AgentRole::Coder);
        assert!(bundle.has_tool("file_write"));
        assert!(bundle.has_tool("file_read"));
        assert!(bundle.has_tool("shell_exec"));
        assert!(bundle.has_tool("git_commit"));
    }

    #[test]
    fn test_coder_bundle_denies_reviewer_tools() {
        let bundle = ToolBundle::for_role(AgentRole::Coder);
        assert!(!bundle.has_tool("verifier_gate"));
        assert!(!bundle.has_tool("ast_grep"));
        assert!(!bundle.has_tool("rule_pack_scan"));
        assert!(!bundle.has_tool("delegate_task"));
    }

    #[test]
    fn test_reviewer_bundle_has_analysis() {
        let bundle = ToolBundle::for_role(AgentRole::Reviewer);
        assert!(bundle.has_tool("verifier_gate"));
        assert!(bundle.has_tool("ast_grep"));
        assert!(bundle.has_tool("rule_pack_scan"));
        assert!(bundle.has_tool("graph_rag_query"));
        assert!(bundle.has_tool("knowledge_query"));
        assert!(bundle.has_tool("file_read"));
    }

    #[test]
    fn test_reviewer_bundle_denies_mutation() {
        let bundle = ToolBundle::for_role(AgentRole::Reviewer);
        assert!(!bundle.has_tool("file_write"));
        assert!(!bundle.has_tool("shell_exec"));
        assert!(!bundle.has_tool("git_commit"));
        assert!(!bundle.has_tool("delegate_task"));
    }

    #[test]
    fn test_manager_bundle_has_delegation() {
        let bundle = ToolBundle::for_role(AgentRole::Manager);
        assert!(bundle.has_tool("delegate_task"));
        assert!(bundle.has_tool("issue_update"));
        assert!(bundle.has_tool("git_commit"));
        assert!(bundle.has_tool("knowledge_query"));
    }

    #[test]
    fn test_manager_bundle_denies_mutation() {
        let bundle = ToolBundle::for_role(AgentRole::Manager);
        assert!(!bundle.has_tool("file_write"));
        assert!(!bundle.has_tool("shell_exec"));
    }

    #[test]
    fn test_reasoner_bundle_readonly() {
        let bundle = ToolBundle::for_role(AgentRole::Reasoner);
        assert!(bundle.has_tool("file_read"));
        assert!(bundle.has_tool("ast_grep"));
        assert!(bundle.has_tool("knowledge_query"));
        assert!(!bundle.has_tool("file_write"));
        assert!(!bundle.has_tool("shell_exec"));
        assert!(!bundle.has_tool("delegate_task"));

        // All tools should be read-only
        for tool in bundle.tools().values() {
            assert!(
                tool.read_only,
                "Reasoner tool '{}' should be read-only",
                tool.name
            );
        }
    }

    #[test]
    fn test_bundle_validate_request() {
        let bundle = ToolBundle::for_role(AgentRole::Reviewer);
        assert!(bundle.validate_request("ast_grep").is_ok());
        assert!(bundle.validate_request("file_write").is_err());
    }

    #[test]
    fn test_bundle_violation_display() {
        let v = BundleViolation {
            role: AgentRole::Reviewer,
            tool_name: "file_write".to_string(),
            reason: "not available".to_string(),
        };
        assert!(v.to_string().contains("not available"));
    }

    #[test]
    fn test_bundle_format_for_prompt() {
        let bundle = ToolBundle::for_role(AgentRole::Coder);
        let prompt = bundle.format_for_prompt();
        assert!(prompt.contains("coder"));
        assert!(prompt.contains("file_write"));
        assert!(prompt.contains("shell_exec"));
    }

    #[test]
    fn test_bundle_category_filter() {
        let bundle = ToolBundle::for_role(AgentRole::Reviewer);
        let ast_tools = bundle.tools_in_category(ToolCategory::AstAnalysis);
        assert!(ast_tools.len() >= 2); // ast_grep + rule_pack_scan
    }

    #[test]
    fn test_bundle_read_only_vs_mutation() {
        let bundle = ToolBundle::for_role(AgentRole::Coder);
        assert!(!bundle.mutation_tools().is_empty());
        assert!(!bundle.read_only_tools().is_empty());
    }

    #[test]
    fn test_no_role_gets_empty_bundle() {
        for role in AgentRole::all() {
            let bundle = ToolBundle::for_role(*role);
            assert!(!bundle.is_empty(), "Role {:?} has empty tool bundle", role);
        }
    }

    #[test]
    fn test_bundle_serde_roundtrip() {
        let bundle = ToolBundle::for_role(AgentRole::Reviewer);
        let json = serde_json::to_string(&bundle).unwrap();
        let parsed: ToolBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.role, AgentRole::Reviewer);
        assert_eq!(parsed.tools().len(), bundle.tools().len());
    }

    #[test]
    fn test_separation_of_concerns() {
        let coder = ToolBundle::for_role(AgentRole::Coder);
        let reviewer = ToolBundle::for_role(AgentRole::Reviewer);

        // Coder should not have verifier tools
        assert!(!coder.category_permitted(ToolCategory::Verifier));
        // Reviewer should not have mutation tools
        assert!(!reviewer.category_permitted(ToolCategory::FileMutation));
        assert!(!reviewer.category_permitted(ToolCategory::ShellExec));
    }
}
