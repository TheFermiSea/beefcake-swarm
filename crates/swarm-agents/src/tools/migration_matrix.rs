//! Tool inventory and rig_derive migration matrix.
//!
//! Classifies each tool by safety category, determines whether it can be
//! migrated to `#[rig_tool]` derive macros, and documents the rationale.
//!
//! # Migration Categories
//!
//! - **DeriveSafe**: Stateless, no security-sensitive logic. Safe to migrate
//!   to `#[rig_tool]` derive macro for reduced boilerplate.
//! - **ManualRequired**: Stateful or has complex construction (e.g., injected
//!   dependencies, sandbox enforcement). Must keep manual `Tool` impl.
//! - **SecuritySensitive**: Handles untrusted input, enforces allowlists, or
//!   performs filesystem mutations with safety guards. Manual impl required
//!   and any changes need security review.

use serde::{Deserialize, Serialize};

/// Migration safety category for a tool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationCategory {
    /// Safe to migrate to `#[rig_tool]` derive macro.
    DeriveSafe,
    /// Requires manual `Tool` impl due to state or complex construction.
    ManualRequired,
    /// Security-sensitive — manual impl required, changes need review.
    SecuritySensitive,
}

/// Whether a tool interacts with the filesystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FilesystemAccess {
    /// No filesystem interaction.
    None,
    /// Reads files/directories only.
    ReadOnly,
    /// Creates, modifies, or deletes files.
    ReadWrite,
}

/// A single entry in the tool migration matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolEntry {
    /// Tool name as registered with Rig (e.g., "read_file").
    pub name: &'static str,
    /// Implementing struct name.
    pub struct_name: &'static str,
    /// Source file (relative to crate root).
    pub source_file: &'static str,
    /// Migration category.
    pub category: MigrationCategory,
    /// Filesystem access pattern.
    pub filesystem: FilesystemAccess,
    /// Whether the tool requires injected state (e.g., working_dir, KnowledgeBase).
    pub requires_state: bool,
    /// Whether the tool enforces sandbox path checks.
    pub sandbox_enforced: bool,
    /// Rationale for the migration classification.
    pub rationale: &'static str,
}

/// Complete migration matrix for all tools in the swarm-agents crate.
///
/// # Usage
///
/// ```rust
/// use swarm_agents::tools::migration_matrix::{TOOL_MATRIX, MigrationCategory};
///
/// let derive_safe: Vec<_> = TOOL_MATRIX.iter()
///     .filter(|t| t.category == MigrationCategory::DeriveSafe)
///     .collect();
/// ```
pub const TOOL_MATRIX: &[ToolEntry] = &[
    // --- Filesystem tools ---
    ToolEntry {
        name: "read_file",
        struct_name: "ReadFileTool",
        source_file: "src/tools/fs_tools.rs",
        category: MigrationCategory::ManualRequired,
        filesystem: FilesystemAccess::ReadOnly,
        requires_state: true,
        sandbox_enforced: true,
        rationale: "Requires working_dir for sandbox enforcement. Sandbox check \
                    is security-critical — path traversal must be caught before \
                    any read. Derive macro cannot enforce this invariant.",
    },
    ToolEntry {
        name: "write_file",
        struct_name: "WriteFileTool",
        source_file: "src/tools/fs_tools.rs",
        category: MigrationCategory::SecuritySensitive,
        filesystem: FilesystemAccess::ReadWrite,
        requires_state: true,
        sandbox_enforced: true,
        rationale: "Creates files and directories. Has blast-radius guard \
                    (rejects >50% file shrinkage). Double-JSON-encoding \
                    unescape logic for Qwen3 compatibility. Sandbox path \
                    validation. All require careful manual control.",
    },
    ToolEntry {
        name: "list_files",
        struct_name: "ListFilesTool",
        source_file: "src/tools/fs_tools.rs",
        category: MigrationCategory::ManualRequired,
        filesystem: FilesystemAccess::ReadOnly,
        requires_state: true,
        sandbox_enforced: true,
        rationale: "Requires working_dir for sandbox enforcement. Filters \
                    hidden files and target/ directories. Sandbox check is \
                    security-critical.",
    },
    ToolEntry {
        name: "edit_file",
        struct_name: "EditFileTool",
        source_file: "src/tools/patch_tool.rs",
        category: MigrationCategory::SecuritySensitive,
        filesystem: FilesystemAccess::ReadWrite,
        requires_state: true,
        sandbox_enforced: true,
        rationale: "Complex search/replace logic with fuzzy whitespace matching \
                    fallback. Blast-radius warning (>50% shrink). Double-JSON \
                    unescape. Single-replacement-per-call limit prevents mass \
                    edits. Security-sensitive mutation path.",
    },
    // --- Execution tools ---
    ToolEntry {
        name: "run_command",
        struct_name: "RunCommandTool",
        source_file: "src/tools/exec_tool.rs",
        category: MigrationCategory::SecuritySensitive,
        filesystem: FilesystemAccess::ReadWrite,
        requires_state: true,
        sandbox_enforced: true,
        rationale: "Executes shell commands. Allowlist enforcement prevents \
                    arbitrary execution. Shell metacharacter blocking prevents \
                    injection. Timeout enforcement. Direct execution (no shell) \
                    is a critical security boundary.",
    },
    ToolEntry {
        name: "run_verifier",
        struct_name: "RunVerifierTool",
        source_file: "src/tools/verifier_tool.rs",
        category: MigrationCategory::ManualRequired,
        filesystem: FilesystemAccess::ReadOnly,
        requires_state: true,
        sandbox_enforced: true,
        rationale: "Runs cargo quality gates via coordination::verifier. \
                    Requires working_dir and optional package filter list. \
                    Structured output parsing (VerifierReport → JSON). \
                    Mode-dependent gate selection (quick/compile/full).",
    },
    // --- External service tools ---
    ToolEntry {
        name: "query_notebook",
        struct_name: "QueryNotebookTool",
        source_file: "src/tools/notebook_tool.rs",
        category: MigrationCategory::ManualRequired,
        filesystem: FilesystemAccess::None,
        requires_state: true,
        sandbox_enforced: false,
        rationale: "Requires injected Arc<dyn KnowledgeBase> dependency. \
                    Graceful error degradation (returns Ok with error message \
                    instead of propagating errors). Role-based notebook routing. \
                    Derive macro cannot express the dependency injection pattern.",
    },
];

/// Number of tools that can safely migrate to `#[rig_tool]`.
pub const DERIVE_SAFE_COUNT: usize = count_by_category(MigrationCategory::DeriveSafe);

/// Number of tools requiring manual `Tool` impl.
pub const MANUAL_REQUIRED_COUNT: usize = count_by_category(MigrationCategory::ManualRequired);

/// Number of security-sensitive tools.
pub const SECURITY_SENSITIVE_COUNT: usize = count_by_category(MigrationCategory::SecuritySensitive);

/// Count tools by migration category (const fn for compile-time evaluation).
const fn count_by_category(cat: MigrationCategory) -> usize {
    let mut count = 0;
    let mut i = 0;
    while i < TOOL_MATRIX.len() {
        if matches!(
            (&TOOL_MATRIX[i].category, &cat),
            (MigrationCategory::DeriveSafe, MigrationCategory::DeriveSafe)
                | (
                    MigrationCategory::ManualRequired,
                    MigrationCategory::ManualRequired
                )
                | (
                    MigrationCategory::SecuritySensitive,
                    MigrationCategory::SecuritySensitive
                )
        ) {
            count += 1;
        }
        i += 1;
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_covers_all_tools() {
        // 7 unique tool implementations (excluding proxy wrappers which
        // delegate entirely to the base tools)
        assert_eq!(
            TOOL_MATRIX.len(),
            7,
            "Matrix should cover all 7 base tool implementations"
        );
    }

    #[test]
    fn test_category_counts_sum() {
        assert_eq!(
            DERIVE_SAFE_COUNT + MANUAL_REQUIRED_COUNT + SECURITY_SENSITIVE_COUNT,
            TOOL_MATRIX.len(),
            "Category counts should sum to total tools"
        );
    }

    #[test]
    fn test_no_derive_safe_tools_currently() {
        // All current tools require state (working_dir or KnowledgeBase),
        // so none are derive-safe. This test documents the current state
        // and will need updating if truly stateless tools are added.
        assert_eq!(
            DERIVE_SAFE_COUNT, 0,
            "No tools are currently derive-safe (all require injected state)"
        );
    }

    #[test]
    fn test_security_sensitive_tools() {
        let security: Vec<_> = TOOL_MATRIX
            .iter()
            .filter(|t| t.category == MigrationCategory::SecuritySensitive)
            .map(|t| t.name)
            .collect();

        // write_file, edit_file, run_command are security-sensitive
        assert!(security.contains(&"write_file"));
        assert!(security.contains(&"edit_file"));
        assert!(security.contains(&"run_command"));
        assert_eq!(security.len(), 3);
    }

    #[test]
    fn test_all_stateful_tools_have_state_flag() {
        for entry in TOOL_MATRIX {
            assert!(
                entry.requires_state,
                "Tool {} should require state (working_dir or KnowledgeBase)",
                entry.name
            );
        }
    }

    #[test]
    fn test_sandbox_enforcement() {
        // All filesystem tools must enforce sandbox
        for entry in TOOL_MATRIX {
            if entry.filesystem != FilesystemAccess::None {
                assert!(
                    entry.sandbox_enforced,
                    "Filesystem tool {} must enforce sandbox",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn test_matrix_serialization() {
        let json = serde_json::to_string_pretty(TOOL_MATRIX).unwrap();
        assert!(json.contains("read_file"));
        assert!(json.contains("security_sensitive"));
        assert!(json.contains("manual_required"));

        // Verify it parses as valid JSON array
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        let arr = parsed.as_array().expect("should be JSON array");
        assert_eq!(arr.len(), TOOL_MATRIX.len());
    }

    #[test]
    fn test_unique_tool_names() {
        let mut names: Vec<&str> = TOOL_MATRIX.iter().map(|t| t.name).collect();
        let count = names.len();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), count, "Tool names must be unique");
    }

    #[test]
    fn test_proxy_wrappers_excluded() {
        // Proxy wrappers (proxy_read_file, etc.) are excluded because they
        // delegate entirely to base tools — migrating the base migrates the proxy.
        for entry in TOOL_MATRIX {
            assert!(
                !entry.name.starts_with("proxy_"),
                "Proxy wrappers should not be in the matrix: {}",
                entry.name
            );
        }
    }
}
