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
//!
//! # Guardrails
//!
//! The [`validate_no_unsafe_derive_migration`] function scans tool source files
//! for `#[rig_tool]` attributes and fails if any `ManualRequired` or
//! `SecuritySensitive` tool has been accidentally annotated. This is enforced
//! in the test suite to catch unsafe migrations in CI.
//!
//! # Policy: Stateful Tool Wrapper Requirements
//!
//! All tools that interact with the filesystem, execute commands, or depend
//! on injected state MUST use manual `impl Tool for T` blocks. This policy
//! exists because:
//!
//! 1. **Sandbox enforcement**: Path traversal checks must run before any I/O.
//!    The derive macro cannot inject pre-call validation.
//! 2. **Blast-radius guards**: Write tools reject changes that shrink files >50%.
//!    This safety logic lives in the `call()` method body.
//! 3. **Command allowlisting**: `run_command` blocks shell metacharacters and
//!    restricts executables to a curated list. Derive cannot express this.
//! 4. **Error resilience**: `query_notebook` returns `Ok(error_message)` instead
//!    of propagating errors, keeping agents functional when KB is down.
//! 5. **State injection**: All tools receive `working_dir` or `Arc<dyn KnowledgeBase>`
//!    at construction time. The derive macro generates `Default`-constructible
//!    tools, which cannot satisfy this requirement.
//!
//! To add a new derive-safe tool, it must:
//! - Take no construction-time state (no `working_dir`, no injected deps)
//! - Perform no filesystem I/O
//! - Have no security-sensitive validation logic
//! - Be added to [`TOOL_MATRIX`] with `MigrationCategory::DeriveSafe`

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

/// Scan tool source files for `#[rig_tool]` attributes and return any that
/// appear on tools classified as `ManualRequired` or `SecuritySensitive`.
///
/// Returns a list of violation descriptions. Empty = all clear.
///
/// This function reads the actual source files at test time to detect if
/// someone adds `#[rig_tool]` to a tool that shouldn't use it.
pub fn validate_no_unsafe_derive_migration() -> Vec<String> {
    let mut violations = Vec::new();

    // Find the crate root by walking up from CARGO_MANIFEST_DIR or using env
    let crate_root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    for entry in TOOL_MATRIX {
        if entry.category == MigrationCategory::DeriveSafe {
            continue; // Derive-safe tools are allowed to use #[rig_tool]
        }

        let source_path = crate_root.join(entry.source_file);
        let content = match std::fs::read_to_string(&source_path) {
            Ok(c) => c,
            Err(_) => continue, // File not found — skip (might be in a different layout)
        };

        // Look for #[rig_tool] attribute near the struct definition
        // We search for the pattern: #[rig_tool] followed by the struct name
        if content.contains("#[rig_tool]") || content.contains("#[rig::tool]") {
            // Check if this specific struct is annotated
            let struct_pattern = format!("struct {}", entry.struct_name);
            if let Some(struct_pos) = content.find(&struct_pattern) {
                // Look backwards from the struct for rig_tool attribute (within 200 chars)
                let search_start = struct_pos.saturating_sub(200);
                let preceding = &content[search_start..struct_pos];
                if preceding.contains("#[rig_tool]") || preceding.contains("#[rig::tool]") {
                    violations.push(format!(
                        "  - {} ({}) in {} is {:?} but has #[rig_tool] attribute. \
                         Remove derive and use manual Tool impl.",
                        entry.struct_name, entry.name, entry.source_file, entry.category
                    ));
                }
            }
        }
    }

    violations
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

    #[test]
    fn test_guardrail_no_unsafe_derive_migration() {
        // Scan tool source files for `#[rig_tool]` attributes on tools classified
        // as ManualRequired or SecuritySensitive. This catches accidental derive
        // migration that bypasses the safety review process.
        let violations = validate_no_unsafe_derive_migration();
        assert!(
            violations.is_empty(),
            "Unsafe derive migration detected! The following tools have \
             #[rig_tool] but are classified as ManualRequired or SecuritySensitive:\n{}",
            violations.join("\n")
        );
    }

    #[test]
    fn test_all_tools_have_rationale() {
        for entry in TOOL_MATRIX {
            assert!(
                !entry.rationale.is_empty(),
                "Tool {} must have a rationale for its classification",
                entry.name
            );
            // Rationale should be substantive (>20 chars)
            assert!(
                entry.rationale.len() > 20,
                "Tool {} rationale is too short: '{}'",
                entry.name,
                entry.rationale
            );
        }
    }

    #[test]
    fn test_security_sensitive_must_have_sandbox_or_allowlist() {
        // SecuritySensitive tools must enforce either sandbox or have
        // explicit security rationale mentioning allowlist/guard.
        for entry in TOOL_MATRIX {
            if entry.category == MigrationCategory::SecuritySensitive {
                let has_security_enforcement = entry.sandbox_enforced
                    || entry.rationale.contains("allowlist")
                    || entry.rationale.contains("guard");
                assert!(
                    has_security_enforcement,
                    "SecuritySensitive tool {} must have sandbox enforcement \
                     or mention allowlist/guard in rationale",
                    entry.name
                );
            }
        }
    }
}
