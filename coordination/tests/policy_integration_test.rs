//! Policy test suite — integration tests for agent profile access control.
//!
//! Enforces anti-pattern policies (no unwrap, no println, no unsafe) and
//! verifies role-based tool access boundaries in the reviewer/verifier loop.

use coordination::agent_profile::{
    AccessLevel, AccessViolation, AgentRole, CapabilityMatrix, PromptContract, ToolCategory,
    ViolationKind,
};

// ── All tool categories for enumeration ───────────────────────────────

const ALL_TOOL_CATEGORIES: &[ToolCategory] = &[
    ToolCategory::FileMutation,
    ToolCategory::FileRead,
    ToolCategory::ShellExec,
    ToolCategory::Verifier,
    ToolCategory::AstAnalysis,
    ToolCategory::DependencyAnalysis,
    ToolCategory::KnowledgeQuery,
    ToolCategory::Delegation,
    ToolCategory::GitOps,
    ToolCategory::IssueTracking,
];

// ── Capability Matrix: cross-role boundary enforcement ─────────────

#[test]
fn policy_coder_cannot_access_verifier() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Coder, ToolCategory::Verifier);
    assert_eq!(access, AccessLevel::Denied);
}

#[test]
fn policy_coder_cannot_access_ast_analysis() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Coder, ToolCategory::AstAnalysis);
    assert_eq!(access, AccessLevel::Denied);
}

#[test]
fn policy_coder_cannot_delegate() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Coder, ToolCategory::Delegation);
    assert_eq!(access, AccessLevel::Denied);
}

#[test]
fn policy_coder_has_mutation_access() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Coder, ToolCategory::FileMutation);
    assert_eq!(access, AccessLevel::Allowed);
}

#[test]
fn policy_coder_has_shell_access() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Coder, ToolCategory::ShellExec);
    assert_eq!(access, AccessLevel::Allowed);
}

#[test]
fn policy_reviewer_cannot_mutate_files() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Reviewer, ToolCategory::FileMutation);
    assert_eq!(access, AccessLevel::Denied);
}

#[test]
fn policy_reviewer_cannot_execute_shell() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Reviewer, ToolCategory::ShellExec);
    assert_eq!(access, AccessLevel::Denied);
}

#[test]
fn policy_reviewer_cannot_use_git() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Reviewer, ToolCategory::GitOps);
    assert_eq!(access, AccessLevel::Denied);
}

#[test]
fn policy_reviewer_has_verifier_access() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Reviewer, ToolCategory::Verifier);
    assert_eq!(access, AccessLevel::Allowed);
}

#[test]
fn policy_reviewer_has_ast_access() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Reviewer, ToolCategory::AstAnalysis);
    assert_eq!(access, AccessLevel::Allowed);
}

#[test]
fn policy_manager_cannot_mutate_files() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Manager, ToolCategory::FileMutation);
    assert_eq!(access, AccessLevel::Denied);
}

#[test]
fn policy_manager_cannot_execute_shell() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Manager, ToolCategory::ShellExec);
    assert_eq!(access, AccessLevel::Denied);
}

#[test]
fn policy_manager_can_delegate() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Manager, ToolCategory::Delegation);
    assert_eq!(access, AccessLevel::Allowed);
}

#[test]
fn policy_manager_has_readonly_verifier() {
    let matrix = CapabilityMatrix::default_matrix();
    let access = matrix.access_level(AgentRole::Manager, ToolCategory::Verifier);
    assert_eq!(access, AccessLevel::ReadOnly);
}

#[test]
fn policy_reasoner_is_readonly() {
    let matrix = CapabilityMatrix::default_matrix();

    // Reasoner should not have mutation, shell, git, or delegation access
    assert_eq!(
        matrix.access_level(AgentRole::Reasoner, ToolCategory::FileMutation),
        AccessLevel::Denied
    );
    assert_eq!(
        matrix.access_level(AgentRole::Reasoner, ToolCategory::ShellExec),
        AccessLevel::Denied
    );
    assert_eq!(
        matrix.access_level(AgentRole::Reasoner, ToolCategory::GitOps),
        AccessLevel::Denied
    );
    assert_eq!(
        matrix.access_level(AgentRole::Reasoner, ToolCategory::Delegation),
        AccessLevel::Denied
    );

    // But can read and analyze
    assert_eq!(
        matrix.access_level(AgentRole::Reasoner, ToolCategory::FileRead),
        AccessLevel::Allowed
    );
    assert_eq!(
        matrix.access_level(AgentRole::Reasoner, ToolCategory::AstAnalysis),
        AccessLevel::Allowed
    );
}

// ── Complete matrix coverage ───────────────────────────────────────

#[test]
fn policy_every_role_tool_combination_defined() {
    let matrix = CapabilityMatrix::default_matrix();

    for role in AgentRole::all() {
        for tool in ALL_TOOL_CATEGORIES {
            let access = matrix.access_level(*role, *tool);
            // Every combination should return a defined level, not panic
            assert!(
                matches!(
                    access,
                    AccessLevel::Allowed | AccessLevel::ReadOnly | AccessLevel::Denied
                ),
                "Undefined access for {:?} + {:?}",
                role,
                tool
            );
        }
    }
}

// ── Symmetry: no role has both mutation and verifier access ────────

#[test]
fn policy_no_role_has_mutation_and_verifier() {
    let matrix = CapabilityMatrix::default_matrix();

    for role in AgentRole::all() {
        let has_mutation =
            matrix.access_level(*role, ToolCategory::FileMutation) == AccessLevel::Allowed;
        let has_verifier =
            matrix.access_level(*role, ToolCategory::Verifier) == AccessLevel::Allowed;

        assert!(
            !(has_mutation && has_verifier),
            "Role {:?} has both mutation AND verifier access — violates separation of concerns",
            role
        );
    }
}

// ── validate_access returns Result correctly ──────────────────────

#[test]
fn policy_validate_access_returns_ok_for_permitted() {
    let matrix = CapabilityMatrix::default_matrix();

    // Coder + FileMutation → Ok(Allowed)
    let result = matrix.validate_access(AgentRole::Coder, ToolCategory::FileMutation);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), AccessLevel::Allowed);

    // Manager + Verifier → Ok(ReadOnly) (ReadOnly is still permitted)
    let result = matrix.validate_access(AgentRole::Manager, ToolCategory::Verifier);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), AccessLevel::ReadOnly);
}

#[test]
fn policy_validate_access_returns_err_for_denied() {
    let matrix = CapabilityMatrix::default_matrix();

    // Reviewer + FileMutation → Err(AccessViolation)
    let result = matrix.validate_access(AgentRole::Reviewer, ToolCategory::FileMutation);
    assert!(result.is_err());

    let violation = result.unwrap_err();
    assert_eq!(violation.role, AgentRole::Reviewer);
    assert_eq!(violation.category, ToolCategory::FileMutation);
    assert_eq!(violation.attempted, AccessLevel::Denied);
    assert!(!violation.rationale.is_empty());
}

// ── Prompt contract validation ─────────────────────────────────────

#[test]
fn policy_coder_prompt_contract() {
    let contract = PromptContract::coder();

    // A valid coder prompt mentions required tools
    let valid = "You are a coder. Use file_write and shell_exec to implement the feature.";
    let violations = contract.validate_prompt(valid);
    assert!(
        violations.is_empty(),
        "Valid coder prompt rejected: {:?}",
        violations
    );

    // A prompt that mentions verifier tools should fail
    let invalid = "You are a coder. Use file_write, shell_exec, and verifier_gate to check.";
    let violations = contract.validate_prompt(invalid);
    assert!(
        violations
            .iter()
            .any(|v| v.kind == ViolationKind::ForbiddenToolMentioned),
        "Coder prompt with verifier tool should be rejected"
    );
}

#[test]
fn policy_reviewer_prompt_contract() {
    let contract = PromptContract::reviewer();

    // A valid reviewer prompt mentions required tools
    let valid = "You are a reviewer. Use verifier_gate and file_read to analyze the code.";
    let violations = contract.validate_prompt(valid);
    assert!(
        violations.is_empty(),
        "Valid reviewer prompt rejected: {:?}",
        violations
    );

    // A prompt that mentions file mutation should fail
    let invalid = "You are a reviewer. Use verifier_gate, file_read, and file_write to fix.";
    let violations = contract.validate_prompt(invalid);
    assert!(
        violations
            .iter()
            .any(|v| v.kind == ViolationKind::ForbiddenToolMentioned),
        "Reviewer prompt with file_write should be rejected"
    );
}

#[test]
fn policy_prompt_missing_required_tool() {
    let contract = PromptContract::coder();

    // Missing shell_exec from required tools
    let prompt = "You are a coder. Use file_write to implement.";
    let violations = contract.validate_prompt(prompt);
    assert!(
        violations
            .iter()
            .any(|v| v.kind == ViolationKind::RequiredToolMissing),
        "Prompt missing required tool should be rejected"
    );
}

// ── Access violation reporting ─────────────────────────────────────

#[test]
fn policy_violation_has_context() {
    let violation = AccessViolation {
        role: AgentRole::Reviewer,
        category: ToolCategory::FileMutation,
        attempted: AccessLevel::Denied,
        rationale: "Reviewer must not mutate code".to_string(),
    };

    assert_eq!(violation.role, AgentRole::Reviewer);
    assert_eq!(violation.category, ToolCategory::FileMutation);
    assert_eq!(violation.attempted, AccessLevel::Denied);
    assert!(violation.rationale.contains("must not mutate"));
}

// ── Summary table smoke test ───────────────────────────────────────

#[test]
fn policy_summary_table_contains_all_roles() {
    let matrix = CapabilityMatrix::default_matrix();
    let table = matrix.summary_table();

    // Table is BTreeMap<AgentRole, BTreeMap<ToolCategory, AccessLevel>>
    assert!(table.contains_key(&AgentRole::Coder), "Table missing coder");
    assert!(
        table.contains_key(&AgentRole::Reviewer),
        "Table missing reviewer"
    );
    assert!(
        table.contains_key(&AgentRole::Manager),
        "Table missing manager"
    );
    assert!(
        table.contains_key(&AgentRole::Reasoner),
        "Table missing reasoner"
    );
    assert_eq!(table.len(), 4);

    // Each role should have all 10 categories
    for role in AgentRole::all() {
        assert_eq!(
            table[role].len(),
            10,
            "Role {:?} should have 10 category entries",
            role
        );
    }
}
