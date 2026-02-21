//! Cross-environment smoke tests — verify core deterministic flows
//! produce identical results regardless of execution environment
//! (dev machine, container, HPC cluster).
//!
//! These tests exercise the pure-logic subsystems that must behave
//! identically across all target environments. No I/O, no network,
//! no filesystem access — purely in-memory verification.

use coordination::agent_profile::{AccessLevel, AgentRole, CapabilityMatrix, ToolCategory};
use coordination::patch::{MatchKind, PatchConfig, PatchEngine, PatchHunk};
use coordination::reviewer_tools::{RulePack, RulePackRegistry, RuleSeverity};
use coordination::shell_safety::{escape_for_ssh, validate_arg};
use coordination::tool_bundle::ToolBundle;

// ─── Agent Profile / Capability Matrix ───────────────────────────────

#[test]
fn smoke_capability_matrix_deterministic() {
    let m1 = CapabilityMatrix::default_matrix();
    let m2 = CapabilityMatrix::default_matrix();

    // Same matrix must yield identical access decisions
    for role in AgentRole::all() {
        let categories = [
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
        for cat in &categories {
            assert_eq!(
                m1.access_level(*role, *cat),
                m2.access_level(*role, *cat),
                "Matrix non-deterministic for {:?}/{:?}",
                role,
                cat
            );
        }
    }
}

#[test]
fn smoke_role_isolation_invariant() {
    let matrix = CapabilityMatrix::default_matrix();

    // Reviewer must NEVER have file mutation
    assert_eq!(
        matrix.access_level(AgentRole::Reviewer, ToolCategory::FileMutation),
        AccessLevel::Denied
    );

    // Coder must NEVER have delegation
    assert_eq!(
        matrix.access_level(AgentRole::Coder, ToolCategory::Delegation),
        AccessLevel::Denied
    );

    // Reasoner must NEVER have mutation or shell
    assert_eq!(
        matrix.access_level(AgentRole::Reasoner, ToolCategory::FileMutation),
        AccessLevel::Denied
    );
    assert_eq!(
        matrix.access_level(AgentRole::Reasoner, ToolCategory::ShellExec),
        AccessLevel::Denied
    );
}

// ─── Tool Bundle Assembly ────────────────────────────────────────────

#[test]
fn smoke_tool_bundles_deterministic() {
    for role in AgentRole::all() {
        let b1 = ToolBundle::for_role(*role);
        let b2 = ToolBundle::for_role(*role);
        assert_eq!(
            b1.tool_names(),
            b2.tool_names(),
            "Bundle non-deterministic for {:?}",
            role
        );
    }
}

#[test]
fn smoke_every_role_has_tools() {
    for role in AgentRole::all() {
        let bundle = ToolBundle::for_role(*role);
        assert!(
            !bundle.is_empty(),
            "Role {:?} should have at least one tool",
            role
        );
    }
}

#[test]
fn smoke_bundle_prompt_format_stable() {
    let bundle = ToolBundle::for_role(AgentRole::Coder);
    let p1 = bundle.format_for_prompt();
    let p2 = bundle.format_for_prompt();
    assert_eq!(p1, p2, "Prompt format should be stable across calls");
}

#[test]
fn smoke_bundle_validate_request_consistency() {
    let bundle = ToolBundle::for_role(AgentRole::Reviewer);

    // Tools in the bundle should validate OK
    for name in bundle.tool_names() {
        assert!(
            bundle.validate_request(name).is_ok(),
            "Tool '{}' in bundle should validate",
            name
        );
    }

    // A known-absent tool should fail
    assert!(bundle.validate_request("file_write").is_err());
}

// ─── Patch Engine ────────────────────────────────────────────────────

#[test]
fn smoke_patch_exact_deterministic() {
    let engine = PatchEngine::default();
    let content = "fn main() {\n    println!(\"hello\");\n}";
    let hunk = PatchHunk {
        old_lines: vec!["    println!(\"hello\");".to_string()],
        new_lines: vec!["    println!(\"world\");".to_string()],
        description: None,
    };

    let r1 = engine.apply(content, &[hunk.clone()]);
    let r2 = engine.apply(content, &[hunk]);
    assert_eq!(r1.success, r2.success);
    assert_eq!(r1.patched_content, r2.patched_content);
    assert_eq!(r1.hunk_results[0].match_kind, MatchKind::Exact);
}

#[test]
fn smoke_patch_whitespace_normalization() {
    let engine = PatchEngine::default();
    // Content has different whitespace than the hunk
    let content = "fn  foo(  x:  i32  ) {\n    x + 1\n}";
    let hunk = PatchHunk {
        old_lines: vec![
            "fn foo( x: i32 ) {".to_string(),
            "   x + 1".to_string(),
            "}".to_string(),
        ],
        new_lines: vec![
            "fn foo(x: i32) {".to_string(),
            "    x + 1".to_string(),
            "}".to_string(),
        ],
        description: None,
    };

    let result = engine.apply(content, &[hunk]);
    assert!(result.success);
    assert!(
        result.hunk_results[0].match_kind == MatchKind::WhitespaceNormalized
            || result.hunk_results[0].match_kind == MatchKind::TrimmedTrailing
    );
}

#[test]
fn smoke_patch_config_roundtrip() {
    let config = PatchConfig::default();
    let json = serde_json::to_string(&config).unwrap();
    let parsed: PatchConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(config.normalize_whitespace, parsed.normalize_whitespace);
    assert_eq!(config.trim_trailing, parsed.trim_trailing);
    assert_eq!(config.max_context_lines, parsed.max_context_lines);
    assert!((config.min_similarity - parsed.min_similarity).abs() < f64::EPSILON);
}

// ─── Rule Pack Registry ──────────────────────────────────────────────

#[test]
fn smoke_rule_packs_deterministic() {
    let r1 = RulePackRegistry::with_defaults();
    let r2 = RulePackRegistry::with_defaults();

    let names1 = r1.pack_names();
    let names2 = r2.pack_names();
    assert_eq!(names1, names2, "Rule pack registry should be deterministic");
}

#[test]
fn smoke_default_rule_packs_non_empty() {
    let registry = RulePackRegistry::with_defaults();
    let packs = registry.pack_names();
    assert!(
        !packs.is_empty(),
        "Default registry should have at least one pack"
    );

    for name in &packs {
        let pack = registry.get(name).unwrap();
        assert!(
            !pack.rules.is_empty(),
            "Pack '{}' should have at least one rule",
            name
        );
    }
}

#[test]
fn smoke_rule_severity_ordering() {
    // Derived Ord uses discriminant order: Error(0) < Warning(1) < Info(2)
    // Ordering must be consistent across environments
    assert!(RuleSeverity::Error < RuleSeverity::Warning);
    assert!(RuleSeverity::Warning < RuleSeverity::Info);
    assert_ne!(RuleSeverity::Error, RuleSeverity::Info);
}

#[test]
fn smoke_rule_registry_version_hash_stable() {
    let r1 = RulePackRegistry::with_defaults();
    let r2 = RulePackRegistry::with_defaults();
    let h1 = r1.version_hash();
    let h2 = r2.version_hash();
    assert_eq!(h1, h2, "Registry version hash should be stable");
}

#[test]
fn smoke_rule_pack_yaml_roundtrip() {
    let mut pack = RulePack::new("test_pack", "A test rule pack");
    pack.rules
        .push(coordination::reviewer_tools::RulePackEntry::pattern_rule(
            "no_unwrap",
            "Avoid .unwrap() in production code",
            RuleSeverity::Warning,
            "rust",
            "$EXPR.unwrap()",
            "safety",
        ));

    let yaml = RulePackRegistry::export_yaml(&pack).unwrap();
    let loaded = RulePackRegistry::load_yaml(&yaml).unwrap();
    assert_eq!(loaded.name, "test_pack");
    assert_eq!(loaded.rules.len(), 1);
    assert_eq!(loaded.rules[0].rule_id, "no_unwrap");
}

// ─── Shell Safety ────────────────────────────────────────────────────

#[test]
fn smoke_shell_escape_deterministic() {
    let inputs = [
        "simple",
        "with spaces",
        "with'quote",
        "$(injection)",
        "; rm -rf /",
        "normal-arg-123",
    ];

    for input in &inputs {
        let e1 = escape_for_ssh(input);
        let e2 = escape_for_ssh(input);
        assert_eq!(e1, e2, "SSH escape non-deterministic for '{}'", input);
    }
}

#[test]
fn smoke_shell_validate_blocks_injection() {
    let dangerous = ["; rm -rf /", "| cat /etc/passwd", "$(whoami)", "`id`"];

    for input in &dangerous {
        assert!(
            validate_arg(input).is_err(),
            "Should block injection: '{}'",
            input
        );
    }
}

#[test]
fn smoke_shell_validate_allows_safe() {
    let safe = [
        "hello",
        "my-file.rs",
        "path/to/file",
        "coordination",
        "123",
        "flag=value",
    ];

    for input in &safe {
        assert!(
            validate_arg(input).is_ok(),
            "Should allow safe arg: '{}'",
            input
        );
    }
}

// ─── Cross-module Integration ────────────────────────────────────────

#[test]
fn smoke_bundle_matches_matrix() {
    let matrix = CapabilityMatrix::default_matrix();

    for role in AgentRole::all() {
        let bundle = ToolBundle::for_role(*role);

        // If file_write is in the bundle, FileMutation must be permitted
        if bundle.has_tool("file_write") {
            assert!(
                matrix.is_permitted(*role, ToolCategory::FileMutation),
                "{:?} has file_write but FileMutation not permitted",
                role
            );
        }

        // If verifier_gate is in the bundle, Verifier must be permitted
        if bundle.has_tool("verifier_gate") {
            assert!(
                matrix.is_permitted(*role, ToolCategory::Verifier),
                "{:?} has verifier_gate but Verifier not permitted",
                role
            );
        }
    }
}

#[test]
fn smoke_patch_on_realistic_rust_code() {
    let engine = PatchEngine::default();

    let content = r#"use std::collections::HashMap;

fn process(items: &[Item]) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for item in items {
        *counts.entry(item.name.clone()).or_insert(0) += 1;
    }
    counts
}

#[cfg(test)]
mod tests {
    use super::*;
}"#;

    let hunk = PatchHunk {
        old_lines: vec![
            "fn process(items: &[Item]) -> HashMap<String, usize> {".to_string(),
            "    let mut counts = HashMap::new();".to_string(),
            "    for item in items {".to_string(),
            "        *counts.entry(item.name.clone()).or_insert(0) += 1;".to_string(),
            "    }".to_string(),
            "    counts".to_string(),
            "}".to_string(),
        ],
        new_lines: vec![
            "fn process(items: &[Item]) -> HashMap<String, usize> {".to_string(),
            "    items.iter().fold(HashMap::new(), |mut acc, item| {".to_string(),
            "        *acc.entry(item.name.clone()).or_insert(0) += 1;".to_string(),
            "        acc".to_string(),
            "    })".to_string(),
            "}".to_string(),
        ],
        description: Some("Refactor to fold".to_string()),
    };

    let result = engine.apply(content, &[hunk]);
    assert!(result.success, "Patch should apply to realistic Rust code");
    let patched = result.patched_content.unwrap();
    assert!(patched.contains("fold"));
    assert!(!patched.contains("let mut counts"));
    // Surrounding code preserved
    assert!(patched.contains("use std::collections::HashMap;"));
    assert!(patched.contains("#[cfg(test)]"));
}
