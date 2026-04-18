---
name: ast-grep
description: Guide for writing ast-grep rules to perform structural code search and analysis. Use when users need to search codebases using Abstract Syntax Tree (AST) patterns, find specific code structures, or perform complex code queries that go beyond simple text search.
---

# ast-grep Code Search

Vendored from https://github.com/ast-grep/agent-skill (MIT license).

## Overview

This skill helps translate natural language queries into ast-grep rules for structural code search. ast-grep uses Abstract Syntax Tree (AST) patterns to match code based on its structure rather than just text, enabling powerful and precise code search across large codebases.

## When to Use This Skill

Use this skill when users:
- Need to search for code patterns using structural matching (e.g., "find all async functions that don't have error handling")
- Want to locate specific language constructs (e.g., "find all function calls with specific parameters")
- Request searches that require understanding code structure rather than just text
- Ask to search for code with particular AST characteristics
- Need to perform complex code queries that traditional text search cannot handle

## Project Context

This repo has ast-grep configured:
- `sgconfig.yml` points to `rules/` directory
- `rules/` contains 7 rule files for Rust safety, security, and quality
- Run `sg scan` to check all rules, or `sg scan --rule rules/<file>.yml` for specific rules

## General Workflow

Follow this process to help users write effective ast-grep rules:

### Step 1: Understand the Query

Clearly understand what the user wants to find. Ask clarifying questions if needed:
- What specific code pattern or structure are they looking for?
- Which programming language? (This repo is Rust)
- Are there specific edge cases or variations to consider?
- What should be included or excluded from matches?

### Step 2: Create Example Code

Write a simple code snippet that represents what the user wants to match. Save this to a temporary file for testing.

**Example (Rust):**
```rust
// test_example.rs
async fn example() -> Result<(), Error> {
    let data = fetch_data().await?;
    Ok(data)
}
```

### Step 3: Write the ast-grep Rule

Translate the pattern into an ast-grep rule. Start simple and add complexity as needed.

**Key principles:**
- Always use `stopBy: end` for relational rules (`inside`, `has`) to ensure search goes to the end of the direction
- Use `pattern` for simple structures
- Use `kind` with `has`/`inside` for complex structures
- Break complex queries into smaller sub-rules using `all`, `any`, or `not`

**Example rule file (test_rule.yml):**
```yaml
id: async-with-await
language: Rust
rule:
  kind: function_item
  has:
    pattern: $EXPR.await
    stopBy: end
```

### Step 4: Test the Rule

Use ast-grep CLI to verify the rule matches the example code. There are two main approaches:

**Option A: Test with inline rules (for quick iterations)**
```bash
echo 'async fn test() { let x = foo().await; }' | sg scan --inline-rules "id: test
language: Rust
rule:
  kind: function_item
  has:
    pattern: \$EXPR.await
    stopBy: end" --stdin
```

**Option B: Test with rule files (recommended for complex rules)**
```bash
sg scan --rule test_rule.yml test_example.rs
```

**Debugging if no matches:**
1. Simplify the rule (remove sub-rules)
2. Add `stopBy: end` to relational rules if not present
3. Use `--debug-query` to understand the AST structure (see below)
4. Check if `kind` values are correct for the language

### Step 5: Search the Codebase

Once the rule matches the example code correctly, search the actual codebase:

**For simple pattern searches:**
```bash
sg run --pattern '$EXPR.unwrap()' --lang rust crates/
```

**For complex rule-based searches:**
```bash
sg scan --rule my_rule.yml crates/
```

**For inline rules (without creating files):**
```bash
sg scan --inline-rules "id: my-rule
language: Rust
rule:
  pattern: \$PATTERN" crates/
```

## ast-grep CLI Commands

### Inspect Code Structure (--debug-query)

Dump the AST structure to understand how code is parsed:

```bash
sg run --pattern 'fn example() { }' \
  --lang rust \
  --debug-query=cst
```

**Available formats:**
- `cst`: Concrete Syntax Tree (shows all nodes including punctuation)
- `ast`: Abstract Syntax Tree (shows only named nodes)
- `pattern`: Shows how ast-grep interprets your pattern

### Test Rules (scan with --stdin)

Test a rule against code snippet without creating files:

```bash
echo 'let x = foo().unwrap();' | sg scan --inline-rules "id: test
language: Rust
rule:
  pattern: \$EXPR.unwrap()" --stdin
```

### Search with Patterns (run)

Simple pattern-based search for single AST node matches:

```bash
sg run --pattern '$EXPR.unwrap()' --lang rust .
sg run --pattern 'panic!($$$ARGS)' --lang rust crates/
sg run --pattern 'unsafe { $$$BODY }' --lang rust coordination/
```

### Search with Rules (scan)

```bash
# Run all project rules
sg scan

# Run specific rule file
sg scan --rule rules/security.yml

# JSON output for scripting
sg scan --json
```

## Tips for Writing Effective Rules

### Always Use stopBy: end

For relational rules, always use `stopBy: end` unless there's a specific reason not to:

```yaml
has:
  pattern: $EXPR.unwrap()
  stopBy: end
```

### Start Simple, Then Add Complexity

1. Try a `pattern` first
2. If that doesn't work, try `kind` to match the node type
3. Add relational rules (`has`, `inside`) as needed
4. Combine with composite rules (`all`, `any`, `not`) for complex logic

### Rust-Specific Kind Names

Common Rust AST node kinds:
- `function_item` — `fn foo() {}`
- `impl_item` — `impl Foo {}`
- `struct_item` — `struct Foo {}`
- `enum_item` — `enum Foo {}`
- `trait_item` — `trait Foo {}`
- `call_expression` — `foo()`
- `macro_invocation` — `println!()`
- `unsafe_block` — `unsafe {}`
- `async_block` — `async {}`
- `match_expression` — `match x {}`
- `let_declaration` — `let x = ...;`
- `use_declaration` — `use std::io;`

### Escaping in Inline Rules

When using `--inline-rules`, escape metavariables:
- Use `\$VAR` instead of `$VAR` (shell interprets `$`)
- Or use single quotes around the whole thing

## Common Rust Patterns

### Find functions containing unwrap

```yaml
rule:
  kind: function_item
  has:
    pattern: $EXPR.unwrap()
    stopBy: end
```

### Find async functions missing error propagation

```yaml
rule:
  all:
    - kind: function_item
    - has:
        pattern: $EXPR.await
        stopBy: end
    - not:
        has:
          regex: "\\?"
          stopBy: end
```

### Find impl blocks for a specific trait

```yaml
rule:
  kind: impl_item
  has:
    kind: type_identifier
    regex: "^Tool$"
    stopBy: end
```

### Find struct fields that are pub

```yaml
rule:
  kind: field_declaration
  has:
    regex: "^pub$"
    stopBy: end
```

## Rule File Format

```yaml
id: rule-name          # Unique identifier
language: Rust         # Target language
severity: error        # error, warning, hint
message: "Description" # Shown to user
note: |                # Detailed explanation
  Multi-line help text
rule:                  # The matching rule
  pattern: ...
constraints:           # Metavariable constraints (top-level!)
  VAR:
    regex: "..."
files:                 # File glob patterns to include
  - "crates/**/*.rs"
ignores:               # File glob patterns to exclude
  - "**/tests/**"
```

## Metavariables

- `$VAR` — Single named node (e.g., `$EXPR`, `$NAME`)
- `$$VAR` — Single unnamed node (operators, punctuation)
- `$$$VAR` — Zero or more nodes (variadic)
- `$_` — Wildcard (non-capturing)
