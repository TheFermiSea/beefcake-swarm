---
name: security-audit
description: Run a security and quality audit on Rust code using AST-grep structural rules. Use when checking code quality, reviewing changes, or before deploying to the swarm.
---

# Security & Quality Audit

Run the project's ast-grep rules to catch safety, security, and quality issues.

## Quick Scan

```bash
# Full scan — all rules against all configured files
sg scan

# JSON output for scripting
sg scan --json 2>/dev/null | python3 -c "
import sys, json
counts = {}
for line in sys.stdin:
    try:
        obj = json.loads(line.strip())
        rid = obj.get('ruleId', 'unknown')
        counts[rid] = counts.get(rid, 0) + 1
    except: pass
for k, v in sorted(counts.items(), key=lambda x: -x[1]):
    print(f'{v:4d}  {k}')
"
```

## Available Rules

| Rule | Severity | What it catches |
|------|----------|-----------------|
| `no-unwrap-in-prod` | warning | `.unwrap()` in non-test code |
| `no-panic-in-prod` | error | `panic!()` in production code |
| `no-todo-macro` | warning | `todo!()` / `unimplemented!()` |
| `no-println-in-prod` | warning | `println!()` / `eprintln!()` instead of tracing |
| `no-std-thread-sleep` | error | Blocking `std::thread::sleep` in async code |
| `no-blocking-stdin` | error | `std::io::stdin()` blocking async runtime |
| `no-unsafe-blocks` | error | Any `unsafe` block |
| `no-process-command-unchecked` | warning | Direct `Command::new()` outside sandbox |
| `tool-missing-sandbox-check` | error | Tool `call()` without `sandbox_check()` |
| `no-bare-expect` | hint | Vague `.expect("failed")` messages |
| `no-silent-error-drop` | warning | `let _ = fallible_op()` hiding errors |

## Targeted Scans

```bash
# Security only
sg scan --rule rules/security.yml

# Tool safety only
sg scan --rule rules/tool-quality.yml

# Async correctness only
sg scan --rule rules/async-safety.yml

# Error handling only
sg scan --rule rules/error-handling.yml
```

## Ad-hoc Pattern Searches

```bash
# Find all .unwrap() calls in a specific file
sg run --pattern '$EXPR.unwrap()' --lang rust crates/swarm-agents/src/orchestrator.rs

# Find all panic! macros
sg run --pattern 'panic!($$$)' --lang rust coordination/src/

# Find unsafe blocks
sg run --pattern 'unsafe { $$$BODY }' --lang rust .

# Find functions that take &self but could take &mut self
sg run --pattern 'fn $NAME(&self, $$$ARGS)' --lang rust crates/
```

## Pre-commit Check

Run before committing to catch issues early:

```bash
# Quick: just errors
sg scan 2>&1 | grep -c 'error\['
# If > 0, fix before committing

# Full report
sg scan
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
```

## Interpreting Results

- **error**: Must fix before merge. These indicate real safety/correctness issues.
- **warning**: Should fix. Technical debt that makes the codebase harder to maintain.
- **hint**: Nice to fix. Suggestions for better practices.

Files in `**/tests/**` and `**/*test*` are excluded from most rules — `.unwrap()` is fine in tests.
