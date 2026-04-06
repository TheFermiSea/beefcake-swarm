# Error Patterns

Common compiler errors encountered by the swarm, grouped by category.
Each category maps to the `ErrorCategory` enum in `coordination/src/feedback/error_parser.rs`.

## Categories

### TypeMismatch

Rustc error codes: E0308, etc. Mismatched types in assignments, return values, or function arguments.

**Typical fixes:** Adjust return type, add `.into()` / `as` cast, wrap in `Ok()`/`Some()`.

**Routing:** Fast tier (GLM-4.7-Flash) handles most cases.

### BorrowChecker

Rustc error codes: E0382, E0502, etc. Use-after-move, conflicting borrows.

**Typical fixes:** Clone the value, restructure scope, switch to `Rc`/`Arc`.

**Routing:** Coder tier (Qwen3.5-27B) for non-trivial cases.

### Lifetime

Rustc error codes: E0106, E0621, etc. Missing or conflicting lifetime annotations.

**Typical fixes:** Add explicit lifetime parameters, restructure to owned types.

**Routing:** Coder tier. Escalates to Reasoning tier if combined with async.

### TraitBound

Rustc error codes: E0277, etc. Missing trait implementations.

**Typical fixes:** Add `#[derive(...)]`, implement the trait, add bounds to generics.

**Routing:** Fast tier for simple derives, Coder tier for manual impls.

### Async

Async/await related errors: Send bounds, pinning issues, lifetime-in-async.

**Typical fixes:** Box the future, add `Send` bound, restructure to avoid holding refs across `.await`.

**Routing:** Reasoning tier (Devstral-24B). These are the hardest for local models.

### Macro

Macro expansion errors. Often from derive macros or proc macros with incorrect input.

**Typical fixes:** Fix macro input syntax, check macro documentation.

**Routing:** Varies by complexity.

### ImportResolution

Missing crate, module, or item. Rustc cannot find the referenced path.

**Typical fixes:** Add dependency to Cargo.toml, fix `use` path, add `pub` visibility.

**Routing:** Fast tier. Usually mechanical.

### Syntax

Parse errors. Malformed Rust syntax.

**Typical fixes:** Fix brackets, semicolons, keyword usage.

**Routing:** Fast tier. Non-retryable if structural.

### Other

Uncategorized errors that do not match the above patterns.
