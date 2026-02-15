# Rig-Core Manager-Worker Migration Plan

## Executive Summary

Migrate `swarm-agents` from its current text-in/text-out agent loop to a **tool-calling Manager-Worker swarm** built on `rig-core` 0.30. The Manager agent (72B) delegates structured tasks to specialized Workers (Coder, Reviewer, Researcher) via rig's native Agent-as-Tool pattern. The existing `coordination` crate (~21k LOC) stays untouched — its deterministic state machines (verifier, escalation, router, work packets) become rig `Tool` implementations that the agents can invoke.

## Architecture Diagram

```
                        ┌──────────────────────────┐
                        │       Beads Issue DB      │
                        │   (bd ready → pick task)  │
                        └────────────┬─────────────┘
                                     │
                        ┌────────────▼─────────────┐
                        │     ORCHESTRATOR LOOP     │
                        │  (Rust main — not an LLM) │
                        │                           │
                        │  1. Pick issue             │
                        │  2. Create worktree        │
                        │  3. Run Manager agent      │
                        │  4. Evaluate outcome       │
                        │  5. Merge or escalate      │
                        └────────────┬─────────────┘
                                     │
                    ┌────────────────▼────────────────┐
                    │         MANAGER AGENT           │
                    │  (OR1-Behemoth 72B on vasp-01)  │
                    │                                  │
                    │  System prompt: "You are the     │
                    │  Swarm Manager. Break down the   │
                    │  task and delegate to workers."   │
                    │                                  │
                    │  Tools:                           │
                    │  ├── delegate_to_coder            │
                    │  ├── delegate_to_reviewer         │
                    │  ├── run_verifier                 │
                    │  ├── read_file                    │
                    │  ├── list_files                   │
                    │  └── run_command                  │
                    └──┬───────────┬───────────┬──────┘
                       │           │           │
          ┌────────────▼──┐  ┌────▼────────┐  ┌▼─────────────────┐
          │  CODER WORKER │  │  REVIEWER   │  │  VERIFIER TOOL   │
          │  (14B strand) │  │  (14B blind)│  │  (deterministic) │
          │               │  │             │  │                  │
          │  Tools:       │  │  No tools — │  │  cargo fmt       │
          │  ├ write_file │  │  text only  │  │  cargo clippy    │
          │  ├ read_file  │  │             │  │  cargo check     │
          │  └ run_command│  │             │  │  cargo test      │
          └───────────────┘  └─────────────┘  └──────────────────┘
```

### Communication Flow

```
Manager ──tool_call──► delegate_to_coder(task)
                              │
                    Coder Agent runs with its own tools
                    (write_file, read_file, run_command)
                              │
                    Returns code + summary to Manager
                              │
Manager ──tool_call──► run_verifier()
                              │
                    Deterministic: fmt → clippy → check → test
                              │
                    Returns VerifierReport to Manager
                              │
Manager ──tool_call──► delegate_to_reviewer(diff)
                              │
                    Reviewer sees ONLY the diff (blind review)
                              │
                    Returns PASS/FAIL + feedback
```

## Current State Analysis

### What exists (crates/swarm-agents/)

| Component | Status | Notes |
|-----------|--------|-------|
| `main.rs` orchestrator loop | Working | Picks issue, creates worktree, runs loop |
| `implementer.rs` (72B agent) | Working | Pure text completion, no tools |
| `validator.rs` (14B reviewer) | Working | Pure text completion, blind diff review |
| `beads_bridge.rs` | Working | CLI subprocess wrapper for `br` |
| `worktree_bridge.rs` | Working | Git worktree create/merge/cleanup |
| `config.rs` | Working | Hardcoded endpoints, no env vars |
| `apply_implementer_changes()` | **STUB** | Returns error — the critical gap |

### What exists (coordination/)

| Module | LOC | Role in migration |
|--------|-----|-------------------|
| `verifier/` | ~2k | Becomes `RunVerifierTool` |
| `escalation/` | ~2k | Drives tier escalation decisions |
| `router/` | ~1.5k | Task classification for tier routing |
| `work_packet/` | ~1k | Structured context format — keep as-is |
| `feedback/` | ~2k | Error parsing — used by verifier |
| `harness/` | ~5k | Session management — future integration |
| `ensemble/` | ~3k | Multi-model voting — future integration |

## Migration Design

### Phase 1: Tool Infrastructure (this PR)

Define the core tool traits and skeleton. No behavioral changes.

**New files:**
- `src/tools/mod.rs` — Tool module root
- `src/tools/fs_tools.rs` — `ReadFileTool`, `WriteFileTool`, `ListFilesTool`
- `src/tools/exec_tool.rs` — `RunCommandTool` (sandboxed)
- `src/tools/verifier_tool.rs` — `RunVerifierTool` (wraps coordination::verifier)
- `src/tools/delegate.rs` — `DelegateToCoderTool`, `DelegateToReviewerTool`

**Tool trait pattern (rig 0.30):**
```rust
use rig::tool::Tool;
use rig::completion::ToolDefinition;
use serde::Deserialize;

#[derive(Deserialize)]
pub struct ReadFileArgs {
    pub path: String,
}

pub struct ReadFileTool {
    pub working_dir: PathBuf,  // Worktree root — sandbox boundary
}

impl Tool for ReadFileTool {
    const NAME: &'static str = "read_file";
    type Error = ToolError;
    type Args = ReadFileArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: "read_file".into(),
            description: "Read file contents from the workspace".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Relative path within workspace" }
                },
                "required": ["path"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        let full = self.working_dir.join(&args.path).canonicalize()?;
        // Sandbox: must stay within working_dir
        if !full.starts_with(&self.working_dir) {
            return Err(ToolError::Sandbox(args.path));
        }
        Ok(std::fs::read_to_string(&full)?)
    }
}
```

### Phase 2: Worker Agents

Replace `Implementer` and `Validator` with tool-equipped rig agents.

**Coder Worker (replaces Implementer):**
```rust
let coder = fast_client         // strand-14B on vasp-02
    .agent(&config.fast_endpoint.model)
    .preamble(CODER_SYSTEM_PROMPT)
    .tool(ReadFileTool::new(&wt_path))
    .tool(WriteFileTool::new(&wt_path))
    .tool(RunCommandTool::new(&wt_path))
    .temperature(0.2)
    .build();
```

**Reviewer Worker (stays tool-free, blind review):**
```rust
let reviewer = fast_client      // strand-14B on vasp-02
    .agent(&config.fast_endpoint.model)
    .preamble(REVIEWER_SYSTEM_PROMPT)
    .temperature(0.1)
    .build();
```

### Phase 3: Manager Agent (Agent-as-Tool)

The Manager uses the 72B reasoning model and delegates via tool calls.

**Key insight from rig-core:** `Agent<M>` already implements `Tool` natively (see `rig/agent/tool.rs`). You can pass a worker agent directly as a tool to the manager.

```rust
// Worker agents implement Tool automatically
let coder_agent = fast_client.agent(...)
    .name("coder")
    .description("Writes Rust code. Give it specific file paths and what to change.")
    .preamble(CODER_SYSTEM_PROMPT)
    .tool(ReadFileTool::new(&wt_path))
    .tool(WriteFileTool::new(&wt_path))
    .build();

let reviewer_agent = fast_client.agent(...)
    .name("reviewer")
    .description("Reviews diffs for bugs, style, correctness. Returns PASS/FAIL.")
    .preamble(REVIEWER_SYSTEM_PROMPT)
    .build();

// Manager gets workers + deterministic tools
let manager = reasoning_client  // OR1-Behemoth 72B on vasp-01
    .agent(&config.reasoning_endpoint.model)
    .preamble(MANAGER_SYSTEM_PROMPT)
    .tool(coder_agent)                         // Agent-as-Tool
    .tool(reviewer_agent)                      // Agent-as-Tool
    .tool(RunVerifierTool::new(&wt_path))      // Deterministic
    .tool(ReadFileTool::new(&wt_path))         // Direct file access
    .tool(ListFilesTool::new(&wt_path))        // Directory listing
    .build();

// Single prompt drives the entire task
let result = manager.prompt(&formatted_work_packet).await?;
```

### Phase 4: Orchestrator Integration

The Rust `main()` stays as the outer loop — it is NOT an LLM. It:
1. Picks beads issues (deterministic)
2. Creates worktrees (deterministic)
3. Builds and runs the Manager agent (LLM)
4. Evaluates the outcome using `coordination::escalation` (deterministic)
5. Merges or escalates (deterministic)

**Escalation integration:**
```rust
// After manager completes, run verifier one final time
let report = Verifier::new(&wt_path, VerifierConfig::full()).run_pipeline().await;

if report.all_green {
    // Manager succeeded
    worktree_bridge.merge_and_remove(&issue.id)?;
    beads.close(&issue.id, Some("Resolved by swarm manager"))?;
} else {
    // Escalation engine decides next step
    let decision = escalation_engine.decide(&mut escalation_state, &report);
    match decision.action {
        EscalationAction::Retry { .. } => { /* re-run with higher tier */ }
        EscalationAction::Escalate { target_tier, .. } => { /* switch model */ }
        EscalationAction::RequestHuman { .. } => { /* create blocking bead */ }
    }
}
```

## Implementation Steps

### Step 1: Add dependencies to Cargo.toml
```toml
rig-core = "0.30"
schemars = "0.8"
thiserror = "2"
```

### Step 2: Create tool module structure
```
src/
├── main.rs              (modify: use new Manager pattern)
├── config.rs            (modify: add env var support, Coder tier)
├── tools/
│   ├── mod.rs           (new: module exports + ToolError type)
│   ├── fs_tools.rs      (new: ReadFile, WriteFile, ListFiles)
│   ├── exec_tool.rs     (new: RunCommand with sandboxing)
│   ├── verifier_tool.rs (new: wraps coordination::verifier)
│   └── delegate.rs      (new: DelegateToCoder, DelegateToReviewer)
├── agents/
│   ├── mod.rs           (new: module exports)
│   ├── manager.rs       (new: Manager agent builder)
│   ├── coder.rs         (new: Coder worker builder, replaces implementer.rs)
│   └── reviewer.rs      (new: Reviewer worker builder, replaces validator.rs)
├── prompts.rs           (new: system prompt constants)
├── beads_bridge.rs      (keep)
└── worktree_bridge.rs   (keep)
```

### Step 3: Implement tools (bottom-up)
1. `ToolError` enum with `Io`, `Sandbox`, `Command`, `Verifier` variants
2. `ReadFileTool` — sandbox-checked `fs::read_to_string`
3. `WriteFileTool` — sandbox-checked `fs::write` with parent dir creation
4. `ListFilesTool` — `fs::read_dir` with gitignore filtering
5. `RunCommandTool` — `std::process::Command` with timeout + working dir
6. `RunVerifierTool` — wraps `coordination::verifier::Verifier`

### Step 4: Build agents (bottom-up)
1. Coder worker: fast_client + tools + CODER_SYSTEM_PROMPT
2. Reviewer worker: fast_client + no tools + REVIEWER_SYSTEM_PROMPT
3. Manager: reasoning_client + workers-as-tools + deterministic tools

### Step 5: Rewire orchestrator loop
1. Replace `implementer.implement()` + `apply_implementer_changes()` with `manager.prompt()`
2. Keep verifier as final deterministic gate (run AFTER manager finishes)
3. Keep escalation state tracking
4. Keep worktree + beads integration

### Step 6: Config improvements
1. Add env var support: `SWARM_FAST_URL`, `SWARM_REASONING_URL`, etc.
2. Add Coder tier (Qwen3-Coder-Next on vasp-02)
3. Add timeout configuration per tool

## Gotchas & Mitigations

### 1. Tool Calling Reliability on Local Models
**Problem:** llama.cpp tool calling is fragile on <70B models.
**Mitigation:**
- Use grammar-constrained sampling (GBNF) on the llama.cpp server
- Keep tool definitions simple — flat JSON objects, no nesting
- The Coder (14B) gets simple tools (read/write/run)
- The Manager (72B) gets delegation tools that return strings

### 2. Context Window Limits
**Problem:** V100S VRAM limits context. Multi-turn tool loops can exhaust it.
**Mitigation:**
- Token-budget context packing (already in `coordination::context_packer`)
- Limit Coder to 3 tool iterations before returning
- Manager gets a single prompt — not a multi-turn conversation
- Use `max_patch_loc` constraint to keep outputs small

### 3. Agent-as-Tool Naming
**Problem:** rig's native `Agent::NAME = "agent_tool"` for ALL agents — name collisions.
**Mitigation:** Override `fn name(&self)` to return unique names (e.g., "coder", "reviewer"). The `AgentBuilder::name()` method handles this.

### 4. Sandboxing
**Problem:** LLM-driven file writes/command execution is dangerous.
**Mitigation:**
- All file tools validate paths stay within worktree root (canonicalize + starts_with)
- `RunCommandTool` only allows allowlisted commands: `cargo`, `git`, `cat`, `ls`
- Command timeout: 120s default, 300s for `cargo test`
- Worktree isolation means worst case = trashing a disposable branch

### 5. Sequential vs Parallel Workers
**Problem:** Agent-as-Tool is sequential — Manager waits for each worker.
**Mitigation:** Sequential is correct for this use case:
- Coder must finish before Reviewer can review
- Verifier must run after Coder writes files
- Future parallel fan-out can use `tokio::join!` at the orchestrator level for multi-issue processing

## Dependency Map

```
rig-core 0.30 ──────► Agent, Tool, CompletionClient, Prompt
coordination ────────► Verifier, EscalationEngine, ContextPacker, WorkPacket
schemars 0.8 ────────► JsonSchema for tool parameter definitions
thiserror 2 ─────────► ToolError enum
tokio 1 (full) ──────► Async runtime
serde/serde_json 1 ──► Serialization
reqwest 0.12 ────────► HTTP client (used by rig internally)
tracing 0.1 ─────────► Structured logging
anyhow 1 ────────────► Top-level error handling in main
```

## Success Criteria

- [ ] `cargo build --workspace` passes with no errors
- [ ] Manager agent can make at least one tool call to a worker
- [ ] Coder worker can read a file and write a modified version
- [ ] RunVerifierTool returns structured pass/fail
- [ ] All file operations are sandbox-checked (no path traversal)
- [ ] RunCommandTool has allowlist + timeout
- [ ] Existing coordination tests still pass
- [ ] End-to-end: pick beads issue → Manager delegates → Coder writes → Verifier checks
