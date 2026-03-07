# Coding Agent Harness Survey

> **Date**: 2026-03-07
> **Author**: Research agents (3x parallel) + synthesis
> **Purpose**: Comprehensive survey of how production coding agent harnesses solve context packing, turn steering, file targeting, multi-agent coordination, and anti-stall detection. Findings inform the beefcake-swarm optimization roadmap.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Agents Surveyed](#agents-surveyed)
3. [Context Packing & File Selection](#context-packing--file-selection)
4. [Turn Steering & Budget Systems](#turn-steering--budget-systems)
5. [Edit Formats & Code Generation](#edit-formats--code-generation)
6. [Multi-Agent Coordination](#multi-agent-coordination)
7. [Anti-Stall & Loop Detection](#anti-stall--loop-detection)
8. [Tool Design Comparison](#tool-design-comparison)
9. [Actionable Recommendations for Beefcake Swarm](#actionable-recommendations-for-beefcake-swarm)
10. [Sources](#sources)

---

## Executive Summary

We surveyed 10+ coding agent harnesses to identify patterns that can improve beefcake-swarm's local-LLM worker reliability (Qwen3.5-397B-A17B on V100S GPUs). Three critical findings:

1. **Aider's repo map** (PageRank + tree-sitter) is the gold standard for context packing. Our alphabetical file walk only reaches `coordination/src/` files, never `crates/swarm-agents/`. Graph-based ranking solves this directly.

2. **Edit format choice has 3x impact on success rate.** Unified diffs score 61% vs search/replace at 20% (Aider benchmark, GPT-4 Turbo). Our `edit_file` uses search/replace blocks.

3. **SWE-agent's linter guardrail** (+3% SWE-bench resolve rate) and **OpenCode's LSP diagnostics after every edit** provide immediate feedback that prevents error compounding — both cheaper than full verifier runs.

---

## Agents Surveyed

| Agent | Language | Model Target | Key Innovation |
|-------|----------|-------------|----------------|
| **Claude Code** | TypeScript | Claude family | Agentic search (no indexing), context compaction |
| **OpenAI Codex CLI** | Rust | GPT-4.1/o3 | Patch-based edits (Lark grammar), encrypted compaction |
| **Google Gemini CLI** | TypeScript | Gemini 3 | Hierarchical GEMINI.md, CodebaseInvestigator subagent |
| **Aider** | Python | Any (LiteLLM) | PageRank repo map, edit format research, Architect mode |
| **OpenCode** | TypeScript | Any (AI SDK) | LSP diagnostics after edit, snapshot rollback, TODO state |
| **Cline** | TypeScript | Any | VS Code integration, proposed multi-agent framework |
| **Roo Code** | TypeScript | Any | Orchestrator pattern, native tool calling migration |
| **SWE-agent** | Python | Any | Agent-Computer Interface (ACI) design, linter guardrail |
| **OpenHands** | Python | Any | Micro-agents, AgentDelegateAction, 87% same-day bug fix |
| **Continue.dev** | TypeScript | Any | XML tool calling, repo maps, MCP + rules files |

---

## Context Packing & File Selection

### Aider: PageRank + Tree-Sitter (Gold Standard)

Aider's repo map is the most sophisticated file-selection system in production:

1. **Tree-sitter parsing**: Extracts symbol definitions (classes, functions, methods) from all source files
2. **Dependency graph**: Builds a NetworkX MultiDiGraph where files are nodes and symbol references are edges
3. **PageRank with personalization**: Ranks files by graph centrality, biased toward files mentioned in the current task
   - Personalization vector prioritizes files added via `--read` or currently in chat
   - Default: `1 / num_nodes` for unspecified files
4. **Binary search**: Fits the maximum number of high-ranking definitions within the token budget
5. **Dynamic token budget**: Defaults to 1K tokens (`--map-tokens`), expands when no files are in chat
   - Formula: `min(map_tokens * map_mul_no_files, max_context_window - 4096)`

**Output format**: Concise map showing key classes/functions/signatures (not full content). Uses `...` to indicate omitted sections. The LLM can request specific files for full details.

**Why this matters for us**: Our current context packer walks 402 `.rs` files alphabetically. Only ~18 fit in the 7471-token budget, and they're all from `coordination/src/` (alphabetically first). Target files in `crates/swarm-agents/src/` are never included. PageRank would rank `runtime_adapter.rs` highly because it's imported by `orchestrator.rs`, `driver.rs`, and test files.

### OpenAI Codex CLI: Byte-Heuristic + Compaction

- **Token estimation**: `APPROX_BYTES_PER_TOKEN = 4` — no real tokenizer, pure byte math
- **Truncation**: Per-item with prefix/suffix split (50/50 budget). Inserts `"...N tokens truncated..."` marker
- **Compaction** (two modes):
  - *Inline*: Sends history to model with prompt: "Create a handoff summary for another LLM that will resume the task"
  - *Remote*: Server-side `compact_conversation_history` API (OpenAI models only)
- **Auto-compaction**: Triggers on `ContextWindowExceeded`. If still over after compaction, progressively drops oldest items
- **Environment context**: Injected as XML at turn start — cwd, shell, date, timezone, network policy, active subagents
- **User message cap**: `COMPACT_USER_MESSAGE_MAX_TOKENS = 20,000`

### Google Gemini CLI: Hierarchical Context + Compression

- **Three-tier context loading**:
  1. Global: `~/.gemini/GEMINI.md`
  2. Workspace: Scans parent directories for GEMINI.md files
  3. Dynamic: When tools access a directory, auto-loads GEMINI.md from ancestors
- **Token estimation**: Local `estimateTokenCountSync()` + Gemini API `countTokens` endpoint
- **Compression threshold**: 50% of model's 1M token limit
- **Two-phase compression**:
  1. *Truncation*: Iterates backwards, truncates tool outputs exceeding 50K token budget to last 30 lines
  2. *Summarization*: Splits at user-turn boundary, preserving 30% (newest), summarizing 70% (oldest)
- **Failure resilience**: After one failed summarization, falls back to truncation-only mode
- **Conductor extension**: Persistent Markdown specs (product definition, tech stack, workflow conventions) that live alongside code

### OpenCode: Read Limits + Auto-Summarization

- **2000-line read limit** per file (paginated via offset)
- **MAX_LINE_LENGTH** truncation on individual lines
- **Auto-summarization** at 90% of context limit: `tokens > (model.info.limit.context - outputLimit) * 0.9`
- **Snapshot-based rollback**: `git write-tree` before changes, `git read-tree` on failure

### OpenHands: Conservative Fixed Window

- **Fixed 32K context** (tuned for open-source LLMs)
- **Condense when exceeded**: Only summarizes when absolutely needed
- **Original memory access**: Agent can open files in sandbox to access full history
- **Research finding**: "Observation masking" matched LLM summarization in cost savings after hyperparameter tuning

### Claude Code: No Indexing, Agentic Search

- **No pre-indexing** — model uses Glob/Grep/Read tools to search on-demand
- **Context compaction** when approaching limits
- Works with Opus-class models (200K+ context, excellent tool use) but would fail with smaller models

### Comparison Table

| Agent | Strategy | Token Budget | Pre-indexes? |
|-------|----------|-------------|-------------|
| **Aider** | PageRank + tree-sitter | 1K-4K default | Yes (graph) |
| **Codex CLI** | Byte-heuristic truncation + compaction | ~258K cap | No |
| **Gemini CLI** | Hierarchical GEMINI.md + 2-phase compression | 1M default | No |
| **OpenCode** | 2000-line limit + auto-summarization | Model-dependent | No |
| **OpenHands** | Fixed 32K + conservative condensing | 32K | No |
| **Claude Code** | Agentic search (no indexing) | 200K+ | No |
| **Our system** | Alphabetical file walk + token budget | ~7.5K | Sort of |

---

## Turn Steering & Budget Systems

### OpenCode: Hard Limits + Permission Gating

The most explicit turn-management system:

```javascript
const stream = streamText({
  stopWhen: async ({ steps }) => steps.length >= 1000 || processor.getShouldStop(),
  maxRetries: 3,
})
```

- **1000-step hard limit** before forced termination
- **Permission-based stopping**: Plan Agent is read-only; Build Agent has full access. Permission rejection stops the agent loop
- **Subagent isolation**: Cannot recursively spawn more subagents (`task: false` in tool config)

### Google Gemini CLI: Turn + Time Limits

- **Default limits**: `DEFAULT_MAX_TURNS = 15`, `DEFAULT_MAX_TIME_MINUTES = 5`
- **Per-agent overrides**: `CodebaseInvestigatorAgent` sets `maxTimeMinutes: 3, maxTurns: 10`
- **DeadlineTimer**: Pauses during user confirmation (human review doesn't count against time)
- **Mandatory completion**: If model stops without calling `complete_task`, treated as error
- **Termination modes**: `ERROR`, `TIMEOUT`, `GOAL`, `MAX_TURNS`, `ABORTED`, `ERROR_NO_COMPLETE_TASK_CALL`

### OpenAI Codex CLI: Model Self-Terminates

- **No explicit turn counter** — agent loop runs until model emits final message without pending tool calls
- **Multi-agent guards**: `agent_max_threads` (concurrent) and `agent_max_depth` (nesting) limits
- **Depth exceeded**: Returns "Agent depth limit reached. Solve the task yourself."
- **User approval modes**: Suggest / Auto Edit / Full Auto provide human-gated turn boundaries

### Aider: Edit Format as Steering

- No explicit turn limits
- **Edit format enforcement** acts as implicit steering — unified diffs make the model produce machine-parsable output instead of conversational text
- **Architect mode**: Two-step (reasoning model proposes, editing model executes) naturally limits each model's scope

### Our System: Write Deadline

- `max_turns_without_write=5` in RuntimeAdapter terminates agents that don't call edit_file/write_file
- `SWARM_MAX_NO_CHANGE=3` circuit breaker for stuck iterations
- `SWARM_MAX_RETRIES=10` overall iteration limit
- **Problem observed**: Qwen3.5 exits the Rig loop by returning text on turn 4 (no tool call), bypassing the write deadline entirely

### Comparison

| Agent | Turn Limit | Time Limit | Write Enforcement |
|-------|-----------|-----------|-------------------|
| **OpenCode** | 1000 steps | None | Permission system |
| **Gemini CLI** | 15 turns | 5 min | `complete_task` required |
| **Codex CLI** | None | None | Model self-terminates |
| **Aider** | None | None | Edit format enforcement |
| **Our system** | 15 turns + write deadline | 15 min HTTP timeout | `max_turns_without_write=5` |

---

## Edit Formats & Code Generation

### Aider's Benchmark: Edit Format Has 3x Impact

From Aider's controlled experiments on GPT-4 Turbo:

| Format | Success Rate | Notes |
|--------|-------------|-------|
| Search/Replace | 20% | Model often uses `# ... original code here ...` lazy comments |
| **Unified Diff** | **61%** | 3x reduction in lazy coding |
| Whole File | ~61% | Comparable success but much higher cost/latency |

**Key insight**: Unified diffs make models treat output as "textual data for a program" rather than conversational text. This behavioral shift is exactly what we need from Qwen3.5 (which tends to dump 4500+ tokens of analysis text instead of tool calls).

### Codex CLI: Patch-Based Edits

- **Primary write tool**: `apply_patch` (not `write_file`)
- Uses a **Lark grammar** (`tool_apply_patch.lark`) for multi-file patches
- Supports renames, precise diffs, and multi-file operations in a single tool call
- More surgical than full-file writes

### Gemini CLI: Search-and-Replace with Fuzzy Fallback

- `edit` tool uses `old_string` / `new_string` semantics
- **Multi-strategy matching**: exact → flexible (whitespace-insensitive) → regex → fuzzy (Levenshtein, 10% threshold)
- **Omission placeholder detector**: Rejects lazy patterns like `// ... rest of code`
- **LLM edit correction**: `FixLLMEditWithInstruction` sends failed edit + error back to model for retry

### SWE-agent: Simplified Actions + Validation

- Small set of simple actions for viewing, searching, editing files
- **Linter guardrail**: Edit command rejected if code isn't syntactically correct
- Selected linter errors shown to agent with before/after snippets
- **Impact**: Without linting, performance drops from 18.0% to 15.0% resolved (+3% from guardrail alone)

### Roo Code: Native Tool Calling Migration

Migrated from XML-based tool calling to native function calling:
- **XML problems**: Latency, accuracy issues, inconsistent formats, parsing complexity
- **Native benefits**: Type safety, eliminates parsing errors, significantly faster edits

### Our System: Search/Replace Blocks via `edit_file`

- `edit_file` tool accepts `old_text` / `new_text` search/replace blocks
- `tool_choice: Required` forces tool calls (partially compensates for search/replace's lower success rate)
- `max_tokens: 4096` caps per-turn waste
- **Gap**: No edit validation before applying, no omission placeholder detection

---

## Multi-Agent Coordination

### OpenCode: Hierarchical Subagents + TODO State

```javascript
export const TaskTool = Tool.define("task", async () => {
  async execute(params, ctx) {
    const agent = await Agent.get(params.subagent_type)
    const session = await Session.create(ctx.sessionID, params.description)
    const result = await Session.prompt({
      sessionID: session.id,
      agent: agent.name,
      tools: { task: false, ...agent.tools }, // No recursive spawning
      parts: [{ type: "text", text: params.prompt }],
    })
  }
})
```

- Each subagent: own context window, can use different LLM, custom tool access, isolated session state
- **Stateless invocations** — cannot send additional messages after spawn
- **State sharing via TODO system**: Global mutable state accessible to all agents
- **Event bus**: Real-time SSE updates for coordination across multiple UI clients

### OpenAI Codex CLI: Full Lifecycle Management

- **Tools**: `spawn_agent`, `send_input`, `resume_agent`, `wait`, `close_agent`
- Each sub-agent gets its own thread, config, and conversation history
- **Context forking**: Parent can optionally clone its history to child (`fork_context`)
- **Guards**: `agent_max_threads` + `agent_max_depth` prevent runaway spawning
- **Nickname system**: Agents get unique names from a pool ("Plato", "Plato the 2nd", etc.)
- **Wait mechanism**: 10s-3600s configurable timeouts

### Google Gemini CLI: Emerging Framework

- **LocalAgentExecutor**: Runs subagent loops with isolated tool registries, dedicated chat instances, independent compression
- **CodebaseInvestigator**: Concrete read-only subagent with structured JSON output (`SummaryOfFindings`, `ExplorationTrace`, `RelevantLocations`). Temperature 0.1, thinking mode, 3-min timeout, 10-turn limit
- **Agent-to-Agent (A2A) server**: HTTP-based remote agents with auth and acknowledgment tracking
- **Recursion prevention**: Subagents cannot call other agents

### Roo Code: Orchestrator Pattern

- **Specialized modes**: Architect (plans), Code (applies diffs), Debug (runs terminals, inspects logs), Ask (Q&A), Custom
- **Orchestrator meta-role**: Coordinates tasks by delegating to modes, switches dynamically based on task requirements
- Not truly multi-agent (single agent, mode switching), but avoids context-loss of handoffs

### Aider: Architect Mode (Two-Model Pipeline)

- **Architect** (reasoning model like o1): Proposes solution in natural language
- **Editor** (code model like DeepSeek/o1-mini): Converts proposal to specific file edits
- **Performance**: o1-preview + DeepSeek = 85% on SWE-bench
- **Trade-off**: Two LLM requests (higher latency) but more reliable edits

### OpenHands: Micro-Agents + Delegation

- **Micro-agents**: Lightweight agents instantiated from natural language
- **Three trigger types**: Always (every session), Keyword (on mention), Manual (user/programmatic)
- **AgentDelegateAction**: Hands off subtasks to most qualified collaborator
- **AgentHub registry**: Specialized agents available for delegation

### Claude Code Teams: Lead + Teammates

- **Lead (Opus)**: Picks issues, assigns to teammates, reviews results
- **Teammates (Sonnet)**: Each works on one issue on a separate branch
- **File locking**: Prevents race conditions during task claiming
- **Mailbox messaging**: 1:1 agent communication
- **Auto-unblocking**: When blocking task completes, dependent tasks become claimable

### Our System: Cloud Manager + Local Workers

- Cloud manager (Claude Opus) plans and delegates via WorkPackets
- Local workers (Qwen3.5 on 3 nodes) execute code changes in git worktrees
- State shared via WorkPacket serialization
- Closest to **Aider's Architect pattern** — reasoning model proposes, editing model executes
- Key difference: We pass structured WorkPackets; Aider passes natural language proposals

### Comparison

| Agent | Architecture | Isolation | State Sharing |
|-------|-------------|-----------|---------------|
| **Codex CLI** | Full lifecycle (spawn/wait/close/resume) | Thread + history | Context forking |
| **OpenCode** | Hierarchical, stateless | Session + context | TODO system |
| **Gemini CLI** | LocalAgentExecutor + A2A server | Tool registry + chat | Structured JSON output |
| **Roo Code** | Mode switching (single agent) | N/A | Shared context |
| **Aider** | Two-model pipeline | Separate prompts | Prompt chaining |
| **OpenHands** | Micro-agents + delegation | Sandbox | File-based |
| **Claude Code** | Lead + teammates | Branch per agent | File locking + mailbox |
| **Our system** | Cloud manager + local workers | Git worktrees | WorkPacket serialization |

---

## Anti-Stall & Loop Detection

### SWE-agent: Linter Guardrail

The most impactful single mechanism:

1. Linter runs when edit command is issued
2. Edit **rejected** if code isn't syntactically correct
3. Selected linter errors shown to agent with before/after file snippets
4. Invalid edits discarded, agent asked to try again

**Impact**: Performance drops from 18.0% to 15.0% without linting (+3% resolve rate from guardrail alone)

**Key benefit**: Prevents error propagation. Common failure mode: agents repeatedly editing same snippet after introducing a syntax error (wrong indentation, extra parenthesis).

### OpenCode: LSP Diagnostics Feedback Loop

```javascript
// After file edit:
await LSP.touchFile(filePath, true)
const diagnostics = await LSP.diagnostics()
```

1. LLM makes a change
2. LSP client sends `textDocument/didChange` over STDIO
3. Language server analyzes (150ms debounce)
4. Diagnostics returned to LLM context
5. LLM adjusts next action based on errors

**Benefits**: Catches type mismatches, undefined variables, import errors immediately. Prevents agent from "going off the rails." Real-time error correction before code execution.

### Google Gemini CLI: Multi-Layer Detection

- **Loop detection events**: `GeminiEventType.LoopDetected` emitted when agent spins without progress
- **Invalid stream detection**: Four cases — `NO_FINISH_REASON`, `NO_RESPONSE_TEXT`, `MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`. Each triggers retry (2 attempts, 500ms base delay)
- **Protocol violation**: Model stopping without `complete_task` = error
- **Curated history**: Strips invalid/empty model outputs to prevent corrupted state propagation
- **Time limits**: `DeadlineTimer` hard-stops subagents after `maxTimeMinutes` + 1-min grace
- **Model fallback**: Switches to alternative models on persistent 429 errors

### OpenAI Codex CLI: Minimal Detection

- **Compaction-on-overflow**: Auto-triggers, progressively drops oldest items if still over
- **Retry with backoff**: Stream errors use exponential backoff up to `stream_max_retries()`
- **Cancellation tokens**: In-flight tool calls aborted when turn ends
- **Approval gates**: Mutating tools require `tool_call_gate` readiness
- **No explicit loop detection**: Model trusted to converge; user interrupts if stuck

### Ralph: Circuit Breaker Pattern

- **Dual-threshold system**:
  - 3 no-progress loops → OPEN state
  - 30-min cooldown
  - HALF_OPEN → test run
  - CLOSED if test passes
- **Same-error detection**: 5 identical errors → circuit opens
- Most sophisticated anti-stall mechanism surveyed

### Our System: Write Deadline + Max No-Change

- `max_turns_without_write=5`: Terminates agents that don't write within N turns
- `SWARM_MAX_NO_CHANGE=3`: Circuit breaker for iterations with no file changes
- `SWARM_MAX_RETRIES=10`: Overall limit
- **Gap**: No per-edit validation, no LSP feedback, no loop detection events

### Comparison

| Agent | Loop Detection | Edit Validation | Feedback Speed |
|-------|---------------|----------------|----------------|
| **SWE-agent** | N/A | Linter guardrail (reject invalid) | Immediate |
| **OpenCode** | Step limit (1000) | LSP diagnostics | ~150ms |
| **Gemini CLI** | Event-based + time limits | None | Turn-level |
| **Codex CLI** | None (model trusted) | None | N/A |
| **Ralph** | Circuit breaker (3 loops) | None | 30-min cooldown |
| **Our system** | Write deadline (5 turns) | Full verifier (post-iteration) | Minutes |

---

## Tool Design Comparison

### Codex CLI Tool Set (Rust)

| Tool | Notes |
|------|-------|
| `apply_patch` | Lark grammar, multi-file, renames, precise diffs |
| `read_file` | Two modes: line-range Slice + indentation-aware Structural |
| `grep_files` | ripgrep wrapper, sorted by modification time |
| `list_dir` | Paginated, depth-limited (default 2) |
| `shell` / `unified_exec` | Full sandbox controls, ZshFork backend |
| `spawn/send/wait/close/resume_agent` | Full sub-agent lifecycle |
| `update_plan` | Structured planning with step statuses |
| `search_tool_bm25` | BM25 tool discovery for MCP tools |

**Key choice**: `apply_patch` (not `write_file`) as primary edit tool — patch-based approach is more surgical.

### Gemini CLI Tool Set (TypeScript)

| Tool | Notes |
|------|-------|
| `edit` (replace) | Multi-strategy: exact → flexible → regex → fuzzy (Levenshtein) |
| `write_file` | Full file create/overwrite |
| `read_file` / `read_many_files` | Line ranges, batch glob reading |
| `grep` | Prefers `git grep`, falls back to `rg` |
| `shell` | Background mode support |
| `save_memory` | Writes to GEMINI.md context file |
| `enter/exit_plan_mode` | Mode switching for plan-then-execute |
| `complete_task` | Mandatory completion signal |

**Key choices**: Omission placeholder detection, LLM edit correction for failed edits, model-family-specific tool schemas.

### Our Tool Set (Rust/Rig)

| Tool | Notes |
|------|-------|
| `read_file` | Line-limited, content truncation at 4000 chars |
| `write_file` | Full file write |
| `edit_file` | Search/replace (`old_text` / `new_text`) |
| `list_files` | Directory listing |
| `run_command` | Shell execution with pipe support |

**Gaps vs industry**: No patch-based edits, no edit validation/rejection, no omission detection, no structured completion signal.

---

## Actionable Recommendations for Beefcake Swarm

Ranked by impact/effort ratio, with direct mappings to the research findings.

### Tier 1: Quick Wins (days)

#### 1. Linter Guardrail (SWE-agent pattern)

Run `cargo check` on the edited file immediately after `edit_file` applies. If it introduces new errors, reject the edit and return the error to the LLM in-context.

- **Evidence**: +3% SWE-bench resolve rate from this single change
- **Implementation**: In the `edit_file` tool handler, after writing the file, run `cargo check --message-format=json` on the crate. If new errors appear that weren't in the pre-edit baseline, revert the edit and return diagnostics
- **Cost**: Low — we already have the verifier's error parser

#### 2. Omission Placeholder Detection (Gemini CLI pattern)

Reject edits containing lazy patterns like `// ... existing code ...`, `// rest of implementation`, `/* ... */`.

- **Evidence**: Gemini CLI's `omissionPlaceholderDetector.ts` catches this common failure mode
- **Implementation**: Regex check in `edit_file` before applying the write

#### 3. Mandatory Completion Signal (Gemini CLI pattern)

Add a `complete_task` tool. If the agent's loop ends without calling it, treat as failure and retry.

- **Evidence**: Gemini CLI treats `ERROR_NO_COMPLETE_TASK_CALL` as error — prevents agents from silently giving up
- **Implementation**: New Rig tool + check in RuntimeAdapter

### Tier 2: Medium Effort (1-2 weeks)

#### 4. Repo Map via Tree-Sitter (Aider pattern)

Replace alphabetical file walking in the context packer with graph-based ranking.

- **Evidence**: Gold standard for context packing. Directly solves our "402 files, only coordination/src/ fits" problem
- **Implementation**: Use `tree-sitter` Rust crate to parse all `.rs` files, build dependency graph, run PageRank, binary-search to fit token budget
- **Alternative**: Simpler version using `cargo metadata` + `use` statement analysis instead of full tree-sitter

#### 5. Snapshot-Based Rollback (OpenCode pattern)

`git write-tree` before each iteration, `git read-tree` on failure.

- **Evidence**: OpenCode uses this to prevent compounding errors across iterations
- **Implementation**: In the orchestrator loop, capture git tree hash before worker runs. On verifier failure, restore and retry with error context

#### 6. Post-Edit Diagnostics (OpenCode LSP pattern)

Run `cargo check --message-format=json` after every edit and feed diagnostics back into the LLM context, not just at verifier time.

- **Evidence**: OpenCode's LSP integration catches type mismatches, undefined variables immediately
- **Implementation**: Modify `edit_file` tool to run lightweight check and append diagnostics to tool result
- **Trade-off**: Adds ~10-30s per edit on our hardware. Could use `cargo check` on just the affected crate

### Tier 3: Larger Effort (weeks)

#### 7. Unified Diff Edit Format (Aider pattern)

Change `edit_file` to accept unified diffs instead of search/replace blocks.

- **Evidence**: 61% vs 20% success rate (GPT-4 Turbo). 3x reduction in lazy coding
- **Risk**: Requires testing with Qwen3.5 specifically — the benchmark was on GPT-4 Turbo
- **Implementation**: New tool `apply_diff` that accepts unified diff format, parses and applies patches
- **Alternative**: Keep search/replace but add Gemini CLI's fuzzy matching fallback chain

#### 8. Circuit Breaker with Cooldown (Ralph pattern)

More sophisticated than our `MAX_NO_CHANGE=3` — track error similarity, implement OPEN/HALF_OPEN/CLOSED states.

- **Evidence**: Ralph's dual-threshold (3 no-progress + 5 same-error) with 30-min cooldown is the most robust anti-stall pattern
- **Implementation**: State machine in the orchestrator loop with configurable thresholds

#### 9. Architect/Editor Separation (Aider pattern)

Formally separate reasoning (Cloud → natural language proposal) from editing (Local → code changes only).

- **Evidence**: o1-preview + DeepSeek = 85% on SWE-bench. Our system is already close to this pattern
- **Implementation**: Cloud manager emits a structured proposal (not a WorkPacket with file contexts). Local worker receives only: (a) the proposal, (b) the target files, (c) current file content. No error history, no escalation context
- **Benefit**: Qwen3.5 only needs to be good at applying changes, not reasoning about architecture

#### 10. CodebaseInvestigator Subagent (Gemini CLI pattern)

Dedicated read-only agent that explores the codebase and returns structured findings before the coding agent starts.

- **Evidence**: Gemini CLI's investigator uses temperature 0.1, structured JSON output, 3-min timeout, 10-turn limit
- **Implementation**: Run on the fast tier (vasp-03) before dispatching to coder tier. Returns `SummaryOfFindings` + `RelevantLocations` that feed into the WorkPacket

---

## Sources

### OpenAI Codex CLI
- [GitHub - openai/codex](https://github.com/openai/codex)
- [Unrolling the Codex Agent Loop](https://openai.com/index/unrolling-the-codex-agent-loop/)
- [Codex CLI Reference](https://developers.openai.com/codex/cli/reference/)
- [How Codex is Built - Pragmatic Engineer](https://newsletter.pragmaticengineer.com/p/how-codex-is-built)

### Google Gemini CLI
- [GitHub - google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli)
- [Gemini CLI Documentation](https://developers.google.com/gemini-code-assist/docs/gemini-cli)
- [Conductor: Context-Driven Development](https://developers.googleblog.com/conductor-introducing-context-driven-development-for-gemini-cli/)
- [Provide context with GEMINI.md files](https://geminicli.com/docs/cli/gemini-md/)
- [Agent mode overview](https://developers.google.com/gemini-code-assist/docs/agent-mode)
- [GitHub - gemini-cli-extensions/conductor](https://github.com/gemini-cli-extensions/conductor)

### Aider
- [GitHub - Aider-AI/aider](https://github.com/Aider-AI/aider)
- [Repository map](https://aider.chat/docs/repomap.html)
- [Building a better repo map with tree-sitter](https://aider.chat/2023/10/22/repomap.html)
- [Separating code reasoning and editing (Architect mode)](https://aider.chat/2024/09/26/architect.html)
- [Unified diffs make GPT-4 Turbo 3X less lazy](https://aider.chat/docs/unified-diffs.html)
- [GPT code editing benchmarks](https://aider.chat/docs/benchmarks.html)
- [Edit formats](https://aider.chat/docs/more/edit-formats.html)

### OpenCode
- [GitHub - anomalyco/opencode](https://github.com/anomalyco/opencode)
- [OpenCode | The open source AI coding agent](https://opencode.ai/)
- [How Coding Agents Actually Work: Inside OpenCode](https://cefboud.com/posts/coding-agents-internals-opencode-deepdive/)
- [LSP Servers](https://opencode.ai/docs/lsp/)
- [Agents](https://opencode.ai/docs/agents/)
- [Context Management and Compaction](https://deepwiki.com/sst/opencode/2.4-context-management-and-compaction)
- [GitHub - code-yeongyu/oh-my-opencode](https://github.com/code-yeongyu/oh-my-opencode)
- [GitHub - darrenhinde/OpenAgentsControl](https://github.com/darrenhinde/OpenAgentsControl)

### Cline
- [Cline - AI Coding, Open Source and Uncompromised](https://cline.bot/)
- [Cline CLI 2.0 Turns Your Terminal Into an AI Agent Control Plane](https://devops.com/cline-cli-2-0-turns-your-terminal-into-an-ai-agent-control-plane/)
- [Feature Request: Multi-Agent Framework](https://github.com/cline/cline/discussions/489)

### Roo Code
- [Roo Code](https://roocode.com/)
- [GitHub - RooCodeInc/Roo-Code](https://github.com/RooCodeInc/Roo-Code)
- [Roo Code vs Cline: Best AI Coding Agents for VS Code](https://www.qodo.ai/blog/roo-code-vs-cline/)
- [Multi Agent Workflow With Roo Code](https://xebia.com/blog/multi-agent-workflow-with-roo-code/)
- [RFC: Native Tool Use for Top-Tier AI Models](https://github.com/RooCodeInc/Roo-Code/issues/4047)

### SWE-agent
- [GitHub - SWE-agent/SWE-agent](https://github.com/SWE-agent/SWE-agent)
- [SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering](https://arxiv.org/abs/2405.15793)
- [Agent-Computer Interface](https://swe-agent.com/0.7/background/aci/)

### OpenHands
- [OpenHands | The Open Platform for Cloud Coding Agents](https://openhands.dev/)
- [GitHub - OpenHands/OpenHands](https://github.com/OpenHands/OpenHands)
- [Memory Management & Context Condense for CodeAct Agent](https://github.com/OpenHands/OpenHands/issues/1748)
- [Cutting Through the Noise: Smarter Context Management for LLM-Powered Agents](https://blog.jetbrains.com/research/2025/12/efficient-context-management/)

### Continue.dev
- [Continue.dev](https://www.continue.dev/)
- [How to Make Agent mode Aware of Codebases and Documentation](https://docs.continue.dev/guides/codebase-documentation-awareness)

### General
- [AI Code Edit Formats Guide 2025: Diff vs Whole File vs Semantic](https://www.morphllm.com/edit-formats)
- [Code Surgery: How AI Assistants Make Precise Edits](https://fabianhertwig.com/blog/coding-assistants-file-edits/)
- [Guardrails for Agentic Coding](https://jvaneyck.wordpress.com/2026/02/22/guardrails-for-agentic-coding-how-to-move-up-the-ladder-without-lowering-your-bar/)
- [Why Agents Get Stuck in Loops (And How to Prevent It)](https://gantz.ai/blog/post/agent-loops/)
- [ralph-claude-code: Circuit Breaker Pattern](https://dev.to/tumf/ralph-claude-code-the-technology-to-stop-ai-agents-how-the-circuit-breaker-pattern-prevents-3di4)
- [Git Worktrees: The Secret Weapon for Running Multiple AI Coding Agents in Parallel](https://medium.com/@mabd.dev/git-worktrees-the-secret-weapon-for-running-multiple-ai-coding-agents-in-parallel-e9046451eb96)
- [Parallelizing AI Coding Agents](https://tessl.io/blog/how-to-parallelize-ai-coding-agents/)
- [Multi-Agent Parallel Execution](https://skywork.ai/blog/agent/multi-agent-parallel-execution-running-multiple-ai-agents-simultaneously/)
