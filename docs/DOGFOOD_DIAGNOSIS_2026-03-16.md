# Dogfood Diagnosis: Qwen3.5-122B Tool Calling Failures on llama-server

**Date:** 2026-03-16
**Author:** Claude Opus 4.6 (research session)
**Status:** Blocking all dogfood progress — zero issues closed in tonight's session

---

## Executive Summary

The swarm dogfood loop ran 8 issues tonight with zero completions. Workers successfully edited files in 1 of 3 runs that reached code execution, but the verifier never passed and no issues were closed. Three distinct infrastructure-level failures compound to make tool-calling unreliable on our Qwen3.5-122B instances.

---

## Infrastructure State (Observed Tonight)

### Compute Nodes

| Node | Model | GGUF | Flags | Context | Slots | Uptime |
|------|-------|------|-------|---------|-------|--------|
| vasp-01 | Qwen3.5-122B-A10B Q4_K_M | 3-shard | `--ctx-size 32768 --n-gpu-layers 99 -ot .ffn_.*_exps.=CPU --threads 16 --batch-size 512 --ubatch-size 512 --cache-type-k q4_0 --cache-type-v q4_0 -fa on --parallel 2 --mlock --cont-batching --jinja` | 32768 (but `/slots` reports **16384 per slot** = 32768/2) | 2 | Since Mar 14, 1148 CPU-min |
| vasp-02 | Qwen3.5-122B-A10B Q4_K_M | 3-shard | Same as vasp-01 | Same (16384 per slot) | 2 | Since Mar 14, 98 CPU-min |
| vasp-03 | Qwen3.5-27B-Distilled Q4_K_M | 1-shard | `--ctx-size 65536 --n-gpu-layers 99 -fa on` | 65536 | 4 | Since Mar 14 |

**Key observation:** vasp-01/02 use `--ctx-size 32768 --parallel 2`, which gives **16384 tokens per slot**. This is the root cause of the context overflow error. The `--jinja` flag IS already set (contradicting earlier hypothesis).

### llama.cpp Build

- **Version:** v8231 (commit c024d8590)
- **Binary:** `/usr/local/bin/llama-server-mmq` (compiled with `GGML_CUDA_FORCE_MMQ=ON` for V100)
- **Includes:** Autoparser refactor (PR #18675), which introduced multiple tool-calling regressions

### Swarm Software

- **Orchestrator:** `crates/swarm-agents` (Rust, using Rig v0.30.0)
- **Cloud manager:** Claude Opus 4.6 via CLIAPIProxy (localhost:8317)
- **Worker tiers:** Coder (vasp-01:8081), Reasoning (vasp-02:8081), Scout (vasp-03:8081)
- **Tool call format:** OpenAI-compatible `/v1/chat/completions` with `tools` parameter

---

## Failure Mode 1: Context Window Overflow (16384 tokens per slot)

### What Happened

The `proxy_planner` worker call returned:
```
HTTP 400: request (17877 tokens) exceeds the available context size (16384 tokens)
```

### Root Cause

`--ctx-size 32768 --parallel 2` divides context evenly: **16384 tokens per slot**. A single worker conversation with:
- System prompt (~2K tokens)
- Tool definitions (~3K tokens for 7 tools)
- One `read_file` of a 355-line Rust file (~7K tokens)
- Conversation history from prior tool calls (~5K tokens)

...easily exceeds 16384 tokens by turn 3-4.

### Fix Required

Increase `--ctx-size` to **65536** on vasp-01 and vasp-02 (matching vasp-03). With `--parallel 2`, this gives 32768 tokens per slot. Alternatively, reduce to `--parallel 1` with `--ctx-size 65536` for 65536 per slot (half the throughput but more headroom).

The `--cache-type-k q4_0 --cache-type-v q4_0` quantized KV cache is already enabled, so memory impact of larger context should be manageable. Verify with:
```bash
# On vasp-01:
/usr/local/bin/llama-server-mmq --model /scratch/ai/models/Qwen3.5-122B-A10B-Q4_K_M-00001-of-00003.gguf \
  --ctx-size 65536 --parallel 2 --n-gpu-layers 99 -ot '.ffn_.*_exps.=CPU' \
  --cache-type-k q4_0 --cache-type-v q4_0 -fa on \
  --threads 16 --batch-size 512 --ubatch-size 512 \
  --mlock --cont-batching --jinja \
  --host 0.0.0.0 --port 8081
```

If OOM, try `--parallel 1 --ctx-size 65536` or `--parallel 2 --ctx-size 49152`.

---

## Failure Mode 2: Tool Call Field Omission (old_content dropped from edit_file)

### What Happened

The worker correctly called `edit_file` with both `old_content` and `new_content` on its first tool call:
```json
{"old_content":"const DEFAULT_TIMEOUT_SECS: u64 = 30;","new_content":"const AST_GREP_TIMEOUT_SECS: u64 = 60;","path":"crates/swarm-agents/src/tools/astgrep_tool.rs"}
```

On subsequent calls within the same conversation, it dropped `old_content`:
```json
{"new_content":"...multi-line replacement...","path":"crates/swarm-agents/src/tools/astgrep_tool.rs"}
```

This pattern repeated on every run — first call correct, subsequent calls broken.

### Root Cause (Compound)

**1. Model behavior ("Lazy Qwen"):** Qwen 3.5 models are trained on XML-style tool calls (`<parameter=name>value</parameter>`) and tend to omit parameters after the first turn, assuming context carries forward. This is acknowledged by Qwen's own docs: *"It is not guaranteed that the model generation will always follow the protocol."*

**2. Autoparser regression:** llama.cpp v8231 includes the PEG autoparser refactor (PR #18675) which has known issues:
- Grammar constraint silently falls back to unconstrained generation when parsing fails ([#19051](https://github.com/ggml-org/llama.cpp/issues/19051))
- PEG parser fails when model outputs text before `<tool_call>` ([#20260](https://github.com/ggml-org/llama.cpp/issues/20260))
- Parser crash on max_tokens truncation ([#20193](https://github.com/ggml-org/llama.cpp/issues/20193)) — we also hit this: `"Failed to parse input at pos 124"` after 33 minutes of generation

**3. reasoning_content not passed back ([QwenLM/Qwen3.5 #26](https://github.com/QwenLM/Qwen3.5/issues/26)):** When `reasoning_content` is not explicitly captured from assistant responses and included in subsequent messages, the model's internal reasoning leaks into the `content` field, corrupting all subsequent turns. Our orchestrator (Rig v0.30.0) does NOT capture or pass back `reasoning_content`. The Qwen3.5 chat template explicitly supports it:
```jinja
{%- if message.reasoning_content is string %}
    {%- set reasoning_content = message.reasoning_content %}
```
But Rig's OpenAI client doesn't extract this field from responses.

**4. We pass `enable_thinking: false` but it may not be honored.** Our coder agent sends:
```rust
// crates/swarm-agents/src/agents/coder.rs:114
"enable_thinking": false
```
But the chat template check is:
```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
```
This only works if `enable_thinking` is passed via `--chat-template-kwargs`, not via the API request body. The server may still be generating `<think>` blocks, which:
- Consume context window tokens
- Confuse the PEG parser
- Cause `reasoning_content` leakage into `content`

### Fixes Required

**Infrastructure (llama-server flags):**
1. Add `--reasoning-format none` to disable thinking entirely for worker calls
2. Optionally add `--chat-template-kwargs '{"enable_thinking":false}'` as a belt-and-suspenders measure
3. Consider adding `--min-p 0.05` (already set per `/slots` output, but verify it's in startup script)

**Software (orchestrator code):**
1. **Client-side tool call validation:** After each tool call response, validate that all `required` fields in the tool schema are present. If missing, return an error to the model: `"edit_file requires old_content. You provided: {keys}. Re-call with all required fields."`
2. **Investigate Rig v0.30.0's handling of `reasoning_content`:** Does Rig strip it? Does it pass it back in the message history? If not, we may need to patch Rig or post-process responses.
3. **Add `"strict": true` to tool definitions** in the JSON schema to signal llama.cpp to enforce grammar constraints on arguments.

**Build upgrade:**
1. Upgrade llama.cpp to b8256+ which contains fixes for #20198 (tool_call args as object vs string), #20260 (peg-native parser with thinking), and #20352 (streaming grammar not applied).

---

## Failure Mode 3: Infinite Tool Call Loop (proxy_search_code)

### What Happened

The manager (Claude Opus 4.6) correctly delegated a rename task to `proxy_rust_coder`. The worker successfully renamed `DEFAULT_TIMEOUT_SECS` → `AST_GREP_TIMEOUT_SECS` in `astgrep_tool.rs` (all 3 occurrences). The manager then called `proxy_search_code("DEFAULT_TIMEOUT_SECS")` to verify no references remained.

The search returned results from **other files** (exec_tool.rs, colgrep_tool.rs, search_code_tool.rs) — these are unrelated constants with the same name in different modules. The manager couldn't distinguish "task-scoped" from "workspace-wide" results and kept re-searching every ~13 seconds for **45+ minutes** (866KB of log), burning cloud API budget without ever running the verifier.

### Root Cause

1. **`proxy_search_code` returns workspace-wide results** with no way to scope to a specific file or set of files. The `glob` parameter exists but the manager didn't use it.
2. **No repeat-call circuit breaker** in the orchestrator. The same tool was called with identical arguments 100+ times without intervention.
3. **The manager still had `proxy_search_code`** — Wait, we removed search tools from the manager bundle! The issue is that the **worker** (proxy_rust_coder) was doing the searching, and workers DO have search tools. The worker was stuck in the loop, not the manager.

Actually, re-examining the logs: the `proxy_search_code` calls were at the **manager level** (outside any `proxy_rust_coder` context). This means either:
- The old binary (pre-search-tool-removal) was still running, OR
- The search calls came from within a worker delegation that returns its conversation to the manager

Looking at the log timestamps and the fact that we deployed the search-tool removal at 23:00 but the s9pz run started at 23:23, the binary WAS rebuilt. However, the `run-3-beefcake-s9pz` log shows `proxy_search_code` calls at the top level. **This needs investigation** — it's possible the search tool removal didn't take effect, or the calls are from within worker sub-conversations.

### Fixes Required

1. **Repeat-call circuit breaker:** Detect when the same tool is called with identical arguments N times (e.g., 3) and inject a forced response: `"You have called this tool 3 times with the same arguments. The results will not change. Move on to the next step."`
2. **Scope `proxy_search_code` results:** Add a `file` parameter or default to only searching files mentioned in the issue/task.
3. **Verify the search tool removal deployed correctly** — check the binary on ai-proxy to confirm `proxy_search_code` is not in the manager's toolset.

---

## Failure Mode 4: Zombie Processes

### What Happened

When the dogfood loop moves to the next issue, it cleans up the worktree (`rm -rf /tmp/beefcake-wt/<id>`) but does NOT kill child processes from the previous run. Specifically, worker sub-processes spawned by `proxy_rust_coder` or `proxy_general_coder` (which make HTTP calls to vasp-01/02) continue running after the parent `swarm-agents` process exits.

These zombie workers have their `working_dir` set to the now-deleted worktree path. Every tool call (even `ls -la /` or `true`) fails with `No such file or directory` because `Command::new().current_dir(deleted_path)` returns ENOENT. They loop indefinitely, writing errors to the old log file.

We observed:
- PID 2522730: jdxc worker from 22:16, still running at 23:45 (1.5 hours)
- PID 2535478: jdxc worker from 22:59, still running at 23:45

### Root Cause

The `swarm-agents` process spawns HTTP requests to vasp-01/02 for worker calls. These are blocking `reqwest` calls within `std::thread::spawn` (because `process_issue` is `!Send`). When the parent process exits, the child threads may continue if they're blocked on an HTTP response from the 122B model (which can take 30+ minutes).

### Fix Required

1. **Process group kill on worktree cleanup:** When cleaning up a worktree, kill the entire process group, not just the parent.
2. **Set HTTP timeouts aggressively:** The `DEFAULT_LOCAL_HTTP_TIMEOUT_SECS` is 2700 (45 minutes). Consider reducing to 600-900 seconds for worker calls.
3. **Check for stale processes before starting a new run** in `dogfood-loop.sh`.

---

## Failure Mode 5: Baseline Verification Failures (cargo fmt)

### What Happened

Two issues (beefcake-jdxc, beefcake-rt4g) failed immediately at baseline verification:
```
Baseline verification FAILED — worktree does not compile/pass tests before agent modifications.
[RED] 0/3 gates passed (1147ms) [fmt:FAIL → clippy:SKIP → check:SKIP]
```

### Root Cause

Our commit `62c5882` (removing search tools from manager bundle) left a long line in a test assertion that `cargo fmt` wanted to split. The worktree was created from this commit, so it failed `cargo fmt --check` before any agent started.

### Fix Applied

Committed `7f8d38c` (cargo fmt fix) and `6cad17e` (shared.rs test fix, `assert_eq!(3, ...)` instead of `6`). Both are pushed to origin but were NOT deployed to ai-proxy before the s9pz run started, so s9pz's verifier found the old test failures.

---

## Current Chat Template Analysis

The `/props` endpoint shows vasp-01/02 are using Qwen3.5's native chat template (from the GGUF). Key observations:

1. **Tool call format is XML-based**, not JSON:
```
<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
</function>
</tool_call>
```

2. **The template supports `reasoning_content`** explicitly:
```jinja
{%- if message.reasoning_content is string %}
    {%- set reasoning_content = message.reasoning_content %}
```

3. **The template supports `enable_thinking`:**
```jinja
{%- if enable_thinking is defined and enable_thinking is false %}
    {{- '<think>\n\n</think>\n\n' }}
```
But this requires `enable_thinking` to be in the template context, not the API request body.

4. **vasp-03 (27B) uses a DIFFERENT template** — JSON-based tool calls:
```
<tool_call>
{"name": "<function-name>", "arguments": <args-json-object>}
</tool_call>
```
This is the older Qwen3 template format, different from Qwen3.5's XML format.

---

## Action Items (Priority Order)

### P0: Infrastructure Changes (No Code Required)

1. **Increase context on vasp-01/02:** Change `--ctx-size 32768` to `--ctx-size 65536` (or at minimum `49152`). This is the single most impactful fix — without it, workers overflow context by turn 3-4.

2. **Add `--reasoning-format none`** to vasp-01/02 startup. This disables `<think>` blocks entirely, saving context tokens and preventing parser confusion.

3. **Restart both 122B instances** with the updated flags.

### P1: Software Fixes (Code Changes)

4. **Client-side tool call validation:** In the tool execution path (`runtime_adapter.rs` or equivalent), after deserializing tool call arguments, check that all `required` fields are present before executing. Return a clear error if not.

5. **Repeat-call circuit breaker:** Track `(tool_name, args_hash)` per agent conversation. After 3 identical calls, inject a synthetic response forcing the agent to move on.

6. **Deploy pending fixes to ai-proxy:** Pull `6cad17e` (shared.rs test fix) and rebuild.

### P2: Deeper Investigation

7. **Rig v0.30.0 `reasoning_content` handling:** Does Rig capture `reasoning_content` from OpenAI-compatible responses? If not, file an issue or patch locally. The Qwen3.5 template corrupts multi-turn conversations without it.

8. **Verify `enable_thinking: false` is reaching the template:** Add logging to show whether `<think>` blocks appear in model responses. If they do despite `enable_thinking: false`, the flag isn't being passed correctly.

9. **Upgrade llama.cpp:** Build b8256+ from source on vasp-03 (build node), deploy to all nodes. Key fixes: #20198, #20260, #20352.

10. **Add `"strict": true` to tool JSON schemas** and test whether it improves grammar enforcement.

### P3: Operational

11. **Kill zombie processes:** Add `pkill -P` or process group cleanup to worktree teardown.

12. **Add `--kill-stale-procs` to dogfood-loop.sh** to clean up before each run.

---

## Reproduction Steps

To reproduce the full failure chain:
```bash
ssh brian@100.105.113.58
cd ~/code/beefcake-swarm
env SWARM_CLOUD_API_KEY=rust-daq-proxy-key \
    SWARM_CLOUD_URL=http://localhost:8317/v1 \
    SWARM_REQUIRE_ANTHROPIC_OWNERSHIP=0 \
    RUST_LOG=debug,hyper=info,reqwest=info,h2=info,rustls=info,tower=info \
    timeout 300 bash scripts/run-swarm.sh \
    --issue test-edit \
    --objective 'In crates/swarm-agents/src/tools/astgrep_tool.rs, rename DEFAULT_TIMEOUT_SECS to AST_GREP_TIMEOUT_SECS and change value from 30 to 60'
```

Expected: Worker edits file, verifier passes, issue closes.
Actual: Worker makes first edit correctly, drops `old_content` on subsequent edits, context overflows by turn 4, no verifier pass.

---

## Log Files

All logs are on ai-proxy at `~/code/beefcake-swarm/logs/dogfood/`:
- `run-3-beefcake-s9pz-20260315-232339.log` (866KB) — Best run: successful edit, then infinite search loop + context overflow
- `run-1-beefcake-jdxc-20260315-221629.log` (727KB) — Zombie process log, shows edit_file field omission pattern
- `run-1-beefcake-jdxc-20260315-225957.log` (224KB) — Second jdxc attempt, shows 500 parser crash

Main loop log: `~/dogfood-no-search-tools.log`
