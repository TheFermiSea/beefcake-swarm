"""
Architect-coder loop for beefcake-swarm.

Two-agent design: MiniMax-M2.7 (via TZ's `code_patch_architect` function)
produces a unified diff that resolves the issue. A cheap, deterministic
"coder" step applies the diff with `git apply` and runs the existing Rust
verifier. On failure, we re-engage the architect with the specific error
(apply conflict or verifier stderr tail) and ask for a revised diff.

Philosophy: MiniMax is a 300B MoE — much stronger per-turn than our
local workers, but 50× slower. Compressing the agent loop from
"N turns of bash commands" to "one diff + apply" exploits the asymmetry
(arXiv 2603.26458 — strong manager + weak worker beats strong single
agent at comparable cost) and sidesteps the "thinks aloud in content"
artefact of MiniMax by only consuming the final diff block.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import time
from typing import Any

import litellm

DEFAULT_TZ_URL = "http://localhost:3000"
DEFAULT_FUNCTION = "code_patch_architect"
MAX_FILE_BYTES = 60_000  # ~15k tokens; trim very long files so the prompt fits
ARCHITECT_TIMEOUT = 1_800  # 30 min; MiniMax can be slow on long outputs

SYSTEM_PROMPT = """You are a senior software engineer. Given a task description
and the current contents of the relevant file(s), respond with a git unified
diff that resolves the task.

Output format:
  • Respond with ONLY the diff, wrapped in a ```diff ... ``` fenced code block.
  • The diff MUST apply cleanly with `git apply` against the current worktree.
  • Include proper `diff --git a/<path> b/<path>` headers, `---` / `+++` lines
    with `a/` and `b/` prefixes, `@@` hunk headers with accurate line ranges,
    and at least 3 lines of unchanged context around each hunk.
  • For multi-file changes, emit multiple `diff --git` sections in one block.
  • Do NOT add prose, plan explanations, or step-by-step reasoning. The
    harness consumes only the diff. Everything else is noise and will be
    discarded.

If you cannot produce a correct diff (e.g. insufficient context), respond
with exactly:
  NEED_CONTEXT: <path/to/file/you/need>
and the harness will re-invoke you with that file's contents attached."""


# ───────────────────────────────────────────────────────────────────────────────
# Prompt assembly
# ───────────────────────────────────────────────────────────────────────────────

def _parse_target_files(issue: dict[str, Any]) -> list[pathlib.PurePosixPath]:
    """Extract likely target paths from the issue description.

    Beefcake issues include `Location: path/to/file.rs:lineno` lines. We also
    look for any standalone `relative/path.rs` or `.py` / `.go` / `.ts` token.
    """
    files: list[pathlib.PurePosixPath] = []
    desc = issue.get("description") or ""
    for m in re.finditer(r"Location:\s*(\S+?):\d+", desc):
        files.append(pathlib.PurePosixPath(m.group(1)))
    # Fallback: any identifier-shaped file ref
    if not files:
        for m in re.finditer(r"[\w./-]+\.(?:rs|py|ts|go|md|toml)", desc):
            files.append(pathlib.PurePosixPath(m.group(0)))
    # Dedup, preserve order
    seen, out = set(), []
    for f in files:
        if str(f) not in seen:
            out.append(f); seen.add(str(f))
    return out


def _read_file_clipped(path: pathlib.Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    try:
        raw = path.read_text(errors="replace")
    except Exception:
        return None
    if len(raw) > MAX_FILE_BYTES:
        raw = raw[: MAX_FILE_BYTES] + f"\n\n[... truncated; full file is {len(raw)} bytes ...]\n"
    return raw


def build_architect_prompt(
    issue: dict[str, Any],
    worktree: pathlib.Path,
    prior_attempts: list[dict] | None = None,
    extra_files: list[pathlib.PurePosixPath] | None = None,
) -> list[dict]:
    """Assemble the architect's messages. Retry attempts are appended as an
    alternating assistant/user conversation so MiniMax sees its own prior
    output + the specific feedback that invalidated it."""
    parts: list[str] = []
    parts.append(f"# Task {issue['id']}")
    parts.append(f"## Title\n{issue['title']}")
    if d := issue.get("description"):
        parts.append(f"## Description\n{d}")

    target_files = _parse_target_files(issue)
    if extra_files:
        for f in extra_files:
            if f not in target_files:
                target_files.append(f)

    if target_files:
        parts.append("## Current file contents")
        for rel in target_files:
            abs_path = worktree / rel
            content = _read_file_clipped(abs_path)
            if content is None:
                parts.append(f"### {rel}\n(file does not exist yet)")
            else:
                parts.append(f"### {rel}\n```\n{content}\n```")
    else:
        # No target specified: give the architect a dir listing so it can ask for context.
        try:
            listing = subprocess.run(
                ["git", "ls-files"], cwd=worktree, capture_output=True, text=True, timeout=15
            ).stdout.strip().splitlines()
            parts.append("## Repo file list (first 200)\n```\n" +
                         "\n".join(listing[:200]) + "\n```")
        except Exception:
            pass

    parts.append("Produce the unified diff now:")

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]
    for attempt in (prior_attempts or []):
        messages.append({"role": "assistant", "content": attempt["response"]})
        messages.append({"role": "user", "content": attempt["feedback"]})
    return messages


# ───────────────────────────────────────────────────────────────────────────────
# Diff extraction
# ───────────────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(
    r"```(?:diff|patch)?\s*\n(?P<body>diff --git .*?)```",
    re.DOTALL,
)


def extract_unified_diff(response_text: str) -> str | None:
    """Pull the unified diff out of a model response."""
    if not response_text:
        return None
    m = _FENCE_RE.search(response_text)
    if m:
        return m.group("body").rstrip() + "\n"
    # Some models emit a raw diff without fences
    if "diff --git " in response_text:
        idx = response_text.index("diff --git ")
        tail = response_text[idx:]
        # Stop at next ``` or end
        end = tail.find("```")
        return (tail[:end] if end >= 0 else tail).rstrip() + "\n"
    return None


def needs_more_context(response_text: str) -> str | None:
    """Detect the NEED_CONTEXT: sentinel the architect can emit."""
    m = re.search(r"NEED_CONTEXT:\s*([^\n]+)", response_text or "")
    return m.group(1).strip() if m else None


# ───────────────────────────────────────────────────────────────────────────────
# Diff application
# ───────────────────────────────────────────────────────────────────────────────

def apply_diff(diff: str, worktree: pathlib.Path) -> dict:
    """`git apply --check` then `git apply`. Returns {ok, error?, files?}."""
    try:
        chk = subprocess.run(
            ["git", "apply", "--check", "-"],
            input=diff, cwd=worktree,
            capture_output=True, text=True, timeout=30,
        )
        if chk.returncode != 0:
            return {"ok": False, "error": f"git apply --check failed:\n{chk.stderr.strip()}"}

        r = subprocess.run(
            ["git", "apply", "-"],
            input=diff, cwd=worktree,
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            return {"ok": False, "error": f"git apply failed:\n{r.stderr.strip()}"}

        changed = subprocess.run(
            ["git", "diff", "--name-only", "HEAD"],
            cwd=worktree, capture_output=True, text=True, timeout=15,
        )
        files = [ln for ln in changed.stdout.splitlines() if ln.strip()]
        return {"ok": True, "files": files}
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "error": f"apply timeout: {e}"}
    except Exception as e:
        return {"ok": False, "error": f"apply exception: {type(e).__name__}: {e}"}


def reset_worktree(worktree: pathlib.Path) -> None:
    """Discard any un-committed changes so the next apply starts clean."""
    subprocess.run(["git", "reset", "--hard", "HEAD"], cwd=worktree,
                   capture_output=True, timeout=30)
    subprocess.run(["git", "clean", "-fd"], cwd=worktree,
                   capture_output=True, timeout=30)


# ───────────────────────────────────────────────────────────────────────────────
# Architect call
# ───────────────────────────────────────────────────────────────────────────────

def call_architect(
    messages: list[dict],
    *,
    tz_url: str = DEFAULT_TZ_URL,
    function_name: str = DEFAULT_FUNCTION,
    variant_name: str | None = None,
    timeout_s: int = ARCHITECT_TIMEOUT,
) -> str:
    """Route through TZ's function_name for architect calls.

    `variant_name=None` uses TZ's configured weights (currently MiniMax 100%).
    Pin a specific variant by passing e.g. variant_name="claude_sonnet" for
    the cloud fallback path.
    """
    if variant_name:
        model = f"openai/tensorzero::function_name::{function_name}::variant_name::{variant_name}"
    else:
        model = f"openai/tensorzero::function_name::{function_name}"

    resp = litellm.completion(
        model=model,
        messages=messages,
        api_base=f"{tz_url.rstrip('/')}/openai/v1",
        api_key="tensorzero",
        timeout=timeout_s,
        temperature=0.0,
    )
    return resp.choices[0].message.content or ""


# ───────────────────────────────────────────────────────────────────────────────
# Main loop
# ───────────────────────────────────────────────────────────────────────────────

def run_architect_coder(
    issue: dict[str, Any],
    worktree: pathlib.Path,
    verifier_fn,
    *,
    max_iters: int = 3,
    tz_url: str = DEFAULT_TZ_URL,
    function_name: str = DEFAULT_FUNCTION,
    variant_name: str | None = None,
) -> dict:
    """Loop: architect → apply → verify → feedback on failure.

    `verifier_fn(worktree) -> {all_green: bool, gates: {name: {passed, stderr_tail, ...}}}`
    """
    attempts: list[dict] = []
    t0 = time.time()
    total_tokens = {"prompt": 0, "completion": 0}
    extra_files: list[pathlib.PurePosixPath] = []

    for iteration in range(1, max_iters + 1):
        messages = build_architect_prompt(issue, worktree,
                                          prior_attempts=attempts,
                                          extra_files=extra_files)
        try:
            response = call_architect(
                messages, tz_url=tz_url,
                function_name=function_name, variant_name=variant_name,
            )
        except Exception as e:
            return {
                "status": "failed", "iterations": iteration,
                "reason": f"architect call failed: {type(e).__name__}: {e}",
                "wall_s": round(time.time() - t0, 2),
            }

        # Architect may ask for additional context instead of producing a diff.
        if ctx := needs_more_context(response):
            ctx_path = pathlib.PurePosixPath(ctx)
            if ctx_path not in extra_files:
                extra_files.append(ctx_path)
                attempts.append({"response": response,
                                 "feedback": f"Attached {ctx_path}. Now emit the diff."})
                continue

        diff = extract_unified_diff(response)
        if not diff:
            attempts.append({
                "response": response,
                "feedback": ("I couldn't find a unified diff in your response. "
                             "Output ONLY a ```diff ... ``` fenced block. "
                             "No prose, no reasoning, no prefix text."),
            })
            continue

        applied = apply_diff(diff, worktree)
        if not applied["ok"]:
            reset_worktree(worktree)
            attempts.append({
                "response": response,
                "feedback": (f"Your diff did not apply cleanly:\n\n{applied['error']}\n\n"
                             "Produce a revised diff that applies against the current state. "
                             "Pay close attention to line numbers, context lines, and file paths."),
            })
            continue

        verify = verifier_fn(worktree)
        if verify.get("all_green"):
            return {
                "status": "resolved",
                "iterations": iteration,
                "wall_s": round(time.time() - t0, 2),
                "files_modified": applied["files"],
                "diff": diff,
                "attempts": len(attempts) + 1,
            }

        # Verifier failed: feed the stderr back to the architect.
        failing_gate = next(
            ((name, g) for name, g in (verify.get("gates") or {}).items() if not g.get("passed")),
            (None, {}),
        )
        gate_name, gate = failing_gate
        err_tail = (gate.get("stderr_tail") or "")[-3000:]
        reset_worktree(worktree)
        attempts.append({
            "response": response,
            "feedback": (
                f"Your diff applied, but the `{gate_name}` gate failed:\n\n"
                f"```\n{err_tail}\n```\n\nProduce a revised diff that fixes the above."
            ),
        })

    return {
        "status": "failed",
        "iterations": max_iters,
        "reason": f"exhausted max_iters={max_iters}",
        "wall_s": round(time.time() - t0, 2),
        "attempts": len(attempts),
    }
