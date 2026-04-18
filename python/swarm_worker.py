"""
beefcake-swarm Python worker — Phase 1 spike.

Reads a WorkerRequest JSON on stdin, runs mini-SWE-agent inside the supplied
git worktree, emits a WorkerOutcome JSON on stdout. The Rust dispatcher owns
beads, worktree lifecycle, verifier, and knowledge-base queries; this worker
is responsible only for the inner LLM-tool loop.

Phase 1: direct CLIAPIProxy endpoint (no LiteLLM Router — added in Phase 2).
Uses mini-swe-agent's bundled default.yaml for the text-based bash extraction
loop (system/instance/observation templates).

Run: python swarm_worker.py --stdin-json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import traceback
from typing import Any

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# ──────────────────────────────────────────────────────────────────────────────
# Model / config setup
# ──────────────────────────────────────────────────────────────────────────────

def _load_bundled_default_config() -> dict:
    """Load mini-swe-agent's bundled default.yaml (agent section)."""
    from minisweagent import config as mswea_config_pkg
    pkg_dir = pathlib.Path(mswea_config_pkg.__file__).parent
    with (pkg_dir / "default.yaml").open() as f:
        return yaml.safe_load(f) or {}


def build_model(model_name: str, *, timeout_s: int = 300):
    """Build a LitellmTextbasedModel routed through the TensorZero gateway.

    Every inference flows through TZ (localhost:3000) for observability + A/B
    variant selection + automatic logging to Postgres/Clickhouse. TZ then
    routes to the backing model (local llama.cpp or cloud via CLIAPIProxy)
    based on the variant config in config/tensorzero.toml.

    `model_name` interpretation:
      • `tensorzero::…`              — use as-is (caller knows TZ routing).
      • `claude-*|gpt-*|gemini-*`    — only valid with SWARM_USE_CLOUD=1;
                                       routes via `tensorzero::model_name::…`.
                                       Without that flag, falls back to the
                                       default function (local variants) so
                                       we don't quietly burn quota.
      • anything else                — treated as a TZ function_name
                                       (default: `worker_code_edit`).
    """
    from minisweagent.models.litellm_textbased_model import LitellmTextbasedModel

    tz_base = os.environ.get("SWARM_TENSORZERO_URL",
                             "http://localhost:3000").rstrip("/")
    api_base = f"{tz_base}/openai/v1"

    CLOUD_PREFIXES = ("claude-", "gpt-", "gemini-")
    if model_name.startswith("tensorzero::"):
        tz_model = model_name
    elif any(model_name.startswith(p) for p in CLOUD_PREFIXES):
        if os.environ.get("SWARM_USE_CLOUD") != "1":
            sys.stderr.write(
                f"WARN build_model: cloud model {model_name!r} requested without "
                f"SWARM_USE_CLOUD=1 — falling back to tz function_name="
                f"worker_code_edit (local variants) so we don't burn quota.\n"
            )
            tz_model = "tensorzero::function_name::worker_code_edit"
        else:
            tz_model = f"tensorzero::model_name::{model_name}"
    else:
        tz_model = f"tensorzero::function_name::{model_name}"

    full_model = f"openai/{tz_model}"

    return LitellmTextbasedModel(
        model_name=full_model,
        model_kwargs={
            "api_base": api_base,
            "api_key": "tensorzero",  # TZ ignores the value; LiteLLM requires one
            "timeout": timeout_s,
            "temperature": 0.0,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task prompt assembly
# ──────────────────────────────────────────────────────────────────────────────

def build_task_prompt(req: dict) -> str:
    """Compose the task string that mini-swe-agent renders into
    instance_template as {{task}}."""
    issue = req["issue"]
    parts = [
        f"# Beads issue {issue['id']}: {issue['title']}",
        "",
        issue.get("description", ""),
        "",
    ]
    if scope := req.get("scope_constraints"):
        if allowed := scope.get("allowed_files"):
            parts += ["## Scope", "Modify only these files:",
                      *[f"- {p}" for p in allowed], ""]
    kb = req.get("knowledge_base") or {}
    if ctx := kb.get("project_brain"):
        parts += ["## Architectural context (project_brain KB)", ctx, ""]
    if dbg := kb.get("debugging_kb"):
        parts += ["## Prior debugging patterns", dbg, ""]
    prior = req.get("prior_context") or {}
    if vr := prior.get("verifier_report"):
        parts += ["## Prior verifier report", "```", str(vr)[:4000], "```", ""]
    parts += [
        "## What counts as done",
        "A change that compiles (`cargo check`), passes lints (`cargo clippy -- -D warnings`),",
        "and leaves tests green (`cargo test`). Before you finish, run `cargo fmt --all` so",
        "the format-check gate passes. `cargo clippy --fix` is fine; it only fixes warnings.",
        "",
        "When you're done, run exactly `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`",
        "followed by a short bullet summary of what you changed. If you conclude",
        "the issue is already resolved or can't be resolved, explain why in that",
        "summary and submit anyway.",
    ]
    return "\n".join(parts)


# ──────────────────────────────────────────────────────────────────────────────
# Git helpers
# ──────────────────────────────────────────────────────────────────────────────

def git_diff(worktree: pathlib.Path) -> str:
    try:
        r = subprocess.run(["git", "diff", "HEAD"], cwd=worktree,
                           capture_output=True, text=True, timeout=30)
        return r.stdout
    except Exception as e:
        return f"<git-diff-failed: {e}>"


def git_changed_files(worktree: pathlib.Path) -> list[str]:
    try:
        r = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=worktree,
                           capture_output=True, text=True, timeout=30)
        return [ln for ln in r.stdout.splitlines() if ln.strip()]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────────────
# Worker entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_worker(req: dict) -> dict:
    from minisweagent.agents.default import DefaultAgent
    from minisweagent.environments.local import LocalEnvironment

    worktree = pathlib.Path(req["worktree_path"]).resolve()
    if not worktree.is_dir():
        return _err(f"worktree path does not exist: {worktree}")

    mc = req.get("model_config") or {}
    primary_model = mc.get("model_name") or "claude-sonnet-4-6"
    step_limit = int(mc.get("max_tool_calls") or 60)
    cost_limit = float(mc.get("cost_limit_usd") or 8.0)
    deadline_s = int(mc.get("deadline_secs") or 1800)

    agent_cfg = _load_bundled_default_config().get("agent", {})
    agent_cfg["step_limit"] = step_limit
    agent_cfg["cost_limit"] = cost_limit

    model = build_model(primary_model, timeout_s=300)
    env = LocalEnvironment(cwd=str(worktree), timeout=300)
    agent = DefaultAgent(model, env, **agent_cfg)

    task = build_task_prompt(req)
    started = time.time()
    exit_status = "unknown"
    submission: str | None = None
    try:
        info = agent.run(task)
        if isinstance(info, dict):
            exit_status = info.get("exit_status") or "unknown"
            submission = info.get("submission")
    except Exception as e:
        exit_status = f"error: {e.__class__.__name__}: {e}"
    wall = time.time() - started

    diff = git_diff(worktree)
    changed = git_changed_files(worktree)
    if exit_status.startswith("error"):
        status = "failed"
    elif not diff.strip():
        status = "no_change"
    else:
        status = "produced_changes"

    return {
        "status": status,
        "exit_status": exit_status,
        "submission": submission,
        "git_diff": diff,
        "files_modified": changed,
        "models_used": [primary_model],
        "iterations": getattr(model, "n_calls", None),
        "cost_usd": round(getattr(model, "cost", 0.0) or 0.0, 4),
        "wall_time_s": round(wall, 2),
        "deadline_exceeded": wall > deadline_s,
    }


def _err(reason: str) -> dict:
    return {
        "status": "failed", "failure_reason": reason,
        "git_diff": None, "files_modified": [],
        "models_used": [], "iterations": 0, "wall_time_s": 0, "cost_usd": 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdin-json", action="store_true",
                    help="Read WorkerRequest JSON on stdin; write WorkerOutcome JSON on stdout")
    ap.add_argument("--request-file", type=pathlib.Path,
                    help="Read WorkerRequest from a JSON file instead of stdin")
    args = ap.parse_args()

    if args.request_file:
        req = json.loads(args.request_file.read_text())
    elif args.stdin_json:
        req = json.loads(sys.stdin.read())
    else:
        ap.error("Pass --stdin-json or --request-file PATH")

    try:
        outcome = run_worker(req)
    except Exception as e:
        outcome = {
            **_err(f"worker crash: {e.__class__.__name__}: {e}"),
            "traceback": traceback.format_exc(),
        }

    json.dump(outcome, sys.stdout)
    sys.stdout.write("\n")
    sys.stdout.flush()
    return 0 if outcome.get("status") != "failed" else 2


if __name__ == "__main__":
    sys.exit(main())
