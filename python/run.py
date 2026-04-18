"""
beefcake-swarm: single-issue end-to-end runner (Phase 2 replacement for
`cargo run -p swarm-agents`).

Given a beads issue id or an --objective, create a worktree, run mini-SWE-agent
inside it, run the verifier, and merge + close on green. Designed to slot in
where scripts/run-swarm.sh currently invokes the Rust binary.

Usage:
  python run.py --issue beefcake-abc123
  python run.py --issue manual-probe --objective 'Reply with OK'
  python run.py --issue beefcake-abc123 --repo-root /path/to/other/repo
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from swarm_worker import run_worker  # reuse the Phase 1 worker

DEFAULT_WT_ROOT = pathlib.Path("/tmp/beefcake-wt")
# Routes through TensorZero. Default is the `worker_code_edit` function (local
# variants A/B-weighted). `SWARM_WORKER_MODEL` lets callers pick a different
# TZ function name, a `tensorzero::…` routing string, or — with
# SWARM_USE_CLOUD=1 — a cloud model alias (claude-sonnet-4-6, etc.).
DEFAULT_MODEL = os.environ.get("SWARM_WORKER_MODEL", "worker_code_edit")


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ──────────────────────────────────────────────────────────────────────────────

def sh(cmd: list[str], cwd: pathlib.Path | None = None,
       timeout: int = 600, env: dict | None = None,
       check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        timeout=timeout, env=env, check=check,
    )


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}Z] {msg}",
          flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Beads integration
# ──────────────────────────────────────────────────────────────────────────────

def bd_bin() -> str:
    return os.environ.get("SWARM_BEADS_BIN", "bd")


def bd_show(issue_id: str) -> dict:
    out = sh([bd_bin(), "show", issue_id, "--json"], timeout=60)
    if out.returncode != 0:
        return {"id": issue_id, "title": issue_id, "description": "", "priority": 2}
    try:
        doc = json.loads(out.stdout)
        if isinstance(doc, list) and doc:
            doc = doc[0]
        return {
            "id": doc.get("id", issue_id),
            "title": doc.get("title", issue_id),
            "description": doc.get("description", ""),
            "priority": doc.get("priority", 2),
        }
    except Exception:
        return {"id": issue_id, "title": issue_id, "description": "", "priority": 2}


def bd_claim(issue_id: str) -> bool:
    out = sh([bd_bin(), "update", issue_id, "--claim"], timeout=30)
    return out.returncode == 0


def bd_close(issue_id: str, reason: str = "resolved by mini-swarm") -> bool:
    out = sh([bd_bin(), "close", issue_id, "--reason", reason], timeout=30)
    return out.returncode == 0


def bd_note(issue_id: str, note: str) -> None:
    # Non-fatal
    sh([bd_bin(), "update", issue_id, "--notes", note[:4000]], timeout=30)


# ──────────────────────────────────────────────────────────────────────────────
# NotebookLM pre-task KB query (failsafe — matches current Rust behavior)
# ──────────────────────────────────────────────────────────────────────────────

def nlm_query(notebook_id: str, question: str, timeout: int = 90) -> str:
    """Query NotebookLM; return the answer text or an empty string on failure.

    Mirrors the Rust `query_kb_with_failsafe` helper: KB unavailability must
    never block the loop. But unlike the old path we log loud WARNINGs and
    exit fast on auth errors (HTTP 400 for a week is what we just fixed)."""
    if not notebook_id:
        return ""
    try:
        out = sh(["nlm", "query", "notebook", notebook_id, question],
                 timeout=timeout)
        if out.returncode != 0:
            log(f"WARN KB query failed ({notebook_id[:8]}…): "
                f"{(out.stderr or out.stdout)[-200:]}")
            return ""
        # nlm query emits JSON with {"value": {"answer": "..."}}
        try:
            doc = json.loads(out.stdout)
            ans = (doc.get("value") or {}).get("answer") or ""
            return ans[:6000]  # cap context growth
        except Exception:
            return out.stdout.strip()[:6000]
    except subprocess.TimeoutExpired:
        log(f"WARN KB query timed out ({notebook_id[:8]}…)")
        return ""
    except Exception as e:
        log(f"WARN KB query error ({notebook_id[:8]}…): {e}")
        return ""


def load_notebook_registry(repo_root: pathlib.Path) -> dict:
    path = repo_root / "notebook_registry.toml"
    if not path.exists():
        return {}
    try:
        import tomllib
        with path.open("rb") as f:
            return tomllib.load(f).get("notebooks", {}) or {}
    except Exception as e:
        log(f"WARN could not parse notebook_registry.toml: {e}")
        return {}


# ──────────────────────────────────────────────────────────────────────────────
# Worktree lifecycle
# ──────────────────────────────────────────────────────────────────────────────

def worktree_create(repo_root: pathlib.Path, issue_id: str,
                    base_branch: str = "main") -> pathlib.Path:
    wt_root = pathlib.Path(os.environ.get("SWARM_WT_ROOT", DEFAULT_WT_ROOT))
    wt_path = wt_root / issue_id
    if wt_path.exists():
        shutil.rmtree(wt_path, ignore_errors=True)
    wt_root.mkdir(parents=True, exist_ok=True)
    sh(["git", "worktree", "prune"], cwd=repo_root, timeout=60)
    branch = f"swarm/{issue_id}"
    # -B recreates the branch if it exists
    sh(["git", "worktree", "add", "-B", branch, str(wt_path), base_branch],
       cwd=repo_root, timeout=120, check=True)
    return wt_path


def worktree_has_changes(wt_path: pathlib.Path) -> bool:
    r = sh(["git", "status", "--porcelain"], cwd=wt_path, timeout=30)
    return bool(r.stdout.strip())


def worktree_commit_if_needed(wt_path: pathlib.Path, issue_id: str) -> bool:
    if not worktree_has_changes(wt_path):
        # Worker may have already committed; just ensure there's something new
        r = sh(["git", "log", "-1", "--oneline"], cwd=wt_path, timeout=15)
        return r.returncode == 0 and bool(r.stdout.strip())
    sh(["git", "add", "-A"], cwd=wt_path, timeout=60)
    msg = f"swarm: resolve {issue_id}\n\nAutomated change by mini-swarm worker."
    r = sh(["git", "commit", "-m", msg], cwd=wt_path, timeout=60)
    return r.returncode == 0


def worktree_merge(repo_root: pathlib.Path, issue_id: str,
                   target: str = "main") -> bool:
    branch = f"swarm/{issue_id}"
    # Fast-forward or merge --no-ff — we use no-ff to preserve branch history
    r = sh(["git", "merge", "--no-ff", "-m", f"swarm: merge {issue_id}", branch],
           cwd=repo_root, timeout=120)
    return r.returncode == 0


def worktree_cleanup(repo_root: pathlib.Path, wt_path: pathlib.Path,
                     issue_id: str) -> None:
    sh(["git", "worktree", "remove", "--force", str(wt_path)],
       cwd=repo_root, timeout=60)
    if wt_path.exists():
        shutil.rmtree(wt_path, ignore_errors=True)
    sh(["git", "worktree", "prune"], cwd=repo_root, timeout=30)


# ──────────────────────────────────────────────────────────────────────────────
# Verifier (cargo check + test; clippy/fmt-check deferred until main is clean)
# ──────────────────────────────────────────────────────────────────────────────

def run_verifier(wt_path: pathlib.Path, skip_tests: bool = False) -> dict:
    gates: dict[str, dict] = {}
    env = {**os.environ, "CARGO_TARGET_DIR": str(wt_path / "target-swarm")}

    # Auto-fmt (normalization — not a gate)
    sh(["cargo", "fmt", "--all"], cwd=wt_path, timeout=120, env=env)

    cmds = [("check", ["cargo", "check", "--workspace", "--all-targets"], 900)]
    if not skip_tests:
        cmds.append(("test", ["cargo", "test", "--workspace", "--",
                              "--test-threads=1"], 1800))

    for name, cmd, timeout in cmds:
        t0 = time.time()
        try:
            r = sh(cmd, cwd=wt_path, timeout=timeout, env=env)
            gates[name] = {
                "passed": r.returncode == 0,
                "duration_s": round(time.time() - t0, 1),
                "stderr_tail": r.stderr[-2000:] if r.stderr else "",
            }
            if r.returncode != 0:
                break
        except subprocess.TimeoutExpired:
            gates[name] = {"passed": False, "duration_s": timeout,
                           "stderr_tail": "<timeout>"}
            break
    return {"all_green": all(g["passed"] for g in gates.values()),
            "gates": gates}


# ──────────────────────────────────────────────────────────────────────────────
# Main orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def process_issue(issue_id: str, *, repo_root: pathlib.Path, model: str,
                  objective: str | None = None, skip_tests: bool = False,
                  keep_on_fail: bool = True, close_on_success: bool = True,
                  query_kb: bool = True,
                  architect_coder: bool = False,
                  architect_function: str = "code_patch_architect",
                  architect_variant: str | None = None,
                  architect_max_iters: int = 3) -> dict:
    t0 = time.time()
    result: dict = {
        "issue_id": issue_id, "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # 1. Resolve issue fields
    if objective:
        issue = {"id": issue_id, "title": objective[:80],
                 "description": objective, "priority": 2}
    else:
        issue = bd_show(issue_id)

    log(f"process_issue: id={issue_id} title={issue['title'][:80]!r}")

    # 2. Pre-task KB enrichment
    kb_context: dict[str, str] = {}
    if query_kb and not objective:
        registry = load_notebook_registry(repo_root)
        if pb := (registry.get("project_brain") or {}).get("id"):
            q = (f"Architectural context for beads issue '{issue['title']}'. "
                 f"What patterns, prior decisions, or gotchas apply?")
            kb_context["project_brain"] = nlm_query(pb, q)
        if db := (registry.get("debugging_kb") or {}).get("id"):
            q = (f"Known fixes or error patterns for: {issue['title']}. "
                 f"Description: {issue.get('description','')[:500]}")
            kb_context["debugging_kb"] = nlm_query(db, q)
    result["kb_hits"] = {k: bool(v) for k, v in kb_context.items()}

    # 3. Worktree
    try:
        wt_path = worktree_create(repo_root, issue_id)
    except Exception as e:
        result.update({"status": "failed", "reason": f"worktree create: {e}",
                       "wall_s": round(time.time() - t0, 1)})
        return result
    log(f"worktree: {wt_path}")

    # 4. Claim issue (non-fatal on failure — may be raced or non-beads ID)
    if not objective:
        bd_claim(issue_id)

    # 5. Invoke inner loop — either architect-coder (MiniMax emits a diff,
    # we apply+verify with retry) or mini-SWE-agent bash-loop (the default).
    if architect_coder:
        log(f"architect-coder mode: function={architect_function} "
            f"variant={architect_variant or 'weighted'}")
        from architect import run_architect_coder
        arch = run_architect_coder(
            issue, wt_path,
            verifier_fn=lambda wt: run_verifier(wt, skip_tests=skip_tests),
            max_iters=architect_max_iters,
            function_name=architect_function,
            variant_name=architect_variant,
        )
        result["architect"] = {k: arch.get(k) for k in
                               ("status", "iterations", "wall_s",
                                "files_modified", "attempts", "reason")}
        # Translate architect result into the same shape downstream code expects.
        if arch["status"] == "resolved":
            outcome = {"status": "produced_changes", "exit_status": "Submitted",
                       "files_modified": arch.get("files_modified", []),
                       "wall_time_s": arch.get("wall_s"),
                       "iterations": arch.get("iterations"),
                       "cost_usd": 0.0}
            # Architect's apply_diff left files in the worktree uncommitted.
            # Downstream commit step below picks them up.
        else:
            outcome = {"status": "failed",
                       "failure_reason": arch.get("reason", "architect failed"),
                       "wall_time_s": arch.get("wall_s")}
    else:
        req = {
            "issue": issue,
            "worktree_path": str(wt_path),
            "iteration": 1,
            "tier": "Worker",
            "model_config": {
                "model_name": model,
                "max_tool_calls": 60,
                "cost_limit_usd": 8.0,
                "deadline_secs": 1800,
            },
            "knowledge_base": kb_context,
            "prior_context": {},
        }
        try:
            outcome = run_worker(req)
        except Exception as e:
            outcome = {"status": "failed", "failure_reason": f"{type(e).__name__}: {e}"}
    result["worker"] = {k: outcome.get(k) for k in
                        ("status", "exit_status", "iterations",
                         "wall_time_s", "cost_usd", "files_modified")}

    # 6. Verifier — skip re-verifying for architect-coder (architect loop
    # already ran + passed the verifier before returning "resolved"). For
    # the mini-SWE-agent path we still gate on a fresh verifier run.
    if outcome.get("status") == "produced_changes":
        committed = worktree_commit_if_needed(wt_path, issue_id)
        result["worker"]["committed"] = committed
        if architect_coder:
            result["verifier"] = {"all_green": True, "gates": {"architect_verified": {"passed": True}}}
            ver_green = committed
        else:
            ver = run_verifier(wt_path, skip_tests=skip_tests)
            result["verifier"] = ver
            ver_green = ver["all_green"] and committed
        if ver_green:
            # 7. Merge + close
            if worktree_merge(repo_root, issue_id):
                result["merged"] = True
                if close_on_success and not objective:
                    result["closed"] = bd_close(issue_id)
                else:
                    result["closed"] = False
            else:
                result["merged"] = False
                bd_note(issue_id, f"merge failed for swarm/{issue_id}; branch retained")
        else:
            bd_note(issue_id, f"verifier failed or uncommitted for {issue_id}")
    else:
        result["verifier"] = None

    # 8. Cleanup
    if result.get("merged") or not keep_on_fail:
        worktree_cleanup(repo_root, wt_path, issue_id)
    # else: leave branch + worktree on disk for inspection

    # 9. Final status
    if result.get("merged") and result.get("verifier", {}).get("all_green"):
        result["status"] = "resolved"
    elif outcome.get("status") == "produced_changes" and not result.get("verifier", {}).get("all_green"):
        result["status"] = "changes_failed_verifier"
    elif outcome.get("status") == "no_change":
        result["status"] = "no_change"
    else:
        result["status"] = "failed"
    result["wall_s"] = round(time.time() - t0, 1)
    return result


def append_summary(row: dict, path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issue", required=True, help="Beads issue ID")
    ap.add_argument("--objective", help="Free-form task (bypasses bd show/claim/close)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repo-root", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--no-close", action="store_true",
                    help="Don't close the beads issue on success (for debugging)")
    ap.add_argument("--architect-coder", action="store_true",
                    help="Use the architect-coder flow (MiniMax emits a unified "
                         "diff, we apply+verify, retry on failure) instead of "
                         "the mini-SWE-agent bash loop.")
    ap.add_argument("--architect-function", default="code_patch_architect",
                    help="TZ function name for the architect call.")
    ap.add_argument("--architect-variant",
                    help="Pin to a specific TZ variant (e.g. 'claude_sonnet'). "
                         "Default: use the function's weighted selection.")
    ap.add_argument("--architect-max-iters", type=int, default=3,
                    help="Max architect-coder retry iterations (default 3).")
    ap.add_argument("--no-kb", action="store_true",
                    help="Skip NotebookLM pre-task queries")
    ap.add_argument("--summary-path", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent
                    / "logs" / "mini-swarm" / "summary.jsonl")
    args = ap.parse_args()

    row = process_issue(
        args.issue,
        repo_root=args.repo_root.resolve(),
        model=args.model,
        objective=args.objective,
        skip_tests=args.skip_tests,
        close_on_success=not args.no_close,
        query_kb=not args.no_kb,
        architect_coder=args.architect_coder,
        architect_function=args.architect_function,
        architect_variant=args.architect_variant,
        architect_max_iters=args.architect_max_iters,
    )
    append_summary(row, args.summary_path)
    log(f"DONE status={row['status']} wall={row['wall_s']}s "
        f"merged={row.get('merged')} closed={row.get('closed')}")
    print(json.dumps(row, indent=2))
    return 0 if row["status"] == "resolved" else 2


if __name__ == "__main__":
    sys.exit(main())
