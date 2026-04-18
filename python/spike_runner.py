"""
Phase 1 spike orchestrator.

For each of N beads issues:
  1. Create a git worktree at /tmp/beefcake-mini-spike/<issue-id> off main.
  2. Invoke swarm_worker.py for that issue inside the worktree.
  3. Run the Rust verifier (cargo fmt/clippy/check/test) on the result.
  4. Record a summary row to logs/mini-spike-after/spike-summary.jsonl.

Runs serially by default (safer for a spike — one verifier at a time avoids
cargo target-dir contention). Pass --parallel N for thread-pool execution.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
WT_ROOT = pathlib.Path("/tmp/beefcake-mini-spike")
LOG_DIR = REPO_ROOT / "logs" / "mini-spike-after"
SUMMARY_PATH = LOG_DIR / "spike-summary.jsonl"
WORKER_PATH = REPO_ROOT / "python" / "swarm_worker.py"


def sh(cmd: list[str], cwd: pathlib.Path | None = None,
       timeout: int = 600, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True,
        timeout=timeout, check=check,
    )


def bd_show(issue_id: str) -> dict:
    """Fetch beads issue fields via bd."""
    out = sh(["bd", "show", issue_id, "--json"], cwd=REPO_ROOT, timeout=60)
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


def create_worktree(issue_id: str) -> pathlib.Path:
    """Create a fresh worktree off main for this issue."""
    wt_path = WT_ROOT / issue_id
    if wt_path.exists():
        shutil.rmtree(wt_path, ignore_errors=True)
        sh(["git", "worktree", "prune"], cwd=REPO_ROOT)
    WT_ROOT.mkdir(parents=True, exist_ok=True)
    branch = f"mini-spike/{issue_id}"
    sh(["git", "worktree", "add", "-B", branch, str(wt_path), "main"],
       cwd=REPO_ROOT, timeout=120)
    return wt_path


def cleanup_worktree(issue_id: str, wt_path: pathlib.Path, keep: bool = False):
    if keep:
        return
    try:
        sh(["git", "worktree", "remove", "--force", str(wt_path)],
           cwd=REPO_ROOT, timeout=60)
    except Exception:
        pass
    if wt_path.exists():
        shutil.rmtree(wt_path, ignore_errors=True)
    sh(["git", "worktree", "prune"], cwd=REPO_ROOT)
    # Also drop the transient branch
    sh(["git", "branch", "-D", f"mini-spike/{issue_id}"],
       cwd=REPO_ROOT, timeout=30)


def run_verifier(wt_path: pathlib.Path, skip_tests: bool = False) -> dict:
    """Run the quality gates. Returns per-gate status.

    Phase 1 gates: cargo check + cargo test. We auto-run `cargo fmt --all`
    as a normalization step (not a gate) and deliberately skip `cargo clippy
    -D warnings` because main has ~10 pre-existing clippy errors in unrelated
    code that would fail every run regardless of worker quality. We'll
    re-enable clippy once main is clippy-clean.
    """
    gates: dict[str, dict] = {}
    env = {**os.environ, "CARGO_TARGET_DIR": str(wt_path / "target-spike")}

    # Normalization — not a gate. Apply fmt so post-fmt-check diffs stay noise-free.
    subprocess.run(["cargo", "fmt", "--all"], cwd=wt_path,
                   capture_output=True, text=True, timeout=120, env=env)

    cargo_cmds = [
        ("check", ["cargo", "check", "--workspace", "--all-targets"], 900),
    ]
    if not skip_tests:
        cargo_cmds.append(("test", ["cargo", "test", "--workspace", "--", "--test-threads=1"], 1800))

    for name, cmd, timeout in cargo_cmds:
        t0 = time.time()
        try:
            r = subprocess.run(
                cmd, cwd=wt_path, capture_output=True, text=True,
                timeout=timeout, env=env,
            )
            gates[name] = {
                "passed": r.returncode == 0,
                "duration_s": round(time.time() - t0, 1),
                "stderr_tail": r.stderr[-2000:] if r.stderr else "",
            }
            if r.returncode != 0:
                break  # fail-fast, matches real pipeline
        except subprocess.TimeoutExpired:
            gates[name] = {"passed": False, "duration_s": timeout, "stderr_tail": "<timeout>"}
            break
        except Exception as e:
            gates[name] = {"passed": False, "duration_s": round(time.time() - t0, 1),
                           "stderr_tail": f"<spawn-error: {e}>"}
            break

    all_green = all(g.get("passed") for g in gates.values())
    return {"all_green": all_green, "gates": gates}


def run_one(issue_id: str, *, model_name: str, skip_tests: bool,
            keep_worktrees: bool, log_path: pathlib.Path) -> dict:
    t0 = time.time()
    wt_path = create_worktree(issue_id)
    issue = bd_show(issue_id)

    request = {
        "issue": issue,
        "worktree_path": str(wt_path),
        "iteration": 1,
        "tier": "Worker",
        "task_prompt": None,  # worker builds from issue fields
        "model_config": {
            "model_name": model_name,
            "max_tool_calls": 60,
            "cost_limit_usd": 8.0,
            "deadline_secs": 1800,
        },
        "knowledge_base": {},
        "prior_context": {},
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as logf:
        logf.write(f"# Spike run: {issue_id} / model={model_name} / "
                   f"start={datetime.utcnow().isoformat()}Z\n")
        logf.write(json.dumps(request, indent=2) + "\n\n=== worker stdout/stderr ===\n")
        logf.flush()
        worker_env = {
            **os.environ,
            # CLIAPIProxy-routed models aren't in LiteLLM's pricing DB; ignore
            # cost-tracking errors so the agent keeps running.
            "MSWEA_COST_TRACKING": "ignore_errors",
        }
        proc = subprocess.run(
            [sys.executable, str(WORKER_PATH), "--stdin-json"],
            input=json.dumps(request), capture_output=True, text=True,
            timeout=2400, env=worker_env,
        )
        logf.write(proc.stderr + "\n=== worker stdout (JSON) ===\n" + proc.stdout + "\n")

    try:
        outcome = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as e:
        outcome = {"status": "failed", "failure_reason": f"worker stdout parse: {e}",
                   "stdout_raw": proc.stdout[-2000:]}

    verifier = None
    if outcome.get("status") == "produced_changes":
        verifier = run_verifier(wt_path, skip_tests=skip_tests)
        outcome["verifier"] = verifier
        with log_path.open("a") as logf:
            logf.write("\n=== verifier ===\n" + json.dumps(verifier, indent=2) + "\n")

    wall = round(time.time() - t0, 1)
    row = {
        "issue_id": issue_id,
        "model": model_name,
        "worker_status": outcome.get("status"),
        "worker_exit_status": outcome.get("exit_status"),
        "verifier_green": (verifier or {}).get("all_green", False),
        "resolved": outcome.get("status") == "produced_changes"
                    and (verifier or {}).get("all_green", False),
        "iterations": outcome.get("iterations"),
        "files_modified": outcome.get("files_modified", []),
        "cost_usd": outcome.get("cost_usd"),
        "wall_time_s": wall,
        "submission": (outcome.get("submission") or "")[:300],
        "log_path": str(log_path),
    }

    # Don't merge; branch stays for inspection unless --no-keep
    cleanup_worktree(issue_id, wt_path, keep=keep_worktrees)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--issues", required=True,
                    help="Space-separated beads issue IDs")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="Primary model alias from litellm.yaml")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Thread-pool size (default: 1 — serial)")
    ap.add_argument("--skip-tests", action="store_true",
                    help="Skip cargo test (faster, less strict)")
    ap.add_argument("--keep-worktrees", action="store_true",
                    help="Leave worktrees on disk after each run for inspection")
    args = ap.parse_args()

    issue_ids = [s for s in args.issues.split() if s.strip()]
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Spike: {len(issue_ids)} issues, model={args.model}, "
          f"parallel={args.parallel}, skip_tests={args.skip_tests}")

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    results: list[dict] = []

    def _run(iid: str) -> dict:
        log_path = LOG_DIR / f"{iid}-{ts}.log"
        return run_one(iid, model_name=args.model,
                       skip_tests=args.skip_tests,
                       keep_worktrees=args.keep_worktrees,
                       log_path=log_path)

    if args.parallel <= 1:
        for iid in issue_ids:
            row = _run(iid)
            results.append(row)
            print(f"  {iid}: resolved={row['resolved']} "
                  f"worker={row['worker_status']} "
                  f"verifier={row['verifier_green']} "
                  f"files={len(row['files_modified'])} "
                  f"wall={row['wall_time_s']}s")
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futures = {ex.submit(_run, iid): iid for iid in issue_ids}
            for fut in as_completed(futures):
                row = fut.result()
                results.append(row)
                print(f"  {row['issue_id']}: resolved={row['resolved']} "
                      f"worker={row['worker_status']} "
                      f"verifier={row['verifier_green']} "
                      f"wall={row['wall_time_s']}s")

    with SUMMARY_PATH.open("a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    resolved = sum(1 for r in results if r["resolved"])
    changed = sum(1 for r in results if r["worker_status"] == "produced_changes")
    print(f"\n=== SPIKE RESULT: {resolved}/{len(results)} resolved, "
          f"{changed} produced changes ===")
    for r in results:
        print(f"  {r['issue_id']:20s} resolved={r['resolved']!s:5} "
              f"worker={r['worker_status']} wall={r['wall_time_s']}s "
              f"files={len(r['files_modified'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
