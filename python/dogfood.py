"""
beefcake-swarm: continuous dogfood loop (Phase 2 replacement for
scripts/dogfood-loop.sh's Rust-swarm invocations).

Pulls beads issues (from --issue-list or `bd ready`), runs them through
python/run.py's process_issue pipeline, appends summaries, loops.

Usage:
  python dogfood.py --discover --cooldown 120
  python dogfood.py --issue-list "beefcake-aaa beefcake-bbb" --max-runs 5
  python dogfood.py --discover --parallel 2 --max-runs 0  # unlimited
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import pathlib
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

from run import process_issue, append_summary, log, bd_bin

REPO_ROOT_DEFAULT = pathlib.Path(__file__).resolve().parent.parent
SUMMARY_DIR_DEFAULT = REPO_ROOT_DEFAULT / "logs" / "dogfood"
LOCK_PATH_TEMPLATE = "/tmp/dogfood-py-{repo_name}.lock"

_stop_requested = False


def _handle_sigterm(signum, frame):
    global _stop_requested
    _stop_requested = True
    log(f"signal {signum} received — finishing current batch then stopping")


def acquire_lock(repo_root: pathlib.Path) -> pathlib.Path:
    """Per-repo lockfile. Mirrors scripts/dogfood-loop.sh semantics so we don't
    race with the legacy loop during the transition period."""
    lock = pathlib.Path(LOCK_PATH_TEMPLATE.format(repo_name=repo_root.name))
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
            # Is the process still alive?
            os.kill(pid, 0)
            raise SystemExit(f"ERROR: Another dogfood loop is running (pid={pid}). "
                             f"Kill it first: kill {pid}")
        except ProcessLookupError:
            log(f"WARN stale lockfile {lock} (pid gone); claiming")
        except (ValueError, PermissionError):
            raise SystemExit(f"ERROR: Lock file {lock} present but unreadable")
    lock.write_text(str(os.getpid()))
    return lock


def release_lock(lock: pathlib.Path) -> None:
    try:
        lock.unlink(missing_ok=True)
    except Exception:
        pass


def discover_ready_issues(limit: int = 10) -> list[str]:
    """List issue IDs ready to work (no blockers, status=open)."""
    try:
        r = subprocess.run(
            [bd_bin(), "ready", "--json", "--limit", str(limit)],
            capture_output=True, text=True, timeout=60,
        )
        if r.returncode != 0:
            log(f"WARN bd ready failed: {r.stderr[:300]}")
            return []
        doc = json.loads(r.stdout)
        if isinstance(doc, dict):
            doc = doc.get("issues") or doc.get("data") or []
        return [d.get("id") for d in doc if d.get("id")]
    except Exception as e:
        log(f"WARN bd ready error: {e}")
        return []


def run_batch(issue_ids: list[str], *, repo_root: pathlib.Path, model: str,
              skip_tests: bool, parallel: int,
              summary_path: pathlib.Path) -> list[dict]:
    """Process a batch of issues. Returns outcome rows."""
    outcomes: list[dict] = []

    def _one(iid: str) -> dict:
        try:
            return process_issue(
                iid, repo_root=repo_root, model=model,
                skip_tests=skip_tests, close_on_success=True,
                query_kb=True,
            )
        except Exception as e:
            return {"issue_id": iid, "status": "failed",
                    "reason": f"driver crash: {type(e).__name__}: {e}",
                    "started_at": datetime.now(timezone.utc).isoformat()}

    if parallel <= 1:
        for iid in issue_ids:
            row = _one(iid)
            outcomes.append(row)
            append_summary(row, summary_path)
            log(f"  → {iid}: {row['status']} wall={row.get('wall_s','?')}s")
            if _stop_requested:
                break
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as ex:
            futures = {ex.submit(_one, iid): iid for iid in issue_ids}
            for fut in concurrent.futures.as_completed(futures):
                row = fut.result()
                outcomes.append(row)
                append_summary(row, summary_path)
                log(f"  → {row['issue_id']}: {row['status']} "
                    f"wall={row.get('wall_s','?')}s")
                if _stop_requested:
                    break
    return outcomes


def main() -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--issue-list", help="Space-separated beads IDs")
    src.add_argument("--discover", action="store_true",
                     help="Auto-fetch issues from `bd ready` each iteration")
    ap.add_argument("--model", default=os.environ.get("SWARM_CLOUD_MODEL", "claude-sonnet-4-6"))
    ap.add_argument("--repo-root", type=pathlib.Path, default=REPO_ROOT_DEFAULT)
    ap.add_argument("--cooldown", type=int, default=60,
                    help="Seconds between batches")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Concurrent issues per batch")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="Stop after N total runs (0 = unlimited)")
    ap.add_argument("--discover-limit", type=int, default=3,
                    help="Issues per batch in --discover mode")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--summary-dir", type=pathlib.Path, default=SUMMARY_DIR_DEFAULT)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    repo_root = args.repo_root.resolve()
    if not (repo_root / ".git").exists():
        raise SystemExit(f"ERROR: {repo_root} is not a git repo")

    lock = acquire_lock(repo_root)
    summary_path = args.summary_dir / "dogfood-summary.jsonl"
    log(f"dogfood loop starting: repo={repo_root.name} model={args.model} "
        f"parallel={args.parallel} cooldown={args.cooldown}s")
    log(f"summary → {summary_path}")

    try:
        issue_list = (args.issue_list or "").split() if args.issue_list else []
        total = 0
        while not _stop_requested:
            if args.discover:
                batch = discover_ready_issues(limit=args.discover_limit)
                if not batch:
                    log("no ready issues; sleeping")
                    time.sleep(args.cooldown)
                    continue
            else:
                if not issue_list:
                    log("issue list exhausted; stopping")
                    break
                # Pull up to `parallel` issues off the front
                batch = issue_list[:max(1, args.parallel)]
                issue_list = issue_list[len(batch):]

            log(f"=== batch of {len(batch)}: {' '.join(batch)} ===")
            run_batch(batch, repo_root=repo_root, model=args.model,
                      skip_tests=args.skip_tests, parallel=args.parallel,
                      summary_path=summary_path)
            total += len(batch)

            if args.max_runs and total >= args.max_runs:
                log(f"max_runs={args.max_runs} reached; stopping")
                break
            if _stop_requested:
                break
            log(f"cooldown {args.cooldown}s…")
            time.sleep(args.cooldown)
    finally:
        release_lock(lock)
        log("dogfood loop exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())
