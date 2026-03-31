#!/usr/bin/env python3
"""Centralized Jules dispatch — routes issues to the correct repo via Jules API.

Reads issues from beads across multiple repos, dispatches Jules sessions
targeting each repo independently. The swarm works locally on one repo
while Jules handles simple fixes across ALL repos in parallel.

Usage:
  python scripts/jules-dispatch.py                      # dispatch from all configured repos
  python scripts/jules-dispatch.py --repo rust-daq      # single repo
  python scripts/jules-dispatch.py --max 50 --dry-run   # preview
  python scripts/jules-dispatch.py --poll               # check session status

Requires: JULES_API_KEY env var (from jules.google.com)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

JULES_API = "https://jules.googleapis.com/v1alpha/sessions"
DISPATCH_LOG = Path.home() / ".cache" / "jules-dispatch" / "sessions.jsonl"
DISPATCH_LOG.parent.mkdir(parents=True, exist_ok=True)

# ── Repo Configuration ───────────────────────────────────────────────────
# Each repo specifies: GitHub owner/name, default branch, beads root,
# and issue patterns suitable for Jules (simple, well-defined tasks).

REPOS = {
    "beefcake-swarm": {
        "github": "TheFermiSea/beefcake-swarm",
        "branch": "main",
        "beads_root": os.path.expanduser("~/code/beefcake-swarm"),
        "patterns": ["Unused function", "TODO:", "Large file:"],
        "quality_gates": (
            "cargo fmt --all -- --check && "
            "cargo clippy --workspace -- -D warnings && "
            "cargo check --workspace && "
            "cargo test -p coordination && cargo test -p swarm-agents"
        ),
        "context": (
            "Rust workspace: coordination/ (deterministic logic) and "
            "crates/swarm-agents/ (LLM orchestrator). "
        ),
    },
    "rust-daq": {
        "github": "TheFermiSea/rust-daq",
        "branch": "main",
        "beads_root": os.path.expanduser("~/code/rust-daq"),
        "patterns": ["Unused function", "TODO:", "bug", "clippy warning"],
        "quality_gates": (
            "cargo fmt --all -- --check && "
            "cargo clippy --workspace -- -D warnings && "
            "cargo check --workspace && cargo test --workspace"
        ),
        "context": (
            "Rust DAQ system for physics data acquisition. "
        ),
    },
}


@dataclass
class DispatchRecord:
    session_id: str
    repo: str
    issue_id: str
    issue_title: str
    timestamp: str
    status: str  # dispatched, completed, failed


def get_api_key() -> str:
    key = os.environ.get("JULES_API_KEY", "")
    if not key:
        print("ERROR: JULES_API_KEY not set", file=sys.stderr)
        sys.exit(1)
    return key


def get_beads_issues(repo_config: dict, max_issues: int) -> list[dict]:
    """Query beads for simple issues matching configured patterns."""
    beads_root = repo_config["beads_root"]
    if not os.path.isdir(beads_root):
        print(f"  WARN: beads root not found: {beads_root}", file=sys.stderr)
        return []

    try:
        result = subprocess.run(
            ["bd", "list", "--status=open"],
            capture_output=True, text=True, timeout=30,
            cwd=beads_root,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    issues = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        # Match any configured pattern
        matched = any(p.lower() in line.lower() for p in repo_config["patterns"])
        if not matched:
            continue

        # Parse: "○ beefcake-xxx ● P2 Title here"
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        # Skip status marker, get ID
        issue_id = parts[1] if not parts[0].startswith("beefcake") else parts[0]
        # Find title after priority
        for i, part in enumerate(parts):
            if part.startswith("P") and len(part) == 2 and part[1].isdigit():
                title = " ".join(parts[i + 1:])
                break
        else:
            title = " ".join(parts[2:])

        issues.append({"id": issue_id, "title": title})
        if len(issues) >= max_issues:
            break

    return issues


def load_dispatched() -> set[str]:
    """Load previously dispatched issue IDs to avoid duplicates."""
    dispatched = set()
    if DISPATCH_LOG.exists():
        for line in DISPATCH_LOG.read_text().strip().split("\n"):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                dispatched.add(record["issue_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return dispatched


def dispatch_jules(
    api_key: str,
    repo_config: dict,
    issue: dict,
    dry_run: bool = False,
) -> str | None:
    """Dispatch a Jules session for a single issue. Returns session ID or None."""
    github_repo = repo_config["github"]
    branch = repo_config["branch"]
    context = repo_config.get("context", "")
    quality_gates = repo_config.get("quality_gates", "")

    prompt = f"""Fix the following issue in the {github_repo} repository.

## Issue: {issue['title']}

Beads ID: `{issue['id']}`

## Instructions

{context}

Quality gates (all must pass before creating PR):
```bash
{quality_gates}
```

Rules:
- Make minimal changes — fix the issue, nothing more
- Do not add comments explaining what the code does
- Do not refactor surrounding code
- Run quality gates before creating the PR
- Title the PR: "fix: {issue['title']}"
"""

    payload = {
        "prompt": prompt,
        "sourceContext": {
            "source": f"sources/github/{github_repo}",
            "githubRepoContext": {
                "startingBranch": branch,
            },
        },
        "requirePlanApproval": False,
        "automationMode": "AUTO_CREATE_PR",
    }

    if dry_run:
        print(f"  [DRY RUN] {github_repo}: {issue['title']} ({issue['id']})")
        return None

    import urllib.request
    import urllib.error

    req = urllib.request.Request(
        JULES_API,
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
            session_id = body.get("name", body.get("sessionId", "unknown"))
            print(f"  Dispatched: {github_repo} #{issue['id']} → session {session_id}")
            return session_id
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"  FAILED ({e.code}): {github_repo} #{issue['id']}: {error_body[:200]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  FAILED: {github_repo} #{issue['id']}: {e}", file=sys.stderr)
        return None


def record_dispatch(session_id: str, repo: str, issue: dict):
    """Append dispatch record to JSONL log."""
    record = {
        "session_id": session_id,
        "repo": repo,
        "issue_id": issue["id"],
        "issue_title": issue["title"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "dispatched",
    }
    with open(DISPATCH_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


def poll_sessions(api_key: str):
    """Check status of dispatched sessions."""
    if not DISPATCH_LOG.exists():
        print("No dispatch log found")
        return

    import urllib.request
    import urllib.error

    records = []
    for line in DISPATCH_LOG.read_text().strip().split("\n"):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    pending = [r for r in records if r["status"] == "dispatched"]
    if not pending:
        print("No pending sessions")
        return

    print(f"Checking {len(pending)} pending sessions...")
    for record in pending:
        sid = record["session_id"]
        try:
            req = urllib.request.Request(
                f"{JULES_API}/{sid}",
                headers={"X-Goog-Api-Key": api_key},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read())
                state = body.get("state", "unknown")
                pr_url = ""
                for activity in body.get("activities", []):
                    if "pullRequestUrl" in str(activity):
                        pr_url = activity.get("pullRequestUrl", "")
                print(f"  {record['repo']} {record['issue_id']}: {state} {pr_url}")
        except Exception as e:
            print(f"  {record['repo']} {record['issue_id']}: error checking: {e}")


def main():
    parser = argparse.ArgumentParser(description="Centralized Jules dispatch")
    parser.add_argument("--repo", help="Target a single repo (key from REPOS config)")
    parser.add_argument("--max", type=int, default=30, help="Max issues per repo (default: 30)")
    parser.add_argument("--dry-run", action="store_true", help="Preview without dispatching")
    parser.add_argument("--poll", action="store_true", help="Check status of dispatched sessions")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between dispatches (seconds)")
    args = parser.parse_args()

    api_key = get_api_key()

    if args.poll:
        poll_sessions(api_key)
        return

    # Select repos
    if args.repo:
        if args.repo not in REPOS:
            print(f"Unknown repo: {args.repo}. Available: {', '.join(REPOS.keys())}")
            sys.exit(1)
        repos = {args.repo: REPOS[args.repo]}
    else:
        repos = REPOS

    dispatched_ids = load_dispatched()
    total_dispatched = 0
    total_skipped = 0

    for repo_name, repo_config in repos.items():
        print(f"\n{'='*60}")
        print(f"Repo: {repo_config['github']}")
        print(f"{'='*60}")

        issues = get_beads_issues(repo_config, args.max)
        if not issues:
            print("  No matching issues found")
            continue

        for issue in issues:
            if issue["id"] in dispatched_ids:
                total_skipped += 1
                continue

            session_id = dispatch_jules(api_key, repo_config, issue, args.dry_run)
            if session_id:
                record_dispatch(session_id, repo_name, issue)
                dispatched_ids.add(issue["id"])
                total_dispatched += 1
                time.sleep(args.delay)  # rate limiting
            elif not args.dry_run:
                # API error — stop dispatching to this repo
                print(f"  Stopping dispatch to {repo_name} after API error")
                break
            else:
                total_dispatched += 1  # count dry-run as dispatched

    print(f"\n{'='*60}")
    print(f"Total dispatched: {total_dispatched} | Skipped (already dispatched): {total_skipped}")
    print(f"Session log: {DISPATCH_LOG}")


if __name__ == "__main__":
    main()
