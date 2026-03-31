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
            "Multi-crate workspace for detector readout, event building, and online analysis."
        ),
    },
    "CF-LIBS-improved": {
        "github": "TheFermiSea/CF-LIBS-improved",
        "branch": "main",
        "beads_root": os.path.expanduser("~/code/CF-LIBS-improved"),
        "patterns": ["TODO:", "bug", "refactor", "cleanup", "test", "type hint", "docstring"],
        "quality_gates": (
            "python -m pytest && "
            "python -m mypy . --ignore-missing-imports || true && "
            "ruff check ."
        ),
        "context": (
            "Python scientific computing: CF-LIBS (Calibration-Free Laser-Induced "
            "Breakdown Spectroscopy) analysis pipeline. NumPy/SciPy/Matplotlib stack. "
            "Planned merge with CF-LIBS repo — keep changes backward-compatible."
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
    """Query beads for simple issues matching configured patterns.

    Fetches full issue details via `bd show` to get file locations and
    descriptions — critical for giving Jules enough context to find the
    right code.
    """
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

        # Fetch full description and labels via `bd show`
        description = ""
        labels = ""
        try:
            show = subprocess.run(
                ["bd", "show", issue_id],
                capture_output=True, text=True, timeout=10,
                cwd=beads_root,
            )
            if show.returncode == 0:
                in_desc = False
                desc_lines = []
                for sl in show.stdout.split("\n"):
                    if sl.strip() == "DESCRIPTION":
                        in_desc = True
                        continue
                    if in_desc:
                        if sl.strip().startswith(("LABELS:", "PARENT", "DEPENDS", "BLOCKS", "CHILDREN")):
                            in_desc = False
                            if sl.strip().startswith("LABELS:"):
                                labels = sl.strip().replace("LABELS:", "").strip()
                            continue
                        desc_lines.append(sl)
                description = "\n".join(desc_lines).strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        issues.append({
            "id": issue_id,
            "title": title,
            "description": description,
            "labels": labels,
        })
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


def _build_task_instructions(title: str, description: str, labels: str) -> str:
    """Generate task-specific instructions based on issue type.

    The key insight from Jules batch #1: generic "fix this" prompts fail.
    Jules needs explicit instructions: which file, which function, what
    action to take (delete, modify, add).
    """
    title_lower = title.lower()

    # Extract file location from description if present
    location = ""
    for line in description.split("\n"):
        if line.strip().lower().startswith("location:"):
            location = line.strip().split(":", 1)[1].strip()
            break

    loc_hint = f"\n**File location:** `{location}`\n" if location else ""

    if "unused function" in title_lower or "dead-code" in labels:
        func_name = title.replace("Unused function:", "").replace("Unused function", "").strip()
        return f"""This is a dead code removal task.
{loc_hint}
1. Search the codebase for the function `{func_name}` using grep or the search tool
2. If the function exists and has NO callers, delete it and any associated test functions
3. If the function is already gone, verify with grep and create an empty PR (or skip)
4. Remove any `use` imports that become unused after the deletion
5. Run `cargo check` (or equivalent) to verify nothing breaks"""

    if "todo:" in title_lower or "fixme" in title_lower:
        return f"""This is a TODO/FIXME resolution task.
{loc_hint}
1. Find the TODO/FIXME comment at the specified location
2. Implement what the comment describes, or remove it if it's obsolete
3. Keep the implementation minimal — do exactly what the comment says"""

    if "vulnerable dependency" in title_lower or "security" in labels:
        dep_name = ""
        for word in title.split():
            if word not in ("Vulnerable", "dependency:", "dependency"):
                dep_name = word
                break
        return f"""This is a dependency vulnerability fix.
{loc_hint}
1. Update the vulnerable dependency `{dep_name}` to the patched version
2. For Rust: edit Cargo.toml version constraint, then run `cargo update -p {dep_name}`
3. For Python: update the version in requirements.txt/pyproject.toml
4. Run tests to verify the update doesn't break anything"""

    if "large file" in title_lower:
        return f"""This is a code organization task — split a large file.
{loc_hint}
1. Identify logical groupings of functions/types in the file
2. Extract cohesive groups into new submodules
3. Re-export public items from the parent module for backward compatibility"""

    if "low test ratio" in title_lower or "low-test-ratio" in labels:
        return f"""This is a test coverage task.
{loc_hint}
1. Identify the key public functions in the module that lack tests
2. Add unit tests for the most important 2-3 functions
3. Focus on edge cases and error paths, not just happy paths"""

    if "complex function" in title_lower:
        return f"""This is a refactoring task — reduce function complexity.
{loc_hint}
1. Identify the function and understand what it does
2. Extract logical sub-operations into well-named helper functions
3. Keep the public API unchanged — only refactor internals"""

    # Generic fallback — still better than before
    return f"""Analyze the issue and implement the fix.
{loc_hint}
1. Read the issue description carefully
2. Search the codebase to understand the current state
3. Make the minimal change needed to resolve the issue"""


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

    description = issue.get("description", "")
    labels = issue.get("labels", "")
    title = issue["title"]

    # Build task-specific instructions based on issue type
    task_instructions = _build_task_instructions(title, description, labels)

    prompt = f"""You are fixing a specific issue in the {github_repo} repository.

## Task: {title}

{f"### Details{chr(10)}{description}" if description else ""}

## What to do

{task_instructions}

## Repository context

{context}

## Verification

Run these quality gates — ALL must pass before creating the PR:
```bash
{quality_gates}
```

## Rules

- Make ONLY the changes needed to fix this specific issue
- Do NOT add comments, docstrings, or documentation
- Do NOT refactor or reformat surrounding code
- Do NOT create any temporary or debug files
- If the function/code is already removed, verify it's gone and close with no changes
- Title the PR: "fix: {title}"
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
