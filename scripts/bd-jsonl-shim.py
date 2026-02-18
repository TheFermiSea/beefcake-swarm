#!/usr/bin/env python3
"""Lightweight bd (beads) shim that operates directly on .beads/issues.jsonl.

Used inside the swarm container where Go bd v0.52.0 requires dolt+CGO
which isn't available. Supports the subset of commands the orchestrator uses:
  - ready --json
  - update <id> --status=<status>
  - close <id> [--reason="..."]
  - show <id> --json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def find_beads_dir():
    """Walk up to find .beads/ directory."""
    path = Path.cwd()
    while path != path.parent:
        beads = path / ".beads"
        if beads.is_dir():
            return beads
        path = path.parent
    return None


def load_issues(beads_dir):
    jsonl_path = beads_dir / "issues.jsonl"
    issues = []
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    issues.append(json.loads(line))
    return issues


def save_issues(beads_dir, issues):
    jsonl_path = beads_dir / "issues.jsonl"
    with open(jsonl_path, "w") as f:
        for issue in issues:
            f.write(json.dumps(issue, separators=(",", ":")) + "\n")


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def resolve_id(issues, short_id):
    """Resolve a short ID to a full issue ID (e.g. 'a8m' -> 'beefcake-swarm-a8m')."""
    # Exact match first
    for issue in issues:
        if issue["id"] == short_id:
            return short_id
    # Suffix match
    for issue in issues:
        if issue["id"].endswith("-" + short_id):
            return issue["id"]
    return short_id  # Return as-is if no match


def cmd_ready(issues, args):
    """Return issues that are open with no unresolved blockers."""
    # Build set of open issue IDs
    open_ids = {i["id"] for i in issues if i.get("status") in ("open", "in_progress")}

    ready = []
    for issue in issues:
        if issue.get("status") != "open":
            continue
        # Check dependencies: skip if any blocking issue is still open
        deps = issue.get("dependencies", [])
        blocked = False
        for dep in deps:
            if dep.get("dependency_type") == "blocks" and dep.get("id") in open_ids:
                blocked = True
                break
        if not blocked:
            ready.append(issue)

    # Sort by priority (lower = higher priority)
    ready.sort(key=lambda i: (i.get("priority", 99), i.get("created_at", "")))

    if "--json" in args:
        print(json.dumps(ready, indent=2))
    else:
        for issue in ready:
            print(f"{issue['id']}  P{issue.get('priority', '?')}  {issue.get('title', '?')}")


def cmd_update(issues, args):
    """Update issue fields."""
    if len(args) < 1:
        print("Usage: bd update <id> --status=<status>", file=sys.stderr)
        sys.exit(1)

    issue_id = resolve_id(issues, args[0])
    target = None
    for issue in issues:
        if issue["id"] == issue_id:
            target = issue
            break

    if not target:
        print(f"Error: issue {issue_id} not found", file=sys.stderr)
        sys.exit(1)

    for arg in args[1:]:
        if arg.startswith("--status="):
            target["status"] = arg.split("=", 1)[1]
        elif arg.startswith("--notes="):
            target["notes"] = arg.split("=", 1)[1]

    target["updated_at"] = now_iso()
    return issues


def cmd_close(issues, args):
    """Close one or more issues."""
    if len(args) < 1:
        print("Usage: bd close <id> [--reason='...']", file=sys.stderr)
        sys.exit(1)

    reason = None
    ids_to_close = []
    for arg in args:
        if arg.startswith("--reason="):
            reason = arg.split("=", 1)[1]
        else:
            ids_to_close.append(resolve_id(issues, arg))

    for issue in issues:
        if issue["id"] in ids_to_close:
            issue["status"] = "closed"
            issue["closed_at"] = now_iso()
            issue["updated_at"] = now_iso()
            if reason:
                issue["close_reason"] = reason

    return issues


def cmd_show(issues, args):
    """Show issue details."""
    if len(args) < 1:
        print("Usage: bd show <id>", file=sys.stderr)
        sys.exit(1)

    issue_id = resolve_id(issues, args[0])
    for issue in issues:
        if issue["id"] == issue_id:
            if "--json" in args:
                print(json.dumps([issue], indent=2))
            else:
                for k, v in issue.items():
                    print(f"  {k}: {v}")
            return

    print(f"Error: issue {issue_id} not found", file=sys.stderr)
    sys.exit(1)


def cmd_version():
    print("bd-jsonl-shim 0.1.0 (JSONL-only, no dolt)")


def main():
    if len(sys.argv) < 2:
        print("Usage: bd <command> [args...]", file=sys.stderr)
        sys.exit(1)

    command = sys.argv[1]

    if command == "--version":
        cmd_version()
        return

    beads_dir = find_beads_dir()
    if not beads_dir:
        print("Error: .beads/ directory not found", file=sys.stderr)
        sys.exit(1)

    issues = load_issues(beads_dir)
    args = sys.argv[2:]

    if command == "ready":
        cmd_ready(issues, args)
    elif command == "update":
        issues = cmd_update(issues, args)
        save_issues(beads_dir, issues)
    elif command == "close":
        issues = cmd_close(issues, args)
        save_issues(beads_dir, issues)
    elif command == "show":
        cmd_show(issues, args)
    elif command == "sync":
        pass  # no-op in container
    else:
        print(f"Error: unknown command '{command}'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
