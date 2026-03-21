#!/usr/bin/env python3
"""Parse ruff JSON output and create beads issues for each unique violation.

Groups violations by (file, rule_code) to avoid creating one issue per line.
Reads JSON from stdin, creates issues via `bd create`.

Usage:
    ruff check cflibs/ --output-format json | python3 lint-to-beads.py --tool ruff
    ruff check cflibs/ --output-format json | python3 lint-to-beads.py --dry-run
"""

import json
import subprocess
import sys
from argparse import ArgumentParser
from collections import defaultdict


def parse_args():
    p = ArgumentParser(description="Create beads issues from ruff JSON output")
    p.add_argument("--tool", default="ruff", help="Tool name for issue titles")
    p.add_argument("--priority", type=int, default=3, help="Issue priority (0-4)")
    p.add_argument("--max-issues", type=int, default=30, help="Max issues to create")
    p.add_argument("--bd", default="bd", help="Beads CLI binary name")
    p.add_argument("--dry-run", action="store_true", help="Print issues without creating")
    return p.parse_args()


def main():
    args = parse_args()

    raw = sys.stdin.read().strip()
    if not raw or raw == "[]":
        return

    try:
        violations = json.loads(raw)
    except json.JSONDecodeError:
        print(f"Failed to parse JSON from {args.tool}", file=sys.stderr)
        return

    # Group by (filename, code) to create one issue per violation type per file
    groups: dict[tuple[str, str], list] = defaultdict(list)
    for v in violations:
        filename = v.get("filename", "unknown")
        code = v.get("code", "unknown")
        groups[(filename, code)].append(v)

    created = 0
    for (filename, code), items in sorted(groups.items()):
        if created >= args.max_issues:
            break

        lines = sorted(set(item.get("location", {}).get("row", 0) for item in items))
        message = items[0].get("message", "lint violation")
        line_str = ", ".join(str(ln) for ln in lines[:5])
        if len(lines) > 5:
            line_str += f" (+{len(lines) - 5} more)"

        title = f"Fix {args.tool} {code}: {message} in {filename}"
        # Truncate title to 120 chars
        if len(title) > 120:
            title = title[:117] + "..."

        description = (
            f"{args.tool} reports {code} ({message}) at {filename} "
            f"line(s) {line_str}.\n\n"
            f"Fix the {len(items)} occurrence(s) of this violation. "
            f"Run `{args.tool} check {filename}` to verify the fix."
        )

        if args.dry_run:
            print(title)
            continue

        try:
            subprocess.run(
                [
                    args.bd, "create",
                    f"--title={title}",
                    f"--description={description}",
                    "--type=bug",
                    f"--priority={args.priority}",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            created += 1
        except subprocess.CalledProcessError as e:
            print(f"Failed to create issue: {e.stderr}", file=sys.stderr)

    if not args.dry_run:
        print(f"Created {created} issues from {args.tool} output", file=sys.stderr)


if __name__ == "__main__":
    main()
