#!/usr/bin/env python3
"""Parse mypy text output and create beads issues for each unique error.

Groups errors by (file, error_code) to avoid one issue per line.
Reads mypy output from stdin, creates issues via `bd create`.

Usage:
    mypy cflibs/ --show-error-codes | python3 mypy-to-beads.py
    mypy cflibs/ --show-error-codes | python3 mypy-to-beads.py --dry-run
"""

import re
import subprocess
import sys
from argparse import ArgumentParser
from collections import defaultdict


def parse_args():
    p = ArgumentParser(description="Create beads issues from mypy output")
    p.add_argument("--priority", type=int, default=2, help="Issue priority (0-4)")
    p.add_argument("--max-issues", type=int, default=30, help="Max issues to create")
    p.add_argument("--bd", default="bd", help="Beads CLI binary name")
    p.add_argument("--dry-run", action="store_true", help="Print issues without creating")
    return p.parse_args()


# mypy output format: cflibs/plasma/state.py:42: error: ... [error-code]
ERROR_RE = re.compile(
    r"^(?P<file>[^:]+):(?P<line>\d+):\s*error:\s*(?P<message>.+?)(?:\s+\[(?P<code>[^\]]+)\])?$"
)


def get_existing_issue_titles(bd: str) -> set[str]:
    """Fetch titles of all open/in_progress issues to prevent duplicates."""
    titles: set[str] = set()
    for status in ("open", "in_progress"):
        try:
            result = subprocess.run(
                [bd, "list", f"--status={status}", "--limit", "0"],
                capture_output=True, text=True, timeout=30,
            )
            for line in result.stdout.splitlines():
                # bd list output: "○ prefix-id ● P2 [bug] Title here"
                # Strip the status/priority prefix to get the title
                # Match after the priority marker (P0-P4)
                m = re.search(r"P\d\s+(?:\[.*?\]\s*)?(.+)$", line)
                if m:
                    titles.add(m.group(1).strip().lower())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    return titles


def title_already_exists(title: str, existing: set[str]) -> bool:
    """Check if a substantially similar title already exists.

    Uses normalized substring matching to catch slight variations
    like different line numbers in otherwise identical issues.
    """
    normalized = title.lower().strip()
    if normalized in existing:
        return True
    # Extract the core signature: mypy [code]: message in file
    # Strip line-number variations by matching on code + file
    core_match = re.match(r"fix mypy \[([^\]]+)\]:.*in\s+(\S+)", normalized)
    if core_match:
        code, filename = core_match.group(1), core_match.group(2)
        pattern = f"fix mypy [{code}]:"
        file_pattern = f"in {filename}"
        for existing_title in existing:
            if pattern in existing_title and file_pattern in existing_title:
                return True
    return False


def main():
    args = parse_args()

    # Group by (file, code)
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for raw_line in sys.stdin:
        line = raw_line.strip()
        m = ERROR_RE.match(line)
        if not m:
            continue
        entry = {
            "file": m.group("file"),
            "line": int(m.group("line")),
            "message": m.group("message").strip(),
            "code": m.group("code") or "unknown",
        }
        groups[(entry["file"], entry["code"])].append(entry)

    if not groups:
        return

    # Fetch existing issue titles ONCE to deduplicate
    existing_titles = set() if args.dry_run else get_existing_issue_titles(args.bd)

    created = 0
    skipped = 0
    for (filename, code), items in sorted(groups.items()):
        if created >= args.max_issues:
            break

        lines = sorted(set(item["line"] for item in items))
        message = items[0]["message"]
        line_str = ", ".join(str(ln) for ln in lines[:5])
        if len(lines) > 5:
            line_str += f" (+{len(lines) - 5} more)"

        title = f"Fix mypy [{code}]: {message} in {filename}"
        if len(title) > 120:
            title = title[:117] + "..."

        # Skip if a matching issue already exists
        if title_already_exists(title, existing_titles):
            skipped += 1
            print(f"Skipped (duplicate): {title}", file=sys.stderr)
            continue

        description = (
            f"mypy reports [{code}] error at {filename} line(s) {line_str}:\n"
            f"{message}\n\n"
            f"Fix the {len(items)} occurrence(s). "
            f"Run `mypy {filename}` to verify the fix."
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
            # Add to existing set so subsequent iterations don't duplicate
            existing_titles.add(title.lower().strip())
        except subprocess.CalledProcessError as e:
            print(f"Failed to create issue: {e.stderr}", file=sys.stderr)

    if not args.dry_run:
        print(
            f"Created {created} issues from mypy output "
            f"({skipped} duplicates skipped)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
