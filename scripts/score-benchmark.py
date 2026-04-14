#!/usr/bin/env python3
"""Score dogfood benchmark results for the autoresearch loop.

Reads dogfood-summary.jsonl and computes a single optimization score.

Score = success_rate * 100 - avg_time/60 - avg_tools/10

Higher is better. Perfect score for a 4-issue batch that all resolve
in 600s with 6 tools each: 100 - 10 - 0.6 = 89.4

Usage:
    python3 scripts/score-benchmark.py                              # Score all results
    python3 scripts/score-benchmark.py --after 2026-03-17T00:00:00  # Score recent only
    python3 scripts/score-benchmark.py --issues a9v8,42s3,r9la,knyz # Score specific issues
    python3 scripts/score-benchmark.py --tensorzero http://localhost:3000  # Post feedback to TZ
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def load_results(summary_path: Path, after: str = None, issue_filter: set = None) -> list:
    """Load and filter results from dogfood-summary.jsonl."""
    results = []
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            # Filter by timestamp
            if after and r.get("timestamp", "") < after:
                continue

            # Filter by issue
            if issue_filter and r["issue"] not in issue_filter:
                continue

            results.append(r)

    return results


def load_telemetry(telemetry_path: Path, issue_filter: set = None) -> dict:
    """Load session metrics from .swarm-telemetry.jsonl keyed by issue_id.

    Returns a dict of issue_id -> latest session metrics for episode_id lookup.
    """
    sessions = {}
    if not telemetry_path.exists():
        return sessions
    with open(telemetry_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
                iid = s.get("issue_id", "")
                if issue_filter and iid not in issue_filter:
                    continue
                sessions[iid] = s  # Keep latest per issue
            except json.JSONDecodeError:
                continue
    return sessions



def parse_log(log_path_str: str, summary_dir: Path) -> tuple[bool, int]:
    """Parse log file to determine success and tool call count.

    Returns (success: bool, tool_count: int)
    """
    if not log_path_str:
        return False, 0

    candidates = []
    p = Path(log_path_str)

    # Check absolute, relative to summary dir, and relative to repo root
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(summary_dir / p)

    if "logs/" in log_path_str:
        repo_root = Path(__file__).resolve().parent.parent
        # The log paths in dogfood-summary.jsonl often have the user's home dir hardcoded,
        # e.g., "/home/brian/code/beefcake-swarm/logs/..."
        # Try to extract relative from "logs/" and prepend repo root
        try:
            rel_log_path = log_path_str[log_path_str.index("logs/"):]
            candidates.append(repo_root / rel_log_path)
        except ValueError:
            pass

        # also if it's relative
        if not p.is_absolute():
            candidates.append(repo_root / log_path_str)

    valid_path = None
    for cand in candidates:
        if cand.exists() and cand.is_file():
            valid_path = cand
            break

    if not valid_path:
        return False, 0

    success = False
    tools = 0
    try:
        with open(valid_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "Issue resolved" in line:
                    success = True
                if "Tool call completed" in line:
                    tools += 1
    except Exception:
        pass

    return success, tools


def score_batch(results: list, summary_dir: Path) -> dict:
    """Compute optimization score for a batch of results."""
    if not results:
        return {"score": 0, "success_rate": 0, "avg_time": 0, "avg_tools": 0, "n": 0}

    n = len(results)

    resolved = 0
    total_tools = 0

    for r in results:
        log_path = r.get("log", "")
        success, tools = parse_log(log_path, summary_dir)
        if success:
            resolved += 1
        total_tools += tools

    success_rate = resolved / n if n > 0 else 0
    avg_time = sum(r.get("elapsed_s", 0) for r in results) / n
    avg_tools = total_tools / n if n > 0 else 0

    score = success_rate * 100 - avg_time / 60 - avg_tools / 10

    return {
        "score": round(score, 2),
        "success_rate": round(success_rate, 3),
        "avg_time": round(avg_time, 1),
        "avg_tools": avg_tools,
        "n": n,
        "resolved": resolved,
    }


def post_tensorzero_feedback(gateway_url: str, sessions: dict, results: list):
    """Post episode-level feedback to TensorZero for each result with an episode_id.

    Reads episode IDs from session telemetry and posts:
    - task_resolved (boolean)
    - iterations_used (float)
    - wall_time_seconds (float)
    """
    if not HAS_REQUESTS:
        print("Warning: 'requests' not installed, skipping TensorZero feedback", file=sys.stderr)
        return

    feedback_url = f"{gateway_url}/feedback"
    posted = 0
    skipped = 0

    for r in results:
        issue_id = r.get("issue", "")
        session = sessions.get(issue_id)
        if not session:
            skipped += 1
            continue

        episode_id = session.get("tensorzero_episode_id")
        if not episode_id:
            skipped += 1
            continue

        success = session.get("success", False)
        iterations = session.get("total_iterations", 0)
        elapsed_ms = session.get("elapsed_ms", 0)
        wall_secs = elapsed_ms / 1000.0

        feedbacks = [
            {"metric_name": "task_resolved", "episode_id": episode_id, "value": success},
            {"metric_name": "iterations_used", "episode_id": episode_id, "value": float(iterations)},
            {"metric_name": "wall_time_seconds", "episode_id": episode_id, "value": wall_secs},
        ]

        for fb in feedbacks:
            try:
                resp = requests.post(feedback_url, json=fb, timeout=10)
                if resp.status_code < 300:
                    posted += 1
                else:
                    print(f"  TZ feedback rejected: {fb['metric_name']} for {issue_id} "
                          f"(HTTP {resp.status_code})", file=sys.stderr)
            except Exception as e:
                print(f"  TZ feedback error: {fb['metric_name']} for {issue_id}: {e}",
                      file=sys.stderr)

    print(f"TensorZero feedback: {posted} posted, {skipped} skipped (no episode_id)")


def main():
    parser = argparse.ArgumentParser(description="Score dogfood benchmark results")
    parser.add_argument("--summary", default="logs/dogfood/dogfood-summary.jsonl",
                        help="Path to dogfood-summary.jsonl")
    parser.add_argument("--telemetry", default=".swarm-telemetry.jsonl",
                        help="Path to .swarm-telemetry.jsonl for episode_id lookup")
    parser.add_argument("--after", help="Only score results after this ISO timestamp")
    parser.add_argument("--issues", help="Comma-separated issue IDs to score")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--tensorzero",
                        default=os.environ.get("SWARM_TENSORZERO_URL", ""),
                        help="TensorZero gateway URL to post feedback (e.g. http://localhost:3000)")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    if not summary_path.exists():
        print(f"Error: {summary_path} not found", file=sys.stderr)
        sys.exit(1)

    issue_filter = set(args.issues.split(",")) if args.issues else None
    results = load_results(summary_path, args.after, issue_filter)

    if not results:
        print("No matching results found.", file=sys.stderr)
        sys.exit(1)

    scores = score_batch(results, summary_path.parent)

    if args.json:
        print(json.dumps(scores))
    else:
        print(f"Benchmark Score: {scores['score']}")
        print(f"  Success rate: {scores['success_rate']*100:.0f}% ({scores['resolved']}/{scores['n']})")
        print(f"  Avg time: {scores['avg_time']:.0f}s ({scores['avg_time']/60:.1f} min)")
        print(f"  Issues scored: {scores['n']}")

        # Per-issue breakdown
        print(f"\nPer-issue:")
        for r in results:
            success, _ = parse_log(r.get("log", ""), summary_path.parent)
            status = "OK" if success else "FAIL"
            print(f"  {r['issue']}: {status} ({r.get('elapsed_s', 0)}s)")

    # Post feedback to TensorZero if URL is provided
    if args.tensorzero:
        telemetry_path = Path(args.telemetry)
        sessions = load_telemetry(telemetry_path, issue_filter)
        print(f"\nPosting feedback to TensorZero at {args.tensorzero}...")
        post_tensorzero_feedback(args.tensorzero, sessions, results)


if __name__ == "__main__":
    main()
