#!/usr/bin/env python3
"""UCB1-based issue selection for the dogfood loop.

Reads ready issues from bd, scores them using UCB1 balancing:
- Exploitation: historical success rate for the issue's error category
- Exploration: under-attempted categories get priority

Usage:
    python3 scripts/ucb1-select.py --experiment-db .swarm/experiment_history.jsonl --top-n 3

Output: space-separated issue IDs (for dogfood-loop.sh consumption)
"""

import json
import math
import subprocess
import sys
import argparse


def load_experiment_history(path):
    """Load past experiment outcomes."""
    history = []
    try:
        with open(path) as f:
            for line in f:
                try:
                    history.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass
    except FileNotFoundError:
        pass
    return history


def get_ready_issues():
    """Get ready issues from bd."""
    result = subprocess.run(
        ["bd", "ready", "--json", "-n", "50"],
        capture_output=True, text=True
    )
    try:
        data = json.loads(result.stdout)
        if isinstance(data, list):
            return data
        # bd ready --json may wrap tasks in {"tasks": [...]}
        if isinstance(data, dict):
            if "tasks" in data:
                return data["tasks"]
            if "bd_stdout" in data:
                inner = data["bd_stdout"]
                if isinstance(inner, list):
                    return inner
                if isinstance(inner, dict) and "tasks" in inner:
                    return inner["tasks"]
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def categorize_issue(issue):
    """Heuristic category detection from issue title and labels."""
    title = issue.get("title", "").lower()
    labels = [l.lower() for l in issue.get("labels", [])]

    # Check labels first (more reliable)
    label_categories = {
        "bug": "bug",
        "feature": "feature",
        "refactor": "refactor",
        "test": "test",
        "docs": "docs",
        "chore": "chore",
        "lint": "lint",
        "benchmark": "benchmark",
        "performance": "performance",
    }
    for label in labels:
        for key, cat in label_categories.items():
            if key in label:
                return cat

    # Fall back to title keyword heuristics
    keyword_categories = [
        (["bug", "fix", "crash", "panic", "error", "broken", "wrong"], "bug"),
        (["feat", "add", "implement", "new", "create"], "feature"),
        (["refactor", "clean", "simplify", "reorganize"], "refactor"),
        (["test", "spec", "assert"], "test"),
        (["unused", "dead_code", "allow", "lint", "clippy", "warning"], "lint"),
        (["benchmark", "perf", "slow", "throughput"], "benchmark"),
        (["doc", "readme", "comment"], "docs"),
        (["chore", "update", "bump", "dependency"], "chore"),
        (["complex", "arch", "design", "plan"], "complex"),
    ]
    for keywords, cat in keyword_categories:
        for kw in keywords:
            if kw in title:
                return cat

    return "unknown"


def ucb1_score(success_rate, total_visits, category_visits, c=1.414):
    """Calculate UCB1 score.

    UCB1 = exploitation + c * sqrt(ln(N) / n_i)
    where N = total attempts, n_i = attempts for this category.
    Unvisited categories get infinite score (explore first).
    """
    if category_visits == 0:
        return float('inf')  # Prioritize unvisited categories
    exploitation = success_rate
    exploration = c * math.sqrt(math.log(max(total_visits, 1)) / category_visits)
    return exploitation + exploration


def main():
    parser = argparse.ArgumentParser(
        description="UCB1-based issue selection for the dogfood loop."
    )
    parser.add_argument(
        "--experiment-db",
        default=".swarm/experiment_history.jsonl",
        help="Path to experiment history JSONL file",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=3,
        help="Number of issues to select",
    )
    parser.add_argument(
        "--c",
        type=float,
        default=1.414,
        help="Exploration coefficient (higher = more exploration)",
    )
    parser.add_argument(
        "--summary-db",
        default="",
        help="Path to dogfood-summary.jsonl (alternative history source)",
    )
    args = parser.parse_args()

    history = load_experiment_history(args.experiment_db)

    # If experiment DB is empty, try the dogfood summary as fallback
    if not history and args.summary_db:
        history = load_experiment_history(args.summary_db)

    issues = get_ready_issues()

    if not issues:
        return

    # Calculate per-category stats from history
    category_stats = {}  # category -> (successes, attempts)
    total_attempts = len(history)
    for exp in history:
        cat = exp.get("error_category", "unknown")
        # Support both experiment_history format and dogfood-summary format
        if "success" in exp:
            succeeded = bool(exp["success"])
        elif "exit_code" in exp:
            succeeded = exp["exit_code"] == 0
        else:
            succeeded = False
        s, a = category_stats.get(cat, (0, 0))
        category_stats[cat] = (s + (1 if succeeded else 0), a + 1)

    # Score each issue
    scored = []
    for issue in issues:
        issue_id = issue.get("task_ref") or issue.get("id", "")
        if not issue_id:
            continue

        category = categorize_issue(issue)

        s, a = category_stats.get(category, (0, 0))
        rate = s / max(a, 1)
        score = ucb1_score(rate, total_attempts, a, args.c)
        scored.append((issue_id, score, category))

    # Sort by UCB1 score descending, take top-n
    scored.sort(key=lambda x: -x[1])
    selected = [issue_id for issue_id, _, _ in scored[:args.top_n]]

    # Output space-separated for shell consumption
    print(" ".join(selected))


if __name__ == "__main__":
    main()
