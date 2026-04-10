#!/usr/bin/env python3
"""Extract preference pairs from TensorZero telemetry for DPO training of SERA-14B.

Queries the TensorZero Postgres database for inference records from the
worker_code_edit and code_fixing functions. For each issue (episode group)
where multiple model variants were attempted, creates preference pairs:
  - winner_trajectory: the variant/model that passed the verifier (task_resolved=true)
  - loser_trajectory:  the variant/model that failed the verifier for the same issue

Per-tool-call granularity is preserved — the RAGEN paper (2024) showed that
coarse outcome-only rewards are insufficient; step-level signal is required
for stable DPO convergence. Each trajectory is a list of tool-call+output dicts
extracted from the TZ chat_inference_data JSONB columns.

Reference: docs/research/agent-harness-survey.md, Q5 section (RAGEN / DPO).

Usage:
    python3 scripts/extract-tz-preferences.py
    python3 scripts/extract-tz-preferences.py --output data/my-pairs.jsonl --since 2026-03-01
    python3 scripts/extract-tz-preferences.py --dry-run --min-pairs 1

Environment variables:
    SWARM_TENSORZERO_PG_URL   Postgres DSN (checked first)
    TENSORZERO_POSTGRES_URL   Fallback DSN
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print(
        "Error: psycopg2-binary is required. Install with: pip install psycopg2-binary",
        file=sys.stderr,
    )
    sys.exit(1)


# Default Postgres DSN — matches infrastructure/tensorzero/docker-compose.yml
DEFAULT_PG_URL = "postgresql://tensorzero:tensorzero@localhost:5433/tensorzero"

# TZ functions whose inferences are useful for worker DPO training.
WORKER_FUNCTIONS = frozenset({
    "worker_code_edit",
    "code_fixing",
    "task_planning",
    "editor_apply",
})

# Tool names that constitute actual edits (write-side actions).
EDIT_TOOL_NAMES = frozenset({
    "edit_file",
    "write_file",
    "proxy_edit_file",
    "proxy_write_file",
    "apply_patch",
    "apply_plan",
})

# Known verifier error category keywords (from coordination/src/verifier/).
ERROR_CATEGORY_KEYWORDS = [
    "borrow",
    "lifetime",
    "trait",
    "type mismatch",
    "async",
    "Send",
    "import",
    "unresolved",
    "unused",
    "fmt",
    "clippy",
    "test",
]


def resolve_pg_url(explicit: str | None) -> str:
    """Return the Postgres DSN from args, env vars, or default."""
    if explicit:
        return explicit
    for var in ("SWARM_TENSORZERO_PG_URL", "TENSORZERO_POSTGRES_URL"):
        val = os.environ.get(var)
        if val:
            return val
    return DEFAULT_PG_URL


def connect_pg(pg_url: str):
    """Open a read-only Postgres connection to TensorZero."""
    try:
        conn = psycopg2.connect(pg_url)
        conn.set_session(readonly=True, autocommit=True)
        return conn
    except psycopg2.Error as exc:
        print(f"Error: Failed to connect to TZ Postgres at {pg_url}: {exc}", file=sys.stderr)
        sys.exit(1)


def fetch_verifier_outcomes(
    conn,
    since: str | None,
    functions: frozenset[str],
) -> dict[str, dict[str, Any]]:
    """Fetch episode-level verifier outcomes (task_resolved boolean feedback).

    Returns a dict keyed by episode_id (str) with:
        resolved: bool
        tags: dict (may include issue_id, repo_id, iterations_used)
        created_at: str ISO
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    query = """
    SELECT
        bf.target_id::text AS episode_id,
        bf.value            AS resolved,
        bf.created_at,
        bf.tags
    FROM tensorzero.boolean_metric_feedback bf
    WHERE bf.metric_name = 'task_resolved'
    """
    params: list[Any] = []

    if since:
        query += " AND bf.created_at >= %s::timestamptz"
        params.append(since)

    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()

    outcomes: dict[str, dict[str, Any]] = {}
    for row in rows:
        episode_id = row["episode_id"]
        outcomes[episode_id] = {
            "resolved": bool(row["resolved"]),
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "tags": dict(row["tags"]) if row["tags"] else {},
        }

    return outcomes


def fetch_inferences_for_episodes(
    conn,
    episode_ids: list[str],
    functions: frozenset[str],
) -> dict[str, list[dict[str, Any]]]:
    """Fetch all chat_inferences for the given episodes, grouped by episode_id.

    Filters to the specified TZ functions (worker_code_edit, code_fixing, etc.).
    Returns dict: episode_id -> list of inference metadata dicts.
    """
    if not episode_ids:
        return {}

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Use ANY(%s) with a list cast — psycopg2 handles the array.
    query = """
    SELECT
        ci.id::text          AS inference_id,
        ci.episode_id::text  AS episode_id,
        ci.function_name,
        ci.variant_name,
        ci.created_at,
        mi.model_name,
        mi.model_provider_name,
        mi.input_tokens,
        mi.output_tokens,
        mi.response_time_ms
    FROM tensorzero.chat_inferences ci
    LEFT JOIN tensorzero.model_inferences mi ON mi.inference_id = ci.id
    WHERE ci.episode_id = ANY(%s::uuid[])
      AND ci.function_name = ANY(%s)
    ORDER BY ci.episode_id, ci.created_at ASC
    """

    cur.execute(query, [episode_ids, list(functions)])
    rows = cur.fetchall()
    cur.close()

    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_episode[row["episode_id"]].append({
            "inference_id": row["inference_id"],
            "episode_id": row["episode_id"],
            "function_name": row["function_name"],
            "variant_name": row["variant_name"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "model_name": row["model_name"],
            "model_provider_name": row["model_provider_name"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "response_time_ms": row["response_time_ms"],
        })

    return dict(by_episode)


def fetch_inference_data(conn, inference_id: str) -> dict[str, Any] | None:
    """Fetch the raw input/output JSONB for one inference from chat_inference_data.

    TZ partitions this table by date; querying the parent lets Postgres route
    to the correct partition automatically.
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    try:
        cur.execute(
            "SELECT input, output FROM tensorzero.chat_inference_data WHERE id = %s::uuid",
            [inference_id],
        )
        row = cur.fetchone()
        cur.close()
        if row is None:
            return None
        return {"input": row["input"], "output": row["output"]}
    except psycopg2.Error as exc:
        cur.close()
        print(
            f"  Warning: Could not fetch data for inference {inference_id}: {exc}",
            file=sys.stderr,
        )
        conn.rollback()
        return None


def extract_tool_calls(output: Any) -> list[dict[str, Any]]:
    """Extract tool-call steps from a TZ inference output blob.

    TZ stores output as either:
      - A list of content blocks (each may be type=tool_call or type=text)
      - A dict with a .content list or .tool_calls list (OpenAI format)
      - A raw string (older TZ versions)

    Returns a list of dicts: [{name, arguments, result, raw_name}, ...]
    preserving order so downstream DPO code sees full trajectories.
    """
    if output is None:
        return []

    # Normalise to a list of content blocks.
    blocks: list[Any] = []
    if isinstance(output, list):
        blocks = output
    elif isinstance(output, dict):
        # OpenAI-style: message.tool_calls or content list
        content = output.get("content") or []
        tool_calls = output.get("tool_calls") or []
        if isinstance(content, list):
            blocks = content
        if tool_calls:
            blocks = blocks + list(tool_calls)
    elif isinstance(output, str):
        # Heuristic: return a single pseudo block for the raw string.
        return [{"type": "text", "content": output, "name": None}]

    calls: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type", "")
        if block_type == "tool_call" or block_type == "tool_use":
            name = (
                block.get("name")
                or block.get("raw_name")
                or block.get("function", {}).get("name", "")
            )
            arguments = block.get("arguments") or block.get("input") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            calls.append({
                "type": "tool_call",
                "name": name,
                "arguments": arguments,
                "result": block.get("result"),
                "raw_name": block.get("raw_name"),
                "is_edit": name in EDIT_TOOL_NAMES if name else False,
            })
        elif block_type == "text":
            # Keep text blocks so the full reasoning trace is preserved.
            calls.append({
                "type": "text",
                "content": block.get("text") or block.get("content") or "",
                "name": None,
                "is_edit": False,
            })
        elif "function" in block:
            # OpenAI tool_call wrapper: {id, type, function: {name, arguments}}
            func = block["function"]
            name = func.get("name", "")
            arguments = func.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {"_raw": arguments}
            calls.append({
                "type": "tool_call",
                "name": name,
                "arguments": arguments,
                "result": None,
                "raw_name": name,
                "is_edit": name in EDIT_TOOL_NAMES,
            })

    return calls


def infer_error_categories(tool_calls: list[dict[str, Any]]) -> list[str]:
    """Heuristically extract error categories from tool call content/arguments.

    Scans argument values and text blocks for known verifier error keywords
    (borrow, lifetime, trait bounds, type mismatch, async/Send, etc.).
    Returns a deduplicated list of matched categories.
    """
    categories: list[str] = []
    haystack = json.dumps(tool_calls, default=str).lower()
    for kw in ERROR_CATEGORY_KEYWORDS:
        if kw.lower() in haystack:
            categories.append(kw)
    return list(dict.fromkeys(categories))  # deduplicate, preserve order


def build_trajectory(
    conn,
    inferences: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a per-tool-call trajectory for a list of inferences in an episode.

    Each inference contributes a step dict:
        {inference_id, function_name, variant_name, model_name,
         tool_calls: [...], input_tokens, output_tokens, response_time_ms}
    """
    steps: list[dict[str, Any]] = []
    for inf in inferences:
        raw = fetch_inference_data(conn, inf["inference_id"])
        if raw is None:
            continue
        tool_calls = extract_tool_calls(raw.get("output"))
        steps.append({
            "inference_id": inf["inference_id"],
            "function_name": inf["function_name"],
            "variant_name": inf["variant_name"],
            "model_name": inf["model_name"],
            "created_at": inf["created_at"],
            "input_tokens": inf["input_tokens"],
            "output_tokens": inf["output_tokens"],
            "response_time_ms": inf["response_time_ms"],
            "tool_calls": tool_calls,
            "edit_count": sum(1 for tc in tool_calls if tc.get("is_edit")),
        })
    return steps


def make_pairs(
    outcomes: dict[str, dict[str, Any]],
    inferences_by_episode: dict[str, list[dict[str, Any]]],
    conn,
    min_pairs: int,
) -> list[dict[str, Any]]:
    """Build preference pairs from episodes where both successes and failures exist.

    Strategy:
      1. Group episodes by issue_id (from the 'tags' field).
      2. For each issue, separate episodes by resolved=true/false.
      3. Pair each winning episode with each losing episode that used a different
         variant, capturing the full tool-call trajectories for both.
      4. Emit one JSONL record per pair.

    If no issue_id tag is available, fall back to grouping by function_name
    so we still get cross-episode pairs within the same task type.
    """
    # Group episodes by (issue_id or "global") -> {success: [...], failure: [...]}
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"success": [], "failure": []})

    for episode_id, outcome in outcomes.items():
        tags = outcome.get("tags", {})
        # Prefer issue_id tag; fall back to function_name from inferences.
        issue_id = tags.get("issue_id") or tags.get("repo_id") or "global"
        bucket = "success" if outcome["resolved"] else "failure"
        groups[issue_id][bucket].append(episode_id)

    pairs: list[dict[str, Any]] = []

    for issue_id, buckets in groups.items():
        winners = buckets["success"]
        losers = buckets["failure"]
        if not winners or not losers:
            continue

        for win_ep in winners:
            win_infs = inferences_by_episode.get(win_ep, [])
            if not win_infs:
                continue
            win_traj = build_trajectory(conn, win_infs)
            if not win_traj:
                continue
            win_model = win_infs[0].get("model_name") or win_infs[0].get("variant_name", "unknown")
            win_variant = win_infs[0].get("variant_name", "unknown")

            for lose_ep in losers:
                lose_infs = inferences_by_episode.get(lose_ep, [])
                if not lose_infs:
                    continue
                # Skip same-variant pairs — they are trivially not comparable.
                lose_variant = lose_infs[0].get("variant_name", "unknown")
                if lose_variant == win_variant:
                    continue

                lose_traj = build_trajectory(conn, lose_infs)
                if not lose_traj:
                    continue
                lose_model = lose_infs[0].get("model_name") or lose_variant

                # Collect metadata from episode outcomes.
                win_outcome = outcomes[win_ep]
                lose_outcome = outcomes[lose_ep]
                win_tags = win_outcome.get("tags", {})
                lose_tags = lose_outcome.get("tags", {})

                error_cats = infer_error_categories(
                    [tc for step in lose_traj for tc in step.get("tool_calls", [])]
                )

                pairs.append({
                    "issue_id": issue_id,
                    "winner_model": win_model,
                    "winner_variant": win_variant,
                    "loser_model": lose_model,
                    "loser_variant": lose_variant,
                    "winner_trajectory": win_traj,
                    "loser_trajectory": lose_traj,
                    "verifier_outcome": {
                        "winner_episode": win_ep,
                        "loser_episode": lose_ep,
                        "winner_resolved": True,
                        "loser_resolved": False,
                        "winner_iterations": win_tags.get("iterations_used"),
                        "loser_iterations": lose_tags.get("iterations_used"),
                        "winner_created_at": win_outcome.get("created_at"),
                        "loser_created_at": lose_outcome.get("created_at"),
                    },
                    "iterations": {
                        "winner": win_tags.get("iterations_used"),
                        "loser": lose_tags.get("iterations_used"),
                    },
                    "error_categories": error_cats,
                    "winner_edit_count": sum(s.get("edit_count", 0) for s in win_traj),
                    "loser_edit_count": sum(s.get("edit_count", 0) for s in lose_traj),
                })

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract preference pairs from TensorZero telemetry for DPO training. "
            "Pairs winning (verifier-pass) trajectories against losing (verifier-fail) "
            "trajectories at per-tool-call granularity."
        )
    )
    parser.add_argument(
        "--pg-url",
        default=None,
        metavar="DSN",
        help=(
            "Postgres DSN for TZ database. "
            "Defaults to SWARM_TENSORZERO_PG_URL env var, then "
            f"{DEFAULT_PG_URL}"
        ),
    )
    parser.add_argument(
        "--output",
        default="data/tz-preference-pairs.jsonl",
        metavar="PATH",
        help="Output JSONL file path (default: data/tz-preference-pairs.jsonl)",
    )
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="Only include episodes created on or after this date",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=1,
        metavar="N",
        help="Minimum number of pairs required to write output (default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview statistics without writing any output file",
    )
    parser.add_argument(
        "--functions",
        default=None,
        metavar="F1,F2",
        help=(
            "Comma-separated TZ function names to include. "
            f"Default: {','.join(sorted(WORKER_FUNCTIONS))}"
        ),
    )

    args = parser.parse_args()

    pg_url = resolve_pg_url(args.pg_url)
    functions = (
        frozenset(f.strip() for f in args.functions.split(","))
        if args.functions
        else WORKER_FUNCTIONS
    )

    print(f"Connecting to TZ Postgres: {pg_url}")
    conn = connect_pg(pg_url)

    print("Fetching verifier outcomes (task_resolved feedback)...")
    outcomes = fetch_verifier_outcomes(conn, since=args.since, functions=functions)
    print(f"  Found {len(outcomes)} episodes with task_resolved feedback")

    resolved_count = sum(1 for o in outcomes.values() if o["resolved"])
    failed_count = len(outcomes) - resolved_count
    print(f"  Resolved (winners): {resolved_count}  |  Failed (losers): {failed_count}")

    if not outcomes:
        print("No episodes found. Let the swarm run longer to accumulate feedback data.")
        conn.close()
        sys.exit(0)

    episode_ids = list(outcomes.keys())

    print(f"Fetching inferences for {len(episode_ids)} episodes (functions: {', '.join(sorted(functions))})...")
    inferences_by_episode = fetch_inferences_for_episodes(conn, episode_ids, functions)
    total_inferences = sum(len(v) for v in inferences_by_episode.values())
    print(f"  Found {total_inferences} inference records across {len(inferences_by_episode)} episodes")

    print("Building preference pairs...")
    pairs = make_pairs(outcomes, inferences_by_episode, conn, min_pairs=args.min_pairs)
    print(f"  Built {len(pairs)} preference pairs")

    conn.close()

    # Summary statistics.
    if pairs:
        model_pairs: dict[str, int] = defaultdict(int)
        for p in pairs:
            key = f"{p['winner_model']} > {p['loser_model']}"
            model_pairs[key] += 1
        print("\nPair breakdown by model matchup:")
        for matchup, count in sorted(model_pairs.items(), key=lambda x: -x[1]):
            print(f"  {matchup}: {count}")

        error_cat_counts: dict[str, int] = defaultdict(int)
        for p in pairs:
            for cat in p["error_categories"]:
                error_cat_counts[cat] += 1
        if error_cat_counts:
            print("\nError categories found in loser trajectories:")
            for cat, count in sorted(error_cat_counts.items(), key=lambda x: -x[1]):
                print(f"  {cat}: {count}")

    if len(pairs) < args.min_pairs:
        print(
            f"\nOnly {len(pairs)} pairs found (need {args.min_pairs}). "
            "Let the swarm run longer to accumulate success/failure data.",
            file=sys.stderr,
        )
        sys.exit(0)

    if args.dry_run:
        print(f"\n[dry-run] Would write {len(pairs)} pairs to {args.output}")
        sys.exit(0)

    # Ensure output directory exists.
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, default=str) + "\n")

    print(f"\nWrote {len(pairs)} preference pairs to {args.output}")


if __name__ == "__main__":
    main()
