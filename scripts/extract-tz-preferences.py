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


# Default Postgres DSN for a local TensorZero database.
#
# Credentials should come from explicit args or environment variables rather than
# being baked into the script.
DEFAULT_PG_URL = "postgresql://localhost:5433/tensorzero"

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


def parse_tool_arguments(arguments: Any) -> Any:
    """Normalize tool arguments to Python objects."""
    if not isinstance(arguments, str):
        return arguments

    try:
        return json.loads(arguments)
    except json.JSONDecodeError:
        return {"_raw": arguments}


def normalize_output_blocks(output: Any) -> list[Any]:
    """Coerce TensorZero output into a list of content/tool-call blocks."""
    if output is None:
        return []
    if isinstance(output, list):
        return output
    if isinstance(output, dict):
        blocks: list[Any] = []
        content = output.get("content")
        if isinstance(content, list):
            blocks.extend(content)
        tool_calls = output.get("tool_calls") or []
        if tool_calls:
            blocks.extend(list(tool_calls))
        return blocks
    if isinstance(output, str):
        return [{"type": "text", "content": output, "name": None}]
    return []


def make_tool_call_step(
    *,
    name: str,
    arguments: Any,
    result: Any,
    raw_name: str | None,
) -> dict[str, Any]:
    """Build a normalized tool-call step."""
    return {
        "type": "tool_call",
        "name": name,
        "arguments": parse_tool_arguments(arguments),
        "result": result,
        "raw_name": raw_name,
        "is_edit": name in EDIT_TOOL_NAMES if name else False,
    }


def extract_block_step(block: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one TensorZero/OpenAI output block into a trajectory step."""
    block_type = block.get("type", "")
    if block_type in {"tool_call", "tool_use"}:
        return make_tool_call_step(
            name=(
                block.get("name")
                or block.get("raw_name")
                or block.get("function", {}).get("name", "")
            ),
            arguments=block.get("arguments") or block.get("input") or {},
            result=block.get("result"),
            raw_name=block.get("raw_name"),
        )
    if block_type == "text":
        return {
            "type": "text",
            "content": block.get("text") or block.get("content") or "",
            "name": None,
            "is_edit": False,
        }
    if "function" in block:
        func = block["function"]
        name = func.get("name", "")
        return make_tool_call_step(
            name=name,
            arguments=func.get("arguments", {}),
            result=None,
            raw_name=name,
        )
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
    calls: list[dict[str, Any]] = []
    for block in normalize_output_blocks(output):
        if not isinstance(block, dict):
            continue
        step = extract_block_step(block)
        if step is not None:
            calls.append(step)

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


def group_episodes_by_issue(
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, dict[str, list[str]]]:
    """Group episode ids into success/failure buckets keyed by issue/repo id."""
    groups: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"success": [], "failure": []})
    for episode_id, outcome in outcomes.items():
        tags = outcome.get("tags", {})
        issue_id = tags.get("issue_id") or tags.get("repo_id") or "global"
        bucket = "success" if outcome["resolved"] else "failure"
        groups[issue_id][bucket].append(episode_id)
    return groups


def collect_episode_contexts(
    conn,
    episode_ids: list[str],
    inferences_by_episode: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build reusable model/trajectory metadata for a set of episodes."""
    contexts: list[dict[str, Any]] = []
    for episode_id in episode_ids:
        inferences = inferences_by_episode.get(episode_id, [])
        if not inferences:
            continue

        trajectory = build_trajectory(conn, inferences)
        if not trajectory:
            continue

        variant = inferences[0].get("variant_name", "unknown")
        contexts.append({
            "episode_id": episode_id,
            "trajectory": trajectory,
            "variant": variant,
            "model": inferences[0].get("model_name") or variant,
        })
    return contexts


def build_pair_record(
    issue_id: str,
    winner: dict[str, Any],
    loser: dict[str, Any],
    outcomes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build one preference-pair record from winner/loser episode contexts."""
    win_ep = winner["episode_id"]
    lose_ep = loser["episode_id"]
    win_traj = winner["trajectory"]
    lose_traj = loser["trajectory"]
    win_outcome = outcomes[win_ep]
    lose_outcome = outcomes[lose_ep]
    win_tags = win_outcome.get("tags", {})
    lose_tags = lose_outcome.get("tags", {})
    error_cats = infer_error_categories(
        [tc for step in lose_traj for tc in step.get("tool_calls", [])]
    )

    return {
        "issue_id": issue_id,
        "winner_model": winner["model"],
        "winner_variant": winner["variant"],
        "loser_model": loser["model"],
        "loser_variant": loser["variant"],
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
        "winner_edit_count": sum(step.get("edit_count", 0) for step in win_traj),
        "loser_edit_count": sum(step.get("edit_count", 0) for step in lose_traj),
    }


def make_pairs_for_issue(
    issue_id: str,
    buckets: dict[str, list[str]],
    outcomes: dict[str, dict[str, Any]],
    inferences_by_episode: dict[str, list[dict[str, Any]]],
    conn,
) -> list[dict[str, Any]]:
    """Build all cross-variant winner/loser pairs for one issue bucket."""
    winners = collect_episode_contexts(conn, buckets["success"], inferences_by_episode)
    losers = collect_episode_contexts(conn, buckets["failure"], inferences_by_episode)
    if not winners or not losers:
        return []

    return [
        build_pair_record(issue_id, winner, loser, outcomes)
        for winner in winners
        for loser in losers
        if loser["variant"] != winner["variant"]
    ]


def make_pairs(
    outcomes: dict[str, dict[str, Any]],
    inferences_by_episode: dict[str, list[dict[str, Any]]],
    conn,
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
    pairs: list[dict[str, Any]] = []
    for issue_id, buckets in group_episodes_by_issue(outcomes).items():
        pairs.extend(
            make_pairs_for_issue(issue_id, buckets, outcomes, inferences_by_episode, conn)
        )

    return pairs


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the preference extraction script."""
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
    return parser


def parse_functions_arg(functions_arg: str | None) -> frozenset[str]:
    """Parse the CLI function filter."""
    if not functions_arg:
        return WORKER_FUNCTIONS
    return frozenset(func.strip() for func in functions_arg.split(","))


def print_pair_summary(pairs: list[dict[str, Any]]) -> None:
    """Print aggregate pair statistics."""
    if not pairs:
        return

    model_pairs: dict[str, int] = defaultdict(int)
    for pair in pairs:
        key = f"{pair['winner_model']} > {pair['loser_model']}"
        model_pairs[key] += 1
    print("\nPair breakdown by model matchup:")
    for matchup, count in sorted(model_pairs.items(), key=lambda item: -item[1]):
        print(f"  {matchup}: {count}")

    error_cat_counts: dict[str, int] = defaultdict(int)
    for pair in pairs:
        for category in pair["error_categories"]:
            error_cat_counts[category] += 1
    if not error_cat_counts:
        return

    print("\nError categories found in loser trajectories:")
    for category, count in sorted(error_cat_counts.items(), key=lambda item: -item[1]):
        print(f"  {category}: {count}")


def write_pairs(output_path: str, pairs: list[dict[str, Any]]) -> None:
    """Persist preference pairs as JSONL."""
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, default=str) + "\n")

    print(f"\nWrote {len(pairs)} preference pairs to {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    pg_url = resolve_pg_url(args.pg_url)
    functions = parse_functions_arg(args.functions)

    print("Connecting to TZ Postgres...")
    conn = connect_pg(pg_url)

    print("Fetching verifier outcomes (task_resolved feedback)...")
    outcomes = fetch_verifier_outcomes(conn, since=args.since)
    print(f"  Found {len(outcomes)} episodes with task_resolved feedback")

    resolved_count = sum(1 for outcome in outcomes.values() if outcome["resolved"])
    failed_count = len(outcomes) - resolved_count
    print(f"  Resolved (winners): {resolved_count}  |  Failed (losers): {failed_count}")

    if not outcomes:
        print("No episodes found. Let the swarm run longer to accumulate feedback data.")
        conn.close()
        sys.exit(0)

    episode_ids = list(outcomes.keys())
    print(
        f"Fetching inferences for {len(episode_ids)} episodes "
        f"(functions: {', '.join(sorted(functions))})..."
    )
    inferences_by_episode = fetch_inferences_for_episodes(conn, episode_ids, functions)
    total_inferences = sum(len(records) for records in inferences_by_episode.values())
    print(f"  Found {total_inferences} inference records across {len(inferences_by_episode)} episodes")

    print("Building preference pairs...")
    pairs = make_pairs(outcomes, inferences_by_episode, conn)
    print(f"  Built {len(pairs)} preference pairs")
    conn.close()

    print_pair_summary(pairs)

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

    write_pairs(args.output, pairs)


if __name__ == "__main__":
    main()
