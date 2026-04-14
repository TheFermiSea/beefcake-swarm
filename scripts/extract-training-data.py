#!/usr/bin/env python3
"""Extract verified successful completions from TensorZero Postgres as LoRA training data.

Queries TZ's inference tables for episodes where task_resolved=true,
extracts the input/output messages including tool calls, and formats
them as JSONL compatible with unsloth/axolotl SFT training.

Usage:
    python3 scripts/extract-training-data.py --output training_data.jsonl
    python3 scripts/extract-training-data.py --output data.jsonl --max-iterations 3 --repo-id beefcake-swarm
    python3 scripts/extract-training-data.py --output data.jsonl --since 2026-03-20 --min-episodes 50

Dependencies: psycopg2-binary, argparse, json (stdlib)
    pip install psycopg2-binary
"""

import argparse
import json
import sys

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print(
        "Error: psycopg2-binary is required. Install with: pip install psycopg2-binary",
        file=sys.stderr,
    )
    sys.exit(1)

DEFAULT_PG_URL = "postgresql://tensorzero:tensorzero@localhost:5433/tensorzero"

# Tool names that indicate productive edits (not just read-only exploration).
EDIT_TOOL_NAMES = frozenset({
    "edit_file",
    "write_file",
    "proxy_edit_file",
    "proxy_write_file",
    "apply_patch",
    "apply_plan",
})


def connect_pg(pg_url: str):
    """Connect to TZ Postgres and return a connection with dict cursors."""
    try:
        conn = psycopg2.connect(pg_url)
        conn.set_session(readonly=True, autocommit=True)
        return conn
    except psycopg2.Error as e:
        print(f"Error: Failed to connect to TZ Postgres at {pg_url}: {e}", file=sys.stderr)
        sys.exit(1)


def fetch_successful_episodes(
    conn,
    since: str | None = None,
    repo_id: str | None = None,
    max_iterations: int | None = None,
) -> list[dict]:
    """Fetch episode IDs where task_resolved=true, with optional filters.

    Returns list of dicts with: episode_id, iterations_used, tags.
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    # Build the query: join boolean_metric_feedback (task_resolved=true)
    # with optional float_metric_feedback (iterations_used) for quality filtering.
    query = """
    WITH resolved_episodes AS (
        SELECT
            bf.target_id AS episode_id,
            bf.created_at,
            bf.tags
        FROM tensorzero.boolean_metric_feedback bf
        WHERE bf.metric_name = 'task_resolved'
          AND bf.value = true
    ),
    iteration_counts AS (
        SELECT
            ff.target_id AS episode_id,
            ff.value AS iterations_used
        FROM tensorzero.float_metric_feedback ff
        WHERE ff.metric_name = 'iterations_used'
    )
    SELECT
        re.episode_id::text,
        re.created_at,
        re.tags,
        ic.iterations_used
    FROM resolved_episodes re
    LEFT JOIN iteration_counts ic ON re.episode_id = ic.episode_id
    WHERE 1=1
    """
    params = []

    if since:
        query += " AND re.created_at >= %s::timestamptz"
        params.append(since)

    if repo_id:
        query += " AND re.tags->>'repo_id' = %s"
        params.append(repo_id)

    if max_iterations is not None:
        query += " AND (ic.iterations_used IS NULL OR ic.iterations_used <= %s)"
        params.append(float(max_iterations))

    query += " ORDER BY re.created_at DESC"

    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()

    episodes = []
    for row in rows:
        episodes.append({
            "episode_id": row["episode_id"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "tags": dict(row["tags"]) if row["tags"] else {},
            "iterations_used": row["iterations_used"],
        })

    return episodes


def fetch_inferences_for_episode(conn, episode_id: str) -> list[dict]:
    """Fetch all chat inferences and their model details for an episode.

    Returns list of dicts with: inference_id, function_name, variant_name,
    model_name, input_tokens, output_tokens, created_at.
    """
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    query = """
    SELECT
        ci.id::text AS inference_id,
        ci.function_name,
        ci.variant_name,
        ci.episode_id::text,
        ci.created_at,
        mi.model_name,
        mi.model_provider_name,
        mi.input_tokens,
        mi.output_tokens,
        mi.response_time_ms
    FROM tensorzero.chat_inferences ci
    LEFT JOIN tensorzero.model_inferences mi ON mi.inference_id = ci.id
    WHERE ci.episode_id = %s::uuid
    ORDER BY ci.created_at ASC
    """

    cur.execute(query, [episode_id])
    rows = cur.fetchall()
    cur.close()

    inferences = []
    for row in rows:
        inferences.append({
            "inference_id": row["inference_id"],
            "function_name": row["function_name"],
            "variant_name": row["variant_name"],
            "episode_id": row["episode_id"],
            "created_at": row["created_at"].isoformat() if row["created_at"] else None,
            "model_name": row["model_name"],
            "model_provider_name": row["model_provider_name"],
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "response_time_ms": row["response_time_ms"],
        })

    return inferences


def fetch_inference_data_batch(conn, inference_ids: list[str]) -> dict[str, dict]:
    """Fetch the actual input/output data for multiple chat inferences in one query.

    TZ partitions chat_inference_data by date. We query the parent table
    and let Postgres route to the correct partition.

    Returns dict mapping inference_id -> {input, output}.
    """
    if not inference_ids:
        return {}

    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

    query = """
    SELECT id::text AS inference_id, input, output
    FROM tensorzero.chat_inference_data
    WHERE id = ANY(%s::uuid[])
    """

    try:
        cur.execute(query, [inference_ids])
        rows = cur.fetchall()
        cur.close()

        result = {}
        for row in rows:
            result[row["inference_id"]] = {"input": row["input"], "output": row["output"]}
        return result
    except psycopg2.Error as e:
        cur.close()
        print(f"  Warning: Could not fetch data for batch of {len(inference_ids)} inferences: {e}", file=sys.stderr)
        # Reset the connection state after error in autocommit mode.
        conn.rollback()
        return {}


def has_edit_tool_calls(output) -> bool:
    """Check if the output contains tool calls for edit/write operations.

    The output format from TZ can be:
    - A list of content blocks (each may have type=tool_call)
    - A single dict with content/tool_calls
    - A raw string
    """
    if output is None:
        return False

    # If output is a list of content blocks (TZ format)
    if isinstance(output, list):
        for block in output:
            if isinstance(block, dict):
                # Check for tool_call type blocks
                if block.get("type") == "tool_call":
                    tool_name = block.get("name", "") or block.get("function", {}).get("name", "")
                    if tool_name in EDIT_TOOL_NAMES:
                        return True
                # Check for raw_name field (TZ variant)
                raw_name = block.get("raw_name", "")
                if raw_name in EDIT_TOOL_NAMES:
                    return True
        return False

    # If output is a dict (OpenAI-style response)
    if isinstance(output, dict):
        # Check message.tool_calls
        tool_calls = output.get("tool_calls", [])
        if not tool_calls:
            message = output.get("message", {})
            if isinstance(message, dict):
                tool_calls = message.get("tool_calls", [])

        for tc in (tool_calls or []):
            func = tc.get("function", {})
            name = func.get("name", "") if isinstance(func, dict) else ""
            if name in EDIT_TOOL_NAMES:
                return True

        # Check content blocks within dict
        content = output.get("content", [])
        if isinstance(content, list):
            return has_edit_tool_calls(content)

        return False

    # String output — check for tool name mentions (heuristic fallback)
    if isinstance(output, str):
        return any(tool in output for tool in EDIT_TOOL_NAMES)

    return False


def format_messages_for_training(input_data, output_data) -> list[dict] | None:
    """Convert TZ inference input/output into OpenAI chat format for SFT training.

    Returns a list of message dicts: [{role, content, ...}, ...]
    Returns None if data is malformed.
    """
    messages = []

    # Process input messages
    if isinstance(input_data, list):
        for msg in input_data:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")

                formatted = {"role": role}

                # Handle content that is a list of content blocks
                if isinstance(content, list):
                    # Flatten text blocks, preserve tool_use/tool_result blocks
                    text_parts = []
                    tool_calls = []
                    for block in content:
                        if isinstance(block, dict):
                            btype = block.get("type", "text")
                            if btype == "text":
                                text_parts.append(block.get("text", ""))
                            elif btype == "tool_use":
                                tool_calls.append({
                                    "id": block.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        "name": block.get("name", ""),
                                        "arguments": json.dumps(block.get("input", {}))
                                        if isinstance(block.get("input"), dict)
                                        else str(block.get("input", "")),
                                    },
                                })
                            elif btype == "tool_result":
                                # Tool results become separate tool messages
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": block.get("tool_use_id", block.get("id", "")),
                                    "content": block.get("content", "")
                                    if isinstance(block.get("content"), str)
                                    else json.dumps(block.get("content", "")),
                                })
                                continue
                        elif isinstance(block, str):
                            text_parts.append(block)

                    if text_parts:
                        formatted["content"] = "\n".join(text_parts)
                    else:
                        formatted["content"] = ""
                    if tool_calls:
                        formatted["tool_calls"] = tool_calls
                elif isinstance(content, str):
                    formatted["content"] = content
                else:
                    formatted["content"] = json.dumps(content)

                messages.append(formatted)
    elif isinstance(input_data, dict):
        # Single message dict
        messages.append(input_data)
    else:
        return None

    # Process output (assistant response)
    if output_data is not None:
        assistant_msg = {"role": "assistant"}

        if isinstance(output_data, list):
            # List of content blocks
            text_parts = []
            tool_calls = []
            for block in output_data:
                if isinstance(block, dict):
                    btype = block.get("type", "text")
                    if btype == "text":
                        text_parts.append(block.get("text", ""))
                    elif btype == "tool_call":
                        name = block.get("name", "") or block.get("raw_name", "")
                        raw_args = block.get("raw_arguments", block.get("arguments", "{}"))
                        tool_calls.append({
                            "id": block.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": raw_args
                                if isinstance(raw_args, str)
                                else json.dumps(raw_args),
                            },
                        })
                elif isinstance(block, str):
                    text_parts.append(block)

            assistant_msg["content"] = "\n".join(text_parts) if text_parts else None
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
        elif isinstance(output_data, dict):
            assistant_msg["content"] = output_data.get("content", "")
            if "tool_calls" in output_data:
                assistant_msg["tool_calls"] = output_data["tool_calls"]
        elif isinstance(output_data, str):
            assistant_msg["content"] = output_data
        else:
            assistant_msg["content"] = json.dumps(output_data)

        messages.append(assistant_msg)

    if not messages:
        return None

    return messages


def extract_tools_from_messages(messages: list[dict]) -> list[dict]:
    """Extract unique tool definitions from message tool_calls for the tools field.

    Returns a list of tool definitions in OpenAI function-calling format.
    This is a best-effort extraction since TZ doesn't store the original tool schemas.
    """
    seen_tools = {}
    for msg in messages:
        for tc in msg.get("tool_calls", []):
            func = tc.get("function", {})
            name = func.get("name", "")
            if name and name not in seen_tools:
                # We only know the name; args schema is not stored in TZ.
                # Provide a minimal tool definition for training.
                seen_tools[name] = {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": f"Tool: {name}",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }

    return list(seen_tools.values())


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code."""
    return len(text) // 4


def main():
    parser = argparse.ArgumentParser(
        description="Extract verified successful completions from TZ Postgres as LoRA training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --output training_data.jsonl
  %(prog)s --output data.jsonl --max-iterations 3 --repo-id beefcake-swarm
  %(prog)s --output data.jsonl --since 2026-03-20 --min-episodes 50 --function worker_code_edit
        """,
    )
    parser.add_argument(
        "--output", "-o",
        default="training_data.jsonl",
        help="Output JSONL file path (default: training_data.jsonl)",
    )
    parser.add_argument(
        "--pg-url",
        default=DEFAULT_PG_URL,
        help=f"TZ Postgres connection URL (default: {DEFAULT_PG_URL})",
    )
    parser.add_argument(
        "--min-episodes",
        type=int,
        default=0,
        help="Minimum episodes to extract; warn if fewer found (default: 0 = no minimum)",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help="Filter by repo_id tag (e.g. beefcake-swarm, rust-daq)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Only include episodes resolved in <= N iterations (fewer = cleaner signal)",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="Only include episodes after this date (ISO format, e.g. 2026-03-20)",
    )
    parser.add_argument(
        "--function",
        default=None,
        help="Filter inferences by TZ function name (e.g. worker_code_edit, code_fixing)",
    )
    parser.add_argument(
        "--require-edits",
        action="store_true",
        default=True,
        help="Only keep completions with edit/write tool calls (default: true)",
    )
    parser.add_argument(
        "--no-require-edits",
        action="store_true",
        help="Include all completions, not just those with edit tool calls",
    )
    parser.add_argument(
        "--include-cloud",
        action="store_true",
        help="Include cloud model inferences (manager/delegation). By default, only local model data is extracted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show statistics without writing output file",
    )

    args = parser.parse_args()

    require_edits = not args.no_require_edits

    print(f"Connecting to TZ Postgres at {args.pg_url}...")
    conn = connect_pg(args.pg_url)

    # Step 1: Find successful episodes
    print("Querying successful episodes (task_resolved=true)...")
    episodes = fetch_successful_episodes(
        conn,
        since=args.since,
        repo_id=args.repo_id,
        max_iterations=args.max_iterations,
    )

    print(f"  Found {len(episodes)} successful episodes")

    if args.min_episodes and len(episodes) < args.min_episodes:
        print(
            f"  Warning: Found {len(episodes)} episodes, below minimum of {args.min_episodes}",
            file=sys.stderr,
        )

    if not episodes:
        print("No successful episodes found matching filters. Exiting.", file=sys.stderr)
        conn.close()
        sys.exit(1)

    # Cloud function names (manager roles — typically not useful for local model LoRA)
    cloud_functions = frozenset({
        "cloud_manager_delegation",
        "architect_plan",
    })

    # Step 2: For each episode, fetch inferences and their data
    training_samples = []
    total_inferences = 0
    skipped_no_data = 0
    skipped_no_edits = 0
    skipped_cloud = 0
    skipped_function = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for i, ep in enumerate(episodes):
        ep_id = ep["episode_id"]
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  Processing episode {i + 1}/{len(episodes)}: {ep_id[:12]}...")

        inferences = fetch_inferences_for_episode(conn, ep_id)
        if not inferences:
            skipped_no_data += 1
            continue

        # Pre-fetch all inference data for this episode in one query
        inf_ids_to_fetch = [inf["inference_id"] for inf in inferences]
        batch_data = fetch_inference_data_batch(conn, inf_ids_to_fetch)

        for inf in inferences:
            total_inferences += 1

            # Filter by function name if specified
            if args.function and inf["function_name"] != args.function:
                skipped_function += 1
                continue

            # Skip cloud model inferences unless explicitly requested
            if not args.include_cloud and inf["function_name"] in cloud_functions:
                skipped_cloud += 1
                continue

            # Fetch the actual message data
            data = batch_data.get(inf["inference_id"])
            if not data:
                skipped_no_data += 1
                continue

            input_data = data["input"]
            output_data = data["output"]

            # Filter: must contain edit tool calls
            if require_edits and not has_edit_tool_calls(output_data):
                skipped_no_edits += 1
                continue

            # Format as training messages
            messages = format_messages_for_training(input_data, output_data)
            if not messages:
                skipped_no_data += 1
                continue

            # Extract tool definitions from the conversation
            tools = extract_tools_from_messages(messages)

            # Build training sample
            sample = {
                "messages": messages,
                "metadata": {
                    "episode_id": ep_id,
                    "function_name": inf["function_name"],
                    "variant_name": inf["variant_name"],
                    "model": inf["model_name"],
                    "iterations": ep["iterations_used"],
                    "repo_id": ep["tags"].get("repo_id"),
                    "source_created_at": inf["created_at"],
                },
            }

            # Only include tools if there are any
            if tools:
                sample["tools"] = tools

            training_samples.append(sample)

            # Accumulate token counts
            if inf["input_tokens"]:
                total_input_tokens += inf["input_tokens"]
            if inf["output_tokens"]:
                total_output_tokens += inf["output_tokens"]

    conn.close()

    # Step 3: Write output
    if not training_samples:
        print("No training samples extracted after filtering. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Deduplicate by inference content hash (same input+output = same sample)
    seen_hashes = set()
    deduped_samples = []
    for sample in training_samples:
        # Hash on the serialized messages (deterministic since we control formatting)
        content_key = json.dumps(sample["messages"], sort_keys=True, ensure_ascii=False)
        h = hash(content_key)
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_samples.append(sample)

    dupes_removed = len(training_samples) - len(deduped_samples)
    training_samples = deduped_samples

    if not args.dry_run:
        with open(args.output, "w") as f:
            for sample in training_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"\nWrote {len(training_samples)} training samples to {args.output}")
    else:
        print("\n[Dry run — no output file written]")

    # Step 4: Print statistics
    # Estimate tokens from output content
    estimated_tokens = 0
    for sample in training_samples:
        estimated_tokens += estimate_tokens(json.dumps(sample["messages"]))

    # Count unique episodes, models, functions
    unique_episodes = set()
    model_counts: dict[str, int] = {}
    function_counts: dict[str, int] = {}
    for sample in training_samples:
        meta = sample["metadata"]
        unique_episodes.add(meta["episode_id"])
        model = meta.get("model") or "unknown"
        model_counts[model] = model_counts.get(model, 0) + 1
        fn = meta.get("function_name") or "unknown"
        function_counts[fn] = function_counts.get(fn, 0) + 1

    iter_values = [
        s["metadata"]["iterations"]
        for s in training_samples
        if s["metadata"].get("iterations") is not None
    ]

    print(f"\n{'=' * 60}")
    print("Training Data Extraction Summary")
    print(f"{'=' * 60}")
    print(f"  Episodes (unique):      {len(unique_episodes)}")
    print(f"  Training samples:       {len(training_samples)}")
    print(f"  Duplicates removed:     {dupes_removed}")
    print(f"  Total inferences seen:  {total_inferences}")
    print(f"  Skipped (no data):      {skipped_no_data}")
    print(f"  Skipped (no edits):     {skipped_no_edits}")
    print(f"  Skipped (cloud):        {skipped_cloud}")
    if args.function:
        print(f"  Skipped (function):     {skipped_function}")
    print(f"  TZ input tokens:        {total_input_tokens:,}")
    print(f"  TZ output tokens:       {total_output_tokens:,}")
    print(f"  Estimated tokens:       ~{estimated_tokens:,}")

    if iter_values:
        avg_iter = sum(iter_values) / len(iter_values)
        print(f"  Avg iterations/episode: {avg_iter:.1f}")

    print("\n  By model:")
    for model, count in sorted(model_counts.items(), key=lambda x: -x[1]):
        print(f"    {model}: {count}")

    print("\n  By function:")
    for fn, count in sorted(function_counts.items(), key=lambda x: -x[1]):
        print(f"    {fn}: {count}")

    print(f"{'=' * 60}")

    # Warn if below minimum
    if args.min_episodes and len(unique_episodes) < args.min_episodes:
        print(
            f"\nWarning: Extracted {len(unique_episodes)} episodes, "
            f"below requested minimum of {args.min_episodes}.",
            file=sys.stderr,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
