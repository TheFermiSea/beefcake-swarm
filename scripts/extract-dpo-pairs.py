#!/usr/bin/env python3
"""Extract DPO (Direct Preference Optimization) training pairs from TensorZero.

Finds episodes where the same bug pattern or similar prompt was handled by
different model variants, one succeeding and one failing. The successful
completion becomes "chosen" and the failed one becomes "rejected".

This teaches the model "this approach works, this one doesn't" — much stronger
signal than SFT alone.

Usage:
    python3 scripts/extract-dpo-pairs.py --output /tmp/dpo-pairs.jsonl
    python3 scripts/extract-dpo-pairs.py --pg-url postgres://... --output pairs.jsonl

Output format (one per line):
    {"prompt": "...", "chosen": "...", "rejected": "...", "metadata": {...}}
"""

import argparse
import json
import os
import sys

try:
    import psycopg2
except ImportError:
    print("Error: psycopg2-binary required. pip install psycopg2-binary", file=sys.stderr)
    sys.exit(1)


def get_pg_url():
    """Find TZ Postgres URL from env or common defaults."""
    for var in ("SWARM_TENSORZERO_PG_URL", "TENSORZERO_POSTGRES_URL"):
        url = os.environ.get(var)
        if url:
            return url
    return "postgres://tensorzero:tensorzero@localhost:5433/tensorzero"


def extract_pairs(pg_url, output_path, min_pairs=5):
    """Extract preference pairs from TZ feedback data."""
    conn = psycopg2.connect(pg_url)
    cur = conn.cursor()

    # Find episodes with task_resolved feedback
    cur.execute("""
        SELECT
            bf.target_id as episode_id,
            bf.value as resolved,
            ci.function_name,
            ci.variant_name,
            ci.input as prompt_input,
            ci.output as completion_output
        FROM tensorzero.boolean_metric_feedback bf
        JOIN tensorzero.chat_inferences ci ON bf.target_id = ci.episode_id
        WHERE bf.metric_name = 'task_resolved'
        ORDER BY ci.function_name, ci.created_at
    """)

    rows = cur.fetchall()
    print(f"Found {len(rows)} episodes with feedback")

    # Group by function_name to find comparable prompts
    by_function = {}
    for episode_id, resolved, func, variant, prompt_input, completion_output in rows:
        if func not in by_function:
            by_function[func] = {"success": [], "failure": []}
        entry = {
            "episode_id": episode_id,
            "variant": variant,
            "prompt": prompt_input,
            "completion": completion_output,
        }
        if resolved:
            by_function[func]["success"].append(entry)
        else:
            by_function[func]["failure"].append(entry)

    # Create pairs: match successes with failures from same function
    pairs = []
    for func, data in by_function.items():
        successes = data["success"]
        failures = data["failure"]
        if not successes or not failures:
            continue

        # Pair each failure with a random success from the same function
        for fail in failures:
            # Pick a success (round-robin)
            success = successes[len(pairs) % len(successes)]

            # Extract the prompt (shared context)
            prompt = fail.get("prompt", "")
            if isinstance(prompt, dict):
                prompt = json.dumps(prompt)

            chosen = success.get("completion", "")
            rejected = fail.get("completion", "")
            if isinstance(chosen, dict):
                chosen = json.dumps(chosen)
            if isinstance(rejected, dict):
                rejected = json.dumps(rejected)

            if not prompt or not chosen or not rejected:
                continue

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "function": func,
                    "chosen_variant": success["variant"],
                    "rejected_variant": fail["variant"],
                    "chosen_episode": success["episode_id"],
                    "rejected_episode": fail["episode_id"],
                },
            })

    cur.close()
    conn.close()

    if len(pairs) < min_pairs:
        print(f"Only {len(pairs)} pairs found (need {min_pairs}). "
              "Let the swarm run longer to accumulate success/failure data.")
        return 0

    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Extracted {len(pairs)} DPO pairs to {output_path}")
    for func in by_function:
        s = len(by_function[func]["success"])
        f_count = len(by_function[func]["failure"])
        if s > 0 or f_count > 0:
            print(f"  {func}: {s} success, {f_count} failure")

    return len(pairs)


def main():
    parser = argparse.ArgumentParser(description="Extract DPO pairs from TZ")
    parser.add_argument("--output", default="/tmp/dpo-pairs.jsonl")
    parser.add_argument("--pg-url", default=None)
    parser.add_argument("--min-pairs", type=int, default=5)
    args = parser.parse_args()

    pg_url = args.pg_url or get_pg_url()
    extract_pairs(pg_url, args.output, args.min_pairs)


if __name__ == "__main__":
    main()
