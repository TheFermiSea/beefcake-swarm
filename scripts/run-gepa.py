#!/usr/bin/env python3
"""Run TensorZero GEPA (prompt optimization) on swarm functions.

Connects to TZ gateway and Postgres to:
  1. Build a dataset from recent successful inferences
  2. Launch GEPA optimization targeting a function/evaluation pair
  3. Poll for completion, printing progress
  4. Output new variant templates and statistics

Requires: tensorzero, psycopg2-binary, asyncio

Usage:
  python scripts/run-gepa.py
  python scripts/run-gepa.py --function code_fixing --evaluation code_fixing_quality
  python scripts/run-gepa.py --max-iterations 10 --analysis-model openai::gpt-5
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone

try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary", file=sys.stderr)
    sys.exit(1)

try:
    from tensorzero import AsyncTensorZeroGateway
except ImportError:
    print("ERROR: tensorzero not installed. Run: pip install tensorzero", file=sys.stderr)
    sys.exit(1)


DEFAULT_GATEWAY_URL = "http://localhost:3000"
DEFAULT_PG_URL = "postgresql://tensorzero:tensorzero@localhost:5433/tensorzero"
DEFAULT_FUNCTION = "worker_code_edit"
DEFAULT_EVALUATION = "worker_code_quality"
DEFAULT_MAX_ITERATIONS = 5
DEFAULT_ANALYSIS_MODEL = "openai::gpt-5-mini"
DEFAULT_MUTATION_MODEL = "openai::gpt-5-mini"
POLL_INTERVAL_SECS = 30
MIN_DATAPOINTS = 3

# Map function names to their known variant lists
FUNCTION_VARIANTS = {
    "worker_code_edit": [
        "qwen35_27b",
        "devstral_24b",
        "sera_14b_worker",
    ],
    "code_fixing": [
        "qwen35_fixer",
        "devstral_fixer",
        "sera_14b_fixer",
    ],
    "architect_plan": ["opus_verbose", "sonnet_concise"],
    "editor_apply": ["qwen_coder", "qwen_reasoning"],
    "task_planning": ["qwen_fast_planner", "qwen_planner"],
}


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def fetch_successful_inference_ids(pg_url: str, function_name: str, limit: int = 50) -> list[str]:
    """Query TZ Postgres for inference IDs where task_resolved=true."""
    log(f"Connecting to Postgres: {pg_url.split('@')[-1]}")

    conn = psycopg2.connect(pg_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            # Find episodes where task_resolved=true, then get their inference IDs
            cur.execute(
                """
                SELECT DISTINCT ci.id::text AS inference_id
                FROM boolean_metric_feedback bmf
                JOIN chat_inferences ci
                    ON ci.episode_id = bmf.target_id
                WHERE bmf.metric_name = 'task_resolved'
                  AND bmf.value = true
                  AND ci.function_name = %s
                ORDER BY ci.id DESC
                LIMIT %s
                """,
                (function_name, limit),
            )
            rows = cur.fetchall()
            ids = [row["inference_id"] for row in rows]
            log(f"Found {len(ids)} successful inference IDs for {function_name}")
            return ids
    finally:
        conn.close()


async def run_gepa(args: argparse.Namespace) -> None:
    gateway_url = args.gateway_url or os.environ.get("SWARM_TENSORZERO_URL", DEFAULT_GATEWAY_URL)
    pg_url = args.pg_url or os.environ.get("SWARM_TENSORZERO_PG_URL", DEFAULT_PG_URL)
    function_name = args.function
    evaluation_name = args.evaluation
    max_iterations = args.max_iterations
    analysis_model = args.analysis_model
    mutation_model = args.mutation_model

    log(f"GEPA optimization for function={function_name}, evaluation={evaluation_name}")
    log(f"Gateway: {gateway_url}")
    log(f"Models: analysis={analysis_model}, mutation={mutation_model}")

    # Step 1: Fetch successful inference IDs from Postgres
    inference_ids = fetch_successful_inference_ids(pg_url, function_name, limit=50)
    if len(inference_ids) < MIN_DATAPOINTS:
        log(
            f"ERROR: Only {len(inference_ids)} successful inferences found "
            f"(minimum {MIN_DATAPOINTS}). Run more swarm iterations first."
        )
        sys.exit(1)

    # Resolve initial variants
    initial_variants = FUNCTION_VARIANTS.get(function_name)
    if initial_variants is None:
        log(f"WARNING: No known variants for {function_name}. GEPA will use gateway defaults.")

    # Step 2: Connect to TZ gateway and create dataset
    dataset_name = f"{function_name}_gepa_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    log(f"Creating dataset: {dataset_name}")

    async with AsyncTensorZeroGateway(gateway_url) as t0:
        # Create the dataset
        dataset = await t0.create_dataset(name=dataset_name)
        log(f"Dataset created: {dataset.name}")

        # Add datapoints from successful inferences
        added = 0
        for inf_id in inference_ids:
            try:
                await t0.add_datapoint_to_dataset(
                    dataset_name=dataset.name,
                    inference_id=inf_id,
                )
                added += 1
            except Exception as e:
                log(f"  Skipping inference {inf_id}: {e}")

        log(f"Added {added}/{len(inference_ids)} datapoints to dataset")

        if added < MIN_DATAPOINTS:
            log(f"ERROR: Only {added} datapoints added (minimum {MIN_DATAPOINTS}). Aborting.")
            sys.exit(1)

        # Step 3: Launch GEPA optimization
        log(f"Launching GEPA (max_iterations={max_iterations})...")

        launch_kwargs = {
            "function_name": function_name,
            "dataset_name": dataset.name,
            "evaluation_name": evaluation_name,
            "analysis_model": analysis_model,
            "mutation_model": mutation_model,
            "max_iterations": max_iterations,
        }
        if initial_variants is not None:
            launch_kwargs["initial_variants"] = initial_variants

        result = await t0.optimization.gepa.launch(**launch_kwargs)
        task_id = result.task_id
        log(f"GEPA launched: task_id={task_id}")

        # Step 4: Poll for completion
        iteration = 0
        while True:
            status = await t0.optimization.gepa.get(task_id=task_id)

            current_status = status.get("status", "unknown") if isinstance(status, dict) else getattr(status, "status", "unknown")
            iteration += 1

            if isinstance(status, dict):
                progress = status.get("progress", {})
                current_iter = progress.get("current_iteration", "?")
                total_iter = progress.get("total_iterations", max_iterations)
                best_score = progress.get("best_score", "?")
            else:
                current_iter = getattr(status, "current_iteration", "?")
                total_iter = getattr(status, "total_iterations", max_iterations)
                best_score = getattr(status, "best_score", "?")

            log(
                f"  Poll #{iteration}: status={current_status}, "
                f"iteration={current_iter}/{total_iter}, best_score={best_score}"
            )

            if current_status == "completed":
                log("GEPA optimization completed!")
                break
            elif current_status == "failed":
                error = status.get("error", "unknown") if isinstance(status, dict) else getattr(status, "error", "unknown")
                log(f"ERROR: GEPA optimization failed: {error}")
                sys.exit(1)
            elif current_status == "cancelled":
                log("GEPA optimization was cancelled.")
                sys.exit(1)

            await asyncio.sleep(POLL_INTERVAL_SECS)

        # Step 5: Print results
        log("=" * 60)
        log("GEPA Optimization Results")
        log("=" * 60)

        if isinstance(status, dict):
            results = status.get("results", {})
            variants = results.get("variants", [])
            stats = results.get("statistics", {})
        else:
            variants = getattr(status, "variants", [])
            stats = getattr(status, "statistics", {})

        if variants:
            log(f"\nNew variant templates ({len(variants)}):")
            for i, variant in enumerate(variants):
                if isinstance(variant, dict):
                    name = variant.get("name", f"variant_{i}")
                    score = variant.get("score", "N/A")
                    template = variant.get("template", "")
                else:
                    name = getattr(variant, "name", f"variant_{i}")
                    score = getattr(variant, "score", "N/A")
                    template = getattr(variant, "template", "")

                log(f"\n--- {name} (score: {score}) ---")
                if template:
                    # Print first 500 chars of template
                    preview = template[:500] + ("..." if len(str(template)) > 500 else "")
                    print(preview)

        if stats:
            log(f"\nStatistics:")
            if isinstance(stats, dict):
                for k, v in stats.items():
                    log(f"  {k}: {v}")
            else:
                log(f"  {stats}")

        # Dump full result as JSON for programmatic consumption
        result_path = f"/tmp/gepa-result-{function_name}-{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}.json"
        try:
            with open(result_path, "w") as f:
                json.dump(
                    status if isinstance(status, dict) else str(status),
                    f,
                    indent=2,
                    default=str,
                )
            log(f"\nFull result written to: {result_path}")
        except Exception as e:
            log(f"Could not write result file: {e}")

        log("\nDone.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run TensorZero GEPA prompt optimization on swarm functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s
  %(prog)s --function code_fixing --evaluation code_fixing_quality
  %(prog)s --max-iterations 10 --analysis-model openai::gpt-5
  %(prog)s --gateway-url http://ai-proxy:3000 --pg-url postgresql://tz:tz@localhost:5433/tensorzero
        """,
    )
    parser.add_argument(
        "--gateway-url",
        default=None,
        help=f"TZ gateway URL (default: $SWARM_TENSORZERO_URL or {DEFAULT_GATEWAY_URL})",
    )
    parser.add_argument(
        "--pg-url",
        default=None,
        help=f"TZ Postgres URL (default: $SWARM_TENSORZERO_PG_URL or {DEFAULT_PG_URL})",
    )
    parser.add_argument(
        "--function",
        default=DEFAULT_FUNCTION,
        help=f"TZ function name to optimize (default: {DEFAULT_FUNCTION})",
    )
    parser.add_argument(
        "--evaluation",
        default=DEFAULT_EVALUATION,
        help=f"TZ evaluation name (default: {DEFAULT_EVALUATION})",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Maximum GEPA iterations (default: {DEFAULT_MAX_ITERATIONS})",
    )
    parser.add_argument(
        "--analysis-model",
        default=DEFAULT_ANALYSIS_MODEL,
        help=f"Model for GEPA analysis step (default: {DEFAULT_ANALYSIS_MODEL})",
    )
    parser.add_argument(
        "--mutation-model",
        default=DEFAULT_MUTATION_MODEL,
        help=f"Model for GEPA mutation step (default: {DEFAULT_MUTATION_MODEL})",
    )

    args = parser.parse_args()
    asyncio.run(run_gepa(args))


if __name__ == "__main__":
    main()
