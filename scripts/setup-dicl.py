#!/usr/bin/env python3
"""Set up TensorZero DICL (Dynamic In-Context Learning) for swarm functions.

Connects to TZ gateway and Postgres to:
  1. Query for successful inferences (task_resolved=true)
  2. Create a TZ dataset with those as datapoints
  3. Run the DICL optimization workflow (embeds examples for retrieval)

Once complete, the DICL variants in tensorzero.toml will dynamically inject
the k most-similar past successful fixes into the prompt at inference time.

Requires: tensorzero, psycopg2-binary, asyncio

Usage:
  python scripts/setup-dicl.py
  python scripts/setup-dicl.py --function code_fixing --template-variant qwen35_fixer
  python scripts/setup-dicl.py --embedding-model nomic_embed --k 5
"""

import argparse
import asyncio
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
    from tensorzero import AsyncTensorZeroGateway, DICLOptimizationConfig
except ImportError:
    print("ERROR: tensorzero not installed. Run: pip install tensorzero", file=sys.stderr)
    sys.exit(1)


DEFAULT_GATEWAY_URL = "http://localhost:3000"
DEFAULT_PG_URL = "postgresql://tensorzero:tensorzero@localhost:5433/tensorzero"
POLL_INTERVAL_SECS = 15
MIN_DATAPOINTS = 3

# Default configurations per function
FUNCTION_DEFAULTS = {
    "worker_code_edit": {
        "template_variant": "qwen35_27b",
        "embedding_model": "nomic_embed",
        "model": "qwen_coder",
        "k": 3,
    },
    "code_fixing": {
        "template_variant": "qwen35_fixer",
        "embedding_model": "nomic_embed",
        "model": "qwen_coder",
        "k": 3,
    },
}


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{ts}] {msg}", flush=True)


def fetch_successful_inference_ids(
    pg_url: str, function_name: str, limit: int = 100
) -> list[str]:
    """Query TZ Postgres for inference IDs where task_resolved=true."""
    log(f"Connecting to Postgres: {pg_url.split('@')[-1]}")

    conn = psycopg2.connect(pg_url)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
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


async def setup_dicl(args: argparse.Namespace) -> None:
    gateway_url = args.gateway_url or os.environ.get(
        "SWARM_TENSORZERO_URL", DEFAULT_GATEWAY_URL
    )
    pg_url = args.pg_url or os.environ.get("SWARM_TENSORZERO_PG_URL", DEFAULT_PG_URL)
    function_name = args.function

    # Resolve defaults for this function
    defaults = FUNCTION_DEFAULTS.get(function_name, FUNCTION_DEFAULTS["worker_code_edit"])
    template_variant = args.template_variant or defaults["template_variant"]
    embedding_model = args.embedding_model or defaults["embedding_model"]
    model = args.model or defaults["model"]
    k = args.k or defaults["k"]

    log(f"DICL setup for function={function_name}")
    log(f"  Template variant: {template_variant}")
    log(f"  Embedding model:  {embedding_model}")
    log(f"  Inference model:  {model}")
    log(f"  k (examples):     {k}")
    log(f"  Gateway:          {gateway_url}")

    # Step 1: Fetch successful inference IDs
    inference_ids = fetch_successful_inference_ids(
        pg_url, function_name, limit=args.max_examples
    )
    if len(inference_ids) < MIN_DATAPOINTS:
        log(
            f"ERROR: Only {len(inference_ids)} successful inferences found "
            f"(minimum {MIN_DATAPOINTS}). Run more swarm iterations to build "
            f"a corpus of successful fixes first."
        )
        sys.exit(1)

    # Step 2: Create dataset and add datapoints
    dataset_name = (
        f"{function_name}_dicl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"
    )
    log(f"Creating dataset: {dataset_name}")

    async with AsyncTensorZeroGateway(gateway_url) as t0:
        dataset = await t0.create_dataset(name=dataset_name)
        log(f"Dataset created: {dataset.name}")

        added = 0
        skipped = 0
        for inf_id in inference_ids:
            try:
                await t0.add_datapoint_to_dataset(
                    dataset_name=dataset.name,
                    inference_id=inf_id,
                )
                added += 1
            except Exception as e:
                skipped += 1
                if skipped <= 5:
                    log(f"  Skipping inference {inf_id}: {e}")
                elif skipped == 6:
                    log("  (suppressing further skip messages)")

        log(f"Added {added}/{len(inference_ids)} datapoints ({skipped} skipped)")

        if added < MIN_DATAPOINTS:
            log(
                f"ERROR: Only {added} datapoints added (minimum {MIN_DATAPOINTS}). "
                f"Aborting DICL setup."
            )
            sys.exit(1)

        # Step 3: Build DICL optimization config
        log("Configuring DICL optimization...")
        config = DICLOptimizationConfig(
            embedding_model=embedding_model,
            k=k,
            model=model,
        )

        # Step 4: Launch DICL optimization workflow
        log(
            f"Launching DICL optimization (embedding {added} examples with "
            f"{embedding_model})..."
        )
        job_handle = await t0.experimental_launch_optimization_workflow(
            function_name=function_name,
            template_variant_name=template_variant,
            dataset_name=dataset.name,
            optimizer_config=config,
        )

        job_id = getattr(job_handle, "optimization_id", None) or getattr(
            job_handle, "task_id", None
        )
        log(f"DICL optimization launched: {job_id or job_handle}")

        # Step 5: Poll for completion
        poll_count = 0
        while True:
            poll_count += 1
            try:
                status = await t0.poll_optimization(job_handle)
            except AttributeError:
                # Older SDK versions may not have poll_optimization
                # Fall back to checking status directly
                log(
                    "  SDK does not support poll_optimization; "
                    "check TZ UI for completion status."
                )
                break

            current_status = (
                status.get("status", "unknown")
                if isinstance(status, dict)
                else getattr(status, "status", "unknown")
            )

            log(f"  Poll #{poll_count}: status={current_status}")

            if current_status == "completed":
                log("DICL optimization completed successfully!")

                # Print summary
                if isinstance(status, dict):
                    num_embedded = status.get("num_embedded", added)
                    variant_name = status.get("variant_name", "dicl_worker")
                else:
                    num_embedded = getattr(status, "num_embedded", added)
                    variant_name = getattr(status, "variant_name", "dicl_worker")

                log(f"\n{'=' * 60}")
                log("DICL Setup Complete")
                log(f"{'=' * 60}")
                log(f"  Function:     {function_name}")
                log(f"  Dataset:      {dataset.name}")
                log(f"  Embedded:     {num_embedded} examples")
                log(f"  k:            {k} examples per inference")
                log(f"  Variant:      {variant_name}")
                log(f"\nThe DICL variant will now inject the {k} most-similar")
                log(f"past successful fixes into the prompt at inference time.")
                log(f"\nEnsure the DICL variant is listed in the function's")
                log(f"experimentation.candidate_variants in tensorzero.toml.")
                break

            elif current_status in ("failed", "error"):
                error = (
                    status.get("error", "unknown")
                    if isinstance(status, dict)
                    else getattr(status, "error", "unknown")
                )
                log(f"ERROR: DICL optimization failed: {error}")
                sys.exit(1)

            elif current_status == "cancelled":
                log("DICL optimization was cancelled.")
                sys.exit(1)

            await asyncio.sleep(POLL_INTERVAL_SECS)

    log("\nDone.")


def setup_all(args: argparse.Namespace) -> None:
    """Set up DICL for all known functions."""
    functions = list(FUNCTION_DEFAULTS.keys())
    log(f"Setting up DICL for all functions: {', '.join(functions)}")

    for func_name in functions:
        log(f"\n{'=' * 60}")
        log(f"Setting up DICL for: {func_name}")
        log(f"{'=' * 60}")

        # Clone args and override function
        func_args = argparse.Namespace(**vars(args))
        func_args.function = func_name
        # Reset per-function overrides to None so defaults are used
        func_args.template_variant = None
        func_args.model = None

        try:
            asyncio.run(setup_dicl(func_args))
        except SystemExit as e:
            if e.code != 0:
                log(f"WARNING: DICL setup failed for {func_name}, continuing...")
            continue


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up TensorZero DICL for swarm functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DICL (Dynamic In-Context Learning) retrieves the k most-similar past
successful fixes and injects them into the prompt at inference time.
This requires an embedding model (nomic-embed) and a corpus of
successful inferences tracked via TensorZero feedback.

Examples:
  %(prog)s                                    # Set up worker_code_edit
  %(prog)s --function code_fixing             # Set up code_fixing
  %(prog)s --all                              # Set up all known functions
  %(prog)s --k 5 --max-examples 200           # More examples, deeper retrieval
  %(prog)s --embedding-model nomic_embed      # Explicit embedding model
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
        default="worker_code_edit",
        help="TZ function name (default: worker_code_edit)",
    )
    parser.add_argument(
        "--template-variant",
        default=None,
        help="Base template variant for DICL (default: per-function)",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Embedding model name in TZ config (default: nomic_embed)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Inference model for DICL variant (default: qwen_coder)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="Number of similar examples to retrieve (default: 3)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=100,
        help="Maximum examples to fetch from Postgres (default: 100)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="setup_all",
        help="Set up DICL for all known functions",
    )

    args = parser.parse_args()

    if args.setup_all:
        setup_all(args)
    else:
        asyncio.run(setup_dicl(args))


if __name__ == "__main__":
    main()
