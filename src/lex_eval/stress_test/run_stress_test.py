#!/usr/bin/env python3
"""Stress-test CLI — multi-target: lex-llm workflows, database, Cortecs.

Concurrency mode::

    uv run python -m lex_eval.stress_test.run_stress_test \\
        --target lex-llm --workflow-id chat_v1_gemma4_26b \\
        --concurrency 1 2 4 8 --n-queries 50

    uv run python -m lex_eval.stress_test.run_stress_test \\
        --target db --db-endpoint vector --db-host http://localhost:8000 \\
        --concurrency 8 --n-queries 100

    uv run python -m lex_eval.stress_test.run_stress_test \\
        --target cortecs --cortecs-host https://api.cortecs.ai \\
        --cortecs-model gemma-4-31b-it \\
        --concurrency 4 8 16 --n-queries 50

Rate mode: add ``--mode rate --rpm 100 --duration-min 10``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Any

from lex_eval.stress_test.load_driver import Runner, drive_load, drive_load_rate
from lex_eval.stress_test.query_sampler import sample_queries
from lex_eval.stress_test.results import (
    StressTestSummary,
    aggregate,
    print_summary,
    save_run_results,
)

SAMPLE_FILE = Path("data/lex-llm-conversations-sample.json")
DEFAULT_HOST = "https://dev1.lex-llm.au.dk"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stress-test under load.",
    )

    p.add_argument(
        "--target",
        choices=["lex-llm", "db", "cortecs"],
        default="lex-llm",
        help="What to stress-test (default: lex-llm).",
    )

    p.add_argument(
        "--mode",
        choices=["concurrency", "rate"],
        default="concurrency",
        help="Load pattern (default: concurrency).",
    )

    # ── concurrency-mode args ──
    p.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="One or more concurrency levels (default: 1 2 4 8).",
    )
    p.add_argument(
        "--n-queries",
        type=int,
        default=50,
        help="Number of queries per concurrency level (default: 50).",
    )

    # ── rate-mode args ──
    p.add_argument(
        "--rpm",
        type=float,
        default=100.0,
        help="Target requests per minute for rate mode (default: 100).",
    )
    p.add_argument(
        "--duration-min",
        type=float,
        default=10.0,
        help="Duration in minutes for rate mode (default: 10).",
    )

    # ── lex-llm target args ──
    p.add_argument(
        "--workflow-id",
        default="chat_v1_gemma4_26b",
        help="Workflow slug (lex-llm target only).",
    )
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="lex-llm service base URL.",
    )
    p.add_argument(
        "--token",
        default=os.getenv("LEX_LLM_TOKEN", "") or None,
        help="X-Auth-Token value (reads $LEX_LLM_TOKEN).",
    )

    # ── db target args ──
    p.add_argument(
        "--db-endpoint",
        choices=["vector", "text"],
        default="vector",
        help="Which db endpoint to hit (default: vector).",
    )
    p.add_argument(
        "--db-host",
        default="http://localhost:8000",
        help="Database service base URL.",
    )
    p.add_argument(
        "--db-index",
        default="lex",
        help="Vector/fulltext index name (default: lex).",
    )
    p.add_argument(
        "--db-top-k",
        type=int,
        default=5,
        help="Top-k for search (default: 5).",
    )
    p.add_argument(
        "--db-subqueries",
        type=int,
        default=4,
        help="Sub-queries per batch (default: 4).",
    )
    p.add_argument(
        "--db-token",
        default=os.getenv("DB_API_TOKEN", "") or None,
        help="X-Auth-Token for remote db (reads $DB_API_TOKEN if not given).",
    )

    # ── shared batching (used by all targets, especially db) ──
    p.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Group queries into batches of this size (joined with newline). "
        "0 = no batching (default). For db target, overrides --db-subqueries.",
    )

    # ── cortecs target args ──
    p.add_argument(
        "--cortecs-host",
        default="https://api.cortecs.ai",
        help="Cortecs API base URL.",
    )
    p.add_argument(
        "--cortecs-model",
        default="gemma-4-31b-it",
        help="Model name for Cortecs (default: gemma-4-31b-it).",
    )
    p.add_argument(
        "--cortecs-api-key",
        default=os.getenv("CORTECS_API_KEY", "") or None,
        help="Cortecs API key (reads $CORTECS_API_KEY if not given).",
    )
    p.add_argument(
        "--cortecs-max-tokens",
        type=int,
        default=1024,
        help="Max tokens to generate per request (default: 1024).",
    )

    # ── shared args ──
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for JSON output files (default: results/).",
    )
    p.add_argument(
        "--sample-file",
        type=Path,
        default=SAMPLE_FILE,
        help=f"Path to conversations sample JSON (default: {SAMPLE_FILE}).",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return p.parse_args(argv)


async def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.sample_file.exists():
        print(f"❌ Sample file not found: {args.sample_file}", file=sys.stderr)
        return 1

    # ── Build the runner ──
    run_one: Runner
    workflow_id: str  # used as label in summary
    runner_kwargs: dict[str, Any]

    if args.target == "lex-llm":
        from lex_eval.stress_test.runners.lex_llm import run_lex_llm

        run_one = run_lex_llm
        workflow_id = args.workflow_id
        runner_kwargs = {
            "host": args.host,
            "workflow_id": args.workflow_id,
            "token": args.token,
            "timeout": args.timeout,
        }
    elif args.target == "db":
        from lex_eval.stress_test.runners.db import run_db_vector, run_db_text

        if args.db_endpoint == "vector":
            run_one = run_db_vector
        else:
            run_one = run_db_text
        workflow_id = f"db-{args.db_endpoint}-{args.db_index}"
        runner_kwargs = {
            "host": args.db_host,
            "index_name": args.db_index,
            "top_k": args.db_top_k,
            "token": args.db_token,
            "timeout": args.timeout,
        }
        # Use --batch-size if given, otherwise --db-subqueries.
        if args.batch_size > 0:
            runner_kwargs["_batch_size"] = args.batch_size
        else:
            runner_kwargs["_batch_size"] = args.db_subqueries
    elif args.target == "cortecs":
        from lex_eval.stress_test.runners.cortecs import run_cortecs

        run_one = run_cortecs
        workflow_id = args.cortecs_model
        runner_kwargs = {
            "host": args.cortecs_host,
            "model": args.cortecs_model,
            "api_key": args.cortecs_api_key,
            "max_tokens": args.cortecs_max_tokens,
            "timeout": args.timeout,
        }
    else:
        raise AssertionError(f"Unknown target: {args.target}")

    if args.mode == "concurrency":
        return await _run_concurrency_mode(args, run_one, workflow_id, runner_kwargs)
    else:
        return await _run_rate_mode(args, run_one, workflow_id, runner_kwargs)


async def _run_concurrency_mode(
    args: argparse.Namespace,
    run_one: Runner,
    workflow_id: str,
    runner_kwargs: dict[str, Any],
) -> int:
    print("⏳ Mining queries from sample file …", flush=True)
    batch_size = runner_kwargs.pop("_batch_size", 1) or 1
    # We need batch_size × n_queries individual queries so we get
    # n_queries HTTP requests after chunking.
    n_individual = args.n_queries * batch_size * max(args.concurrency)
    queries = sample_queries(
        args.sample_file,
        n=n_individual,
        seed=42,
    )
    if len(queries) < args.n_queries * batch_size:
        print(
            f"⚠ Only {len(queries)} individual queries available "
            f"(wanted {args.n_queries * batch_size}). "
            f"Running with {len(queries)}.",
            file=sys.stderr,
        )
        effective_requests = max(1, len(queries) // batch_size)
    else:
        effective_requests = args.n_queries
        queries = queries[: args.n_queries * batch_size * max(args.concurrency)]

    summaries: list[StressTestSummary] = []

    for concurrency in args.concurrency:
        batch = queries[: effective_requests * batch_size]

        results, wall_clock_s = await drive_load(
            run_one=run_one,
            queries=batch,
            concurrency=concurrency,
            batch_size=batch_size,
            request_timeout=args.timeout,
            runner_kwargs=runner_kwargs,
        )

        summary = aggregate(
            results,
            workflow_id,
            wall_clock_s,
            target=args.target,
            concurrency=concurrency,
        )
        summaries.append(summary)

        saved_path = save_run_results(results, summary, args.output_dir)
        print_summary(summary)
        print(f"  📁 Results saved to {saved_path}\n")

    if len(summaries) > 1:
        print_combined_table(summaries)

    return 0


async def _run_rate_mode(
    args: argparse.Namespace,
    run_one: Runner,
    workflow_id: str,
    runner_kwargs: dict[str, Any],
) -> int:
    batch_size = runner_kwargs.pop("_batch_size", 1) or 1
    # The rate driver draws batch_size queries per HTTP request, so we
    # need batch_size × (rpm × duration) individual queries for full coverage.
    pool_size = max(200, int(args.rpm * args.duration_min * batch_size))
    print(f"⏳ Mining ~{pool_size} queries from sample file …", flush=True)
    queries = sample_queries(
        args.sample_file,
        n=pool_size,
        seed=42,
        max_scanned=5000,
    )
    print(f"  Collected {len(queries)} queries")

    results, wall_clock_s = await drive_load_rate(
        run_one=run_one,
        queries=queries,
        rpm=args.rpm,
        duration_min=args.duration_min,
        batch_size=batch_size,
        request_timeout=args.timeout,
        runner_kwargs=runner_kwargs,
    )

    summary = aggregate(
        results,
        workflow_id,
        wall_clock_s,
        target=args.target,
        target_rpm=args.rpm,
        duration_min=args.duration_min,
    )

    saved_path = save_run_results(results, summary, args.output_dir)
    print_summary(summary)
    print(f"  📁 Results saved to {saved_path}\n")
    return 0


def print_combined_table(summaries: list[StressTestSummary]) -> None:
    print(f"\n{'=' * 100}")
    print("COMBINED COMPARISON")
    print(f"{'=' * 100}")
    header = (
        f"{'Level':>8} {'Succ':>5} {'TTFT p50':>9} {'TTFT p95':>9} "
        f"{'E2E p50':>9} {'E2E p95':>9} {'RPS':>7} {'Spill%':>7}"
    )
    print(header)
    print("-" * 100)
    for s in summaries:
        level = f"c{s.concurrency}" if s.concurrency is not None else "-"
        print(
            f"{level:>8} {s.success_count:>5} "
            f"{s.ttft_p50_ms:>8.0f}ms {s.ttft_p95_ms:>8.0f}ms "
            f"{s.e2e_p50_ms:>8.0f}ms {s.e2e_p95_ms:>8.0f}ms "
            f"{s.throughput_rps:>6.1f} {s.backend_spillover_pct:>6.1f}%"
        )
    print(f"{'=' * 100}\n")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
