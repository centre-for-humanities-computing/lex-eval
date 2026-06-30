#!/usr/bin/env python3
"""Stress-test CLI for the lex-llm inference server.

Concurrency mode (fixed concurrency caps)::

    uv run python -m lex_eval.stress_test.run_stress_test \\
        --mode concurrency \\
        --workflow-id beta_workflow_v4_local \\
        --concurrency 1 2 4 8 \\
        --n-queries 50 \\
        --output-dir results/

Rate mode (sustained requests/min for a duration)::

    uv run python -m lex_eval.stress_test.run_stress_test \\
        --mode rate \\
        --workflow-id beta_workflow_v4_local \\
        --rpm 100 \\
        --duration-min 10 \\
        --output-dir results/
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from lex_eval.stress_test.load_driver import drive_load, drive_load_rate
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
        description="Stress-test lex-llm workflow under load.",
    )
    p.add_argument(
        "--workflow-id",
        default="chat_v1_gemma4_26b",
        help="Workflow slug to test.",
    )
    p.add_argument(
        "--mode",
        choices=["concurrency", "rate"],
        default="concurrency",
        help="Load pattern: fixed concurrency or sustained request rate "
        "(default: concurrency).",
    )

    # -- concurrency-mode args
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

    # -- rate-mode args
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

    # -- shared args
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory for JSON output files (default: results/).",
    )
    p.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"lex-llm service base URL (default: {DEFAULT_HOST}).",
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
        default=300.0,
        help="Per-request timeout in seconds (default: 300).",
    )
    p.add_argument(
        "--token",
        default=os.getenv("LEX_LLM_TOKEN", "") or None,
        help="X-Auth-Token value (default: $LEX_LLM_TOKEN env var).",
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

    if args.mode == "concurrency":
        return await _run_concurrency_mode(args)
    else:
        return await _run_rate_mode(args)


async def _run_concurrency_mode(args: argparse.Namespace) -> int:
    """Fixed-concurrency stress test — one run per concurrency level."""
    print("⏳ Mining queries from sample file …", flush=True)
    queries = sample_queries(
        args.sample_file,
        n=args.n_queries * max(args.concurrency),
        seed=42,
    )
    if len(queries) < args.n_queries:
        print(
            f"⚠ Only {len(queries)} queries available "
            f"(wanted {args.n_queries}). "
            f"Running with {len(queries)}.",
            file=sys.stderr,
        )
        effective_n = len(queries)
    else:
        effective_n = args.n_queries
        queries = queries[: args.n_queries * max(args.concurrency)]

    summaries: list[StressTestSummary] = []

    for concurrency in args.concurrency:
        batch = queries[:effective_n]

        results, wall_clock_s = await drive_load(
            base_url=args.host,
            workflow_id=args.workflow_id,
            queries=batch,
            concurrency=concurrency,
            token=args.token,
            request_timeout=args.timeout,
        )

        summary = aggregate(
            results,
            args.workflow_id,
            wall_clock_s,
            concurrency=concurrency,
        )
        summaries.append(summary)

        saved_path = save_run_results(results, summary, args.output_dir)
        print_summary(summary)
        print(f"  📁 Results saved to {saved_path}\n")

    if len(summaries) > 1:
        print_combined_table(summaries)

    return 0


async def _run_rate_mode(args: argparse.Namespace) -> int:
    """Sustained-rate stress test."""
    # Rate mode needs enough queries to cycle through without obvious
    # repetition.  Scan more conversations but keep a reasonable cap.
    pool_size = max(200, int(args.rpm * args.duration_min / 2))
    print(
        f"⏳ Mining ~{pool_size} queries from sample file …",
        flush=True,
    )
    queries = sample_queries(
        args.sample_file,
        n=pool_size,
        seed=42,
        max_scanned=5000,
    )
    print(f"  Collected {len(queries)} queries")

    results, wall_clock_s = await drive_load_rate(
        base_url=args.host,
        workflow_id=args.workflow_id,
        queries=queries,
        rpm=args.rpm,
        duration_min=args.duration_min,
        token=args.token,
        request_timeout=args.timeout,
    )

    summary = aggregate(
        results,
        args.workflow_id,
        wall_clock_s,
        target_rpm=args.rpm,
        duration_min=args.duration_min,
    )

    saved_path = save_run_results(results, summary, args.output_dir)
    print_summary(summary)
    print(f"  📁 Results saved to {saved_path}\n")

    return 0


def print_combined_table(summaries: list[StressTestSummary]) -> None:
    """Print a side-by-side comparison of multiple concurrency runs."""
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
