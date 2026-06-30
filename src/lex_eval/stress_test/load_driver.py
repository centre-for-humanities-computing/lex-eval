"""Async load driver supporting two modes:

* **concurrency** — fire *N* queries with a fixed concurrency cap and wait
  for all to complete.
* **rate** — fire queries at a steady rate (requests/min) for a fixed
  duration, cycling through the query pool as needed.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
import uuid
from typing import List

import httpx

from lex_eval.stress_test.results import SingleRunResult
from lex_eval.stress_test.streaming_parser import run_single_stress

logger = logging.getLogger(__name__)


async def _worker(
    semaphore: asyncio.Semaphore,
    base_url: str,
    workflow_id: str,
    query: str,
    token: str | None,
    timeout: float,
    results: List[SingleRunResult],
    completed: List[int],
    total: int,
) -> None:
    """Run one request, bounded by the semaphore, and collect its result."""
    async with semaphore:
        conversation_id = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            result = await run_single_stress(
                client,
                base_url,
                workflow_id,
                query,
                conversation_id=conversation_id,
                token=token,
                timeout=timeout,
            )
        results.append(result)
        done = completed[0] + 1
        completed[0] = done
        status = "✓" if result.success else "✗"
        print(f"  [{done}/{total}] {status} {query[:80]}", flush=True)


async def drive_load(
    base_url: str,
    workflow_id: str,
    queries: List[str],
    *,
    concurrency: int = 1,
    token: str | None = None,
    request_timeout: float = 300.0,
) -> tuple[List[SingleRunResult], float]:
    """Run all *queries* against *workflow_id* with a fixed concurrency cap.

    Args:
        base_url: lex-llm service base URL.
        workflow_id: Workflow to test.
        queries: List of query strings.
        concurrency: Maximum simultaneous in-flight requests.
        token: ``X-Auth-Token`` value.
        request_timeout: Per-request timeout in seconds.

    Returns:
        ``(results, wall_clock_s)``.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: List[SingleRunResult] = []
    completed = [0]  # mutable counter shared across workers
    total = len(queries)

    print(f"\n▶ Concurrency={concurrency}, {total} queries, workflow={workflow_id}")
    print(f"  Target: {base_url}")

    t0 = time.monotonic()

    tasks = [
        _worker(
            semaphore,
            base_url,
            workflow_id,
            query,
            token,
            request_timeout,
            results,
            completed,
            total,
        )
        for query in queries
    ]
    # Add a hard deadline: total wall-clock = per-request timeout + 10 %
    # buffer so we never wait longer than the worst-case serial execution.
    overall_deadline = request_timeout * len(queries) + 60.0
    done, pending = await asyncio.wait(
        [asyncio.ensure_future(t) for t in tasks],
        timeout=overall_deadline,
    )
    if pending:
        print(
            f"  ⚠ {len(pending)} request(s) did not complete within "
            f"{overall_deadline:.0f}s — cancelling.",
            flush=True,
        )
        for task in pending:
            task.cancel()

    elapsed = time.monotonic() - t0
    print(
        f"  ✔ Done in {elapsed:.1f}s — "
        f"{sum(1 for r in results if r.success)}/{total} successful"
    )

    return results, elapsed


async def drive_load_rate(
    base_url: str,
    workflow_id: str,
    queries: List[str],
    *,
    rpm: float = 100.0,
    duration_min: float = 10.0,
    token: str | None = None,
    request_timeout: float = 300.0,
) -> tuple[List[SingleRunResult], float]:
    """Fire requests at a steady *rpm* rate for *duration_min* minutes.

    Queries are drawn round-robin from *queries*, cycling as needed.

    Requests are launched on schedule and run to completion in the
    background — the driver does **not** cancel in-flight requests when
    the duration elapses, only stops launching new ones.

    Args:
        base_url: lex-llm service base URL.
        workflow_id: Workflow to test.
        queries: Pool of query strings (cycled through).
        rpm: Target requests per minute.
        duration_min: How long to keep launching new requests.
        token: ``X-Auth-Token`` value.
        request_timeout: Per-request timeout in seconds.

    Returns:
        ``(results, wall_clock_s)``.
    """
    interval_s = 60.0 / rpm  # seconds between launches
    duration_s = duration_min * 60.0
    query_cycle = itertools.cycle(queries)

    results: List[SingleRunResult] = []
    # Track active tasks so we can wait for them to settle after the
    # launch window ends.
    active_tasks: set[asyncio.Task[None]] = set()
    launched = [0]
    completed = [0]

    print(
        f"\n▶ Rate mode: {rpm:.0f} req/min for {duration_min:.0f} min, "
        f"workflow={workflow_id}"
    )
    print(f"  Target: {base_url}")
    print(f"  Launch interval: {interval_s:.2f}s  (~{rpm / 60:.1f} req/s)")

    t0 = time.monotonic()
    t_deadline = t0 + duration_s

    async def _launch_one(query: str) -> None:
        cid = str(uuid.uuid4())
        async with httpx.AsyncClient() as client:
            result = await run_single_stress(
                client,
                base_url,
                workflow_id,
                query,
                conversation_id=cid,
                token=token,
                timeout=request_timeout,
            )
        results.append(result)
        done = completed[0] + 1
        completed[0] = done
        status = "✓" if result.success else "✗"
        print(f"  [{done} done] {status} {query[:80]}", flush=True)

    try:
        while time.monotonic() < t_deadline:
            t_next = time.monotonic() + interval_s
            query = next(query_cycle)
            launched[0] += 1

            task = asyncio.create_task(_launch_one(query))
            active_tasks.add(task)
            # Remove task from set when done (fire-and-forget cleanup).
            task.add_done_callback(active_tasks.discard)

            # Sleep until next launch slot (may be negative if we're behind).
            delay = t_next - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

    except asyncio.CancelledError:
        pass

    t_launch_end = time.monotonic()
    print(
        f"  Launch window ended — {launched[0]} requests fired. "
        f"Waiting for in-flight requests to complete …"
    )

    # Wait for all in-flight requests to finish (or until they time out).
    if active_tasks:
        await asyncio.wait(active_tasks, timeout=request_timeout)

    elapsed = time.monotonic() - t0
    print(
        f"  ✔ Done in {elapsed:.1f}s — "
        f"{sum(1 for r in results if r.success)}/{completed[0]} successful "
        f"(launch window: {t_launch_end - t0:.1f}s)"
    )

    return results, elapsed
