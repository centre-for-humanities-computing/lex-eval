"""Async load driver supporting two modes:

* **concurrency** — fire *N* queries with a fixed concurrency cap and wait
  for all to complete.
* **rate** — fire queries at a steady rate (requests/min) for a fixed
  duration, cycling through the query pool as needed.

The driver is target-agnostic: it accepts any async callable with
signature ``(client, query, **kwargs) -> SingleRunResult``.

When ``batch_size > 1``, queries are grouped into chunks and joined
with newlines before being passed to the runner.  The ``total`` and
``results`` list count *chunks* (HTTP requests), not individual queries.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from typing import Any, Callable, Coroutine, Iterator, List

import httpx

from lex_eval.stress_test.results import SingleRunResult

logger = logging.getLogger(__name__)

Runner = Callable[..., Coroutine[Any, Any, SingleRunResult]]


def _batch(queries: List[str], size: int) -> Iterator[str]:
    """Group *queries* into chunks of *size*, joining each chunk with \n."""
    for i in range(0, len(queries), size):
        yield "\n".join(queries[i : i + size])


async def _worker(
    semaphore: asyncio.Semaphore,
    run_one: Runner,
    query: str,
    runner_kwargs: dict[str, Any],
    results: List[SingleRunResult],
    completed: List[int],
    total: int,
) -> None:
    """Run one request, bounded by the semaphore, and collect its result."""
    async with semaphore:
        async with httpx.AsyncClient() as client:
            result = await run_one(client, query, **runner_kwargs)
        results.append(result)
        done = completed[0] + 1
        completed[0] = done
        status = "✓" if result.success else "✗"
        # Show first sub-query only so the output stays readable.
        first = query.split("\n")[0][:80]
        print(f"  [{done}/{total}] {status} {first}", flush=True)


async def drive_load(
    run_one: Runner,
    queries: List[str],
    *,
    concurrency: int = 1,
    batch_size: int = 1,
    request_timeout: float = 300.0,
    runner_kwargs: dict[str, Any] | None = None,
) -> tuple[List[SingleRunResult], float]:
    """Run all *queries* with a fixed concurrency cap.

    Args:
        run_one: Async callable ``(client, query, **kw) -> SingleRunResult``.
        queries: List of query strings.
        concurrency: Maximum simultaneous in-flight requests (chunks).
        batch_size: Group queries into chunks of this size (joined with \n).
        request_timeout: Per-request timeout in seconds.
        runner_kwargs: Extra kwargs forwarded to *run_one*.

    Returns:
        ``(results, wall_clock_s)``.  ``len(results) = ceil(len(queries) / batch_size)``.
    """
    runner_kwargs = dict(runner_kwargs or {})

    chunks = list(_batch(queries, batch_size))
    semaphore = asyncio.Semaphore(concurrency)
    results: List[SingleRunResult] = []
    completed = [0]
    total = len(chunks)

    target_desc = runner_kwargs.get("host", runner_kwargs.get("workflow_id", "?"))
    subq = f" ({batch_size} sub-queries/batch)" if batch_size > 1 else ""
    print(
        f"\n▶ Concurrency={concurrency}, {total} requests{subq}, target={target_desc}"
    )

    t0 = time.monotonic()

    tasks = [
        _worker(semaphore, run_one, chunk, runner_kwargs, results, completed, total)
        for chunk in chunks
    ]
    overall_deadline = request_timeout * total + 60.0
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
    run_one: Runner,
    queries: List[str],
    *,
    rpm: float = 100.0,
    duration_min: float = 10.0,
    batch_size: int = 1,
    request_timeout: float = 300.0,
    runner_kwargs: dict[str, Any] | None = None,
) -> tuple[List[SingleRunResult], float]:
    """Fire requests at a steady *rpm* rate for *duration_min* minutes.

    Queries are drawn round-robin from *queries*, cycling as needed.
    When *batch_size > 1*, each "request" sent to the runner is a
    newline-joined chunk of *batch_size* queries.  The *rpm* rate
    controls how many *chunks* (HTTP requests) are launched per minute.

    Args:
        run_one: Async callable ``(client, query, **kw) -> SingleRunResult``.
        queries: Pool of query strings (cycled individually, batched).
        rpm: Target HTTP *requests* per minute.
        duration_min: How long to keep launching new requests.
        batch_size: Sub-queries per HTTP request (joined with \n).
        request_timeout: Per-request timeout in seconds.
        runner_kwargs: Extra kwargs forwarded to *run_one*.

    Returns:
        ``(results, wall_clock_s)``.
    """
    runner_kwargs = dict(runner_kwargs or {})

    interval_s = 60.0 / rpm
    duration_s = duration_min * 60.0
    query_cycle = itertools.cycle(queries)

    results: List[SingleRunResult] = []
    active_tasks: set[asyncio.Task[None]] = set()
    launched = [0]
    completed = [0]

    target_desc = runner_kwargs.get("host", runner_kwargs.get("workflow_id", "?"))
    subq = f" ({batch_size} sub-queries/request)" if batch_size > 1 else ""
    print(
        f"\n▶ Rate mode: {rpm:.0f} requests/min for {duration_min:.0f} min{subq}, "
        f"target={target_desc}"
    )
    print(f"  Launch interval: {interval_s:.2f}s  (~{rpm / 60:.1f} req/s)")

    t0 = time.monotonic()
    t_deadline = t0 + duration_s

    async def _launch_one(chunk: str) -> None:
        async with httpx.AsyncClient() as client:
            result = await run_one(client, chunk, **runner_kwargs)
        results.append(result)
        done = completed[0] + 1
        completed[0] = done
        status = "✓" if result.success else "✗"
        first = chunk.split("\n")[0][:80]
        print(f"  [{done} done] {status} {first}", flush=True)

    try:
        while time.monotonic() < t_deadline:
            t_next = time.monotonic() + interval_s

            # Draw batch_size queries and join them.
            sub_queries = [next(query_cycle) for _ in range(batch_size)]
            chunk = "\n".join(sub_queries)
            launched[0] += 1

            task = asyncio.create_task(_launch_one(chunk))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)

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

    if active_tasks:
        await asyncio.wait(active_tasks, timeout=request_timeout)

    elapsed = time.monotonic() - t0
    print(
        f"  ✔ Done in {elapsed:.1f}s — "
        f"{sum(1 for r in results if r.success)}/{completed[0]} successful "
        f"(launch window: {t_launch_end - t0:.1f}s)"
    )

    return results, elapsed
