"""Runner for the database batch vector / fulltext search API.

Targets:

    POST /vector-search/indexes/{index_name}/batch
    POST /text-search/indexes/{index_name}/batch

The runner treats the entire batch call as a single "request" — the
query string encodes the multiple sub-queries as newline-separated.
"""

from __future__ import annotations

import time

import httpx

from lex_eval.stress_test.results import RunTimings, SingleRunResult

DB_DEFAULT_INDEX = "lex"


async def run_db_vector(
    client: httpx.AsyncClient,
    query: str,
    *,
    host: str = "http://localhost:8000",
    index_name: str = DB_DEFAULT_INDEX,
    top_k: int = 5,
    token: str | None = None,
    timeout: float = 30.0,
) -> SingleRunResult:
    """Batch vector search.

    *query* is newline-joined sub-queries (pre-batched by the load driver).
    """
    sub_qs = [s for s in query.split("\n") if s.strip()]
    url = f"{host}/vector-search/indexes/{index_name}/batch"
    headers: dict[str, str] = {}
    if token:
        headers["X-Auth-Token"] = token

    timings = RunTimings()
    t_start = time.monotonic()

    try:
        response = await client.post(
            url,
            json={
                "queries": [[s, "query"] for s in sub_qs],
                "top_k": top_k,
            },
            headers=headers,
            timeout=timeout,
        )
        timings.http_status = response.status_code
        elapsed = time.monotonic() - t_start
        timings.e2e_ms = elapsed * 1000
        timings.ttft_ms = elapsed * 1000  # non-streaming: e2e IS ttft

        if response.status_code != 200:
            body = await response.aread()
            timings.error = (
                f"HTTP {response.status_code}: {body.decode(errors='replace')[:300]}"
            )
    except httpx.TimeoutException:
        timings.error = "Request timed out"
        timings.e2e_ms = (time.monotonic() - t_start) * 1000
    except httpx.RequestError as exc:
        timings.error = f"Request error: {exc!s}"[:300]
        timings.e2e_ms = (time.monotonic() - t_start) * 1000

    return SingleRunResult(
        query=query,
        conversation_id="",
        run_id="",
        timings=timings,
    )


async def run_db_text(
    client: httpx.AsyncClient,
    query: str,
    *,
    host: str = "http://localhost:8000",
    index_name: str = DB_DEFAULT_INDEX,
    top_k: int = 50,
    token: str | None = None,
    timeout: float = 30.0,
) -> SingleRunResult:
    """Batch fulltext search.

    *query* is newline-joined sub-queries (pre-batched by the load driver).
    """
    sub_qs = [s for s in query.split("\n") if s.strip()]

    url = f"{host}/text-search/indexes/{index_name}/batch"
    headers: dict[str, str] = {}
    if token:
        headers["X-Auth-Token"] = token

    timings = RunTimings()
    t_start = time.monotonic()

    try:
        response = await client.post(
            url,
            json={
                "queries": sub_qs,
                "top_k": top_k,
            },
            headers=headers,
            timeout=timeout,
        )
        timings.http_status = response.status_code
        elapsed = time.monotonic() - t_start
        timings.e2e_ms = elapsed * 1000
        timings.ttft_ms = elapsed * 1000

        if response.status_code != 200:
            body = await response.aread()
            timings.error = (
                f"HTTP {response.status_code}: {body.decode(errors='replace')[:300]}"
            )
    except httpx.TimeoutException:
        timings.error = "Request timed out"
        timings.e2e_ms = (time.monotonic() - t_start) * 1000
    except httpx.RequestError as exc:
        timings.error = f"Request error: {exc!s}"[:300]
        timings.e2e_ms = (time.monotonic() - t_start) * 1000

    return SingleRunResult(
        query=query,
        conversation_id="",
        run_id="",
        timings=timings,
    )
