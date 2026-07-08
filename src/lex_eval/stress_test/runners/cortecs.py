"""Runner for the Cortecs OpenAI-compatible API.

Target: POST /v1/chat/completions (SSE streaming).
Captures TTFT, token count, and per-token TPS.
"""

from __future__ import annotations

import json
import os
import time

import httpx

from lex_eval.stress_test.results import RunTimings, SingleRunResult


async def run_cortecs(
    client: httpx.AsyncClient,
    query: str,
    *,
    host: str,
    model: str = "gemma-4-31b-it",
    api_key: str | None = None,
    max_tokens: int = 1024,
    timeout: float = 300.0,
) -> SingleRunResult:
    """Run one chat completion against Cortecs with SSE streaming.

    Args:
        client: ``httpx.AsyncClient`` (per-request).
        query: User message content.
        host: Cortecs base URL, e.g. ``https://api.cortecs.ai``.
        model: Model name.
        api_key: ``Authorization: Bearer`` token (reads ``CORTECS_API_KEY``
            env var if not provided).
        max_tokens: Max tokens to generate.
        timeout: Read timeout in seconds.
    """
    api_key = api_key or os.getenv("CORTECS_API_KEY")

    url = f"{host}/v1/chat/completions"
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    timings = RunTimings()
    t_start = time.monotonic()
    t_first_chunk: float | None = None
    t_end: float | None = None
    token_count = 0

    try:
        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(connect=10.0, read=timeout, write=30.0, pool=10.0),
        ) as response:
            timings.http_status = response.status_code

            if response.status_code != 200:
                body = await response.aread()
                timings.error = (
                    f"HTTP {response.status_code}: "
                    f"{body.decode(errors='replace')[:300]}"
                )
                timings.e2e_ms = (time.monotonic() - t_start) * 1000
                return SingleRunResult(
                    query=query,
                    conversation_id="",
                    run_id="",
                    timings=timings,
                )

            async for line in response.aiter_lines():
                if not line:
                    continue
                # SSE format: "data: {...}"
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: ") :]
                if data_str == "[DONE]":
                    t_end = time.monotonic()
                    break

                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    if "content" in delta:
                        if t_first_chunk is None:
                            t_first_chunk = time.monotonic()
                        token_count += 1
                    # Some APIs send usage in the final chunk.
                    usage = chunk.get("usage", {})
                    if usage:
                        token_count = usage.get("completion_tokens", token_count)

    except httpx.TimeoutException:
        timings.error = "Request timed out"
    except httpx.RequestError as exc:
        timings.error = f"Request error: {exc!s}"[:300]
    except Exception:
        timings.error = "Unexpected error"

    t_end = t_end or time.monotonic()
    timings.e2e_ms = (t_end - t_start) * 1000
    if t_first_chunk is not None:
        timings.ttft_ms = (t_first_chunk - t_start) * 1000
    timings.tokens_generated = token_count
    gen_time_ms = timings.e2e_ms - timings.ttft_ms
    if gen_time_ms > 0 and token_count > 0:
        timings.tps = token_count / (gen_time_ms / 1000)

    return SingleRunResult(
        query=query,
        conversation_id="",
        run_id="",
        timings=timings,
    )
