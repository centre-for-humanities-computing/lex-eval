"""Async streaming NDJSON parser for lex-llm workflow events.

Unlike the existing ``LexLLMConnector`` which buffers the entire response
body before parsing, this module processes NDJSON lines as they arrive,
recording precise timestamps for TTFT / E2E metrics.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import List, Optional

import httpx

from lex_eval.stress_test.results import (
    LLMCallRecord,
    RunTimings,
    SingleRunResult,
    WorkflowStepRecord,
)

logger = logging.getLogger(__name__)

DEFAULT_AUTH_HEADER = "X-Auth-Token"


def _env_token() -> Optional[str]:
    return os.getenv("LEX_LLM_TOKEN")


async def run_single_stress(
    client: httpx.AsyncClient,
    base_url: str,
    workflow_id: str,
    user_input: str,
    *,
    conversation_id: str,
    token: Optional[str] = None,
    timeout: float = 300.0,
) -> SingleRunResult:
    """Execute one workflow request with streaming event capture.

    Args:
        client: Shared ``httpx.AsyncClient`` (reuse across requests).
        base_url: lex-llm service base URL, e.g. ``http://localhost:8011``.
        workflow_id: Workflow slug.
        user_input: The user's query text.
        conversation_id: Unique conversation id for this request.
        token: ``X-Auth-Token`` value; read from ``LEX_LLM_TOKEN`` env var
            if omitted.
        timeout: Per-request timeout in seconds.

    Returns:
        ``SingleRunResult`` with timings and LLM call records.
    """
    token = token or _env_token()

    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if token:
        headers[DEFAULT_AUTH_HEADER] = token

    payload = {
        "user_input": user_input,
        "conversation_history": [],
        "conversation_id": conversation_id,
    }

    url = f"{base_url}/workflows/{workflow_id}/run"

    timings = RunTimings()
    llm_calls: List[LLMCallRecord] = []
    steps: List[WorkflowStepRecord] = []
    run_id = ""
    conv_id = conversation_id
    backend_spilled = False

    t_start = time.monotonic()
    t_first_chunk: Optional[float] = None
    t_end: Optional[float] = None

    try:
        # Use an explicit httpx.Timeout so the *read* timeout is distinct
        # from connect/write/pool.  This ensures a stalled stream (server
        # stops sending NDJSON lines) is killed after `timeout` seconds
        # rather than waiting indefinitely.
        http_timeout = httpx.Timeout(
            connect=10.0,
            read=timeout,
            write=30.0,
            pool=10.0,
        )
        async with client.stream(
            "POST", url, json=payload, headers=headers, timeout=http_timeout
        ) as response:
            timings.http_status = response.status_code

            if response.status_code != 200:
                # Try to read error body
                body = await response.aread()
                timings.error = (
                    f"HTTP {response.status_code}: "
                    f"{body.decode(errors='replace')[:300]}"
                )
                return SingleRunResult(
                    query=user_input,
                    conversation_id=conv_id,
                    run_id="",
                    timings=timings,
                    llm_calls=[],
                    steps=[],
                    backend_spilled=False,
                )

            # The response is NDJSON — read lines as they arrive.
            async for line_bytes in response.aiter_lines():
                line = line_bytes.strip()
                if not line:
                    continue

                event = json.loads(line)
                event_type = event.get("event")

                if event_type == "stream_start":
                    conv_id = event.get("conversation_id", conv_id)
                    run_id = event.get("run_id", "")

                elif event_type == "text_chunk":
                    if t_first_chunk is None:
                        t_first_chunk = time.monotonic()

                elif event_type == "workflow_step":
                    data = event.get("data", {})
                    if data.get("status") == "completed":
                        output = data.get("output", {})
                        step_duration_ms = output.get("duration_ms", 0)
                        step_name = data.get("name", "unknown")

                        # Record step timing (for profiling).
                        steps.append(
                            WorkflowStepRecord(
                                name=step_name, duration_ms=step_duration_ms
                            )
                        )

                        # Capture LLM call details.
                        step_llm_calls = output.get("llm_calls", [])
                        for call in step_llm_calls:
                            rec = LLMCallRecord(
                                step_name=step_name,
                                backend=call.get("backend", "unknown"),
                                model=call.get("model", "unknown"),
                                trigger=call.get("trigger", "unknown"),
                                duration_ms=step_duration_ms,
                            )
                            llm_calls.append(rec)
                            if rec.backend != "primary":
                                backend_spilled = True

                elif event_type == "stream_end":
                    t_end = time.monotonic()

                elif event_type == "workflow_metrics":
                    # Fallback: extract e2e_ms from explicit metrics event
                    pass

            t_end = t_end or time.monotonic()

    except httpx.TimeoutException:
        timings.error = "Request timed out"
    except httpx.RequestError as exc:
        timings.error = f"Request error: {exc!s}"[:300]
    except Exception:
        logger.exception("Unexpected error during stress request")
        timings.error = "Unexpected error (see log)"

    # Compute derived timings
    if t_first_chunk is not None:
        timings.ttft_ms = (t_first_chunk - t_start) * 1000
    timings.e2e_ms = ((t_end or time.monotonic()) - t_start) * 1000

    return SingleRunResult(
        query=user_input,
        conversation_id=conv_id,
        run_id=run_id,
        timings=timings,
        llm_calls=llm_calls,
        steps=steps,
        backend_spilled=backend_spilled,
    )
