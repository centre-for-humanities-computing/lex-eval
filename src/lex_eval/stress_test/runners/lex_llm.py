"""Runner for the lex-llm workflow API.

Target: POST /workflows/{workflow_id}/run (NDJSON streaming).
Uses the existing ``streaming_parser`` for event capture.
"""

from __future__ import annotations

import os
from typing import Any, Protocol

import httpx

from lex_eval.stress_test.results import SingleRunResult


class Runner(Protocol):
    """Protocol for stress-test runner callables."""

    async def __call__(
        self, client: httpx.AsyncClient, query: str, **target_kwargs: Any
    ) -> SingleRunResult: ...


async def run_lex_llm(
    client: httpx.AsyncClient,
    query: str,
    *,
    host: str,
    workflow_id: str,
    token: str | None = None,
    timeout: float = 300.0,
) -> SingleRunResult:
    """Run one lex-llm workflow request.

    Args:
        client: ``httpx.AsyncClient`` (per-request).
        query: The user input text.
        host: lex-llm service base URL.
        workflow_id: Workflow slug.
        token: ``X-Auth-Token`` value.
        timeout: Per-request timeout in seconds.
    """
    from lex_eval.stress_test.streaming_parser import run_single_stress

    token = token or os.getenv("LEX_LLM_TOKEN")
    import uuid

    return await run_single_stress(
        client,
        host,
        workflow_id,
        query,
        conversation_id=str(uuid.uuid4()),
        token=token,
        timeout=timeout,
    )
