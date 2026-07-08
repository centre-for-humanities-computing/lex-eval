"""Per-target runner modules.

Each module exposes an async ``run_one`` callable with signature::

    async def run_one(
        client: httpx.AsyncClient,
        query: str,
        **target_kwargs,
    ) -> SingleRunResult:
        ...
"""

from __future__ import annotations
