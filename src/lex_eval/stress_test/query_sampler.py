"""Stream-parse the large conversations sample JSON to extract user queries.

The sample file (data/lex-llm-conversations-sample.json) is a single JSON
array; each element has shape:

    {
        "conversation_id": "...",
        "history": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

We use ``ijson`` to avoid loading the entire file into memory.
"""

from __future__ import annotations

import random
import textwrap
from pathlib import Path
from typing import Iterator, List, Optional

import ijson  # type: ignore[import-untyped]


def stream_queries(
    path: Path, *, min_length: int = 3, max_queries: Optional[int] = None
) -> Iterator[str]:
    """Yield the first user-turn from every conversation in the sample file.

    Args:
        path: Path to the NDJSON conversations file.
        min_length: Skip queries shorter than this many characters.
        max_queries: Stop after yielding this many queries (None = unlimited).

    Yields:
        Query strings extracted from the first ``user`` message of each
        conversation.
    """
    yielded = 0
    with path.open("rb") as fh:
        for item in ijson.items(fh, "item"):
            history = item.get("history", [])
            for turn in history:
                if turn.get("role") == "user":
                    content = (turn.get("content") or "").strip()
                    if len(content) >= min_length:
                        # Trim extremely long queries to avoid token waste.
                        content = textwrap.shorten(content, width=600, placeholder="…")
                        yield content
                        yielded += 1
                        if max_queries is not None and yielded >= max_queries:
                            return
                    break  # only first user turn per conversation


def sample_queries(
    path: Path,
    n: int = 100,
    *,
    min_length: int = 3,
    seed: int = 42,
    max_scanned: int = 2000,
) -> List[str]:
    """Return a list of *n* queries sampled from ``path``.

    By default the function scans up to 2000 conversations but only
    selects *n* of them uniformly at random to keep the sample
    representative while bounding runtime.

    Args:
        path: Path to the JSON conversations file.
        n: Desired sample size.
        min_length: Mimimum query length in characters.
        seed: RNG seed for reproducibility.
        max_scanned: Collect this many queries before sampling.

    Returns:
        List of query strings.
    """
    rng = random.Random(seed)
    pool: List[str] = []
    for query in stream_queries(path, min_length=min_length, max_queries=max_scanned):
        pool.append(query)
    if len(pool) <= n:
        return pool
    return rng.sample(pool, n)
