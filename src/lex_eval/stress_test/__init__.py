"""Stress-test framework for lex-llm inference server.

Captures per-request timing (TTFT, E2E), backend spillover tracking,
and aggregates results across concurrency levels.
"""

from __future__ import annotations
