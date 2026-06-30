"""Results models and aggregation for stress-test runs."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field


class LLMCallRecord(BaseModel):
    """Record of a single LLM call within a workflow step."""

    step_name: str
    backend: str  # "primary" or "backup"
    model: str
    trigger: str  # "ok" or reason for fallback
    duration_ms: float


class RunTimings(BaseModel):
    """Timing breakdown for a single stress-test run."""

    ttft_ms: float = 0.0  # time to first text_chunk
    e2e_ms: float = 0.0  # stream_start → stream_end
    http_status: int = 0
    error: Optional[str] = None


class SingleRunResult(BaseModel):
    """Complete result from one stress-test request."""

    query: str
    conversation_id: str
    run_id: str
    timings: RunTimings
    llm_calls: List[LLMCallRecord] = Field(default_factory=list)
    backend_spilled: bool = False

    @property
    def success(self) -> bool:
        return self.timings.error is None


class StressTestSummary(BaseModel):
    """Aggregate statistics for a stress-test run.

    For *concurrency* mode, ``concurrency`` is set and rate fields are 0.
    For *rate* mode, ``target_rpm`` / ``duration_min`` are set and
    ``concurrency`` is ``None``.
    """

    mode: str  # "concurrency" or "rate"
    workflow_id: str
    total_requests: int
    success_count: int
    error_count: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_mean_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float
    e2e_mean_ms: float
    backend_spillover_pct: float
    throughput_rps: float  # requests completed / total wall-clock seconds
    wall_clock_s: float
    errors: List[str] = Field(default_factory=list)
    # Concurrency-mode fields
    concurrency: int | None = None
    # Rate-mode fields
    target_rpm: float = 0.0
    duration_min: float = 0.0


def _percentile(values: List[float], pct: float) -> float:
    """Compute percentile using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (pct / 100.0)
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_vals):
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[f]


def aggregate(
    results: List[SingleRunResult],
    workflow_id: str,
    wall_clock_s: float,
    *,
    concurrency: int | None = None,
    target_rpm: float = 0.0,
    duration_min: float = 0.0,
) -> StressTestSummary:
    """Aggregate individual run results into a summary.

    Args:
        results: Per-request results from the load driver.
        workflow_id: Workflow that was tested.
        wall_clock_s: Total wall-clock seconds for the run batch.
        concurrency: Concurrency level (set for concurrency mode, None for rate).
        target_rpm: Target requests per minute (rate mode only).
        duration_min: Target duration in minutes (rate mode only).
    """
    successful = [r for r in results if r.success]
    errors = [r.timings.error for r in results if not r.success and r.timings.error]

    ttft_vals = [r.timings.ttft_ms for r in successful if r.timings.ttft_ms > 0]
    e2e_vals = [r.timings.e2e_ms for r in successful if r.timings.e2e_ms > 0]
    spillover_count = sum(1 for r in successful if r.backend_spilled)

    total = len(results)
    mode = "concurrency" if concurrency is not None else "rate"

    return StressTestSummary(
        mode=mode,
        concurrency=concurrency,
        workflow_id=workflow_id,
        total_requests=total,
        success_count=len(successful),
        error_count=total - len(successful),
        ttft_p50_ms=_percentile(ttft_vals, 50),
        ttft_p95_ms=_percentile(ttft_vals, 95),
        ttft_p99_ms=_percentile(ttft_vals, 99),
        ttft_mean_ms=statistics.mean(ttft_vals) if ttft_vals else 0.0,
        e2e_p50_ms=_percentile(e2e_vals, 50),
        e2e_p95_ms=_percentile(e2e_vals, 95),
        e2e_p99_ms=_percentile(e2e_vals, 99),
        e2e_mean_ms=statistics.mean(e2e_vals) if e2e_vals else 0.0,
        backend_spillover_pct=spillover_count / len(successful) * 100
        if successful
        else 0.0,
        throughput_rps=total / wall_clock_s if wall_clock_s > 0 else 0.0,
        wall_clock_s=wall_clock_s,
        errors=[str(e) for e in errors],
        target_rpm=target_rpm,
        duration_min=duration_min,
    )


def save_run_results(
    results: List[SingleRunResult],
    summary: StressTestSummary,
    output_dir: Path,
) -> Path:
    """Save per-request results and summary to JSON files.

    Returns the path to the summary file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    wf = summary.workflow_id.replace("/", "_")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    if summary.mode == "concurrency" and summary.concurrency is not None:
        label = f"c{summary.concurrency}"
    else:
        label = f"r{summary.target_rpm:.0f}"

    detail_path = output_dir / f"{wf}_{ts}_{label}_details.json"
    summary_path = output_dir / f"{wf}_{ts}_{label}_summary.json"

    detail_path.write_text(
        json.dumps([r.model_dump() for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary_path.write_text(
        summary.model_dump_json(indent=2),
        encoding="utf-8",
    )

    return summary_path


def print_summary(summary: StressTestSummary) -> None:
    """Print a human-readable summary table."""
    print(f"\n{'=' * 60}")
    if summary.mode == "concurrency":
        print(f"STRESS-TEST RESULTS — concurrency={summary.concurrency}")
    else:
        print(
            f"STRESS-TEST RESULTS — rate={summary.target_rpm:.0f} req/min, "
            f"duration={summary.duration_min:.0f}min"
        )
    print(f"Workflow: {summary.workflow_id}")
    print(f"{'=' * 60}")
    print(
        f"  Requests:  {summary.total_requests} "
        f"({summary.success_count} ok, {summary.error_count} errors)"
    )
    print(f"  Wall time: {summary.wall_clock_s:.1f}s")
    print(f"  Throughput:{summary.throughput_rps:.2f} req/s")
    print()
    print(
        f"  TTFT (ms): p50={summary.ttft_p50_ms:.0f}  "
        f"p95={summary.ttft_p95_ms:.0f}  "
        f"p99={summary.ttft_p99_ms:.0f}  "
        f"mean={summary.ttft_mean_ms:.0f}"
    )
    print(
        f"  E2E  (ms): p50={summary.e2e_p50_ms:.0f}  "
        f"p95={summary.e2e_p95_ms:.0f}  "
        f"p99={summary.e2e_p99_ms:.0f}  "
        f"mean={summary.e2e_mean_ms:.0f}"
    )
    print(f"  Scaleway spillover: {summary.backend_spillover_pct:.1f}%")
    if summary.errors:
        print(f"\n  Errors ({len(summary.errors)}):")
        for err in summary.errors[:5]:
            print(f"    - {err[:120]}")
        if len(summary.errors) > 5:
            print(f"    ... and {len(summary.errors) - 5} more")
    print(f"{'=' * 60}\n")
