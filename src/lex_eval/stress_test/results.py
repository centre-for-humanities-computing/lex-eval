"""Results models and aggregation for stress-test runs."""

from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
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


class WorkflowStepRecord(BaseModel):
    """Timing for one workflow step (completed)."""

    name: str
    duration_ms: float
    llm_calls: List[LLMCallRecord] = Field(default_factory=list)


class RunTimings(BaseModel):
    """Timing breakdown for a single stress-test run."""

    ttft_ms: float = 0.0  # time to first text_chunk / first byte
    e2e_ms: float = 0.0  # request start → response end
    http_status: int = 0
    error: Optional[str] = None
    tokens_generated: int = 0  # only populated for LLM streaming targets
    tps: float = 0.0  # tokens per second (tokens_generated / (e2e_ms - ttft_ms))


class SingleRunResult(BaseModel):
    """Complete result from one stress-test request."""

    query: str
    conversation_id: str
    run_id: str
    timings: RunTimings
    llm_calls: List[LLMCallRecord] = Field(default_factory=list)
    steps: List[WorkflowStepRecord] = Field(default_factory=list)
    backend_spilled: bool = False

    @property
    def success(self) -> bool:
        return self.timings.error is None


class BackendStats(BaseModel):
    """Aggregate latency per backend + model combination."""

    backend: str
    model: str
    step_name: str
    call_count: int
    p50_ms: float
    p95_ms: float
    mean_ms: float


class ErrorBreakdown(BaseModel):
    """Categorized error counts."""

    timeout: int = 0
    http_502: int = 0
    http_503: int = 0
    http_4xx: int = 0
    http_5xx_other: int = 0
    other: int = 0
    total: int = 0


class StepStats(BaseModel):
    """Aggregate timing for one workflow step across all runs."""

    name: str
    count: int  # how many runs had this step
    p50_ms: float
    p95_ms: float
    mean_ms: float
    total_ms: float  # sum of all durations


class StressTestSummary(BaseModel):
    """Aggregate statistics for a stress-test run.

    For *concurrency* mode, ``concurrency`` is set and rate fields are 0.
    For *rate* mode, ``target_rpm`` / ``duration_min`` are set and
    ``concurrency`` is ``None``.
    """

    mode: str  # "concurrency" or "rate"
    target: str  # "lex-llm", "db", or "cortecs"
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
    tps_p50: float = 0.0
    tps_p95: float = 0.0
    tps_mean: float = 0.0
    tokens_total: int = 0
    backend_spillover_pct: float
    throughput_rps: float  # requests completed / total wall-clock seconds
    wall_clock_s: float
    errors: List[str] = Field(default_factory=list)
    error_breakdown: ErrorBreakdown = Field(default_factory=ErrorBreakdown)
    step_stats: List[StepStats] = Field(default_factory=list)
    backend_stats: List[BackendStats] = Field(default_factory=list)
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
    target: str = "lex-llm",
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
    tps_vals = [r.timings.tps for r in successful if r.timings.tps > 0]
    tokens_total = sum(r.timings.tokens_generated for r in successful)
    spillover_count = sum(1 for r in successful if r.backend_spilled)
    total = len(results)
    mode = "concurrency" if concurrency is not None else "rate"

    # Aggregate per-step durations across all successful runs.
    step_durations: dict[str, list[float]] = defaultdict(list)
    # Aggregate LLM call latencies per (backend, model, step).
    backend_durations: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in successful:
        for step in r.steps:
            step_durations[step.name].append(step.duration_ms)
        for call in r.llm_calls:
            backend_durations[(call.backend, call.model, call.step_name)].append(
                call.duration_ms
            )

    step_stats = []
    for name in sorted(step_durations):
        vals = step_durations[name]
        step_stats.append(
            StepStats(
                name=name,
                count=len(vals),
                p50_ms=_percentile(vals, 50),
                p95_ms=_percentile(vals, 95),
                mean_ms=statistics.mean(vals),
                total_ms=sum(vals),
            )
        )

    backend_stats = []
    for (backend, model, step_name), vals in sorted(backend_durations.items()):
        backend_stats.append(
            BackendStats(
                backend=backend,
                model=model,
                step_name=step_name,
                call_count=len(vals),
                p50_ms=_percentile(vals, 50),
                p95_ms=_percentile(vals, 95),
                mean_ms=statistics.mean(vals),
            )
        )

    # Classify errors.
    error_breakdown = ErrorBreakdown()
    for r in results:
        if r.success:
            continue
        err = (r.timings.error or "").lower()
        if "timed out" in err:
            error_breakdown.timeout += 1
        elif "502" in err:
            error_breakdown.http_502 += 1
        elif "503" in err:
            error_breakdown.http_503 += 1
        elif re.search(r"http 4\d\d", err):
            error_breakdown.http_4xx += 1
        elif re.search(r"http 5\d\d", err):
            error_breakdown.http_5xx_other += 1
        else:
            error_breakdown.other += 1
    error_breakdown.total = sum(
        getattr(error_breakdown, f) for f in error_breakdown.model_fields
    )

    return StressTestSummary(
        mode=mode,
        target=target,
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
        tps_p50=_percentile(tps_vals, 50),
        tps_p95=_percentile(tps_vals, 95),
        tps_mean=statistics.mean(tps_vals) if tps_vals else 0.0,
        tokens_total=tokens_total,
        backend_spillover_pct=spillover_count / len(successful) * 100
        if successful
        else 0.0,
        throughput_rps=total / wall_clock_s if wall_clock_s > 0 else 0.0,
        wall_clock_s=wall_clock_s,
        errors=[str(e) for e in errors],
        target_rpm=target_rpm,
        duration_min=duration_min,
        step_stats=step_stats,
        backend_stats=backend_stats,
        error_breakdown=error_breakdown,
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
    if summary.tokens_total > 0:
        print(
            f"  Tokens:   {summary.tokens_total} total  "
            f"TPS p50={summary.tps_p50:.0f}  p95={summary.tps_p95:.0f}  "
            f"mean={summary.tps_mean:.0f}"
        )
    _print_error_breakdown(summary)
    if summary.backend_stats:
        _print_backend_breakdown(summary)
    if summary.step_stats:
        _print_step_breakdown(summary)
    print(f"{'=' * 60}\n")


def _print_error_breakdown(summary: StressTestSummary) -> None:
    eb = summary.error_breakdown
    if eb.total == 0:
        return
    print(f"\n  ERROR BREAKDOWN ({eb.total} total):")
    parts: list[tuple[str, int]] = [
        ("502 Proxy Error", eb.http_502),
        ("503 Unavailable", eb.http_503),
        ("Timeouts", eb.timeout),
        ("Other 4xx", eb.http_4xx),
        ("Other 5xx", eb.http_5xx_other),
        ("Other", eb.other),
    ]
    for label, count in parts:
        if count:
            print(f"    {label:<20} {count:>4}")


def _print_backend_breakdown(summary: StressTestSummary) -> None:
    """Print per-backend/model latency for profiling the LLM orchestrator."""
    print("\n  BACKEND / MODEL LATENCY (p50 | p95 | mean):")
    for b in sorted(
        summary.backend_stats,
        key=lambda x: (x.step_name, x.backend, x.model),
    ):
        print(
            f"    [{b.step_name:<30}] {b.backend}/{b.model:<25} "
            f"{b.p50_ms:>7.0f}ms {b.p95_ms:>7.0f}ms {b.mean_ms:>7.0f}ms  "
            f"(n={b.call_count})"
        )


def _print_step_breakdown(summary: StressTestSummary) -> None:
    """Print per-step timing breakdown for workflow profiling."""
    print("\n  WORKFLOW STEP BREAKDOWN (p50 | p95 | mean):")
    # Sort by total time descending so the bottleneck appears first.
    for s in sorted(summary.step_stats, key=lambda x: x.total_ms, reverse=True):
        pct = s.total_ms / sum(ss.total_ms for ss in summary.step_stats) * 100
        print(
            f"    {s.name:<40} "
            f"{s.p50_ms:>7.0f}ms {s.p95_ms:>7.0f}ms {s.mean_ms:>7.0f}ms  "
            f"({pct:.0f}% of total step time, n={s.count})"
        )
