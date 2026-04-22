from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import BaseModel, Field


class CachedNodeRunResult(BaseModel):
    """Outcome of running a single target node (and its prerequisites) for one case.

    Produced by :meth:`CachedNodeRunner.run_case` and consumed by the CLI
    (`nexagauge run …`) plus any callable that wraps the runner — notably the
    adapter in ``nexagauge`` that writes per-case JSON into ``--output-dir``.

    Fields:
        node_name: Target node requested by the caller (e.g. ``"geval_steps"``).
        case_id: Stable identifier pulled from the input record.
        executed_nodes: Nodes actually run for this case (cache miss).
        cached_nodes: Nodes served from the node-level cache (cache hit).
        node_output: Patch emitted by the *target* node only — what downstream
            reporting typically cares about when the target isn't ``report``.
        final_state: Merged :class:`EvalCase` state after the full plan ran;
            report aggregation reads from this.
        elapsed_ms: Wall-clock for the full per-case plan.
        node_timings: Per-node wall-clock in milliseconds. Cache hits are
            recorded as ``0.0``. Populated for every node the plan touched
            (executed or cached) so callers can render p50/p95 summaries.
    """

    node_name: str
    case_id: str
    executed_nodes: list[str]
    cached_nodes: list[str]
    node_output: dict[str, Any]
    final_state: dict[str, Any]
    elapsed_ms: float
    node_timings: dict[str, float] = Field(default_factory=dict)


class BatchRunResult(BaseModel):
    """Aggregated result shape for a whole-batch (non-streaming) execution.

    Retained as a convenience container for callers that collect all outcomes
    before returning. The streaming path (:meth:`CachedNodeRunner.run_cases_iter`)
    yields :class:`CaseRunOutcome` instances instead and is preferred for CLI
    progress reporting.

    Fields:
        results: Successful per-case results in submission order.
        failures: ``(case_id, traceback_text)`` pairs for cases that raised.
    """

    results: list[CachedNodeRunResult]
    failures: list[tuple[str, str]]


class CaseRunOutcome(BaseModel):
    """One element yielded by :meth:`CachedNodeRunner.run_cases_iter`.

    Guarantees output in *submission order* even when workers finish out of
    order — the runner buffers completions and releases them contiguously from
    ``emit_index``. Exactly one of ``result`` / ``error`` is set.

    Fields:
        index: Zero-based position within the input iterable.
        case_id: Resolved case identifier (``"unknown-case"`` if missing).
        result: Populated on success.
        error: Formatted ``str(exc) + traceback`` on failure.
    """

    index: int
    case_id: str
    result: CachedNodeRunResult | None = None
    error: str | None = None


@dataclass(frozen=True)
class _RunPlanContext:
    """Per-target execution topology reused across many cases."""

    node_name: str
    plan: tuple[str, ...]
    plan_index: Mapping[str, int]
    direct_prereqs: Mapping[str, tuple[str, ...]]
    dependents: Mapping[str, tuple[str, ...]]
    plan_transitive_prereqs: Mapping[str, tuple[str, ...]]
