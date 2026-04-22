from __future__ import annotations

import time
import traceback
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
from typing import Any, Iterable, Iterator, Mapping

from ng_core.cache import NodeCacheBackend, cache_read_allowed, cache_write_allowed
from ng_core.types import EvalCase

from ng_graph.log import get_node_logger
from ng_graph.registry import NODE_FNS
from ng_graph.topology import DEBUG_SKIP_NODES, METRIC_NODES

from .fingerprints import (
    _cache_key_for_step,
    _cache_namespace_mode,
    _case_id,
    _case_value,
    _compute_case_fingerprint,
    _step_fingerprint_for_node_in_plan,
)
from .plan import build_run_plan_context
from .types import CachedNodeRunResult, CaseRunOutcome, _RunPlanContext


def _debug_log_running(node_name: str, case_id: str) -> None:
    if node_name in DEBUG_SKIP_NODES:
        return
    get_node_logger(node_name).start(f"Running for case={case_id}")


def _debug_log_elapsed(node_name: str, case_id: str, elapsed_ms: float) -> None:
    if node_name in DEBUG_SKIP_NODES:
        return
    get_node_logger(node_name).info(f"case={case_id} elapsed={elapsed_ms:.1f}ms")


def _build_initial_state(
    case: dict[str, Any], *, execution_mode: str, target_node: str
) -> EvalCase:
    """Construct the initial :class:`EvalCase` state from a raw input record."""
    record = {
        "case_id": _case_id(case),
        "generation": _case_value(case, "generation"),
        "question": _case_value(case, "question"),
        "reference": _case_value(case, "reference"),
        "context": _case_value(case, "context") or [],
        "geval": _case_value(case, "geval"),
        "redteam": _case_value(case, "redteam"),
    }
    return EvalCase(
        record=record,
        llm_overrides=_case_value(case, "llm_overrides"),
        target_node=target_node,
        execution_mode=execution_mode,
        estimated_costs={},
        node_model_usage={},
        reference_files=_case_value(case, "reference_files") or [],
    )


def _merge_state_patch(state: dict[str, Any], patch: Mapping[str, Any]) -> None:
    """Apply a node's output patch into ``state`` in place."""
    for key, value in patch.items():
        if key == "estimated_costs" and isinstance(value, Mapping):
            existing = state.get("estimated_costs")
            merged = dict(existing) if isinstance(existing, Mapping) else {}
            merged.update(dict(value))
            state["estimated_costs"] = merged
            continue
        if key == "node_model_usage" and isinstance(value, Mapping):
            existing = state.get("node_model_usage")
            merged = dict(existing) if isinstance(existing, Mapping) else {}
            merged.update(dict(value))
            state["node_model_usage"] = merged
            continue
        state[key] = value


class CachedNodeRunner:
    """Execute a target node (plus its prerequisites + report) with per-node caching."""

    def __init__(self, cache_store: NodeCacheBackend) -> None:
        self._cache = cache_store

    @staticmethod
    def _build_run_plan_context(*, node_name: str) -> _RunPlanContext:
        return build_run_plan_context(node_name=node_name)

    @staticmethod
    def _normalize_max_in_flight(*, max_workers: int, max_in_flight: int | None) -> int:
        workers = max(1, max_workers)
        if max_in_flight is None:
            return max(1, workers * 2)
        return max(workers, max_in_flight)

    def _read_step_cache(self, cache_key: str) -> dict[str, Any] | None:
        entry = self._cache.get_entry_by_key(cache_key)
        if entry is None:
            return None
        return entry["node_output"]

    def _read_step_cache_if_allowed(
        self,
        *,
        cache_key: str,
        node_name: str,
        execution_mode: str,
        force: bool,
    ) -> dict[str, Any] | None:
        if force:
            return None
        if not cache_read_allowed(execution_mode=execution_mode, node_name=node_name):
            return None
        return self._read_step_cache(cache_key)

    def _write_step_cache(
        self,
        *,
        cache_key: str,
        node_name: str,
        output: dict[str, Any],
        execution_mode: str,
        case_fingerprint: str,
    ) -> None:
        self._cache.put_by_key(
            cache_key,
            node_name,
            output,
            metadata={
                "execution_mode": execution_mode,
                "case_fingerprint": case_fingerprint,
                "cache_schema": "v2",
            },
        )

    def run_case(
        self,
        *,
        case: dict[str, Any],
        node_name: str,
        plan_context: _RunPlanContext | None = None,
        force: bool = False,
        execution_mode: str = "run",
        debug: bool = False,
    ) -> CachedNodeRunResult:
        plan_ctx = plan_context or self._build_run_plan_context(node_name=node_name)
        if plan_ctx.node_name != node_name:
            raise ValueError(
                f"Plan context node '{plan_ctx.node_name}' does not match requested '{node_name}'."
            )

        t0 = time.monotonic()
        case_id_for_log = _case_id(case)

        plan = plan_ctx.plan
        plan_index = plan_ctx.plan_index
        direct_prereqs = plan_ctx.direct_prereqs
        dependents = plan_ctx.dependents
        plan_transitive_prereqs = plan_ctx.plan_transitive_prereqs

        state: EvalCase = _build_initial_state(
            case,
            execution_mode=execution_mode,
            target_node=node_name,
        )
        state["__cache_store"] = self._cache

        case_fingerprint = _compute_case_fingerprint(case)
        cache_mode = _cache_namespace_mode(execution_mode=execution_mode)

        step_fingerprints = {
            step: _step_fingerprint_for_node_in_plan(
                case_fingerprint=case_fingerprint,
                node_name=step,
                state=state,
                execution_mode=cache_mode,
                plan_transitive_prereqs=plan_transitive_prereqs,
            )
            for step in plan
        }

        cache_keys = {
            step: _cache_key_for_step(
                case_fingerprint=case_fingerprint,
                node_name=step,
                step_fingerprint=step_fingerprints[step],
                execution_mode=cache_mode,
            )
            for step in plan
        }

        executed: list[str] = []
        cached: list[str] = []
        node_output: dict[str, Any] = {}
        node_timings: dict[str, float] = {}

        remaining_prereqs = {step: len(direct_prereqs[step]) for step in plan}
        ready: set[str] = {step for step, count in remaining_prereqs.items() if count == 0}
        submitted: set[str] = set()
        resolved: set[str] = set()
        completed_outputs: dict[str, dict[str, Any]] = {}
        in_flight: dict[Any, str] = {}
        emit_index = 0

        def _drain_in_order_merges() -> None:
            nonlocal emit_index, node_output
            while emit_index < len(plan):
                step_name = plan[emit_index]
                output = completed_outputs.get(step_name)
                if output is None:
                    break
                _merge_state_patch(state, output)
                if step_name == node_name:
                    node_output = output
                emit_index += 1

        def _resolve_step(
            *,
            step_name: str,
            output: dict[str, Any],
            was_cached: bool,
            elapsed_ms: float,
        ) -> None:
            completed_outputs[step_name] = output
            resolved.add(step_name)

            if was_cached:
                cached.append(step_name)
                node_timings[step_name] = 0.0
            else:
                executed.append(step_name)
                node_timings[step_name] = elapsed_ms
                if debug:
                    _debug_log_elapsed(step_name, case_id_for_log, elapsed_ms)
                if cache_write_allowed(execution_mode=execution_mode, node_name=step_name):
                    self._write_step_cache(
                        cache_key=cache_keys[step_name],
                        node_name=step_name,
                        output=output,
                        execution_mode=execution_mode,
                        case_fingerprint=case_fingerprint,
                    )

            for dependent in dependents.get(step_name, ()):
                remaining_prereqs[dependent] -= 1
                if (
                    remaining_prereqs[dependent] == 0
                    and dependent not in resolved
                    and dependent not in submitted
                ):
                    ready.add(dependent)

        def _timed_step_run(step_name: str, snapshot: dict[str, Any]) -> tuple[dict[str, Any], float]:
            node_t0 = time.monotonic()
            out = NODE_FNS[step_name](snapshot)
            return out, (time.monotonic() - node_t0) * 1000

        case_workers = max(4, len(METRIC_NODES))
        with ThreadPoolExecutor(max_workers=case_workers) as pool:
            while len(resolved) < len(plan):
                _drain_in_order_merges()

                made_progress = False
                ready_now = sorted(
                    (step for step in ready if step not in resolved and step not in submitted),
                    key=lambda step: plan_index[step],
                )
                for step in ready_now:
                    ready.discard(step)
                    cached_output = self._read_step_cache_if_allowed(
                        cache_key=cache_keys[step],
                        node_name=step,
                        execution_mode=execution_mode,
                        force=force,
                    )
                    if cached_output is not None:
                        _resolve_step(
                            step_name=step,
                            output=cached_output,
                            was_cached=True,
                            elapsed_ms=0.0,
                        )
                        made_progress = True
                        continue

                    if debug:
                        _debug_log_running(step, case_id_for_log)
                    future = pool.submit(_timed_step_run, step, deepcopy(state))
                    in_flight[future] = step
                    submitted.add(step)
                    made_progress = True

                if len(resolved) >= len(plan):
                    break

                if made_progress:
                    continue

                if not in_flight:
                    missing = [step for step in plan if step not in resolved]
                    raise RuntimeError(f"Node scheduler deadlocked; unresolved nodes: {missing}")

                done, _ = wait(set(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    step = in_flight.pop(future)
                    output, elapsed_ms = future.result()
                    _resolve_step(
                        step_name=step,
                        output=output,
                        was_cached=False,
                        elapsed_ms=elapsed_ms,
                    )

        _drain_in_order_merges()

        elapsed_ms = (time.monotonic() - t0) * 1000

        return CachedNodeRunResult(
            node_name=node_name,
            case_id=_case_id(case),
            executed_nodes=executed,
            cached_nodes=cached,
            node_output=node_output,
            final_state=dict(state),
            elapsed_ms=elapsed_ms,
            node_timings=node_timings,
        )

    def run_cases_iter(
        self,
        *,
        cases: Iterable[dict[str, Any]],
        node_name: str,
        force: bool = False,
        execution_mode: str = "run",
        max_workers: int = 1,
        max_in_flight: int | None = None,
        continue_on_error: bool = True,
        debug: bool = False,
    ) -> Iterator[CaseRunOutcome]:
        plan_context = self._build_run_plan_context(node_name=node_name)
        workers = max(1, max_workers)
        in_flight_limit = self._normalize_max_in_flight(
            max_workers=workers,
            max_in_flight=max_in_flight,
        )

        if workers == 1:
            for idx, case in enumerate(cases):
                case_id = _case_id(case)
                try:
                    result = self.run_case(
                        case=case,
                        node_name=node_name,
                        plan_context=plan_context,
                        force=force,
                        execution_mode=execution_mode,
                        debug=debug,
                    )
                    yield CaseRunOutcome(index=idx, case_id=result.case_id, result=result)
                except Exception as exc:
                    yield CaseRunOutcome(
                        index=idx,
                        case_id=case_id,
                        error=f"{exc}\n{traceback.format_exc()}",
                    )
                    if not continue_on_error:
                        return
            return

        case_iter = iter(cases)
        submit_index = 0
        emit_index = 0
        stop_submitting = False
        source_exhausted = False
        first_failure_index: int | None = None

        pending: dict[Any, tuple[int, str]] = {}
        buffered_results: dict[int, CachedNodeRunResult] = {}
        buffered_failures: dict[int, tuple[str, str]] = {}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                while (
                    not stop_submitting and not source_exhausted and len(pending) < in_flight_limit
                ):
                    try:
                        case = next(case_iter)
                    except StopIteration:
                        source_exhausted = True
                        break

                    idx = submit_index
                    submit_index += 1
                    case_id = _case_id(case)
                    future = pool.submit(
                        self.run_case,
                        case=case,
                        node_name=node_name,
                        plan_context=plan_context,
                        force=force,
                        execution_mode=execution_mode,
                        debug=debug,
                    )
                    pending[future] = (idx, case_id)

                if not pending:
                    break

                done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    idx, case_id = pending.pop(future)
                    try:
                        buffered_results[idx] = future.result()
                    except Exception as exc:
                        buffered_failures[idx] = (case_id, f"{exc}\n{traceback.format_exc()}")
                        if first_failure_index is None:
                            first_failure_index = idx
                        if not continue_on_error:
                            stop_submitting = True

                while True:
                    if emit_index in buffered_results:
                        result = buffered_results.pop(emit_index)
                        yield CaseRunOutcome(index=emit_index, case_id=result.case_id, result=result)
                        emit_index += 1
                        continue

                    if emit_index in buffered_failures:
                        case_id, error = buffered_failures.pop(emit_index)
                        yield CaseRunOutcome(index=emit_index, case_id=case_id, error=error)
                        emit_index += 1

                        if (
                            not continue_on_error
                            and first_failure_index is not None
                            and emit_index > first_failure_index
                        ):
                            for pending_future in pending:
                                pending_future.cancel()
                            return
                        continue

                    break

            while True:
                if emit_index in buffered_results:
                    result = buffered_results.pop(emit_index)
                    yield CaseRunOutcome(index=emit_index, case_id=result.case_id, result=result)
                    emit_index += 1
                    continue

                if emit_index in buffered_failures:
                    case_id, error = buffered_failures.pop(emit_index)
                    yield CaseRunOutcome(index=emit_index, case_id=case_id, error=error)
                    emit_index += 1

                    if not continue_on_error:
                        return
                    continue

                break
