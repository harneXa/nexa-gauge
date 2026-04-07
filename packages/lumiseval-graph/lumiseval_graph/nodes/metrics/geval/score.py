"""GEval scoring node.

Consumes resolved GEval metrics from `EvalCase.node_geval_steps`.
"""

from __future__ import annotations

import asyncio
from enum import Enum
from typing import Optional

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from pydantic import BaseModel

from lumiseval_core.constants import METRIC_PASS_THRESHOLD
from lumiseval_core.geval_cache import (
    GEVAL_STEPS_PARSER_VERSION,
    GEVAL_STEPS_PROMPT_VERSION,
)
from lumiseval_core.types import EvalCase, GevalMetricResolved, GevalScorePayload

from lumiseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.base import BaseMetricNode

log = get_node_logger("geval")


class MetricCategory(str, Enum):
    ANSWER = "answer"


class MetricResult(BaseModel):
    name: str
    category: str = MetricCategory.ANSWER.value
    score: float | None = None
    passed: bool | None = None
    reasoning: str | None = None
    error: str | None = None


_FIELD_TO_PARAM = {
    "question": LLMTestCaseParams.INPUT,
    "generation": LLMTestCaseParams.ACTUAL_OUTPUT,
    "reference": LLMTestCaseParams.EXPECTED_OUTPUT,
    "context": LLMTestCaseParams.CONTEXT,
}


class GevalNode(BaseMetricNode):
    """Evaluate generation quality against resolved GEval metrics."""

    node_name = "geval"
    prompt_version = GEVAL_STEPS_PROMPT_VERSION
    parser_version = GEVAL_STEPS_PARSER_VERSION

    @staticmethod
    def _missing_required_fields(
        *,
        record_fields: list[str],
        generation: str,
        question: Optional[str],
        reference: Optional[str],
        context: Optional[list[str]],
    ) -> list[str]:
        missing: list[str] = []
        for field_name in record_fields:
            if field_name == "generation" and not (generation and generation.strip()):
                missing.append("generation")
            if field_name == "question" and not (question and question.strip()):
                missing.append("question")
            if field_name == "reference" and not (reference and reference.strip()):
                missing.append("reference")
            if field_name == "context" and not any(
                isinstance(item, str) and item.strip() for item in (context or [])
            ):
                missing.append("context")
        return missing

    async def _evaluate_metric(
        self,
        *,
        metric: GevalMetricResolved,
        generation: str,
        question: Optional[str],
        reference: Optional[str],
        context: Optional[list[str]],
    ) -> MetricResult:
        """Run one GEval scoring call for a single resolved metric."""
        test_case = LLMTestCase(
            input=question or "",
            actual_output=generation,
            expected_output=reference or "",
            context=context or [],
        )
        g_eval = GEval(
            name=metric.name,
            criteria=f"Evaluate this response for '{metric.name}' using the provided evaluation steps.",
            evaluation_steps=metric.effective_steps,
            evaluation_params=[_FIELD_TO_PARAM[field_name] for field_name in metric.record_fields],
            model=self.judge_model,
        )
        await asyncio.get_event_loop().run_in_executor(None, g_eval.measure, test_case)

        score = g_eval.score or 0.0
        return MetricResult(
            name=metric.name,
            score=score,
            passed=score >= METRIC_PASS_THRESHOLD,
            reasoning=g_eval.reason or "",
        )

    def run(self, case: EvalCase) -> GevalScorePayload:  # type: ignore[override]
        """Score resolved GEval metrics from case payload only."""
        resolved_metrics = case.node_geval_steps.metrics if case.node_geval_steps else []
        if not resolved_metrics:
            return GevalScorePayload(metrics=[], cost=None)

        generation = case.input_payload.generation.text if case.input_payload.generation else ""
        question = case.input_payload.question.text if case.input_payload.question else None
        reference = case.input_payload.reference.text if case.input_payload.reference else None
        context = case.input_payload.context.text if case.input_payload.context else None

        resolved_by_index: dict[int, GevalMetricResolved] = {
            idx: metric for idx, metric in enumerate(resolved_metrics)
        }
        indexed_results: dict[int, MetricResult] = {}

        async def _run_all() -> list[MetricResult]:
            task_indices: list[int] = []
            tasks: list[asyncio.Future] = []

            for idx, metric in resolved_by_index.items():
                missing_fields = self._missing_required_fields(
                    record_fields=list(metric.record_fields),
                    generation=generation,
                    question=question,
                    reference=reference,
                    context=context,
                )
                if missing_fields:
                    indexed_results[idx] = MetricResult(
                        name=metric.name,
                        error=(
                            "Skipped GEval metric due to missing required record fields: "
                            f"{', '.join(sorted(missing_fields))}."
                        ),
                    )
                    continue

                task_indices.append(idx)
                tasks.append(
                    asyncio.create_task(
                        self._evaluate_metric(
                            metric=metric,
                            generation=generation,
                            question=question,
                            reference=reference,
                            context=context,
                        )
                    )
                )

            if tasks:
                evaluated = await asyncio.gather(*tasks)
                for idx, result in zip(task_indices, evaluated):
                    indexed_results[idx] = result

            if not indexed_results:
                return []
            max_index = max(indexed_results.keys())
            return [indexed_results[i] for i in range(max_index + 1) if i in indexed_results]

        results = asyncio.run(_run_all())
        log.info(f"GEval evaluated metrics={len(results)}")
        return GevalScorePayload(metrics=results, cost=self.cost_estimate(
            input_tokens=0.0,
            output_tokens=0.0,
        ))

    def cost_estimate(
        self,
        input_tokens
        output_tokens
    ):
        """Estimate GEval scoring cost (one call per record per GEval metric)."""
        per_call_cost = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens, pricing, "output"
        )
        return EstimatePayload(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=per_call_cost
       )

    # @staticmethod
    # def cost_formula(cost_meta) -> str:
    #     """Human-readable formula for GEval scoring cost."""
    #     eligible_records = int(getattr(cost_meta, "eligible_records", 0) or 0)
    #     rule_count = int(getattr(cost_meta, "rule_count", 0) or 0)
    #     calls = eligible_records * rule_count
    #     input_tokens = round(float(getattr(cost_meta, "avg_input_tokens", 0.0) or 0.0))
    #     output_tokens = round(float(getattr(cost_meta, "avg_output_tokens", 0.0) or 0.0))
    #     total_tokens = calls * (input_tokens + output_tokens)
    #     return (
    #         f"calls         = {calls}  ({eligible_records} recs × {rule_count} GEval metrics)\n"
    #         f"input_tokens  = {input_tokens} (GEval scoring overhead) tok/call\n"
    #         f"output_tokens = {output_tokens} (GEval scoring output) tok/call\n"
    #         f"total_tokens  = {calls} × ({input_tokens} + {output_tokens}) = {total_tokens} tok"
    #     )
