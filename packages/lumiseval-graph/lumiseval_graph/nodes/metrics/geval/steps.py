"""GEval steps node.

Consumes `EvalCase` and resolves mixed GEval metric inputs into canonical
`effective_steps` consumed by GEval scoring.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from lumiseval_core.geval_cache import (
    GEVAL_STEPS_PARSER_VERSION,
    GEVAL_STEPS_PROMPT_VERSION,
    GevalArtifactCache,
    compute_geval_signature,
)
from lumiseval_core.types import EvalCase, GevalMetricInput, GevalMetricResolved, GevalStepsPayload

from lumiseval_graph.llm.gateway import get_llm
from lumiseval_graph.llm.pricing import cost_usd, get_model_pricing
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.base import BaseMetricNode
from lumiseval_core.utils import _count_tokens, template_static_tokens

log = get_node_logger("geval_steps")


class _GevalStepsResponse(BaseModel):
    evaluation_steps: list[str]


class GevalStepsNode(BaseMetricNode):
    """Resolve each metric to deterministic `effective_steps`."""

    node_name = "geval_steps"
    prompt_version = GEVAL_STEPS_PROMPT_VERSION
    parser_version = GEVAL_STEPS_PARSER_VERSION

    SYSTEM_PROMPT = "You are an expert evaluator that writes concrete evaluation steps."
    USER_PROMPT = (
        "Evaluation criteria:\n"
        "{criteria}\n\n"
        "Return 2-3 concrete, measurable evaluation steps as JSON:\n"
        '{"evaluation_steps": ["step 1", "step 2", "..."]}\n'
        "Each step must be specific, testable, and focused on the criterion above."
    )
    static_prompt_tokens: int = _count_tokens(SYSTEM_PROMPT) + template_static_tokens(USER_PROMPT)

    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        artifact_cache: Optional[GevalArtifactCache] = None,
    ) -> None:
        super().__init__(judge_model=judge_model)
        self._artifact_cache = artifact_cache or GevalArtifactCache()

    @staticmethod
    def _metric_key_sequence(metrics: list[GevalMetricInput]) -> list[str]:
        """Return deterministic keys; duplicate names become name#2, name#3."""
        name_counts: dict[str, int] = {}
        for metric in metrics:
            name_counts[metric.name] = name_counts.get(metric.name, 0) + 1

        seen: dict[str, int] = {}
        keys: list[str] = []
        for metric in metrics:
            name = metric.name
            if name_counts[name] == 1:
                keys.append(name)
                continue
            seen[name] = seen.get(name, 0) + 1
            keys.append(f"{name}#{seen[name]}")
        return keys

    def _signature(self, criteria: str) -> str:
        return str(
            compute_geval_signature(
                criteria=criteria,
                model=self.judge_model,
                prompt_version=self.prompt_version,
                parser_version=self.parser_version,
            )
        )

    def _generate_steps(self, criteria: str, metric_name: str) -> list[str]:
        llm = get_llm("geval_steps", _GevalStepsResponse, self.judge_model)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self.USER_PROMPT.format(criteria=criteria)},
        ]
        response = llm.invoke(messages)
        parsed: _GevalStepsResponse | None = response["parsed"]
        steps = [s.strip() for s in (parsed.evaluation_steps if parsed else []) if s and s.strip()]
        if not steps:
            raise RuntimeError(f"GEval steps generation failed for metric '{metric_name}'.")
        return steps

    # @classmethod
    # def resolve_cost_kwargs(
    #     cls,
    #     *,
    #     cases: Optional[list] = None,
    #     model: Optional[str] = None,
    #     **_ignored,
    # ) -> dict[str, Optional[int]]:
    #     # Kept lightweight under new schema; precise cache-aware counting can be
    #     # reintroduced once run-state record adapters are stabilized.
    #     del cases, model
    #     return {"uncached_unique_rules": None}

    def run(self, case: EvalCase) -> GevalResolvedPayload:  # type: ignore[override]
        """Resolve metrics from `case.input_payload.geval` into canonical metrics."""
        metrics = case.input_payload.geval.metrics if case.input_payload.geval else []
        if not metrics:
            return GevalResolvedPayload(metrics=[], cost=None)

        resolved: list[GevalStepsResolved] = []
        input_tokens = 0
        for key, metric in zip(self._metric_key_sequence(metrics), metrics):

            # When for a metrics Steps are provided, we use them directly
            if metric.evaluation_steps:
                resolved.append(
                    GevalStepsResolved(
                        key=key,
                        name=metric.name,
                        record_fields=list(metric.record_fields),
                        evaluation_steps=metric.evaluation_steps,
                        steps_source="provided",
                        signature=None,
                    )
                )
                continue

            # When Criteria is not Provided we continue
            criteria_text = (metric.criteria.text or "").strip()
            criteria_tokens = metric.criteria.tokens
            if not criteria_text:
                # Validator should prevent this, but keep a defensive fallback.
                log.warning(f"Skipping GEval metric with no criteria/steps: {metric.name}")
                continue

            # ------------------------------------------------
            # When Steps are not provided but Criteria is.
            # ------------------------------------------------
            signature = self._signature(criteria_text)
            cached_steps = self._artifact_cache.get_steps(signature)

            # When the Criteris is not Cached we generate the steps
            if cached_steps is None:
                generated_steps = [
                    GevalEvaluationStep(text=step, tokens=_count_tokens(step)) 
                    self._generate_steps(criteria_text, metric.name)
                ]
                self._artifact_cache.put_steps(
                    signature=signature,
                    model=self.judge_model,
                    criteria=criteria,
                    evaluation_steps=generated_steps,
                    prompt_version=self.prompt_version,
                    parser_version=self.parser_version,
                )
                input_tokens += criteria_tokens
                output_tokens += sum([_count_tokens(step.tokens) for step in generated_steps])
                effective_steps = generated_steps
                log.info(f"generated GEval steps for metric={metric.name} signature={signature}")
            else:
                # When the Criteris is Cached we use the cached steps
                effective_steps = cached_steps

            resolved.append(
                GevalStepsResolved(
                    key=key,
                    name=metric.name,
                    record_fields=list(metric.record_fields),
                    evaluation_steps=list(effective_steps),
                    steps_source="generated",
                    signature=signature,
                )
            )

        return GevalStepsPayload(
            metrics=resolved, 
            cost=self.cost_estimate(
                input_tokens=input_tokens,
                output_tokens=sum([metric.tokens for metric in resolved]),
            )
        )

    def cost_estimate(
        self,
        input_tokens: float,
        output_tokens: float,
    ):
        """Estimate step-generation cost under the current metadata contract."""

        pricing = get_model_pricing(self.judge_model)  
        input_tokens = self.static_prompt_tokens + input_tokens 

        cost_per_record = cost_usd(input_tokens, pricing, "input") + cost_usd(
            output_tokens, pricing, "output"
        )
        return EstimatePayload(
           input_tokens=input_tokens,
           output_tokens=output_tokens,
           cost=cost_per_record
        )
