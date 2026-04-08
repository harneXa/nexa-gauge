"""Adversarial node: bias + toxicity probes via DeepEval."""

from deepeval.metrics import BiasMetric, ToxicityMetric
from deepeval.test_case import LLMTestCase

from lumiseval_core.constants import METRIC_PASS_THRESHOLD
from lumiseval_core.types import CostEstimate, Item, MetricCategory, MetricResult, RedteamMetrics
from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.base import BaseMetricNode

log = get_node_logger("redteam")


class RedteamNode(BaseMetricNode):
    node_name = "redteam"

    def _run_metric(self, metric, test_case: LLMTestCase, name: str) -> tuple[MetricResult, float]:
        metric.measure(test_case)
        raw = metric.score
        score = round(1.0 - raw, 4) if raw is not None else None
        passed = score is not None and score >= METRIC_PASS_THRESHOLD
        metric_cost = float(getattr(metric, "evaluation_cost", 0.0) or 0.0)
        log.info(f"  {name}_score={score} raw={raw} cost={metric_cost}")
        return (
            MetricResult(
                name=name,
                category=MetricCategory.ANSWER,
                score=score,
                result=None,
                error=None,
            ),
            metric_cost,
        )

    def run(self, item: Item) -> RedteamMetrics:  # type: ignore[override]
        test_case = LLMTestCase(input="", actual_output=item.text)
        bias_result, bias_cost = self._run_metric(
            BiasMetric(model=self.judge_model),
            test_case,
            "bias",
        )
        tox_result, tox_cost = self._run_metric(
            ToxicityMetric(model=self.judge_model),
            test_case,
            "toxicity",
        )
        total_cost = bias_cost + tox_cost
        return RedteamMetrics(
            metrics=[bias_result, tox_result],
            cost=CostEstimate(
                cost=total_cost,
                input_tokens=None,
                output_tokens=None,
            ),
        )

    def estimate(self, input_tokens: float, output_tokens: float) -> CostEstimate:  # type: ignore[override]
        # Redteam runs through DeepEval internals; no reliable token breakdown here.
        return CostEstimate(cost=0.0, input_tokens=None, output_tokens=None)
