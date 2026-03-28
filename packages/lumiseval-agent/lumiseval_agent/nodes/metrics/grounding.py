"""
Hallucination Node — detects hallucinations using DeepEval HallucinationMetric.

Compares the generation against retrieved context passages.
Returns a single MetricResult with a hallucination-free score (1.0 = no hallucination).

score = 1.0 - raw_hallucination_score so that higher is always better.
"""

from deepeval.metrics import HallucinationMetric
from deepeval.test_case import LLMTestCase
from lumiseval_core.constants import METRIC_PASS_THRESHOLD
from lumiseval_core.types import MetricCategory, MetricResult

from lumiseval_agent.log import get_node_logger

log = get_node_logger("grounding")


def run(
    generation: str,
    context: list[str],
    judge_model: str = "gpt-4o-mini",
) -> MetricResult:
    """Measure hallucination in the generation relative to retrieved evidence.

    Args:
        generation: The LLM-generated text to evaluate.
        context: Context retrieval
        judge_model: LiteLLM model string for the judge LLM.

    Returns:
        MetricResult where score=1.0 means no hallucination detected.
    """
    test_case = LLMTestCase(
        input="",
        actual_output=generation,
        context=context,
    )
    metric = HallucinationMetric(model=judge_model)
    metric.measure(test_case)

    raw_score = metric.score or 0.0
    score = round(1.0 - raw_score, 4)

    return MetricResult(
        name="hallucination",
        category=MetricCategory.ANSWER,
        score=score,
        passed=score >= METRIC_PASS_THRESHOLD,
        reasoning=getattr(metric, "reason", None),
    )
