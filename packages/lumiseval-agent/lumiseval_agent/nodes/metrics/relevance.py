"""
RAGAS Node — computes faithfulness, answer relevancy, and optionally context
precision/recall using a single ragas.evaluate() call.

faithfulness and answer_relevancy measure answer quality (ANSWER category).
context_precision and context_recall measure retrieval quality (RETRIEVAL category).

A single evaluate() call is used for all metrics to avoid redundant LLM calls.

TODO: Wire LiteLLM as the judge LLM inside RAGAS so billing is unified.
"""

from typing import Optional, List

from datasets import Dataset
from lumiseval_core.types import MetricCategory, MetricResult
from ragas import evaluate
from ragas.metrics import AnswerRelevancy, Faithfulness

from lumiseval_agent.log import get_node_logger

log = get_node_logger("relevance")


def run(
    context: List[str],
    claims: List[str],
    generation: str,
    question: Optional[str] = None,
    ground_truth: Optional[str] = None,
    judge_model: str = "gpt-4o-mini",
    enable_faithfulness: bool = True,
    enable_answer_relevancy: bool = True,
) -> list[MetricResult]:
    """Compute RAGAS metrics on the generation + retrieved evidence.

    Args:
        generation: The LLM-generated text to evaluate.
        context: Evidence passages from the Evidence Router.
        question: The original query/question (improves answer relevancy score).
        ground_truth: Optional reference answer (enables context_recall).
        judge_model: LiteLLM model string used as the RAGAS judge LLM.
        enable_faithfulness: Whether to compute faithfulness.
        enable_answer_relevancy: Whether to compute answer relevancy.

    Returns:
        list[MetricResult] — one per computed metric.
        faithfulness / answer_relevancy → ANSWER category.
        context_precision / context_recall → RETRIEVAL category (if ground_truth provided).
    """

    metrics = []
    if enable_faithfulness:
        metrics.append(Faithfulness())
    if enable_answer_relevancy:
        metrics.append(AnswerRelevancy())

    if not metrics:
        return []

    data: dict = {
        "question": [question or ""],
        "answer": [generation],
        "contexts": [context],
    }
    if ground_truth:
        data["ground_truth"] = [ground_truth]
        log.info("ground_truth provided — context_recall enabled")

    dataset = Dataset.from_dict(data)
    result = evaluate(dataset, metrics=metrics)
    scores = result.to_pandas().iloc[0].to_dict()

    log.success(
        f"faithfulness={scores.get('faithfulness')}  "
        f"answer_relevancy={scores.get('answer_relevancy')}"
    )

    results: list[MetricResult] = []
    if enable_faithfulness and scores.get("faithfulness") is not None:
        results.append(
            MetricResult(
                name="faithfulness",
                category=MetricCategory.ANSWER,
                score=scores["faithfulness"],
            )
        )
    if enable_answer_relevancy and scores.get("answer_relevancy") is not None:
        results.append(
            MetricResult(
                name="answer_relevancy",
                category=MetricCategory.ANSWER,
                score=scores["answer_relevancy"],
            )
        )
    if scores.get("context_precision") is not None:
        results.append(
            MetricResult(
                name="context_precision",
                category=MetricCategory.RETRIEVAL,
                score=scores["context_precision"],
            )
        )
    if scores.get("context_recall") is not None:
        results.append(
            MetricResult(
                name="context_recall",
                category=MetricCategory.RETRIEVAL,
                score=scores["context_recall"],
            )
        )

    return results
