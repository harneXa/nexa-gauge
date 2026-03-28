"""
Eval Node (eval & Scoring) — rolls up all metric outputs into a single EvalReport.

Organises results into two high-level quality dimensions:
  retrieval_score — how well evidence was retrieved
                    (context_precision + context_recall + evidence_support_rate)
  answer_score    — how good the answer is
                    (faithfulness + answer_relevancy + hallucination + rubric + privacy/bias)

composite_score = simple average of the two dimension scores.

Confidence band is the standard deviation of per-claim retrieval scores.
"""

import statistics
from typing import Optional

from lumiseval_core.types import (
    CostEstimate,
    EvalJobConfig,
    EvalReport,
    MetricCategory,
    MetricResult,
    QualityScore,
)

from lumiseval_agent.log import get_node_logger

log = get_node_logger("eval")


def _avg(metrics: list[MetricResult]) -> Optional[float]:
    """Simple unweighted average of metrics that have a score."""
    scored = [m.score for m in metrics if m.score is not None]
    return sum(scored) / len(scored) if scored else None


def aggregate(
    job_id: str,
    hallucination_metrics: list[MetricResult],
    ragas_metrics: list[MetricResult],
    adversarial_metrics: list[MetricResult],
    rubric_metrics: list[MetricResult],
    cost_estimate: Optional[CostEstimate],
    cost_actual_usd: float,
    job_config: EvalJobConfig,
) -> EvalReport:
    """Assemble all metric results into a final EvalReport."""
    warnings: list[str] = []

    # Synthesise evidence_support_rate from claim verdicts and add to the pool
    all_metrics: list[MetricResult] = list(
        hallucination_metrics + ragas_metrics + adversarial_metrics + rubric_metrics
    )

    # Partition — vulnerability markers (score=0 presence flags) go to warnings, not scores
    retrieval_metrics = [
        m for m in all_metrics
        if m.category == MetricCategory.RETRIEVAL and m.score is not None
    ]
    answer_metrics = [
        m for m in all_metrics
        if m.category == MetricCategory.ANSWER
        and m.score is not None
        and not m.name.startswith("vulnerability_")
    ]

    # Surface vulnerability hits and metric errors as warnings
    for m in all_metrics:
        if m.name.startswith("vulnerability_") and m.reasoning:
            warnings.append(f"Adversarial [{m.name}]: {m.reasoning[:120]}")
    for m in all_metrics:
        if m.error:
            warnings.append(f"{m.name} error: {m.error}")

    # Compute per-dimension quality scores (simple unweighted average)
    retrieval_avg = _avg(retrieval_metrics)
    answer_avg = _avg(answer_metrics)

    retrieval_qs = QualityScore(
        score=round(retrieval_avg, 4) if retrieval_avg is not None else None,
        metrics=retrieval_metrics,
    )
    answer_qs = QualityScore(
        score=round(answer_avg, 4) if answer_avg is not None else None,
        metrics=answer_metrics,
    )

    # Composite — simple average of the two dimension scores
    top_scores = [s for s in [retrieval_avg, answer_avg] if s is not None]
    composite_score: Optional[float] = None
    if top_scores:
        composite_score = round(sum(top_scores) / len(top_scores), 4)
        log.info(
            f"retrieval_score={retrieval_avg}  "
            f"answer_score={answer_avg}  "
            f"composite={composite_score}"
        )

    # Confidence band — std dev of per-claim retrieval scores
    retrieval_scores = [p.retrieval_score for v in claim_verdicts for p in v.passages]
    confidence_band: Optional[float] = None
    if len(retrieval_scores) >= 2:
        confidence_band = round(statistics.stdev(retrieval_scores), 4)

    if cost_estimate and cost_estimate.approximate_warning:
        warnings.append(cost_estimate.approximate_warning)

    evaluation_incomplete = not retrieval_metrics and not answer_metrics and not claim_verdicts

    return EvalReport(
        job_id=job_id,
        composite_score=composite_score,
        confidence_band=confidence_band,
        retrieval_score=retrieval_qs,
        answer_score=answer_qs,
        claim_verdicts=claim_verdicts,
        cost_estimate=cost_estimate,
        cost_actual_usd=round(cost_actual_usd, 6),
        evaluation_incomplete=evaluation_incomplete,
        warnings=warnings,
    )
