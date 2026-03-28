"""
Adversarial Node — runs bias and vulnerability probes.

Combines DeepEval BiasMetric with a Giskard adversarial scan.
Only activated when EvalJobConfig.enable_adversarial=True.

Bias score follows the DeepEval convention (1.0 = passed = unbiased).
Each Giskard vulnerability is surfaced as a separate MetricResult with score=0.0
and passed=False so it appears as a warning in the final report.

TODO: Add PrivacyMetric when it becomes available in the installed DeepEval version.
TODO: Map Giskard severity levels to score weights instead of binary 0/1.
"""

from typing import Optional

try:
    import giskard
    import pandas as pd
    _GISKARD_AVAILABLE = True
except ImportError:
    _GISKARD_AVAILABLE = False

from deepeval.metrics import BiasMetric
from deepeval.test_case import LLMTestCase
from lumiseval_core.constants import ADVERSARIAL_DEFAULT_PROBE_CATEGORIES, METRIC_PASS_THRESHOLD
from lumiseval_core.types import MetricCategory, MetricResult

from lumiseval_agent.log import get_node_logger

log = get_node_logger("redteam")

_DEFAULT_PROBE_CATEGORIES = ADVERSARIAL_DEFAULT_PROBE_CATEGORIES


def run(
    generation: str,
    judge_model: str = "gpt-4o-mini",
    probe_categories: Optional[list[str]] = None,
) -> list[MetricResult]:
    """Run bias and adversarial vulnerability probes.

    Args:
        generation: The LLM-generated text to evaluate.
        judge_model: LiteLLM model string for DeepEval metrics.
        probe_categories: Giskard probe categories to scan.

    Returns:
        list[MetricResult] — bias score, plus one entry per Giskard vulnerability.
    """
    results: list[MetricResult] = []

    # ── DeepEval: Bias ──────────────────────────────────────────────────────
    test_case = LLMTestCase(input="", actual_output=generation)
    bias = BiasMetric(model=judge_model)
    bias.measure(test_case)
    log.info(f"  bias_score={bias.score}")

    results.append(
        MetricResult(
            name="bias",
            category=MetricCategory.ANSWER,
            score=bias.score,
            passed=bias.score is not None and bias.score >= METRIC_PASS_THRESHOLD,
            reasoning=getattr(bias, "reason", None),
        )
    )

    # ── Giskard: adversarial vulnerability scan ─────────────────────────────
    if not _GISKARD_AVAILABLE:
        log.info("Giskard not installed — skipping vulnerability scan")
    else:
        probe_categories = probe_categories or _DEFAULT_PROBE_CATEGORIES
        log.info(f"Probing Giskard categories: {probe_categories}")

        def _predict(df):
            return pd.Series([generation] * len(df))

        giskard_model = giskard.Model(
            model=_predict,
            model_type="text_generation",
            name="lumiseval_probe_target",
            description="LumisEval adversarial probe target",
            feature_names=["input"],
        )
        scan_results = giskard.scan(giskard_model, only=probe_categories)
        issues = scan_results.issues if hasattr(scan_results, "issues") else []
        log.info(f"Giskard scan complete — {len(issues)} issue(s) found")

        for issue in issues:
            group = str(getattr(issue, "group", "unknown"))
            description = str(getattr(issue, "description", ""))
            log.info(f"  [vulnerability] {group}: {description[:80]}")
            results.append(
                MetricResult(
                    name=f"vulnerability_{group}",
                    category=MetricCategory.ANSWER,
                    score=0.0,
                    passed=False,
                    reasoning=description,
                )
            )

    return results
