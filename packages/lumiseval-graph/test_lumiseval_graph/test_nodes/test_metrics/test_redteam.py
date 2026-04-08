# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_metrics/test_redteam.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_metrics/test_redteam.py::test_run_returns_bias_and_toxicity_metrics_with_aggregated_cost
# uv run pytest -s -k "redteam" packages/lumiseval-graph/test_lumiseval_graph/test_nodes/test_metrics/test_redteam.py

import pytest

from lumiseval_core.types import Item, MetricCategory
import lumiseval_graph.nodes.metrics.redteam as redteam_module
from lumiseval_graph.nodes.metrics.redteam import RedteamNode


def test_run_returns_bias_and_toxicity_metrics_with_aggregated_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeBiasMetric:
        def __init__(self, model: str):
            self.model = model
            self.score = None
            self.evaluation_cost = None

        def measure(self, _test_case):
            self.score = 0.2
            self.evaluation_cost = 0.04

    class FakeToxicityMetric:
        def __init__(self, model: str):
            self.model = model
            self.score = None
            self.evaluation_cost = None

        def measure(self, _test_case):
            self.score = 0.5
            self.evaluation_cost = 0.06

    monkeypatch.setattr(redteam_module, "BiasMetric", FakeBiasMetric)
    monkeypatch.setattr(redteam_module, "ToxicityMetric", FakeToxicityMetric)

    node = RedteamNode(judge_model="gpt-4o-mini")
    result = node.run(Item(text="Some generation", tokens=3.0))

    assert len(result.metrics) == 2
    assert result.metrics[0].name == "bias"
    assert result.metrics[1].name == "toxicity"
    assert all(m.category == MetricCategory.ANSWER for m in result.metrics)

    # DeepEval raw score is inverted in RedteamNode: score = 1 - raw
    assert result.metrics[0].score == 0.8
    assert result.metrics[1].score == 0.5

    assert result.cost.input_tokens is None
    assert result.cost.output_tokens is None
    assert result.cost.cost == pytest.approx(0.10)


def test_run_defaults_cost_to_zero_when_metric_cost_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeBiasMetric:
        def __init__(self, model: str):
            self.model = model
            self.score = None
            self.evaluation_cost = None

        def measure(self, _test_case):
            self.score = 0.1
            self.evaluation_cost = None

    class FakeToxicityMetric:
        def __init__(self, model: str):
            self.model = model
            self.score = None

        def measure(self, _test_case):
            self.score = 0.3
            # no evaluation_cost attribute on purpose

    monkeypatch.setattr(redteam_module, "BiasMetric", FakeBiasMetric)
    monkeypatch.setattr(redteam_module, "ToxicityMetric", FakeToxicityMetric)

    node = RedteamNode(judge_model="gpt-4o-mini")
    result = node.run(Item(text="Another generation", tokens=4.0))

    assert len(result.metrics) == 2
    assert result.cost.cost == 0.0
    assert result.cost.input_tokens is None
    assert result.cost.output_tokens is None


def test_estimate_returns_zero_and_unknown_tokens() -> None:
    node = RedteamNode(judge_model="gpt-4o-mini")
    est = node.estimate(123.0, 456.0)

    assert est.cost == 0.0
    assert est.input_tokens is None
    assert est.output_tokens is None
