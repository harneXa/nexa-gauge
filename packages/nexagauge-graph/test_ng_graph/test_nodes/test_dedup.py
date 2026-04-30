# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_dedup.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_dedup.py::test_run_returns_refiner_artifacts_from_mocked_deduplicate
# uv run pytest -s -k "refiner" packages/nexagauge-graph/test_ng_graph/test_nodes/test_dedup.py

import pytest
from ng_core.types import Item
from ng_graph.nodes import refiner as refiner_module
from ng_graph.nodes.refiner import RefinerNode


def test_run_returns_refiner_artifacts_from_mocked_deduplicate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    items = [
        Item(text="Paris is the capital of France.", tokens=7, confidence=0.95),
        Item(text="France's capital is Paris.", tokens=6, confidence=0.90),
        Item(text="Tokyo is the capital of Japan.", tokens=7, confidence=0.93),
    ]

    def fake_deduplicate(_items, top_k=None):
        del top_k
        return [0, 2], {1: 0}

    monkeypatch.setattr(refiner_module, "deduplicate", fake_deduplicate)

    node = RefinerNode()
    result = node.run(items)

    assert len(result.items) == 2
    assert result.items[0].text == "Paris is the capital of France."
    assert result.items[1].text == "Tokyo is the capital of Japan."
    assert result.indices == [0, 2]
    assert result.dropped == 1
    assert result.dedup_map == {1: 0}
    assert result.cost.input_tokens == 0.0
    assert result.cost.output_tokens == 0.0
    assert result.cost.cost == 0.0


def test_estimate_returns_zero_cost() -> None:
    node = RefinerNode()
    estimate = node.estimate(123.0, 456.0)

    assert estimate.input_tokens == 0.0
    assert estimate.output_tokens == 0.0
    assert estimate.cost == 0.0
