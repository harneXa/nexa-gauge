# Debug commands:
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_claim_extractor.py
# uv run pytest -s packages/nexagauge-graph/test_ng_graph/test_nodes/test_claim_extractor.py::test_run_builds_claim_artifacts_with_mocked_llm
# uv run pytest -s -k "claim_extractor" packages/nexagauge-graph/test_ng_graph/test_nodes/test_claim_extractor.py

import hashlib
import time
from types import SimpleNamespace

import pytest
from ng_core.types import Chunk, Item
from ng_graph.nodes import claim_extractor as claim_module
from ng_graph.nodes.claim_extractor import ClaimExtractorNode


def _make_chunk(index: int, text: str) -> Chunk:
    return Chunk(
        index=index,
        item=Item(text=text, tokens=float(len(text.split()))),
        char_start=0,
        char_end=len(text),
        sha256=hashlib.sha256(text.encode()).hexdigest(),
    )


def test_run_builds_claim_artifacts_with_mocked_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": SimpleNamespace(
                    claims=["The Eiffel Tower is located in Paris."],
                    confidences=[0.93],
                ),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 25,
                    "total_tokens": 125,
                },
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(claim_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    node = ClaimExtractorNode(model="gpt-4o-mini")
    chunks = [
        _make_chunk(0, "The Eiffel Tower stands in Paris, France."),
        _make_chunk(1, "It was completed in 1889."),
    ]

    result = node.run(chunks)

    assert len(result.claims) == 2
    assert result.claims[0].item.text == "The Eiffel Tower is located in Paris."
    assert result.claims[0].source_chunk_index == 0
    assert result.claims[1].source_chunk_index == 1

    assert result.cost.input_tokens == 200  # 100 per chunk * 2 chunks
    assert result.cost.output_tokens == 50  # 25 per chunk * 2 chunks
    assert result.cost.cost > 0


def test_run_raises_on_parsing_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLLM:
        def invoke(self, _messages):
            return {
                "parsed": None,
                "parsing_error": ValueError("bad parse"),
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(claim_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    node = ClaimExtractorNode(model="gpt-4o-mini")
    chunks = [_make_chunk(0, "Short chunk")]

    with pytest.raises(ValueError, match="bad parse"):
        node.run(chunks)


def test_parallel_and_serial_runs_return_same_ordered_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLLM:
        def invoke(self, messages):
            content = messages[1]["content"]
            if "chunk-0" in content:
                time.sleep(0.04)
                claim = "claim-for-0"
            elif "chunk-1" in content:
                time.sleep(0.01)
                claim = "claim-for-1"
            else:
                claim = "claim-for-2"
            return {
                "parsed": SimpleNamespace(claims=[claim], confidences=[0.8]),
                "parsing_error": None,
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 10,
                    "total_tokens": 60,
                },
                "model": "gpt-4o-mini",
            }

    monkeypatch.setattr(claim_module, "get_llm", lambda *_args, **_kwargs: FakeLLM())

    chunks = [
        _make_chunk(0, "chunk-0 text"),
        _make_chunk(1, "chunk-1 text"),
        _make_chunk(2, "chunk-2 text"),
    ]

    monkeypatch.setattr(claim_module, "CLAIMS_MAX_WORKERS", 1)
    serial = ClaimExtractorNode(model="gpt-4o-mini").run(chunks)

    monkeypatch.setattr(claim_module, "CLAIMS_MAX_WORKERS", 8)
    parallel = ClaimExtractorNode(model="gpt-4o-mini").run(chunks)

    assert [c.item.text for c in serial.claims] == [c.item.text for c in parallel.claims]
    assert [c.source_chunk_index for c in serial.claims] == [
        c.source_chunk_index for c in parallel.claims
    ]
    assert serial.cost.input_tokens == parallel.cost.input_tokens
    assert serial.cost.output_tokens == parallel.cost.output_tokens
