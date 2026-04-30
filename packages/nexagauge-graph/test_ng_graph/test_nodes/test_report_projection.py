from __future__ import annotations

from ng_core.types import (
    Chunk,
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    GroundingMetrics,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
)
from ng_graph.nodes import report


def test_report_aggregate_emits_state_key_sections() -> None:
    state = {
        "target_node": "grounding",
        "inputs": Inputs(
            case_id="case-1",
            generation=Item(text="Paris is in France.", tokens=4.0),
            question=Item(text="Where is Paris?", tokens=3.0),
            context=Item(text="Paris is in France.", tokens=4.0),
            reference=Item(text="Paris", tokens=1.0),
            has_generation=True,
            has_question=True,
            has_context=True,
            has_reference=True,
        ),
        "generation_chunk": ChunkArtifacts(
            chunks=[
                Chunk(
                    index=0,
                    item=Item(text="Paris is in France.", tokens=4.0),
                    char_start=0,
                    char_end=19,
                    sha256="abc",
                )
            ],
            cost=CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0),
        ),
        "generation_claims": ClaimArtifacts(
            claims=[
                Claim(item=Item(text="Paris is in France.", tokens=4.0), source_chunk_index=0),
            ],
            cost=CostEstimate(cost=0.1, input_tokens=10.0, output_tokens=3.0),
        ),
    }

    out = report.aggregate(state=state)

    assert out["target_node"] == "grounding"
    assert out["input"]["case_id"] == "case-1"
    assert out["generation_chunk"]["text"] == ["Paris is in France."]
    assert out["generation_claims"]["text"] == ["Paris is in France."]
    assert "generation_refined_chunks" not in out


def test_report_aggregate_projects_metric_wrappers() -> None:
    state = {
        "target_node": "grounding",
        "inputs": Inputs(
            case_id="case-2", generation=Item(text="A", tokens=1.0), has_generation=True
        ),
        "grounding_metrics": GroundingMetrics(
            metrics=[
                MetricResult(
                    name="grounding",
                    category=MetricCategory.ANSWER,
                    score=1.0,
                    verdict="PASSED",
                )
            ],
            cost=CostEstimate(cost=0.2, input_tokens=3.0, output_tokens=1.0),
        ),
    }

    out = report.aggregate(state=state)

    assert out["grounding_metrics"]["metrics"][0]["name"] == "grounding"
    assert out["grounding_metrics"]["metrics"][0]["verdict"] == "PASSED"
    assert out["grounding_metrics"]["cost"]["cost"] == 0.2
