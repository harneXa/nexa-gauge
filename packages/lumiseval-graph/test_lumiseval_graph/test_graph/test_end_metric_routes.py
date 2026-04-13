# Debug commands:
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_metric_targets_contains_only_that_branch
# uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_eval_contains_all_metric_branches

from __future__ import annotations

from collections.abc import Callable

import pytest
from lumiseval_core.types import (
    Chunk,
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    Inputs,
    Item,
    MetricCategory,
    MetricResult,
)


def _empty_metric_groups() -> dict[str, list[MetricResult]]:
    return {
        "grounding_metrics": [],
        "relevance_metrics": [],
        "redteam_metrics": [],
        "geval_metrics": [],
        "reference_metrics": [],
    }


@pytest.mark.parametrize(
    ("target_node", "group_key", "metric_name", "category"),
    [
        ("grounding", "grounding_metrics", "grounding", MetricCategory.ANSWER),
        ("relevance", "relevance_metrics", "answer_relevancy", MetricCategory.ANSWER),
        ("reference", "reference_metrics", "rouge_l", MetricCategory.RETRIEVAL),
        ("geval", "geval_metrics", "geval_coherence", MetricCategory.ANSWER),
        ("redteam", "redteam_metrics", "vulnerability_prompt_injection", MetricCategory.ANSWER),
    ],
)
def test_report_for_metric_targets_contains_only_that_branch(
    target_node: str,
    group_key: str,
    metric_name: str,
    category: MetricCategory,
    make_metric: Callable[[str, float, MetricCategory], MetricResult],
    graph_module,
) -> None:
    """
    pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_metric_targets_contains_only_that_branch
    """
    groups = _empty_metric_groups()
    groups[group_key] = [make_metric(metric_name, 0.8, category)]

    state = {
        "job_id": f"job-{target_node}",
        **groups,
        "cost_estimate": None,
        "cost_actual_usd": 0.0,
    }

    eval_out = graph_module.node_eval(state)
    print("eval_out ", eval_out)
    state.update(eval_out)
    report_out = graph_module.node_report(state)
    report = report_out["report"]

    assert isinstance(report, dict)
    assert report["target_node"] == "eval"
    assert isinstance(report["metrics"], list)
    assert len(report["metrics"]) == 1
    assert report["metrics"][0]["name"] == metric_name


def test_report_for_eval_contains_all_metric_branches(
    make_metric: Callable[[str, float, MetricCategory], MetricResult],
    graph_module,
) -> None:
    """
    pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_eval_contains_all_metric_branches
    """
    state = {
        "job_id": "job-eval",
        "grounding_metrics": [make_metric("grounding", 1.0, MetricCategory.ANSWER)],
        "relevance_metrics": [make_metric("answer_relevancy", 0.9, MetricCategory.ANSWER)],
        "redteam_metrics": [
            make_metric("vulnerability_prompt_injection", 0.2, MetricCategory.ANSWER)
        ],
        "geval_metrics": [make_metric("geval_coherence", 0.7, MetricCategory.ANSWER)],
        "reference_metrics": [make_metric("rouge_l", 0.6, MetricCategory.RETRIEVAL)],
        "cost_estimate": None,
        "cost_actual_usd": 0.0,
    }

    eval_out = graph_module.node_eval(state)
    state.update(eval_out)
    report_out = graph_module.node_report(state)
    report = report_out["report"]

    assert isinstance(report, dict)
    assert report["target_node"] == "eval"
    metrics = report["metrics"]
    assert isinstance(metrics, list)
    assert len(metrics) == 5
    assert [m["name"] for m in metrics] == [
        "grounding",
        "answer_relevancy",
        "vulnerability_prompt_injection",
        "geval_coherence",
        "rouge_l",
    ]


def test_node_eval_is_orchestration_only(graph_module) -> None:
    assert graph_module.node_eval({}) == {}


def test_report_for_grounding_target_includes_inputs_and_branch_nodes(graph_module) -> None:
    """
    pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py::test_report_for_grounding_target_includes_inputs_and_branch_nodes
    """
    state = {
        "target_node": "grounding",
        "record": {
            "case_id": "case-1",
            "generation": "Paris is the capital of France.",
            "question": "What is the capital of France?",
            "reference": "Paris",
            "context": "France has Paris as its capital.",
        },
        "inputs": Inputs(
            case_id="case-1",
            generation=Item(text="Paris is the capital of France.", tokens=7.0),
            question=Item(text="What is the capital of France?", tokens=7.0),
            context=Item(text="France has Paris as its capital.", tokens=7.0),
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
                    item=Item(text="Paris is the capital of France.", tokens=7.0),
                    char_start=0,
                    char_end=31,
                    sha256="abc123",
                )
            ],
            cost=CostEstimate(cost=0.0, input_tokens=0.0, output_tokens=0.0),
        ),
        "generation_claims": ClaimArtifacts(
            claims=[
                Claim(
                    item=Item(text="Paris is the capital of France.", tokens=7.0),
                    source_chunk_index=0,
                    confidence=0.95,
                )
            ],
            cost=CostEstimate(cost=0.001, input_tokens=10.0, output_tokens=3.0),
        ),
        "generation_dedup_claims": ClaimArtifacts(
            claims=[
                Claim(
                    item=Item(text="Paris is the capital of France.", tokens=7.0),
                    source_chunk_index=0,
                    confidence=0.95,
                )
            ],
            cost=CostEstimate(cost=0.0, input_tokens=None, output_tokens=None),
        ),
        "grounding_metrics": graph_module.GroundingMetrics(
            metrics=[
                MetricResult(
                    name="grounding",
                    category=MetricCategory.ANSWER,
                    score=1.0,
                )
            ],
            cost=CostEstimate(cost=0.002, input_tokens=20.0, output_tokens=2.0),
        ),
        "node_model_usage": {
            "claims": {
                "used_models": ["openai/gpt-4o-mini"],
                "used_model_counts": {"openai/gpt-4o-mini": 1},
                "total_calls": 1,
                "fallback_hits": 0,
            },
            "grounding": {
                "used_models": ["openai/gpt-4o"],
                "used_model_counts": {"openai/gpt-4o": 1},
                "total_calls": 1,
                "fallback_hits": 1,
            },
        },
        "estimated_costs": {},
    }

    report_out = graph_module.node_report(state)
    report = report_out["report"]

    assert report["target_node"] == "grounding"
    assert report["inputs"]["case_id"] == "case-1"
    assert report["inputs"]["question"] == "What is the capital of France?"
    assert report["inputs"]["generation"] == "Paris is the capital of France."
    assert report["inputs"]["context"] == "France has Paris as its capital."
    assert report["inputs"]["reference"] == "Paris"
    assert report["branch"] == ["scan", "chunk", "claims", "dedup", "grounding"]
    assert set(report["nodes"].keys()) == {"scan", "chunk", "claims", "dedup", "grounding"}
    assert report["nodes"]["chunk"]["eligible"] is True
    assert report["nodes"]["chunk"]["output"] == {
        "chunk_texts": ["Paris is the capital of France."],
    }
    assert report["nodes"]["claims"]["output"] == {
        "claim_texts": ["Paris is the capital of France."],
    }
    assert report["nodes"]["dedup"]["output"] == {
        "claim_texts": ["Paris is the capital of France."],
    }
    assert report["nodes"]["grounding"]["output"] == {
        "metric_scores": [1.0],
        "metric_errors": [None],
    }
    assert report["nodes"]["claims"]["cost"]["cost"] == pytest.approx(0.001)
    assert report["nodes"]["grounding"]["model"]["fallback_hits"] == 1
    assert report["nodes"]["grounding"]["model"]["fallback_used"] is True
