from __future__ import annotations

from lumiseval_graph.nodes import report


def test_extract_path_scalar() -> None:
    data = {"inputs": {"case_id": "case-1"}}
    assert report._extract_path(data, "inputs.case_id") == "case-1"


def test_extract_path_list_wildcard() -> None:
    data = {
        "claims": [
            {"item": {"text": "A", "tokens": 1.0}},
            {"item": {"text": "B", "tokens": 2.0}},
        ]
    }
    assert report._extract_path(data, "claims[*].item.text") == ["A", "B"]
    assert report._extract_path(data, "claims[*].item.tokens") == [1.0, 2.0]


def test_extract_path_missing_returns_nullish_defaults() -> None:
    data = {"claims": None}
    assert report._extract_path(data, "claims[*].item.text") == []
    assert report._extract_path(data, "claims.item.text") is None


def test_project_node_output_uses_declared_map() -> None:
    output = {
        "claims": [
            {"item": {"text": "Alpha", "tokens": 5.0}},
            {"item": {"text": "Beta", "tokens": 4.0}},
        ],
        "cost": {"cost": 0.1},
    }
    projected = report._project_node_output("claims", output)
    assert projected == {
        "claim_texts": ["Alpha", "Beta"],
    }


def test_project_node_output_returns_none_for_unmapped_nodes() -> None:
    output = {"foo": "bar"}
    assert report._project_node_output("unknown-node", output) is None
