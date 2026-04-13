"""Structured per-record report aggregation helpers."""

from __future__ import annotations

from typing import Any, Mapping

from lumiseval_core.config import config as cfg
from lumiseval_core.types import CostEstimate, Inputs, MetricResult
from lumiseval_graph.llm.config import get_node_config
from lumiseval_graph.topology import NODES_BY_NAME

LLM_NODES: frozenset[str] = frozenset(
    {"claims", "grounding", "relevance", "redteam", "geval_steps", "geval"}
)

NODE_OUTPUT_SOURCE_KEYS: dict[str, str] = {
    "chunk": "generation_chunk",
    "claims": "generation_claims",
    "dedup": "generation_dedup_claims",
    "geval_steps": "geval_steps",
    "grounding": "grounding_metrics",
    "relevance": "relevance_metrics",
    "redteam": "redteam_metrics",
    "geval": "geval_metrics",
    "reference": "reference_metrics",
}

NODE_OUTPUT_PROJECTION_MAP: dict[str, dict[str, str]] = {
    # "scan": {
    #     "has_generation": "has_generation",
    #     "has_question": "has_question",
    #     "has_context": "has_context",
    #     "has_reference": "has_reference",
    #     "has_geval": "has_geval",
    #     "has_redteam": "has_redteam",
    # },
    "chunk": {
        "chunk_texts": "chunks[*].item.text",
        # "chunk_tokens": "chunks[*].item.tokens",
    },
    "claims": {
        "claim_texts": "claims[*].item.text",
        # "claim_tokens": "claims[*].item.tokens",
    },
    "dedup": {
        "claim_texts": "claims[*].item.text",
        # "claim_tokens": "claims[*].item.tokens",
    },
    "geval_steps": {
        "metric_keys": "resolved_steps[*].key",
        "metric_names": "resolved_steps[*].name",
        "item_fields": "resolved_steps[*].item_fields",
        "steps_source": "resolved_steps[*].steps_source",
        "step_texts": "resolved_steps[*].evaluation_steps[*].text",
    },
    "grounding": {
        # "metric_names": "metrics[*].name",
        # "metric_categories": "metrics[*].category",
        "metric_scores": "metrics[*].score",
        "metric_errors": "metrics[*].error",
    },
    "relevance": {
        # "metric_names": "metrics[*].name",
        # "metric_categories": "metrics[*].category",
        "metric_scores": "metrics[*].score",
        "metric_errors": "metrics[*].error",
    },
    "redteam": {
        # "metric_names": "metrics[*].name",
        # "metric_categories": "metrics[*].category",
        "metric_scores": "metrics[*].score",
        "metric_errors": "metrics[*].error",
    },
    "geval": {
        # "metric_names": "metrics[*].name",
        # "metric_categories": "metrics[*].category",
        "metric_scores": "metrics[*].score",
        "metric_errors": "metrics[*].error",
    },
    "reference": {
        # "metric_names": "metrics[*].name",
        # "metric_categories": "metrics[*].category",
        "metric_scores": "metrics[*].score",
        "metric_errors": "metrics[*].error",
    },
    "eval": {"joined": "joined"},
}

METRIC_GROUP_KEYS: tuple[str, ...] = (
    "grounding_metrics",
    "relevance_metrics",
    "redteam_metrics",
    "geval_metrics",
    "reference_metrics",
)

ZERO_COST = {
    "cost": 0.0,
    "input_tokens": None,
    "output_tokens": None,
}


def _to_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    return value


def _to_json_metric(metric: MetricResult) -> dict[str, Any]:
    if hasattr(metric, "model_dump"):
        return metric.model_dump()
    return {
        "name": getattr(metric, "name", None),
        "category": getattr(metric, "category", None),
        "score": getattr(metric, "score", None),
        "result": getattr(metric, "result", None),
        "error": getattr(metric, "error", None),
    }


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if hasattr(value, "text"):
        return str(getattr(value, "text"))
    text = str(value)
    return text if text else None


def _normalize_cost(value: Any) -> dict[str, float | None]:
    if value is None:
        return dict(ZERO_COST)

    if isinstance(value, CostEstimate):
        estimate = value
    elif isinstance(value, Mapping):
        estimate = CostEstimate.model_validate(value)
    else:
        return dict(ZERO_COST)

    return {
        "cost": float(estimate.cost or 0.0),
        "input_tokens": float(estimate.input_tokens)
        if estimate.input_tokens is not None
        else None,
        "output_tokens": float(estimate.output_tokens)
        if estimate.output_tokens is not None
        else None,
    }


def _unwrap_metrics(value: Any) -> list[MetricResult]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return list(getattr(value, "metrics", None) or [])


def _branch_nodes(target_node: str) -> list[str]:
    spec = NODES_BY_NAME.get(target_node)
    if spec is None:
        return []
    return [*spec.prerequisites, target_node]


def _node_eligible(node_name: str, inputs: Inputs | None) -> bool:
    if node_name in {"scan", "eval", "report"}:
        return True
    if inputs is None:
        return False

    spec = NODES_BY_NAME[node_name]
    if spec.requires_generation and not bool(inputs.has_generation):
        return False
    if spec.requires_context and not bool(inputs.has_context):
        return False
    if spec.requires_question and not bool(inputs.has_question):
        return False
    if spec.requires_geval and not bool(inputs.has_geval):
        return False
    if spec.requires_reference and not bool(inputs.has_reference):
        return False
    return True


def _project_inputs(state: Mapping[str, Any]) -> dict[str, Any]:
    inputs = state.get("inputs")
    record = state.get("record") if isinstance(state.get("record"), Mapping) else {}
    record_case_id = str((record or {}).get("case_id") or "")

    if inputs is None:
        return {
            "case_id": record_case_id or None,
            "question": _as_text((record or {}).get("question")),
            "generation": _as_text((record or {}).get("generation")),
            "context": _as_text((record or {}).get("context")),
            "reference": _as_text((record or {}).get("reference")),
        }

    return {
        "case_id": str(getattr(inputs, "case_id", "") or record_case_id or ""),
        "question": _as_text(getattr(inputs, "question", None)),
        "generation": _as_text(getattr(inputs, "generation", None)),
        "context": _as_text(getattr(inputs, "context", None)),
        "reference": _as_text(getattr(inputs, "reference", None)),
    }


def _project_scan_output(state: Mapping[str, Any]) -> dict[str, Any]:
    inputs = state.get("inputs")
    if inputs is None:
        return {}
    return {
        "has_generation": bool(getattr(inputs, "has_generation", False)),
        "has_question": bool(getattr(inputs, "has_question", False)),
        "has_context": bool(getattr(inputs, "has_context", False)),
        "has_reference": bool(getattr(inputs, "has_reference", False)),
        "has_geval": bool(getattr(inputs, "has_geval", False)),
        "has_redteam": bool(getattr(inputs, "has_redteam", False)),
    }


def _raw_node_output(node_name: str, state: Mapping[str, Any]) -> Any:
    if node_name == "scan":
        return _project_scan_output(state)
    if node_name == "eval":
        return {"joined": True}

    state_key = NODE_OUTPUT_SOURCE_KEYS.get(node_name)
    if state_key is None:
        return None
    return _to_jsonable(state.get(state_key))


def _extract_path(value: Any, path: str) -> Any:
    segments = [seg for seg in path.split(".") if seg]
    if not segments:
        return value

    def _walk(current: Any, index: int) -> Any:
        if index >= len(segments):
            return current

        seg = segments[index]
        wildcard = seg.endswith("[*]")
        key = seg[:-3] if wildcard else seg

        if key:
            if not isinstance(current, Mapping):
                return [] if wildcard else None
            current = current.get(key)

        if wildcard:
            if not isinstance(current, list):
                return []
            return [_walk(item, index + 1) for item in current]

        return _walk(current, index + 1)

    return _walk(value, 0)


def _project_node_output(node_name: str, output: Any) -> Any:
    projection_fields = NODE_OUTPUT_PROJECTION_MAP.get(node_name)
    if projection_fields is None:
        return None
    if output is None:
        return None
    return {name: _extract_path(output, path) for name, path in projection_fields.items()}


def _node_output(node_name: str, state: Mapping[str, Any]) -> Any:
    raw_output = _raw_node_output(node_name, state)
    return _project_node_output(node_name, raw_output)


def _node_cost(node_name: str, state: Mapping[str, Any]) -> dict[str, float | None]:
    state_key = NODE_OUTPUT_SOURCE_KEYS.get(node_name)
    artifact = state.get(state_key) if state_key else None
    artifact_cost = getattr(artifact, "cost", None)
    if artifact_cost is not None:
        return _normalize_cost(artifact_cost)

    estimated = state.get("estimated_costs")
    if isinstance(estimated, Mapping) and node_name in estimated:
        return _normalize_cost(estimated[node_name])

    return dict(ZERO_COST)


def _node_model(node_name: str, state: Mapping[str, Any]) -> dict[str, Any]:
    uses_llm = node_name in LLM_NODES
    if not uses_llm:
        return {
            "uses_llm": False,
            "configured_primary": None,
            "configured_fallback": None,
            "used_models": [],
            "used_model_counts": {},
            "total_calls": 0,
            "fallback_hits": 0,
            "fallback_used": False,
        }

    node_cfg = get_node_config(node_name, llm_overrides=state.get("llm_overrides"))
    configured_primary = node_cfg.model or cfg.LLM_MODEL
    runtime = {}
    runtime_all = state.get("node_model_usage")
    if isinstance(runtime_all, Mapping):
        runtime = runtime_all.get(node_name) or {}

    used_models = list(runtime.get("used_models") or [])
    model_counts = runtime.get("used_model_counts") or {}
    total_calls = int(runtime.get("total_calls", 0) or 0)
    fallback_hits = int(runtime.get("fallback_hits", 0) or 0)

    return {
        "uses_llm": True,
        "configured_primary": configured_primary,
        "configured_fallback": node_cfg.fallback_model,
        "used_models": used_models,
        "used_model_counts": model_counts,
        "total_calls": total_calls,
        "fallback_hits": fallback_hits,
        "fallback_used": fallback_hits > 0,
    }


def _collect_metrics(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    merged: list[MetricResult] = []
    for key in METRIC_GROUP_KEYS:
        merged.extend(_unwrap_metrics(state.get(key)))
    return [_to_json_metric(metric) for metric in merged]


def aggregate(*, state: Mapping[str, Any]) -> dict[str, Any]:
    target_node = str(state.get("target_node") or "eval")
    branch_nodes = _branch_nodes(target_node)
    inputs = state.get("inputs")

    nodes_payload: dict[str, Any] = {}
    for node_name in branch_nodes:
        nodes_payload[node_name] = {
            "eligible": _node_eligible(node_name, inputs),
            "output": _node_output(node_name, state),
            "cost": _node_cost(node_name, state),
            "model": _node_model(node_name, state),
        }

    cost_estimate = state.get("cost_estimate")
    return {
        "target_node": target_node,
        "inputs": _project_inputs(state),
        "branch": branch_nodes,
        "nodes": nodes_payload,
        "metrics": _collect_metrics(state),
        "cost_actual_usd": float(state.get("cost_actual_usd") or 0.0),
        "cost_estimate": _normalize_cost(cost_estimate) if cost_estimate is not None else None,
    }
