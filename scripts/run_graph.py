"""
Debug runner for the lumiseval graph pipeline.

# ── Basic usage ────────────────────────────────────────────────────────────────

  # Run all nodes on all records
  python scripts/run_graph.py sample.json

  # Run only the grounding node
  python scripts/run_graph.py sample.json --node grounding

  # Run a slice of records (indices 0–2 inclusive)
  python scripts/run_graph.py sample.json --start 0 --end 3

  # Run only relevance on records 2–4
  python scripts/run_graph.py sample.json --node relevance --start 2 --end 5

# ── Runtime model overrides (--llm-model) ─────────────────────────────────────

  # Override the model for a single node
  python scripts/run_graph.py sample.json --llm-model grounding=openai/gpt-4o-mini

  # Override models for multiple nodes (repeat the flag)
  python scripts/run_graph.py sample.json \
      --start 0 --end 3
      --node grounding \
      --llm-model grounding=openai/gpt-4o-mini \
      --llm-model claims=openai/gpt-4o-mini

  # Node aliases are accepted — "node_grounding" resolves to "grounding"
  python scripts/run_graph.py sample.json --node grounding --llm-model node_grounding=openai/gpt-4o-mini

# ── Fallback model overrides (--llm-fallback) ─────────────────────────────────

  # If the primary model fails, fall back to another
  python scripts/run_graph.py sample.json \
      --start 0 --end 3 \
      --node grounding \
      --llm-model grounding=openai/gpt-4o \
      --llm-fallback grounding=openai/gpt-4o-mini

# ── Temperature overrides (--llm-temp) ────────────────────────────────────────

  python scripts/run_graph.py sample.json --llm-temp claims=0.2

# ── Combined example ──────────────────────────────────────────────────────────

  python scripts/run_graph.py sample.json \
      --node grounding \
      --start 0 --end 5 \
      --llm-model grounding=openai/gpt-4o-mini \
      --llm-fallback grounding=openai/gpt-4o \
      --llm-temp grounding=0.1


{
    {
        name: "bias",
        "rubric": {
            "violations": [
                ...,
                ...
            ]
             "non-violations": [
                ...,
                ...
            ]
        },
        "item_fields": ["generation"],
    }
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, cast

from langgraph.graph import END, StateGraph
from lumiseval_core.utils import pprint_model
from lumiseval_graph.graph import (
    EvalCase,
    node_eval,
    node_generation_chunk,
    node_generation_claims,
    node_generation_claims_dedup,
    node_geval,
    node_geval_steps,
    node_grounding,
    node_metadata_scanner,
    node_redteam,
    node_reference,
    node_relevance,
    node_report,
)
from lumiseval_graph.llm.config import (
    RuntimeLLMOverrides,
    normalize_node_name,
    normalize_runtime_overrides,
)

_C = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "header": "\033[95m",  # bright magenta  — record banner
    "node": "\033[94m",  # bright blue     — node name
    "key": "\033[96m",  # bright cyan     — dict key
    "value": "\033[92m",  # bright green    — value preview
    "none": "\033[90m",  # grey            — None / empty
    "sep": "\033[33m",  # yellow          — separator lines
    "skip": "\033[90m",  # grey            — verbose artifacts
}


def build_graph(
    enabled_metrics: set[str] | None = None,
) -> StateGraph:
    run = enabled_metrics or {
        "claims",
        "grounding",
        "relevance",
        "redteam",
        "geval_steps",
        "geval",
        "reference",
    }
    print("runrunrunrun: ", run)

    g = StateGraph(EvalCase)

    # Trunk — always present
    g.add_node("metadata_scanner", node_metadata_scanner)
    g.add_node("eval", node_eval)
    g.add_node("report", node_report)
    g.set_entry_point("metadata_scanner")

    # Claims branch — only when grounding or relevance is requested
    if "claims" in run or "grounding" in run or "relevance" in run:
        g.add_node("generation_chunk", node_generation_chunk)
        g.add_node("generation_claims", node_generation_claims)
        g.add_node("generation_claims_dedup", node_generation_claims_dedup)
        g.add_edge("metadata_scanner", "generation_chunk")
        g.add_edge("generation_chunk", "generation_claims")
        g.add_edge("generation_claims", "generation_claims_dedup")

        for name, node_fn in [("grounding", node_grounding), ("relevance", node_relevance)]:
            if name in run:
                g.add_node(name, node_fn)
                g.add_edge("generation_claims_dedup", name)
                g.add_edge(name, "eval")

    # Geval — needs its own steps node upstream, branches off scanner
    if "geval" in run:
        g.add_edge("metadata_scanner", "geval_steps")
        g.add_node("geval_steps", node_geval_steps)
        if "geval_steps" not in run:
            g.add_edge("geval_steps", "geval")
            g.add_node("geval", node_geval)
            g.add_edge("geval", "eval")
        else:
            g.add_edge("geval_steps", "eval")

    # Reference & redteam — branch off scanner directly
    for name, node_fn in [("reference", node_reference), ("redteam", node_redteam)]:
        if name in run:
            g.add_node(name, node_fn)
            g.add_edge("metadata_scanner", name)
            g.add_edge(name, "eval")

    # If nothing is enabled, scanner goes straight to eval
    if not run:
        g.add_edge("metadata_scanner", "eval")

    g.add_edge("eval", "report")
    g.add_edge("report", END)
    return g


def _parse_kv_pair(raw: str, *, flag_name: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError(f"Invalid {flag_name} '{raw}'. Expected format: <node>=<value>.")
    node_name, value = raw.split("=", 1)
    node_name = normalize_node_name(node_name, strict=True)
    value = value.strip()
    if not value:
        raise ValueError(f"Invalid {flag_name} '{raw}'. Value cannot be empty.")
    return node_name, value


def _build_llm_overrides(
    *,
    model_pairs: list[str],
    fallback_pairs: list[str],
    temp_pairs: list[str],
) -> Optional[RuntimeLLMOverrides]:
    if not model_pairs and not fallback_pairs and not temp_pairs:
        return None

    models: dict[str, str] = {}
    fallback_models: dict[str, str] = {}
    temperatures: dict[str, float] = {}

    for raw in model_pairs:
        node_name, model = _parse_kv_pair(raw, flag_name="--llm-model")
        models[node_name] = model

    for raw in fallback_pairs:
        node_name, model = _parse_kv_pair(raw, flag_name="--llm-fallback")
        fallback_models[node_name] = model

    for raw in temp_pairs:
        node_name, temp_raw = _parse_kv_pair(raw, flag_name="--llm-temp")
        temperatures[node_name] = float(temp_raw)

    return normalize_runtime_overrides(
        RuntimeLLMOverrides(
            models=models,
            fallback_models=fallback_models,
            temperatures=temperatures,
        )
    )


# ── Dataset runner ─────────────────────────────────────────────────────────


def run_dataset(
    input_path: str | Path,
    *,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    node: Optional[str] = None,
    llm_overrides: Optional[RuntimeLLMOverrides] = None,
) -> list[dict[str, Any]]:
    """Run one record at a time from a JSON file.

    Args:
        debug: When True, print each node's output to stdout with colour.
        node:  When set, only print output for this specific node (implies debug).
    """
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        records: list[dict[str, Any]] = raw
    elif isinstance(raw, dict):
        records = [cast(dict[str, Any], raw)]
    else:
        raise ValueError("Input JSON must be an object or a list of objects")

    selected = records[start_idx:end_idx] if end_idx is not None else records[start_idx:]
    enabled_metrics = {node} if node else None
    app = build_graph(enabled_metrics=enabled_metrics).compile()
    outputs: list[dict[str, Any]] = []

    for offset, record in enumerate(selected):
        idx = start_idx + offset
        case_id = str(record.get("case_id", f"record-{idx}"))
        record["case_id"] = case_id

        invoke_payload: dict[str, Any] = {"record": record}
        if llm_overrides:
            invoke_payload["llm_overrides"] = llm_overrides
        final_state = cast(dict[str, Any], app.invoke(invoke_payload))
        pprint_model(final_state)
        outputs.append(final_state)

    def _serialise(obj: Any) -> Any:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        raise TypeError(type(obj))

    out_name = f"data/{node}.json" if node else "data/output.json"
    out_path = Path(out_name)
    out_path.write_text(json.dumps(outputs, indent=2, default=_serialise), encoding="utf-8")
    print(f"\n{_C['header']}{_C['bold']}  Saved → {out_path.resolve()}{_C['reset']}")

    return outputs


# ── CLI entry point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    # python scripts/run_graph.py sample.json --debug
    # python scripts/run_graph.py sample.json --debug --node grounding
    parser = argparse.ArgumentParser(description="Run the lumiseval graph pipeline.")
    parser.add_argument("input", nargs="?", default="sample.json", help="Path to input JSON file")
    parser.add_argument("--start", type=int, default=0, help="Start record index")
    parser.add_argument("--end", type=int, default=None, help="End record index (exclusive)")
    parser.add_argument(
        "--node", type=str, default=None, help="Only print output for this node name"
    )
    parser.add_argument(
        "--llm-model",
        action="append",
        default=[],
        help="Per-run node model override in format <node>=<model>. Repeatable.",
    )
    parser.add_argument(
        "--llm-fallback",
        action="append",
        default=[],
        help="Per-run node fallback model override in format <node>=<model>. Repeatable.",
    )
    parser.add_argument(
        "--llm-temp",
        action="append",
        default=[],
        help="Per-run node temperature override in format <node>=<float>. Repeatable.",
    )
    args = parser.parse_args()
    llm_overrides = _build_llm_overrides(
        model_pairs=args.llm_model,
        fallback_pairs=args.llm_fallback,
        temp_pairs=args.llm_temp,
    )

    results = run_dataset(
        input_path=args.input,
        start_idx=args.start,
        end_idx=args.end,
        node=args.node,
        llm_overrides=llm_overrides,
    )

"""
    python scripts/run_graph.py sample.json --node grounding


    '/Volumes/Raid1CrucialHD/sardhendu/workspace/lumis-eval/data/claims.json' Is my claims.json.
  For all records where input.has_generation=True. I need you to write me the code to go
  through `generation_claims` and print me for each `tokens` and the cost. for eaxmple "record
  8382be2ebb83986d  ------------ \ntext = The Eiffel Tower is a wrought-iron ....
  input_tokens=285.0, output_tokens=28, cost=0.000005955". Write me the script here /Volumes/Raid1CrucialHD/sardhendu/workspace/lumis-eval/scripts/count_tokens
"""
