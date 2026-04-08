"""
LangGraph Orchestration Graph — the core evaluation pipeline.

Node sequence:
  scan → chunk → claims → dedup →
  geval_steps →
  [parallel: relevance, grounding, redteam, geval, reference] →
  eval → result

TODO:
  - Implement async TaskIQ dispatch for batch jobs.
  - Stream progress to CLI/API consumers.
  - Persist EvalReport to SQLite via SQLModel.
"""

import logging
import uuid
from typing import Any, NotRequired, Optional, TypedDict, cast

from langgraph.graph import END, StateGraph
from lumiseval_core.config import config as cfg
from lumiseval_core.constants import DEFAULT_DATASET_NAME, DEFAULT_SPLIT
from lumiseval_core.types import (
    ChunkArtifacts,
    Claim,
    ClaimArtifacts,
    CostEstimate,
    DedupArtifacts,
    Geval,
    GevalMetrics,
    GevalStepsArtifacts,
    GroundingMetrics,
    Inputs,
    Item,
    MetricResult,
    ReferenceMetrics,
    RedteamMetrics,
    RelevanceMetrics,
)
from lumiseval_evidence.indexer import index_file

from .llm import get_judge_model
from .log import get_node_logger, print_pipeline_footer, print_pipeline_header
from .nodes import claim_extractor, eval
from .nodes.chunk_extractor import ChunkExtractorNode
from .nodes.dedup import DedupNode
from .nodes.metrics.geval import GevalNode, GevalStepsNode
from .nodes.metrics.grounding import GroundingNode
from .nodes.metrics.redteam import RedteamNode
from .nodes.metrics.reference import ReferenceNode
from .nodes.metrics.relevance import RelevanceNode
from .observability import observe, score_trace, update_trace
from .scanner import scan as scan_record

logger = logging.getLogger(__name__)

# Module-level node loggers — one per pipeline node
_log_scanner = get_node_logger("scan")
_log_chunker = get_node_logger("chunk")
_log_claims = get_node_logger("claims")
_log_mmr = get_node_logger("dedup")
_log_ragas = get_node_logger("relevance")
_log_hallucination = get_node_logger("grounding")
_log_adversarial = get_node_logger("redteam")
_log_geval_steps = get_node_logger("geval_steps")
_log_geval = get_node_logger("geval")
_log_reference = get_node_logger("reference")
_log_agg = get_node_logger("eval")


# ── Graph state ────────────────────────────────────────────────────────────
class EvalCase(TypedDict):
    """Canonical dataset row used by adapters and dataset runners."""
    # Required
    case_id: str
    # Pipeline inputs
    inputs: Optional[Inputs]

    # Pipeline artifacts — None until the corresponding node runs
    generation_chunk: Optional[ChunkArtifacts]
    generation_claims: Optional[ClaimArtifacts]
    generation_dedup_claims: Optional[ClaimArtifacts]
    grounding_metrics: Optional[GroundingMetrics]
    relevance_metrics: Optional[RelevanceMetrics]
    redteam_metrics: Optional[RedteamMetrics]
    geval_steps: Optional[GevalStepsArtifacts]
    geval_metrics: Optional[GevalMetrics]
    reference_metrics: Optional[ReferenceMetrics]


@observe(name="node_scan")
def node_metadata_scanner(state: dict[str, Any]) -> dict[str, Any]:
    """
    In LangGraph, each node gets the current full state snapshot.
    So after node_metadata_scanner returns {"inputs": ...}, that patch is merged into state, and downstream nodes like node_chunk receive the same full state with
    inputs included.

    Important in your current file:

    - node_chunk still reads state["generation"], not state["inputs"].
    - So inputs is available downstream, but node_chunk won’t use it unless you change it.
    """
    # Per-record scan only: hydrate `inputs` from the current state payload.
    geval_raw = state.get("geval")
    if hasattr(geval_raw, "model_dump"):
        geval_raw = geval_raw.model_dump()

    raw_record = {
        "case_id": state.get("case_id") or "record-0",
        "generation": state.get("generation"),
        "question": state.get("question"),
        "reference": state.get("reference"),
        "context": state.get("context"),
        "geval": geval_raw,
        "reference_files": state.get("reference_files") or [],
    }

    case = scan_record(raw_record, idx=0)
    return {"inputs": case.get("inputs")}


@observe(name="node_generation_chunk")
def node_generation_chunk(state: EvalCase) -> EvalCase:
    if not state["inputs"].generation:
        return {"generation_chunk": None}
    text = state["inputs"].generation.text
    chunks: ChunkArtifacts = ChunkExtractorNode(
        chunk_size=GENERATION_CHUNK_SIZE_TOKENS
    ).run(record=text)
    return {"generation_chunk": chunks}


@observe(name="node_generation_claims")
def node_generation_claims(state: EvalCase) -> EvalCase:
    if not state["inputs"].generation or not state.get("generation_chunk"):
        return {"generation_claims": None}
    model = get_judge_model("node_generation_claims")
    claims = claim_extractor.ClaimExtractorNode(model=model).run(
        state["generation_chunk"].chunks
    )
    return {"generation_claims": claims}


@observe(name="node_generation_claims_dedup")
def node_generation_claims_dedup(state: EvalCase) -> EvalCase:
    if not state["generation_claims"]:
        return {"generation_dedup_claims": None}
    unique_items, dedup_map = DedupNode().run(items=state["generation_claims"].claims)
    selected_ids = set(dedup_map.values())

    claims = []
    costs = []
    for claim, cost in zip(
        state["generation_claims"].claims, state["generation_claims"].cost
    ):
        if claim.id in selected_ids:
            claims.append(claim)
            costs.append(cost)

    return {"generation_dedup_claims": ClaimArtifacts(claims=claims, cost=costs)}


@observe(name="node_grounding")
def node_grounding(state: EvalCase) -> EvalCase:
    claims: list[Claim] = state["generation_dedup_claims"].claims or []
    context = state["context"]
    model = get_judge_model("node_grounding")
    results = GroundingNode(judge_model=model).run(
        claims=claims,
        context=context,
        enable_grounding=state["inputs"].enable_grounding,
    )
    return {"grounding_metrics": results}



@observe(name="node_relevance")
def node_relevance(state: EvalCase) -> EvalCase:
    claims: list[Claim] = state["generation_dedup_claims"].claims or []
    question = state["inputs"].question
    model = get_judge_model("node_relevance")
    results = RelevanceNode(judge_model=model).run(
        claims=claims,
        question=state.get("question"),
        enable_relevance=state["inputs"].enable_relevance,
    )
    return {"relevance_metrics": results}


@observe(name="node_redteam")
def node_redteam(state: EvalCase) -> EvalCase:
    model = get_judge_model("node_redteam")
    results = RedteamNode(judge_model=model).run(item=state["inputs"].generation)
    return {"redteam_metrics": results}



@observe(name="node_geval_steps")
def node_geval_steps(state: EvalCase) -> EvalCase:
    geval_cfg = state["inputs"].geval
    metrics = geval_cfg.metrics if geval_cfg is not None else []
    if not state["inputs"].enable_geval or not metrics:
        return {"geval_steps_by_signature": {}}

    model = get_judge_model("geval_steps")
    steps_by_signature = GevalStepsNode(judge_model=model).run(metrics=metrics)
    return {"geval_steps_by_signature": steps_by_signature}



@observe(name="node_geval")
def node_geval(state: EvalCase) -> EvalCase:
    geval_cfg = state["inputs"].geval
    metrics = geval_cfg.metrics if geval_cfg is not None else []
    if not state["inputs"].enable_geval or not metrics:
        return {"geval_metrics": []}

    model = get_judge_model("geval")
    results = GevalNode(judge_model=model).run(
        metrics=metrics,
        generation=state.get("inputs").generation,
        question=state.get("inputs").question,
        reference=state.get("inputs").reference,
        context=state.get("inputs").context,
        steps_by_signature=state.get("geval_steps_by_signature") or {},
    )
    return {"geval_metrics": results}


@observe(name="node_reference")
def node_reference(state: EvalCase) -> EvalCase:
    if not state["inputs"].enable_reference:
        return {"reference_metrics": []}
    reference = state.get("reference")
    if not reference:
        return {"reference_metrics": []}
    results = ReferenceNode().run(
        generation=state["generation"],
        reference=reference,
        enable_generation_metrics=True,
    )
    return {"reference_metrics": results}


@observe(name="node_eval")
def node_eval(state: EvalCase) -> EvalCase:
    report = eval.aggregate(
        job_id=state.get("job_id", ""),
        grounding_metrics=state.get("grounding_metrics") or [],
        relevance_metrics=state.get("relevance_metrics") or [],
        redteam_metrics=state.get("redteam_metrics") or [],
        geval_metrics=state.get("geval_metrics") or [],
        reference_metrics=state.get("reference_metrics") or [],
        cost_estimate=state.get("cost_estimate"),
        cost_actual_usd=state.get("cost_actual_usd", 0.0),
    )
    return {"report": report}


# ── Graph construction ─────────────────────────────────────────────────────


def build_graph() -> StateGraph:
    g = StateGraph(EvalCase)

    g.add_node("metadata_scanner", node_metadata_scanner)
    g.add_node("generation_chunk", node_generation_chunk)
    g.add_node("generation_claims", node_generation_claims)
    g.add_node("generation_claims_dedup", node_generation_claims_dedup)
    g.add_node("geval_steps", node_geval_steps)
    g.add_node("relevance", node_relevance)
    g.add_node("grounding", node_grounding)
    g.add_node("redteam", node_redteam)
    g.add_node("geval", node_geval)
    g.add_node("reference", node_reference)
    g.add_node("eval", node_eval)

    g.set_entry_point("metadata_scanner")
    g.add_edge("metadata_scanner", "generation_chunk")
    g.add_edge("metadata_scanner", "geval_steps")
    g.add_edge("metadata_scanner", "redteam")
    g.add_edge("metadata_scanner", "reference")
    g.add_edge("generation_chunk", "generation_claims")
    g.add_edge("generation_claims", "generation_claims_dedup")
    g.add_edge("geval_steps", "geval")
    g.add_edge("generation_claims_dedup", "relevance")
    g.add_edge("generation_claims_dedup", "grounding")
    g.add_edge("relevance", "eval")
    g.add_edge("grounding", "eval")
    g.add_edge("redteam", "eval")
    g.add_edge("geval", "eval")
    g.add_edge("reference", "eval")
    g.add_edge("eval", END)

    return g




def build_initial_state(
    generation: str = None,
    question: Optional[str] = None,
    reference: Optional[str] = None,
    context: Optional[list[str]] = None,
    geval: Optional[Any] = None,
    reference_files: Optional[list[str]] = None,
) -> EvalCase:

    case_id = str(raw_record.get("case_id") or f"record-{uuid.uuid4().hex[:8]}")
    judge_model = DEFAULT_JUDGE_MODEL

    return EvalCase(
        case_id=case_id,
        inputs=Inputs(
            generation=raw_record.get("generation"),
            context=raw_record.get("context"),
            question=raw_record.get("question"),
            reference=raw_record.get("reference"),
            geval=raw_record.get("geval")
            
        )
    )


@observe(name="lumiseval_pipeline")
def run_graph(case: EvalCase) -> dict[str, Any]:
    app = build_graph().compile()
    return cast(dict[str, Any], app.invoke(case))


def run_dataset(
    input_path: str | Path,
    *,
    start_idx: int = 0,
    end_idx: Optional[int] = None,
    continue_on_error: bool = True,
) -> list[dict[str, Any]]:
    """Run one record at a time in sequence from a JSON file."""
    raw = json.loads(Path(input_path).read_text(encoding="utf-8"))
    records: list[dict[str, Any]]
    if isinstance(raw, list):
        records = cast(list[dict[str, Any]], raw)
    elif isinstance(raw, dict):
        records = [cast(dict[str, Any], raw)]
    else:
        raise ValueError("Input JSON must be an object or a list of objects")

    selected = records[start_idx:end_idx] if end_idx is not None else records[start_idx:]
    app = build_graph().compile()
    outputs: list[dict[str, Any]] = []

    for offset, record in enumerate(selected):
        idx = start_idx + offset
        state = build_initial_state(record=record)
        try:
            final_state = cast(dict[str, Any], app.invoke(state))
            outputs.append(final_state)
        except Exception as exc:
            if not continue_on_error:
                raise
            outputs.append(
                {
                    "case_id": record.get("case_id", f"record-{idx}"),
                    "error": str(exc),
                    "record_index": idx,
                }
            )

    return outputs

if __name__ == "__main__":
    run_dataset(input_path="data/sample.json", start_idx=0, end_idx=1)