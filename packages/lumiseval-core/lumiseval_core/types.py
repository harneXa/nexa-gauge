"""Shared domain types for lumis-eval."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lumiseval_core.constants import (
    AVG_CLAIM_TOKENS,
    AVG_DEEPEVAL_INPUT_OVERHEAD_TOKENS,
    AVG_DEEPEVAL_OUTPUT_OVERHEAD_TOKENS,
    AVG_GEVAL_INPUT_OVERHEAD_TOKENS,
    AVG_GEVAL_OUTPUT_OVERHEAD_TOKENS,
    AVG_OUTPUT_TOKENS_BOOLEAN_VERDICT,
    AVG_OUTPUT_TOKENS_JSON_VERDICT,
    DEFAULT_DATASET_NAME,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SPLIT,
    EVIDENCE_VERDICT_SUPPORTED_THRESHOLD,
    GENERATION_CHUNK_SIZE_TOKENS,
    SCORE_WEIGHT_ANSWER_RELEVANCY,
    SCORE_WEIGHT_EVIDENCE_SUPPORT_RATE,
    SCORE_WEIGHT_FAITHFULNESS,
    SCORE_WEIGHT_GEVAL,
    SCORE_WEIGHT_HALLUCINATION,
    SCORE_WEIGHT_SAFETY,
)

# ── Enums ──────────────────────────────────────────────────────────────────


class ClaimVerdict(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNVERIFIABLE = "UNVERIFIABLE"


class EvidenceSource(str, Enum):
    LOCAL = "local"
    MCP = "mcp"
    WEB = "web"
    NONE = "none"


class MetricCategory(str, Enum):
    RETRIEVAL = "retrieval"  # was evidence retrieval correct/complete?
    ANSWER = "answer"  # is the answer correct, relevant, and safe?


# ── Core domain models ─────────────────────────────────────────────────────


_ALLOWED_GEVAL_RECORD_FIELDS = {"question", "generation", "reference", "context"}




# ------------------------------------------------------------
# New Nodes
# ------------------------------------------------------------


GevalRecordField = Literal["question", "generation", "reference", "context"]

class Item(BaseModel):
    id: str = ""
    text: str
    tokens: float
    cached: bool = False

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            self.id = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]

class GevalMetricInput(BaseModel):
    """Input carried by a metric per record"""
    name: str
    record_fields: list[GevalRecordField] = Field(default_factory=lambda: ["generation"])
    criteria: Record | None = None
    evaluation_steps: list[Record] 

class Geval(BaseModel):
    """Input contract carried by record input payload."""
    metrics: list[GevalMetricInput] = Field(default_factory=list)

class Chunk(BaseModel):
    index: int
    item: Item
    char_start: int
    char_end: int
    sha256: str

class Claim(BaseModel):
    item: Items
    source_chunk_index: Optional[int] = None
    confidence: float = 1.0
    extraction_failed: bool = False



class Inputs(BaseModel):
    generation: Record
    question: Optional[Record] = None
    reference: Optional[Record] = None
    context: Optional[Record] = None
    geval: Optional[Geval] = None

    has_generation: bool = False
    has_question: bool = False
    has_reference: bool = False
    has_context: bool = False
    has_geval: bool = False


class CostEstimate(BaseModel):
    cost: float
    input_tokens: float
    output_tokens: float

# -----
# NODES Essentials
#
class ChunkArtifacts(BaseModel):
    chunks: list[Chunk]
    cost:  CostEstimate

class ClaimArtifacts(BaseModel):
    claims: list[Claim]
    cost: list[CostEstimate]

class DedupArtifacts(BaseModel):
    items: list[Record]
    dropped: int
    dedup_map: dict[int, int]
    cost: CostEstimate

class GroundingMetrics(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate

class RelevanceMetrics(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate

class RedteamMetrics(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate


class GevalStepsResolved(BaseModel):
    """This is per metric per record."""
    key: str
    name: str
    record_fields: list[GevalRecordField]
    evaluation_steps: list[GevalEvaluationStep]
    steps_source: Literal["provided", "generated"]
    signature: str | None = None



class GevalStepsArtifacts(BaseModel):
    resolved_steps: list[GevalStepsResolved] = Field(default_factory=list)
    cost: CostEstimate | None = None


class GevalMetrics(BaseModel):
    metrics: list[MetricResult] = Field(default_factory=list)
    cost: CostEstimate | None = None



class ReferencePayload(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate

class EvalPayload(BaseModel):
    metrics: list[MetricResult]
    cost: CostEstimate

