"""
Dedupe Node — orchestrates claim deduplication strategies.

This node is intentionally modeled as a first-class graph node (with
``cost_estimate`` and ``cost_formula`` methods) so cost estimation and node
registry behavior remain uniform across the pipeline.
"""

from typing import Literal

from lumiseval_core.dedupe.mmr import deduplicate as deduplicate_mmr
from lumiseval_core.types import Claim, NodeCostBreakdown

from lumiseval_graph.log import get_node_logger

log = get_node_logger("dedupe")

DedupeStrategy = Literal["mmr"]


class DedupeNode:
    """Run claim deduplication with a pluggable strategy.

    The default strategy is ``mmr``. More strategies can be added later without
    changing graph topology or CLI node names.
    """

    node_name = "dedupe"

    def __init__(self, strategy: DedupeStrategy = "mmr", judge_model: str = "") -> None:
        # judge_model is accepted for interface uniformity with other nodes.
        _ = judge_model
        self.strategy = strategy

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} strategy={self.strategy!r}>"

    def run(self, *, claims: list[Claim]) -> list[Claim]:
        """Deduplicate claims and return the unique claim list."""
        if not claims:
            return []

        if self.strategy == "mmr":
            unique_claims, _dedup_map = deduplicate_mmr(claims)
            log.info(f"dedupe strategy={self.strategy}  in={len(claims)}  out={len(unique_claims)}")
            return unique_claims

        raise ValueError(f"Unknown dedupe strategy: {self.strategy}")

    def cost_estimate(self, **_ignored) -> NodeCostBreakdown:
        """Dedupe uses only local embedding compute; no judge LLM calls."""
        return NodeCostBreakdown(model_calls=0, cost_usd=0.0)

    @staticmethod
    def cost_formula(*_args, **_kwargs) -> str:
        """Human-readable cost explanation used by cost table breakdown."""
        return "No LLM calls (local embedding deduplication)."
