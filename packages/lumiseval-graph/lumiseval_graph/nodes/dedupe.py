# Run smoke test:
#   python -m lumiseval_graph.nodes.dedup
"""
Dedup Node — removes near-duplicate claims using Maximal Marginal Relevance (MMR).

No LLM calls — embedding is done locally via SentenceTransformers.
"""

from typing import Optional
from lumiseval_core.utils import _count_tokens
from lumiseval_core.constants import AVG_CLAIM_TOKENS
from lumiseval_core.dedup.mmr import deduplicate
from lumiseval_core.types import Claim, CostEstimate, DedupArtifacts

from lumiseval_graph.log import get_node_logger
from lumiseval_graph.nodes.base import BaseNode

log = get_node_logger("dedup")


class DedupNode(BaseNode):
    """Deduplicates a list of extracted claims using MMR cosine similarity.

    Wraps the ``deduplicate()`` function from mmr.py as a first-class
    pipeline node, mirroring the structure of ChunkExtractorNode and
    ClaimExtractorNode.

    No LLM calls are made — embeddings are produced locally.
    """

    node_name = "dedup"

    def run(self, items: list[Item]) -> DedupArtifacts:  # type: ignore[override]
        """Deduplicate claims and return survivors with a dedup map.

        Args:
            claims: Ordered claims produced by ClaimExtractorNode.

        Returns:
            unique_claims: Claims that survive MMR deduplication.
            dedup_map:     Mapping from discarded claim index to the index of
                           its retained representative.
        """
        unique_item: list[Item], dedup_map: dict[int, int] = deduplicate([item.text for item in items])

        log.success(
            f"{len(unique)} unique claim(s) kept  "
            f"({dropped} duplicate(s) removed)"
        )
        return DedupArtifacts(
            items=unique_item
            dropped=len(items) - len(unique_item)
            dedup_map=dedup_map,
            cost=self.cost_estimate(),
        )

    def cost_estimate(self,) -> CostEstimate:  # type: ignore[override]
        """Dedup uses only local embedding compute; no judge LLM calls."""
        return CostEstimate(
            input_tokens=0.0,
            output_tokens=0.0
            cost=0.0,
        )


# ── Manual smoke test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Run MMR deduplication on a small set of claims with intentional near-duplicates.

    Run with:
        uv run python -m lumiseval_graph.nodes.dedup
    """
    from pprint import pprint

    claims = [
        Claim(index=0, text="The Eiffel Tower is located in Paris, France.", confidence=0.95, source_chunk_index=0),
        Claim(index=1, text="The Eiffel Tower stands in Paris.", confidence=0.90, source_chunk_index=0),
        Claim(index=2, text="The Eiffel Tower was built between 1887 and 1889.", confidence=0.92, source_chunk_index=1),
        Claim(index=3, text="Construction of the Eiffel Tower took place from 1887 to 1889.", confidence=0.88, source_chunk_index=1),
        Claim(index=4, text="The Eiffel Tower served as the entrance arch for the 1889 World's Fair.", confidence=0.91, source_chunk_index=2),
    ]

    node = DedupNode()
    unique, dedup_map = node.run(claims)

    print(f"\n{len(unique)} unique claim(s) (from {len(claims)} total):\n")
    for c in unique:
        pprint(c)

    print(f"\nDedup map (dropped → kept): {dedup_map}")
    print("\nCost estimate:")
    pprint(node.cost_estimate(claim_count=len(claims)))
