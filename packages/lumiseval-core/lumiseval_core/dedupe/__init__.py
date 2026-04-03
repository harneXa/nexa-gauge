"""Claim deduplication algorithms shared across pipeline layers.

This package holds strategy implementations (for example MMR) that can be
invoked by graph nodes and potentially other orchestration surfaces.
"""

from lumiseval_core.dedupe.mmr import deduplicate

__all__ = ["deduplicate"]

