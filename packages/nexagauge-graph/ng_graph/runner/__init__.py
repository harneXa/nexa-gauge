"""Cache-aware runner package.

Public compatibility surface intentionally mirrors historical
``ng_graph.runner`` module imports.
"""

from __future__ import annotations

from ng_graph.registry import NODE_FNS

from .engine import CachedNodeRunner
from .fingerprints import _case_id
from .plan import _plan_nodes
from .types import BatchRunResult, CachedNodeRunResult, CaseRunOutcome

__all__ = [
    "BatchRunResult",
    "CachedNodeRunResult",
    "CachedNodeRunner",
    "CaseRunOutcome",
    "NODE_FNS",
    "_case_id",
    "_plan_nodes",
]
