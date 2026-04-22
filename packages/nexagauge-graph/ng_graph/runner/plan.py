from __future__ import annotations

from typing import Mapping

from ng_graph.registry import NODE_FNS
from ng_graph.topology import NODES_BY_NAME, transitive_prerequisites

from .types import _RunPlanContext


def _plan_direct_prerequisites(plan: list[str]) -> dict[str, tuple[str, ...]]:
    """Return direct prerequisite edges restricted to ``plan`` nodes only."""
    plan_set = set(plan)
    return {
        node: tuple(parent for parent in NODES_BY_NAME[node].prerequisites if parent in plan_set)
        for node in plan
    }


def _plan_transitive_prerequisites(
    *,
    node_name: str,
    plan: list[str],
    direct_prereqs: Mapping[str, tuple[str, ...]],
) -> tuple[str, ...]:
    """Return all ancestors of ``node_name`` constrained to the current plan."""
    seen: set[str] = set()

    def _visit(name: str) -> None:
        for parent in direct_prereqs.get(name, ()):
            if parent in seen:
                continue
            seen.add(parent)
            _visit(parent)

    _visit(node_name)
    return tuple(node for node in plan if node in seen)


def _plan_nodes(node_name: str) -> list[str]:
    """Build the ordered list of nodes to execute for a requested target.

    Returns ``[*prerequisites, target, "eval", "report"]``. Every plan funnels
    through ``eval`` before ``report``: utility and metric nodes never reach
    ``report`` directly — ``eval`` is the sole subscriber of ``report``.
    """
    plan = list(transitive_prerequisites(node_name)) + [node_name]
    if node_name not in ("eval", "report") and "eval" in NODE_FNS and "eval" not in plan:
        plan.append("eval")
    if node_name != "report" and "report" in NODE_FNS and "report" not in plan:
        plan.append("report")
    return plan


def build_run_plan_context(*, node_name: str) -> _RunPlanContext:
    """Build reusable plan topology for a target node."""
    if node_name not in NODE_FNS:
        valid = ", ".join(sorted(NODE_FNS))
        raise ValueError(f"Unknown node '{node_name}'. Valid options: {valid}.")

    plan_list = _plan_nodes(node_name)
    plan = tuple(plan_list)
    plan_index = {step: idx for idx, step in enumerate(plan)}
    direct_prereqs = _plan_direct_prerequisites(plan_list)
    dependents: dict[str, list[str]] = {step: [] for step in plan}
    for child, parents in direct_prereqs.items():
        for parent in parents:
            dependents[parent].append(child)
    plan_transitive_prereqs = {
        step: _plan_transitive_prerequisites(
            node_name=step,
            plan=plan_list,
            direct_prereqs=direct_prereqs,
        )
        for step in plan
    }

    return _RunPlanContext(
        node_name=node_name,
        plan=plan,
        plan_index=plan_index,
        direct_prereqs=direct_prereqs,
        dependents={step: tuple(children) for step, children in dependents.items()},
        plan_transitive_prereqs=plan_transitive_prereqs,
    )
