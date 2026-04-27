"""Shared helpers for top-level metric verdict normalization."""

from __future__ import annotations

PASSED_VERDICT = "PASSED"
FAILED_VERDICT = "FAILED"


def verdict_from_passed(passed: bool) -> str:
    """Map a boolean metric outcome to the standardized verdict label."""
    return PASSED_VERDICT if passed else FAILED_VERDICT


def verdict_from_score(score: float | None, threshold: float) -> str | None:
    """Derive standardized verdict from a numeric score using a pass threshold."""
    if score is None:
        return None
    return verdict_from_passed(score >= threshold)
