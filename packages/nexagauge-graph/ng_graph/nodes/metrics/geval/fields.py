"""Shared display-name mapping for G-Eval item fields.

The step-generation and scoring prompts both refer to case fields by
human-readable names. Keeping the map in one module guarantees the two prompts
stay consistent — if scoring renames ``"Actual Output"`` to ``"Generation"``,
step generation follows automatically.
"""

from __future__ import annotations

FIELD_DISPLAY_NAMES: dict[str, str] = {
    "question": "Input",
    "generation": "Actual Output",
    "reference": "Expected Output",
    "context": "Context",
}


def format_param_names(item_fields: list[str]) -> str:
    """Render item_fields as a human-readable, order-preserving comma list."""
    return ", ".join(FIELD_DISPLAY_NAMES.get(f, f) for f in item_fields)
