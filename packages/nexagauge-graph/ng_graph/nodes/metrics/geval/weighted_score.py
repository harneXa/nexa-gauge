"""G-Eval probability-weighted expected-score calculation.

Pure math with no LLM dependencies. Given the integer score the judge committed
to and the provider's per-token logprobs, compute E[score] across the decimal
tokens the model almost-chose. Normalization to [0, 1] is the caller's job.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

PROB_FLOOR = 0.01
LOG_PROB_FLOOR = math.log(PROB_FLOOR)


def calculate_weighted_summed_score(
    raw_score: int,
    content_tokens: Sequence[Mapping],
    *,
    score_min: int = 1,
    score_max: int = 10,
) -> float:
    """Return the logprob-weighted expected score, or ``raw_score`` if no signal.

    ``content_tokens`` is the provider-agnostic form of LiteLLM's
    ``choice.logprobs.content``: a sequence of dict-like entries with ``token``,
    ``logprob``, and ``top_logprobs`` (list of ``{token, logprob}``).
    """

    target = str(raw_score)
    score_entry: Mapping | None = None
    for entry in content_tokens:
        token = str(entry.get("token", "")).strip()
        if token == target:
            score_entry = entry
            break

    if score_entry is None:
        return float(raw_score)

    candidates: list[tuple[int, float]] = []
    for alt in score_entry.get("top_logprobs", []) or []:
        token = str(alt.get("token", "")).strip()
        if not token.isdecimal():
            continue
        value = int(token)
        if value < score_min or value > score_max:
            continue
        logprob = float(alt.get("logprob", LOG_PROB_FLOOR - 1.0))
        if logprob < LOG_PROB_FLOOR:
            continue
        candidates.append((value, logprob))

    if not candidates:
        return float(raw_score)

    weighted_sum = 0.0
    total_prob = 0.0
    for value, logprob in candidates:
        prob = math.exp(logprob)
        weighted_sum += value * prob
        total_prob += prob

    if total_prob <= 0.0:
        return float(raw_score)

    return weighted_sum / total_prob
