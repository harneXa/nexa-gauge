import numpy as np
from ng_core.dedup import mmr
from ng_core.types import Item


class FakeModel:
    def __init__(self, embeddings: dict[str, list[float]]) -> None:
        self.embeddings = embeddings

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        del show_progress_bar
        return np.array([self.embeddings[text] for text in texts], dtype=float)


def _item(text: str, confidence: float) -> Item:
    return Item(text=text, tokens=1, confidence=confidence)


def test_deduplicate_limits_unique_items_to_top_k(monkeypatch) -> None:
    monkeypatch.setattr(
        mmr,
        "_get_model",
        lambda: FakeModel(
            {
                "highest": [1.0, 0.0, 0.0],
                "middle": [0.0, 1.0, 0.0],
                "lowest": [0.0, 0.0, 1.0],
            }
        ),
    )

    selected_indices, dedup_map = mmr.deduplicate(
        [
            _item("lowest", 0.1),
            _item("highest", 0.9),
            _item("middle", 0.5),
        ],
        similarity_threshold=0.99,
        lmb=1.0,
        top_k=2,
    )

    assert selected_indices == [1, 2]
    assert dedup_map == {}


def test_deduplicate_top_k_keeps_duplicate_map_for_threshold_matches(monkeypatch) -> None:
    monkeypatch.setattr(
        mmr,
        "_get_model",
        lambda: FakeModel(
            {
                "representative": [1.0, 0.0],
                "duplicate": [1.0, 0.0],
                "distinct": [0.0, 1.0],
            }
        ),
    )

    selected_indices, dedup_map = mmr.deduplicate(
        [
            _item("representative", 0.9),
            _item("duplicate", 0.8),
            _item("distinct", 0.7),
        ],
        similarity_threshold=0.99,
        lmb=1.0,
        top_k=2,
    )

    assert selected_indices == [0, 2]
    assert dedup_map == {1: 0}


def test_deduplicate_top_k_zero_returns_no_items(monkeypatch) -> None:
    def fail_if_called() -> FakeModel:
        raise AssertionError("model should not be initialized for top_k=0")

    monkeypatch.setattr(mmr, "_get_model", fail_if_called)

    selected_indices, dedup_map = mmr.deduplicate([_item("one", 1.0)], top_k=0)

    assert selected_indices == []
    assert dedup_map == {}
