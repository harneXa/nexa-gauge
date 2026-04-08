## Short Summary

- Added a new graph-focused test suite under `test_graph`.
- Created shared fixtures in `conftest.py` for simple metric creation and stable `graph` module import.
- Added `test_end_metric_routes.py` with branch tests for:
  - `grounding`
  - `relevance`
  - `reference`
  - `geval`
  - `redteam`
  - `eval`
- Each branch test simulates state, calls `node_eval` then `node_report`, and asserts expected metrics in final `report`.
- Added a small assertion that `node_eval` is orchestration-only.
- Kept tests simple, deterministic, and free of external API calls.
- Verified by running:
  - `uv run pytest -s packages/lumiseval-graph/test_lumiseval_graph/test_graph/test_end_metric_routes.py`
  - Result: all tests passed.
