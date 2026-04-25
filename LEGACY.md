# Legacy / Deprecated Code

Symbols and flags still present for backward compatibility. Each entry lists the condition under which removal is safe.

## `packages/nexagauge-core/ng_core/types.py::LegacyGEvalMetric`
- **Status:** Kept for cache/tests/adapters compatibility.
- **Evidence:** line 49 docstring — "Legacy GEval metric shape used by cache/tests/adapters."
- **Safe to remove when:** all cached artifacts using the old shape have expired, and tests/adapters have been migrated to the new GEval shape.

## `packages/nexagauge-core/ng_core/types.py::LegacyGEvalConfig`
- **Status:** Kept for cache/tests/adapters compatibility.
- **Evidence:** line 58 docstring — "Legacy GEval config shape used by cache/tests/adapters."
- **Safe to remove when:** same condition as `LegacyGEvalMetric`.

## `apps/nexagauge-apps/ng_cli/main.py::run` / `estimate` wrappers
- **Status:** Backward-compatible callables used by tests and programmatic callers.
- **Evidence:** lines 49-64 docstrings — "Backward-compatible callable used by tests and programmatic callers."
- **Safe to remove when:** tests import the typer commands directly (`from ng_cli.run import run as run_command`) instead of these wrappers.

## `apps/nexagauge-apps/ng_cli/run.py` flag `--yes` / `-y`
- **Status:** Accepted but deprecated — `run` no longer prompts.
- **Evidence:** line 94 help text — "Deprecated: run executes immediately and no longer needs confirmation."
- **Safe to remove when:** next minor version bump; users have had one release cycle of deprecation warnings.

## `apps/nexagauge-apps/ng_cli/run.py` flags `--web-search`, `--evidence-threshold`
- **Status:** Accepted but currently unused.
- **Evidence:** lines 88-89 help text.
- **Safe to remove when:** decision is made to either wire them up or drop them (see `TODO.md`).

## `apps/nexagauge-apps/ng_cli/util.py` parameter `legacy_model`
- **Status:** Backward-compat alias allowing `--model` and `--llm-model` to be used interchangeably.
- **Evidence:** lines 149, 167-175.
- **Safe to remove when:** the `--model` flag is removed from `run` and `estimate`.
