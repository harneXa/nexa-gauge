<!-- pr-snapshot: 74d92dcea30baa791334f8aff3eed3c46a9210e6 -->

# PR: Full Refactor — CLI Decomposition, Graph Overhaul, Package Consolidation
**Branch:** `full-refactor` → `main`
**Date:** 2026-04-12

## Summary

Decomposes the CLI monolith into focused submodules, consolidates the package architecture by removing three unused packages (`lumiseval-api`, `lumiseval-ingest`, `lumiseval-evidence`), and substantially rewrites the graph execution layer. Introduces a `CachedNodeRunner` with per-node key-value caching, restructures all metric nodes into typed subpackages (geval, redteam, reference), and adds a `scanner.py` node and `topology.py` for explicit graph topology management. Expands test coverage across all layers with dedicated test suites for every node, metric, and execution path.

## What Changed

### CLI (`apps/lumiseval-cli`)
- Splits the `main.py` monolith (~300 lines) into three dedicated modules: `cli/run.py` (eval execution), `cli/estimate.py` (cost estimation), and `cli/util.py` (shared helpers and defaults)
- `main.py` is now a thin compatibility shim that re-exports the split commands for backward-compatible programmatic callers
- Moves dataset adapters (`huggingface.py`, `local_file.py`) directly into the CLI package under `adapters/`, eliminating the dependency on the removed `lumiseval-ingest` package

### Graph (`packages/lumiseval-graph`)
- Renames and significantly extends `node_runner.py` → `runner.py` as `CachedNodeRunner`: per-node cache read/write, concurrent record execution via `ThreadPoolExecutor`, streaming result yields
- Extracts `topology.py` from `graph.py` to own graph topology definition (`METRIC_NODES`, `NODES_BY_NAME`)
- Adds `nodes/base.py` as a typed base contract for all node functions
- Adds `nodes/scanner.py` (~310 lines): document scanning and chunking as an explicit first-stage graph node
- Adds `nodes/chunk_extractor.py`, `nodes/dedup.py`, `nodes/report.py`
- Restructures `metrics/geval/` into a full subpackage: `cache.py`, `score.py`, `steps.py` (was a single file)
- Restructures `metrics/redteam/` into a subpackage: `redteam.py`, `bias.py`, `toxicity.py`
- Adds `metrics/reference.py` for reference-based scoring
- Removes `cost_estimator.py`, `eval.py`, `metrics/base.py`, `metrics/rubric.py`, `metrics/token_utils.py`
- Pins `litellm==1.82.6`; adds `rouge-score`, `nltk`, `semchunk` dependencies; drops `lumiseval-ingest` and `lumiseval-evidence` deps

### Core (`packages/lumiseval-core`)
- Significantly extends `cache.py` with `build_node_cache_key`, `cache_read_allowed`, `cache_write_allowed`, `compute_case_hash`
- Adds `geval_cache.py` for GEval-specific cache logic
- Adds `utils.py` with shared utilities
- Refactors `types.py` (field renames and schema updates)
- Moves `dedup/mmr.py` into core from the removed evidence package

### Removed Packages
- **`lumiseval-api`**: FastAPI REST server removed (3 files deleted, package deregistered)
- **`lumiseval-ingest`**: Adapter and scanner logic consolidated into CLI; package deleted
- **`lumiseval-evidence`**: LanceDB indexer and Tavily router removed; package deleted

### Documentation
- Major rewrite of `docs/architecture.md` (111 lines changed)
- New `docs/cli-code-flow.md` (222 lines): documents the CLI command dispatch and adapter flow
- New `docs/execution-model.md` (56 lines): documents the `CachedNodeRunner` execution model
- Substantial rewrite of `docs/get-started.md` (~430 lines)

### Tests
- New CLI tests: `test_llm_cli_interface.py`, `test_local_adapter_streaming.py`
- New core tests: `test_cache.py` (expanded), `test_geval_cache.py`
- New graph tests covering: end-to-end metric routes, estimate mode, LLM routing, runner streaming, graph overrides, all metric nodes (geval, redteam, grounding, reference, relevance), scanner, chunk extractor, dedup, LLM config and gateway

## Why These Changes

- Splitting the CLI monolith reduces cognitive load when extending or testing individual commands — `run`, `estimate`, and shared utilities are now independently readable and testable
- Consolidating `lumiseval-ingest` adapters into the CLI eliminates an unnecessary abstraction layer that had no callers outside of the CLI
- Removing `lumiseval-evidence` and `lumiseval-api` cuts dead weight: the LanceDB indexer and FastAPI server had no active integration points in the current execution model
- Extracting `topology.py` decouples graph shape from graph execution, making it straightforward to add or remove nodes without touching the runner
- The `CachedNodeRunner` replaces the previous stateless runner with per-node caching, enabling incremental re-evaluation without re-running the full pipeline on every call
- Restructuring metrics into subpackages (geval, redteam) reflects their growing complexity and makes individual metric logic navigable

## Test Plan

- [ ] Run test suite: `make test`
- [ ] Run graph-specific tests with output: `make test_graph`
- [ ] Manual: run `uv run lumiseval run --help` and confirm CLI commands are intact
- [ ] Manual: run `uv run lumiseval estimate --help` and confirm estimate command is intact
- [ ] Manual: execute a sample eval run using `sample.json` to confirm the full pipeline executes end-to-end

**Test files added/modified:**
- `test_lumiseval_cli/test_llm_cli_interface.py` — CLI run/estimate command integration
- `test_lumiseval_cli/test_local_adapter_streaming.py` — local file adapter streaming behavior
- `test_lumiseval_core/test_cache.py` — expanded node cache key and write-guard logic
- `test_lumiseval_core/test_geval_cache.py` — GEval cache round-trips
- `test_lumiseval_graph/test_graph/test_runner_streaming.py` — `CachedNodeRunner` streaming and concurrency
- `test_lumiseval_graph/test_graph/test_estimate_mode.py` — cost estimation path
- `test_lumiseval_graph/test_graph/test_end_metric_routes.py` — full graph route coverage
- `test_lumiseval_graph/test_graph/test_llm_routing.py` — LLM gateway routing
- `test_lumiseval_graph/test_graph/test_run_graph_overrides.py` — user LLM overrides
- `test_lumiseval_graph/test_nodes/` — per-node tests (scanner, chunk extractor, dedup, all metric nodes)

## Notes for Reviewer

Three full packages were deleted intentionally:
- **`lumiseval-api`** (`apps/lumiseval-api/`): The FastAPI server was a stub and had no callers. The `Makefile` still has an `api:` target pointing to it — this should be removed in a follow-up or the API resurrected as a proper package.
- **`lumiseval-ingest`** (`packages/lumiseval-ingest/`): All active adapter logic has been moved into `lumiseval-cli/adapters/`. The package's test files (`test_dataset_adapters.py`, `test_scanner.py`) are deleted alongside it.
- **`lumiseval-evidence`** (`packages/lumiseval-evidence/`): The LanceDB + Tavily integration was not wired into the main execution path. If external document retrieval is added back, it should be a fresh design.

The root `pyproject.toml` no longer references `lumiseval-ingest` or `lumiseval-evidence` as workspace members — confirm the workspace `members = ["packages/*", "apps/*"]` glob is still correct given the deletions.
