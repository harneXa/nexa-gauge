# Execution Model

This document describes how execution works today in `packages/lumiseval-graph/lumiseval_graph/runner.py`.

## 1) Core Executor

`CachedNodeRunner` executes cases node-by-node with cache awareness.

Two public paths:
- `run_case(...)` for one case
- `run_cases_iter(...)` for streaming many cases with optional concurrency

## 2) Initial State Construction

Each case is converted into an `EvalCase`-shaped state with:
- `record` (`case_id`, `generation`, `question`, `reference`, `context`, `geval`, `redteam`)
- `target_node`
- `execution_mode` (`run` or `estimate`)
- `estimated_costs` (empty dict)
- `node_model_usage` (empty dict)
- optional `llm_overrides` and `reference_files`

## 3) Plan Construction

Node plan is:
- `prerequisites(target_node) + [target_node]`
- plus `report` when target is `eval` or a metric node

Metric node set (from topology):
- `relevance`, `grounding`, `redteam`, `geval`, `reference`

## 4) Cache Keying and Fingerprints

Cache key is built from:
- case fingerprint (`compute_case_hash`)
- execution mode
- node name
- route fingerprint (dependency path + node routing config)

Route fingerprint includes model-routing inputs via `get_node_config`:
- resolved model
- fallback model
- temperature

This means model-routing changes invalidate cache for affected nodes.

## 5) Cache Read/Write Policy

Policy in `lumiseval_core/cache.py`:
- reads allowed in both `run` and `estimate`
- writes allowed in `run`
- estimate writes are disabled by default
- `eval` and `report` are never cacheable

Estimate fallback behavior:
- if estimate key misses, runner attempts run-mode cache lookup for same node route

## 6) Execution Semantics

### Sequential path
For non-eval targets, steps run in plan order:
1. optional cache hit path
2. else execute `NODE_FNS[step](state)`
3. merge patch into state
4. optional cache write

### Eval metric fan-out path
For `target_node == "eval"`:
- when the plan reaches first metric node, all metric siblings are processed together
- cache hits are applied first
- misses run concurrently in `ThreadPoolExecutor`
- each metric task receives `deepcopy(state)` to avoid shared mutation races
- resulting patches are merged back in stable metric order

## 7) State Merge Rules

Patch merge is key-aware:
- `estimated_costs` merges by dict update
- `node_model_usage` merges by dict update
- all other keys overwrite directly

This allows each node to contribute partial cost/model info without losing prior entries.

## 8) Batch Streaming (`run_cases_iter`)

`run_cases_iter` supports:
- single-worker deterministic loop
- multi-worker pool with bounded in-flight submissions
- ordered emission by original input index

Fail-fast behavior:
- when `continue_on_error=False`, stream stops after first ordered failure is emitted

## 9) Estimate vs Run

`execution_mode="run"`:
- real node execution, full writes to cache where allowed

`execution_mode="estimate"`:
- node estimate logic populates `estimated_costs`
- run-mode cache can satisfy unchanged prerequisite work
- CLI aggregates estimated costs into branch summary tables

## 10) Report Node Placement

`report` is appended automatically for metric/eval targets by runner planning logic.

Report generation is state-projection only (`report.aggregate`) and does not perform LLM calls.
