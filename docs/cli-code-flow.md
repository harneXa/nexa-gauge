# CLI Code Flow

This document describes the actual command flow for:
- `nexagauge run <node_name>`
- `nexagauge estimate <node_name>`

Source files:
- `apps/nexagauge-cli/ng_cli/main.py`
- `apps/nexagauge-cli/ng_cli/cli/run.py`
- `apps/nexagauge-cli/ng_cli/cli/estimate.py`
- `apps/nexagauge-cli/ng_cli/cli/util.py`

## 1) Command Entry

`main.py` registers two Typer commands:
- `run`
- `estimate`

```text
nexagauge
  ├── run <node_name> ...
  └── estimate <node_name> ...
```

Both commands share helper utilities for:
- target node validation
- branch planning
- model override parsing
- routing summary rendering

## 2) Shared Pre-Execution Steps

Both commands execute this setup sequence:

1. Validate target node (`_resolve_target_node`) against `NODES_BY_NAME`.
2. Parse and normalize model/fallback overrides (`_resolve_runtime_llm_overrides`).
3. Print branch LLM routing table (`_print_llm_routing_summary`).
4. Build cache store:
   - `CacheStore` (default)
   - `NoOpCacheStore` when `--no-cache`
5. Build `CachedNodeRunner`.
6. Resolve adapter and iterate selected rows:
   - `create_dataset_adapter(source, adapter, config_name, revision)`
   - `iter_cases(split=..., limit=...)`
   - apply `start/end/limit` slicing
   - inject per-case `llm_overrides`

## 3) `nexagauge run` Flow

File: `cli/run.py`

```text
run()
  ├─ resolve target + model routing
  ├─ adapter.iter_cases(...)
  └─ runner.run_cases_iter(... execution_mode="run")
       ├─ per case success:
       │    - accumulate executed/cached counters
       │    - if --output-dir and final_state has "report": write JSON
       └─ per case failure:
            - capture case_id + error
```

### run output behavior
- Always prints final summary counts:
  - total cases
  - succeeded / failed
  - executed steps / cached steps
- If `--output-dir` is provided, one JSON file is written per case **only when** `final_state["report"]` exists.

Practical implication:
- Targets like `grounding`, `relevance`, `redteam`, `geval`, `reference`, and `eval` usually produce `report`.
- Intermediate targets like `scan`, `chunk`, `claims`, `dedup` typically do not.

## 4) `nexagauge estimate` Flow

File: `cli/estimate.py`

```text
estimate()
  ├─ resolve target + model routing
  ├─ adapter.iter_cases(...)
  └─ runner.run_cases_iter(... execution_mode="estimate")
       ├─ aggregate per-node estimated_costs
       ├─ aggregate per-node stats (executed/cached/estimated/eligible)
       └─ render estimate summary table + total estimate
```

### estimate output columns
Rendered table includes per branch node:
- node_name
- model
- status
- cached / uncached counts
- uncached eligible count and percent
- cost_estimate

Status is derived from node cost and execution stats:
- `billable`
- `zero_cost`
- `cached_only`
- `skipped/ineligible`
- `not_reached`
- `failed`

## 5) Adapter Resolution Details

File: `adapters/registry.py`

`create_dataset_adapter(...)` rules:
- `adapter=local` -> local file adapter
- `adapter=huggingface` -> hf adapter
- `adapter=auto`:
  - `hf://...` source -> hf adapter
  - existing path -> local adapter
  - else -> input parse error

Local adapter supports:
- `.json` (object or array of objects)
- `.jsonl`
- `.csv`
- fallback to plain text single-case (`{"generation": <file text>}`)

HF adapter requires `datasets` package.

## 6) CLI Flags That Matter Most

Common flags:
- target/data: `node_name`, `--input`, `--adapter`, `--split`, `--start`, `--end`, `--limit`
- model routing: `--model`, `--llm-model`, `--llm-fallback`
- cache/execution: `--force`, `--no-cache`, `--cache-dir`, `--max-workers`, `--max-in-flight`, `--continue-on-error`

Run-only:
- `--output-dir`
- `--yes` (deprecated no-op)
- `--web-search`, `--evidence-threshold` are currently accepted but not used in `run` implementation.

## 7) Where CLI Meets Graph Execution

`CachedNodeRunner` is the bridge from CLI to graph nodes.

It is responsible for:
- initial `EvalCase` state construction
- prerequisite planning
- cache reads/writes
- estimate-mode run-cache fallback
- parallel metric execution for eval target
- ordered streaming outcomes

For deeper runner semantics, see `docs/execution-model.md`.
