# NexaGauge Architecture

This document reflects the current implementation in:
- `apps/nexagauge-cli`
- `packages/nexagauge-core`
- `packages/nexagauge-graph`

## 1) System Overview

NexaGauge is a CLI-first evaluation system that executes a typed graph pipeline over dataset rows.

```mermaid
flowchart LR
  U[User] --> CLI[Typer CLI]
  CLI --> AD[Dataset Adapter\nlocal | huggingface]
  AD --> R[CachedNodeRunner]
  R --> G[Graph Node Functions]
  G --> S[EvalCase State]
  S --> REP[report.aggregate]
  REP --> OUT[Per-case JSON report]

  subgraph EXT[External dependencies]
    LLM[LiteLLM]
    DE[DeepEval]
    NLTK[ROUGE/BLEU/METEOR]
  end

  G -.-> LLM
  G -.-> DE
  G -.-> NLTK
```

## 2) Main Subsystems

### CLI (`apps/nexagauge-cli`)
- Commands:
  - `nexagauge run <node_name>`
  - `nexagauge estimate <node_name>`
- Data source adapters:
  - Local file adapter (`json`, `jsonl`, `csv`, text fallback)
  - Hugging Face adapter (`hf://...`)
- Runtime model routing:
  - global and per-node model/fallback override handling

### Core (`packages/nexagauge-core`)
- Shared typed contracts (`types.py`)
- Environment-backed config (`config.py`)
- Cache backend and key utilities (`cache.py`)
- Error types (`errors.py`)

### Graph (`packages/nexagauge-graph`)
- Node topology registry (`topology.py`)
- Node function registry (`registry.py`)
- Graph node implementations (`graph.py`, `nodes/*`)
- Cache-aware executor (`runner.py`)
- Report projection contract (`nodes/report.py`)

## 3) Data Contracts

### Input row (raw)
Scanner accepts multiple aliases and normalizes to `Inputs`.

Key normalized fields:
- `case_id`
- `generation` (required to run meaningful evaluation)
- `question` (optional)
- `context` (optional)
- `reference` (optional)
- `geval` (optional)
- `redteam` (optional)

### Graph state (`EvalCase`)
Important keys in runtime state:
- control: `target_node`, `execution_mode`, `llm_overrides`
- normalized input: `inputs`
- node artifacts:
  - `generation_chunk`
  - `generation_claims`
  - `generation_dedup_claims`
  - `grounding_metrics`
  - `relevance_metrics`
  - `redteam_metrics`
  - `geval_steps`
  - `geval_metrics`
  - `reference_metrics`
- cost/model bookkeeping:
  - `estimated_costs`
  - `node_model_usage`

### Output report
`report.aggregate(state=...)` builds report from declarative `REPORT_VISIBILITY`.

Always present:
- `target_node`
- `input`

Conditionally present (omitted when source artifact is `None`):
- `chunks`, `claims`, `claims_unique`, `geval_steps`, `grounding`, `relevance`, `redteam`, `geval`, `reference`

## 4) Node Topology and Eligibility

Canonical nodes:
- `scan`, `chunk`, `claims`, `dedup`, `geval_steps`, `relevance`, `grounding`, `redteam`, `geval`, `reference`, `eval`, `report`

Eligibility flags are declared in `topology.py` and enforced by node logic:
- `requires_generation`
- `requires_question`
- `requires_context`
- `requires_geval`
- `requires_reference`

Current practical gates in node execution:
- `chunk/claims/dedup`: generation required
- `relevance`: generation + question
- `grounding`: generation + context
- `geval_steps/geval`: generation + GEval metrics
- `reference`: generation + reference

## 5) Execution Paths

### `run`
- Builds initial state from raw row.
- Plans prerequisite chain + target.
- Executes with node-level cache reuse.
- For `target=eval`, metric nodes execute in parallel.
- Emits final per-case state; writes `report` JSON when present.

### `estimate`
- Uses same runner with `execution_mode="estimate"`.
- Executes node `estimate()` logic where available.
- Aggregates uncached estimated cost rows per node.
- Reuses run cache entries for matching routes.

## 6) Caching Model

Cache key is opaque and route-aware:
- case fingerprint (input content)
- execution mode
- node name
- node route fingerprint (includes model/fallback/temp route)

Policies:
- cache reads: allowed in run and estimate
- estimate may reuse run-mode cache
- `eval` and `report` are non-cacheable

## 7) Concurrency Model

Two layers of concurrency:
- record-level concurrency via `run_cases_iter(..., max_workers=...)`
- metric fan-out concurrency inside `run_case` when target is `eval`

Result ordering is preserved in streaming iterator output.

## 8) Current Boundaries

- API server package was removed; CLI is the active interface.
- Legacy ingest/evidence packages were removed; adapter and scanner logic now live in CLI/graph packages.
- `report` is now a projection of state, not a separate post-processing service.
