# Latency Analysis (`ng_core` + `ng_graph`)

## Scope
This analysis covers:
- `packages/nexagauge-core/ng_core`
- `packages/nexagauge-graph/ng_graph`

It is a code-path analysis (not runtime-profiler output), focused on where latency and throughput are most likely constrained.

## Executive Summary
The slowest parts are dominated by LLM-call orchestration shape, not pure Python compute:

1. Metric fan-out is only parallelized when target node is exactly `eval` (`runner.py:541`). If callers run `report`, metric nodes can execute serially and increase per-case latency.
2. Claims extraction is one LLM call per chunk in a serial loop (`claim_extractor.py:54-55`).
3. Redteam runs one LLM call per metric in a serial loop (`redteam.py:301-302`, `_evaluate_metric` at `redteam.py:190`).
4. GEval scoring is already parallel per metric (`geval/score.py:322`), but GEval step generation is per-metric looped (`geval/steps.py:270`) and only fast if cache hits (`geval/steps.py:294`).
5. Dedup uses embedding + pairwise similarity loops (`mmr.py:47`, `mmr.py:58+`) and can become CPU-heavy as claim count grows.

## Slowest Processes (Ranked)

### 1) Metric phase can become serial for non-`eval` targets
- Evidence: `if node_name == "eval" ...` guard in `ng_graph/runner.py:541`.
- Impact: If target is `report`, you can lose parallel execution of `relevance/grounding/redteam/geval/reference`, turning `max(metric_times)` into approximately `sum(metric_times)`.
- Throughput effect: Large drop in cases/hour when metrics are LLM-bound.

### 2) Claims extraction serial LLM loop
- Evidence: `for chunk in chunks: response = llm.invoke(...)` in `ng_graph/nodes/claim_extractor.py:54-55`.
- Impact: Latency scales linearly with number of chunks.
- Throughput effect: Long generations (many chunks) create a hard per-case bottleneck.

### 3) Redteam serial per-metric loop
- Evidence: `for metric in metrics_to_run` then `_evaluate_metric` LLM call (`ng_graph/nodes/metrics/redteam/redteam.py:301-302`, `:206`).
- Impact: Latency scales linearly with number of redteam metrics.
- Throughput effect: Increased rubric count directly increases wall time.

### 4) GEval step generation on cache misses
- Evidence: loop over metrics (`geval/steps.py:270`), generate + write cache (`:316`), cache read (`:294`).
- Impact: First run for new criteria signatures can be slow; subsequent runs are much faster.
- Throughput effect: Cold-start penalty is significant in new datasets/configs.

### 5) CPU-heavy dedup for larger claim sets
- Evidence: embedding generation (`mmr.py:47`) + repeated cosine similarity in loop (`mmr.py:58+`).
- Impact: O(N²)-like behavior in pairwise scoring region.
- Throughput effect: noticeable when extracted claims per case become large.

### 6) Cache serialization/deserialization overhead
- Evidence: per-node read/write and JSON (de)serialization (`ng_core/cache.py:495`, `:507`, `:542`; runner read/write calls at `runner.py:387`, `:462`).
- Impact: Usually secondary vs LLM latency, but non-trivial at high case volume.
- Throughput effect: reduced worker efficiency on IO-bound hosts.

## What To Do First (Highest ROI)

1. Remove `node_name == "eval"` restriction for metric fan-out in runner.
2. Parallelize claims extraction with a bounded worker pool (or async semaphore).
3. Parallelize redteam metric evaluations similarly.
4. Ensure caller sets `run_cases_iter(max_workers>1)` (`runner.py:718` default is `1`).
5. Keep GEval step cache warm (reuse criteria + item_fields signatures).

## LLM Calls: What You Can and Cannot Improve
You cannot fully control provider-side model latency, but you can reduce end-to-end latency by reducing call count and prompt/token size:

- Add explicit `max_tokens` to structured judge calls in `llm/gateway.py` to cap slow long-tail completions.
- Keep prompts compact (especially grounding/context payloads).
- Prefer faster judge models for high-volume nodes (`claims`, `redteam`, `geval`) where quality permits.
- Retain cache-friendly stable prompts/config versions (avoid unnecessary prompt-version churn).

If provider latency spikes, your best defense is concurrency + caching + fewer/lighter calls.

## Suggested Measurement Plan (to validate improvements)
1. Add per-node timing counters in `CachedNodeRunner.run_case`.
2. Benchmark cold-cache vs warm-cache.
3. Benchmark `target=eval` vs `target=report` to confirm metric fan-out behavior.
4. Benchmark case-level workers (`1, 2, 4, 8`) to find saturation point.
5. Track p50/p95 per node and per case before/after each optimization.
