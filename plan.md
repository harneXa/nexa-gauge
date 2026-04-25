## Low-Latency Runner + Debug Timing Plan

  ### Summary

  - Implement all requested optimizations in one pass: record-level parallelism remains, metric-
    branch parallelism is generalized, and inner-node serial LLM loops are parallelized for
    claims, redteam, and geval_steps.
  - Add per-node timing counters in CachedNodeRunner.run_case and expose both per-case debug
    timing and cross-case “slowest nodes” summaries when --debug is enabled.
  - Use env-driven inner-node concurrency controls with default cap 8 (your selected profile).

  ### Implementation Changes

  - Runner scheduling: Update ng_graph.runner.CachedNodeRunner.run_case to parallelize metric
    nodes whenever the metric segment is present in the plan and prerequisites are already
    satisfied, not only when node_name == "eval". This must apply to report target paths too.
  - Runner timing model: Add per-node timing capture in run_case for each step with explicit
    fields: node, executed_count, cached_count, execute_ms_total, cache_lookup_ms_total,
    last_execute_ms, last_cache_lookup_ms. Capture timing for both normal-step path and metric
    fan-out path.
  - Runner result interface: Extend CachedNodeRunResult with a timing payload (e.g.
    node_timings) so CLI can aggregate cross-case hotspot stats without recomputation.
  - Debug behavior: Under debug=True, print per-case node timing lines from runner and include
    cache-hit vs executed distinction.
  - CLI debug summary: In both run and estimate, aggregate result.node_timings across outcomes
    and print a final sorted slow-node table (descending by execute_ms_total) with columns:
    node, executed, cached, total execute ms, avg execute ms.
  - Claims parallelism: In ClaimExtractorNode.run, replace serial chunk loop with bounded
    parallel execution using env cap NEXAGAUGE_CLAIMS_MAX_CONCURRENCY (default 8). Preserve
    output order by chunk index and preserve existing cost aggregation semantics.
  - Redteam parallelism: In RedteamNode.run, evaluate metrics concurrently with env cap
    NEXAGAUGE_REDTEAM_MAX_CONCURRENCY (default 8). Preserve metric output order as input metric
    order.
  - GEval steps parallelism: In GevalStepsNode.run, concurrently generate steps only for metrics
    that are neither provided nor cache-hit, with env cap NEXAGAUGE_GEVAL_STEPS_MAX_CONCURRENCY
    (default 8). Keep cache-read/provided paths immediate and final resolved list in original
    input order.
  - Thread-safe model usage accounting: For nodes parallelized internally, avoid mutating shared
    usage counters from worker threads. Collect per-task model usage locally and finalize once
    after join via deterministic aggregation.
  - Configuration surface: Add the three concurrency env vars to config (ng_core.config)
    and .env.example. No CLI flags added.

  ### Test Plan

  - Runner metric fan-out for report: Add test proving node_name="report" executes metric
    segment in parallel (wall-time bound with artificial sleeps) and still merges outputs in
    deterministic metric order.
  - Runner timing payload: Add tests that run_case(..., debug=True) populates per-node timings
    for executed and cached paths, including metric fan-out execution.
  - CLI slow-node summary: Add tests for run/estimate debug aggregation helper to verify sort
    order and totals from multiple outcomes.
  - Claims concurrency: Add unit test with delayed fake LLM that verifies reduced wall-time vs
    ordering.
    ordering/signatures remain correct.
  - Regression: Ensure existing caching behavior tests pass unchanged, including estimate-mode
    reuse and non-cacheable eval/report behavior.

  ### Assumptions and Defaults

  - Default per-case inner-node caps are 8 for claims, redteam, and GEval-step generation.
  - Record-level concurrency (--max-workers, --max-in-flight) remains unchanged.
  - No cache key format changes are introduced.
  - Debug timing is observability-only; it does not alter execution/caching semantics.



















  Here is Claude's plan:                                                                           
╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌
 Runner latency optimization + per-node timing                                                    
                                                        
 Context                                                                                        

 packages/nexagauge-graph/ng_graph/runner.py executes the eval graph serially within a case,
 except for the final metric fan-out — and even that fan-out only triggers when the target is
 exactly eval (guard at runner.py:541). Several metric nodes also have internal serial LLM loops.
  The user already authored latency.md today (Apr 21, 2026) identifying these bottlenecks; this
 plan implements its recommendations plus per-node timing.

 Current state (from exploration):

 - Cross-record parallelism exists at runner.py:793 via ThreadPoolExecutor(max_workers=workers);
 CLI flag --max-workers defaults to 1 (run.py:104-109).
 - Metric fan-out exists at runner.py:589 with per-future deepcopy(state), but is gated to
 node_name == "eval" (runner.py:541). For target=report the metrics run serially.
 - Utility chain runs strictly sequentially (runner.py:538-697).
 - Topology (topology.py:85-205): scan → chunk → claims → dedup → {relevance, grounding};
 independent branches scan → redteam, scan → reference, and scan → geval_steps → geval.
 Redteam/reference/geval_steps do not need dedup.
 - Per-item serial loops: claims per-chunk (claim_extractor.py:54-63), redteam per-metric
 (redteam.py:301-308), geval_steps per-metric with cache check (geval/steps.py:270-336).
 - Observability: whole-case elapsed_ms only (runner.py:514, 699). --debug flag toggles per-node
 start-of-execution log lines via NodeLogger (run.py:127-135, log.py).
 - LLM client (gateway.py): sync StructuredLLM.invoke wrapping litellm.completion; no async
 entrypoint — but thread-safe in practice, which is how GEval scoring already uses
 asyncio.to_thread.

 Why thread-based: the runner is already threaded end-to-end; LiteLLM sync calls are thread-safe;
  matching the existing pattern avoids a disruptive async refactor.

 Approach

 Three independent layers, each deliverable on its own:

 1. Observability first — per-node timing under existing --debug. Instrument before optimizing so
  phases 2/3 have measured deltas.
 2. DAG-level scheduler — replace the linear plan walker with a scheduler that submits each node
 as soon as its prereqs complete. Drops the node_name == "eval" guard. Lets
 redteam/reference/geval_steps/geval run concurrently with the claims→dedup chain.
 3. Intra-node parallelism — bounded thread pools inside claims, redteam, and geval_steps to
 remove per-item serial loops.

 A small safety net (Phase 4) bounds total in-flight LLM calls so the multiplied concurrency
 doesn't blow provider rate limits.

 ---
 Phase 1 — Per-node timing under --debug

 Files:
 - packages/nexagauge-graph/ng_graph/runner.py (CachedNodeRunResult at :47-72; run_case at
 :473-714)
 - apps/nexagauge-apps/ng_cli/run.py (:192-241, end-of-run summary block)

 Changes:
 1. Extend CachedNodeRunResult with node_timings: dict[str, float] (ms per executed node,
 cache-hits recorded as 0.0).
 2. In run_case, wrap the two execution sites with time.monotonic():
   - Single-node call at runner.py:682 (output = NODE_FNS[step](state)).
   - Each future inside the metric fan-out at runner.py:593-614.
 3. When debug=True, emit get_node_logger(step).info(f"elapsed {ms:.1f}ms") right after
 completion. Continue to honor DEBUG_SKIP_NODES (topology.py:221) for log output — but always
 record timings into the dict so they surface in the summary.
 4. In run.py after the outcome loop (around :231), when debug=True, print a Rich table of
 per-node p50/p95/sum across all successful outcomes. Reuse console from run.py:24.

 No new CLI flag. This extends the existing --debug.

 ---
 Phase 2 — DAG-level scheduler

 Files:
 - packages/nexagauge-graph/ng_graph/runner.py (replace run_case body at :538-697)
 - Reuse transitive_prerequisites at topology.py:224-242 to build the working set.

 Scheduler shape:

 - Build the plan as the set {target} ∪ transitive_prerequisites(target) (ordered by NODE_ORDER
 from topology.py:212 for determinism).
 - Maintain a single ThreadPoolExecutor(max_workers=max(len(METRIC_NODES), 4)) for the case.
 - For each node n, a future is submittable once every name in NODES_BY_NAME[n].prerequisites has
  resolved (either via cache hit or future completion).
 - Each submission receives deepcopy(state_at_submission_time) — the same isolation technique
 already used at runner.py:593. This keeps every node reading a stable view of its transitive
 prereqs' merged patches.
 - On completion: merge the patch into the canonical state in NODE_ORDER to keep ordering
 deterministic (buffer completions that arrive out of topo order, drain in order).
 - Cache-hit nodes resolve synchronously before submission — they never occupy a worker.
 - Drops the node_name == "eval" guard at runner.py:541 entirely. The eval node becomes a normal
 DAG node whose prereqs (the metric set) happen to be leaf-ish.

 Cache fingerprint rule (the delicate bit):

 The current linear chain computes step_fingerprint = H(parent_fingerprint, node, state, mode)
 (runner.py:644-655). In a DAG:

 - step_fingerprint(n) = H(merge_fingerprint(prereqs of n), n, state_view, mode) where
 merge_fingerprint(ps) = H(sorted([step_fingerprint(p) for p in ps])) — deterministic
 combination.
 - For nodes with a single linear chain (e.g. scan → chunk → claims → dedup), this must reduce to
  the same fingerprint the current code produces. Verify this in Phase 2 tests before removing
 the old code path. If the reduction doesn't hold exactly, we either (a) add a compat shim
 preserving linear-chain fingerprints literally, or (b) accept a one-time cache invalidation —
 user's call.

 Preserved invariants:
 - Output merge into final state follows NODE_ORDER (stable, deterministic).
 - executed/cached lists report in completion order but rendered in NODE_ORDER at the top of
 CachedNodeRunResult.
 - case_fingerprint computation unchanged.
 - _read_step_cache_with_run_fallback / _write_step_cache (runner.py:420-471) unchanged — only
 the orchestration around them changes.

 Expected wall-clock win: for target=report with cold cache, independent branches (redteam,
 reference, geval_steps, geval_steps→geval) now overlap with
 chunk→claims→dedup→{grounding,relevance}. Upper bound: wall time drops from sum(all nodes) to
 max(longest_chain).

 ---
 Phase 3 — Intra-node parallelism

 3A. Claims per-chunk fan-out

 File: packages/nexagauge-graph/ng_graph/nodes/claim_extractor.py:46-100

 - Replace the for chunk in chunks: loop at :54-92 with a bounded ThreadPoolExecutor fan-out.
 - Worker count: min(len(chunks), CLAIMS_MAX_WORKERS) with CLAIMS_MAX_WORKERS = 8 as a module
 constant.
 - Preserve ordering: collect (chunk.index, response) tuples, sort by index before flattening
 into all_claims. The existing source_chunk_index field (:89) stays correct by construction.
 - _record_model_response (:64) mutates self._model_usage — make that block run under a
 threading.Lock on the node instance, since concurrent LLM completions will race the usage
 accumulator. Same for costs.append (:74).

 3B. Redteam per-metric fan-out

 File: packages/nexagauge-graph/ng_graph/nodes/metrics/redteam/redteam.py:280-326

 - Replace the for metric in metrics_to_run: loop at :301-317 with a bounded ThreadPoolExecutor.
 - Worker count: min(len(metrics_to_run), REDTEAM_MAX_WORKERS) with REDTEAM_MAX_WORKERS = 8.
 - Preserve results order: use executor.map or index-ordered future collection so results align
 with metrics_to_run.
 - _record_model_response is invoked inside _evaluate_metric at :225 — protect with a lock
 identical to 3A.
 - The per-metric log.info(...) at :313-317 can fire in completion order — acceptable.

 3C. GEval steps per-metric fan-out on cache miss

 File: packages/nexagauge-graph/ng_graph/nodes/metrics/geval/steps.py:260-347

 - First pass (sequential): iterate metrics at :270; handle metric.evaluation_steps branch
 (:273-286) and cache-hit branch (:296-307) inline — these are O(ms).
 - Second pass (concurrent): collect cache-miss entries; submit all
 _generate_steps(criteria_text, item_fields, metric.name) calls via ThreadPoolExecutor bounded by
  GEVAL_STEPS_MAX_WORKERS = 8.
 - Third pass (sequential): write cache (_write_cached_steps, :316-324) and append to resolved in
  original metric order. The cache write mutates disk and benefits little from parallelism; keep
 serial.
 - Cumulative cost/tokens accumulation at :312-314 must either run under a lock during concurrent
  phase, or (simpler) accumulate from the future results in the serial third pass.

 ---
 Phase 4 — Global LLM concurrency cap (safety net)

 File: packages/nexagauge-graph/ng_graph/llm/gateway.py (StructuredLLM.invoke at :90+)

 - Add a module-level threading.BoundedSemaphore. Wrap the body of invoke with with
 _LLM_SEMAPHORE: before self._call(...) (gateway.py:102).
 - Configurable via --llm-concurrency N CLI option (default 16) plumbed through run_command → a
 module setter set_llm_concurrency(n).
 - Why: after phases 2+3, a single record can have ~10 in-flight calls (claims chunks + redteam
 metrics + GEval steps + 2 direct metrics). Multiplied by --max-workers=4 this is ~40 in-flight.
 Without a cap, provider rate limits will 429.

 ---
 Verification

 1. Cold-cache benchmark (user runs):
 rm -rf .cache
 uv run ng run report -i <sample>.json -n 20 --debug --max-workers 4
 1. Compare end-of-run per-node p50/p95 table against a pre-change baseline saved to latency.md
 or a scratch file.
 2. Correctness parity: run target=report on 5 cases before and after each phase; diff the
 emitted JSON reports (_write_report_json at run.py:225). Deterministic nodes (reference, dedup)
 must byte-match; LLM nodes may differ only in non-deterministic fields (reasoning text) —
 numeric scores should match for same inputs + seeds.
 3. Metric fan-out applies to non-eval targets (Phase 2): run target=report --debug; in the
 summary table, confirm redteam+reference wall-clock overlaps claims+dedup (their per-node
 elapsed times sum to more than the case elapsed_ms).
 4. Cache fingerprint stability (Phase 2): run target=dedup twice; second run's
 cached=['scan','chunk','claims','dedup'] and executed=[]. If regression, apply the compat-shim
 path documented in Phase 2.
 5. Intra-node ordering (Phase 3): add one unit test per node asserting that running with 1
 worker vs N workers produces identical claims/results/resolved_steps lists for a fixed
 deterministic stub LLM.
 6. Existing test suite:
 uv run pytest packages/nexagauge-graph packages/nexagauge-core

 Risks

 - Cache fingerprint regression (Phase 2): highest-risk change. Mitigation: compat shim
 preserving linear-chain hashes; full test pass with an existing cache before merging.
 - Thread-safety of LLM usage accumulators (Phase 3): each node's _record_model_response and
 costs list must be lock-protected.
 - Provider rate limits (Phases 2+3 combined): Phase 4 semaphore is the mitigation; default 16 is
  conservative.
 - deepcopy(state) overhead growing with DAG fan-out (Phase 2): each node submission deepcopies
 state. For small state this is sub-ms; if state becomes large (e.g. many chunks), this is a
 future concern — flag but don't fix now.

 Out of scope

 - Async rewrite of StructuredLLM (.ainvoke) — would unify the threaded and asyncio paths but is
 a bigger refactor. Revisit if Phase 4 semaphore becomes a bottleneck.
 - Embedding-stage dedup CPU optimization (mmr.py:47, 58+, item #5 in latency.md) — out of scope
 for this runner change.