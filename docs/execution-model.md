# Execution Model: Sequential vs Parallel Work

This document traces how LumisEval executes today and separates three different questions:

- Are cases processed in parallel?
- Are pipeline nodes processed in parallel?
- Does a single node do internal concurrent work?

It describes the current repository code only.

## Source of Truth

These files define the current execution behavior:

- `apps/lumiseval-cli/lumiseval_cli/main.py`
- `apps/lumiseval-api/lumiseval_api/main.py`
- `packages/lumiseval-core/lumiseval_core/pipeline.py`
- `packages/lumiseval-graph/lumiseval_graph/node_runner.py`
- `packages/lumiseval-graph/lumiseval_graph/graph.py`
- `packages/lumiseval-graph/lumiseval_graph/nodes/metrics/rubric.py`

The key split is:

- CLI dataset execution uses `CachedNodeRunner.run_case(...)`
- API request execution uses `run_graph(...)`

## Pipeline Topology

The pipeline dependencies come from `NodeSpec.prerequisites` in `lumiseval_core/pipeline.py` and the LangGraph edges in `lumiseval_graph/graph.py`.

Canonical shape:

```text
scan
  -> chunk
      -> claims
          -> dedupe
              -> relevance
              -> grounding
  -> redteam
  -> rubric
  -> reference

all metric outputs -> eval
```

Dependency summary:

- `chunk` depends on `scan`
- `claims` depends on `scan`, `chunk`
- `dedupe` depends on `scan`, `chunk`, `claims`
- `relevance` depends on `scan`, `chunk`, `claims`, `dedupe`, and a question
- `grounding` depends on `scan`, `chunk`, `claims`, `dedupe`, and context
- `redteam` depends on `scan`
- `rubric` depends on `scan` and rubric rules
- `reference` depends on `scan` and a reference answer
- `eval` depends on every metric branch

Important implication:

- Dependent nodes cannot start until their prerequisites have produced state.
- A branch may be independent in the graph even if it is still executed sequentially by a specific runner.

## Execution Paths

### CLI path: `lumiseval run ...`

Entry path:

```text
apps/lumiseval-cli/lumiseval_cli/main.py
  run(...)
    -> _load_cases(...)
    -> _build_job_config(...)
    -> CachedNodeRunner(cache_store)
    -> for case in cases:
         runner.run_case(case=case, node_name=target_node, ...)
```

What is sequential at the CLI level:

- Cases are processed one at a time with `for case in cases:`
- The prerequisite chain is executed in order
- Non-`eval` targets never enter the metric fan-out block

Inside `CachedNodeRunner.run_case(...)`, the plan is built as:

```text
_plan_nodes(node_name)
  -> prerequisites + target node
```

That means:

- `lumiseval run relevance ...` executes `scan -> chunk -> claims -> dedupe -> relevance`
- `lumiseval run grounding ...` executes `scan -> chunk -> claims -> dedupe -> grounding`
- `lumiseval run eval ...` executes the full plan and may fan out the metric group

### API path: `POST /jobs`

Entry path:

```text
apps/lumiseval-api/lumiseval_api/main.py
  create_job(...)
    -> _run_one(request)
      -> run_graph(...)
        -> build_initial_state(...)
        -> build_graph().compile()
        -> graph.invoke(initial_state)
```

What is sequential at the API level:

- A list payload is processed with `[_run_one(item) for item in request]`
- The API code does not use `CachedNodeRunner`
- The API code does not contain the CLI thread-pool fan-out block

Important implication:

- The only explicit repo-local metric-node fan-out lives in `CachedNodeRunner.run_case(...)`
- The API path does not use that code path
- `run_graph(...)` is still a synchronous call boundary in this repo
- This document does not assume any additional LangGraph scheduling beyond what is visible in repository code

## Where Parallelism Exists

### 1. Case-level parallelism

Current behavior:

- CLI: no case-level parallelism
- API list payload: no request-level parallelism

Why:

- CLI loops with `for case in cases:`
- API list handling uses a plain list comprehension over `_run_one(...)`

### 2. Node-level parallelism across sibling metric nodes

Current behavior:

- Yes, but only in the CLI `CachedNodeRunner` path
- Only when `node_name == "eval"`

The explicit parallel fan-out is in `packages/lumiseval-graph/lumiseval_graph/node_runner.py`:

```text
if node_name == "eval" and step == "relevance":
  for metric_step in METRIC_NODES:
    classify as skipped / cached / to_run

  with ThreadPoolExecutor(max_workers=len(to_run)) as pool:
    futures = {pool.submit(NODE_FNS[m], dict(state)): m for m in to_run}
```

What this does:

1. Waits until upstream prerequisites have populated `state`
2. Builds the metric group from `METRIC_NODES`
3. Skips ineligible metric nodes
4. Reuses cache for already-computed metric nodes
5. Runs only uncached eligible metric nodes in a thread pool
6. Passes each worker a cloned `dict(state)` instead of sharing mutable state
7. Merges outputs back into the main state in stable metric order after completion

What this does not do:

- It does not parallelize `scan`, `chunk`, `claims`, or `dedupe`
- It does not parallelize different cases
- It does not run `eval` until all metric outputs are ready
- It does not apply when the CLI target is a single metric like `relevance`

### 3. Intra-node concurrency inside `rubric`

Current behavior:

- Yes, in both CLI and API paths if the rubric node runs

`RubricNode.run()` does its own concurrent work:

```text
async def _run_all():
  tasks = [self._evaluate_rule(rule, generation) for rule in rubric]
  return await asyncio.gather(*tasks)
```

Then each rule evaluation calls `run_in_executor(...)` around `g_eval.measure(...)`.

What this means:

- Multiple rubric rules may be evaluated concurrently inside the rubric node
- This is internal node concurrency, not graph-node concurrency
- It does not make `scan`, `chunk`, `claims`, `dedupe`, or `eval` run in parallel

## Worked Example

Use a record that enables every branch:

```json
{
  "case_id": "paris-1",
  "question": "What city is the Eiffel Tower in?",
  "generation": "The Eiffel Tower is in Paris, France.",
  "context": ["The Eiffel Tower is a wrought-iron tower in Paris."],
  "reference": "The Eiffel Tower is located in Paris.",
  "rubric": [
    {
      "id": "R-1",
      "statement": "Mentions Paris",
      "pass_condition": "The answer must mention Paris."
    },
    {
      "id": "R-2",
      "statement": "Mentions France",
      "pass_condition": "The answer must mention France."
    }
  ]
}
```

### Example A: CLI `lumiseval run eval --input sample.json`

Execution for one case:

```text
case loop
  -> scan                sequential
  -> chunk               sequential
  -> claims              sequential
  -> dedupe              sequential
  -> relevance           may overlap with other metric siblings
  -> grounding           may overlap with other metric siblings
  -> redteam             may overlap with other metric siblings
  -> rubric              may overlap with other metric siblings
       -> rule R-1       may overlap with rule R-2 inside rubric
       -> rule R-2       may overlap with rule R-1 inside rubric
  -> reference           may overlap with other metric siblings
  -> eval                sequential join/aggregation
```

Same flow as a timeline:

```text
time ->

case #1:
  scan -> chunk -> claims -> dedupe -> [ relevance | grounding | redteam | rubric | reference ] -> eval

case #2:
  starts only after case #1 finishes in the current CLI implementation
```

Interpretation:

- The left side of the graph is strictly ordered
- The metric group is the only explicit node-level parallel fan-out in the repo
- `eval` is a join point and waits for all metric outputs

### Example B: CLI `lumiseval run relevance --input sample.json`

Execution for one case:

```text
scan -> chunk -> claims -> dedupe -> relevance
```

Interpretation:

- No metric fan-out happens here
- The runner only executes the strict prerequisite chain for the requested target

### Example C: API `POST /jobs`

Execution path:

```text
POST /jobs
  -> _run_one(request)
    -> run_graph(...)
      -> graph.invoke(initial_state)
```

Practical interpretation from this repository's code:

- The API path does not enter the `CachedNodeRunner` thread-pool fan-out
- The API path still uses the same node implementations
- If rubric runs, rubric rules may still overlap inside `RubricNode.run()`
- Outside of node-local behavior like rubric, the repo does not add explicit parallel node execution around `run_graph()`

## Execution Matrix

| Scope | CLI `run eval` | CLI `run <single-metric>` | API `POST /jobs` |
|---|---|---|---|
| Cases / requests | Sequential | Sequential | Sequential |
| `scan -> chunk -> claims -> dedupe` | Sequential | Sequential | Sequential in repo code |
| Metric siblings (`relevance`, `grounding`, `redteam`, `rubric`, `reference`) | Parallel fan-out in `CachedNodeRunner` | No | No repo-local fan-out |
| Rubric rules inside the rubric node | Concurrent | Concurrent | Concurrent |
| Final `eval` aggregation | Sequential join | Not applicable unless target is `eval` | Sequential in repo code |

## Bottom Line

If you want to know where LumisEval explicitly runs independent work in parallel today, there are two places:

- `CachedNodeRunner.run_case(...)` for the metric fan-out during CLI `run eval`
- `RubricNode.run()` for concurrent rule evaluation inside the rubric node

If you want to know whether dependent nodes overlap, the answer in current repo code is no:

- `scan -> chunk -> claims -> dedupe` is sequential
- `eval` waits for metric outputs
- CLI cases are processed one by one
- API requests in one payload are processed one by one
