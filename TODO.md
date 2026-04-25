# TODO

## Future work scaffolded but not wired up
- [ ] Async TaskIQ dispatch for batch jobs (`taskiq` dependency removed from `packages/nexagauge-graph/pyproject.toml`; re-add when implementing).
- [ ] Stream progress to CLI/API consumers.
- [ ] Persist `EvalReport` to SQLite via SQLModel.
- [ ] Decide whether to re-add `ragas` for metric comparison baselines (removed during cleanup — was declared but zero imports).
- [ ] Wire up `--web-search` and `--evidence-threshold` CLI flags or remove them (`apps/nexagauge-apps/ng_cli/run.py:88-89`).

## Previously tracked (carried over from `TODOS.md`)
- [ ] PR.md is tracked but listed in `.gitignore`. The `.gitignore` entry has no effect because the file was already committed before the ignore rule was added. Options: leave it (harmless), or `git rm --cached PR.md && git commit -m "stop tracking PR.md"` so future edits stay local.
- [x] ~~Stale `lumis_*` references from the rename~~ — `LUMISEVAL_CACHE_DIR` env var has been renamed to `NEXAGAUGE_CACHE_DIR`. Verify any remaining comment references (e.g., `ng_graph/registry.py`) are cleaned up.
