# Contributing to engram

Thanks for your interest! engram is a small, readable FastAPI service over a
pluggable store. Everything runs in containers — you don't need a local Python
environment to develop or test.

## Dev loop

```bash
# run the full test suite (spins up Neo4j + pgvector, runs pytest)
docker compose --profile test run --rm tests

# lint (same as CI)
docker compose --profile test run --rm tests ruff check app scripts tests

# fast, no-DB subset (engramdb is in-process, so most tests need no server)
docker compose --profile test run --rm --no-deps tests \
  python -m pytest tests/ --ignore=tests/test_integration_neo4j.py \
  --ignore=tests/test_integration_pgvector.py -q
```

CI runs `ruff check app scripts tests` and the test suite; please make sure both
pass before opening a PR.

## Guidelines

- **Match the surrounding style** — comment density, naming, and idiom. The code
  favours clear, descriptive comments over terse one-liners.
- **Keep claims honest and measured.** engram's positioning is *parity* retrieval
  quality + a performance/operational win — not a benchmark-quality crown. Any
  benchmark claim in code or docs must be reproducible (see [bench/](bench/)) and,
  for comparative claims, significance-tested (bootstrap CIs + paired tests, as in
  `bench/compare.py`). No un-verified superiority claims.
- **New store backends** register on `STORES` and implement the `Store` protocol
  (`app/store.py`); new pipeline stages register on their strategy registries
  (`app/pipeline.py`, `app/rerank.py`).
- **Opt-in by default** for new signals/features (a flag in `app/config.py` +
  `SEARCH_TUNABLE_FIELDS` if it's search-time), so the default pipeline is stable.

## Reporting issues

Include the engram version (`GET /` or `app/main.py`), the `STORE_BACKEND`, and a
minimal repro. For retrieval-quality reports, a small golden set you can run
through `POST /eval` is ideal.
