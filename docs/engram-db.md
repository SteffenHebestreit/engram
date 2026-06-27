# Engram-DB — purpose-built retrieval store (design + profiling)

Two ways to run engram, same pipeline:
- **Engram-Layer** (today) — the pipeline over a plug-and-play DB
  (`STORE_BACKEND=neo4j|pgvector`). Maximum feature/ops maturity for free.
- **Engram-DB** (this doc) — a third `Store` backend, purpose-built so the things
  we currently orchestrate across a general-purpose DB happen natively in one
  process. **Same quality** (models + pipeline math are backend-agnostic),
  **lower latency / memory / cost** — the payoff growing with corpus size.

It is a drop-in because everything already talks to the `Store` protocol
([app/store.py](../app/store.py)); `search.py` / `ingest.py` don't change.

## Decision discipline: profile before building

A custom store can only speed up what an engine owns — the **store** queries and
the in-process **CPU** stages. It cannot touch the **models** (query embedding +
HyDE LLM + cross-encoder reranker), which are backend-agnostic network/GPU calls.
So the build is only worth it to the extent the store is a real share of latency.
[bench/profile_latency.py](../bench/profile_latency.py) measures exactly that.

## Findings (2026-06, Neo4j backend, `--fake-models`)

`--fake-models` strips the model calls (random embeddings + identity reranker) to
isolate the **controllable** latency — store + CPU — and lets us sweep corpus size
without a GPU. Mean ms/query, synthetic corpus:

| corpus | end-to-end | STORE | ├ retrieval (vec+FTS, concurrent) | ├ graph (siblings+PPR) | CPU/other |
|---|---|---|---|---|---|
| 300 docs | 73 | 66 (90.6%) | 24 | **42** | 7 |
| 2000 docs | 120 | 111 (92.5%) | 37 | **74** | 9 |

Three things this tells us:
1. **The store dominates the controllable budget (~90–92%)** and **grows with the
   corpus** (66→111 ms as docs go 300→2000). CPU/fusion is small (~7–9 ms) — little
   upside in fusing CPU stages into the query.
2. **The graph stage (sibling expansion + Personalized PageRank) is the single
   biggest store cost — larger than vector + fulltext** (42 vs 24 ms; 74 vs 37 ms),
   and it grows fastest. This is the surprise: the lever is the **graph traversal /
   PPR**, not the vector ANN most people assume. On Neo4j this includes the GDS
   projection round-trip a co-located engine would eliminate outright.
3. **CPU/other is ~7–9 ms** and flat — fusion/median/MMR/scoring are not the bottleneck.

## The missing number (gated): the real-models share

The honest per-query *ceiling* depends on the models' share, which needs the live
embedding + reranker endpoints (unreachable from the test container here; same
gating as the GPU benches). Run it where the endpoints are up:

```bash
python -m bench.profile_latency --docs 500 --queries 50          # real end-to-end
python -m bench.profile_latency --docs 50000 --queries 50 --fake-models  # store at scale
```

Expectation to verify: with a HyDE LLM round-trip + a cross-encoder over the
shortlist, **models are typically 100s of ms**, so at *small* scale they dominate
end-to-end and Engram-DB's per-query latency ceiling is modest. **But** (a) the
store share rises with corpus size (above), and (b) for **throughput / cost** the
store + CPU is what you replicate per node — model calls are external and
batchable — so Engram-DB's win shows up in QPS/$ and at scale, not just p50.

## What the data implies for the build (decide-after-profiling)

- **Don't write a DBMS from scratch.** Compose proven embedded engines in one
  process behind the `Store` protocol (the pragmatic tier): vector ANN
  (usearch / LanceDB / Faiss, with int8/binary quantization), lexical/BM25
  (Tantivy), KV+persistence (redb / RocksDB / LanceDB), and our **graph + PPR
  in-process** (PPR is localized power iteration — we already seed it on the hits).
- **Prioritize by where the time is:** (1) co-locate graph + vector so PPR/sibling
  expansion stops round-tripping (the biggest measured cost), (2) quantize vectors
  (targets the retrieval cost + memory at scale), (3) leave CPU fusion mostly as-is.
- **Re-measure with real models** to set the honest per-query ceiling, and **sweep
  to 50k–500k docs** to size the scale win before committing to the native tier.

## Backend comparison (Engram-Layer: Neo4j vs pgvector) — which to use now

Same profiler, same synthetic corpus, `--fake-models`, mean ms/query:

| metric | Neo4j 300 | pgvector 300 | Neo4j 2000 | pgvector 2000 |
|---|---|---|---|---|
| end-to-end | 73 | **66** | 120 | **59** |
| vector retrieval (vec+FTS) | 24 | **7.6** | 37 | **11** |
| graph (siblings ± PPR) | 42 | 54 | 74 | 42 |
| CPU/other | 7 | 4.6 | 9 | 4.8 |

Clean, consistent signals:
- **pgvector vector retrieval is 2–3× faster** (7.6 vs 24 ms; 11 vs 37 ms) and
  scales better — pgvector HNSW + `tsvector` beats Neo4j's vector + fulltext here.
- **Neo4j end-to-end grows with the corpus** (73→120 ms) while pgvector stays
  flat/low (66→59) — Neo4j's PPR/GDS projection cost scales with graph size.
- The **graph row is noisy and partly an artifact**: the synthetic generator uses
  a ~40-word vocabulary, so the `HAS_KEYWORD` graph is unrealistically dense and
  the sibling join is worst-case on *both* backends (real corpora are far sparser).
  Same corpus both sides → the *comparison* is fair; the absolute graph numbers
  over-state realistic load. Neo4j additionally computes **PPR** here (a quality
  signal) that pgvector does not.

What pgvector gives up (feature, not latency): **PPR graph proximity** (→ decay
fallback), the **community/theme layer**, and the **structured-entity graph**.
Multi-tenancy, recency, contextual retrieval, learned-sparse, and sibling
expansion work on both.

**Recommendation for the current state of the project: stay on Neo4j as the
default.** The graph layer (PPR-in-the-ranker, communities, entity graph) is
engram's headline differentiator (see [competitive-scorecard.md](competitive-scorecard.md)),
the project is still establishing that moat, and at today's corpus sizes the gap
is tens of ms — quality/features win over raw speed now.

**Choose pgvector when** ops simplicity dominates (you already run Postgres — one
fewer system than Neo4j+GDS), the corpus is large (better latency scaling), or the
workload is plain hybrid retrieval where PPR tends to wash out at the reranker
anyway (our BEIR finding). It's a first-class lighter alternative, not a downgrade
for those cases.

**Caveat:** this compares *latency + features*. Whether PPR/communities lift
*quality* on your corpus needs a real `/eval` with live endpoints (gated here) —
run it before dropping the graph for a connected/multi-hop corpus.

## Status

- `bench/profile_latency.py` shipped (this branch). Fake-models numbers above are
  real; real-models + large-scale sweep are the next measurements (need the
  endpoints / a bigger box).
- Ambition tier deferred until the real-models share + scale curve are in hand —
  per the "decide after profiling" call.
