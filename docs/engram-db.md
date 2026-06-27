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

## TL;DR — evaluation verdict (2026-06-27)

A full eval campaign (latency profiling + scale sweep + quality on SciFact,
HotpotQA, MuSiQue + a chunking ablation, all on local CPU models) settled the
store question and reshaped the Engram-DB design:

1. **Default to pgvector + `GRAPH_PROXIMITY_MODE=decay`.** pgvector beats Neo4j on
   standard retrieval (SciFact nDCG@10 0.749 vs 0.733), ties on multi-hop, runs
   faster end-to-end at every scale, and needs only Postgres. Keep Neo4j only for
   its unique features (communities, entity graph, deeper recall@100).
2. **Drop PPR.** Personalized PageRank adds **zero** quality over trivial decay
   (confirmed on saturated *and* non-saturated multi-hop) yet is Neo4j's
   fastest-growing latency cost (graph stage 46→145 ms from 2k→20k docs).
3. **Engram-DB design:** skip PageRank entirely; the **graph/keyword-sibling
   expansion is the real scaling bottleneck on both backends** (the dominant cost
   at 20k) — that's what a custom engine must make efficient (bounded fan-out,
   indexed joins), not the vector ANN. Compose embedded engines, don't write a DBMS.
4. **Chunking:** don't over-split coherent docs (engram's default size is good);
   NEXT_CHUNK's value is context-completeness, not doc-rank (needs a chunk-level
   metric to quantify — still open).

Details, tables, and caveats below.

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

### Quality (real pipeline, BEIR SciFact — no graph exercised)

`bench/run_benchmark.py`, real engram pipeline on local CPU models (MiniLM-384 +
ms-marco cross-encoder), full 5183-doc corpus, 100 test queries, content + BM25
channels (SciFact is single-chunk with no keywords → no graph, so this isolates
the **vector + lexical + rerank** path):

| metric | Neo4j | pgvector |
|---|---|---|
| nDCG@10 | 0.7330 | **0.7494** |
| Recall@10 | 0.8260 | **0.8530** |
| Recall@100 | **0.9700** | 0.9150 |
| MAP | 0.7042 | **0.7141** |
| ingest / search time | 218s / 205s | **136s / 88s** |

- **On standard (non-graph) retrieval, pgvector matches or slightly *beats* Neo4j
  on top-k quality** (nDCG@10 +1.6, Recall@10 +2.7) **and is ~2× faster** to
  ingest and query. So pgvector isn't merely "lighter" — for non-graph workloads
  it's the better pick on both quality and speed.
- **But Neo4j has higher Recall@100** (0.970 vs 0.915): its vector index recalls
  ~5.5 pts more gold deep in the pool. pgvector's HNSW misses some deep candidates
  (a tunable `ef_search` issue) — they can never be reranked into the top-k. It
  didn't hurt top-10 here, but on a corpus where deep recall matters, raise
  pgvector's `ef_search` (cf. the per-tenant `ef_search` bump already shipped).
- **The whole Neo4j case therefore rests on the graph** (PPR proximity,
  communities, entity graph) — which SciFact does not exercise. Whether that earns
  Neo4j's 2× latency is the multi-hop question (below).

### Quality (real pipeline, HotpotQA multi-hop — the graph's home turf)

`bench/multihop.py`, 250 questions, 2477 passages, MiniLM-384 + ms-marco, full
graph pipeline (YAKE keywords → `HAS_KEYWORD` graph; passages are single-chunk so
no `NEXT_CHUNK`). Recall of the supporting passages:

| system | Recall@2 | Recall@5 | Recall@10 |
|---|---|---|---|
| bm25 | 0.494 | 0.654 | 0.832 |
| dense | 0.540 | 0.732 | 0.846 |
| dense+rerank | 0.684 | 0.814 | 0.908 |
| engram — Neo4j + **PPR** | 0.690 | **0.838** | **0.932** |
| engram — Neo4j + **decay** | 0.690 | **0.838** | **0.932** |
| engram — pgvector + decay | 0.684 | 0.820 | 0.914 |

Two decision-changing results:
1. **PPR == decay, to the digit.** Personalized PageRank — Neo4j's signature
   capability and *the single most expensive store operation* (the profiler's
   heaviest cost) — adds **zero** quality over simple per-hop decay here. The
   graph *expansion* (pulling in keyword-siblings) is what lifts engram over
   dense+rerank (+2.4 pt Recall@5/@10); the expensive *proximity algorithm* on top
   contributes nothing measurable.
2. **pgvector + decay nearly matches Neo4j** (0.914 vs 0.932 @10; within ~1.8 pt),
   capturing most of the graph-expansion benefit through its SQL keyword-sibling
   join — no GDS required.

### Quality — MuSiQue (non-saturated multi-hop, the validation set)

`bench/multihop.py` with `BENCH_MULTIHOP_DATASET=musique` (200 questions, 2882
passages, MiniLM). MuSiQue is built to defeat single-hop shortcuts — dense+rerank
hits only Recall@10 0.65 here (vs 0.91 on HotpotQA), so graph/PPR have real
headroom to show value:

| system | Recall@2 | Recall@5 | Recall@10 |
|---|---|---|---|
| dense+rerank | 0.4650 | 0.5925 | 0.6525 |
| engram — Neo4j + **PPR** | 0.4775 | 0.5950 | 0.6575 |
| engram — Neo4j + **decay** | 0.4775 | **0.5975** | **0.6600** |
| engram — pgvector + decay | 0.4725 | 0.5925 | **0.6675** |

The HotpotQA findings **hold on the harder set**, de-caveated:
- **PPR still adds nothing** — decay *equals or edges* PPR (R@5 0.5975 vs 0.5950,
  R@10 0.6600 vs 0.6575). Confirmed on a saturated *and* a non-saturated multi-hop
  benchmark: Personalized PageRank does not earn its cost.
- **pgvector ties Neo4j** (and wins Recall@10 0.6675 vs 0.6600), at ~2× the ingest
  speed (20s vs 49s).
- The graph *expansion* adds only ~+1 pt over dense+rerank even here — a minor
  positive, not a game-changer, for passage-retrieval multi-hop.

**Remaining quality gap (one mechanism still dormant):** all benchmarks so far are
single-chunk passages, so `NEXT_CHUNK` (sequence) expansion never fired.

### Chunking ablation (NEXT_CHUNK) — SciFact, forced multi-chunk

To exercise `NEXT_CHUNK`, SciFact abstracts were force-split (`CHUNK_TARGET_CHARS=400`,
`METADATA_EXTRACTOR=none` so the *only* expansion is the sequence chain). pgvector,
100 queries, doc-level scoring:

| config | nDCG@10 | Recall@10 | Recall@100 |
|---|---|---|---|
| A — small chunks + NEXT_CHUNK (hops=2) | 0.7252 | 0.8560 | 0.8880 |
| B — small chunks + no expansion (seed=0) | 0.7252 | 0.8560 | 0.8880 |
| C — large chunks (1800) baseline | **0.7498** | **0.8680** | 0.8680 |

- **A == B to four decimals → NEXT_CHUNK is invisible to *document-level* retrieval,
  by construction.** Sequence siblings are same-document chunks, and the doc is
  already represented by its best chunk, so pulling neighbours in cannot change the
  doc's rank. NEXT_CHUNK's real payoff is **context completeness** (handing the
  agent the adjacent chunks around a hit) — measurable only with a *chunk-level*
  answer-context metric, which standard IR benchmarks (doc-level qrels) don't have.
  So the chunking thesis is neither confirmed nor refuted here — it needs the right
  metric, not just the right corpus.
- **Over-splitting a coherent short document hurts** (C beats A by +2.5 nDCG):
  SciFact abstracts are ~one good chunk; fragmenting them degrades each fragment's
  embedding. engram's larger default chunk size is well-chosen for abstract-length
  content. (Small chunks do raise Recall@100 — more surface area — but rank top-k
  worse.)

This is a *separate* claim from the store/PPR decision, which is settled.

### Latency scale sweep (profiler, realistic-corpus, `--fake-models`)

Mean ms/query, per-doc jargon corpus (sparse keyword graph), models excluded:

| stage | Neo4j 2k | pgvector 2k | Neo4j 20k | pgvector 20k |
|---|---|---|---|---|
| **end-to-end** | 78 | **28** | 178 | **131** |
| store | 72 | 23 | 171 | 125 |
| ├─ retrieval (vec+FTS) | 26 | 12 | **26** | 40 |
| ├─ graph (siblings ± PPR) | 46 | 11 | **145** | 85 |
| CPU/other | 6 | 5 | 7 | 5 |

What scales (and what doesn't):
- **pgvector is faster end-to-end at every size** (28 vs 78 @2k; 131 vs 178 @20k),
  though the relative gap narrows with scale (2.8× → 1.36×).
- **The graph stage is the scaling bottleneck on *both* backends** — it dominates
  at 20k (Neo4j 145/178 = 81%; pgvector 85/131 = 65%). The keyword-sibling
  expansion grows with corpus size on each. **This, not the vector ANN, is the #1
  thing Engram-DB must make efficient at scale** (bounded fan-out, indexed joins).
- **Neo4j's graph grows fastest** (46 → 145 ms) — that extra growth is PPR/GDS, the
  op proven to add zero quality. (A neo4j+decay run at 20k decomposes siblings vs
  PPR — see below.)
- **Surprise reversal in vector retrieval at scale:** Neo4j's vector index stays
  flat (26 → 26 ms) while pgvector's HNSW grows (12 → 40 ms) and *overtakes* it.
  pgvector's deep recall + retrieval cost is `ef_search`-bound — tune it at scale.

The earlier (dense ~40-word vocab) latency numbers over-stated the graph stage on
both backends and are superseded by this realistic-corpus sweep.

What pgvector gives up (feature, not latency): **PPR graph proximity** (→ decay
fallback), the **community/theme layer**, and the **structured-entity graph**.
Multi-tenancy, recency, contextual retrieval, learned-sparse, and sibling
expansion work on both.

**Recommendation (updated by the quality evals — reverses the earlier
latency-only call).** The data now points to **pgvector + decay as the strong
default for the current state**:
- it **beats** Neo4j on standard retrieval top-k (SciFact nDCG@10 0.749 vs 0.733)
  and **nearly matches** it on multi-hop (0.914 vs 0.932 @10),
- at **~2× the speed** (ingest + query) and far simpler ops (Postgres only, no
  Neo4j + GDS),
- and **PPR — the costliest store op — buys no measurable quality** over decay, so
  Neo4j's signature feature isn't paying its way on these benchmarks.

So: default to **pgvector**, and set **`GRAPH_PROXIMITY_MODE=decay`** (don't pay
for GDS PageRank until a corpus proves it helps).

**Stay on Neo4j when** you specifically need what only it offers: the
**community/theme layer** (global "what are the themes" search), the
**structured-entity graph**, or deeper candidate recall (Recall@100 0.970 vs
0.915 — or just raise pgvector's `ef_search`). These are real but
workload-specific, not the default.

**Confidence:** the "drop PPR" call is now confirmed on **both** a saturated
(HotpotQA) and a non-saturated (MuSiQue) multi-hop benchmark — not a saturation
artifact. Open (smaller) gaps: the `NEXT_CHUNK` chunking thesis (needs a
long-document corpus; a *separate* claim), communities/entity-graph quality (no
bench yet), and absolute numbers under the production embedder (BGE-M3, gated on
GPU) — none of which is expected to change the cross-backend *deltas*.

## Status

- `bench/profile_latency.py` shipped (this branch). Fake-models numbers above are
  real; real-models + large-scale sweep are the next measurements (need the
  endpoints / a bigger box).
- Ambition tier deferred until the real-models share + scale curve are in hand —
  per the "decide after profiling" call.
