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

1. **Drop PPR — the one unambiguous win.** Set `GRAPH_PROXIMITY_MODE=decay`:
   Personalized PageRank adds **zero** quality over trivial decay (confirmed on
   *saturated* HotpotQA **and** *non-saturated* MuSiQue) yet is the biggest,
   fastest-growing store cost — **~65% of latency at 20k; removing it makes Neo4j
   2.7× faster** (178→67 ms). Applies to any backend.
2. **Backend choice is scale-dependent (decay vs decay; quality ~tied).** The
   earlier "pgvector is ~2× faster" was an artifact of comparing against Neo4j+PPR.
   Fair comparison: **pgvector wins at small scale** (28 vs 52 ms @2k, simpler ops)
   but **Neo4j+decay scales far better and wins large** (67 vs 131 ms @20k — Neo4j
   nearly flat, pgvector grows steeply). And Neo4j+decay needs **no GDS**, so its
   ops gap mostly disappears. Quality: pgvector slightly better SciFact top-k
   (0.749 vs 0.733), Neo4j slightly better multi-hop recall — a wash. Pick by
   scale + which DB you already run; keep Neo4j for communities / entity graph.
3. **Engram-DB design:** skip PageRank; the **graph/keyword-sibling expansion is
   the scaling bottleneck**, and **native graph adjacency beats a SQL self-join at
   scale** (Neo4j's traversal grew 20→29 ms vs pgvector's join 11→85 ms). So a
   custom engine should use native adjacency + skip PPR — compose embedded engines,
   don't write a DBMS.
4. **Chunking:** don't over-split coherent docs (engram's default size is good);
   NEXT_CHUNK's value is context-completeness, not doc-rank (needs a chunk-level
   metric to quantify — still open).
5. **Engram-DB ships and keeps the quality win — now verified on the real
   production stack.** A GPU head-to-head with `bge-m3` + `bge-reranker-v2-m3`
   (not just the CPU MiniLM floor) shows **engramdb ties the engram-layer (Neo4j)**
   — SciFact 0.7389 vs 0.7373, NFCorpus 0.3397 vs 0.3378 nDCG@10 — while **beating
   standard 2-stage RAG by +1.39 / +0.73 nDCG@10**. Same quality, fastest backend.
   See "Real production-model head-to-head" below.

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

| metric | Neo4j SciFact | pgvector SciFact | Neo4j NFCorpus | pgvector NFCorpus |
|---|---|---|---|---|
| nDCG@10 | 0.7330 | **0.7494** | 0.3963 | **0.4002** |
| Recall@10 | 0.8260 | **0.8530** | 0.1828 | **0.1919** |
| Recall@100 | **0.9700** | 0.9150 | **0.3060** | 0.2577 |
| MAP | 0.7042 | **0.7141** | **0.1924** | 0.1884 |
| ingest time | 218s | **136s** | 159s | **104s** |

**The same shape holds on both datasets** (NFCorpus confirms it's not SciFact-
specific):
- **pgvector slightly *beats* Neo4j on top-k** (nDCG@10, Recall@10 on both) **and
  is faster** to ingest. For non-graph retrieval it's the better pick on quality
  and speed, not merely "lighter."
- **Neo4j has higher Recall@100 on both** (0.970 vs 0.915; 0.306 vs 0.258) — its
  vector index recalls more gold deep in the pool; pgvector's HNSW misses some
  deep candidates (tunable via `ef_search` — cf. the per-tenant bump shipped). It
  didn't hurt top-10 here, but matters where deep recall feeds the reranker.
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

### Chunk-level NEXT_CHUNK — SQuAD answer-span coverage (the right metric)

The doc-level ablation can't see NEXT_CHUNK, so this uses a chunk-level metric with
no fuzzy qrels: each SQuAD article = one document (paragraphs concatenated), chunked
small (`CHUNK_TARGET_CHARS=300`, `METADATA_EXTRACTOR=none`); **answer-coverage@10** =
the gold answer string appears in the returned chunks. pgvector, 400 questions / 48
articles:

| arm | answer-coverage@10 |
|---|---|
| NEXT_CHUNK **on** (seed=8, hops=2) | 0.9450 (378/400) |
| NEXT_CHUNK **off** (seed=0) | 0.9425 (377/400) |

- **+0.25 pt — literally one question out of 400.** Even with a metric built to
  expose it, NEXT_CHUNK adds essentially nothing here: SQuAD answers sit in a single
  paragraph the question matches directly, so dense+rerank already covers them and
  there's no split-context for the sequence chain to recover.
- **Verdict on the chunking thesis:** across *every* benchmark we can run —
  doc-level (SciFact) and chunk-level (SQuAD) — NEXT_CHUNK does not move retrieval
  metrics. Its only plausible remaining value is **context completeness for a
  generative reader** (adjacent chunks improving the *answer*, not the *retrieval*),
  which needs an LLM + answer-quality metric to measure (gated). On retrieval
  metrics it's a no-op — keep it for read-time context, don't expect ranking lift.

This (and the store/PPR decision) means **engram's graph layer adds ≤ ~2 pt only on
multi-hop keyword bridges; PPR and NEXT_CHUNK add ~0** on everything measured — the
dense embedder + cross-encoder reranker already do the work. The graph is cheap
insurance + context, not a ranking lever, on standard benchmarks.

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

The Neo4j columns above run its **default `ppr`**. That is the confound: PPR
dominates and grows. The PPR decomposition (Neo4j 20k, `GRAPH_PROXIMITY_MODE=decay`):

| Neo4j 20k | end-to-end | graph |
|---|---|---|
| +PPR (default) | 178 | 145 |
| **+decay (PPR off)** | **67** | **29** |

→ **PPR costs ~115 ms (≈65% of latency) at 20k for zero quality** — dropping it
makes Neo4j **2.7× faster**. So the fair comparison is **decay vs decay**:

| decay vs decay — end-to-end (ms) | 2k | 5k | 10k | 20k |
|---|---|---|---|---|
| Neo4j + decay | 52 | 57 | **55** | **67** |
| pgvector + decay | **28** | 57 | 133 | 131 |

**Crossover ≈ 5k documents.** Below it pgvector wins; at ~5k they tie (~57 ms);
above it **Neo4j+decay wins and stays nearly flat (~55–67 ms across 2k→20k)** while
pgvector degrades to ~130 ms (its HNSW retrieval + SQL keyword-join grow with the
corpus; numbers are also noisier run-to-run). Actionable rule: **pgvector under
~5k docs, Neo4j+decay above** (real-corpus / real-models would shift the exact
threshold, but Neo4j's flat scaling vs pgvector's steep growth is the robust shape).

Stage detail at the endpoints (decay vs decay):

| stage | Neo4j 2k | pgvector 2k | Neo4j 20k | pgvector 20k |
|---|---|---|---|---|
| retrieval | 26 | 12 | 30 | 40 |
| graph | 20 | 11 | 29 | 85 |

What this actually shows (corrects the earlier "pgvector is ~2× faster" call,
which was comparing pgvector+decay against Neo4j+**PPR**):
- **There is a crossover.** pgvector wins at small scale (28 vs 52 @2k); **Neo4j
  wins at large scale (67 vs 131 @20k) and scales far better** — Neo4j is nearly
  flat (52→67 over 10× data) while pgvector grows steeply (28→131).
- **Why:** pgvector's HNSW retrieval grows (12→40) *and* its SQL keyword-sibling
  join grows hard (11→85); Neo4j's vector index (26→30) and native graph traversal
  (20→29) barely move. **Native graph adjacency beats a SQL self-join at scale.**
- **The graph/sibling-expansion stage is still the scaling bottleneck** — but
  Neo4j's traversal handles it ~3× better than pgvector's join at 20k. **Engram-DB
  lesson: use native graph adjacency, not SQL joins, and skip PPR.**
- **Drop PPR is the one unambiguous win** (2.7× faster, zero quality, any backend).

The earlier (dense ~40-word vocab) latency numbers over-stated the graph stage on
both backends and are superseded by this realistic-corpus sweep.

What pgvector gives up (feature, not latency): **PPR graph proximity** (→ decay
fallback), the **community/theme layer**, and the **structured-entity graph**.
Multi-tenancy, recency, contextual retrieval, learned-sparse, and sibling
expansion work on both.

**Recommendation (final — corrected after the PPR decomposition).**

1. **Drop PPR, set `GRAPH_PROXIMITY_MODE=decay`** — the one unambiguous, backend-
   independent win: zero quality cost (HotpotQA + MuSiQue), and it removes ~65% of
   latency at 20k (Neo4j 178→67 ms). PPR/GDS is dead weight; decay is the default.

2. **Then the backend is a scale + ops choice, not a clear winner** (quality ties
   with decay; SciFact top-k slightly favours pgvector, multi-hop recall slightly
   favours Neo4j):
   - **Small / simple deployments → pgvector.** Faster at small scale (28 vs 52 ms
     @2k) and one system (Postgres) if you don't otherwise run Neo4j.
   - **Large corpora / agent-memory at scale → Neo4j + decay.** It scales far
     better (nearly flat 52→67 ms vs pgvector 28→131 ms over 10× data) because
     native graph traversal + its vector index beat pgvector's SQL keyword-join +
     HNSW as the corpus grows. With PPR gone, **Neo4j+decay needs no GDS**, so the
     ops gap largely closes.
   - **Communities / entity graph** remain Neo4j-only either way.

This *corrects* the earlier "pgvector is the strong default / ~2× faster" call,
which compared pgvector+decay against Neo4j running **PPR by default** — an unfair
baseline. Apples-to-apples (decay vs decay), it's a scale-dependent crossover.

**Confidence:** "drop PPR" is confirmed on saturated (HotpotQA) + non-saturated
(MuSiQue) multi-hop, and the latency cost is decomposed at scale — high. The
backend crossover is from one synthetic-corpus sweep (fake models) — directional,
worth confirming with real models + a real corpus. Open gaps: `NEXT_CHUNK`
chunking thesis (needs a long-doc corpus + chunk-level metric), communities/
entity-graph quality (no bench), BGE-M3 absolute numbers (GPU) — none expected to
change these deltas.

## Status

- `bench/profile_latency.py` shipped. Fake-models latency numbers above are real;
  the **real production-model quality run is now done** (`BAAI/bge-m3` +
  `BAAI/bge-reranker-v2-m3` on the RTX 4080 — see "Real production-model
  head-to-head" below). A large-scale (50k+) real-corpus *latency* sweep is the
  remaining measurement.
- **Prototype shipped** ([app/store_engramdb.py](../app/store_engramdb.py),
  `STORE_BACKEND=engramdb`) — Tier 1 of the plan, embodying the findings: embedded
  single-process store with in-memory vector (brute-force cosine), BM25 (inverted
  index over text+summary+context → contextual BM25), **native-adjacency** graph
  (NEXT_CHUNK + keyword, no SQL self-join), **decay only (no PPR)**, plus
  multi-tenancy / recency / sparse / near-dup / feedback. Optional pickle snapshot
  (`ENGRAMDB_PATH`). Community synthesis + structured-entity graph deliberately
  omitted. Runs the full pipeline + passes an in-process test suite (no server).
### Prototype benchmark — Engram-DB **wins across the board**

Same profiler, `--fake-models`, decay-vs-decay, end-to-end ms/query (engramdb runs
in-process; the others over a container socket):

| docs | Neo4j+decay | pgvector | engramdb (matmul) | **engramdb + ANN** |
|---|---|---|---|---|
| 2k | 52 | 28 | 12.7 | **8.6** |
| 20k | 67 | 131 | 78.5 | **41.6** |
| 50k | — | — | — | 122.8 |
| ingest 20k | ~49 s | ~25 s | 19 s | **19 s** |

- **engramdb is fastest at every compared scale** — ~3–6× faster than the backends
  at 2k, and at 20k (**41.6 ms**) it beats Neo4j+decay (67) *and* pgvector (131).
  The win comes from being in-process (no socket / query-parse round-trip), a
  **native-adjacency** graph (best graph stage: 0.6 / 5.6 ms vs 11–85), and an ANN
  index (usearch) for sub-linear vector search.

**Quality is preserved (the win we must not lose).** Real pipeline (local MiniLM +
ms-marco), 100 queries — engramdb vs neo4j / pgvector:

| dataset | nDCG@10 | Recall@10 | Recall@100 | MAP |
|---|---|---|---|---|
| SciFact | 0.736 (vs 0.733 / 0.749) | 0.836 (0.826 / 0.853) | **0.970** (0.970 / 0.915) | 0.705 (0.704 / 0.714) |
| NFCorpus | 0.395 (vs 0.396 / 0.400) | 0.181 (0.183 / 0.192) | **0.306** (0.306 / 0.258) | 0.191 (0.192 / 0.188) |

engramdb is in the band on every metric, and its **Recall@100 ties Neo4j (the best)
and beats pgvector** — the usearch ANN is not dropping recall, and the in-house
BM25 fuses equivalently. So engramdb is **same quality, fastest backend** — the
speed win does not cost quality.
- The **matmul → ANN** swap mattered: it cut 20k retrieval 51 → 28 ms (and 2k
  12.7 → 8.6). Same quality — usearch returns the exact-match top hits the tests
  assert.
- **Vector quantization (memory moat) — implemented + quality-verified.**
  `ENGRAMDB_QUANTIZATION` = `f16` (default) / `f32` / `i8` (usearch index `dtype`)
  / `b1` (binary; a hamming shortlist + exact-cosine rescore — see below). Floor
  stack (MiniLM), SciFact:

  | quant | nDCG@10 | Recall@10 | Recall@100 | memory |
  |---|---|---|---|---|
  | f32 | 0.7360 | 0.8360 | 0.9700 | 1× |
  | **f16** (default) | 0.7360 | 0.8360 | 0.9700 | **½×** |
  | i8 | 0.7358 | 0.8360 | 0.9600 | **¼×** |

  **f16 is lossless for ranking** (identical to f32) → free 2× memory, the default.
  **i8 keeps top-k identical** (only deep Recall@100 dips 1pt, still > pgvector's
  0.915) at 4× savings. So the memory win does **not** cost the quality win.

  **b1 (binary, 32× smaller) — verified on the real production stack.** 1 bit/dim
  is too coarse to *rank* directly, so engramdb takes a `k`×16 **hamming shortlist**
  then **rescores it with exact cosine** (2-stage) — the binary stage only has to
  keep the right docs in a generous shortlist, the rescore restores precision. The
  vector cache holds only the bits (32× smaller than f32); the rescore reads the
  ~80 shortlisted f32 vectors from the chunk store. The full 32× *RAM* win
  additionally needs those chunk embeddings out of core (the on-disk/mmap segment
  format, below); in this in-memory prototype they stay resident for MMR/dedup. On
  bge-m3 + bge-reranker-v2-m3, SciFact, **b1 ties full precision to 3 decimals**:

  | engramdb (bge-m3, SciFact) | nDCG@10 | Recall@10 | MAP | P@10 |
  |---|---|---|---|---|
  | f32 | 0.7389 | 0.8529 | 0.7027 | 0.0960 |
  | **b1** (32× smaller) | **0.7390** | **0.8529** | **0.7030** | **0.0960** |

  So the whole quant ladder — f16 (½×) / i8 (¼×) / **b1 (1/32×)** — is quality-safe;
  b1 is the deep memory moat for very large / unbounded corpora. (Naive 1-bit
  hamming *without* the rescore is not quality-safe — the rescore is what makes it
  work; covered by `test_b1_quantization_preserves_ranking`.)
- **Remaining (lower-value) levers:** the one scaling bottleneck left is the BM25
  fulltext (vector is now sub-ms), but that's mostly a *synthetic-query artifact*
  (profiler query terms are super-common words → huge postings; real queries are
  selective), so block-max/WAND BM25 is deferred until a real corpus shows it
  matters. An incremental on-disk segment format (today: in-memory + optional
  pickle snapshot) is the other production-hardening follow-up. *(The real-model
  quality run is done — see below.)*

### Real production-model head-to-head (bge-m3 + bge-reranker-v2-m3, RTX 4080)

The quality numbers above use the local **MiniLM floor stack** (CPU). To confirm
the win survives the **real production models**, the same head-to-head harness
([bench/compare.py](../bench/compare.py)) was run on the GPU with engram's
configured stack — `BAAI/bge-m3` (1024-d) embeddings + `BAAI/bge-reranker-v2-m3` —
scoring four systems on *identical* models: `bm25`, `dense`, `dense+rerank` (the
standard 2-stage RAG baseline), and `engram` on both the **engram-layer (Neo4j)**
and **engramdb** backends. Rerank depth 100; full BEIR test sets; f32.

**SciFact** (300 queries, 5183 docs):

| system | nDCG@10 | Recall@10 | MAP | P@10 |
|---|---|---|---|---|
| bm25 | 0.6519 | 0.7740 | 0.6132 | 0.0850 |
| dense | 0.6415 | 0.7751 | 0.5990 | 0.0870 |
| dense+rerank *(standard 2-stage)* | 0.7250 | 0.8246 | 0.6934 | 0.0933 |
| engram · Neo4j *(engram-layer)* | 0.7373 | 0.8529 | 0.7007 | 0.0960 |
| **engram · engramdb** | **0.7389** | **0.8529** | **0.7027** | **0.0960** |

**NFCorpus** (323 queries, 3633 docs):

| system | nDCG@10 | Recall@10 | MAP | P@10 |
|---|---|---|---|---|
| bm25 | 0.3062 | 0.1521 | 0.1368 | 0.2180 |
| dense | 0.3174 | 0.1504 | 0.1446 | 0.2316 |
| dense+rerank | 0.3324 | 0.1614 | 0.1518 | 0.2412 |
| engram · Neo4j *(engram-layer)* | 0.3378 | 0.1642 | 0.1547 | 0.2430 |
| **engram · engramdb** | **0.3397** | **0.1659** | **0.1562** | **0.2452** |

Two things this proves on the **real** models, not the floor stack:

1. **engramdb preserves the engram-layer quality win.** engram·engramdb vs
   engram·Neo4j is +0.0016 (SciFact) / +0.0019 (NFCorpus) nDCG@10 with identical
   Recall@10 on SciFact — a tie within ANN/tie-break noise. (The `bm25` / `dense` /
   `dense+rerank` rows came out **byte-identical** across the two backend runs,
   confirming the harness is reproducible and only the backend changed.) engramdb's
   usearch ANN + in-house contextual BM25 retrieve equivalently to Neo4j's HNSW.
2. **The architecture beats standard RAG on real models too.** engram·engramdb
   over `dense+rerank` (the strong competitor): **+1.39 nDCG / +2.83 recall@10**
   (SciFact), **+0.73 / +0.45** (NFCorpus); over naive dense, **+9.7 / +2.2** nDCG.
   So the pipeline's lift is not a small-model artifact — it holds with a 2026 SOTA
   embedder + reranker. (Both backends sit ~+1.2–1.4 over `dense+rerank`; engramdb
   is the faster of the two at equal quality.)
