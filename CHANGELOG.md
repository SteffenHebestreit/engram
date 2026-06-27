# Changelog

All notable changes to **engram**. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are tagged in git.

## [Unreleased]

### Added
- **Engram-DB `b1` binary quantization (32× smaller vectors), quality-verified.**
  `ENGRAMDB_QUANTIZATION=b1` packs each vector to 1 bit/dim and retrieves in two
  stages — a `k`×16 **hamming shortlist** then an **exact-cosine rescore** of that
  shortlist — so ranking stays full-precision. On the real production stack
  (`bge-m3` + `bge-reranker-v2-m3`, SciFact) b1 **ties f32 to 3 decimals**
  (nDCG@10 0.7390 vs 0.7389, identical Recall@10) at 1/32 the vector memory. The
  earlier naive `dtype=b1` path (no rescore) was broken/quality-risky and is
  replaced. Covered by `test_b1_quantization_preserves_ranking`. The deep tier of
  the memory ladder: f16 (½×) / i8 (¼×) / **b1 (1/32×)**, all quality-safe.

### Verified
- **Engram-DB quality parity proven on the real production stack** (not just the
  CPU MiniLM floor): GPU head-to-head with `bge-m3` + `bge-reranker-v2-m3`
  ([bench/compare.py](bench/compare.py), now with bootstrap CIs + paired sign
  tests + a hybrid+rerank control) — engram·engramdb **ties** engram·Neo4j (SciFact
  0.7389 vs 0.7373, NFCorpus 0.3377 vs 0.3378) and **beats pgvector** (SciFact
  0.7232, its BM25 channel underperforms); confirmed on multi-hop (engramdb decay
  ties Neo4j PPR) and with Qwen3-Reranker (0.7738 vs 0.7723). **Honest correction**
  (after an adversarial review): engram's lift over a strong `dense+rerank` /
  `hybrid+rerank` baseline is **not statistically significant** — the median/MMR/
  graph stages add no robust single-hop or multi-hop nDCG with production models;
  the architecture is a robustness floor + operational value, and the reranker is
  the real quality lever. See [docs/engram-db.md](docs/engram-db.md).

## [0.4.0] - 2026-06-27

### Added
- **Engram-DB embedded backend** (`STORE_BACKEND=engramdb`, **experimental**) — a
  purpose-built, single-process retrieval store distilled from the v0.3.0
  evaluation: in-process vector search (usearch ANN, `ENGRAMDB_QUANTIZATION`
  f16/f32/i8/b1), BM25 over text+summary+context (contextual BM25), and a
  **native-adjacency** graph (NEXT_CHUNK + keyword) with decay proximity — **no
  PPR/GDS, no community synthesis, no structured-entity graph, no server**.
  Optional pickle snapshot (`ENGRAMDB_PATH`); supports multi-tenancy, recency,
  sparse, near-dup links and feedback. See [docs/engram-db.md](docs/engram-db.md).
  - **Fastest backend at every measured scale, same quality.** Profiler
    (decay-vs-decay, end-to-end): 8.6 ms @2k / 41.6 ms @20k vs Neo4j+decay
    52 / 67 and pgvector 28 / 131; ingest 10–25× faster. Quality on SciFact +
    NFCorpus matches Neo4j/pgvector (e.g. SciFact nDCG@10 0.736, Recall@100 0.970
    — ties Neo4j, beats pgvector). f16 quantization is lossless for ranking; i8
    keeps top-k identical at ¼ the memory. *(Quality parity re-verified
    post-release on the real `bge-m3` + `bge-reranker-v2-m3` stack — see
    [Unreleased].)*

### Changed
- New optional dependency `usearch` (the engramdb ANN index; other backends
  ignore it, and engramdb falls back to an exact matmul if it is absent).

## [0.3.0] - 2026-06-27

### Added

- **Multi-tenant isolation** (`tenant_id` on ingest + search, opt-in) — the
  biggest adoption gap for SaaS RAG, and security-sensitive: every one of the
  four chunk-surfacing reads filters by tenant (dense `vector_search`, BM25
  `fulltext_search`, ingest-dedup `nearest_chunks` filter in-query; graph
  siblings — which reach another tenant via a shared keyword — are filtered in
  the pipeline). Ingest ids are namespaced per tenant so identical content / a
  reused `document_id` can't collide or let one tenant overwrite another's doc.
  Neo4j over-fetches the ANN top-k before filtering; pgvector raises
  `hnsw.ef_search` in-transaction (both keep the filtered top-k full). Gated by a
  **0%-cross-tenant-leakage** test on both live backends.
- **Contextual Retrieval** (`CONTEXTUAL_RETRIEVAL_ENABLED`, opt-in) — Anthropic's
  technique: at ingest an LLM writes a short document-situating context per chunk,
  prepended before embedding so the content vector encodes document-level identity
  (which entity/section/period it belongs to) instead of just the bare passage.
  A change to the embedding *geometry* — the one layer a reranker can't overwrite —
  and complementary to `NEXT_CHUNK` expansion (doc identity baked in at index time
  vs. neighbour context at read time). Degrades to the bare chunk when the LLM is
  down; part of the schema signature. Wired through the existing `CHANNEL_SOURCES`
  seam. Includes **contextual BM25**: the context is also indexed for fulltext
  (Neo4j: alongside text/summary; pgvector: a separate `context_tsv` generated
  column) so the lexical channel benefits too — Anthropic's larger reported gain —
  additively (empty when off → unchanged behaviour).
- **Recency / temporal decay** (`RECENCY_ENABLED`, opt-in) — the agent-*memory*
  signal: after reranking, blend an exponential recency factor on each candidate's
  document age into the final ordering, so among similarly-relevant results the
  newer ones rank higher (what Mem0/Zep/Letta do and pure-relevance RAG ignores).
  Applied **post-rerank**, so it's an orthogonal signal the cross-encoder can't
  overwrite. No schema/ingest change — reuses document `created_at` via a batched
  read (like the sparse/near-dup reads), so the hot retrieval queries are
  untouched. `RECENCY_WEIGHT` / `RECENCY_HALF_LIFE_DAYS` tune it; `SearchResult`
  gains `recency_score`.

- **Reranker sidecar** (`deploy/reranker`) — serves Qwen3-Reranker in engram's
  reranker wire format, since TEI can't serve its causal-LM format. The measured
  **+3.15 / +3.84 nDCG@10** (SciFact / NFCorpus) reranker upgrade, made
  deployable.
- **Adaptive query routing** (`ROUTER_STRATEGY=heuristic`) — a no-LLM `ROUTERS`
  registry + classifier that auto-selects a preset per query (factoid →
  balanced, complex/thematic → max_quality); an explicit preset always wins.
 
- **Implicit-relevance feedback** (`POST /feedback`, MCP `mark_used`) — an agent
  reports which chunks it grounded its answer on; engram persists the
  (query → used-chunk) positives as the foundation for offline hard-negative
  mining + weight tuning.

### Changed

- **`GRAPH_PROXIMITY_MODE` now defaults to `decay`** (Personalized PageRank is
  opt-in). A full store/graph evaluation showed PPR adds **no measurable quality**
  over trivial per-hop decay on both saturated (HotpotQA) and non-saturated
  (MuSiQue) multi-hop benchmarks, while being the single most expensive,
  fastest-growing store operation (~65% of search latency at 20k docs; ~2.7×
  faster to drop it) and requiring the Neo4j GDS plugin. Set
  `GRAPH_PROXIMITY_MODE=ppr` to opt back in.

### Evaluation & docs

- **Store/graph evaluation** ([docs/engram-db.md](docs/engram-db.md)) — latency
  profiling + scale sweep + PPR decomposition, quality on SciFact, NFCorpus,
  HotpotQA and MuSiQue, and a chunking ablation. Headlines: **pgvector** wins
  below ~5k docs, **Neo4j + decay** above (it scales nearly flat while pgvector
  grows); the graph adds ≤ ~2 pt on multi-hop only, PPR and `NEXT_CHUNK` add ~0 —
  the dense embedder + cross-encoder reranker do the work. This is also the design
  brief for a future purpose-built **Engram-DB** store (native adjacency + decay,
  no PPR).
- **Competitive scorecard** ([docs/competitive-scorecard.md](docs/competitive-scorecard.md))
  vs LangChain/LlamaIndex, RAGFlow, Weaviate, Zep/Mem0.
- **Benchmark harness** extended: `bench/profile_latency.py` (latency breakdown +
  scale sweep), parameterized datasets (`BENCH_DATASET`,
  `BENCH_MULTIHOP_DATASET=musique`), and `bench/chunk_context.py`. Bigger-model
  runs remain queued in [bench/PENDING.md](bench/PENDING.md).

## [0.2.0] - 2026-06-26

### Added
- **Pluggable storage** — a `Store` protocol with **Neo4j** (graph + vector + GDS
  Personalized PageRank) and **PostgreSQL + pgvector** backends (`STORE_BACKEND`).
- **Learned-sparse channel** (BGE-M3 `lexical_weights`), opt-in `SPARSE_ENABLED` —
  an exact-term signal folded into the fused score.
- **ColBERT** late-interaction reranker strategy (`RERANKER_STRATEGY=colbert`).
- **Asymmetric query/passage instruction prefixes** for instruction-tuned
  embedders (E5/GTE/Qwen3): `QUERY_INSTRUCTION` / `PASSAGE_INSTRUCTION`.
- **Judge-free eval harness** — `POST /eval`: IR metrics (nDCG/Recall/P@k/MAP)
  with bootstrap confidence intervals + **per-channel gold-hit attribution**, via
  the new `SearchResult.channels` provenance.
- **Memory write-path (M1)** — `nearest_chunks` primitive + non-destructive
  near-duplicate linking/collapse (`DEDUP_ENABLED`).
- **MCP server** (`app/mcp_server.py`) — `search` / `get_chunk_context` /
  `list_documents` / `search_themes` as MCP tools.
- **Community/theme layer** (Leiden + LLM reports), **search presets**
  (cheap/balanced/max_quality), **observability**, **structured-entity ingest**.
- **Benchmark harness** (`bench/`) with [RESULTS.md](bench/RESULTS.md), and a
  reference **BGE-M3 sidecar** (`deploy/bge-m3`) serving dense + sparse + ColBERT.

### Changed
- Chunk overlap now defaults to **0** — the `NEXT_CHUNK` graph recovers seam
  context, so overlap is redundant (and semantic chunking largely unnecessary).
- Renamed advancedRAG → **engram**; new README + logo.

### Notes
- Controlled benchmarks (same models, only the architecture changes): engram's
  architecture showed a **+1.6 to +2.2 nDCG@10** point estimate over naive
  dense+rerank on SciFact. **⚠️ Superseded — see [Unreleased]:** a later rigorous
  re-run (bootstrap CIs + paired sign tests + a hybrid+rerank control) found this
  delta is **not statistically significant** and is mostly the BM25 channel, not
  the graph/median/MMR. The **reranker is the one highest-leverage, robust lever**;
  upstream signals (sparse, graph, stronger embedder) cap at it. Full study +
  honest negatives in [bench/RESULTS.md](bench/RESULTS.md).

## [0.1.2] - 2026-06-14

### Added
- Fulltext-only fallback when the embedding endpoint is unavailable (search
  degrades to lexical retrieval instead of failing).

## [0.1.1] - 2026-06-14

### Added
- Reranker fallback: degrade to the fused score when the reranker is down.

## [0.1.0] - 2026-06-14

### Added
- Initial release — graph-augmented RAG over Neo4j: HyDE, 4-channel DBSF fusion
  (content / summary / keywords + BM25 fulltext), shared-keyword graph +
  Personalized PageRank proximity, median-proximity scoring, MMR shortlist,
  cross-encoder rerank, and autocut. Docker-only setup; tests in containers.
