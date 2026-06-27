# Changelog

All notable changes to **engram**. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); versions are tagged in git.

## [Unreleased]

Feature branches built and tested, awaiting merge:

- **Recency / temporal decay** (`RECENCY_ENABLED`, opt-in) — the agent-*memory*
  signal: after reranking, blend an exponential recency factor on each candidate's
  document age into the final ordering, so among similarly-relevant results the
  newer ones rank higher (what Mem0/Zep/Letta do and pure-relevance RAG ignores).
  Applied **post-rerank**, so it's an orthogonal signal the cross-encoder can't
  overwrite. No schema/ingest change — reuses document `created_at` via a batched
  read (like the sparse/near-dup reads), so the hot retrieval queries are
  untouched. `RECENCY_WEIGHT` / `RECENCY_HALF_LIFE_DAYS` tune it; `SearchResult`
  gains `recency_score`. Branch `recency-scoring`.

- **Reranker sidecar** (`deploy/reranker`) — serves Qwen3-Reranker in engram's
  reranker wire format, since TEI can't serve its causal-LM format. The measured
  **+3.15 / +3.84 nDCG@10** (SciFact / NFCorpus) reranker upgrade, made
  deployable. Branch `reranker-sidecar`.
- **Adaptive query routing** (`ROUTER_STRATEGY=heuristic`) — a no-LLM `ROUTERS`
  registry + classifier that auto-selects a preset per query (factoid →
  balanced, complex/thematic → max_quality); an explicit preset always wins.
  Branch `adaptive-routing`.
- **Implicit-relevance feedback** (`POST /feedback`, MCP `mark_used`) — an agent
  reports which chunks it grounded its answer on; engram persists the
  (query → used-chunk) positives as the foundation for offline hard-negative
  mining + weight tuning. Branch `feedback-loop`.

Pending bigger-model benchmarks are queued in [bench/PENDING.md](bench/PENDING.md)
(need a higher-/unified-memory box).

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
  architecture adds **+1.6 to +2.2 nDCG@10 / +3–4 recall@10 over naive
  dense+rerank** on SciFact. The **reranker is the highest-leverage lever**;
  upstream signals (sparse, graph, even a stronger embedder) cap at it on
  saturated benchmarks. Full study + honest negatives in
  [bench/RESULTS.md](bench/RESULTS.md).

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
