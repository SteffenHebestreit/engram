# engram competitive roadmap — how to win, not tie

> **⚠️ Superseded for prioritization by [`competitive-plan.md`](competitive-plan.md)**
> (written after the learned-sparse channel + ColBERT reranker shipped). That
> plan is the decision-ready, adversarially-verified build order; this document
> remains the fuller competitor analysis and feature catalogue it was distilled
> from. Tier 1 (A learned-sparse, B ColBERT) below is **SHIPPED**.

**North star:** engram is *the* retrieval **tool** an agent reaches for to talk
to documents — best-in-class **document embeddings**, **semantic search**, and
**search-result quality**. engram does **not** generate answers; it hands the
calling agent the most relevant, complete, well-ranked **context** (including
multi-hop bridges and thematic/global context), and the agent writes the answer.
The win condition: when an agent needs document retrieval, the right tool to
plug in is engram.

Synthesis of a competitor study (GraphRAG, LightRAG, HippoRAG 2, RAGFlow,
LlamaIndex) against engram's architecture and our controlled benchmark result.
Goal: turn the multi-hop **tie** into a **clear** advantage on *retrieval
quality* — the context engram returns must be measurably better than any
alternative an agent could call.

## The interface gap (engram is consumed by agents)

engram exposes HTTP today. Agents increasingly discover and call tools over
**MCP** (Model Context Protocol). To *be the retrieval tool agents reach for*:

- **MCP server** — expose engram's capabilities as MCP tools (`search`,
  `get_chunk_context`, `list/search_communities`, `ingest`) with clear schemas
  and descriptions, so any MCP-speaking agent (Claude, etc.) plugs engram in as
  a drop-in retrieval tool. A thin adapter over the existing FastAPI handlers
  (same store/pipeline); ship it as `engram-mcp` (stdio + HTTP/SSE). **Effort M.
  This is how engram gets adopted as the tool, not just an API.** Pair the tool
  descriptions with the per-result score provenance engram already returns so
  the agent can reason about *why* a result was retrieved.

So the north-star stack is two layers engram owns, plus the interface:
1. **Best retrieval** (Tier 1 below) → best document-embeddings + semantic search.
2. **Best reasoning-retrieval** (Tier 2/3) → multi-hop + global *context* (not
   answers), so the agent can answer hard questions from what engram returns.
3. **First-class agent interface** (MCP) → engram is the tool that gets called.

(Answer generation, citations, faithfulness checking = the *agent's* job, not
engram's. engram's job is to make the agent's answer perfect by feeding it
perfect context.)

## The competitors, honestly

| solution | stars | real edge | key weakness |
|---|---|---|---|
| **Microsoft GraphRAG** | ~34k | global/sensemaking via Leiden community summaries (answers "main themes" vector RAG can't) | ~1000× indexing cost; costly global queries; no incremental update; only LLM-judge evals |
| **LightRAG** | ~37k | cheap + **incremental** updates; dual-level (entity/relation) keyword retrieval | shallow **1-hop only**; quality bound by LLM extraction; needs ≥32B model; thin win vs GraphRAG |
| **HippoRAG 2** | ~3.8k | **query→triple PPR** seeding → strict multi-hop upgrade, *no* simple-QA regression (+13.9 R@5 on 2Wiki) | LLM OpenIE per corpus; immature production plumbing; gains concentrated on 2-hop+ |
| **RAGFlow** | ~84k | **DeepDoc** layout/table parsing; **3-way hybrid (dense+sparse+BM25) + ColBERT** late-interaction rerank | heavy deploy; ColBERT tensor storage blows up ~150×; no published QA accuracy |
| **LlamaIndex** | ~50k | breadth: composable retrievers (auto-merge, recursive, agentic, PropertyGraph) | a framework not a tuned system; tuning burden on you; no first-party accuracy benchmark |

**Two patterns jump out:** (1) the strongest *measured* retrieval result in the
field is RAGFlow/Infinity's **3-way hybrid + ColBERT rerank** — and engram
already runs the exact model (BGE-M3) that produces all three signals but uses
only the dense one. (2) The graph-RAG winners (HippoRAG 2, GraphRAG-DRIFT,
StepChain) all beat a SOTA embedder on multi-hop via **query decomposition /
iterative re-seeding + a real entity graph** — not the single-pass keyword
expansion engram does today.

## The winning thesis

engram is uniquely positioned: it **already pays for BGE-M3** (so sparse +
ColBERT are *free* upgrades others need extra models for) and **already has the
graph machinery** (registries, keyword graph, PPR, a *dormant* community layer).
So the highest-impact features are cheap *extensions* for engram but core
rewrites for competitors. The plan:

1. **Activate BGE-M3's free signals** → widen the retrieval lead (beats RAGFlow's hybrid with no new model).
2. **Add iterative decomposition + a real entity KG + entity-seeded PPR** → convert the multi-hop tie into a win (HippoRAG-2 recipe, improved).
3. **Light up the dormant community layer** for sensemaking → a capability dense+rerank *cannot* match by construction.
4. All behind **adaptive routing** so factoid queries stay fast (no regression — HippoRAG-2's selling point).

---

## Prioritized roadmap

### Tier 1 — Free wins (exploit BGE-M3's discarded outputs). **VALIDATED** (`bench/hybrid_probe.py`).

> **Measured on SciFact (BGE-M3, nDCG@10):** dense (engram today) **0.642** →
> dense+sparse **0.682** (+4.0) → dense+sparse+ColBERT **0.700** (+5.8). The
> retrieval stage jumps ~+9% for free (same model). This is the highest-confidence
> clear-advantage feature — **build next.**

- **A. Learned-sparse channel — SHIPPED (opt-in, default off), with an honest
  caveat.** BGE-M3's `lexical_weights` are fetched at ingest, stored per chunk
  (Neo4j JSON / pgvector `JSONB`), and folded into the **fused score** as an
  exact-term re-scoring of the candidate pool (one batched `get_sparse_weights`
  read; no sparse index; degrades to zeros when down). **Measured (`BENCH_SPARSE=1`,
  SciFact, 4 configs):** at engram's **default rerank depth (15)** sparse is a
  **clear win — +1.1 nDCG@10 / +1.8 recall@10** (best config measured: 0.7428);
  reranker-OFF +1.6 / +2.3; it only washes out at an *artificially deep* rerank
  depth (100), where the cross-encoder re-scores nearly the whole pool (see
  [RESULTS.md §1c](../bench/RESULTS.md)). Mechanism: sparse improves the
  *shortlist*, so it pays exactly when the shortlist is the bottleneck (the
  realistic case). **Candidate-expansion (F3) was evaluated and DROPPED:**
  `/eval` per-channel attribution shows the dense channel surfaces *all* SciFact
  gold hits (fulltext/graph are redundant subsets; 9 unique to dense), so a
  sparse *retrieval* channel has no headroom there, and engram already runs BM25
  for exact terms. Sparse's value is **re-ranking** the pool (measured), not new
  recall — the shipped re-scoring integration is the right one.
- **B. ColBERT late-interaction reranker — SHIPPED** as the `colbert` `RERANKERS`
  strategy (MaxSim over BGE-M3 `colbert_vecs` via a `/rerank_colbert` endpoint,
  shortlist-only → no storage blow-up). Honest framing: **not** a quality lift
  over engram's default cross-encoder (bge-reranker-v2-m3 is stronger) — its win
  is being ~100× cheaper, so it's the *fast* late-interaction option (RAGFlow's
  headline feature) for latency-sensitive / reranker-on-CPU deploys. Select per
  request via `reranker_strategy=colbert`.
- **M. Learned fusion weights** — fit per-channel weights (incl. sparse) on a dev
  set; Infinity showed the optimal blend is far from uniform (80% sparse).
  Multiplier on A. **(M)** — next.

### Tier 2 — The multi-hop tie-breaker. Needs a generative LLM (we can serve one on the 4080).
- **C. Iterative / query-decomposition retrieval** — split a complex query into sub-questions, retrieve each, seed graph expansion from hop-1 winners, union → one final rerank. Short-circuits on simple queries. *The direct fix: a 2nd query seeded by the hop-1 entity reaches the bridge passage no single embedding of the original question can.* → **win on 2Wiki/MuSiQue (3–4 hop)**. **(L)**
- **D. LLM entity/relation extractor → typed KG** — per-chunk `(subject, relation, object)` triples + synonym edges, written via engram's **existing** `upsert_entities/relations` + a `GraphProfile`. Turns "keyword co-occurrence" into a traversable reasoning graph. HippoRAG-2 strict-upgrade. **(L)**
- **E. HippoRAG-2 entity-seeded PPR** — seed PageRank on query→triple-matched entity nodes *and* passages (+ an LLM "recognition" filter), not just chunk hits. *Exactly fixes the benchmark root cause* (PPR is redundant today because it's seeded on the same chunks the embedder already found). **(M)**
- **F. Bridge-path traversal expander** — pull chunks on the shortest typed-relation *path between* two seeds (StepChain), not just 1-hop neighbours. **(M)**

### Tier 3 — A capability they can't match by construction
- **G. Global/sensemaking synthesis** — reuse engram's **already-built but unused** community layer (Leiden + LLM reports + `community_vectors`) for GraphRAG-style map-reduce with dynamic community selection (answers LazyGraphRAG's cost critique). Answers "what are the main themes?" — which *no* embedder can retrieve. **(M)**

### Tier 4 — Adoption moats (real-world, beyond leaderboards)
- **H. Eval + retrieval-trace harness** — golden-set scoring endpoint + **per-stage attribution** ("the sparse channel / keyword-graph recovered this gold hit the dense vector missed"). Makes engram's advantage *visible per corpus* and drives self-tuning. The whole category ships only biased LLM-judge win-rates; engram can offer real, reproducible eval. **(M)**
- **I. Structured ingestion + layout-aware chunking** — DeepDoc-class PDF/table/layout parsing behind a `PARSERS` registry, with page/bbox provenance + citations. engram is plaintext-only today; this is RAGFlow's enterprise moat. **(L)**
- **J. Metadata-filtered / multi-tenant retrieval** — tenant_id + attribute filters + recency-decay threaded through every channel and the graph walk. A hard SaaS/compliance gate no competitor benchmark measures. **(M)**
- **K. Parent-child / auto-merging retrieval** — small chunks for precision, return the coherent parent for context (LlamaIndex/RAGFlow primitive). **(M)**

### Tier 5 — Orchestration & quality multipliers
- **L. Adaptive query routing** — classify factoid / multi-hop / global and dispatch to flat / iterative / community mode. Pays the heavy cost only when it helps → captures multi-hop+global wins with **no factoid regression**. **(M)**
- **N. Prompt auto-tuning for extraction** — domain-adaptive entity/relation prompts (GraphRAG ships this; multiplies the KG features, helps small local LLMs). **(M)**

---

## Recommended sequence

1. **Tier 1 (A+B+M)** first — free, highest-confidence, benchmarkable *now* without an LLM; widens our existing BEIR lead and gives a measured edge over RAGFlow's hybrid.
2. **Stand up a local LLM on the GPU**, then **Tier 2 (C+D+E+F)** — the actual fix for the multi-hop tie; re-benchmark HotpotQA/2Wiki to show tie→win.
3. **L (routing)** to protect factoid speed, then **G** (sensemaking) and the **H/I/J** moats.

Full research with sources + per-feature rationale is in the workflow result.
