# engram competitive scorecard (2026-06)

A decision-oriented snapshot: where engram stands against the most popular
retrieval/memory solutions **after this development run** (multi-tenancy,
contextual retrieval, recency, reranker sidecar, adaptive routing, feedback —
all built + tested, awaiting merge). Companion to the deeper
[competitive-plan.md](competitive-plan.md) / [competitive-roadmap.md](competitive-roadmap.md);
this one is the "so where are we, and what's the move" summary.

Grounded in a June-2026 web scan (sources at the bottom). Honest-framing rule:
**TS** = table-stakes/parity, **CA** = clear advantage, **GAP** = we lag.

## The field (and what category each is in)

engram is a **retrieval + memory engine** an agent calls over MCP/HTTP — *not*
an answer generator and *not* an orchestration framework. So the comparison
splits by category:

| Solution | Category | Their notable edge (2026) |
|---|---|---|
| **LangChain / LangGraph** | Orchestration framework | 700+ integrations; LangGraph agentic-RAG graphs (query-analysis → retrieve → grade → web-fallback → generate); HyDE / ContextualCompression / cross-encoder rerank as components |
| **LlamaIndex** | Orchestration framework | Composable memory blocks (FactExtraction, VectorMemory); "retrieval harness for agents"; the default RAG data layer |
| **RAGFlow (InfiniFlow)** | End-to-end RAG engine | DeepDoc layout-aware parsing (tables/bbox); GraphRAG; RAPTOR/Ψ-RAG hierarchical trees; agentic reasoning loop; tensor (ColBERT) rerank |
| **Weaviate** | Vector DB / platform | Native hybrid (Relative Score Fusion = weighted-sum) + built-in rerankers; multi-tenancy + RBAC + replication; Agent Skills (Feb 2026) |
| **Zep (Graphiti) / Mem0** | Agent-memory layer | Zep: bi-temporal knowledge graph (valid_from/valid_to/invalid_at), 63.8% LongMemEval vs Mem0 49.0%; time-aware fact invalidation |

**Frameworks (LangChain/LlamaIndex) are a different category** — engram is a thing
you *plug into* them, not a competitor to their ecosystem. We win there by being
the best *engine* their agents can call, exposed over MCP. The real head-to-head
is RAGFlow / Weaviate (engines) and Zep / Mem0 (memory).

## Dimension-by-dimension

| Dimension | engram now | vs field | Verdict |
|---|---|---|---|
| Hybrid retrieval (dense+lexical+fusion) | 4 channels, DBSF convex fusion | Weaviate/RAGFlow have hybrid; ours adds 3 dense views + learned-sparse | **TS** |
| Reranking | cross-encoder + Qwen3 sidecar + ColBERT strategy + autocut | parity; Qwen3-4B swing still unmeasured | **TS** (CA pending bench) |
| Graph-augmented ranking | NEXT_CHUNK + HAS_KEYWORD + **PPR proximity in the ranker** | most bolt graph on *beside* retrieval (GraphRAG); ours is *in* the fused score | **CA** |
| Contextual Retrieval | embeddings **+ BM25**, built into ingest | LangChain has ContextualCompression (different); few engines bake in Anthropic-style contextual ingest | **CA** (measurement gated) |
| Multi-tenant isolation | per-doc `tenant_id`, all 4 reads, 0%-leak test | Weaviate has it; frameworks/most engines don't | **TS** (CA vs frameworks) |
| Recency / temporal | exponential decay, post-rerank blend | memory systems have recency; RAG engines usually don't | **TS** (vs memory) / **CA** (vs RAG) |
| Bi-temporal fact invalidation | recency only — **no valid_from/valid_to/invalidation** | Zep's core moat; the memory frontier | **GAP** |
| Judge-free eval + per-channel attribution | `POST /eval`, bootstrap CIs, `unique_to_channel` | competitors ship LLM-judge winrates | **CA** |
| Measured architecture delta | **+1.6–2.2 nDCG / +3–4 recall** over naive dense+rerank, models fixed | almost nobody isolates architecture from model | **CA** |
| Document understanding (layout/tables/PDF) | text in; **no DeepDoc-style parsing** | RAGFlow's headline feature | **GAP** |
| Hierarchical summarization | Leiden community/theme layer | RAGFlow RAPTOR/Ψ-RAG trees — ours is graph-community, not recursive tree | **partial** |
| Agentic retrieval loop | single-shot pipeline + **heuristic** routing | 2026 frontier = a reasoning loop (reformulate/re-search) + *learned* per-query policy | **GAP** (partly by design — the loop lives in the calling agent) |
| Agent interface | **MCP server** (search / context / themes / feedback) | the right surface for "the tool agents reach for" | **CA** |

## Where engram already wins

The **combination in one engine** is rare: graph-in-the-ranker + contextual
retrieval (both halves) + recency + multi-tenancy + judge-free eval + MCP, on two
pluggable backends, with a **measured** architecture delta. Frameworks make you
assemble it; vector DBs lack the graph + memory lifecycle; memory layers lack the
retrieval depth. That intersection is the moat — *provably better context an agent
can plug in*, not a winrate claim.

## Where engram lags — and how to close it (prioritized)

1. **Merge + measure (no new code).** Six tested branches are unmerged; the
   biggest *measured* win available is the **Qwen3-Reranker-4B** swing (+12.7
   BEIR vs bge) + the Contextual-Retrieval on-corpus A/B — both queued in
   [../bench/PENDING.md](../bench/PENDING.md). This converts "cited" into "ours".
2. **Bi-temporal memory (own the "memory" claim vs Zep).** Add edge validity
   (valid_from/valid_to/invalid_at) + supersession to the M1 write-path. This is
   the memory frontier and engram's *name* (…Memory) promises it. Gated on a
   local LLM for contradiction detection (the no-LLM recency slice already
   shipped). Highest-leverage *differentiator* build.
3. **Document understanding (close the RAGFlow gap).** A pluggable layout/table
   parser (DeepDoc-style) at the ingest seam — biggest ingestion-breadth gap;
   could wrap an existing parser rather than build from scratch.
4. **Agentic retrieval mode.** An optional self-critique / iterate-and-re-search
   loop (or richer per-result signals for the *agent's* loop) to match the 2026
   agentic-RAG frontier — natural fit for the MCP tool positioning.
5. **Learned retrieval policy.** Upgrade adaptive-routing from heuristic to
   learned, trained on the **feedback-loop** signal already being collected —
   turns two shipped branches into a compounding advantage.

**Bottom line:** engram is competitive *as an engine today* and differentiated on
graph-in-ranker + eval honesty + the feature intersection. The clear-advantage
upside is (a) proving the gated measurements and (b) bi-temporal memory to win the
memory category outright. Building more un-gated retrieval features has hit
diminishing returns — the next leverage is merge → measure → memory.

## Sources

- [RAGFlow releases](https://ragflow.io/docs/release_notes) · [RAGFlow repo](https://github.com/infiniflow/ragflow) · [RAGFlow: From RAG to Context (2025 review)](https://ragflow.io/blog/rag-review-2025-from-rag-to-context)
- [LlamaIndex OSS frameworks](https://www.llamaindex.ai/llamaindex) · [Best AI agent memory frameworks 2026](https://machinelearningmastery.com/the-6-best-ai-agent-memory-frameworks-you-should-try-in-2026/)
- [Weaviate hybrid search docs](https://docs.weaviate.io/weaviate/search/hybrid) · [Weaviate reranking](https://docs.weaviate.io/weaviate/concepts/reranking) · [Weaviate platform](https://weaviate.io/platform)
- [Zep vs Mem0 (Graphiti)](https://vectorize.io/articles/mem0-vs-zep) · [State of AI agent memory 2026](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [Next-gen agentic RAG with LangGraph (2026)](https://medium.com/@vinodkrane/next-generation-agentic-rag-with-langgraph-2026-edition-d1c4c068d2b8) · [Enterprise RAG platforms comparison 2026](https://atlan.com/know/enterprise-rag-platforms-comparison/)
