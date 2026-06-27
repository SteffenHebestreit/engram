# Pending benchmark runs (need a higher-memory GPU)

These runs were **blocked by the 16 GB RTX 4080**, not by the method. A 4B
embedder barely fits 16 GB alongside a reranker (97% VRAM → memory-thrashing →
~4× slowdown → a ~2 h run), and an 8B model + reranker doesn't fit at all. They
are queued for a box with more (ideally **unified**) memory — e.g. a **Strix Halo
/ Ryzen AI Max** with 96–128 GB unified memory, where 4B/8B models fit easily.

> **Hardware note for Strix Halo (AMD):** the bench's `docker-compose.gpu.yml`
> reserves an **NVIDIA** device. On AMD you need a **ROCm** device mapping
> instead (`/dev/kfd` + `/dev/dri`, `HSA_OVERRIDE_GFX_VERSION` for gfx1151) and a
> ROCm PyTorch wheel in `bench/Dockerfile.gpu`. Alternatively run **CPU-only**
> (drop the gpu override) — slow on compute but the 128 GB unified memory removes
> the OOM/thrash bottleneck entirely, so the big models at least *run*.

## What we already measured (the baselines to compare against)

| config | engram nDCG@10 (SciFact / NFCorpus) | source |
|---|---|---|
| BGE-M3 + bge-reranker-v2-m3 | 0.7408 / 0.3411 | RESULTS §1 |
| BGE-M3 + **Qwen3-Reranker-0.6B** | **0.7723 / 0.3795** | RESULTS §1e |
| Qwen3-Embedding-0.6B + bge-reranker | 0.7409 / — | RESULTS §1d *(prompt-confounded; superseded)* |

`engram − dense+rerank` is a positive *point estimate* (~+1.4 nDCG) but, on a
rigorous re-run (bootstrap CIs + paired sign tests + a hybrid+rerank control), is
**NOT statistically significant** (95%CI straddles 0; sign-p > 0.05) and is a
**tie vs hybrid+rerank** — most of it is the BM25 channel, not graph/median/MMR.
The **verified, robust** wins are: the **reranker** (+3–4 nDCG, every system),
and the **backend/quant** (engramdb ties Neo4j / beats pgvector, b1 ties f32,
faster). See [docs/engram-db.md](../docs/engram-db.md) for the corrected analysis.

## The runs to do (each ALONE — never run two GPU benches in parallel: shared DB + GPU; `docker stop` to kill)

All inherit the **prompt fix** (`embedder.default_prompt_name = None`, so engram's
E1 instruction is the only one applied). Set `EMBEDDING_DIM` to the model's native
dim. `QUERY_INSTRUCTION` uses Qwen3's `Instruct: <task>\nQuery:` format
(documents get none — `PASSAGE_INSTRUCTION` stays empty).

### 1. Bigger embedder, isolated (the open "is it negligible?" question)

Does a **stronger embedder** help engram once a **strong reranker** can exploit
its better candidates? Compare against `BGE-M3 + Qwen3-Reranker-0.6B` (= 0.7723).

```bash
# Qwen3-Embedding-4B (2560-dim) + Qwen3-Reranker-0.6B
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm --build \
  -e BENCH_EMBED_MODEL=Qwen/Qwen3-Embedding-4B -e EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B \
  -e EMBEDDING_DIM=2560 \
  -e "QUERY_INSTRUCTION=Instruct: Given a web search query, retrieve relevant passages that answer the query
Query:" \
  -e BENCH_RERANK_MODEL=Qwen/Qwen3-Reranker-0.6B -e BENCH_DATASETS=scifact,nfcorpus \
  runner python -m bench.compare

# Qwen3-Embedding-8B (4096-dim) — only on the big-memory box
#   ...same as above but EMBED/EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B, EMBEDDING_DIM=4096
```

**Decision rule:** if engram beats 0.7723 → a bigger embedder *does* earn its keep
(retract "negligible"); if it ties → the reranker truly caps it even at 4B/8B,
and BGE-M3 stays the robust default.

### 2. Bigger reranker (the biggest swing left)

Qwen3-Reranker-**4B** scores **69.76 nDCG@10 on BEIR vs bge's 57.03 (+12.7)** —
the 0.6B already gave engram +3.15/+3.84; the 4B could push further. Keep BGE-M3
so only the reranker changes; compare against the 0.6B-reranker rows above.

```bash
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm \
  -e BENCH_RERANK_MODEL=Qwen/Qwen3-Reranker-4B -e BENCH_DATASETS=scifact,nfcorpus \
  runner python -m bench.compare
# Qwen3-Reranker-4B is a CrossEncoder via sentence-transformers (no harness change).
```

### 3. All-SOTA combo

`Qwen3-Embedding-4B (or 8B) + Qwen3-Reranker-4B` — the ceiling of what these open
models can do on engram. Compare engram vs `dense+rerank` to confirm the
**architecture delta still holds (and grows) at the top of the model stack**.

### 4. Chunking / overlap ablation (needs a LONG-document corpus, not BEIR)

`CHUNK_OVERLAP_CHARS` now defaults to **0** (the `NEXT_CHUNK` graph recovers seam
context). BEIR docs are single-chunk so this is unmeasurable there. On a
long-document set (or a customer corpus via `POST /eval`), A/B `dedup`/overlap and
a future `semantic` chunker to *quantify* the "chunking is a robustness property"
claim (RESULTS §"the architecture, not the models").

### 5. Contextual Retrieval (needs an LLM at ingest + a multi-doc corpus)

`CONTEXTUAL_RETRIEVAL_ENABLED=true` prepends an LLM-written document-situating
context to each chunk before embedding (Anthropic reports **−35% retrieval
failures** for contextual embeddings, **−49%** with contextual BM25 too). It is a
**geometry** change — it disambiguates near-identical chunks across *different*
documents — so **BEIR can't show it**: SciFact/NFCorpus docs are single-chunk, so
there is no within-corpus ambiguity for the context to resolve. Measure on a
**long-document, multi-document** corpus (or a customer corpus via `POST /eval`)
where the same phrasing recurs across documents:

```bash
# A/B over a long-doc corpus: baseline vs contextual embeddings.
# Needs LLM_API_BASE reachable at ingest (one extra call per chunk).
#   ingest twice (CONTEXTUAL_RETRIEVAL_ENABLED false / true) into separate
#   stores, then POST /eval the same golden set against each.
```

Both halves ship: **contextual embeddings** (context prepended before embedding)
and **contextual BM25** (the context is also indexed for fulltext — Neo4j indexes
`c.context`; pgvector adds a `context_tsv` generated column). So the A/B can also
isolate the two — embeddings-only vs +BM25 — to attribute the lift per channel.

**Decision rule:** report the nDCG@10 / recall@10 delta from engram's own
`/eval` (judge-free) — claim a win only if it clears the baseline beyond the
bootstrap CI. Until measured on such a corpus it ships as an opt-in feature with
the citation, not a claimed number.

## Also worth doing on the bigger box

- **Local generative LLM** (the unified memory fits a 14–32B model): unblocks
  LLM entity-KG extraction and query decomposition for multi-hop — currently gated
  on no online LLM. See [docs/memory-writepath-plan.md](../docs/memory-writepath-plan.md).
