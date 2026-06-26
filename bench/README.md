# Benchmarks

Head-to-head comparisons of engram against the standard RAG retrieval strategies
(`bm25`, `dense` = naive vector RAG, `dense+rerank` = standard 2-stage, `engram`).
**Same datasets, same embedding model, same reranker, same metrics — only the
architecture changes**, so the difference is the architecture, not the models.

**📊 Results + analysis: [RESULTS.md](RESULTS.md).**

## The two benchmarks

- **BEIR retrieval** ([compare.py](compare.py)) — SciFact + NFCorpus, nDCG@10 /
  Recall / MAP. Retrieval quality.
- **Multi-hop** ([multihop.py](multihop.py)) — HotpotQA, recall@2/@5 of the
  *linked* supporting passages (HippoRAG's metric). engram runs its full graph
  pipeline (YAKE keyword graph + GDS PageRank).

## The two configs

| config | embedder | reranker | where |
|---|---|---|---|
| **production** | `BAAI/bge-m3` (1024-d) | `BAAI/bge-reranker-v2-m3` | GPU (NVIDIA Container Toolkit) |
| **floor** | `all-MiniLM-L6-v2` (384-d) | `ms-marco-MiniLM-L-6-v2` | CPU |

The harness wires the models straight into engram's seams (no HTTP shim), so the
real pipeline runs unchanged. Models are configurable via `BENCH_EMBED_MODEL` /
`BENCH_RERANK_MODEL`.

## Run

```bash
# CPU floor
docker compose -f bench/docker-compose.yml run --rm --build runner python -m bench.compare
docker compose -f bench/docker-compose.yml run --rm --build multihop

# production stack on GPU (needs NVIDIA Container Toolkit)
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm --build runner python -m bench.compare
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm --build multihop
```

Runs as an isolated `engrambench` compose project on a throwaway database — it
never touches your dev store. Model downloads + datasets are cached across runs.
`bge-reranker-v2-m3` is heavy (~17 min per BEIR dataset on a 4080); the CPU floor
is faster per-pair but caps out on a weak embedder.

### Measuring the learned-sparse lift

Set `BENCH_SPARSE=1` to add an `engram+sparse` system: a second engram pass with
the BGE-M3 learned-sparse channel on (`SPARSE_WEIGHT` tunes its share, default
0.2). It re-ingests so the sparse term-weights are stored, loads a local
FlagEmbedding BGE-M3 for the sparse vectors, and reports `engram` vs
`engram+sparse` side by side — the same model, only the discarded sparse signal
added. Probe (`hybrid_probe.py`) isolated the lift at +4.0 nDCG@10 on SciFact;
this measures it inside the full pipeline (where the cross-encoder re-scores the
shortlist, so the carry-through differs).

```bash
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml \
    run --rm --build -e BENCH_SPARSE=1 -e BENCH_DATASETS=scifact \
    runner python -m bench.compare
```

## What it does *not* exercise

No generative LLM is available in this environment, so HyDE and **LLM** keyword
extraction are off — the multi-hop graph is built from **YAKE** statistical
keywords (LLM/entity extraction would link it more precisely). BEIR docs are
single-chunk, so sequence/PPR expansion only matters on multi-hop.
