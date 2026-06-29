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

## Picking an extraction model ([extractor_bench.py](extractor_bench.py))

A different question from the retrieval benchmarks above: *if* you turn on the
opt-in LLM metadata extractor (`METADATA_EXTRACTOR=default`), which model should
it use? Extraction is the high-volume per-chunk call (3-8 keywords + a
one-sentence summary as JSON), so you want the smallest model that's still fast
*and* faithful. This harness compares any models you can serve, head-to-head,
over engram's **real** extraction prompt + parser, with the no-LLM `yake` path as
the baseline row.

It reports, per model: throughput (chunks/s at a concurrency), p50/p95 latency,
**mean completion tokens** (a bloated count instantly outs a Qwen3 *thinking*
model run in the wrong mode), strict-JSON rate, schema adherence (3-8 keywords +
one sentence), and judge-free **faithfulness proxies** (cosine of the summary and
of the keywords to the chunk, via a local embedder).

```bash
cat > models.json <<'JSON'
[ {"label":"qwen3-1.7b","api_base":"http://localhost:8001/v1","model":"Qwen/Qwen3-1.7B-Instruct-2507","max_tokens":96,"json_mode":true},
  {"label":"qwen3-0.6b","api_base":"http://localhost:8002/v1","model":"Qwen/Qwen3-0.6B","max_tokens":96,"json_mode":true} ]
JSON
python -m bench.extractor_bench --models models.json --corpus docs/ --concurrency 8 --max-chunks 100 --out extractor.json
```

`--use-settings` also tests whatever `EXTRACTION_LLM_*`/`LLM_*` point at;
`--no-quality` skips the embedder (perf + reliability only); omit `--corpus` for
the built-in sample chunks. The proxies are a fast iteration loop — the **gold**
quality check is still downstream retrieval nDCG: run the BEIR/multi-hop
benchmarks above with the chosen extractor on a multi-chunk corpus.

## Evaluating a LIVE model stack ([live_eval.py](live_eval.py))

`compare.py`/`run_benchmark.py` wire *local* sentence-transformers models into the
seams. `live_eval.py` instead drives engram's **real HTTP pipeline** against a live
OpenAI-compatible stack (LM Studio / vLLM / TEI / ...) over an in-process `engramdb`
store — so you can pick models on *your* hardware with real BEIR nDCG, including
instruction-tuned embedders served correctly. It reuses `compare.py`'s
`paired_delta` (bootstrap CI + sign test), writes per-query scores (`BENCH_OUT`),
and compares against a previous run (`BENCH_COMPARE_TO`).

```bash
# A/B two embedders with a paired significance test (reranker off isolates the embedder):
common="STORE_BACKEND=engramdb SCHEMA_GUARD_MODE=off EMBEDDING_API_BASE=http://host:1234/v1 \
  EMBEDDING_DIM=1024 RERANKER_ENABLED=false HYDE_ENABLED=false SPARSE_ENABLED=false \
  METADATA_EXTRACTOR=none SUMMARY_CHANNEL_ENABLED=false KEYWORDS_CHANNEL_ENABLED=false \
  BENCH_DATA_DIR=./beir BENCH_DATASET=scifact"
env $common EMBEDDING_MODEL=BAAI/bge-m3 BENCH_OUT=bge.json python -m bench.live_eval
env $common EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
  QUERY_INSTRUCTION=$'Instruct: Given a search query, retrieve relevant passages\nQuery: ' \
  BENCH_OUT=qwen.json BENCH_COMPARE_TO=bge.json python -m bench.live_eval
```

**Measuring the reranker leg (the actual quality lever).** A server like LM Studio
can't serve a reranker, so load one in-process with `BENCH_LOCAL_RERANKER` (needs
`torch`+`sentence-transformers`, ideally a GPU) and run `RERANKER_ENABLED=true
RERANKER_STRATEGY=local`. This answers the §1d question on *your* data — does a
strong reranker wash out the embedder gap?

```bash
env $common_but_reranker_on EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B \
  RERANKER_ENABLED=true RERANKER_STRATEGY=local \
  BENCH_LOCAL_RERANKER=Qwen/Qwen3-Reranker-0.6B python -m bench.live_eval
```

Measured so far (see RESULTS §1d-live): qwen3-embedding-0.6b beats bge-m3 on SciFact
content-only by a **significant** +3.4 nDCG@10 (p=0.003) — but raw-dense only; §1d
shows the reranker likely collapses it, and bge-m3 is the cross-domain-robust default.

## What it does *not* exercise

No generative LLM is available in this environment, so HyDE and **LLM** keyword
extraction are off — the multi-hop graph is built from **YAKE** statistical
keywords (LLM/entity extraction would link it more precisely). BEIR docs are
single-chunk, so sequence/PPR expansion only matters on multi-hop.
