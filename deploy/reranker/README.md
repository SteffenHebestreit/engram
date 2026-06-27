# Qwen3-Reranker sidecar — the measured retrieval upgrade

engram's controlled benchmarks found the **reranker is the single highest-leverage
retrieval lever** (it's the one stage nothing downstream washes out). Swapping the
default `bge-reranker-v2-m3` for **Qwen3-Reranker** lifted engram by:

| dataset | bge-reranker-v2-m3 | Qwen3-Reranker-0.6B |
|---|---|---|
| SciFact nDCG@10 | 0.7408 | **0.7723 (+3.15)** |
| NFCorpus nDCG@10 | 0.3411 | **0.3795 (+3.84)** |

It's a drop-in, **multilingual** (100+ languages, like bge-reranker-v2-m3) upgrade.
The catch: Qwen3-Reranker is a *causal-LM* reranker, so TEI's classifier rerank
endpoint can't serve it. This sidecar can — it loads it via sentence-transformers
`CrossEncoder` and speaks engram's reranker wire format (both `tei` and `jina`).

## Run it with engram

```bash
# from the repo root — overlay the reranker onto the root stack
docker compose -f docker-compose.yml -f deploy/reranker/docker-compose.yml up --build
```

The overlay starts the sidecar and points engram's `api` at it
(`RERANKER_API_BASE` / `RERANKER_MODEL` / `RERANKER_FORMAT=tei`).

## Standalone

```bash
docker build -t engram-reranker deploy/reranker
docker run --rm -p 8091:8091 engram-reranker
curl -s localhost:8091/rerank -H 'content-type: application/json' \
     -d '{"query":"how do I rotate the signing key?","texts":["rotate keys via /admin","unrelated text"]}'
# -> [{"index":0,"score":...},{"index":1,"score":...}]
```

## Notes

- **Model size.** `Qwen3-Reranker-0.6B` is the default (fast, the measured win).
  `Qwen3-Reranker-4B` is stronger still (69.76 vs bge's 57.03 on BEIR) but wants a
  GPU with enough memory — see [bench/PENDING.md](../../bench/PENDING.md).
- **GPU.** CPU image by default (fine for the 0.6B on small shortlists). For the
  4B/8B, build `FROM` an `nvidia/cuda` + torch base, set `USE_FP16=true`, and run
  with the NVIDIA Container Toolkit.
- **Degrades gracefully.** If the sidecar is down, engram falls back to the fused
  score (same as any reranker-down), so this is a quality upgrade, not a new hard
  dependency.
- The model loads **lazily** on first request; the first call pays the load cost.
