# BGE-M3 multi-output sidecar

A reference embedding service for engram that serves **all three** of BGE-M3's
outputs from one model, one forward pass:

| engram setting | endpoint | signal |
|---|---|---|
| `EMBEDDING_API_BASE` | `POST /v1/embeddings` | dense vectors (OpenAI-compatible) |
| `SPARSE_API_BASE` | `POST /embed_sparse` | learned-sparse term weights |
| `COLBERT_API_BASE` | `POST /rerank_colbert` | ColBERT MaxSim late-interaction rerank |

engram already runs BGE-M3 for dense retrieval but discards the sparse and
ColBERT signals. Point all three env vars at this one container and they light
up with **no extra model** — dense, learned-sparse, and ColBERT from a single
BGE-M3 deployment.

## Run it with engram

```bash
# from the repo root — overlay the sidecar onto the root stack
docker compose -f docker-compose.yml -f deploy/bge-m3/docker-compose.yml up --build
```

The overlay starts the sidecar and sets `SPARSE_ENABLED=true` plus the three
endpoint vars on engram's `api`. To also swap the cross-encoder for the cheap
ColBERT reranker, add `RERANKER_STRATEGY: colbert` to the `api` environment.

## Standalone

```bash
docker build -t engram-bge-m3 deploy/bge-m3
docker run --rm -p 8090:8090 engram-bge-m3
curl -s localhost:8090/embed_sparse -H 'content-type: application/json' \
     -d '{"input": ["ERR_SSL_VERSION_MISMATCH on port 8443"]}'
```

## Notes

- **CPU by default.** Fine for query-time encoding and small/medium corpora. For
  large-corpus ingest, build `FROM` an `nvidia/cuda` + torch base, set
  `USE_FP16=true`, and run with the NVIDIA Container Toolkit.
- The model loads **lazily** on first request, so the container starts fast and
  the healthcheck can gate readiness; the first call pays the load cost.
- Tunables (env): `MODEL_NAME`, `USE_FP16`, `MAX_LENGTH` (512), `BATCH_SIZE` (32).
- It is **optional**. The core engram image never imports it; the dense pipeline
  runs against any OpenAI-compatible `/embeddings` endpoint exactly as before.
- All three engram-side paths degrade gracefully if the sidecar is down (dense
  search falls back to fulltext; sparse and ColBERT fall back to no-ops), so this
  is a quality upgrade, never a new hard dependency.
