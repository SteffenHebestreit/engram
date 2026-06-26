"""Reference multi-output BGE-M3 sidecar for engram.

One BGE-M3 model, one forward pass, three outputs — so a single service backs
*all* of engram's embedding needs:

  POST /v1/embeddings      OpenAI-compatible dense vectors  (EMBEDDING_API_BASE)
  POST /embed_sparse       learned-sparse term weights      (SPARSE_API_BASE)
  POST /rerank_colbert     ColBERT MaxSim late-interaction  (COLBERT_API_BASE)

engram talks to embedding/reranker endpoints over HTTP (see config.py); point
all three of the env vars above at this one container and the discarded sparse +
ColBERT signals light up with no extra model. Optional — the core image does not
depend on it; deploy it (CPU or GPU) only when you turn sparse/ColBERT on.

  EMBEDDING_API_BASE=http://bge-m3:8090/v1
  SPARSE_API_BASE=http://bge-m3:8090
  COLBERT_API_BASE=http://bge-m3:8090
  SPARSE_ENABLED=true            # to use the sparse channel
  RERANKER_STRATEGY=colbert      # to use the cheap late-interaction reranker

Model + fp16 are env-configurable (MODEL_NAME, USE_FP16). The model loads lazily
on first request so the container starts fast and a readiness probe can wait.
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = os.environ.get("MODEL_NAME", "BAAI/bge-m3")
USE_FP16 = os.environ.get("USE_FP16", "true").lower() == "true"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))

app = FastAPI(title="engram BGE-M3 sidecar")
_model = None


def model():
    """Load the BGE-M3 model once, on first use (keeps startup cheap)."""
    global _model
    if _model is None:
        from FlagEmbedding import BGEM3FlagModel

        _model = BGEM3FlagModel(MODEL_NAME, use_fp16=USE_FP16)
    return _model


class EmbeddingsRequest(BaseModel):
    input: list[str] | str
    model: str | None = None


class SparseRequest(BaseModel):
    input: list[str] | str
    model: str | None = None


class ColbertRequest(BaseModel):
    query: str
    texts: list[str]
    model: str | None = None


def _as_list(value: list[str] | str) -> list[str]:
    return [value] if isinstance(value, str) else list(value)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "loaded": _model is not None}


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingsRequest) -> dict:
    """OpenAI-compatible dense embeddings (engram's EMBEDDING_API_BASE)."""
    texts = _as_list(req.input)
    if not texts:
        return {"object": "list", "data": [], "model": MODEL_NAME}
    out = model().encode(
        texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
        return_dense=True, return_sparse=False, return_colbert_vecs=False,
    )["dense_vecs"]
    return {
        "object": "list",
        "model": MODEL_NAME,
        "data": [
            {"object": "embedding", "index": i, "embedding": vec.tolist()}
            for i, vec in enumerate(out)
        ],
    }


@app.post("/embed_sparse")
def embed_sparse(req: SparseRequest) -> dict:
    """BGE-M3 learned-sparse term weights (engram's SPARSE_API_BASE)."""
    texts = _as_list(req.input)
    if not texts:
        return {"data": [], "model": MODEL_NAME}
    weights = model().encode(
        texts, batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
        return_dense=False, return_sparse=True, return_colbert_vecs=False,
    )["lexical_weights"]
    return {
        "model": MODEL_NAME,
        "data": [
            {"index": i, "lexical_weights": {str(t): float(w) for t, w in m.items()}}
            for i, m in enumerate(weights)
        ],
    }


@app.post("/rerank_colbert")
def rerank_colbert(req: ColbertRequest) -> dict:
    """ColBERT late-interaction (MaxSim) rerank of `texts` against `query`
    (engram's COLBERT_API_BASE; select with RERANKER_STRATEGY=colbert)."""
    if not req.texts:
        return {"data": [], "model": MODEL_NAME}
    m = model()
    encoded = m.encode(
        [req.query, *req.texts], batch_size=BATCH_SIZE, max_length=MAX_LENGTH,
        return_dense=False, return_sparse=False, return_colbert_vecs=True,
    )["colbert_vecs"]
    q_vecs, doc_vecs = encoded[0], encoded[1:]
    return {
        "model": MODEL_NAME,
        "data": [
            {"index": i, "score": float(m.colbert_score(q_vecs, dv))}
            for i, dv in enumerate(doc_vecs)
        ],
    }
