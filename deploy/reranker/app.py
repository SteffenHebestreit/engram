"""Reference reranker sidecar for engram — serves Qwen3-Reranker (or any
sentence-transformers CrossEncoder) in engram's reranker wire format.

engram's controlled benchmarks found the **reranker** is the highest-leverage
retrieval lever: swapping `bge-reranker-v2-m3` for **Qwen3-Reranker** lifts engram
+3.15 / +3.84 nDCG@10 (SciFact / NFCorpus) — a drop-in, multilingual upgrade
(see bench/RESULTS.md §1e). But Qwen3-Reranker is a causal-LM reranker, so TEI's
classifier rerank endpoint can't serve it. This sidecar can: it loads it via
sentence-transformers `CrossEncoder` and exposes engram's reranker endpoints.

Point engram at it:
  RERANKER_API_BASE=http://reranker:8091
  RERANKER_MODEL=Qwen/Qwen3-Reranker-0.6B
  RERANKER_FORMAT=tei            # this sidecar speaks both "tei" and "jina"

Endpoints (match app/rerank.py's two supported formats):
  POST /rerank  {"query", "texts": [...]}                 -> [{"index", "score"}]            (tei)
  POST /rerank  {"query", "documents": [...], "top_n"?}   -> {"results": [{"index","relevance_score"}]} (jina)
A single handler detects which by the `texts` vs `documents` key.

MODEL_NAME / USE_FP16 / MAX_LENGTH are env-configurable. The model loads lazily on
first request so the container starts fast and a readiness probe can gate it.
"""

import os

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3-Reranker-0.6B")
USE_FP16 = os.environ.get("USE_FP16", "true").lower() == "true"
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))

app = FastAPI(title="engram reranker sidecar")
_model = None


def model():
    """Load the CrossEncoder once, on first use (keeps startup cheap)."""
    global _model
    if _model is None:
        import torch
        from sentence_transformers import CrossEncoder

        kwargs = {"max_length": MAX_LENGTH}
        if USE_FP16 and torch.cuda.is_available():
            kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
        _model = CrossEncoder(MODEL_NAME, **kwargs)
    return _model


class RerankRequest(BaseModel):
    query: str
    # tei callers send `texts`; jina callers send `documents` — accept either
    texts: list[str] | None = None
    documents: list[str] | None = None
    model: str | None = None
    top_n: int | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "loaded": _model is not None}


@app.post("/rerank")
def rerank(req: RerankRequest):
    """Score each text/document against the query. Returns the **jina** shape
    when called with `documents`, else the **tei** shape — so engram's `tei` and
    `jina` reranker formats both work against this one endpoint."""
    jina = req.documents is not None
    texts = req.documents if jina else (req.texts or [])
    if not texts:
        return {"results": []} if jina else []

    scores = [float(s) for s in model().predict([(req.query, t) for t in texts])]
    if jina:
        ranked = sorted(
            ({"index": i, "relevance_score": s} for i, s in enumerate(scores)),
            key=lambda r: r["relevance_score"],
            reverse=True,
        )
        if req.top_n:
            ranked = ranked[: req.top_n]
        return {"results": ranked}
    return [{"index": i, "score": s} for i, s in enumerate(scores)]
