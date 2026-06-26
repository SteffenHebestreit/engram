from typing import Awaitable, Callable

import httpx

from .config import get_settings
from .registry import Registry

# A reranker scores `texts` against `query`, returning one score per text in
# input order, [] when there is nothing to rank, or None when it is unavailable
# (so the pipeline can fall back to the fused score). Swap the cross-encoder for
# a different provider/algorithm by registering a new strategy.
Reranker = Callable[[httpx.AsyncClient, str, list[str]], Awaitable["list[float] | None"]]
RERANKERS: Registry[Reranker] = Registry("reranker")


def get_reranker(name: str) -> Reranker:
    return RERANKERS.get(name)


@RERANKERS.register("http")
async def rerank(
    client: httpx.AsyncClient, query: str, texts: list[str]
) -> list[float] | None:
    """Score each text against the query via an HTTP cross-encoder endpoint.
    Returns one score per text in input order, an empty list when there is
    nothing to rank, or None when the reranker endpoint is unavailable or
    returns something unusable.

    Returning None (rather than raising) lets the search pipeline fall back to
    the fused score and keep working when the reranker is down, the same way
    HyDE and PPR degrade. Any connection error, HTTP error or malformed
    response collapses to None.

    Supports two wire formats:
      tei:  POST {base}/rerank {"query", "texts"}           -> [{"index", "score"}]
      jina: POST {base}/rerank {"model", "query", "documents"}
            -> {"results": [{"index", "relevance_score"}]}
    """
    if not texts:
        return []
    settings = get_settings()

    headers = {}
    if settings.reranker_api_key:
        headers["Authorization"] = f"Bearer {settings.reranker_api_key}"

    url = f"{settings.reranker_api_base.rstrip('/')}/rerank"
    if settings.reranker_format == "jina":
        body = {
            "model": settings.reranker_model,
            "query": query,
            "documents": texts,
            "top_n": len(texts),
        }
    else:
        body = {"query": query, "texts": texts}

    try:
        resp = await client.post(
            url, json=body, headers=headers, timeout=settings.request_timeout
        )
        resp.raise_for_status()
        payload = resp.json()

        items = payload["results"] if isinstance(payload, dict) else payload
        scores = [0.0] * len(texts)
        for item in items:
            score = item.get("relevance_score", item.get("score", 0.0))
            scores[item["index"]] = float(score)
        return scores
    except Exception:
        return None


@RERANKERS.register("colbert")
async def rerank_colbert(
    client: httpx.AsyncClient, query: str, texts: list[str]
) -> list[float] | None:
    """ColBERT late-interaction reranker: score the shortlist by MaxSim over
    BGE-M3 multi-vectors via a dedicated endpoint.

    Same drop-in contract as the cross-encoder strategy — one score per text in
    input order, [] for nothing to rank, None when the endpoint is unconfigured
    or unavailable (search falls back to the fused score). This is the *cheap*
    late-interaction option, not a quality upgrade over the default
    cross-encoder; see the colbert_* settings in config.py.

    Endpoint contract:
      POST {colbert_api_base}/rerank_colbert {"model", "query", "texts": [...]}
      -> {"data": [{"index": i, "score": float}]}  (or a bare list of the same)
    """
    if not texts:
        return []
    settings = get_settings()
    if not settings.colbert_api_base:
        return None

    headers = {}
    if settings.colbert_api_key:
        headers["Authorization"] = f"Bearer {settings.colbert_api_key}"
    url = f"{settings.colbert_api_base.rstrip('/')}/rerank_colbert"
    body = {"model": settings.colbert_model, "query": query, "texts": texts}
    try:
        resp = await client.post(
            url, json=body, headers=headers, timeout=settings.request_timeout
        )
        resp.raise_for_status()
        payload = resp.json()
        items = payload["data"] if isinstance(payload, dict) else payload
        scores = [0.0] * len(texts)
        for item in items:
            scores[item["index"]] = float(item["score"])
        return scores
    except Exception:
        return None
