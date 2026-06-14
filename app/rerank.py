import httpx

from .config import get_settings


async def rerank(
    client: httpx.AsyncClient, query: str, texts: list[str]
) -> list[float] | None:
    """Score each text against the query. Returns one score per text in input
    order, an empty list when there is nothing to rank, or None when the
    reranker endpoint is unavailable or returns something unusable.

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
        resp = await client.post(url, json=body, headers=headers, timeout=120)
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
