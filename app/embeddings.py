import asyncio

import httpx

from .config import get_settings


async def embed_texts(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts via the OpenAI-compatible /embeddings endpoint.

    Large inputs are split into batches of `embedding_batch_size` with up to
    `embedding_concurrency` requests in flight: embedding servers commonly cap
    the per-request batch size, and several bounded requests pipeline better
    than one oversized one. Output order matches input order.
    """
    if not texts:
        return []
    settings = get_settings()
    batch_size = max(1, settings.embedding_batch_size)
    if len(texts) <= batch_size:
        return await _embed_batch(client, texts)

    semaphore = asyncio.Semaphore(max(1, settings.embedding_concurrency))

    async def bounded(batch: list[str]) -> list[list[float]]:
        async with semaphore:
            return await _embed_batch(client, batch)

    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    results = await asyncio.gather(*(bounded(b) for b in batches))
    return [emb for part in results for emb in part]


async def _embed_batch(client: httpx.AsyncClient, texts: list[str]) -> list[list[float]]:
    settings = get_settings()

    headers = {}
    if settings.embedding_api_key:
        headers["Authorization"] = f"Bearer {settings.embedding_api_key}"

    resp = await client.post(
        f"{settings.embedding_api_base.rstrip('/')}/embeddings",
        json={"model": settings.embedding_model, "input": texts},
        headers=headers,
        timeout=settings.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # the API may return items out of order; restore input order
    data.sort(key=lambda item: item["index"])
    return [item["embedding"] for item in data]


async def embed_text(client: httpx.AsyncClient, text: str) -> list[float]:
    return (await embed_texts(client, [text]))[0]


async def embed_sparse_texts(
    client: httpx.AsyncClient, texts: list[str]
) -> list[dict[str, float]] | None:
    """BGE-M3 learned-sparse term weights per text from the multi-output endpoint.

    Returns one `{token: weight}` map per text, or None on any failure / when no
    sparse endpoint is configured — sparse is opt-in and degrades gracefully
    (the dense channels carry the search), like the HyDE / reranker fallbacks.
    """
    if not texts:
        return []
    settings = get_settings()
    if not settings.sparse_api_base:
        return None

    headers = {}
    if settings.sparse_api_key:
        headers["Authorization"] = f"Bearer {settings.sparse_api_key}"
    try:
        resp = await client.post(
            f"{settings.sparse_api_base.rstrip('/')}/embed_sparse",
            json={"model": settings.sparse_model, "input": texts},
            headers=headers,
            timeout=settings.request_timeout,
        )
        resp.raise_for_status()
        data = resp.json()["data"]
        data.sort(key=lambda item: item["index"])
        return [
            {str(k): float(v) for k, v in item["lexical_weights"].items()}
            for item in data
        ]
    except Exception:
        return None


async def embed_sparse_text(
    client: httpx.AsyncClient, text: str
) -> dict[str, float] | None:
    out = await embed_sparse_texts(client, [text])
    return out[0] if out else None
