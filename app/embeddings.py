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
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()["data"]
    # the API may return items out of order; restore input order
    data.sort(key=lambda item: item["index"])
    return [item["embedding"] for item in data]


async def embed_text(client: httpx.AsyncClient, text: str) -> list[float]:
    return (await embed_texts(client, [text]))[0]
