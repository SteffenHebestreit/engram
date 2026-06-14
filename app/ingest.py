import asyncio
import hashlib

import httpx
from neo4j import AsyncDriver

from . import graph
from .channels import get_channel_source, resolve_vector_channels
from .chunking import get_chunker
from .config import get_settings
from .embeddings import embed_texts
from .llm import get_extractor


def compute_document_id(text: str, explicit: str | None = None) -> str:
    """Stable identifier for a document.

    A caller can pass its own `document_id` (its existing handle for the doc in
    whatever "context" it tracks); otherwise the id is the SHA-256 of the text,
    so the same content always maps to the same id. Either way the id is
    *recomputable*, so the document can be deleted later without having stored
    the value we returned — and re-ingesting the same document replaces it
    instead of creating a duplicate.
    """
    if explicit:
        cleaned = explicit.strip()
        if cleaned:
            return cleaned
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def ingest_document(
    driver: AsyncDriver,
    http: httpx.AsyncClient,
    text: str,
    title: str = "",
    source: str = "",
    document_id: str | None = None,
) -> tuple[str, int, list[str]]:
    """Full ingestion pipeline: chunk -> LLM metadata -> 3x embeddings -> graph.

    Returns (document_id, chunk_count, distinct keywords). A non-empty `source`
    is required: every document is reference-counted by the sources that pulled
    it in, so it can always be cleaned up (no document can sit unreferenced and
    leave chunks orphaned). Re-ingesting the same id replaces the previous
    version (its chunks, edges and orphaned keywords are removed first).
    """
    source = source.strip()
    if not source:
        raise ValueError("a non-empty source is required")

    settings = get_settings()
    doc_id = compute_document_id(text, document_id)
    content_addressed = document_id is None

    existing = await graph.get_document(driver, doc_id)

    # fast path: the same content is already ingested (the id is the text hash),
    # so the chunks are identical — just register this source instead of
    # re-chunking and re-embedding. Multiple sources can reference one document;
    # its nodes are only deleted once the last source is removed.
    if existing is not None and content_addressed:
        await graph.add_document_source(driver, doc_id, source)
        return doc_id, existing["chunk_count"], existing["keywords"]

    chunker = get_chunker(settings.chunk_strategy)
    chunks = chunker(text, settings)
    if not chunks:
        raise ValueError("no chunks produced from input text")

    extractor = get_extractor(settings.metadata_extractor)
    semaphore = asyncio.Semaphore(max(1, settings.extraction_concurrency))

    async def extract(chunk: str):
        async with semaphore:
            return await extractor(http, chunk)

    metadata = await asyncio.gather(*(extract(c) for c in chunks))

    # one independent embedding space per active vector channel (default:
    # content, summary, keywords); each channel derives its embed text from the
    # chunk + metadata via its registered source
    channels = resolve_vector_channels(settings)
    channel_inputs = [
        get_channel_source(ch.source)(chunks, metadata) for ch in channels
    ]
    channel_embs = await asyncio.gather(
        *(embed_texts(http, inputs) for inputs in channel_inputs)
    )

    chunk_rows = [
        {
            "id": f"{doc_id}:{seq}",
            "seq": seq,
            "text": chunks[seq],
            "summary": metadata[seq].summary,
            "keywords": metadata[seq].keywords,
            "embeddings": {
                ch.embedding_prop: channel_embs[ci][seq]
                for ci, ch in enumerate(channels)
            },
        }
        for seq in range(len(chunks))
    ]

    # union this source into the document's reference set, preserving any
    # sources from a previous version we're about to replace
    sources = sorted(set(existing["sources"] if existing else []) | {source})

    # idempotent re-ingest: drop any previous version of this id (chunks,
    # edges, orphaned keywords) right before writing the new one, so a
    # re-ingested document never leaves stale nodes behind
    if existing is not None:
        await graph.delete_document(driver, doc_id)
    await graph.save_document(driver, doc_id, title, sources, chunk_rows)

    all_keywords = sorted({kw.lower() for m in metadata for kw in m.keywords})
    return doc_id, len(chunks), all_keywords
