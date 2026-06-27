import asyncio
import hashlib
from typing import TYPE_CHECKING

import httpx

from .channels import get_channel_source, resolve_vector_channels
from .chunking import get_chunker
from .config import get_settings
from .embeddings import embed_texts
from .llm import ExtractionResult, generate_chunk_context, get_extractor

if TYPE_CHECKING:
    from .store import Store


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
    store: "Store",
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

    existing = await store.get_document(doc_id)

    # fast path: the same content is already ingested (the id is the text hash),
    # so the chunks are identical — just register this source instead of
    # re-chunking and re-embedding. Multiple sources can reference one document;
    # its nodes are only deleted once the last source is removed.
    if existing is not None and content_addressed:
        await store.add_document_source(doc_id, source)
        return doc_id, existing["chunk_count"], existing["keywords"]

    chunker = get_chunker(settings.chunk_strategy)
    chunks = chunker(text, settings)
    if not chunks:
        raise ValueError("no chunks produced from input text")

    channels = resolve_vector_channels(settings)
    embedding_props = [ch.embedding_prop for ch in channels]

    # incremental re-ingest: reuse the stored vectors + metadata of any chunk
    # whose text is byte-identical to one in this document's previous version,
    # so only changed chunks pay for fresh LLM extraction + embedding. Only reuse
    # a chunk that has a vector for every active channel.
    reuse: dict[str, dict] = {}
    if existing is not None and settings.reuse_unchanged_chunks:
        for old in await store.fetch_document_chunks(doc_id, embedding_props):
            if all(old["embeddings"].get(p) is not None for p in embedding_props):
                reuse.setdefault(old["text"], old)

    extractor = get_extractor(settings.metadata_extractor)
    semaphore = asyncio.Semaphore(max(1, settings.extraction_concurrency))

    async def extract(chunk: str):
        async with semaphore:
            return await extractor(http, chunk)

    # extract metadata only for chunks that are not being reused
    fresh_idx = [i for i, c in enumerate(chunks) if c not in reuse]
    fresh_set = set(fresh_idx)
    fresh_meta = await asyncio.gather(*(extract(chunks[i]) for i in fresh_idx))
    fresh_meta_by_idx = dict(zip(fresh_idx, fresh_meta))

    metadata = [
        fresh_meta_by_idx[i]
        if i in fresh_set
        else ExtractionResult(
            keywords=reuse[c]["keywords"] or [], summary=reuse[c]["summary"] or ""
        )
        for i, c in enumerate(chunks)
    ]

    # Contextual Retrieval (opt-in): the LLM writes a short document-situating
    # context per fresh chunk, attached to its metadata so the `contextual_content`
    # channel source prepends it before embedding. The whole document is the
    # shared prefix across a doc's chunk calls (prompt-cache friendly). Degrades
    # to no context for a chunk when the LLM is unavailable; reused chunks keep
    # the context already baked into their stored vector.
    if settings.contextual_retrieval_enabled and fresh_idx:

        async def _context(i: int) -> tuple[int, str]:
            async with semaphore:
                return i, await generate_chunk_context(http, text, chunks[i])

        for i, ctx in await asyncio.gather(*(_context(i) for i in fresh_idx)):
            if ctx:
                metadata[i]["context"] = ctx

    # one independent embedding space per active vector channel; embed only the
    # non-reused chunks per channel, then splice the reused vectors back by index.
    # The passage-side instruction (empty by default) is prepended for
    # instruction-tuned embedders; it is part of the schema signature so reused
    # vectors were embedded with the same prefix.
    passage_prefix = settings.passage_instruction

    async def channel_vectors(ch) -> list:
        inputs = get_channel_source(ch.source)(chunks, metadata)
        fresh = await embed_texts(http, [passage_prefix + inputs[i] for i in fresh_idx])
        fresh_by_idx = dict(zip(fresh_idx, fresh))
        return [
            fresh_by_idx[i]
            if i in fresh_set
            else reuse[c]["embeddings"][ch.embedding_prop]
            for i, c in enumerate(chunks)
        ]

    channel_embs = await asyncio.gather(*(channel_vectors(ch) for ch in channels))

    # optional BGE-M3 learned-sparse term weights per chunk (opt-in). Computed
    # over the chunk content text — the same text the query sparse vector is
    # matched against at search time — and folded into the fused score there as
    # an exact-term signal. Fresh chunks only; unchanged chunks reuse their
    # stored weights, like the dense vectors. Degrades to None (no sparse stored)
    # when the endpoint is down, so it never blocks ingest.
    sparse_by_seq: list[dict | None] = [None] * len(chunks)
    if settings.sparse_enabled:
        from .embeddings import embed_sparse_texts

        fresh_sparse = await embed_sparse_texts(http, [chunks[i] for i in fresh_idx])
        if fresh_sparse is not None:
            fresh_sparse_by_idx = dict(zip(fresh_idx, fresh_sparse))
            sparse_by_seq = [
                fresh_sparse_by_idx[i]
                if i in fresh_set
                else reuse[c].get("sparse_weights")
                for i, c in enumerate(chunks)
            ]

    # memory write-path (M1): link a fresh chunk that is a near-duplicate of an
    # existing chunk in *another* document to its canonical (non-destructive — the
    # chunk is still stored). Reuses the store's nearest_chunks primitive over the
    # content vector; opt-in. Reused (byte-identical) chunks are skipped.
    near_dup_by_seq: list[str | None] = [None] * len(chunks)
    if settings.dedup_enabled:
        content_ci = next(
            (i for i, ch in enumerate(channels) if ch.embedding_prop == "content_embedding"),
            0,
        )
        content_vecs = channel_embs[content_ci]

        async def _nearest(i: int) -> tuple[int, str | None]:
            near = await store.nearest_chunks(
                content_vecs[i],
                settings.dedup_candidate_k,
                settings.dedup_cosine_threshold,
                exclude_doc_id=doc_id,
            )
            return i, (near[0]["id"] if near else None)

        for i, canonical in await asyncio.gather(*(_nearest(i) for i in fresh_idx)):
            near_dup_by_seq[i] = canonical

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
            "sparse_weights": sparse_by_seq[seq],
            "near_dup_of": near_dup_by_seq[seq],
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
        await store.delete_document(doc_id)
    await store.save_document(doc_id, title, sources, chunk_rows)

    all_keywords = sorted({kw.lower() for m in metadata for kw in m.keywords})
    return doc_id, len(chunks), all_keywords
