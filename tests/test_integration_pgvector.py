"""End-to-end test against a live PostgreSQL + pgvector.

Embedding / LLM / reranker services are mocked; the real SQL in
app.store_pgvector is exercised: document + chunk persistence, the seq-based
NEXT_CHUNK analogue, keyword-join siblings, pgvector similarity search and
tsvector fulltext search.

Skipped automatically when Postgres is not reachable (run the `postgres`
compose service, or `docker compose run --rm tests`).
"""

import hashlib

import numpy as np
import pytest

from app import ingest as ingest_mod, rerank as rerank_mod, search as search_mod
from app.channels import resolve_vector_channels
from app.chunking import get_chunker
from app.config import get_settings
from app.ingest import compute_document_id
from app.llm import ExtractionResult
from app.store_pgvector import PgvectorStore

SETTINGS = get_settings()
DIM = SETTINGS.embedding_dim

_PARA = " ".join(f"Topic {{n}} sentence {i} with several filler words." for i in range(35))
DOC_TEXT = "\n\n".join(_PARA.format(n=n) for n in range(3))
CHUNK_KEYWORDS = [["alpha", "graph"], ["beta"], ["alpha", "vector"]]


def fake_embedding(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
    vec = np.random.RandomState(seed).normal(size=DIM)
    return (vec / np.linalg.norm(vec)).tolist()


async def _connect_or_skip() -> PgvectorStore:
    store = PgvectorStore(SETTINGS.postgres_dsn)
    try:
        await store.connect()
        await store.verify_connectivity()
    except Exception:
        await store.close()
        pytest.skip("Postgres not reachable; run: docker compose --profile pgvector up -d postgres")
    return store


async def test_ingest_and_search_end_to_end(monkeypatch):
    store = await _connect_or_skip()
    await store.init_schema()

    call_count = {"n": 0}

    async def fake_extract_metadata(client, chunk):
        idx = min(call_count["n"], 2)
        call_count["n"] += 1
        return ExtractionResult(
            keywords=CHUNK_KEYWORDS[idx], summary=f"Summary of part {idx}."
        )

    async def fake_embed_texts(client, texts):
        return [fake_embedding(t) for t in texts]

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract_metadata)
    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed_texts)

    doc_id = compute_document_id(DOC_TEXT)
    chunks = get_chunker(SETTINGS.chunk_strategy)(DOC_TEXT, SETTINGS)
    chunk_ids = [f"{doc_id}:{i}" for i in range(len(chunks))]

    returned_id, chunk_count, keywords = await ingest_mod.ingest_document(
        store, None, DOC_TEXT, title="it-doc", source="test"
    )
    try:
        assert returned_id == doc_id
        assert chunk_count == 3
        assert set(keywords) == {"alpha", "graph", "beta", "vector"}

        content_channel = resolve_vector_channels(SETTINGS)[0]

        # --- pgvector similarity finds the exact chunk ---
        hits = await store.vector_search(
            content_channel, fake_embedding(chunks[0]), 3
        )
        assert hits[0]["id"] == chunk_ids[0]
        assert hits[0]["score"] > 0.99

        # --- tsvector fulltext finds the lexically matching chunk ---
        ft_hits = await store.fulltext_search("Topic 0", 5)
        assert ft_hits
        assert ft_hits[0]["id"] == chunk_ids[0]

        # --- sibling expansion: seq walk + shared-keyword join ---
        sibs = await store.fetch_siblings([chunk_ids[0]], 5, 3)
        relations = {(s["id"], s["via"]): s for s in sibs}
        seq1 = relations[(chunk_ids[1], "sequence")]
        assert (seq1["direction"], seq1["distance"]) == ("after", 1)
        seq2 = relations[(chunk_ids[2], "sequence")]
        assert (seq2["direction"], seq2["distance"]) == ("after", 2)
        # chunk 2 shares the keyword "alpha" with the seed
        kw = relations[(chunk_ids[2], "keyword")]
        assert (kw["direction"], kw["distance"], kw["strength"]) == ("lateral", 1, 1.0)

        # --- full search pipeline (graph_proximity -> None -> decay fallback) ---
        async def fake_embed_text(client, text):
            return fake_embedding(chunks[0])

        async def fake_rerank(client, query, texts):
            return [1.0 / (i + 1) for i in range(len(texts))]

        monkeypatch.setattr(search_mod, "embed_text", fake_embed_text)
        monkeypatch.setitem(rerank_mod.RERANKERS._items, "http", fake_rerank)

        results = await search_mod.search(store, None, "what is topic zero?")
        assert results
        assert results[0].chunk_id == chunk_ids[0]
        returned = {r.chunk_id for r in results}
        assert {chunk_ids[1], chunk_ids[2]} <= returned

        # incremental-reuse read path returns chunks with their stored vectors
        cached = await store.fetch_document_chunks(doc_id, ["content_embedding"])
        assert len(cached) == 3
        assert all(len(c["embeddings"]["content_embedding"]) == DIM for c in cached)

        # nearest_chunks parity with the neo4j backend: a chunk's own content
        # vector finds it back at ~1.0 cosine; excluding its document returns none
        c0_emb = fake_embedding(chunks[0])
        near = await store.nearest_chunks(c0_emb, 3, 0.9)
        assert any(n["id"] == chunk_ids[0] and n["sim"] > 0.99 for n in near)
        excluded = await store.nearest_chunks(c0_emb, 3, 0.9, exclude_doc_id=doc_id)
        assert all(n["doc_id"] != doc_id for n in excluded)  # this doc is excluded

        # feedback capture: only existing chunks are recorded (parity with neo4j)
        n = await store.record_feedback(
            "what is topic zero?", [chunk_ids[0], chunk_ids[1], "missing:9"]
        )
        assert n == 2
    finally:
        deleted = await store.delete_document(doc_id)
        assert deleted == 3
        assert await store.delete_document(doc_id) is None
        await store.close()


async def test_sparse_weights_round_trip():
    """BGE-M3 sparse term-weight maps survive save -> get_sparse_weights and the
    reuse read path, stored in the JSONB `sparse_weights` column."""
    store = await _connect_or_skip()
    await store.init_schema()
    doc_id = "sparse-pg-doc"
    channels = resolve_vector_channels(SETTINGS)
    vec = [1.0] + [0.0] * (DIM - 1)
    embeddings = {ch.embedding_prop: vec for ch in channels}
    chunks = [
        {
            "id": f"{doc_id}:0", "seq": 0, "text": "first", "summary": "s0",
            "keywords": ["k0"], "embeddings": embeddings,
            "sparse_weights": {"101": 0.5, "202": 1.25},
        },
        {
            "id": f"{doc_id}:1", "seq": 1, "text": "second", "summary": "s1",
            "keywords": ["k1"], "embeddings": embeddings,
            "sparse_weights": None,  # a chunk ingested without sparse weights
        },
    ]
    ids = [f"{doc_id}:0", f"{doc_id}:1"]
    try:
        await store.save_document(doc_id, "sparse-doc", ["test"], chunks)

        weights = await store.get_sparse_weights(ids)
        assert weights == {f"{doc_id}:0": {"101": 0.5, "202": 1.25}}

        cached = await store.fetch_document_chunks(doc_id, ["content_embedding"])
        by_text = {c["text"]: c for c in cached}
        assert by_text["first"]["sparse_weights"] == {"101": 0.5, "202": 1.25}
        assert by_text["second"]["sparse_weights"] is None
    finally:
        await store.delete_document(doc_id)
        await store.close()


async def test_near_dup_links_round_trip():
    """Memory write-path: near_dup_of survives save -> get_near_dup_links (parity)."""
    store = await _connect_or_skip()
    await store.init_schema()
    doc_id = "neardup-pg-doc"
    channels = resolve_vector_channels(SETTINGS)
    vec = [1.0] + [0.0] * (DIM - 1)
    embeddings = {ch.embedding_prop: vec for ch in channels}
    chunks = [
        {
            "id": f"{doc_id}:0", "seq": 0, "text": "a", "summary": "", "keywords": [],
            "embeddings": embeddings, "near_dup_of": "canonical:7",
        },
        {
            "id": f"{doc_id}:1", "seq": 1, "text": "b", "summary": "", "keywords": [],
            "embeddings": embeddings, "near_dup_of": None,
        },
    ]
    ids = [f"{doc_id}:0", f"{doc_id}:1"]
    try:
        await store.save_document(doc_id, "t", ["s"], chunks)
        assert await store.get_near_dup_links(ids) == {f"{doc_id}:0": "canonical:7"}
    finally:
        await store.delete_document(doc_id)
        await store.close()
