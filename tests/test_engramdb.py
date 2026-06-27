"""Engram-DB embedded store — end-to-end via the real pipeline + protocol checks.

Runs fully in-process (no Neo4j/Postgres), so unlike the other backend tests it
never skips. Exercises ingest → vector + fulltext + native-adjacency graph
expansion → the full search pipeline, plus multi-tenant isolation, recency,
feedback, and pickle persistence. graph_proximity is None by design (decay path).
"""

import hashlib

import numpy as np

from app import ingest as ingest_mod
from app import rerank as rerank_mod
from app import search as search_mod
from app.channels import resolve_vector_channels
from app.chunking import get_chunker
from app.config import get_settings
from app.llm import ExtractionResult
from app.store_engramdb import EngramDBStore

SETTINGS = get_settings()
DIM = SETTINGS.embedding_dim

_PARA = " ".join(f"Topic {{n}} sentence {i} with several filler words." for i in range(35))
DOC = "\n\n".join(_PARA.format(n=n) for n in range(3))
KW = [["alpha", "graph"], ["beta"], ["alpha", "vector"]]


def fake_embedding(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
    v = np.random.RandomState(seed).normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()


def _patch_ingest(monkeypatch):
    counter = {"n": 0}

    async def fake_extract(client, chunk):
        idx = min(counter["n"], 2)
        counter["n"] += 1
        return ExtractionResult(keywords=KW[idx], summary=f"Summary of part {idx}.")

    async def fake_embed_texts(client, texts):
        return [fake_embedding(t) for t in texts]

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)
    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed_texts)


async def test_end_to_end(monkeypatch):
    store = EngramDBStore()
    await store.connect()
    await store.init_schema()
    chunks = get_chunker(SETTINGS.chunk_strategy)(DOC, SETTINGS)
    assert len(chunks) == 3
    _patch_ingest(monkeypatch)

    doc_id, count, keywords = await ingest_mod.ingest_document(
        store, None, DOC, title="t", source="s", document_id="d"
    )
    ids = [f"{doc_id}:{i}" for i in range(3)]
    assert count == 3
    assert set(keywords) == {"alpha", "graph", "beta", "vector"}

    content = resolve_vector_channels(SETTINGS)[0]
    # vector search finds the exact chunk back at ~1.0 cosine
    hits = await store.vector_search(content, fake_embedding(chunks[0]), 3)
    assert hits[0]["id"] == ids[0] and hits[0]["score"] > 0.99

    # BM25 fulltext returns the matching chunks
    ft = await store.fulltext_search("topic", 5)
    assert {h["id"] for h in ft} >= set(ids)

    # native-adjacency graph: NEXT_CHUNK seq sibling + shared-keyword sibling
    sibs = await store.fetch_siblings([ids[0]], 5, 3)
    rel = {(s["id"], s["via"]) for s in sibs}
    assert (ids[1], "sequence") in rel       # chunk 1 follows chunk 0
    assert (ids[2], "keyword") in rel        # chunk 2 shares "alpha" with chunk 0

    # decay by design — no PPR
    assert await store.graph_proximity([ids[0]], ids, 0.85) is None

    # full pipeline end to end
    async def fake_embed_text(client, text):
        return fake_embedding(chunks[0])

    async def fake_rerank(client, query, texts):
        return [1.0 / (i + 1) for i in range(len(texts))]

    monkeypatch.setattr(search_mod, "embed_text", fake_embed_text)
    monkeypatch.setitem(rerank_mod.RERANKERS._items, "http", fake_rerank)
    results = await search_mod.search(
        store, None, "what is topic zero?", tuning={"hyde_enabled": False}
    )
    assert results and results[0].chunk_id == ids[0]
    assert {ids[1], ids[2]} <= {r.chunk_id for r in results}

    # incremental-reuse read path returns stored vectors
    cached = await store.fetch_document_chunks(doc_id, ["content_embedding"])
    assert len(cached) == 3
    assert all(len(c["embeddings"]["content_embedding"]) == DIM for c in cached)
    await store.close()


async def test_tenant_isolation(monkeypatch):
    store = EngramDBStore()
    await store.connect()
    await store.init_schema()
    _patch_ingest(monkeypatch)
    a, _, _ = await ingest_mod.ingest_document(
        store, None, DOC, source="s", document_id="x", tenant_id="A"
    )
    b, _, _ = await ingest_mod.ingest_document(
        store, None, DOC, source="s", document_id="x", tenant_id="B"
    )
    assert a != b  # ids namespaced per tenant
    content = resolve_vector_channels(SETTINGS)[0]
    q = fake_embedding(get_chunker(SETTINGS.chunk_strategy)(DOC, SETTINGS)[0])
    a_hits = await store.vector_search(content, q, 20, "A")
    assert a_hits and all(h["id"].startswith("A:") for h in a_hits)
    all_hits = await store.vector_search(content, q, 20)
    assert any(h["id"].startswith("B:") for h in all_hits)  # B reachable unfiltered
    ft_a = await store.fulltext_search("topic", 20, "A")
    assert ft_a and not any(h["id"].startswith("B:") for h in ft_a)
    await store.close()


async def test_recency_feedback_and_persistence(monkeypatch, tmp_path):
    path = str(tmp_path / "engramdb.pkl")
    store = EngramDBStore(path)
    await store.connect()
    await store.init_schema()
    _patch_ingest(monkeypatch)
    doc_id, _, _ = await ingest_mod.ingest_document(
        store, None, DOC, source="s", document_id="d"
    )
    ids = [f"{doc_id}:{i}" for i in range(3)]

    ages = await store.get_chunk_recency(ids)
    assert ids[0] in ages and 0.0 <= ages[ids[0]] < 3600.0

    assert await store.record_feedback("q", [ids[0], "missing"], "qid") == 1

    await store.close()  # snapshot to disk

    # reload from the snapshot: data survives
    store2 = EngramDBStore(path)
    await store2.connect()
    doc = await store2.get_document(doc_id)
    assert doc and doc["chunk_count"] == 3
    near = await store2.nearest_chunks(
        fake_embedding(get_chunker(SETTINGS.chunk_strategy)(DOC, SETTINGS)[0]),
        3, 0.9,
    )
    assert any(n["id"] == ids[0] and n["sim"] > 0.99 for n in near)
    await store2.close()
