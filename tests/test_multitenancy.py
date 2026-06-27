"""Tenant-isolation security gate (both backends).

Isolation is all-or-nothing: every chunk-surfacing read must filter by tenant,
or one tenant's documents leak into another's results. This exercises the
complete list against a live store — vector_search, fulltext_search,
nearest_chunks (filtered in-query) and fetch_siblings (returns the tenant_id the
search layer filters on) — plus the search pipeline end to end, plus the
ingest-identity holes (content-addressed collision + explicit-id collision).

Two tenants ingest *byte-identical* content, so the only thing keeping tenant B
out of a tenant-A read is the filter (identical text → identical embeddings,
identical fulltext terms, shared keywords → cross-tenant graph siblings).

Skipped automatically when the backend is not reachable.
"""

import hashlib

import numpy as np
import pytest

from app import graph, ingest as ingest_mod, rerank as rerank_mod, search as search_mod
from app.channels import resolve_vector_channels
from app.chunking import get_chunker
from app.config import get_settings
from app.ingest import compute_document_id
from app.llm import ExtractionResult
from app.store_neo4j import Neo4jStore
from app.store_pgvector import PgvectorStore

SETTINGS = get_settings()
DIM = SETTINGS.embedding_dim

_PARA = " ".join(f"Topic {{n}} sentence {i} with several filler words." for i in range(35))
DOC_TEXT = "\n\n".join(_PARA.format(n=n) for n in range(3))
# every chunk carries the "shared" keyword, so a tenant-A seed's HAS_KEYWORD
# siblings reach tenant-B chunks — the exact cross-tenant graph leak to filter
CHUNK_KEYWORDS = [["shared", "graph"], ["shared", "beta"], ["shared", "vector"]]

TENANT_A = "tenant-a"
TENANT_B = "tenant-b"


def fake_embedding(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
    vec = np.random.RandomState(seed).normal(size=DIM)
    return (vec / np.linalg.norm(vec)).tolist()


def _patch_pipeline(monkeypatch, chunks):
    """Mock the embedding / extraction / rerank services (no network)."""

    async def fake_extract(client, chunk):
        idx = chunks.index(chunk) if chunk in chunks else 0
        return ExtractionResult(
            keywords=CHUNK_KEYWORDS[idx], summary=f"Summary of part {idx}."
        )

    async def fake_embed_texts(client, texts):
        return [fake_embedding(t) for t in texts]

    async def fake_embed_text(client, text):
        # query embeds onto chunk 0, so chunk 0 is the strongest seed
        return fake_embedding(chunks[0])

    async def fake_rerank(client, query, texts):
        return [1.0 / (i + 1) for i in range(len(texts))]

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)
    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(search_mod, "embed_text", fake_embed_text)
    monkeypatch.setitem(rerank_mod.RERANKERS._items, "http", fake_rerank)


async def _assert_isolation(store, monkeypatch):
    await store.init_schema()
    chunks = get_chunker(SETTINGS.chunk_strategy)(DOC_TEXT, SETTINGS)
    assert len(chunks) == 3
    _patch_pipeline(monkeypatch, chunks)

    content_channel = resolve_vector_channels(SETTINGS)[0]
    q_emb = fake_embedding(chunks[0])

    # identical content under two tenants → two disjoint id namespaces
    a_doc, _, _ = await ingest_mod.ingest_document(
        store, None, DOC_TEXT, source="s", tenant_id=TENANT_A
    )
    b_doc, _, _ = await ingest_mod.ingest_document(
        store, None, DOC_TEXT, source="s", tenant_id=TENANT_B
    )
    # explicit-id collision: same handle under both tenants must NOT collide
    a_dup, _, _ = await ingest_mod.ingest_document(
        store, None, "x", source="s", document_id="dup", tenant_id=TENANT_A
    )
    b_dup, _, _ = await ingest_mod.ingest_document(
        store, None, "x", source="s", document_id="dup", tenant_id=TENANT_B
    )

    def is_b(cid: str) -> bool:
        return cid.startswith(f"{TENANT_B}:")

    try:
        # --- identity: collisions are namespaced apart ---
        assert a_doc == compute_document_id(DOC_TEXT, tenant_id=TENANT_A)
        assert a_doc != b_doc  # same bytes, different tenant → different id
        assert a_dup == f"{TENANT_A}:dup" and b_dup == f"{TENANT_B}:dup"
        assert a_dup != b_dup

        # --- read path 1: dense vector search filters in-query ---
        a_hits = await store.vector_search(content_channel, q_emb, 20, TENANT_A)
        assert a_hits and not any(is_b(h["id"]) for h in a_hits)
        all_hits = await store.vector_search(content_channel, q_emb, 20)
        assert any(is_b(h["id"]) for h in all_hits)  # B *is* reachable unfiltered

        # --- read path 2: fulltext search filters in-query ---
        a_ft = await store.fulltext_search("Topic", 20, TENANT_A)
        assert a_ft and not any(is_b(h["id"]) for h in a_ft)
        all_ft = await store.fulltext_search("Topic", 20)
        assert any(is_b(h["id"]) for h in all_ft)

        # --- read path 3: nearest_chunks (ingest dedup) filters in-query ---
        a_near = await store.nearest_chunks(q_emb, 20, -1.0, tenant_id=TENANT_A)
        assert a_near and not any(is_b(n["id"]) for n in a_near)
        all_near = await store.nearest_chunks(q_emb, 20, -1.0)
        assert any(is_b(n["id"]) for n in all_near)

        # --- read path 4: graph siblings carry tenant_id for the search filter ---
        # a tenant-A seed reaches tenant-B chunks via the shared keyword; the rows
        # carry the discriminator search.py drops them on
        sibs = await store.fetch_siblings([f"{a_doc}:0"], 10, 3)
        assert any(s.get("tenant_id") == TENANT_B for s in sibs)  # leak reachable
        kept = [s for s in sibs if s.get("tenant_id") == TENANT_A]
        assert kept and not any(is_b(s["id"]) for s in kept)

        # full-pipeline tuning: drop HyDE (no LLM) and autocut (so the non-vacuity
        # check below can't be trimmed away), and widen top-k to see the whole pool
        tune = {"hyde_enabled": False, "autocut_enabled": False, "final_top_k": 20}

        # --- the gate: full pipeline never surfaces a B chunk for a tenant-A query
        results = await search_mod.search(
            store, None, "what is topic zero?", tenant_id=TENANT_A, tuning=tune
        )
        assert results
        assert not any(is_b(r.chunk_id) for r in results)
        assert all(r.document_id in (a_doc, a_dup) for r in results)

        # untenanted search sees everything (proves the gate is real filtering)
        mixed = await search_mod.search(
            store, None, "what is topic zero?", tuning=tune
        )
        assert any(is_b(r.chunk_id) for r in mixed)

        # --- isolation of destructive ops: deleting B's docs leaves A's intact ---
        await store.delete_document(b_doc)
        await store.delete_document(b_dup)
        still = await store.vector_search(content_channel, q_emb, 20, TENANT_A)
        assert still and not any(is_b(h["id"]) for h in still)
    finally:
        for d in (a_doc, b_doc, a_dup, b_dup):
            await store.delete_document(d)


async def test_tenant_isolation_neo4j(monkeypatch):
    driver = graph.create_driver()
    try:
        await driver.verify_connectivity()
    except Exception:
        await driver.close()
        pytest.skip("Neo4j not reachable; run: docker compose up -d")
    store = Neo4jStore(driver)
    try:
        await _assert_isolation(store, monkeypatch)
    finally:
        await store.close()


async def test_tenant_isolation_pgvector(monkeypatch):
    store = PgvectorStore(SETTINGS.postgres_dsn)
    try:
        await store.connect()
        await store.verify_connectivity()
    except Exception:
        await store.close()
        pytest.skip(
            "Postgres not reachable; run: docker compose --profile pgvector up -d postgres"
        )
    try:
        await _assert_isolation(store, monkeypatch)
    finally:
        await store.close()
