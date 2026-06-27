"""Recency / temporal decay (the agent-memory signal): a chunk's document age is
read once per search and blended into the final ordering. These live tests verify
the actual age query (Cypher / SQL) on both backends — a just-ingested document's
chunks report a small, non-negative age in seconds.

Skipped automatically when a backend is not reachable.
"""

import hashlib

import numpy as np
import pytest

from app import graph, ingest as ingest_mod
from app.config import get_settings
from app.llm import ExtractionResult
from app.store_neo4j import Neo4jStore
from app.store_pgvector import PgvectorStore

SETTINGS = get_settings()
DIM = SETTINGS.embedding_dim


def _fake_embedding(text: str) -> list[float]:
    v = np.random.RandomState(
        int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
    ).normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()


def _patch_ingest(monkeypatch):
    async def fake_extract(client, chunk):
        return ExtractionResult(keywords=["k"], summary="s")

    async def fake_embed_texts(client, texts):
        return [_fake_embedding(t) for t in texts]

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)
    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed_texts)


async def _assert_fresh_recency(store, monkeypatch):
    _patch_ingest(monkeypatch)
    await store.init_schema()
    doc_id, _, _ = await ingest_mod.ingest_document(
        store, None, "alpha beta gamma", source="s", document_id="recencydoc"
    )
    try:
        ages = await store.get_chunk_recency([f"{doc_id}:0"])
        assert f"{doc_id}:0" in ages
        # just created: a small, non-negative age in seconds (generous upper bound
        # to tolerate slow CI / clock skew between app and DB)
        assert 0.0 <= ages[f"{doc_id}:0"] < 3600.0
        # unknown ids are simply absent (caller defaults them to neutral)
        assert await store.get_chunk_recency([]) == {}
    finally:
        await store.delete_document(doc_id)


async def test_recency_neo4j(monkeypatch):
    driver = graph.create_driver()
    try:
        await driver.verify_connectivity()
    except Exception:
        await driver.close()
        pytest.skip("Neo4j not reachable; run: docker compose up -d")
    store = Neo4jStore(driver)
    try:
        await _assert_fresh_recency(store, monkeypatch)
    finally:
        await store.close()


async def test_recency_pgvector(monkeypatch):
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
        await _assert_fresh_recency(store, monkeypatch)
    finally:
        await store.close()
