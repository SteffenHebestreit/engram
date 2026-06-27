"""Contextual Retrieval (Anthropic): an LLM-written, document-situating context
is prepended to each chunk before embedding (contextual embeddings) and indexed
for fulltext (contextual BM25), so both retrieval channels encode document-level
identity. Opt-in (`contextual_retrieval_enabled`); a no-op by default. The
embeddings half is a geometry change (search untouched); the BM25 half stores +
indexes the context additively (empty when off, so behaviour is unchanged)."""

import hashlib

import numpy as np
import pytest

from app import graph
from app import ingest as ingest_mod
from app.channels import get_channel_source, resolve_vector_channels
from app.config import Settings, get_settings
from app.llm import ExtractionResult
from app.store_neo4j import Neo4jStore
from app.store_pgvector import PgvectorStore

DIM = get_settings().embedding_dim


def _fake_vec(seed: str) -> list[float]:
    v = np.random.RandomState(
        int.from_bytes(hashlib.md5(seed.encode()).digest()[:4], "big")
    ).normal(size=DIM)
    return (v / np.linalg.norm(v)).tolist()


def _chunk_with_context(chunk_id: str, text: str, context: str) -> dict:
    """A save_document chunk dict carrying a `context` field + embeddings for every
    active channel. Tests contextual BM25 directly via save_document — no ingest
    flag, so the schema signature is unchanged (the context column/index is
    additive + unconditional)."""
    embeddings = {
        ch.embedding_prop: _fake_vec(chunk_id)
        for ch in resolve_vector_channels(get_settings())
    }
    return {
        "id": chunk_id, "seq": 0, "text": text, "summary": "",
        "keywords": [], "embeddings": embeddings, "context": context,
    }


class _IngestStore:
    """Minimal Store stand-in: no prior version, records the saved rows."""

    def __init__(self):
        self.rows = None

    async def get_document(self, doc_id):
        return None

    async def save_document(self, doc_id, title, sources, rows):
        self.rows = rows


def _patch_ingest(monkeypatch, settings, embedded):
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(
        ingest_mod, "get_chunker", lambda name: lambda text, s: ["alpha", "beta"]
    )

    async def fake_extract(client, chunk):
        return ExtractionResult(keywords=[], summary="")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    async def fake_embed(client, texts):
        embedded.extend(texts)
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)


# ── the channel source in isolation ──────────────────────────────────────────
def test_contextual_content_source_prepends_context():
    src = get_channel_source("contextual_content")
    meta = [
        ExtractionResult(keywords=[], summary="", context="From the Q2 report."),
        ExtractionResult(keywords=[], summary="", context=""),  # no context
    ]
    out = src(["revenue grew 12%", "bare chunk"], meta)
    assert out == ["From the Q2 report.\n\nrevenue grew 12%", "bare chunk"]


# ── ingest wiring ────────────────────────────────────────────────────────────
async def test_ingest_prepends_generated_context(monkeypatch):
    settings = Settings(
        contextual_retrieval_enabled=True,
        summary_channel_enabled=False,
        keywords_channel_enabled=False,
    )
    embedded: list[str] = []
    _patch_ingest(monkeypatch, settings, embedded)

    async def fake_context(client, document, chunk):
        return f"ctx[{chunk}]"

    monkeypatch.setattr(ingest_mod, "generate_chunk_context", fake_context)

    await ingest_mod.ingest_document(
        _IngestStore(), None, "alpha\n\nbeta", source="s", document_id="d"
    )
    # the content channel embedded each chunk WITH its situating context prepended
    assert embedded == ["ctx[alpha]\n\nalpha", "ctx[beta]\n\nbeta"]


async def test_ingest_degrades_when_context_unavailable(monkeypatch):
    settings = Settings(
        contextual_retrieval_enabled=True,
        summary_channel_enabled=False,
        keywords_channel_enabled=False,
    )
    embedded: list[str] = []
    _patch_ingest(monkeypatch, settings, embedded)

    async def empty_context(client, document, chunk):
        return ""  # LLM down / empty reply

    monkeypatch.setattr(ingest_mod, "generate_chunk_context", empty_context)

    await ingest_mod.ingest_document(
        _IngestStore(), None, "alpha\n\nbeta", source="s", document_id="d"
    )
    # no context -> bare chunk embedded, ingest never breaks
    assert embedded == ["alpha", "beta"]


async def test_ingest_stores_context_in_rows(monkeypatch):
    settings = Settings(
        contextual_retrieval_enabled=True,
        summary_channel_enabled=False,
        keywords_channel_enabled=False,
    )
    _patch_ingest(monkeypatch, settings, [])

    async def fake_context(client, document, chunk):
        return f"ctx[{chunk}]"

    monkeypatch.setattr(ingest_mod, "generate_chunk_context", fake_context)

    store = _IngestStore()
    await ingest_mod.ingest_document(
        store, None, "alpha\n\nbeta", source="s", document_id="d"
    )
    # the generated context is persisted on each row (so it can be indexed for BM25)
    assert [r["context"] for r in store.rows] == ["ctx[alpha]", "ctx[beta]"]


async def test_contextual_disabled_does_not_call_llm(monkeypatch):
    settings = Settings(
        summary_channel_enabled=False, keywords_channel_enabled=False
    )  # contextual_retrieval_enabled defaults False
    embedded: list[str] = []
    _patch_ingest(monkeypatch, settings, embedded)

    async def boom_context(client, document, chunk):
        raise AssertionError("context must not be generated when disabled")

    monkeypatch.setattr(ingest_mod, "generate_chunk_context", boom_context)

    await ingest_mod.ingest_document(
        _IngestStore(), None, "alpha\n\nbeta", source="s", document_id="d"
    )
    assert embedded == ["alpha", "beta"]  # unchanged default behaviour


# ── schema signature guard ───────────────────────────────────────────────────
def test_contextual_changes_schema_signature():
    base = graph.schema_signature(Settings())
    changed = graph.schema_signature(Settings(contextual_retrieval_enabled=True))
    assert base != changed  # stored content vectors changed -> guard must notice


def test_contextual_disabled_signature_unchanged():
    # the default-off case must not perturb an existing store's signature
    assert graph.schema_signature(Settings()) == graph.schema_signature(
        Settings(contextual_retrieval_enabled=False)
    )


# ── contextual BM25 (live; the context is indexed for fulltext too) ───────────
# A term that appears ONLY in the situating context (not in the chunk text or
# summary) must be findable by fulltext search — proof the context is indexed.
_CTX = "this passage concerns the zzqcontextterm subsystem"


async def test_contextual_bm25_neo4j():
    driver = graph.create_driver()
    try:
        await driver.verify_connectivity()
    except Exception:
        await driver.close()
        pytest.skip("Neo4j not reachable; run: docker compose up -d")
    store = Neo4jStore(driver)
    try:
        # the shared test DB may hold the old [text, summary] fulltext index from
        # another test; drop it so init_schema rebuilds it including c.context
        async with driver.session() as s:
            await s.run(f"DROP INDEX {graph.FULLTEXT_INDEX} IF EXISTS")
        await store.init_schema()
        await store.save_document(
            "ctxbm25", "t", ["s"],
            [_chunk_with_context("ctxbm25:0", "alpha widget", _CTX)],
        )
        # term only in the context -> found via the contextual BM25 index
        ctx_hits = await store.fulltext_search("zzqcontextterm", 5)
        assert any(h["id"] == "ctxbm25:0" for h in ctx_hits)
        # regression: a term in the chunk text is still found
        text_hits = await store.fulltext_search("widget", 5)
        assert any(h["id"] == "ctxbm25:0" for h in text_hits)
    finally:
        await graph.delete_document(driver, "ctxbm25")
        await store.close()


async def test_contextual_bm25_pgvector():
    store = PgvectorStore(get_settings().postgres_dsn)
    try:
        await store.connect()
        await store.verify_connectivity()
    except Exception:
        await store.close()
        pytest.skip(
            "Postgres not reachable; run: docker compose --profile pgvector up -d postgres"
        )
    try:
        await store.init_schema()  # additive: adds context + context_tsv + index
        await store.save_document(
            "ctxbm25pg", "t", ["s"],
            [_chunk_with_context("ctxbm25pg:0", "alpha widget", _CTX)],
        )
        ctx_hits = await store.fulltext_search("zzqcontextterm", 5)
        assert any(h["id"] == "ctxbm25pg:0" for h in ctx_hits)
        text_hits = await store.fulltext_search("widget", 5)
        assert any(h["id"] == "ctxbm25pg:0" for h in text_hits)
    finally:
        await store.delete_document("ctxbm25pg")
        await store.close()
