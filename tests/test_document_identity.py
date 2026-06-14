import hashlib

import pytest

import app.ingest as ingest_mod
from app.config import Settings
from app.ingest import compute_document_id
from app.llm import ExtractionResult


def test_content_addressed_id_is_deterministic():
    assert compute_document_id("hello") == hashlib.sha256(b"hello").hexdigest()
    assert compute_document_id("hello") == compute_document_id("hello")
    assert compute_document_id("hello") != compute_document_id("world")


def test_explicit_id_overrides_and_blank_falls_back():
    assert compute_document_id("hello", "ctx-42") == "ctx-42"
    assert compute_document_id("hello", "  ctx-42  ") == "ctx-42"
    # empty / whitespace-only explicit ids fall back to the content hash
    content_hash = hashlib.sha256(b"hello").hexdigest()
    assert compute_document_id("hello", "") == content_hash
    assert compute_document_id("hello", "   ") == content_hash


async def _run_ingest(monkeypatch, *, existing=None, document_id=None, source="src"):
    settings = Settings()
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_mod, "get_chunker", lambda name: lambda text, s: [text])

    async def fake_extract(client, chunk):
        return ExtractionResult(keywords=["k"], summary="s")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    async def fake_embed(client, texts):
        return [[0.0] for _ in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)

    calls = {"deleted": [], "saved": [], "added_source": []}

    async def fake_get(driver, doc_id):
        return existing

    async def fake_delete(driver, doc_id):
        calls["deleted"].append(doc_id)
        return 1

    async def fake_save(driver, doc_id, title, sources, rows):
        calls["saved"].append((doc_id, sources))

    async def fake_add(driver, doc_id, src):
        calls["added_source"].append((doc_id, src))

    monkeypatch.setattr(ingest_mod.graph, "get_document", fake_get)
    monkeypatch.setattr(ingest_mod.graph, "delete_document", fake_delete)
    monkeypatch.setattr(ingest_mod.graph, "save_document", fake_save)
    monkeypatch.setattr(ingest_mod.graph, "add_document_source", fake_add)

    doc_id, n, kws = await ingest_mod.ingest_document(
        None, None, "hello", source=source, document_id=document_id
    )
    return doc_id, n, kws, calls


async def test_fresh_ingest_saves_with_its_source(monkeypatch):
    doc_id, n, kws, calls = await _run_ingest(monkeypatch, existing=None)
    assert calls["deleted"] == []
    assert calls["added_source"] == []
    assert calls["saved"] == [(doc_id, ["src"])]
    assert doc_id == hashlib.sha256(b"hello").hexdigest()


async def test_content_addressed_reingest_just_registers_the_source(monkeypatch):
    existing = {"sources": ["src"], "chunk_count": 3, "keywords": ["k"]}
    doc_id, n, kws, calls = await _run_ingest(
        monkeypatch, existing=existing, source="other"
    )
    # fast path: identical content, so no re-embed and no replace — just add the
    # new source reference; existing chunk_count/keywords are returned
    assert calls["saved"] == []
    assert calls["deleted"] == []
    assert calls["added_source"] == [(doc_id, "other")]
    assert (n, kws) == (3, ["k"])


async def test_explicit_id_reingest_replaces_and_unions_sources(monkeypatch):
    existing = {"sources": ["a"], "chunk_count": 1, "keywords": ["x"]}
    doc_id, n, kws, calls = await _run_ingest(
        monkeypatch, existing=existing, document_id="ctx-1", source="b"
    )
    assert doc_id == "ctx-1"
    assert calls["deleted"] == ["ctx-1"]  # old version replaced
    assert calls["saved"] == [("ctx-1", ["a", "b"])]  # sources unioned + sorted
    assert calls["added_source"] == []


async def test_client_supplied_id_fresh(monkeypatch):
    doc_id, n, kws, calls = await _run_ingest(
        monkeypatch, existing=None, document_id="ctx-42"
    )
    assert doc_id == "ctx-42"
    assert calls["saved"] == [("ctx-42", ["src"])]


async def test_ingest_requires_a_non_empty_source():
    # every document must be reference-counted, so a source is mandatory
    with pytest.raises(ValueError, match="source"):
        await ingest_mod.ingest_document(None, None, "hello", source="   ")
