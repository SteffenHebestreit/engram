"""Incremental re-ingest: unchanged chunks reuse stored vectors + metadata."""

from app import ingest as ingest_mod
from app.config import Settings
from app.llm import ExtractionResult

# default channels => these three embedding props per chunk
PROPS = ["content_embedding", "summary_embedding", "keywords_embedding"]


class FakeStore:
    """Minimal Store stand-in recording what gets saved; holds one prior version."""

    def __init__(self, existing, old_chunks):
        self._existing = existing
        self._old_chunks = old_chunks
        self.saved_rows = None
        self.deleted = False

    async def get_document(self, doc_id):
        return self._existing

    async def fetch_document_chunks(self, doc_id, embedding_props):
        return self._old_chunks

    async def delete_document(self, doc_id):
        self.deleted = True
        return len(self._old_chunks)

    async def save_document(self, doc_id, title, sources, rows):
        self.saved_rows = rows

    async def add_document_source(self, doc_id, source):
        raise AssertionError("not the fast path in this test")


async def test_unchanged_chunks_are_reused(monkeypatch):
    settings = Settings()  # reuse_unchanged_chunks defaults True
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    # new version chunks into ["A", "C"]: A is unchanged, C is new
    monkeypatch.setattr(ingest_mod, "get_chunker", lambda name: lambda text, s: ["A", "C"])

    extracted = []

    async def fake_extract(client, chunk):
        extracted.append(chunk)
        return ExtractionResult(keywords=[f"kw-{chunk}"], summary=f"sum-{chunk}")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    embedded = []

    async def fake_embed(client, texts):
        embedded.append(list(texts))
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)

    # prior version had chunks "A" and "B", each with all three channel vectors
    old_chunks = [
        {
            "text": "A",
            "summary": "old-sum-A",
            "keywords": ["old-kw-A"],
            "embeddings": {p: [9.0] for p in PROPS},
        },
        {
            "text": "B",
            "summary": "old-sum-B",
            "keywords": ["old-kw-B"],
            "embeddings": {p: [8.0] for p in PROPS},
        },
    ]
    store = FakeStore(
        existing={"sources": ["s1"], "chunk_count": 2, "keywords": ["old-kw-a"]},
        old_chunks=old_chunks,
    )

    await ingest_mod.ingest_document(
        store, None, "irrelevant text", source="s2", document_id="doc-1"
    )

    # only the new chunk "C" was extracted; "A" reused its stored metadata
    assert extracted == ["C"]
    # each channel embedded exactly one item (the fresh chunk), never "A";
    # the content channel embeds the raw chunk text
    assert [len(batch) for batch in embedded] == [1, 1, 1]
    assert ["A"] not in embedded
    assert embedded[0] == ["C"]  # content channel = raw text of the fresh chunk

    rows = {r["text"]: r for r in store.saved_rows}
    # "A" reused the stored vectors + metadata verbatim
    assert rows["A"]["summary"] == "old-sum-A"
    assert rows["A"]["keywords"] == ["old-kw-A"]
    assert rows["A"]["embeddings"]["content_embedding"] == [9.0]
    # "C" got fresh metadata + a freshly computed embedding (len("C") == 1)
    assert rows["C"]["summary"] == "sum-C"
    assert rows["C"]["embeddings"]["content_embedding"] == [1.0]


async def test_reuse_disabled_recomputes_everything(monkeypatch):
    settings = Settings(reuse_unchanged_chunks=False)
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_mod, "get_chunker", lambda name: lambda text, s: ["A", "C"])

    extracted = []

    async def fake_extract(client, chunk):
        extracted.append(chunk)
        return ExtractionResult(keywords=["k"], summary="s")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    async def fake_embed(client, texts):
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)

    store = FakeStore(
        existing={"sources": ["s1"], "chunk_count": 2, "keywords": []},
        old_chunks=[
            {"text": "A", "summary": "x", "keywords": [], "embeddings": {p: [9.0] for p in PROPS}}
        ],
    )
    await ingest_mod.ingest_document(
        store, None, "irrelevant", source="s2", document_id="doc-1"
    )
    # with reuse off, both chunks are re-extracted despite "A" being unchanged
    assert sorted(extracted) == ["A", "C"]
