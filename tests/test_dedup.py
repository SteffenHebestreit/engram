"""Memory write-path: non-destructive near-duplicate linking + collapse (M1)."""

from app import ingest as ingest_mod
from app import search as search_mod
from app.config import Settings
from app.llm import ExtractionResult


class _LinkStore:
    def __init__(self, links: dict[str, str]):
        self._links = links

    async def get_near_dup_links(self, chunk_ids):
        return {i: self._links[i] for i in chunk_ids if i in self._links}


async def test_collapse_keeps_best_retrieved_per_cluster():
    pool = [
        {"id": "a", "retrieval_score": 0.9},
        {"id": "b", "retrieval_score": 0.5},  # near-dup of a
        {"id": "c", "retrieval_score": 0.7},  # distinct
    ]
    out = await search_mod._collapse_near_dups(_LinkStore({"b": "a"}), pool)
    assert {c["id"] for c in out} == {"a", "c"}  # b folded into a's cluster
    assert next(c for c in out if c["id"] == "a")["near_dup_of"] is None


async def test_collapse_noop_without_links():
    pool = [{"id": "a", "retrieval_score": 0.9}, {"id": "b", "retrieval_score": 0.5}]
    out = await search_mod._collapse_near_dups(_LinkStore({}), pool)
    assert len(out) == 2


async def test_collapse_keeps_dup_when_canonical_absent():
    # b links to a, but a isn't in the pool — b survives as the representative
    pool = [{"id": "b", "retrieval_score": 0.5}, {"id": "c", "retrieval_score": 0.7}]
    out = await search_mod._collapse_near_dups(_LinkStore({"b": "a"}), pool)
    assert {c["id"] for c in out} == {"b", "c"}
    assert next(c for c in out if c["id"] == "b")["near_dup_of"] == "a"


async def test_collapse_follows_chains_without_looping():
    pool = [
        {"id": "a", "retrieval_score": 0.3},
        {"id": "b", "retrieval_score": 0.9},
        {"id": "c", "retrieval_score": 0.5},
    ]
    # a -> b -> c chain (and a self-cycle guard via c -> a would not loop forever)
    out = await search_mod._collapse_near_dups(_LinkStore({"a": "b", "b": "c"}), pool)
    assert {c["id"] for c in out} == {"b"}  # whole chain collapses; best (b) kept


class _DedupIngestStore:
    """Ingest stand-in: no prior version; one near-dup hit for any vector."""

    def __init__(self, canonical_id: str | None):
        self._canonical = canonical_id
        self.rows = None

    async def get_document(self, doc_id):
        return None

    async def nearest_chunks(
        self, embedding, k, min_sim, exclude_doc_id=None, tenant_id=None
    ):
        return [{"id": self._canonical, "sim": 0.99}] if self._canonical else []

    async def save_document(self, doc_id, title, sources, rows):
        self.rows = rows


async def test_ingest_links_near_duplicate_chunks(monkeypatch):
    settings = Settings(
        dedup_enabled=True,
        summary_channel_enabled=False,
        keywords_channel_enabled=False,
    )
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_mod, "get_chunker", lambda name: lambda text, s: ["alpha"])

    async def fake_extract(client, chunk):
        return ExtractionResult(keywords=[], summary="")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    async def fake_embed(client, texts):
        return [[1.0] for _ in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)

    store = _DedupIngestStore(canonical_id="other-doc:3")
    await ingest_mod.ingest_document(store, None, "x", source="s", document_id="d")
    assert store.rows[0]["near_dup_of"] == "other-doc:3"


async def test_ingest_no_link_when_dedup_disabled(monkeypatch):
    settings = Settings(dedup_enabled=False)
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_mod, "get_chunker", lambda name: lambda text, s: ["alpha"])

    async def fake_extract(client, chunk):
        return ExtractionResult(keywords=["k"], summary="s")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    async def fake_embed(client, texts):
        return [[1.0] for _ in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)

    # nearest_chunks must not even be called when dedup is off
    class _NoDedupStore(_DedupIngestStore):
        async def nearest_chunks(self, *a, **k):
            raise AssertionError("nearest_chunks must not be called when dedup off")

    store = _NoDedupStore(canonical_id="x")
    await ingest_mod.ingest_document(store, None, "x", source="s", document_id="d")
    assert store.rows[0]["near_dup_of"] is None
