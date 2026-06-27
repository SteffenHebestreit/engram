"""Contextual Retrieval (Anthropic): an LLM-written, document-situating context
is prepended to each chunk before embedding, so the content vector encodes
document-level identity. Opt-in (`contextual_retrieval_enabled`); a no-op by
default. A geometry change — search is untouched, only the ingest-time embedded
text differs."""

from app import graph
from app import ingest as ingest_mod
from app.channels import get_channel_source
from app.config import Settings
from app.llm import ExtractionResult


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
