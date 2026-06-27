"""Asymmetric query/passage instruction prefixes for instruction-tuned embedders
(E5 / GTE / Qwen3-Embedding / ...). Empty by default — a no-op for BGE-M3."""

from app import graph
from app import ingest as ingest_mod
from app import search as search_mod
from app.config import Settings
from app.llm import ExtractionResult
from app.store_neo4j import Neo4jStore


class _IngestStore:
    """Minimal Store stand-in: no prior version, records the saved rows."""

    def __init__(self):
        self.rows = None

    async def get_document(self, doc_id):
        return None

    async def save_document(self, doc_id, title, sources, rows):
        self.rows = rows


async def test_ingest_prepends_passage_instruction(monkeypatch):
    # content-only channel set keeps the assertion focused on the one channel
    settings = Settings(
        passage_instruction="passage: ",
        summary_channel_enabled=False,
        keywords_channel_enabled=False,
    )
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(
        ingest_mod, "get_chunker", lambda name: lambda text, s: ["alpha", "beta"]
    )

    async def fake_extract(client, chunk):
        return ExtractionResult(keywords=[], summary="")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    embedded: list[str] = []

    async def fake_embed(client, texts):
        embedded.extend(texts)
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)

    await ingest_mod.ingest_document(
        _IngestStore(), None, "x", source="s", document_id="d"
    )

    # the content channel embedded each chunk WITH the passage prefix
    assert embedded == ["passage: alpha", "passage: beta"]


async def test_ingest_default_has_no_prefix(monkeypatch):
    settings = Settings(summary_channel_enabled=False, keywords_channel_enabled=False)
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(
        ingest_mod, "get_chunker", lambda name: lambda text, s: ["alpha"]
    )

    async def fake_extract(client, chunk):
        return ExtractionResult(keywords=[], summary="")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    embedded: list[str] = []

    async def fake_embed(client, texts):
        embedded.extend(texts)
        return [[1.0] for _ in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed)
    await ingest_mod.ingest_document(
        _IngestStore(), None, "x", source="s", document_id="d"
    )
    assert embedded == ["alpha"]  # unchanged: no instruction by default


def _stub_empty_retrieval(monkeypatch):
    async def empty_vec(driver, index_name, embedding, k, tenant_id=None):
        return []

    async def empty_ft(driver, query, k, tenant_id=None):
        return []

    monkeypatch.setattr(graph, "vector_search", empty_vec)
    monkeypatch.setattr(graph, "fulltext_search", empty_ft)


async def test_search_prepends_query_instruction(monkeypatch):
    settings = Settings(query_instruction="query: ", hyde_enabled=False)
    monkeypatch.setattr(search_mod, "get_settings", lambda: settings)
    _stub_empty_retrieval(monkeypatch)

    captured: list[str] = []

    async def fake_embed_text(client, text):
        captured.append(text)
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(search_mod, "embed_text", fake_embed_text)

    await search_mod.search(Neo4jStore(None), None, "what is x")
    # the dense query was embedded with the instruction; fulltext keeps raw text
    assert captured == ["query: what is x"]


async def test_hyde_applies_query_and_passage_instructions(monkeypatch):
    settings = Settings(
        query_instruction="query: ",
        passage_instruction="passage: ",
        hyde_enabled=True,
        hyde_max_query_words=8,
    )
    monkeypatch.setattr(search_mod, "get_settings", lambda: settings)
    _stub_empty_retrieval(monkeypatch)

    async def fake_generate(client, q):
        return "a hypothetical answer"

    captured: dict[str, list[str]] = {}

    async def fake_embed_texts(client, texts):
        captured["texts"] = list(texts)
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    monkeypatch.setattr(search_mod, "generate_hypothetical_answer", fake_generate)
    monkeypatch.setattr(search_mod, "embed_texts", fake_embed_texts)

    await search_mod.search(Neo4jStore(None), None, "short q")
    # query side gets the query instruction; the answer-shaped hypothetical gets
    # the passage instruction
    assert captured["texts"] == ["query: short q", "passage: a hypothetical answer"]


def test_passage_instruction_changes_schema_signature():
    base = graph.schema_signature(Settings())
    changed = graph.schema_signature(Settings(passage_instruction="passage: "))
    assert base != changed  # stored geometry changed -> guard must notice


def test_query_instruction_does_not_change_schema_signature():
    base = graph.schema_signature(Settings())
    same = graph.schema_signature(Settings(query_instruction="query: "))
    assert base == same  # query side never touches stored vectors
