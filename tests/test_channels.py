from app import graph
from app import ingest as ingest_mod
from app.channels import (
    VectorChannel,
    get_channel_source,
    resolve_vector_channels,
)
from app.config import Settings
from app.llm import ExtractionResult
from app.store_neo4j import Neo4jStore


def test_default_channels_match_legacy_shape():
    channels = resolve_vector_channels(Settings())
    assert [(c.index, c.embedding_prop, c.weight) for c in channels] == [
        ("chunk_content_idx", "content_embedding", 1.0),
        ("chunk_summary_idx", "summary_embedding", 0.9),
        ("chunk_keywords_idx", "keywords_embedding", 0.8),
    ]
    # legacy per-channel weight env overrides still flow through
    reweighted = resolve_vector_channels(Settings(summary_channel_weight=0.5))
    assert reweighted[1].weight == 0.5


def test_channel_enable_flags_drop_channels():
    # disabling both leaves only the canonical content channel (1 embedding/chunk)
    only_content = resolve_vector_channels(
        Settings(summary_channel_enabled=False, keywords_channel_enabled=False)
    )
    assert [c.name for c in only_content] == ["content"]
    # disabling just keywords keeps content + summary
    no_kw = resolve_vector_channels(Settings(keywords_channel_enabled=False))
    assert [c.name for c in no_kw] == ["content", "summary"]


def test_explicit_channel_override_is_used_verbatim():
    custom = [
        VectorChannel(
            name="content",
            index="chunk_content_idx",
            embedding_prop="content_embedding",
            source="text",
            weight=1.0,
        )
    ]
    assert resolve_vector_channels(Settings(vector_channels=custom)) == custom


def test_channel_sources_derive_embed_text():
    meta = [
        ExtractionResult(keywords=["a", "b"], summary="sum one"),
        ExtractionResult(keywords=[], summary="sum two"),
    ]
    chunks = ["chunk one", "chunk two"]
    assert get_channel_source("text")(chunks, meta) == chunks
    assert get_channel_source("summary")(chunks, meta) == ["sum one", "sum two"]
    # keywords joined, falling back to the summary when a chunk has none
    assert get_channel_source("keywords")(chunks, meta) == ["a, b", "sum two"]


async def test_ingest_builds_embeddings_map_per_channel(monkeypatch):
    custom = [
        VectorChannel(
            name="content",
            index="chunk_content_idx",
            embedding_prop="content_embedding",
            source="text",
            weight=1.0,
        ),
        VectorChannel(
            name="kw",
            index="chunk_kw_idx",
            embedding_prop="kw_embedding",
            source="keywords",
            weight=0.5,
        ),
    ]
    settings = Settings(vector_channels=custom)
    monkeypatch.setattr(ingest_mod, "get_settings", lambda: settings)
    monkeypatch.setattr(ingest_mod, "get_chunker", lambda name: lambda text, s: [text])

    async def fake_extract(client, chunk):
        return ExtractionResult(keywords=["k"], summary="s")

    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract)

    async def fake_embed_texts(client, texts):
        return [[float(len(t))] for t in texts]

    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed_texts)

    captured = {}

    async def fake_get(driver, doc_id):
        return None

    async def fake_save(driver, doc_id, title, sources, chunk_rows):
        captured["rows"] = chunk_rows

    monkeypatch.setattr(graph, "get_document", fake_get)
    monkeypatch.setattr(graph, "save_document", fake_save)

    await ingest_mod.ingest_document(
        Neo4jStore(None), None, "hello", title="t", source="src"
    )

    row = captured["rows"][0]
    # one embedding entry per channel, keyed by the channel's embedding_prop
    assert set(row["embeddings"]) == {"content_embedding", "kw_embedding"}
    assert row["embeddings"]["content_embedding"] == [5.0]  # len("hello")
    assert row["embeddings"]["kw_embedding"] == [1.0]  # len("k")
