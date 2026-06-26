import pytest

from app import graph
from app.channels import VectorChannel
from app.config import Settings
from app.store import Store, create_store
from app.store_neo4j import Neo4jStore


def test_create_store_defaults_to_neo4j():
    store = create_store(Settings())
    assert isinstance(store, Neo4jStore)
    # the Neo4j store satisfies the runtime-checkable Store protocol
    assert isinstance(store, Store)


def test_pgvector_backend_is_selectable():
    from app.store_pgvector import PgvectorStore

    store = create_store(Settings(store_backend="pgvector"))
    assert isinstance(store, PgvectorStore)
    assert isinstance(store, Store)


def test_unknown_backend_raises_listing_options():
    with pytest.raises(KeyError, match="unknown store 'redis'.*neo4j"):
        create_store(Settings(store_backend="redis"))


async def test_neo4j_store_vector_search_delegates_with_channel_index(monkeypatch):
    captured = {}

    async def fake_vector_search(driver, index_name, embedding, k):
        captured["args"] = (driver, index_name, embedding, k)
        return [{"id": "x"}]

    monkeypatch.setattr(graph, "vector_search", fake_vector_search)

    channel = VectorChannel(
        name="content",
        index="chunk_content_idx",
        embedding_prop="content_embedding",
        source="text",
        weight=1.0,
    )
    store = Neo4jStore(driver="sentinel-driver")
    out = await store.vector_search(channel, [0.1, 0.2], 5)

    assert out == [{"id": "x"}]
    # the store passes its own driver and the channel's index name through
    assert captured["args"] == ("sentinel-driver", "chunk_content_idx", [0.1, 0.2], 5)


async def test_neo4j_store_graph_proximity_delegates_to_ppr(monkeypatch):
    async def fake_ppr(driver, seed_ids, candidate_ids, damping):
        return {"c": 0.5}

    monkeypatch.setattr(graph, "ppr_proximity", fake_ppr)
    store = Neo4jStore(driver=None)
    assert await store.graph_proximity(["s"], ["c"], 0.85) == {"c": 0.5}
