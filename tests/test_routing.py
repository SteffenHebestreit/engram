"""Adaptive query routing (app/routing.py): the no-LLM heuristic classifier and
its integration into search() (enabled by ROUTER_STRATEGY, overridable per call)."""

from app import graph
from app import search as search_mod
from app.config import Settings
from app.routing import get_router
from app.store_neo4j import Neo4jStore

R = get_router("heuristic")
S = Settings()  # hyde_max_query_words = 8


def test_factoid_short_query_uses_balanced():
    assert R("engram default reranker port", S) == ("factoid", "balanced")


def test_long_query_routes_complex():
    label, preset = R(
        "how do I configure the reranker endpoint for a production deployment today", S
    )
    assert (label, preset) == ("complex", "max_quality")


def test_comparative_cue_routes_complex():
    assert R("compare neo4j and pgvector", S) == ("complex", "max_quality")


def test_thematic_cue_routes_global():
    assert R("what are the main themes across these documents", S) == (
        "global",
        "max_quality",
    )


# ── integration into search() ────────────────────────────────────────────────


def _stub_empty(monkeypatch):
    async def fake_embed_text(client, text):
        return [1.0, 0.0, 0.0]

    async def empty_vec(driver, index_name, embedding, k):
        return []

    async def empty_ft(driver, query, k):
        return []

    monkeypatch.setattr(search_mod, "embed_text", fake_embed_text)
    monkeypatch.setattr(graph, "vector_search", empty_vec)
    monkeypatch.setattr(graph, "fulltext_search", empty_ft)


async def test_search_consults_router_when_enabled(monkeypatch):
    settings = Settings(router_strategy="heuristic", hyde_enabled=False)
    monkeypatch.setattr(search_mod, "get_settings", lambda: settings)
    _stub_empty(monkeypatch)

    seen = []

    def spy(name):
        def router(q, s):
            seen.append(q)
            return ("complex", "max_quality")
        return router

    monkeypatch.setattr(search_mod, "get_router", spy)
    await search_mod.search(Neo4jStore(None), None, "compare neo4j and pgvector")
    assert seen == ["compare neo4j and pgvector"]  # the router was consulted


async def test_explicit_preset_overrides_router(monkeypatch):
    settings = Settings(router_strategy="heuristic", hyde_enabled=False)
    monkeypatch.setattr(search_mod, "get_settings", lambda: settings)
    _stub_empty(monkeypatch)

    seen = []

    def spy(name):
        def router(q, s):
            seen.append(q)
            return ("complex", "max_quality")
        return router

    monkeypatch.setattr(search_mod, "get_router", spy)
    # caller named its own preset -> the router must NOT be consulted
    await search_mod.search(
        Neo4jStore(None), None, "compare neo4j and pgvector", tuning={"preset": "cheap"}
    )
    assert seen == []


async def test_router_off_by_default(monkeypatch):
    settings = Settings(hyde_enabled=False)  # router_strategy defaults ""
    monkeypatch.setattr(search_mod, "get_settings", lambda: settings)
    _stub_empty(monkeypatch)

    def boom(name):
        raise AssertionError("router must not be consulted when disabled")

    monkeypatch.setattr(search_mod, "get_router", boom)
    await search_mod.search(Neo4jStore(None), None, "anything")  # must not raise
