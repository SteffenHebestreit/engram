"""Unit tests for community-synthesis orchestration (app/community.py)."""

import pytest

from app import community as community_mod
from app.community import _top_keywords, build_communities, search_communities


def test_top_keywords_ranks_by_frequency():
    lists = [["a", "b"], ["A", "c"], ["a"]]  # case-insensitive
    assert _top_keywords(lists, limit=2) == ["a", "b"] or _top_keywords(lists, limit=2)[0] == "a"
    assert _top_keywords(lists)[0] == "a"  # most frequent


class FakeStore:
    def __init__(self, detected, vectors=None):
        self._detected = detected
        self._vectors = vectors or []
        self.saved = None

    async def detect_communities(self, min_size):
        return self._detected

    async def save_communities(self, communities):
        self.saved = communities
        return len(communities)

    async def community_vectors(self):
        return self._vectors


async def test_build_communities_persists_with_reports(monkeypatch):
    detected = [
        {
            "id": 1,
            "chunk_ids": ["d:0", "d:1"],
            "summaries": ["about cats", "more cats"],
            "keyword_lists": [["cat", "pet"], ["cat"]],
        }
    ]
    store = FakeStore(detected)

    async def fake_report(client, summaries, keywords):
        return {"title": "Cats", "summary": "All about cats."}

    async def fake_embed(client, text):
        return [1.0, 0.0, 0.0]

    monkeypatch.setattr(community_mod, "generate_community_report", fake_report)
    monkeypatch.setattr(community_mod, "embed_text", fake_embed)

    result = await build_communities(store, http=object(), generate_reports=True)
    assert result == {"communities": 1}
    comm = store.saved[0]
    assert comm["title"] == "Cats"
    assert comm["summary"] == "All about cats."
    assert comm["keywords"][0] == "cat"          # most frequent keyword
    assert comm["chunk_ids"] == ["d:0", "d:1"]
    assert comm["report_embedding"] == [1.0, 0.0, 0.0]  # report embedded for search


async def test_search_communities_ranks_by_cosine(monkeypatch):
    vectors = [
        {"id": "1", "title": "Cats", "summary": "", "keywords": ["cat"],
         "member_count": 3, "report_embedding": [1.0, 0.0]},
        {"id": "2", "title": "Dogs", "summary": "", "keywords": ["dog"],
         "member_count": 2, "report_embedding": [0.0, 1.0]},
    ]
    store = FakeStore(detected=[], vectors=vectors)

    async def fake_embed(client, text):
        return [0.9, 0.1]  # closest to the "Cats" community

    monkeypatch.setattr(community_mod, "embed_text", fake_embed)
    ranked = await search_communities(store, http=object(), query="felines", top_k=2)
    assert [c["id"] for c in ranked] == ["1", "2"]
    assert ranked[0]["score"] > ranked[1]["score"]


async def test_search_communities_empty_when_no_vectors():
    store = FakeStore(detected=[], vectors=[])
    assert await search_communities(store, http=object(), query="x") == []


async def test_build_communities_without_reports_skips_llm(monkeypatch):
    detected = [
        {"id": 7, "chunk_ids": ["d:0"], "summaries": ["x"], "keyword_lists": [["k"]]}
    ]
    store = FakeStore(detected)

    async def boom(client, summaries, keywords):
        raise AssertionError("LLM must not be called when reports are off")

    monkeypatch.setattr(community_mod, "generate_community_report", boom)
    result = await build_communities(store, http=None, generate_reports=False)
    assert result == {"communities": 1}
    assert store.saved[0]["title"] == ""


async def test_unsupported_backend_raises():
    class NoCommunityStore:
        async def detect_communities(self, min_size):
            return None  # e.g. pgvector

    with pytest.raises(NotImplementedError, match="neo4j backend with the GDS"):
        await build_communities(NoCommunityStore(), http=None)
