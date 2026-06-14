import math

import pytest

from app import search as search_mod
from app.config import Settings

# embeddings chosen so A/B/D/E/F are cohesive and C is the outlier
CHUNKS = {
    "A": {"emb": [1.0, 0.0, 0.0]},
    "B": {"emb": [0.9, 0.1, 0.0]},
    "C": {"emb": [0.0, 1.0, 0.0]},
    "D": {"emb": [0.95, 0.05, 0.0]},
    "E": {"emb": [0.85, 0.15, 0.0]},
    "F": {"emb": [0.92, 0.08, 0.0]},
}

RERANK_SCORES = {"B": 0.9, "A": 0.8, "F": 0.7, "D": 0.5, "E": 0.3, "C": 0.1}

# DBSF maps a two-value channel to [2/3, 1/3] (z = +/-1 over +/-3 sigma) and a
# single-value channel to [0.5]; total channel weight = 1.0 + 0.9 + 0.8 + 0.7
HI, LO, SOLO = 2 / 3, 1 / 3, 0.5
TOTAL_W = 3.4
RETR = {
    "A": (1.0 * HI + 0.9 * SOLO + 0.7 * LO) / TOTAL_W,  # content + summary + fulltext
    "B": (1.0 * LO) / TOTAL_W,                          # content only
    "C": (0.8 * SOLO) / TOTAL_W,                        # keywords only
    "F": (0.7 * HI) / TOTAL_W,                          # fulltext only
}


def _hit(chunk_id: str, score: float) -> dict:
    return {
        "id": chunk_id,
        "doc_id": "doc1",
        "text": chunk_id,
        "summary": f"summary {chunk_id}",
        "keywords": ["kw"],
        "content_embedding": CHUNKS[chunk_id]["emb"],
        "score": score,
    }


def _sib(chunk_id: str, seed: str, via: str, direction: str, distance: int, strength: float) -> dict:
    return {
        **_hit(chunk_id, 0.0),
        "seed_id": seed,
        "via": via,
        "direction": direction,
        "distance": distance,
        "strength": strength,
    }


@pytest.fixture
def patched(monkeypatch):
    # deterministic unit-test settings: no HyDE detour, no autocut trimming
    settings = Settings(hyde_enabled=False, autocut_enabled=False)
    monkeypatch.setattr(search_mod, "get_settings", lambda: settings)

    async def fake_embed_text(client, text):
        return [1.0, 0.0, 0.0]

    async def fake_vector_search(driver, index_name, embedding, k):
        return {
            "chunk_content_idx": [_hit("A", 0.9), _hit("B", 0.8)],
            "chunk_summary_idx": [_hit("A", 0.95)],
            "chunk_keywords_idx": [_hit("C", 0.5)],
        }[index_name]

    async def fake_fulltext_search(driver, query, k):
        # Lucene-style unbounded scores; F is the best lexical match
        return [{**_hit("F", 8.0)}, {**_hit("A", 4.0)}]

    async def fake_fetch_siblings(driver, chunk_ids, kw_limit, max_hops):
        sibs = []
        if "A" in chunk_ids:
            sibs.append(_sib("D", "A", "sequence", "after", 1, 1.0))
        if "B" in chunk_ids:
            sibs.append(_sib("E", "B", "keyword", "lateral", 1, 2.0))
        return sibs

    async def fake_ppr(driver, seed_ids, candidate_ids, damping):
        return None  # exercise the decay fallback by default

    async def fake_rerank(client, query, texts):
        return [RERANK_SCORES[t] for t in texts]

    monkeypatch.setattr(search_mod, "embed_text", fake_embed_text)
    monkeypatch.setattr(search_mod, "rerank", fake_rerank)
    monkeypatch.setattr(search_mod.graph, "vector_search", fake_vector_search)
    monkeypatch.setattr(search_mod.graph, "fulltext_search", fake_fulltext_search)
    monkeypatch.setattr(search_mod.graph, "fetch_siblings", fake_fetch_siblings)
    monkeypatch.setattr(search_mod.graph, "ppr_proximity", fake_ppr)
    return settings


async def test_full_pipeline(patched):
    results = await search_mod.search(None, None, "test query")
    by_id = {r.chunk_id: r for r in results}

    assert set(by_id) == {"A", "B", "C", "D", "E", "F"}

    # convex fusion over DBSF-normalized channels: corroboration adds, so A
    # (content + summary + fulltext) clearly outscores single-channel hits
    for cid, expected in RETR.items():
        assert math.isclose(by_id[cid].retrieval_score, expected, abs_tol=1e-4)
    assert by_id["A"].retrieval_score > 2 * by_id["F"].retrieval_score
    assert by_id["A"].origin == "vector"
    assert by_id["F"].origin == "fulltext"

    # direct hits sit at graph distance 0 with full proximity
    for cid in "ABCF":
        assert by_id[cid].graph_distance == 0
        assert by_id[cid].graph_proximity == 1.0

    # sibling expansion (decay fallback): D one sequence hop after A,
    # E a keyword sibling of B with 2 shared keywords
    assert by_id["D"].origin == "sibling:sequence:after"
    assert by_id["D"].graph_distance == 1
    assert math.isclose(by_id["D"].graph_proximity, 0.7, abs_tol=1e-4)
    assert math.isclose(
        by_id["D"].retrieval_score, 0.7 * by_id["A"].retrieval_score, abs_tol=1e-4
    )
    assert by_id["E"].origin == "sibling:keyword:lateral"
    assert math.isclose(by_id["E"].graph_proximity, 0.6, abs_tol=1e-4)
    assert math.isclose(
        by_id["E"].retrieval_score, 0.6 * by_id["B"].retrieval_score, abs_tol=1e-4
    )

    # median proximity: outlier C scores below every cohesive result
    cohesive_min = min(by_id[c].median_score for c in "ABDEF")
    assert by_id["C"].median_score < cohesive_min

    # fused = 0.55 * retrieval + 0.30 * median + 0.15 * graph proximity
    a = by_id["A"]
    assert math.isclose(
        a.fused_score,
        0.55 * a.retrieval_score + 0.30 * a.median_score + 0.15 * a.graph_proximity,
        abs_tol=1e-3,
    )

    # final order comes from the reranker
    assert [r.chunk_id for r in results] == ["B", "A", "F", "D", "E", "C"]


async def test_ppr_proximity_replaces_decay(patched, monkeypatch):
    async def fake_ppr(driver, seed_ids, candidate_ids, damping):
        return {"D": 0.9, "E": 0.55}

    monkeypatch.setattr(search_mod.graph, "ppr_proximity", fake_ppr)
    results = await search_mod.search(None, None, "test query")
    by_id = {r.chunk_id: r for r in results}

    # PPR values are used instead of the decay table; distance still reported
    assert math.isclose(by_id["D"].graph_proximity, 0.9, abs_tol=1e-4)
    assert by_id["D"].graph_distance == 1
    assert math.isclose(
        by_id["D"].retrieval_score, 0.9 * by_id["A"].retrieval_score, abs_tol=1e-4
    )
    assert math.isclose(by_id["E"].graph_proximity, 0.55, abs_tol=1e-4)


async def test_autocut_trims_after_score_cliff(patched, monkeypatch):
    patched.autocut_enabled = True
    cliff_scores = {"B": 0.95, "A": 0.9, "F": 0.85, "D": 0.2, "E": 0.15, "C": 0.1}

    async def fake_rerank(client, query, texts):
        return [cliff_scores[t] for t in texts]

    monkeypatch.setattr(search_mod, "rerank", fake_rerank)
    results = await search_mod.search(None, None, "test query")
    # the rerank scores fall off a cliff between F (0.85) and D (0.2):
    # everything from the cliff on is cut
    assert [r.chunk_id for r in results] == ["B", "A", "F"]


async def test_hyde_blends_query_and_hypothetical(patched, monkeypatch):
    patched.hyde_enabled = True
    seen_embeddings = []

    async def fake_generate(client, query):
        return "a hypothetical answer"

    async def fake_embed_texts(client, texts):
        assert texts == ["test query", "a hypothetical answer"]
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]

    async def recording_vector_search(driver, index_name, embedding, k):
        seen_embeddings.append(embedding)
        return []

    async def fake_fulltext(driver, query, k):
        return []

    monkeypatch.setattr(search_mod, "generate_hypothetical_answer", fake_generate)
    monkeypatch.setattr(search_mod, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(search_mod.graph, "vector_search", recording_vector_search)
    monkeypatch.setattr(search_mod.graph, "fulltext_search", fake_fulltext)

    await search_mod.search(None, None, "test query")

    # 50/50 blend of query and hypothetical embedding, re-normalized
    expected = 1 / math.sqrt(2)
    for emb in seen_embeddings:
        assert math.isclose(emb[0], expected, abs_tol=1e-6)
        assert math.isclose(emb[1], expected, abs_tol=1e-6)
        assert emb[2] == 0.0


async def test_top_k_limits_results(patched):
    results = await search_mod.search(None, None, "test query", top_k=2)
    assert len(results) == 2
    assert [r.chunk_id for r in results] == ["B", "A"]


async def test_tuning_overrides_search_shaping(patched):
    # final_top_k via per-request tuning trims to the top 2 by rerank score
    results = await search_mod.search(
        None, None, "test query", tuning={"final_top_k": 2}
    )
    assert [r.chunk_id for r in results] == ["B", "A"]


async def test_tuning_rejects_non_tunable_field(patched):
    with pytest.raises(ValueError, match="non-tunable"):
        await search_mod.search(None, None, "test query", tuning={"neo4j_password": "x"})


async def test_reranker_down_falls_back_to_fused_score(patched, monkeypatch):
    async def fake_rerank(client, query, texts):
        return None  # reranker unavailable

    monkeypatch.setattr(search_mod, "rerank", fake_rerank)
    results = await search_mod.search(None, None, "test query")

    # search still returns results; final order and rerank_score come from the
    # fused score instead of breaking
    assert results
    for r in results:
        assert r.rerank_score == r.fused_score
    fused = [r.fused_score for r in results]
    assert fused == sorted(fused, reverse=True)


async def test_embeddings_down_falls_back_to_fulltext_only(patched, monkeypatch):
    calls = {"vector": 0}

    async def boom_embed(client, text):
        raise RuntimeError("embedding endpoint down")

    async def counting_vector_search(driver, index_name, embedding, k):
        calls["vector"] += 1
        return []

    monkeypatch.setattr(search_mod, "embed_text", boom_embed)
    monkeypatch.setattr(search_mod.graph, "vector_search", counting_vector_search)

    results = await search_mod.search(None, None, "test query")

    # no vector channel was queried, yet search still returns lexical results
    assert calls["vector"] == 0
    ids = {r.chunk_id for r in results}
    assert "F" in ids  # F is the fulltext-only hit in the fixture
    assert by_id_origin(results, "F") == "fulltext"


def by_id_origin(results, chunk_id):
    return next(r.origin for r in results if r.chunk_id == chunk_id)


async def test_no_hits_returns_empty(monkeypatch):
    async def fake_embed_text(client, text):
        return [1.0, 0.0, 0.0]

    async def fake_vector_search(driver, index_name, embedding, k):
        return []

    async def fake_fulltext_search(driver, query, k):
        return []

    monkeypatch.setattr(search_mod, "embed_text", fake_embed_text)
    monkeypatch.setattr(search_mod.graph, "vector_search", fake_vector_search)
    monkeypatch.setattr(search_mod.graph, "fulltext_search", fake_fulltext_search)

    assert await search_mod.search(None, None, "anything") == []
