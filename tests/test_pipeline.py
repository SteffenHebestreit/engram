import math

import pytest

from app import pipeline
from app.channels import resolve_vector_channels
from app.config import Settings


def _sib(chunk_id, via, distance, strength):
    return {"id": chunk_id, "via": via, "distance": distance, "strength": strength}


def test_default_fusion_convex_combines_channels():
    settings = Settings()
    channels = resolve_vector_channels(settings)  # weights 1.0 / 0.9 / 0.8
    # one hit in the content channel, two in summary so DBSF maps to {2/3, 1/3}
    channel_hits = [
        [{"id": "A", "score": 0.9}],
        [{"id": "A", "score": 0.9}, {"id": "B", "score": 0.1}],
        [],
    ]
    fused = pipeline.get_fusion("dbsf_convex")(
        channel_hits, channels, [], settings.fulltext_channel_weight, settings
    )
    total = 1.0 + 0.9 + 0.8 + 0.7
    # A: content solo (0.5) * 1.0 + summary high (2/3) * 0.9
    assert math.isclose(
        fused["A"]["retrieval_score"], (1.0 * 0.5 + 0.9 * (2 / 3)) / total, abs_tol=1e-9
    )
    assert fused["A"]["origin"] == "vector"


def test_custom_fusion_is_selectable():
    @pipeline.FUSIONS.register("first-only")
    def _first_only(channel_hits, channels, fulltext_hits, fulltext_weight, settings):
        return {"only": {"retrieval_score": 1.0}}

    try:
        assert pipeline.get_fusion("first-only")(None, None, None, 0.0, None) == {
            "only": {"retrieval_score": 1.0}
        }
    finally:
        pipeline.FUSIONS._items.pop("first-only", None)


async def test_default_expander_calls_fetch_siblings(monkeypatch):
    seen = {}

    async def fake_fetch(driver, ids, kw_limit, hops):
        seen["args"] = (ids, kw_limit, hops)
        return [{"id": "sib"}]

    monkeypatch.setattr(pipeline.graph, "fetch_siblings", fake_fetch)
    out = await pipeline.get_expander("sequence_keyword")(None, ["seed"], Settings())
    assert out == [{"id": "sib"}]
    assert seen["args"] == (["seed"], Settings().keyword_sibling_limit, Settings().sequence_max_hops)


async def test_decay_proximity_formulas():
    settings = Settings()
    sibs = [
        _sib("seq", "sequence", 2, 1.0),  # 0.7 ** 2
        _sib("kw2", "keyword", 1, 2.0),   # 0.4 + 0.1 * 2
        _sib("kw9", "keyword", 1, 9.0),   # shared capped at 3 -> 0.4 + 0.3
    ]
    out = await pipeline.get_proximity("decay")(None, [], sibs, settings)
    assert math.isclose(out[0], 0.49, abs_tol=1e-9)
    assert math.isclose(out[1], 0.6, abs_tol=1e-9)
    assert math.isclose(out[2], 0.7, abs_tol=1e-9)


async def test_ppr_proximity_falls_back_to_decay_per_sibling(monkeypatch):
    async def fake_ppr(driver, seed_ids, cand_ids, damping):
        return {"X": 0.9}  # Y missing -> decay fallback

    monkeypatch.setattr(pipeline.graph, "ppr_proximity", fake_ppr)
    sibs = [_sib("X", "sequence", 1, 1.0), _sib("Y", "sequence", 1, 1.0)]
    out = await pipeline.get_proximity("ppr")(None, ["s"], sibs, Settings())
    assert math.isclose(out[0], 0.9, abs_tol=1e-9)
    assert math.isclose(out[1], 0.7, abs_tol=1e-9)  # 0.7 ** 1
