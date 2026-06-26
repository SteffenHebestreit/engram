"""Swappable pipeline-stage strategies behind registries.

The search pipeline has three stages whose *algorithm* (not just weights) a
deployment might want to swap: how channel scores are fused, how the graph is
expanded around seeds, and how graph proximity is computed. Each is a registry
with the current implementation as the default, so `search()` reads as a fixed
skeleton selecting strategies by config key, and an alternative (e.g. RRF
fusion, a typed-relation expander) is a registration rather than an edit.

Defaults are extracted verbatim from the original inline pipeline; the scoring
tests pin the exact math.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Awaitable, Callable, Protocol

from .registry import Registry
from .scoring import dbsf_normalize

if TYPE_CHECKING:
    from .channels import VectorChannel
    from .config import Settings
    from .store import Store


# ── Fusion ──────────────────────────────────────────────────────────────────
# Combine the per-channel hit lists (vector channels + fulltext) into a
# candidate pool keyed by chunk id, each with a `retrieval_score`.

class Fusion(Protocol):
    def __call__(
        self,
        channel_hits: list[list[dict[str, Any]]],
        channels: "list[VectorChannel]",
        fulltext_hits: list[dict[str, Any]],
        fulltext_weight: float,
        settings: "Settings",
    ) -> dict[str, dict[str, Any]]: ...


FUSIONS: Registry[Fusion] = Registry("fusion")


@FUSIONS.register("dbsf_convex")
def _fuse_dbsf_convex(
    channel_hits: list[list[dict[str, Any]]],
    channels: "list[VectorChannel]",
    fulltext_hits: list[dict[str, Any]],
    fulltext_weight: float,
    settings: "Settings",
) -> dict[str, dict[str, Any]]:
    """DBSF-normalize each channel, then convex-combine: a chunk corroborated
    by several channels outranks a single-channel hit."""
    candidates: dict[str, dict[str, Any]] = {}
    total_weight = sum(ch.weight for ch in channels) + fulltext_weight

    def add_channel(
        hits: list[dict[str, Any]], weight: float, origin: str, name: str
    ) -> None:
        normalized = dbsf_normalize([hit["score"] for hit in hits])
        for hit, norm_score in zip(hits, normalized):
            cand = candidates.get(hit["id"])
            if cand is None:
                cand = candidates[hit["id"]] = {
                    **hit,
                    "weighted_sum": 0.0,
                    "graph_proximity": 1.0,
                    "graph_distance": 0,
                    "origin": origin,
                    # provenance: which retrieval channels surfaced this chunk
                    # (feeds per-stage attribution in the eval harness)
                    "channels": [],
                }
            cand["weighted_sum"] += weight * norm_score
            cand["channels"].append(name)
            if origin == "vector":
                cand["origin"] = "vector"

    for channel, hits in zip(channels, channel_hits):
        add_channel(hits, channel.weight, "vector", channel.name)
    add_channel(fulltext_hits, fulltext_weight, "fulltext", "fulltext")

    for cand in candidates.values():
        cand["retrieval_score"] = cand.pop("weighted_sum") / total_weight
    return candidates


def get_fusion(name: str) -> Fusion:
    return FUSIONS.get(name)


# ── Expander ────────────────────────────────────────────────────────────────
# Find graph neighbours of the seed chunks to pull into the candidate pool.

Expander = Callable[
    ["Store", list[str], "Settings"], Awaitable[list[dict[str, Any]]]
]
EXPANDERS: Registry[Expander] = Registry("expander")


@EXPANDERS.register("sequence_keyword")
async def _expand_sequence_keyword(
    store: "Store", seed_ids: list[str], settings: "Settings"
) -> list[dict[str, Any]]:
    """Directional NEXT_CHUNK walk + shared-keyword siblings (the document
    profile's traversal)."""
    return await store.fetch_siblings(
        seed_ids, settings.keyword_sibling_limit, settings.sequence_max_hops
    )


def get_expander(name: str) -> Expander:
    return EXPANDERS.get(name)


# ── Proximity ───────────────────────────────────────────────────────────────
# Graph-proximity indication value per expanded sibling (parallel to the
# siblings list), feeding both the inherited score and the fused score.

Proximity = Callable[
    ["Store", list[str], list[dict[str, Any]], "Settings"],
    Awaitable[list[float]],
]
PROXIMITIES: Registry[Proximity] = Registry("proximity")


def _decay_proximity(sib: dict[str, Any], settings: "Settings") -> float:
    """Fixed per-hop fade: sequence siblings decay with distance, keyword
    siblings start from a base plus a per-shared-keyword bonus (capped at 3)."""
    if sib["via"] == "sequence":
        return settings.sequence_proximity_decay ** sib["distance"]
    shared = min(sib["strength"], 3.0)
    return (
        settings.keyword_sibling_base_decay
        + settings.keyword_sibling_shared_bonus * shared
    )


@PROXIMITIES.register("ppr")
async def _proximity_ppr(
    store: "Store",
    seed_ids: list[str],
    siblings: list[dict[str, Any]],
    settings: "Settings",
) -> list[float]:
    """Graph-activation proximity from the seeds (personalized PageRank on the
    Neo4j backend), with the decay table as the per-sibling fallback when the
    backend reports no graph-proximity capability (e.g. GDS missing, or the
    pgvector backend)."""
    ppr: dict[str, float] | None = None
    if siblings:
        sibling_ids = list({s["id"] for s in siblings})
        ppr = await store.graph_proximity(
            seed_ids, sibling_ids, settings.ppr_damping
        )
    out: list[float] = []
    for sib in siblings:
        value = ppr.get(sib["id"]) if ppr else None
        out.append(value if value is not None else _decay_proximity(sib, settings))
    return out


@PROXIMITIES.register("decay")
async def _proximity_decay(
    store: "Store",
    seed_ids: list[str],
    siblings: list[dict[str, Any]],
    settings: "Settings",
) -> list[float]:
    return [_decay_proximity(sib, settings) for sib in siblings]


def get_proximity(name: str) -> Proximity:
    return PROXIMITIES.get(name)
