"""Named search presets — convenience bundles over per-request tuning.

A preset is just a dict of `SEARCH_TUNABLE_FIELDS` overrides with a name, so it
flows through the existing `Settings.tuned()` validation unchanged. It saves a
caller from spelling out the same five knobs every time to get a
"cheap"/"balanced"/"max-quality" shape.

Selection (lowest to highest precedence):
1. `SEARCH_PRESET` env default (process-wide baseline; "" = none),
2. a `preset` key inside a `/search` `tuning` object (per request),
3. explicit per-request `tuning` fields (always win over the preset).

Presets are intentionally a thin layer, not a second config system: every value
must be a tunable search-shaping field (enforced at import below).
"""

from __future__ import annotations

from .config import SEARCH_TUNABLE_FIELDS

# Each preset's keys must all be in SEARCH_TUNABLE_FIELDS. Values are starting
# points — tune them to your corpus.
PRESETS: dict[str, dict] = {
    # the documented defaults: no overlay
    "balanced": {},
    # minimize per-query latency/cost: no HyDE LLM hop, no cross-encoder, decay
    # proximity instead of GDS PageRank, shallower channels
    "cheap": {
        "hyde_enabled": False,
        "reranker_enabled": False,
        "graph_proximity_mode": "decay",
        "top_k_per_index": 8,
        "seed_count": 4,
    },
    # widen recall and diversity, rerank a deeper shortlist
    "max_quality": {
        "hyde_enabled": True,
        "top_k_per_index": 20,
        "seed_count": 12,
        "rerank_top_k": 30,
        "mmr_lambda": 0.5,
    },
}

# fail fast if a preset references a non-tunable field
for _name, _overlay in PRESETS.items():
    _bad = set(_overlay) - SEARCH_TUNABLE_FIELDS
    if _bad:
        raise ValueError(f"preset {_name!r} has non-tunable fields: {sorted(_bad)}")


def get_preset(name: str) -> dict:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; available: {sorted(PRESETS)}"
        ) from None


def apply_preset(tuning: dict | None, default_preset: str = "") -> dict:
    """Merge the selected preset with the per-request tuning.

    The preset name is `tuning['preset']` if present, else `default_preset`.
    Returns a plain tunable dict (the `preset` meta-key removed) with explicit
    per-request fields layered on top of the preset overlay. Raises ValueError
    on an unknown preset name.
    """
    tuning = dict(tuning or {})
    preset_name = tuning.pop("preset", None) or default_preset
    overlay = get_preset(preset_name) if preset_name else {}
    return {**overlay, **tuning}
