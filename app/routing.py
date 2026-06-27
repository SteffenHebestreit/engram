"""Adaptive query routing — classify a query and pick the pipeline shape.

The 2026 consensus (GraphRAG-DRIFT, HippoRAG-2's "no simple-QA regression",
production RAG guides) is that one pipeline depth doesn't fit every query: a
factoid shouldn't pay for the widest recall + deepest rerank, and a thematic or
multi-hop question shouldn't be answered by the shallow path. A router classifies
the query and selects a **preset** (see app/presets.py), so a caller — typically
an agent hitting `/search` — gets the right shape automatically instead of
hand-tuning every call.

Routers are a registry, like the other pipeline seams: the built-in `heuristic`
router is pure lexical rules (no LLM); a deployment can register its own (e.g. an
LLM/embedding classifier) without touching `search()`. A request that names its
own `preset`/tuning always wins, so routing never overrides an explicit choice.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .registry import Registry

if TYPE_CHECKING:
    from .config import Settings

# A router maps (query, settings) -> (route_label, preset_name). The label is for
# provenance/diagnostics; the preset name flows through the normal preset/tuning
# resolution. Return preset "" to mean "no overlay" (the documented defaults).
Router = Callable[[str, "Settings"], "tuple[str, str]"]
ROUTERS: Registry[Router] = Registry("router")

# thematic / sensemaking cues — "what are the main themes across the docs"
_GLOBAL_CUES = (
    "main theme", "key theme", "overview", "summar", "what are the",
    "across all", "key topics", "main points", "overall", "in general",
    "high level", "high-level",
)
# multi-hop / comparative cues — answers that need bridging or comparison
_COMPLEX_CUES = (
    "compare", "comparison", "relationship between", "how does", "why does",
    "difference between", "differences between", "impact of", "related to",
    " versus ", " vs ", "trade-off", "tradeoff", "pros and cons",
)


@ROUTERS.register("heuristic")
def _heuristic_router(query: str, settings: "Settings") -> tuple[str, str]:
    """No-LLM query-type classifier.

    - *global* (thematic/sensemaking) and *complex* (multi-hop/comparative, or
      simply long) queries widen recall + rerank depth (`max_quality`), since
      they need more of the corpus surfaced to answer.
    - *factoid* queries use the standard `balanced` shape — no reason to pay for
      the widest recall on a single-fact lookup.
    """
    q = query.lower().strip()
    if any(cue in q for cue in _GLOBAL_CUES):
        return ("global", "max_quality")
    long_query = len(q.split()) > settings.hyde_max_query_words
    if long_query or any(cue in q for cue in _COMPLEX_CUES):
        return ("complex", "max_quality")
    return ("factoid", "balanced")


def get_router(name: str) -> Router:
    return ROUTERS.get(name)
