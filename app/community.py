"""Community synthesis — a GraphRAG-style global/theme layer (opt-in, offline).

engram's default retrieval is local (per-query). For corpus-wide
"what are the main themes?" questions, `build_communities` clusters the
chunk/keyword graph with Leiden, names each cluster with an LLM-written report,
and persists a Community layer (`(:Community)` + `(:Chunk)-[:IN_COMMUNITY]->`)
in the same store. It is meant to run as an offline batch (a script or an
explicit endpoint), never on the search hot path.

It is a Neo4j + GDS capability: `store.detect_communities()` returns None on a
backend without Leiden (e.g. pgvector), and `build_communities` raises
NotImplementedError so the caller can surface a clear 501.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import httpx
import numpy as np

from .config import get_settings
from .embeddings import embed_text
from .llm import generate_community_report

if TYPE_CHECKING:
    from .store import Store


def _top_keywords(keyword_lists: list[list[str]], limit: int = 8) -> list[str]:
    """Most frequent keywords across a community's member chunks."""
    counter: Counter[str] = Counter()
    for kws in keyword_lists:
        for kw in kws or []:
            counter[kw.lower()] += 1
    return [kw for kw, _ in counter.most_common(limit)]


async def build_communities(
    store: "Store", http: httpx.AsyncClient | None, generate_reports: bool = True
) -> dict:
    """Detect communities, (optionally) write LLM reports, and persist the layer.

    Returns `{"communities": n}`. Raises NotImplementedError when the store has
    no community-detection capability (no GDS/Leiden).
    """
    settings = get_settings()
    detected = await store.detect_communities(settings.community_min_size)
    if detected is None:
        raise NotImplementedError(
            "community detection requires the neo4j backend with the GDS plugin"
        )

    communities = []
    for comm in detected:
        keywords = _top_keywords(comm["keyword_lists"])
        record = {
            "id": comm["id"],
            "chunk_ids": comm["chunk_ids"],
            "keywords": keywords,
            "title": "",
            "summary": "",
        }
        if generate_reports and http is not None:
            report = await generate_community_report(http, comm["summaries"], keywords)
            if report:
                record["title"] = report["title"]
                record["summary"] = report["summary"]
        # embed a text view of the community so it can be ranked by a global
        # query later; needs the embedding endpoint (skipped when http is None)
        if http is not None:
            report_text = " ".join(
                p for p in [record["title"], record["summary"], ", ".join(keywords)] if p
            ).strip()
            if report_text:
                record["report_embedding"] = await embed_text(http, report_text)
        communities.append(record)

    await store.save_communities(communities)
    return {"communities": len(communities)}


async def search_communities(
    store: "Store", http: httpx.AsyncClient, query: str, top_k: int = 10
) -> list[dict]:
    """Rank the community/theme layer against a query (GraphRAG global search).

    Embeds the query and scores each community by cosine similarity to its
    report embedding. Returns the top-k as `{id, title, summary, keywords,
    member_count, score}`, or [] when no community has an embedding.
    """
    vectors = await store.community_vectors()
    if not vectors:
        return []
    q = np.asarray(await embed_text(http, query), dtype=np.float64)
    q_norm = np.linalg.norm(q) or 1.0
    scored = []
    for comm in vectors:
        v = np.asarray(comm["report_embedding"], dtype=np.float64)
        v_norm = np.linalg.norm(v) or 1.0
        score = float(q @ v / (q_norm * v_norm))
        scored.append(
            {
                "id": comm["id"],
                "title": comm["title"],
                "summary": comm["summary"],
                "keywords": comm["keywords"],
                "member_count": comm["member_count"],
                "score": round(score, 4),
            }
        )
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]
