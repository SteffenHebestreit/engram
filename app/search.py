import asyncio
from typing import TYPE_CHECKING

import httpx
import numpy as np
from neo4j import AsyncDriver

from . import graph
from .channels import resolve_vector_channels
from .config import get_settings

if TYPE_CHECKING:
    from .config import Settings
from .embeddings import embed_text, embed_texts
from .llm import generate_hypothetical_answer
from .models import SearchResult
from .pipeline import get_expander, get_fusion, get_proximity
from .rerank import rerank
from .scoring import autocut, median_proximity_scores, mmr_select


async def search(
    driver: AsyncDriver,
    http: httpx.AsyncClient,
    query: str,
    top_k: int | None = None,
    tuning: dict | None = None,
) -> list[SearchResult]:
    """Search pipeline:

    1. embed the query (short queries: HyDE — blend in the embedding of an
       LLM-written hypothetical answer), hit all three vector indexes
       (content/summary/keywords) plus the fulltext index as a lexical
       fourth channel
    2. DBSF-normalize each channel and fuse as a convex combination, so a
       chunk corroborated by several channels outranks a single-channel hit
    3. expand the top seeds along the directional NEXT_CHUNK chain and via
       shared keywords; graph proximity comes from personalized PageRank
       seeded on the direct hits (decay-based fallback when GDS is missing)
    4. score every candidate by proximity to the median of the result set and
       fuse retrieval, median and graph proximity into a total value
    5. pick the rerank shortlist with MMR so near-duplicate neighbours don't
       crowd out other relevant regions
    6. rerank, sort by reranker score and autocut after the first score cliff
    """
    settings = get_settings().tuned(tuning or {})
    final_top_k = top_k or settings.final_top_k

    # the lexical channel only needs the raw query text, so it runs while the
    # query embedding is computed — which can take a while when HyDE adds an
    # LLM round trip
    fulltext_task = asyncio.create_task(
        graph.fulltext_search(driver, query, settings.top_k_per_index)
    )
    try:
        query_emb = await _query_embedding(http, query, settings)
    except BaseException:
        fulltext_task.cancel()
        raise

    channels = resolve_vector_channels(settings)
    *channel_hits, fulltext_hits = await asyncio.gather(
        *(
            graph.vector_search(driver, ch.index, query_emb, settings.top_k_per_index)
            for ch in channels
        ),
        fulltext_task,
    )

    # fuse the channel hit lists into a candidate pool with retrieval scores
    # (default: DBSF-normalize each channel, then convex-combine so a chunk
    # corroborated by several channels outranks a single-channel hit)
    candidates = get_fusion(settings.fusion_strategy)(
        list(channel_hits),
        channels,
        fulltext_hits,
        settings.fulltext_channel_weight,
        settings,
    )

    if not candidates:
        return []

    # graph expansion around the strongest seeds
    seeds = sorted(candidates.values(), key=lambda c: c["retrieval_score"], reverse=True)
    seeds = seeds[: settings.seed_count]
    seed_scores = {s["id"]: s["retrieval_score"] for s in seeds}

    siblings = await get_expander(settings.expander_strategy)(
        driver, list(seed_scores), settings
    )

    # graph proximity per sibling (parallel to `siblings`): personalized
    # PageRank spreads activation from the seeds and accumulates over multiple
    # paths; the decay formula is the fallback when GDS is unavailable
    proximities = await get_proximity(settings.graph_proximity_mode)(
        driver, list(seed_scores), siblings, settings
    )

    for sib, proximity in zip(siblings, proximities):
        # total value a sibling inherits: seed score scaled by graph proximity
        score = seed_scores[sib["seed_id"]] * proximity
        existing = candidates.get(sib["id"])
        if existing is None:
            candidates[sib["id"]] = {
                **sib,
                "retrieval_score": score,
                "graph_proximity": proximity,
                "graph_distance": sib["distance"],
                "origin": f"sibling:{sib['via']}:{sib['direction']}",
            }
            continue
        # already pooled (direct hit or sibling of another seed): keep the
        # best score and the closest graph relation independently
        if score > existing["retrieval_score"]:
            existing["retrieval_score"] = score
        if proximity > existing["graph_proximity"]:
            existing["graph_proximity"] = proximity
        if existing["origin"].startswith("sibling:") and sib["distance"] < existing["graph_distance"]:
            existing["graph_distance"] = sib["distance"]
            existing["origin"] = f"sibling:{sib['via']}:{sib['direction']}"

    pool = list(candidates.values())

    # proximity to the median of the whole result set; outliers score low
    median_scores = median_proximity_scores([c["content_embedding"] for c in pool])
    for cand, median_score in zip(pool, median_scores):
        cand["median_score"] = median_score
        cand["fused_score"] = (
            settings.retrieval_weight * cand["retrieval_score"]
            + settings.median_weight * median_score
            + settings.graph_proximity_weight * cand["graph_proximity"]
        )

    # MMR shortlist: overlapping/adjacent chunks carry near-identical content,
    # so pure top-k by fused score would feed the reranker redundant text
    picked = mmr_select(
        [c["fused_score"] for c in pool],
        [c["content_embedding"] for c in pool],
        settings.rerank_top_k,
        settings.mmr_lambda,
    )
    shortlist = [pool[i] for i in picked]

    rerank_scores = await rerank(http, query, [c["text"] for c in shortlist])
    for cand, score in zip(shortlist, rerank_scores):
        cand["rerank_score"] = score
    shortlist.sort(key=lambda c: c["rerank_score"], reverse=True)

    final = shortlist[:final_top_k]
    if settings.autocut_enabled:
        keep = autocut(
            [c["rerank_score"] for c in final],
            settings.autocut_min_keep,
            settings.autocut_min_gap,
        )
        final = final[:keep]

    return [
        SearchResult(
            chunk_id=c["id"],
            document_id=c["doc_id"],
            text=c["text"],
            summary=c["summary"] or "",
            keywords=c["keywords"] or [],
            origin=c["origin"],
            graph_distance=c["graph_distance"],
            graph_proximity=round(c["graph_proximity"], 4),
            retrieval_score=round(c["retrieval_score"], 4),
            median_score=round(c["median_score"], 4),
            fused_score=round(c["fused_score"], 4),
            rerank_score=round(c["rerank_score"], 4),
        )
        for c in final
    ]


async def _query_embedding(
    http: httpx.AsyncClient, query: str, settings: "Settings"
) -> list[float]:
    """Embedding of the query; short queries get the HyDE treatment.

    Short/ambiguous queries embed far from the answer-shaped chunks they
    should match. HyDE bridges that asymmetry by blending in the embedding of
    a hypothetical answer. Long queries skip it (HyDE can slightly hurt
    well-formed queries), and any LLM failure falls back to the plain query.
    The fulltext channel always keeps the raw query text.
    """
    use_hyde = (
        settings.hyde_enabled
        and len(query.split()) <= settings.hyde_max_query_words
    )
    if use_hyde:
        hypothetical = await generate_hypothetical_answer(http, query)
        if hypothetical:
            query_emb, hypo_emb = await embed_texts(http, [query, hypothetical])
            w = settings.hyde_query_weight
            blended = w * np.asarray(query_emb) + (1.0 - w) * np.asarray(hypo_emb)
            norm = np.linalg.norm(blended)
            if norm > 0:
                return (blended / norm).tolist()
    return await embed_text(http, query)
