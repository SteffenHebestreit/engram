import asyncio
import logging
import time
from typing import TYPE_CHECKING

import httpx
import numpy as np

from .channels import resolve_vector_channels
from .config import get_settings

if TYPE_CHECKING:
    from .config import Settings
    from .store import Store
from .embeddings import embed_sparse_text, embed_text, embed_texts
from .llm import generate_hypothetical_answer
from .models import SearchResult
from .pipeline import get_expander, get_fusion, get_proximity
from .presets import apply_preset
from .rerank import get_reranker
from .routing import get_router
from .scoring import (
    autocut,
    median_proximity_scores,
    mmr_select,
    recency_blend,
    recency_decay,
    sparse_scores,
)

log = logging.getLogger(__name__)


async def search(
    store: "Store",
    http: httpx.AsyncClient,
    query: str,
    top_k: int | None = None,
    tuning: dict | None = None,
    tenant_id: str | None = None,
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

    With `tenant_id`, every chunk-surfacing read is restricted to that tenant for
    0% cross-tenant leakage: the dense + fulltext channels filter in-scan, and the
    graph siblings (which can reach another tenant's chunk via a shared keyword)
    are filtered here before they enter the pool. `None` = untenanted search.
    """
    started = time.perf_counter()
    base = get_settings()
    # adaptive routing: when enabled and the caller didn't name its own preset,
    # a router classifies the query and picks the preset (pipeline shape). An
    # explicit per-request preset always wins, so routing never overrides it.
    route = None
    default_preset = base.search_preset
    if base.router_strategy and not (tuning and tuning.get("preset")):
        route, default_preset = get_router(base.router_strategy)(query, base)
    # resolve the preset (router/env default or tuning["preset"]) into concrete
    # overrides, with explicit per-request fields winning, then validate
    settings = base.tuned(apply_preset(tuning, default_preset))
    final_top_k = top_k or settings.final_top_k

    # the lexical channel only needs the raw query text, so it runs while the
    # query embedding is computed — which can take a while when HyDE adds an
    # LLM round trip
    fulltext_task = asyncio.create_task(
        store.fulltext_search(query, settings.top_k_per_index, tenant_id)
    )
    try:
        query_emb = await _query_embedding(http, query, settings)
    except Exception:
        # embedding endpoint unreachable: degrade to fulltext-only search rather
        # than breaking. The vector channels are skipped; the lexical channel
        # plus the graph/median/rerank stages still run on the chunk embeddings
        # already stored in Neo4j.
        log.warning("embedding endpoint unavailable; falling back to fulltext-only search")
        query_emb = None
    except BaseException:
        # genuine cancellation / interpreter shutdown: don't swallow it
        fulltext_task.cancel()
        raise

    # no query embedding -> no vector channels; the lexical channel carries it
    channels = resolve_vector_channels(settings) if query_emb is not None else []
    *channel_hits, fulltext_hits = await asyncio.gather(
        *(
            store.vector_search(ch, query_emb, settings.top_k_per_index, tenant_id)
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
        store, list(seed_scores), settings
    )

    # graph expansion can cross tenants: a HAS_KEYWORD sibling (or any sibling row)
    # may belong to another tenant. The seeds are already tenant-filtered, but the
    # siblings are not — drop any that aren't this tenant's before they enter the
    # pool (the final isolation gate on the graph path).
    if tenant_id is not None:
        siblings = [s for s in siblings if s.get("tenant_id") == tenant_id]

    # graph proximity per sibling (parallel to `siblings`): personalized
    # PageRank spreads activation from the seeds and accumulates over multiple
    # paths; the decay formula is the fallback when GDS is unavailable
    proximities = await get_proximity(settings.graph_proximity_mode)(
        store, list(seed_scores), siblings, settings
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
                "channels": [f"graph:{sib['via']}"],
            }
            continue
        # already pooled (direct hit or sibling of another seed): keep the
        # best score and the closest graph relation independently
        if score > existing["retrieval_score"]:
            existing["retrieval_score"] = score
        if proximity > existing["graph_proximity"]:
            existing["graph_proximity"] = proximity
        # credit the graph reach in the provenance even when a retrieval channel
        # also surfaced this chunk (so attribution counts graph's contribution)
        existing.setdefault("channels", []).append(f"graph:{sib['via']}")
        if existing["origin"].startswith("sibling:") and sib["distance"] < existing["graph_distance"]:
            existing["graph_distance"] = sib["distance"]
            existing["origin"] = f"sibling:{sib['via']}:{sib['direction']}"

    pool = list(candidates.values())

    # memory write-path (M1): collapse near-duplicate clusters so a re-ingested
    # paraphrase flood can't crowd distinct material out of the shortlist. Opt-in;
    # non-destructive (the duplicate chunks still exist, just not surfaced twice).
    if settings.dedup_enabled:
        pool = await _collapse_near_dups(store, pool)

    # optional learned-sparse (BGE-M3 lexical) re-scoring of the candidate pool:
    # an exact-term signal that dense pooling smooths away (rare entities, IDs,
    # numbers). Opt-in and degrades to all-zeros when off / endpoint down.
    sparse_pool_scores = await _sparse_pool_scores(store, http, query, pool, settings)

    # proximity to the median of the whole result set; outliers score low
    median_scores = median_proximity_scores([c["content_embedding"] for c in pool])
    for cand, median_score, sparse_score in zip(pool, median_scores, sparse_pool_scores):
        cand["median_score"] = median_score
        # sparse is a re-scoring signal over the existing pool, not a retrieval
        # channel — it doesn't *surface* candidates, so its contribution lives in
        # `sparse_score`, not in `channels` (which is retrieval provenance)
        cand["sparse_score"] = sparse_score
        cand["fused_score"] = (
            settings.retrieval_weight * cand["retrieval_score"]
            + settings.median_weight * median_score
            + settings.graph_proximity_weight * cand["graph_proximity"]
            + settings.sparse_weight * sparse_score
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

    rerank_scores = None
    if settings.reranker_enabled:
        reranker = get_reranker(settings.reranker_strategy)
        rerank_scores = await reranker(http, query, [c["text"] for c in shortlist])
    rerank_fallback = rerank_scores is None
    if rerank_fallback:
        # reranker unavailable or disabled: degrade to the fused score as the
        # final signal rather than failing the whole search (cf. HyDE and PPR)
        log.warning("reranker unavailable or disabled; falling back to fused score")
        rerank_scores = [c["fused_score"] for c in shortlist]
    for cand, score in zip(shortlist, rerank_scores):
        cand["rerank_score"] = score
        cand["recency_score"] = 0.0

    # recency / temporal decay (opt-in, the agent-memory signal): blend an
    # exponential recency factor on each candidate's document age into the final
    # ordering, so among similarly-relevant results the newer ones rank higher.
    # Applied AFTER reranking, so it is an orthogonal signal the cross-encoder
    # can't overwrite. The ranking key is the blend; pure relevance otherwise.
    rank_scores = list(rerank_scores)
    if settings.recency_enabled and shortlist:
        ages = await store.get_chunk_recency([c["id"] for c in shortlist])
        if ages:
            half_life = settings.recency_half_life_days * 86400.0
            recency = [
                recency_decay(ages[c["id"]], half_life) if c["id"] in ages else 0.5
                for c in shortlist
            ]
            rank_scores = recency_blend(rerank_scores, recency, settings.recency_weight)
            for cand, rec in zip(shortlist, recency):
                cand["recency_score"] = rec
    for cand, rank in zip(shortlist, rank_scores):
        cand["_rank_score"] = rank
    shortlist.sort(key=lambda c: c["_rank_score"], reverse=True)

    final = shortlist[:final_top_k]
    if settings.autocut_enabled:
        keep = autocut(
            [c["_rank_score"] for c in final],
            settings.autocut_min_keep,
            settings.autocut_min_gap,
        )
        final = final[:keep]

    # one structured diagnostics line per search (visible at LOG_LEVEL=DEBUG)
    log.debug(
        "search done: route=%s query_words=%d embed_fallback=%s candidates=%d "
        "shortlist=%d rerank_fallback=%s results=%d elapsed_ms=%.1f",
        route or "-",
        len(query.split()),
        query_emb is None,
        len(candidates),
        len(shortlist),
        rerank_fallback,
        len(final),
        (time.perf_counter() - started) * 1000.0,
    )

    return [
        SearchResult(
            chunk_id=c["id"],
            document_id=c["doc_id"],
            text=c["text"],
            summary=c["summary"] or "",
            keywords=c["keywords"] or [],
            origin=c["origin"],
            channels=sorted(set(c.get("channels", []))),
            near_dup_of=c.get("near_dup_of"),
            graph_distance=c["graph_distance"],
            graph_proximity=round(c["graph_proximity"], 4),
            retrieval_score=round(c["retrieval_score"], 4),
            median_score=round(c["median_score"], 4),
            sparse_score=round(c.get("sparse_score", 0.0), 4),
            fused_score=round(c["fused_score"], 4),
            rerank_score=round(c["rerank_score"], 4),
            recency_score=round(c.get("recency_score", 0.0), 4),
        )
        for c in final
    ]


async def _collapse_near_dups(store: "Store", pool: list[dict]) -> list[dict]:
    """Collapse near-duplicate clusters in the candidate pool to one
    representative each (the best-retrieved member), so re-ingested paraphrases
    don't crowd out distinct chunks. Non-destructive — the dropped members still
    exist in the store; they're just not surfaced twice in one result set. The
    surviving representative records its canonical link in `near_dup_of`."""
    links = await store.get_near_dup_links([c["id"] for c in pool])
    if not links:
        return pool

    def canonical(cid: str) -> str:
        # follow the link chain to the cluster's canonical id (cycle-guarded)
        seen: set[str] = set()
        while cid in links and cid not in seen:
            seen.add(cid)
            cid = links[cid]
        return cid

    best: dict[str, dict] = {}
    for cand in pool:
        cand["near_dup_of"] = links.get(cand["id"])
        key = canonical(cand["id"])
        current = best.get(key)
        if current is None or cand["retrieval_score"] > current["retrieval_score"]:
            best[key] = cand
    return list(best.values())


async def _sparse_pool_scores(
    store: "Store",
    http: httpx.AsyncClient,
    query: str,
    pool: list[dict],
    settings: "Settings",
) -> list[float]:
    """Per-candidate learned-sparse score in [0, 1], or all-zeros.

    Embeds the query's BGE-M3 sparse vector and dots it against each candidate's
    stored sparse weights (fetched in one batched store read over the pool). Any
    missing piece — feature off, no sparse endpoint, weights never ingested —
    yields zeros, so the dense pipeline is unaffected (cf. HyDE / reranker
    fallbacks).
    """
    if not settings.sparse_enabled:
        return [0.0] * len(pool)
    query_sparse = await embed_sparse_text(http, query)
    if not query_sparse:
        return [0.0] * len(pool)
    weights = await store.get_sparse_weights([c["id"] for c in pool])
    if not weights:
        return [0.0] * len(pool)
    return sparse_scores(query_sparse, [weights.get(c["id"]) for c in pool])


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
    # instruction-tuned embedders want a task instruction on the query side
    # (empty by default — a no-op for BGE-M3); the HyDE hypothetical is
    # answer/passage-shaped, so it takes the passage-side instruction instead
    q_instr = settings.query_instruction
    use_hyde = (
        settings.hyde_enabled
        and len(query.split()) <= settings.hyde_max_query_words
    )
    if use_hyde:
        hypothetical = await generate_hypothetical_answer(http, query)
        if hypothetical:
            query_emb, hypo_emb = await embed_texts(
                http, [q_instr + query, settings.passage_instruction + hypothetical]
            )
            w = settings.hyde_query_weight
            blended = w * np.asarray(query_emb) + (1.0 - w) * np.asarray(hypo_emb)
            norm = np.linalg.norm(blended)
            if norm > 0:
                return (blended / norm).tolist()
    return await embed_text(http, q_instr + query)
