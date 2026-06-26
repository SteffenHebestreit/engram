"""First-party, judge-free retrieval evaluation + per-stage attribution.

The RAG field reports LLM-judge win-rates — biased, irreproducible, and silent
about *why* a system retrieved what it did. engram scores retrieval against a
golden set (query -> relevant document ids) with standard IR metrics on *your*
corpus, reproducibly (bootstrap confidence intervals, no LLM in the loop), and —
uniquely — **attributes each recovered gold hit to the channel that surfaced
it**, using the per-result `channels` provenance `search()` already returns.

So you can see which part of the pipeline earns its keep on your data, e.g.
"the sparse channel uniquely recovered 11 gold hits the dense vectors missed" —
the kind of falsifiable, per-stage claim no competitor's eval can produce.

Everything here is pure (operates on plain data structures), so it is fully
unit-tested without a store; `run_evaluation()` is the thin adapter that calls
`search()` over a golden set and feeds these functions.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np

# ── IR metrics (binary relevance: a qrel value > 0 means relevant) ────────────
# Formulas match bench/compare.py so harness numbers are directly comparable.


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [qrel.get(d, 0) for d in ranked[:k]]
    idcg = _dcg(sorted(qrel.values(), reverse=True)[:k])
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    rel = {d for d, s in qrel.items() if s > 0}
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def precision_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    rel = {d for d, s in qrel.items() if s > 0}
    return len(set(ranked[:k]) & rel) / k if k else 0.0


def average_precision(ranked: list[str], qrel: dict[str, int]) -> float:
    rel = {d for d, s in qrel.items() if s > 0}
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked):
        if d in rel:
            hits += 1
            total += hits / (i + 1)
    return total / len(rel)


def bootstrap_ci(
    values: list[float], n: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """(mean, lo, hi) percentile bootstrap CI over per-query scores.

    Deterministic for a given seed, so a reported interval is reproducible.
    """
    if not values:
        return (0.0, 0.0, 0.0)
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if len(arr) == 1:
        return (mean, mean, mean)
    rng = np.random.default_rng(seed)
    means = arr[rng.integers(0, len(arr), size=(n, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return (mean, lo, hi)


# ── ranking helpers ───────────────────────────────────────────────────────────


def _get(result: Any, field: str) -> Any:
    """Read a field from a SearchResult-like object or a plain dict."""
    if isinstance(result, dict):
        return result.get(field)
    return getattr(result, field, None)


def ranked_docs(results: list[Any]) -> list[str]:
    """Ordered, de-duplicated document ids from a ranked result list (a document
    can contribute several chunks; keep its best/first rank)."""
    seen: dict[str, None] = {}
    for r in results:
        doc = _get(r, "document_id")
        if doc is not None and doc not in seen:
            seen[doc] = None
    return list(seen)


# ── scoring a whole run ───────────────────────────────────────────────────────


def score_results(
    results_by_query: dict[str, list[Any]],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (10,),
) -> dict[str, Any]:
    """Aggregate nDCG@k / Recall@k / P@k (per k) + MAP over the evaluated
    queries, each with a bootstrap 95% CI. `results_by_query` maps query id to a
    ranked list of SearchResult-like items; documents are scored (chunks rolled
    up to their document)."""
    qids = [q for q in results_by_query if q in qrels]
    ranked = {q: ranked_docs(results_by_query[q]) for q in qids}

    per_query: dict[str, list[float]] = {}
    for k in ks:
        per_query[f"nDCG@{k}"] = [ndcg_at_k(ranked[q], qrels[q], k) for q in qids]
        per_query[f"Recall@{k}"] = [recall_at_k(ranked[q], qrels[q], k) for q in qids]
        per_query[f"P@{k}"] = [precision_at_k(ranked[q], qrels[q], k) for q in qids]
    per_query["MAP"] = [average_precision(ranked[q], qrels[q]) for q in qids]

    metrics = {}
    for name, vals in per_query.items():
        mean, lo, hi = bootstrap_ci(vals)
        metrics[name] = {"mean": mean, "ci95": [lo, hi]}
    return {"n_queries": len(qids), "metrics": metrics}


# ── per-stage attribution (the differentiator) ────────────────────────────────


def attribute_channels(
    results_by_query: dict[str, list[Any]],
    qrels: dict[str, dict[str, int]],
    k: int | None = None,
) -> dict[str, Any]:
    """Attribute each recovered gold document to the channel(s) that surfaced it.

    For every relevant document that appears in the (optionally top-k) results,
    union the `channels` of its matching chunks. Returns, per channel, how many
    gold hits it surfaced and — the key number — how many it surfaced
    **uniquely** (no other channel found them, so they are lost without it).
    """
    by_channel: Counter[str] = Counter()
    unique_to_channel: Counter[str] = Counter()
    gold_retrieved = 0

    for qid, results in results_by_query.items():
        relevant = {d for d, s in qrels.get(qid, {}).items() if s > 0}
        if not relevant:
            continue
        ranked = results[:k] if k is not None else results
        doc_channels: dict[str, set[str]] = {}
        for r in ranked:
            doc = _get(r, "document_id")
            if doc in relevant:
                doc_channels.setdefault(doc, set()).update(_get(r, "channels") or [])
        for chans in doc_channels.values():
            gold_retrieved += 1
            for c in chans:
                by_channel[c] += 1
            if len(chans) == 1:
                unique_to_channel[next(iter(chans))] += 1

    return {
        "gold_hits_retrieved": gold_retrieved,
        "by_channel": dict(by_channel.most_common()),
        "unique_to_channel": dict(unique_to_channel.most_common()),
    }


def evaluate(
    results_by_query: dict[str, list[Any]],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (10,),
    attribution_k: int | None = 10,
) -> dict[str, Any]:
    """Full report: scored metrics (with CIs) + per-channel gold-hit attribution."""
    return {
        **score_results(results_by_query, qrels, ks),
        "attribution": attribute_channels(results_by_query, qrels, attribution_k),
    }


async def run_evaluation(
    store: Any,
    http: Any,
    golden: dict[str, dict[str, int]],
    queries: dict[str, str],
    ks: tuple[int, ...] = (10,),
    top_k: int = 50,
    tuning: dict | None = None,
) -> dict[str, Any]:
    """Run engram `search()` over a golden set and produce the eval report.

    `golden` maps query id -> {document_id: relevance}; `queries` maps query id
    -> query text. Thin adapter around the pure functions above.
    """
    from .search import search

    results_by_query: dict[str, list[Any]] = {}
    for qid in golden:
        if qid not in queries:
            continue
        results_by_query[qid] = await search(
            store, http, queries[qid], top_k=top_k, tuning=tuning
        )
    return evaluate(results_by_query, golden, ks)
