"""Where does a search actually spend its time?

This decides whether a purpose-built **Engram-DB** is worth building: it can only
speed up the parts an engine owns — the **store** queries (vector / fulltext /
graph / batched metadata reads) and the in-process **CPU** stages (fusion /
median / MMR / scoring). It can NOT speed up the **models** (query embedding +
HyDE LLM + the cross-encoder reranker), which are backend-agnostic network/GPU
calls. So the question is simply: what fraction of end-to-end latency is the
store?

It times one real `search()` per query, attributing time to:
  * MODELS   — query embedding, HyDE LLM, sparse-query embedding, reranker
  * STORE    — the store calls, on the CRITICAL PATH: the per-channel vector
               searches + fulltext run concurrently (asyncio.gather), so their
               contribution is max(...), not the sum; the sequential phases
               (sibling expansion, graph proximity, near-dup / sparse / recency
               reads) add on top
  * CPU/other— the residual (fusion, median proximity, MMR, scoring, Python glue)

Modes:
  * default        — real endpoints (honest end-to-end); needs EMBEDDING + RERANKER
  * --fake-models  — random embeddings + identity reranker, so MODELS ≈ 0. Use
                     this to isolate the store and sweep corpus size (the scale
                     question) without a GPU.

Run (from repo root, with the configured STORE_BACKEND reachable):
    python -m bench.profile_latency --docs 500 --queries 50
    python -m bench.profile_latency --docs 50000 --queries 50 --fake-models

The verdict line reports the store's share of end-to-end latency = the best-case
ceiling on what Engram-DB could shave off.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time
from collections import defaultdict

import numpy as np

from app import ingest as ingest_mod
from app import search as search_mod
from app.config import get_settings
from app.llm import ExtractionResult
from app.store import create_store

# the active per-query timing bucket: {label: [durations...]}; wrappers append here
_CURRENT: dict[str, list[float]] | None = None

_VOCAB = (
    "neural retrieval embedding vector graph proximity rerank fusion sparse dense "
    "tenant recency context chunk document keyword median pagerank cosine index "
    "memory agent pipeline latency throughput quantization channel lexical hybrid "
    "transformer attention semantic corpus passage relevance ranking similarity"
).split()


def _doc_text(seed: int, paras: int = 3, sents: int = 6) -> str:
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(paras):
        sentences = []
        for _ in range(sents):
            words = rng.choice(_VOCAB, size=rng.randint(8, 16))
            sentences.append(" ".join(words).capitalize() + ".")
        out.append(" ".join(sentences))
    return "\n\n".join(out)


def _query_text(seed: int) -> str:
    rng = np.random.RandomState(100_000 + seed)
    return " ".join(rng.choice(_VOCAB, size=rng.randint(4, 9)))


def _timed(label: str, fn):
    async def wrapper(*args, **kwargs):
        t = time.perf_counter()
        try:
            return await fn(*args, **kwargs)
        finally:
            if _CURRENT is not None:
                _CURRENT[label].append(time.perf_counter() - t)

    return wrapper


def _instrument_store(store) -> None:
    """Wrap every store read so its wall time lands in the active query bucket."""
    for name in (
        "vector_search", "fulltext_search", "fetch_siblings", "graph_proximity",
        "get_near_dup_links", "get_sparse_weights", "get_chunk_recency",
        "nearest_chunks",
    ):
        if hasattr(store, name):
            setattr(store, name, _timed(f"store.{name}", getattr(store, name)))


def _instrument_models(monkeypatch_fakes: bool) -> None:
    """Time the model calls search() makes (embedding / HyDE / rerank). In fake
    mode, replace them with cheap local stand-ins so MODELS ≈ 0."""
    dim = int(get_settings().embedding_dim)

    if monkeypatch_fakes:
        def _rand_vec() -> list[float]:
            v = np.random.RandomState().normal(size=dim)
            return (v / np.linalg.norm(v)).tolist()

        async def fake_embed_text(client, text):
            return _rand_vec()

        async def fake_embed_texts(client, texts):
            return [_rand_vec() for _ in texts]

        async def fake_hyde(client, query):
            return None  # skip the HyDE LLM round trip

        search_mod.embed_text = fake_embed_text
        search_mod.embed_texts = fake_embed_texts
        search_mod.generate_hypothetical_answer = fake_hyde
        # identity reranker: keep fused order, no network
        import app.rerank as rerank_mod

        async def fake_rerank(client, query, texts):
            return [1.0 / (i + 1) for i in range(len(texts))]

        rerank_mod.RERANKERS._items["http"] = fake_rerank

    # wrap (real or fake) model calls so they show up in the MODELS bucket
    search_mod.embed_text = _timed("model.embed_query", search_mod.embed_text)
    search_mod.embed_texts = _timed("model.embed_texts", search_mod.embed_texts)
    search_mod.generate_hypothetical_answer = _timed(
        "model.hyde_llm", search_mod.generate_hypothetical_answer
    )
    if getattr(search_mod, "embed_sparse_text", None):
        search_mod.embed_sparse_text = _timed(
            "model.embed_sparse", search_mod.embed_sparse_text
        )
    _orig_get_reranker = search_mod.get_reranker
    search_mod.get_reranker = lambda strat: _timed(
        "model.rerank", _orig_get_reranker(strat)
    )


def _bucketize(q: dict[str, list[float]]) -> dict[str, float]:
    """Reduce one query's raw call timings to critical-path buckets (seconds)."""
    def s(label: str) -> float:
        return sum(q.get(label, []))

    # the per-channel vector searches + fulltext are gathered → concurrent, so
    # their critical-path cost is the SLOWEST one, not their sum
    concurrent = q.get("store.vector_search", []) + q.get("store.fulltext_search", [])
    retrieval = max(concurrent) if concurrent else 0.0
    graph = s("store.fetch_siblings") + s("store.graph_proximity")
    batched = (
        s("store.get_near_dup_links")
        + s("store.get_sparse_weights")
        + s("store.get_chunk_recency")
        + s("store.nearest_chunks")
    )
    store = retrieval + graph + batched
    models = (
        s("model.embed_query") + s("model.embed_texts") + s("model.hyde_llm")
        + s("model.embed_sparse") + s("model.rerank")
    )
    return {
        "store.retrieval": retrieval, "store.graph": graph,
        "store.batched": batched, "store": store, "models": models,
    }


def _pct(part: float, whole: float) -> float:
    return 100.0 * part / whole if whole > 0 else 0.0


def _stats(xs: list[float]) -> tuple[float, float, float]:
    xs = sorted(xs)
    p = lambda f: xs[min(len(xs) - 1, int(f * len(xs)))] * 1000.0  # noqa: E731
    return statistics.mean(xs) * 1000.0, p(0.5), p(0.95)


async def main() -> None:
    global _CURRENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=500)
    ap.add_argument("--queries", type=int, default=50)
    ap.add_argument("--fake-models", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    store = create_store(settings)
    await store.connect()
    try:
        await store.verify_connectivity()
    except Exception as exc:
        raise SystemExit(f"store ({settings.store_backend}) unreachable: {exc}")
    await store.init_schema()

    # ── ingest the synthetic corpus ──────────────────────────────────────────
    if args.fake_models:
        d = int(settings.embedding_dim)

        async def fake_embed_texts(client, texts):
            return [
                (lambda v: (v / np.linalg.norm(v)).tolist())(
                    np.random.RandomState().normal(size=d)
                )
                for _ in texts
            ]

        async def fake_extract(client, chunk):
            return ExtractionResult(keywords=chunk.split()[:4], summary=chunk[:80])

        ingest_mod.embed_texts = fake_embed_texts
        ingest_mod.get_extractor = lambda name: fake_extract
        http = None
    else:
        import httpx

        http = httpx.AsyncClient()

    print(
        f"backend={settings.store_backend} docs={args.docs} queries={args.queries} "
        f"fake_models={args.fake_models}"
    )
    print(f"ingesting {args.docs} synthetic documents…")
    t0 = time.perf_counter()
    for i in range(args.docs):
        await ingest_mod.ingest_document(
            store, http, _doc_text(i), source="profile", document_id=f"profile:{i}"
        )
    print(f"  ingest done in {time.perf_counter() - t0:.1f}s")

    _instrument_store(store)
    _instrument_models(args.fake_models)

    # ── timed search runs ────────────────────────────────────────────────────
    per_query: list[dict[str, float]] = []
    end_to_end: list[float] = []
    # warm up (JIT caches, connection pool, OS page cache) — not measured
    _CURRENT = defaultdict(list)
    await search_mod.search(store, http, _query_text(0))

    for i in range(args.queries):
        _CURRENT = defaultdict(list)
        t = time.perf_counter()
        await search_mod.search(store, http, _query_text(i))
        end_to_end.append(time.perf_counter() - t)
        per_query.append(_bucketize(_CURRENT))
    _CURRENT = None

    # ── report ───────────────────────────────────────────────────────────────
    def col(key: str) -> list[float]:
        return [q[key] for q in per_query]

    e_mean, e_p50, e_p95 = _stats(end_to_end)
    s_mean = statistics.mean(col("store")) * 1000.0
    m_mean = statistics.mean(col("models")) * 1000.0
    cpu_mean = max(0.0, e_mean - s_mean - m_mean)

    print("\n=== latency breakdown (mean ms/query unless noted) ===")
    print(f"  end-to-end      {e_mean:8.2f}   (p50 {e_p50:.2f} / p95 {e_p95:.2f})")
    print(f"  MODELS          {m_mean:8.2f}   {_pct(m_mean, e_mean):5.1f}%   "
          f"(embed + HyDE + rerank — Engram-DB CANNOT change)")
    print(f"  STORE           {s_mean:8.2f}   {_pct(s_mean, e_mean):5.1f}%   "
          f"(Engram-DB's target)")
    print(f"    ├─ retrieval  {statistics.mean(col('store.retrieval'))*1000:8.2f}   "
          f"(max of concurrent vector channels + fulltext)")
    print(f"    ├─ graph      {statistics.mean(col('store.graph'))*1000:8.2f}   "
          f"(siblings + proximity/PPR)")
    print(f"    └─ batched    {statistics.mean(col('store.batched'))*1000:8.2f}   "
          f"(near-dup / sparse / recency reads)")
    print(f"  CPU/other       {cpu_mean:8.2f}   {_pct(cpu_mean, e_mean):5.1f}%   "
          f"(fusion / median / MMR / scoring / glue — Engram-DB could fuse some)")

    ceiling = _pct(s_mean, e_mean)
    print(
        f"\nverdict: the store is {ceiling:.1f}% of end-to-end latency → that is the "
        f"best-case ceiling\non what Engram-DB could shave off (it cannot touch the "
        f"{_pct(m_mean, e_mean):.1f}% in models).\n"
        "Sweep --docs to see how the store share grows with corpus size."
    )

    # cleanup
    for i in range(args.docs):
        await store.delete_document(f"profile:{i}")
    await store.close()
    if http is not None:
        await http.aclose()


if __name__ == "__main__":
    asyncio.run(main())
