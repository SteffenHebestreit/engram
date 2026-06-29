"""Evaluate the REAL engram pipeline against a LIVE OpenAI-compatible stack.

Unlike `bench/run_benchmark.py` (which monkeypatches *local* sentence-transformers
models into the seams), this drives engram exactly as in production — real HTTP
embedding / reranker / LLM calls to whatever `EMBEDDING_API_BASE` etc. point at
(LM Studio, vLLM, TEI, ...) — over an in-process `engramdb` store, so no database
to deploy. Use it to pick models on YOUR hardware with real BEIR nDCG.

Everything is read from engram's Settings (env), so you compare models by changing
env between runs and holding the rest fixed. Ingest + query run concurrently
(`BENCH_CONCURRENCY`) since each chunk is its own HTTP round trip.

  STORE_BACKEND=engramdb SCHEMA_GUARD_MODE=off \
  EMBEDDING_API_BASE=http://host:1234/v1 EMBEDDING_MODEL=... EMBEDDING_DIM=1024 \
  RERANKER_ENABLED=false HYDE_ENABLED=false SPARSE_ENABLED=false \
  METADATA_EXTRACTOR=none SUMMARY_CHANNEL_ENABLED=false KEYWORDS_CHANNEL_ENABLED=false \
  BENCH_DATA_DIR=e:/tmp/beir BENCH_DATASET=scifact \
  python -m bench.live_eval

Reports nDCG@10, Recall@10/100, MAP, P@10 — the standard SciFact metrics.
"""

import asyncio
import json
import os
import time
from pathlib import Path

import httpx

import bench.run_benchmark as rb
from app.config import get_settings
from app.ingest import ingest_document
from app.search import search
from app.store import create_store
from bench.compare import paired_delta  # bootstrap-CI + sign-test, the repo's bar

CONCURRENCY = int(os.environ.get("BENCH_CONCURRENCY", "12"))
MAX_QUERIES = int(os.environ.get("BENCH_MAX_QUERIES", "0"))


def _maybe_register_local_reranker() -> None:
    """Optionally load a sentence-transformers CrossEncoder reranker IN-PROCESS
    (`BENCH_LOCAL_RERANKER=<hf id>`, e.g. BAAI/bge-reranker-v2-m3 or
    Qwen/Qwen3-Reranker-0.6B) and register it as RERANKERS['local'], so the FULL
    pipeline can be measured WITH a reranker even when the model server can't serve
    one (LM Studio). Then run with RERANKER_ENABLED=true RERANKER_STRATEGY=local.
    No-op when unset or when torch/sentence-transformers aren't installed — so this
    must run on a box with the ML stack (and ideally a GPU)."""
    name = os.environ.get("BENCH_LOCAL_RERANKER")
    if not name:
        return
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        print(f"  BENCH_LOCAL_RERANKER={name} set but sentence-transformers is "
              "missing — skipping (run this on a box with torch+sentence-transformers).",
              flush=True)
        return
    ce = CrossEncoder(name, max_length=int(os.environ.get("BENCH_RERANK_MAX_LEN", "512")))
    from app.rerank import RERANKERS

    async def _local_rerank(client, query, texts):
        if not texts:
            return []
        return [float(s) for s in ce.predict([(query, t) for t in texts])]

    RERANKERS.register("local", _local_rerank)
    print(f"  local reranker loaded: {name}", flush=True)


async def _retry(make_coro, attempts: int = 5, base: float = 1.0):
    """Await a coroutine with backoff — a single transient 5xx from the model
    server must not lose a multi-minute run (or silently drop a corpus doc)."""
    for i in range(attempts):
        try:
            return await make_coro()
        except Exception:
            if i == attempts - 1:
                raise
            await asyncio.sleep(base * (i + 1))


async def main() -> None:
    rb.DATA_DIR = Path(os.environ.get("BENCH_DATA_DIR", "/data"))
    rb.DATASET = os.environ.get("BENCH_DATASET", "scifact")
    d = rb.ensure_dataset()
    corpus = {o["_id"]: o for o in rb.load_jsonl(d / "corpus.jsonl")}
    queries = {o["_id"]: o["text"] for o in rb.load_jsonl(d / "queries.jsonl")}
    qrels = rb.load_qrels(d / "qrels" / "test.tsv")
    test_qids = [q for q in qrels if q in queries]
    if MAX_QUERIES > 0:
        test_qids = test_qids[:MAX_QUERIES]

    s = get_settings()
    print(
        f"dataset={rb.DATASET} corpus={len(corpus)} queries={len(test_qids)}\n"
        f"  embedder={s.embedding_model} dim={s.embedding_dim} "
        f"query_instruction={'set' if s.query_instruction else 'none'}\n"
        f"  reranker={'on:' + s.reranker_model if s.reranker_enabled else 'OFF'} "
        f"hyde={s.hyde_enabled} sparse={s.sparse_enabled} extractor={s.metadata_extractor}\n"
        f"  channels={'content-only' if not (s.summary_channel_enabled or s.keywords_channel_enabled) else 'multi'} "
        f"backend={s.store_backend} concurrency={CONCURRENCY}",
        flush=True,
    )

    _maybe_register_local_reranker()

    store = create_store(s)
    await store.connect()
    await store.init_schema()

    limits = httpx.Limits(max_connections=max(16, CONCURRENCY * 2))
    async with httpx.AsyncClient(limits=limits) as http:
        # ── ingest (concurrent: each doc is its own embedding round trip) ──
        sem = asyncio.Semaphore(CONCURRENCY)
        done = 0

        async def ingest_one(did: str, doc: dict):
            nonlocal done
            text = f"{doc.get('title', '')}\n{doc.get('text', '')}".strip()
            if not text:
                return
            async with sem:
                await _retry(lambda: ingest_document(
                    store, http, text, title=doc.get("title", ""),
                    source="beir", document_id=did))
            done += 1
            if done % 500 == 0:
                print(f"  ingested {done}/{len(corpus)}", flush=True)

        t0 = time.perf_counter()
        await asyncio.gather(*(ingest_one(did, doc) for did, doc in corpus.items()))
        ingest_s = time.perf_counter() - t0
        print(f"ingest done in {ingest_s:.0f}s ({len(corpus) / ingest_s:.1f} docs/s)", flush=True)

        # ── search (concurrent) ──
        qsem = asyncio.Semaphore(CONCURRENCY)
        results: dict[str, dict[str, float]] = {}

        async def query_one(qid: str):
            async with qsem:
                hits = await _retry(lambda: search(store, http, queries[qid], top_k=100))
            scored: dict[str, float] = {}
            for rank, r in enumerate(hits):
                # reranker may be off -> fall back to descending rank as the score
                sc = r.rerank_score if r.rerank_score else (1.0 / (rank + 1))
                if r.document_id not in scored or sc > scored[r.document_id]:
                    scored[r.document_id] = sc
            results[qid] = scored

        t0 = time.perf_counter()
        await asyncio.gather(*(query_one(qid) for qid in test_qids))
        print(f"search done in {time.perf_counter() - t0:.0f}s", flush=True)

    await store.close()

    # ── evaluate ──
    def avg(fn) -> float:
        return sum(fn(qid) for qid in test_qids) / len(test_qids)

    def ranked_for(qid: str) -> list[str]:
        return sorted(results[qid], key=results[qid].get, reverse=True)

    metrics = {
        "nDCG@10": avg(lambda q: rb.ndcg_at_k(ranked_for(q), qrels[q], 10)),
        "Recall@10": avg(lambda q: rb.recall_at_k(ranked_for(q), qrels[q], 10)),
        "Recall@100": avg(lambda q: rb.recall_at_k(ranked_for(q), qrels[q], 100)),
        "MAP": avg(lambda q: rb.average_precision(ranked_for(q), qrels[q])),
        "P@10": avg(lambda q: rb.precision_at_k(ranked_for(q), qrels[q], 10)),
    }
    print(f"\n=== engram on BEIR {rb.DATASET} :: {s.embedding_model} "
          f"(reranker {'on:' + s.reranker_model if s.reranker_enabled else 'OFF'}) ===")
    for name, value in metrics.items():
        print(f"  {name:<12} {value:.4f}")

    # per-query results (for a later paired comparison) + an optional paired test
    # against a previous run, using the repo's bootstrap-CI + sign-test bar.
    label = f"{s.embedding_model}|rerank={s.reranker_model if s.reranker_enabled else 'off'}|" \
            f"channels={'content' if not (s.summary_channel_enabled or s.keywords_channel_enabled) else 'multi'}"
    out_path = os.environ.get("BENCH_OUT")
    if out_path:
        Path(out_path).write_text(json.dumps({
            "label": label, "embedder": s.embedding_model, "metrics": metrics,
            "results": results, "qids": test_qids,
        }), encoding="utf-8")
        print(f"wrote per-query results to {out_path}", flush=True)

    compare_path = os.environ.get("BENCH_COMPARE_TO")
    if compare_path and Path(compare_path).exists():
        prev = json.loads(Path(compare_path).read_text(encoding="utf-8"))
        common = [q for q in test_qids if q in prev["results"]]
        pd = paired_delta(results, prev["results"], qrels, common)
        decisive = (pd["lo"] > 0 or pd["hi"] < 0) and pd["p"] < 0.05
        print(f"\n=== paired nDCG@10: this vs {prev.get('label', 'prev')} (n={len(common)}) ===")
        print(f"  mean delta {pd['mean']:+.4f}  95% CI [{pd['lo']:+.4f}, {pd['hi']:+.4f}]  "
              f"win/tie/loss {pd['wins']}/{pd['ties']}/{pd['losses']}  sign-test p={pd['p']:.4f}")
        print(f"  -> {'SIGNIFICANT (CI excludes 0 and p<0.05)' if decisive else 'n.s. — an honest TIE by the repo bar'}")


if __name__ == "__main__":
    asyncio.run(main())
