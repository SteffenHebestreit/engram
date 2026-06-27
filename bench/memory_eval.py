"""Agent-memory write-path eval — does recorded feedback make retrieval IMPROVE?

This tests engram's one structurally-unique capability: a *stateful* retriever
that learns from which chunks an agent actually used. A stateless RAG pipeline
(dense+rerank, hybrid+rerank) cannot do this — it has no memory of past usage.

Protocol (per dataset, real production models, engramdb backend):
  1. Ingest the BEIR corpus.
  2. Split the test queries into HISTORY (a past agent session) and TEST (held out).
  3. COLD: run each TEST query with the memory boost OFF → baseline nDCG@10/recall.
  4. Record HISTORY feedback: for each history query, the gold chunks it "used",
     stored WITH the history query's embedding (the learning signal).
  5. WARM: run each TEST query with the memory boost ON. A test query benefits when
     a *similar* past (history) query used a chunk that is also relevant here —
     the realistic "the agent has asked related things before" case.
  6. Compare WARM vs COLD per TEST query (paired sign test + bootstrap 95%CI),
     over ALL test queries AND over the "memory-applicable" subset (test queries
     that actually have a ≥min_sim history neighbour sharing a gold doc).

Run (GPU, engramdb):
  docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm \
    -e STORE_BACKEND=engramdb -e EMBEDDING_DIM=1024 \
    -e BENCH_EMBED_MODEL=BAAI/bge-m3 -e EMBEDDING_MODEL=BAAI/bge-m3 \
    --no-deps runner python -m bench.memory_eval
"""

import asyncio
import json
import math
import os
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA_DIR = Path("/data")
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATASETS = os.environ.get("BENCH_MEMORY_DATASETS", "nfcorpus,scifact").split(",")
HISTORY_FRAC = float(os.environ.get("BENCH_MEMORY_HISTORY_FRAC", "0.6"))
MIN_SIM = float(os.environ.get("BENCH_MEMORY_MIN_SIM", "0.7"))
TOP_K = 100


def ensure_dataset(name: str) -> Path:
    d = DATA_DIR / name
    if (d / "corpus.jsonl").exists():
        return d
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / f"{name}.zip"
    print(f"downloading {BEIR_URL.format(name=name)} ...", flush=True)
    urllib.request.urlretrieve(BEIR_URL.format(name=name), zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    return d


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            qid, did, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, qrel, k):
    idcg = _dcg(sorted(qrel.values(), reverse=True)[:k])
    return _dcg([qrel.get(d, 0) for d in ranked[:k]]) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, qrel, k):
    rel = {d for d, s in qrel.items() if s > 0}
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def paired(deltas, seed=0):
    """mean delta + bootstrap 95%CI + win/tie/loss + two-sided sign-test p."""
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    nd = wins + losses
    p = 1.0 if nd == 0 else math.erfc((abs(wins - losses) / math.sqrt(nd)) / math.sqrt(2))
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, len(arr), size=(1000, len(arr)))].mean(axis=1)
    return {"mean": float(arr.mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "wins": wins,
            "ties": len(deltas) - wins - losses, "losses": losses, "p": p}


async def main():
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embed_model = os.environ.get("BENCH_EMBED_MODEL", "BAAI/bge-m3")
    rerank_model = os.environ.get("BENCH_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    print(f"loading models: {embed_model} + {rerank_model} ...", flush=True)
    embedder = SentenceTransformer(embed_model)
    embedder.max_seq_length = 512
    embedder.default_prompt_name = None
    reranker = CrossEncoder(rerank_model, max_length=512)
    print(
        f"CONFIG backend={os.environ.get('STORE_BACKEND','?')} embed={embed_model} "
        f"rerank={rerank_model} history_frac={HISTORY_FRAC} min_sim={MIN_SIM}",
        flush=True,
    )

    _cache: dict[str, list[float]] = {}

    async def embed_texts(client, texts):
        texts = list(texts)
        miss = [t for t in texts if t not in _cache]
        if miss:
            for t, e in zip(miss, embedder.encode(miss, normalize_embeddings=True)):
                _cache[t] = e.tolist()
        return [_cache[t] for t in texts]

    async def embed_text(client, text):
        return (await embed_texts(client, [text]))[0]

    async def local_rerank(client, query, texts):
        return [float(s) for s in reranker.predict([(query, t) for t in texts])] if texts else []

    import app.embeddings as emb_mod
    import app.ingest as ingest_mod
    import app.search as search_mod
    from app.config import get_settings
    from app.ingest import ingest_document
    from app.rerank import RERANKERS
    from app.search import search
    from app.store import create_store

    emb_mod.embed_texts = ingest_mod.embed_texts = search_mod.embed_texts = embed_texts
    emb_mod.embed_text = search_mod.embed_text = embed_text
    RERANKERS.register("local", local_rerank)

    store = create_store(get_settings())
    await store.connect()
    await store.init_schema()
    doc_chunks = getattr(store, "_doc_chunks", None)
    if doc_chunks is None:
        raise SystemExit("memory_eval requires the engramdb backend (STORE_BACKEND=engramdb)")

    cold_t = {"memory_boost_enabled": False, "hyde_enabled": False}
    warm_t = {"memory_boost_enabled": True, "memory_boost_min_sim": MIN_SIM,
              "memory_boost_max_neighbors": 50, "memory_boost_weight": 0.5,
              "hyde_enabled": False}

    for name in DATASETS:
        print(f"\n########## dataset: {name} ##########", flush=True)
        d = ensure_dataset(name)
        corpus = {o["_id"]: o for o in load_jsonl(d / "corpus.jsonl")}
        queries = {o["_id"]: o["text"] for o in load_jsonl(d / "queries.jsonl")}
        qrels = load_qrels(d / "qrels" / "test.tsv")
        qids = sorted(q for q in qrels if q in queries)
        doc_ids = list(corpus)

        # wipe + ingest
        for doc in await store.list_documents():
            await store.delete_document(doc["id"])
        t0 = time.perf_counter()
        for did in doc_ids:
            text = f"{corpus[did].get('title','')}\n{corpus[did].get('text','')}".strip()
            if text:
                await ingest_document(store, None, text, title=corpus[did].get("title", ""),
                                      source="beir", document_id=did)
        print(f"ingested {len(doc_ids)} docs in {time.perf_counter()-t0:.0f}s", flush=True)

        # deterministic history/test split
        cut = int(len(qids) * HISTORY_FRAC)
        history, test = qids[:cut], qids[cut:]
        print(f"queries: {len(history)} history / {len(test)} test", flush=True)

        def gold_chunks(qid):
            ids = []
            for did, s in qrels[qid].items():
                if s > 0:
                    ids.extend(doc_chunks.get(did, []))
            return ids

        async def run(qid, tuning):
            hits = await search(store, None, queries[qid], top_k=TOP_K, tuning=tuning)
            scored = {}
            for r in hits:
                if r.document_id not in scored or r.rerank_score > scored[r.document_id]:
                    scored[r.document_id] = r.rerank_score
            return sorted(scored, key=scored.get, reverse=True)

        # 1) COLD (memory off) on test
        cold = {}
        for i, qid in enumerate(test):
            cold[qid] = await run(qid, cold_t)
            if i % 50 == 0:
                print(f"  cold {i}/{len(test)}", flush=True)

        # 2) record history feedback (gold chunks + history query embedding)
        hist_embs = {qid: np.asarray(await embed_text(None, queries[qid]), dtype=np.float32)
                     for qid in history}
        recorded = 0
        for qid in history:
            gc = gold_chunks(qid)
            if gc:
                recorded += await store.record_feedback(
                    queries[qid], gc, qid, query_embedding=hist_embs[qid].tolist())
        print(f"recorded {recorded} feedback links from history", flush=True)

        # 3) WARM (memory on) on test
        warm = {}
        for i, qid in enumerate(test):
            warm[qid] = await run(qid, warm_t)
            if i % 50 == 0:
                print(f"  warm {i}/{len(test)}", flush=True)

        # memory-applicable subset: a test query with a >=MIN_SIM history neighbour
        # that shares a gold doc (the only queries memory CAN help)
        hist_mat = np.stack([hist_embs[q] / (np.linalg.norm(hist_embs[q]) or 1) for q in history])
        hist_gold = {q: {dd for dd, s in qrels[q].items() if s > 0} for q in history}
        applicable = set()
        for qid in test:
            qv = np.asarray(_cache[queries[qid]], dtype=np.float32)
            qv = qv / (np.linalg.norm(qv) or 1)
            sims = hist_mat @ qv
            tg = {dd for dd, s in qrels[qid].items() if s > 0}
            for j, hq in enumerate(history):
                if sims[j] >= MIN_SIM and (hist_gold[hq] & tg):
                    applicable.add(qid)
                    break

        # 4) score
        def report(subset, label):
            if not subset:
                print(f"  [{label}] (empty)", flush=True)
                return
            dn = [ndcg_at_k(warm[q], qrels[q], 10) - ndcg_at_k(cold[q], qrels[q], 10) for q in subset]
            dr = [recall_at_k(warm[q], qrels[q], 10) - recall_at_k(cold[q], qrels[q], 10) for q in subset]
            cn = sum(ndcg_at_k(cold[q], qrels[q], 10) for q in subset) / len(subset)
            wn = sum(ndcg_at_k(warm[q], qrels[q], 10) for q in subset) / len(subset)
            pn, pr = paired(dn), paired(dr)
            sig = "n.s." if pn["lo"] <= 0 <= pn["hi"] else "SIGNIFICANT"
            print(f"  [{label}] n={len(subset)}  cold nDCG@10={cn:.4f} → warm={wn:.4f}", flush=True)
            print(f"      ΔnDCG@10={pn['mean']:+.4f} 95%CI[{pn['lo']:+.4f},{pn['hi']:+.4f}] "
                  f"win/tie/loss={pn['wins']}/{pn['ties']}/{pn['losses']} sign-p={pn['p']:.4f} ({sig})", flush=True)
            print(f"      ΔRecall@10={pr['mean']:+.4f} 95%CI[{pr['lo']:+.4f},{pr['hi']:+.4f}] "
                  f"win/tie/loss={pr['wins']}/{pr['ties']}/{pr['losses']} sign-p={pr['p']:.4f}", flush=True)

        print(f"\n=== {name}: agent-memory WARM vs COLD ({embed_model} + {rerank_model}) ===", flush=True)
        report(test, "all test queries")
        report(sorted(applicable), "memory-applicable subset")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
