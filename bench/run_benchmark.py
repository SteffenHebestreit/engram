"""Retrieval benchmark: run engram's real pipeline over BEIR SciFact.

There are no live model endpoints in this environment, so we drive engram with
a local embedding model (all-MiniLM-L6-v2) and a local cross-encoder reranker
(ms-marco-MiniLM-L-6-v2), wired straight into engram's embedding + reranker
seams. The full pipeline otherwise runs unchanged (DBSF fusion of the dense +
fulltext channels, median-proximity, MMR shortlist, cross-encoder rerank).
Metadata extraction + HyDE are off (no generative LLM available), so the
summary/keyword channels and keyword-graph expansion are inactive — SciFact
docs are single-chunk abstracts anyway.

Reports nDCG@10, Recall@10/100, MAP, P@10 — the standard SciFact metrics.
"""

import asyncio
import json
import os
import time
import urllib.request
import zipfile
from pathlib import Path

DATA_DIR = Path("/data")
# optional cap on the number of test queries evaluated (0 = all). Sampling keeps
# CPU runs tractable; the mean nDCG/recall is a fine estimate over ~100 queries.
MAX_QUERIES = int(os.environ.get("BENCH_MAX_QUERIES", "0"))
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


# ── dataset ──────────────────────────────────────────────────────────────────
def ensure_dataset() -> Path:
    d = DATA_DIR / "scifact"
    if (d / "corpus.jsonl").exists():
        return d
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "scifact.zip"
    print(f"downloading {SCIFACT_URL} ...", flush=True)
    urllib.request.urlretrieve(SCIFACT_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    return d


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with open(path, encoding="utf-8") as f:
        next(f)  # header: query-id  corpus-id  score
        for line in f:
            qid, did, score = line.strip().split("\t")
            qrels.setdefault(qid, {})[did] = int(score)
    return qrels


# ── pure-Python IR metrics (binary relevance) ────────────────────────────────
import math


def _dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    gains = [qrel.get(d, 0) for d in ranked[:k]]
    ideal = sorted(qrel.values(), reverse=True)[:k]
    idcg = _dcg(ideal)
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    rel = {d for d, s in qrel.items() if s > 0}
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def precision_at_k(ranked: list[str], qrel: dict[str, int], k: int) -> float:
    rel = {d for d, s in qrel.items() if s > 0}
    return len(set(ranked[:k]) & rel) / k


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


# ── local models wired into engram's seams ───────────────────────────────────
def install_local_models():
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print("loading models (all-MiniLM-L6-v2 + ms-marco cross-encoder)...", flush=True)
    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    async def embed_texts(client, texts):
        if not texts:
            return []
        return embedder.encode(list(texts), normalize_embeddings=True).tolist()

    async def embed_text(client, text):
        return embedder.encode([text], normalize_embeddings=True)[0].tolist()

    async def local_rerank(client, query, texts):
        if not texts:
            return []
        scores = reranker.predict([(query, t) for t in texts])
        return [float(s) for s in scores]

    import app.embeddings as emb_mod
    import app.ingest as ingest_mod
    import app.search as search_mod
    from app.rerank import RERANKERS

    emb_mod.embed_texts = embed_texts
    emb_mod.embed_text = embed_text
    ingest_mod.embed_texts = embed_texts
    search_mod.embed_text = embed_text
    search_mod.embed_texts = embed_texts
    RERANKERS.register("local", local_rerank)


async def main():
    d = ensure_dataset()
    corpus = {o["_id"]: o for o in load_jsonl(d / "corpus.jsonl")}
    queries = {o["_id"]: o["text"] for o in load_jsonl(d / "queries.jsonl")}
    qrels = load_qrels(d / "qrels" / "test.tsv")
    test_qids = [q for q in qrels if q in queries]
    if MAX_QUERIES > 0:
        test_qids = test_qids[:MAX_QUERIES]
    print(
        f"corpus={len(corpus)} queries(test)={len(test_qids)} "
        f"backend={os.environ.get('STORE_BACKEND', 'neo4j')}",
        flush=True,
    )

    install_local_models()

    from app.config import get_settings
    from app.ingest import ingest_document
    from app.search import search
    from app.store import create_store

    store = create_store(get_settings())
    await store.connect()
    await store.init_schema()

    # ── ingest ──
    t0 = time.perf_counter()
    for i, (did, doc) in enumerate(corpus.items()):
        text = f"{doc.get('title', '')}\n{doc.get('text', '')}".strip()
        if not text:
            continue
        await ingest_document(
            store, None, text, title=doc.get("title", ""), source="beir", document_id=did
        )
        if i % 1000 == 0:
            print(f"  ingested {i}/{len(corpus)}", flush=True)
    print(f"ingest done in {time.perf_counter() - t0:.0f}s", flush=True)

    # ── search ──
    t0 = time.perf_counter()
    results: dict[str, dict[str, float]] = {}
    for j, qid in enumerate(test_qids):
        hits = await search(store, None, queries[qid], top_k=100)
        scored: dict[str, float] = {}
        for r in hits:
            # one document may yield several chunks; keep its best score
            if r.document_id not in scored or r.rerank_score > scored[r.document_id]:
                scored[r.document_id] = r.rerank_score
        results[qid] = scored
        if j % 50 == 0:
            print(f"  queried {j}/{len(test_qids)}", flush=True)
    print(f"search done in {time.perf_counter() - t0:.0f}s", flush=True)
    await store.close()

    # ── evaluate ──
    def avg(fn) -> float:
        return sum(fn(qid) for qid in test_qids) / len(test_qids)

    def ranked_for(qid: str) -> list[str]:
        return sorted(results[qid], key=results[qid].get, reverse=True)

    metrics = {
        "nDCG@10": avg(lambda q: ndcg_at_k(ranked_for(q), qrels[q], 10)),
        "Recall@10": avg(lambda q: recall_at_k(ranked_for(q), qrels[q], 10)),
        "Recall@100": avg(lambda q: recall_at_k(ranked_for(q), qrels[q], 100)),
        "MAP": avg(lambda q: average_precision(ranked_for(q), qrels[q])),
        "P@10": avg(lambda q: precision_at_k(ranked_for(q), qrels[q], 10)),
    }
    print("\n=== engram on BEIR SciFact (dense MiniLM + BM25 fusion + cross-encoder) ===")
    for name, value in metrics.items():
        print(f"  {name:<12} {value:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
