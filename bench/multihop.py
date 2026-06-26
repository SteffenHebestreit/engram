"""Multi-hop retrieval benchmark: engram vs standard RAG strategies on HotpotQA.

This is the turf graph-RAG systems (HippoRAG, GraphRAG) compete on: multi-hop
questions whose answer spans two supporting passages that must be retrieved
together. The metric is recall@2/@5 of the supporting passages — exactly what
HippoRAG reports.

Unlike the BEIR run, engram here uses its FULL graph pipeline:
  * YAKE keyword extraction (no LLM) -> shared-keyword graph linking passages
  * keyword-sibling graph expansion + GDS personalized-PageRank proximity
so a passage retrieved for one hop can pull in its bridge-linked partner.

Systems (all over MiniLM embeddings / ms-marco cross-encoder, same corpus):
  bm25 / dense (naive vector RAG) / dense+rerank (2-stage) / engram (graph).

HotpotQA distractor dev, first BENCH_MULTIHOP_N (default 500) questions.
"""

import asyncio
import os
import re
import time

import numpy as np

N = int(os.environ.get("BENCH_MULTIHOP_N", "500"))
RERANK_DEPTH = int(os.environ.get("BENCH_RERANK_DEPTH", "50"))
TOP_K = 50

_TEXT2EMB: dict[str, list[float]] = {}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def recall_at_k(ranked, gold, k):
    return len(set(ranked[:k]) & gold) / len(gold) if gold else 0.0


def score_system(runs, questions):
    def avg(fn):
        return sum(fn(qid, gold) for qid, _, gold in questions) / len(questions)

    def ranked(qid):
        return sorted(runs[qid], key=runs[qid].get, reverse=True)

    return {
        "Recall@2": avg(lambda qid, g: recall_at_k(ranked(qid), g, 2)),
        "Recall@5": avg(lambda qid, g: recall_at_k(ranked(qid), g, 5)),
        "Recall@10": avg(lambda qid, g: recall_at_k(ranked(qid), g, 10)),
    }


async def main():
    from datasets import load_dataset
    from sentence_transformers import CrossEncoder, SentenceTransformer

    print(f"loading HotpotQA distractor dev (first {N})...", flush=True)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ds = ds.select(range(min(N, len(ds))))

    corpus: dict[str, str] = {}
    questions: list[tuple[str, str, set]] = []
    for ex in ds:
        ctx = ex["context"]
        for title, sents in zip(ctx["title"], ctx["sentences"]):
            corpus.setdefault(title, " ".join(sents))
        gold = set(ex["supporting_facts"]["title"])
        questions.append((ex["id"], ex["question"], gold))
    doc_ids = list(corpus)
    doc_texts = [corpus[t] for t in doc_ids]
    print(f"corpus={len(doc_ids)} passages, questions={len(questions)}", flush=True)

    embed_model = os.environ.get("BENCH_EMBED_MODEL", "BAAI/bge-m3")
    rerank_model = os.environ.get("BENCH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"loading models: {embed_model} + {rerank_model} ...", flush=True)
    embedder = SentenceTransformer(embed_model)
    embedder.max_seq_length = 512  # cap (BGE-M3 defaults to 8192 — far too slow on CPU)
    reranker = CrossEncoder(rerank_model, max_length=512)

    def cross_encode(query, texts):
        return [float(s) for s in reranker.predict([(query, t) for t in texts])] if texts else []

    # encode corpus + questions once (also primes the engram cache)
    t0 = time.perf_counter()
    doc_matrix = np.asarray(
        embedder.encode(doc_texts, normalize_embeddings=True, batch_size=64), dtype=np.float32
    )
    q_texts = [q for _, q, _ in questions]
    q_matrix = np.asarray(
        embedder.encode(q_texts, normalize_embeddings=True, batch_size=64), dtype=np.float32
    )
    for t, e in zip(doc_texts, doc_matrix):
        _TEXT2EMB[t] = e.tolist()
    print(f"encoded in {time.perf_counter() - t0:.0f}s", flush=True)

    runs: dict[str, dict[str, dict[str, float]]] = {}

    # ── bm25 ──
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(t) for t in doc_texts])
    runs["bm25"] = {}
    for qid, q, _ in questions:
        s = bm25.get_scores(tokenize(q))
        top = np.argsort(s)[::-1][:TOP_K]
        runs["bm25"][qid] = {doc_ids[i]: float(s[i]) for i in top}

    # ── dense (naive vector RAG) ──
    runs["dense"] = {}
    for idx, (qid, q, _) in enumerate(questions):
        sims = doc_matrix @ q_matrix[idx]
        top = np.argsort(sims)[::-1][:TOP_K]
        runs["dense"][qid] = {doc_ids[i]: float(sims[i]) for i in top}

    # ── dense + cross-encoder rerank (2-stage) ──
    runs["dense+rerank"] = {}
    for idx, (qid, q, _) in enumerate(questions):
        sims = doc_matrix @ q_matrix[idx]
        cand = np.argsort(sims)[::-1][:RERANK_DEPTH]
        ce = cross_encode(q, [doc_texts[i] for i in cand])
        runs["dense+rerank"][qid] = {doc_ids[i]: sc for i, sc in zip(cand, ce)}
    print("baselines done", flush=True)

    # ── engram (full graph pipeline) ──
    async def embed_texts(client, texts):
        texts = list(texts)
        if not texts:
            return []
        missing = [t for t in texts if t not in _TEXT2EMB]
        if missing:
            for t, e in zip(missing, embedder.encode(missing, normalize_embeddings=True)):
                _TEXT2EMB[t] = e.tolist()
        return [_TEXT2EMB[t] for t in texts]

    async def embed_text(client, text):
        return (await embed_texts(client, [text]))[0]

    async def local_rerank(client, query, texts):
        return cross_encode(query, list(texts))

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
    # start clean (the shared bench db may carry data from a prior run)
    async with store._driver.session() as session:
        await session.run(
            "MATCH (n) WHERE n:Chunk OR n:Document OR n:Keyword OR n:Community "
            "DETACH DELETE n"
        )

    t0 = time.perf_counter()
    for i, did in enumerate(doc_ids):
        await ingest_document(store, None, doc_texts[i], title=did, source="hotpot", document_id=did)
        if i % 2000 == 0:
            print(f"  engram ingested {i}/{len(doc_ids)}", flush=True)
    print(f"engram ingest in {time.perf_counter() - t0:.0f}s", flush=True)

    t0 = time.perf_counter()
    runs["engram"] = {}
    for j, (qid, q, _) in enumerate(questions):
        hits = await search(store, None, q, top_k=TOP_K)
        scored: dict[str, float] = {}
        for r in hits:
            if r.document_id not in scored or r.rerank_score > scored[r.document_id]:
                scored[r.document_id] = r.rerank_score
        runs["engram"][qid] = scored
        if j % 100 == 0:
            print(f"  engram queried {j}/{len(questions)}", flush=True)
    print(f"engram query in {time.perf_counter() - t0:.0f}s", flush=True)
    await store.close()

    # ── report ──
    systems = ["bm25", "dense", "dense+rerank", "engram"]
    metrics = ["Recall@2", "Recall@5", "Recall@10"]
    scores = {sys: score_system(runs[sys], questions) for sys in systems}
    print(f"\n=== HotpotQA multi-hop retrieval ({embed_model} + {rerank_model}) ===")
    print(f"{'system':<14}" + "".join(f"{m:>12}" for m in metrics))
    for sys in systems:
        print(f"{sys:<14}" + "".join(f"{scores[sys][m]:>12.4f}" for m in metrics))


if __name__ == "__main__":
    asyncio.run(main())
