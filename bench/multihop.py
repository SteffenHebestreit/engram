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


def paired_recall(runs_a, runs_b, questions, k=5, seed=0):
    """Paired engram-vs-baseline on per-question Recall@k: mean delta + bootstrap
    95% CI + win/tie/loss + two-sided sign-test p (so a multi-hop 'lift' is tested,
    not just eyeballed — same rigor as the BEIR harness)."""
    import math as _m

    def ranked(runs, qid):
        return sorted(runs[qid], key=runs[qid].get, reverse=True)

    deltas = [
        recall_at_k(ranked(runs_a, qid), g, k) - recall_at_k(ranked(runs_b, qid), g, k)
        for qid, _, g in questions
    ]
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    nd = wins + losses
    p = 1.0 if nd == 0 else _m.erfc((abs(wins - losses) / _m.sqrt(nd)) / _m.sqrt(2))
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, len(arr), size=(1000, len(arr)))].mean(axis=1)
    return {"mean": float(arr.mean()), "lo": float(np.percentile(boot, 2.5)),
            "hi": float(np.percentile(boot, 97.5)), "wins": wins,
            "ties": len(deltas) - wins - losses, "losses": losses, "p": p}


async def main():
    from datasets import load_dataset
    from sentence_transformers import CrossEncoder, SentenceTransformer

    # dataset switch: HotpotQA (saturated) or MuSiQue (harder, multi-hop-by-design)
    dataset = os.environ.get("BENCH_MULTIHOP_DATASET", "hotpot")
    corpus: dict[str, str] = {}
    questions: list[tuple[str, str, set]] = []
    if dataset == "musique":
        print(f"loading MuSiQue validation (first {N})...", flush=True)
        ds = load_dataset("dgslibisey/MuSiQue", split="validation")
        ds = ds.select(range(min(N, len(ds))))
        for ex in ds:
            gold = set()
            for p in ex["paragraphs"]:
                title = p["title"]
                corpus.setdefault(title, p["paragraph_text"])
                if p.get("is_supporting"):
                    gold.add(title)
            questions.append((ex["id"], ex["question"], gold))
    else:
        print(f"loading HotpotQA distractor dev (first {N})...", flush=True)
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
        ds = ds.select(range(min(N, len(ds))))
        for ex in ds:
            ctx = ex["context"]
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                corpus.setdefault(title, " ".join(sents))
            gold = set(ex["supporting_facts"]["title"])
            questions.append((ex["id"], ex["question"], gold))
    doc_ids = list(corpus)
    doc_texts = [corpus[t] for t in doc_ids]
    print(
        f"dataset={dataset} corpus={len(doc_ids)} passages questions={len(questions)}",
        flush=True,
    )

    embed_model = os.environ.get("BENCH_EMBED_MODEL", "BAAI/bge-m3")
    rerank_model = os.environ.get("BENCH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"loading models: {embed_model} + {rerank_model} ...", flush=True)
    embedder = SentenceTransformer(embed_model)
    embedder.max_seq_length = 512  # cap (BGE-M3 defaults to 8192 — far too slow on CPU)
    reranker = CrossEncoder(rerank_model, max_length=512)
    print(
        "CONFIG "
        f"backend={os.environ.get('STORE_BACKEND','neo4j')} embed={embed_model} "
        f"rerank={rerank_model} embedding_dim={os.environ.get('EMBEDDING_DIM','(default)')} "
        f"quant={os.environ.get('ENGRAMDB_QUANTIZATION','(default)')} "
        f"graph_proximity={os.environ.get('GRAPH_PROXIMITY_MODE','(default)')} "
        f"rerank_depth={RERANK_DEPTH} top_k={TOP_K} dataset={dataset} n={N}",
        flush=True,
    )

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

    # ── hybrid (dense + BM25 via RRF) + rerank — isolates fusion from the graph ──
    runs["hybrid+rerank"] = {}
    for idx, (qid, q, _) in enumerate(questions):
        sims = doc_matrix @ q_matrix[idx]
        dense_rank = np.argsort(sims)[::-1][:200]
        bm_rank = np.argsort(bm25.get_scores(tokenize(q)))[::-1][:200]
        rrf: dict[int, float] = {}
        for r, i in enumerate(dense_rank):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + r)
        for r, i in enumerate(bm_rank):
            rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + r)
        cand = sorted(rrf, key=rrf.get, reverse=True)[:RERANK_DEPTH]
        ce = cross_encode(q, [doc_texts[i] for i in cand])
        runs["hybrid+rerank"][qid] = {doc_ids[i]: sc for i, sc in zip(cand, ce)}
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
    # start clean (the shared bench db may carry data from a prior run). Neo4j is
    # wiped via the driver; other backends (pgvector) are run on a fresh DB
    # instead, so this is a neo4j-only convenience, guarded for backend-agnostic use.
    if getattr(store, "_driver", None) is not None:
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
    systems = ["bm25", "dense", "dense+rerank", "hybrid+rerank", "engram"]
    systems = [s for s in systems if s in runs]
    metrics = ["Recall@2", "Recall@5", "Recall@10"]
    scores = {sys: score_system(runs[sys], questions) for sys in systems}
    print(f"\n=== {dataset} multi-hop retrieval ({embed_model} + {rerank_model}, n={len(questions)}) ===")
    print(f"{'system':<16}" + "".join(f"{m:>12}" for m in metrics))
    for sys in systems:
        print(f"{sys:<16}" + "".join(f"{scores[sys][m]:>12.4f}" for m in metrics))
    # paired significance: does engram's GRAPH pipeline beat the rerank baselines on
    # multi-hop Recall@5? This is the decisive test of whether the architecture adds
    # measurable quality where single-hop BEIR cannot exercise it.
    for base in ("dense+rerank", "hybrid+rerank"):
        if "engram" in runs and base in runs:
            d = paired_recall(runs["engram"], runs[base], questions, k=5)
            sig = "n.s." if d["lo"] <= 0 <= d["hi"] else "SIGNIFICANT"
            print(
                f"  paired engram−{base}: ΔRecall@5={d['mean']:+.4f} "
                f"95%CI[{d['lo']:+.4f},{d['hi']:+.4f}] "
                f"win/tie/loss={d['wins']}/{d['ties']}/{d['losses']} "
                f"sign-p={d['p']:.4f} ({sig})"
            )


if __name__ == "__main__":
    asyncio.run(main())
