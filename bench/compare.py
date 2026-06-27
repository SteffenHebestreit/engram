"""Head-to-head retrieval comparison: engram vs standard RAG strategies.

Same datasets, same embedding model, same cross-encoder, same metrics — only
the retrieval *architecture* changes. That isolates what each approach
contributes (running external frameworks with different models/LLMs cannot).

Systems (all over MiniLM embeddings / ms-marco cross-encoder):
  * bm25         — classic lexical retrieval (rank-bm25)
  * dense        — single-vector cosine (naive vector RAG)
  * dense+rerank — dense retrieval then cross-encoder rerank (standard 2-stage)
  * engram       — full pipeline: DBSF fusion of dense + BM25 channels,
                   median-proximity, MMR shortlist, cross-encoder rerank

Datasets: BEIR SciFact + NFCorpus (override via BENCH_DATASETS=scifact,nfcorpus).
Reports nDCG@10, Recall@10, MAP, P@10 per system per dataset.
"""

import asyncio
import json
import math
import os
import re
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

DATA_DIR = Path("/data")
BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
DATASETS = os.environ.get("BENCH_DATASETS", "scifact,nfcorpus").split(",")
RERANK_DEPTH = int(os.environ.get("BENCH_RERANK_DEPTH", "50"))
# optional cap on test queries (0 = all); keeps heavy CPU model runs tractable
MAX_QUERIES = int(os.environ.get("BENCH_MAX_QUERIES", "0"))
TOP_K = 100


# ── dataset ──────────────────────────────────────────────────────────────────
def ensure_dataset(name: str) -> Path:
    d = DATA_DIR / name
    if (d / "corpus.jsonl").exists():
        return d
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = BEIR_URL.format(name=name)
    zip_path = DATA_DIR / f"{name}.zip"
    print(f"downloading {url} ...", flush=True)
    urllib.request.urlretrieve(url, zip_path)
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


# ── metrics (binary relevance) ───────────────────────────────────────────────
def _dcg(gains):
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked, qrel, k):
    gains = [qrel.get(d, 0) for d in ranked[:k]]
    idcg = _dcg(sorted(qrel.values(), reverse=True)[:k])
    return _dcg(gains) / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, qrel, k):
    rel = {d for d, s in qrel.items() if s > 0}
    return len(set(ranked[:k]) & rel) / len(rel) if rel else 0.0


def precision_at_k(ranked, qrel, k):
    rel = {d for d, s in qrel.items() if s > 0}
    return len(set(ranked[:k]) & rel) / k


def average_precision(ranked, qrel):
    rel = {d for d, s in qrel.items() if s > 0}
    if not rel:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(ranked):
        if d in rel:
            hits += 1
            total += hits / (i + 1)
    return total / len(rel)


def per_query_ndcg(runs, qrels, qids):
    """Per-query nDCG@10 list (for bootstrap CIs + paired significance tests)."""
    def ranked(q):
        return sorted(runs[q], key=runs[q].get, reverse=True)

    return [ndcg_at_k(ranked(q), qrels[q], 10) for q in qids]


def score_system(runs, qrels, qids):
    def avg(fn):
        return sum(fn(q) for q in qids) / len(qids)

    def ranked(q):
        return sorted(runs[q], key=runs[q].get, reverse=True)

    return {
        "nDCG@10": avg(lambda q: ndcg_at_k(ranked(q), qrels[q], 10)),
        "Recall@10": avg(lambda q: recall_at_k(ranked(q), qrels[q], 10)),
        # Recall@100 is the metric most sensitive to ANN recall (usearch vs HNSW),
        # so it surfaces real backend differences the @10 metrics smooth away.
        "Recall@100": avg(lambda q: recall_at_k(ranked(q), qrels[q], 100)),
        "MAP": avg(lambda q: average_precision(ranked(q), qrels[q])),
        "P@10": avg(lambda q: precision_at_k(ranked(q), qrels[q], 10)),
    }


def paired_delta(runs_a, runs_b, qrels, qids, seed=0):
    """Paired engram-vs-baseline comparison on per-query nDCG@10: mean delta with
    a bootstrap CI, plus win/tie/loss counts and a two-sided sign-test p-value.
    A 'tie' is only honest when the delta CI straddles 0 / the sign test is n.s."""
    import math as _m

    def ranked(runs, q):
        return sorted(runs[q], key=runs[q].get, reverse=True)

    deltas = [
        ndcg_at_k(ranked(runs_a, q), qrels[q], 10) - ndcg_at_k(ranked(runs_b, q), qrels[q], 10)
        for q in qids
    ]
    wins = sum(1 for d in deltas if d > 1e-9)
    losses = sum(1 for d in deltas if d < -1e-9)
    ties = len(deltas) - wins - losses
    # exact two-sided sign test over decisive (non-tie) queries, normal-approx
    nd = wins + losses
    if nd == 0:
        p = 1.0
    else:
        z = abs(wins - losses) / _m.sqrt(nd)
        p = _m.erfc(z / _m.sqrt(2))  # two-sided normal approximation
    arr = np.asarray(deltas, dtype=np.float64)
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    boot = arr[rng.integers(0, len(arr), size=(1000, len(arr)))].mean(axis=1)
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    return {"mean": mean, "lo": lo, "hi": hi, "wins": wins, "ties": ties,
            "losses": losses, "p": p}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


# shared embedding cache so engram reuses the corpus/query encodings
_TEXT2EMB: dict[str, list[float]] = {}


async def main():
    from sentence_transformers import CrossEncoder, SentenceTransformer

    embed_model = os.environ.get("BENCH_EMBED_MODEL", "BAAI/bge-m3")
    rerank_model = os.environ.get("BENCH_RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    print(f"loading models: {embed_model} + {rerank_model} ...", flush=True)
    embedder = SentenceTransformer(embed_model)
    embedder.max_seq_length = 512  # cap (BGE-M3 defaults to 8192 — far too slow on CPU)
    # Some instruction-tuned embedders (e.g. Qwen3-Embedding) ship a default
    # 'query' prompt that sentence-transformers auto-applies to EVERY encode call
    # — which would wrongly prefix documents and double-prefix queries, since
    # engram applies its own asymmetric QUERY/PASSAGE instructions (E1). Disable
    # the auto-prompt so engram's E1 prepending is the single instruction source.
    embedder.default_prompt_name = None
    reranker = CrossEncoder(rerank_model, max_length=512)

    # print the full effective config so a result log is reproducible from itself
    # (quant + embedding_dim + proximity mode are runtime overrides, not committed)
    def _env(k, d="(default)"):
        return os.environ.get(k, d)
    print(
        "CONFIG "
        f"backend={_env('STORE_BACKEND','neo4j')} embed={embed_model} "
        f"rerank={rerank_model} embedding_dim={_env('EMBEDDING_DIM')} "
        f"quant={_env('ENGRAMDB_QUANTIZATION')} graph_proximity={_env('GRAPH_PROXIMITY_MODE')} "
        f"rerank_depth={RERANK_DEPTH} top_k={TOP_K} rerank_top_k={_env('RERANK_TOP_K')} "
        f"mmr_lambda={_env('MMR_LAMBDA')} hyde={_env('HYDE_ENABLED')} "
        f"sparse={'on' if os.environ.get('BENCH_SPARSE')=='1' else 'off'}",
        flush=True,
    )

    def encode(texts, cache=False):
        embs = embedder.encode(
            list(texts), normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )
        if cache:
            for t, e in zip(texts, embs):
                _TEXT2EMB[t] = e.tolist()
        return np.asarray(embs, dtype=np.float32)

    def cross_encode(query, texts):
        if not texts:
            return []
        return [float(s) for s in reranker.predict([(query, t) for t in texts])]

    # wire engram's seams to the cached encoder + a local reranker
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
    from app.eval import attribute_channels, bootstrap_ci
    from app.ingest import ingest_document
    from app.rerank import RERANKERS
    from app.search import search
    from app.store import create_store

    emb_mod.embed_texts = ingest_mod.embed_texts = search_mod.embed_texts = embed_texts
    emb_mod.embed_text = search_mod.embed_text = embed_text
    RERANKERS.register("local", local_rerank)

    # optional second engram pass with BGE-M3's learned-sparse channel on, to
    # measure the sparse lift inside the FULL pipeline (the probe measured it in
    # isolation). Wire a local FlagEmbedding sparse encoder to the same seams.
    bench_sparse = os.environ.get("BENCH_SPARSE", "0") == "1"
    if bench_sparse:
        from FlagEmbedding import BGEM3FlagModel

        print("loading FlagEmbedding BGE-M3 for the sparse channel ...", flush=True)
        sparse_model = BGEM3FlagModel(embed_model, use_fp16=True)

        async def embed_sparse_texts(client, texts):
            texts = list(texts)
            if not texts:
                return []
            lw = sparse_model.encode(
                texts, batch_size=64, max_length=512,
                return_dense=False, return_sparse=True, return_colbert_vecs=False,
            )["lexical_weights"]
            return [{str(t): float(w) for t, w in m.items()} for m in lw]

        async def embed_sparse_text(client, text):
            return (await embed_sparse_texts(client, [text]))[0]

        # ingest lazy-imports embed_sparse_texts from app.embeddings; search
        # binds embed_sparse_text at module load — patch both namespaces
        emb_mod.embed_sparse_texts = embed_sparse_texts
        search_mod.embed_sparse_text = embed_sparse_text

    store = create_store(get_settings())
    await store.connect()
    await store.init_schema()

    summary: dict[str, dict[str, dict]] = {}
    raw_runs: dict[str, tuple] = {}  # name -> (runs, qrels, qids) for CIs + paired tests
    # per-channel gold-hit attribution (engram's eval harness) per dataset
    attributions: dict[str, dict[str, dict]] = {}

    for name in DATASETS:
        print(f"\n########## dataset: {name} ##########", flush=True)
        d = ensure_dataset(name)
        corpus = {o["_id"]: o for o in load_jsonl(d / "corpus.jsonl")}
        queries = {o["_id"]: o["text"] for o in load_jsonl(d / "queries.jsonl")}
        qrels = load_qrels(d / "qrels" / "test.tsv")
        qids = [q for q in qrels if q in queries]
        if MAX_QUERIES > 0:
            qids = qids[:MAX_QUERIES]
        doc_ids = list(corpus)
        doc_texts = [
            f"{corpus[i].get('title', '')}\n{corpus[i].get('text', '')}".strip()
            for i in doc_ids
        ]
        print(f"corpus={len(doc_ids)} queries={len(qids)}", flush=True)

        # encode everything once (also fills the engram cache)
        t0 = time.perf_counter()
        doc_matrix = encode(doc_texts, cache=True)
        query_emb = {q: encode([queries[q]], cache=True)[0] for q in qids}
        print(f"encoded in {time.perf_counter() - t0:.0f}s", flush=True)

        runs: dict[str, dict[str, dict[str, float]]] = {}

        # ── bm25 ──
        tb = time.perf_counter()
        bm25_model = _bm25(doc_texts)
        runs["bm25"] = {}
        for q in qids:
            scores = bm25_model.get_scores(tokenize(queries[q]))
            top = np.argsort(scores)[::-1][:TOP_K]
            runs["bm25"][q] = {doc_ids[i]: float(scores[i]) for i in top}
        print(f"  bm25 in {time.perf_counter() - tb:.0f}s", flush=True)

        # ── dense (naive vector RAG) ──
        tb = time.perf_counter()
        runs["dense"] = {}
        for q in qids:
            sims = doc_matrix @ query_emb[q]
            top = np.argsort(sims)[::-1][:TOP_K]
            runs["dense"][q] = {doc_ids[i]: float(sims[i]) for i in top}
        print(f"  dense in {time.perf_counter() - tb:.0f}s", flush=True)

        # ── dense + cross-encoder rerank (standard 2-stage) ──
        tb = time.perf_counter()
        runs["dense+rerank"] = {}
        for q in qids:
            sims = doc_matrix @ query_emb[q]
            cand = np.argsort(sims)[::-1][:RERANK_DEPTH]
            ce = cross_encode(queries[q], [doc_texts[i] for i in cand])
            runs["dense+rerank"][q] = {doc_ids[i]: s for i, s in zip(cand, ce)}
        print(f"  dense+rerank in {time.perf_counter() - tb:.0f}s", flush=True)

        # ── hybrid (dense + BM25 via RRF) + rerank — the honest 2026 control ──
        # Isolates engram's *fusion* from its median/MMR/graph stages: if engram
        # ties this, its single-hop lift over dense+rerank is the hybrid fusion,
        # not the graph machinery (which BEIR single-chunk docs don't exercise).
        tb = time.perf_counter()
        runs["hybrid+rerank"] = {}
        for q in qids:
            sims = doc_matrix @ query_emb[q]
            dense_rank = np.argsort(sims)[::-1][:200]
            bm = bm25_model.get_scores(tokenize(queries[q]))
            bm_rank = np.argsort(bm)[::-1][:200]
            rrf: dict[int, float] = {}
            for r, i in enumerate(dense_rank):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + r)
            for r, i in enumerate(bm_rank):
                rrf[int(i)] = rrf.get(int(i), 0.0) + 1.0 / (60 + r)
            cand = sorted(rrf, key=rrf.get, reverse=True)[:RERANK_DEPTH]
            ce = cross_encode(queries[q], [doc_texts[i] for i in cand])
            runs["hybrid+rerank"][q] = {doc_ids[i]: s for i, s in zip(cand, ce)}
        print(f"  hybrid+rerank in {time.perf_counter() - tb:.0f}s", flush=True)
        print("baselines done", flush=True)

        # ── engram (full pipeline) ──
        await _wipe(store)
        t0 = time.perf_counter()
        for i, did in enumerate(doc_ids):
            if doc_texts[i]:
                await ingest_document(
                    store, None, doc_texts[i],
                    title=corpus[did].get("title", ""), source="beir", document_id=did,
                )
        print(f"engram ingest {len(doc_ids)} docs in {time.perf_counter() - t0:.0f}s", flush=True)
        t0 = time.perf_counter()
        runs["engram"] = {}
        eng_results: dict[str, list] = {}
        for j, q in enumerate(qids):
            hits = await search(store, None, queries[q], top_k=TOP_K)
            eng_results[q] = hits
            scored: dict[str, float] = {}
            for r in hits:
                if r.document_id not in scored or r.rerank_score > scored[r.document_id]:
                    scored[r.document_id] = r.rerank_score
            runs["engram"][q] = scored
            if j % 100 == 0:
                print(f"  engram queried {j}/{len(qids)}", flush=True)
        print(f"engram query in {time.perf_counter() - t0:.0f}s", flush=True)
        attributions[name] = {"engram": attribute_channels(eng_results, qrels, 10)}

        # ── engram + learned-sparse (re-ingest so sparse weights are stored) ──
        if bench_sparse:
            settings = get_settings()
            settings.sparse_enabled = True
            settings.sparse_weight = float(os.environ.get("SPARSE_WEIGHT", "0.2"))
            await _wipe(store)
            t0 = time.perf_counter()
            for i, did in enumerate(doc_ids):
                if doc_texts[i]:
                    await ingest_document(
                        store, None, doc_texts[i],
                        title=corpus[did].get("title", ""), source="beir", document_id=did,
                    )
            print(
                f"engram+sparse ingest {len(doc_ids)} docs in "
                f"{time.perf_counter() - t0:.0f}s (sparse_weight={settings.sparse_weight})",
                flush=True,
            )
            t0 = time.perf_counter()
            runs["engram+sparse"] = {}
            eng_sparse_results: dict[str, list] = {}
            for j, q in enumerate(qids):
                hits = await search(store, None, queries[q], top_k=TOP_K)
                eng_sparse_results[q] = hits
                scored = {}
                for r in hits:
                    if r.document_id not in scored or r.rerank_score > scored[r.document_id]:
                        scored[r.document_id] = r.rerank_score
                runs["engram+sparse"][q] = scored
                if j % 100 == 0:
                    print(f"  engram+sparse queried {j}/{len(qids)}", flush=True)
            print(f"engram+sparse query in {time.perf_counter() - t0:.0f}s", flush=True)
            attributions[name]["engram+sparse"] = attribute_channels(
                eng_sparse_results, qrels, 10
            )
            settings.sparse_enabled = False

        summary[name] = {sys: score_system(r, qrels, qids) for sys, r in runs.items()}
        raw_runs[name] = (dict(runs), qrels, qids)

    await store.close()

    # ── report ──
    systems = ["bm25", "dense", "dense+rerank", "hybrid+rerank", "engram"]
    if bench_sparse:
        systems.append("engram+sparse")
    systems = [s for s in systems if s in summary[DATASETS[0]]]
    metrics = ["nDCG@10", "Recall@10", "Recall@100", "MAP", "P@10"]
    for name in DATASETS:
        runs_n, qrels_n, qids_n = raw_runs[name]
        print(f"\n=== {name} ({embed_model} + {rerank_model}, n={len(qids_n)}) ===")
        print(f"{'system':<16}" + "".join(f"{m:>12}" for m in metrics) + "   nDCG@10 95%CI")
        for sys in systems:
            row = summary[name][sys]
            mean, lo, hi = bootstrap_ci(per_query_ndcg(runs_n[sys], qrels_n, qids_n))
            ci = f"[{lo:.4f},{hi:.4f}]"
            print(f"{sys:<16}" + "".join(f"{row[m]:>12.4f}" for m in metrics) + f"   {ci}")
        # paired significance: engram vs the two rerank baselines (is the lift real?
        # is any backend delta a tie?). Reports mean delta + 95% CI + sign-test p.
        for base in ("dense+rerank", "hybrid+rerank"):
            if "engram" in runs_n and base in runs_n:
                d = paired_delta(runs_n["engram"], runs_n[base], qrels_n, qids_n)
                sig = "n.s." if d["lo"] <= 0 <= d["hi"] else "SIGNIFICANT"
                print(
                    f"  paired engram−{base}: Δave_nDCG@10={d['mean']:+.4f} "
                    f"95%CI[{d['lo']:+.4f},{d['hi']:+.4f}] "
                    f"win/tie/loss={d['wins']}/{d['ties']}/{d['losses']} "
                    f"sign-p={d['p']:.3f} ({sig})"
                )
        # per-channel gold-hit attribution (which channel surfaced each gold hit,
        # and — the key line — which it recovered uniquely; engram's eval harness)
        for sys, attr in attributions.get(name, {}).items():
            print(
                f"  [{sys}] gold@10={attr['gold_hits_retrieved']} "
                f"by_channel={attr['by_channel']} unique={attr['unique_to_channel']}"
            )


def _bm25(doc_texts):
    from rank_bm25 import BM25Okapi

    return BM25Okapi([tokenize(t) for t in doc_texts])


async def _wipe(store):
    # clear documents between datasets, backend-agnostic. Fast path for neo4j
    # (one Cypher DETACH DELETE); otherwise drop via the Store protocol so the
    # in-process engramdb backend (no _driver) is wiped the same way.
    driver = getattr(store, "_driver", None)
    if driver is not None:
        async with driver.session() as session:
            await session.run(
                "MATCH (n) WHERE n:Chunk OR n:Document OR n:Keyword OR n:Community "
                "DETACH DELETE n"
            )
        return
    for doc in await store.list_documents():
        await store.delete_document(doc["id"])


if __name__ == "__main__":
    asyncio.run(main())
