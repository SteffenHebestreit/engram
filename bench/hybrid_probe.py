"""Probe: do BGE-M3's discarded sparse + ColBERT outputs beat dense-only?

engram uses only BGE-M3's DENSE vector. This measures nDCG@10 on BEIR SciFact
for dense / sparse(learned-lexical) / dense+sparse hybrid / +ColBERT rerank —
ALL from the same BGE-M3 forward pass — to validate (and size) the #1
retrieval-quality feature before wiring it into engram.

Run on GPU:
  docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml \
      run --rm --build --no-deps runner python -m bench.hybrid_probe
"""

import os
import time

import numpy as np
from scipy.sparse import csr_matrix

from bench.compare import ensure_dataset, load_jsonl, load_qrels, ndcg_at_k

DATASET = os.environ.get("BENCH_DATASET", "scifact")
RERANK_DEPTH = int(os.environ.get("BENCH_RERANK_DEPTH", "100"))
W_DENSE = float(os.environ.get("BENCH_W_DENSE", "0.5"))
VOCAB = 250002  # BGE-M3 / XLM-RoBERTa vocab size


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def _sparse_vec(lw: dict) -> csr_matrix:
    if not lw:
        return csr_matrix((1, VOCAB), dtype=np.float32)
    cols = [int(t) for t in lw]
    vals = [float(v) for v in lw.values()]
    return csr_matrix((vals, ([0] * len(cols), cols)), shape=(1, VOCAB), dtype=np.float32)


def main():
    d = ensure_dataset(DATASET)
    corpus = {o["_id"]: o for o in load_jsonl(d / "corpus.jsonl")}
    queries = {o["_id"]: o["text"] for o in load_jsonl(d / "queries.jsonl")}
    qrels = load_qrels(d / "qrels" / "test.tsv")
    qids = [q for q in qrels if q in queries]
    doc_ids = list(corpus)
    doc_texts = [
        f"{corpus[i].get('title', '')}\n{corpus[i].get('text', '')}".strip() for i in doc_ids
    ]
    print(f"{DATASET}: corpus={len(doc_ids)} queries={len(qids)} (w_dense={W_DENSE})", flush=True)

    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    t0 = time.time()
    doc_out = model.encode(
        doc_texts, batch_size=32, max_length=512,
        return_dense=True, return_sparse=True, return_colbert_vecs=True,
    )
    print(f"encoded corpus in {time.time() - t0:.0f}s", flush=True)
    doc_dense = np.asarray(doc_out["dense_vecs"], dtype=np.float32)
    doc_cv = doc_out["colbert_vecs"]

    # docs x vocab learned-sparse matrix (CSR) for fast full-corpus sparse scoring
    rows, cols, vals = [], [], []
    for i, lw in enumerate(doc_out["lexical_weights"]):
        for tok, w in lw.items():
            rows.append(i)
            cols.append(int(tok))
            vals.append(float(w))
    doc_csr = csr_matrix((vals, (rows, cols)), shape=(len(doc_ids), VOCAB), dtype=np.float32)

    q_out = model.encode(
        [queries[q] for q in qids], batch_size=32, max_length=512,
        return_dense=True, return_sparse=True, return_colbert_vecs=True,
    )
    q_dense = np.asarray(q_out["dense_vecs"], dtype=np.float32)
    q_lw = q_out["lexical_weights"]
    q_cv = q_out["colbert_vecs"]

    systems = ["dense", "sparse", "dense+sparse", "dense+sparse+colbert"]
    runs: dict[str, dict] = {s: {} for s in systems}

    t0 = time.time()
    for idx, qid in enumerate(qids):
        ds = doc_dense @ q_dense[idx]
        ss = np.asarray((doc_csr @ _sparse_vec(q_lw[idx]).T).todense()).ravel()
        fused = W_DENSE * _minmax(ds) + (1.0 - W_DENSE) * _minmax(ss)
        for name, scores in [("dense", ds), ("sparse", ss), ("dense+sparse", fused)]:
            top = np.argsort(scores)[::-1][:100]
            runs[name][qid] = {doc_ids[i]: float(scores[i]) for i in top}
        cand = np.argsort(fused)[::-1][:RERANK_DEPTH]
        cb = [float(model.colbert_score(q_cv[idx], doc_cv[i])) for i in cand]
        runs["dense+sparse+colbert"][qid] = {doc_ids[i]: s for i, s in zip(cand, cb)}
        if idx % 100 == 0:
            print(f"  scored {idx}/{len(qids)}", flush=True)
    print(f"scored in {time.time() - t0:.0f}s", flush=True)

    print(f"\n=== BGE-M3 hybrid probe on {DATASET} (nDCG@10) ===")
    for name in systems:
        avg = sum(
            ndcg_at_k(sorted(runs[name][q], key=runs[name][q].get, reverse=True), qrels[q], 10)
            for q in qids
        ) / len(qids)
        print(f"  {name:<24} {avg:.4f}")


if __name__ == "__main__":
    main()
