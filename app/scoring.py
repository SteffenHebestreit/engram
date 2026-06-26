import numpy as np


def dbsf_normalize(scores: list[float]) -> list[float]:
    """Distribution-based score fusion normalization.

    Z-score each value against the channel's own score distribution, clip to
    +/- 3 sigma, and map to [0, 1]. Unlike min-max this is robust to a single
    outlier hit compressing the rest of the channel into a meaningless band.
    """
    if not scores:
        return []
    arr = np.asarray(scores, dtype=np.float64)
    std = arr.std()
    if std == 0:
        return [0.5] * len(scores)
    z = np.clip((arr - arr.mean()) / std, -3.0, 3.0)
    return ((z + 3.0) / 6.0).tolist()


def mmr_select(
    relevance: list[float],
    embeddings: list[list[float]],
    k: int,
    lambda_: float,
) -> list[int]:
    """Maximal Marginal Relevance: greedily pick k indices balancing relevance
    against redundancy with what is already selected.

    mmr_i = lambda * rel_i - (1 - lambda) * max_j(cos(emb_i, emb_j selected))

    Relevance is min-max normalized internally so lambda trades off against
    cosine similarity on a comparable scale. Returns selected indices in pick
    order (best first).
    """
    n = len(relevance)
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return sorted(range(n), key=lambda i: relevance[i], reverse=True)

    rel = np.asarray(relevance, dtype=np.float64)
    spread = rel.max() - rel.min()
    rel = (rel - rel.min()) / spread if spread > 0 else np.full(n, 0.5)

    emb = np.asarray(embeddings, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1)
    norms[norms == 0] = 1.0
    emb = emb / norms[:, None]

    selected = [int(np.argmax(rel))]
    # running max cosine similarity of each candidate to the selected set
    max_sim = emb @ emb[selected[0]]
    while len(selected) < k:
        mmr = lambda_ * rel - (1.0 - lambda_) * max_sim
        mmr[selected] = -np.inf
        pick = int(np.argmax(mmr))
        selected.append(pick)
        max_sim = np.maximum(max_sim, emb @ emb[pick])
    return selected


def autocut(scores: list[float], min_keep: int, min_gap: float) -> int:
    """How many of the descending-sorted scores to keep.

    Min-max normalizes the scores (scale-agnostic, reranker output may be
    logits or probabilities) and cuts before the first drop of at least
    `min_gap` between consecutive results, never keeping fewer than
    `min_keep`. Returns len(scores) when no such cliff exists.
    """
    n = len(scores)
    if n <= min_keep:
        return n
    arr = np.asarray(scores, dtype=np.float64)
    spread = arr.max() - arr.min()
    if spread == 0:
        return n
    arr = (arr - arr.min()) / spread
    for i in range(max(min_keep, 1), n):
        if arr[i - 1] - arr[i] >= min_gap:
            return i
    return n


def sparse_dot(a: dict[str, float], b: dict[str, float]) -> float:
    """Dot product of two BGE-M3 learned-sparse term-weight maps (over shared
    terms) — the exact-term match score the dense vector smooths away."""
    if not a or not b:
        return 0.0
    if len(a) > len(b):  # iterate the smaller map
        a, b = b, a
    return sum(weight * b.get(term, 0.0) for term, weight in a.items())


def sparse_scores(
    query_sparse: dict[str, float] | None,
    candidate_sparse: list[dict[str, float] | None],
) -> list[float]:
    """Score each candidate's sparse map against the query's, normalized to
    [0, 1] (by the max) so it combines with the other fused-score components."""
    if not query_sparse:
        return [0.0] * len(candidate_sparse)
    raw = [sparse_dot(query_sparse, cand or {}) for cand in candidate_sparse]
    top = max(raw, default=0.0)
    if top <= 0:
        return [0.0] * len(raw)
    return [value / top for value in raw]


def median_proximity_scores(embeddings: list[list[float]]) -> list[float]:
    """Score each embedding by cosine similarity to the element-wise median
    vector of the whole result set, mapped to [0, 1].

    Results that sit close to the "center of mass" of what the search found
    score high; outliers far from the median score low.
    """
    if not embeddings:
        return []
    if len(embeddings) == 1:
        return [1.0]

    arr = np.asarray(embeddings, dtype=np.float64)
    median_vec = np.median(arr, axis=0)

    median_norm = np.linalg.norm(median_vec)
    if median_norm == 0:
        return [0.5] * len(embeddings)

    row_norms = np.linalg.norm(arr, axis=1)
    row_norms[row_norms == 0] = 1.0
    cosine = (arr @ median_vec) / (row_norms * median_norm)

    # cosine in [-1, 1] -> [0, 1]
    return ((cosine + 1.0) / 2.0).tolist()
