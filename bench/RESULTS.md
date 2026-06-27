# engram benchmark results

Controlled, reproducible comparisons of engram against the standard RAG
retrieval strategies. **Same datasets, same embedding model, same cross-encoder,
same metrics — only the retrieval *architecture* changes.** That isolates what
each approach contributes; running external frameworks (LightRAG, HippoRAG,
Haystack) head-to-head would confound the result with their different
embedders/LLMs, so we instead reimplement the standard strategies on identical
components.

| system | what it is |
|---|---|
| `bm25` | classic lexical retrieval (rank-bm25) |
| `dense` | single-vector cosine — **naive vector RAG** (LlamaIndex/LangChain default) |
| `dense+rerank` | dense retrieval → cross-encoder rerank — the **standard 2-stage** production RAG |
| `engram` | DBSF fusion of dense+BM25 channels → median-proximity → MMR → cross-encoder rerank (+ keyword graph + PageRank on multi-hop) |

Two configurations, both reported:

- **Production (GPU)** — `BAAI/bge-m3` (1024-d) + `BAAI/bge-reranker-v2-m3`, engram's
  actual default stack, on an RTX 4080. **The headline numbers.**
- **Floor (CPU)** — `all-MiniLM-L6-v2` + `ms-marco-MiniLM-L-6-v2`, a deliberately
  weak/old, CPU-friendly stack — a lower bound, and useful for the
  embedder-strength analysis below.

(`bm25` is identical across configs — it uses no embeddings — a handy consistency
check: every `bm25` row matches across the two runs.)

---

## What this measures: the architecture, not the models

Every system here — `dense`, `dense+rerank`, `engram` — uses the **identical
embedder and reranker**. So `engram − dense+rerank` cancels the models out. That
delta is **positive in every config tested** (point estimates, single unseeded runs):

| config (embedder + reranker) | engram | dense+rerank | engram−dense+rerank ΔnDCG@10 | Δrecall@10 |
|---|---|---|---|---|
| BGE-M3 + bge-reranker-v2-m3 (SciFact) | 0.7408 | 0.7250 | +1.58 | +3.10 |
| BGE-M3 + **Qwen3-Reranker-0.6B** (SciFact) | 0.7723 | 0.7508 | +2.15 | +4.00 |
| Qwen3-Embed-0.6B + bge-reranker (SciFact) | 0.7409 | 0.7344 | +0.65 | +0.13 |
| BGE-M3 + bge-reranker-v2-m3 (NFCorpus) | 0.3411 | 0.3324 | +0.87 | +0.52 |

> **⚠️ Correction (2026-06-27, after an adversarial review + a rigorous re-run with
> bootstrap CIs, paired sign tests, and a `hybrid(dense+BM25 RRF)+rerank` control —
> see [docs/engram-db.md](../docs/engram-db.md)):** the `engram − dense+rerank` delta
> above is **NOT statistically significant on single-hop BEIR** (paired 95%CI
> straddles 0; sign-p 0.27 SciFact / 0.41 NFCorpus; ~83% of queries tie), and it is
> **mostly the added BM25 channel, not** the graph / median-proximity / MMR: a
> `hybrid+rerank` baseline (dense+BM25 fusion, *no* graph/median/MMR) already reaches
> 0.7357 on SciFact, and `engram − hybrid+rerank` is +0.003 (n.s.). On single-chunk
> single-hop corpora MMR is a no-op (λ=1.0), PPR is off, and the keyword graph
> contributes ~0 unique gold hits, so these benches **cannot** credit engram's
> distinctive stages. Treat the rows above as "engram ≥ naive 2-stage", and look to
> the **multi-hop** benchmark below for whether the graph adds *significant* recall.

Two structural observations (now read in light of that correction):

- The architecture is a **robustness layer**: its recall gain is large with a
  weaker embedder (BGE-M3 +3–4) and ≈0 with a SOTA embedder (Qwen3 +0.1) — when
  the embedder already retrieves everything, there's less for fusion+graph to add.
- With a strong reranker the point estimate *rose* to +2.3 (Qwen3-Reranker), **but
  per the ⚠️ correction above this is not statistically significant** (26 win / 26
  loss, sign-p 1.0) and is a tie vs a hybrid+rerank control — the reranker upgrade
  (+3.15 nDCG, real and robust) lifts everything; it is not evidence the
  architecture compounds. The honest lever here is the **reranker**, not the graph.

A useful property for trusting the deltas: even when an embedder run is mis-
prompted (§1d), the architecture Δ stays valid, because `engram` and
`dense+rerank` share the *same* (mis-)embeddings — the confound cancels in the
subtraction. The model swaps below move the *absolute* ceiling (available to
anyone); the table above is what engram itself contributes.

The model choices are reported below for completeness, but read them as "engram
on top of model X," and read the table above as "engram vs. the alternative
*architecture*, model held constant."

**An architectural property these benchmarks *can't* show: chunking robustness.**
BEIR documents are single-chunk, so chunk boundaries never come into play here —
but on real (long) documents, naive RAG depends heavily on chunk size / overlap /
semantic splitting to avoid losing context across a boundary. engram's
`NEXT_CHUNK` graph retrieves a hit's neighbouring chunks, so a passage split
across a boundary is recovered at retrieval time. That makes overlap redundant
(engram defaults `CHUNK_OVERLAP_CHARS=0`) and semantic chunking largely
unnecessary — chunking becomes a robustness property instead of a tuning problem.
It's a real architectural advantage; it just needs a long-document eval (or the
`/eval` harness on your own corpus) to quantify, which BEIR can't provide.

---

## Benchmark 1 — BEIR retrieval (SciFact, NFCorpus)

Standard IR benchmark; headline metric nDCG@10.

### Production stack (BGE-M3 + bge-reranker-v2-m3, GPU)

**SciFact** (5,183 docs · 300 queries)

| system | nDCG@10 | Recall@10 | MAP | P@10 |
|---|---|---|---|---|
| bm25 | 0.6519 | 0.7740 | 0.6132 | 0.0850 |
| dense (naive RAG) | 0.6415 | 0.7751 | 0.5990 | 0.0870 |
| dense+rerank (2-stage) | 0.7250 | 0.8246 | 0.6934 | 0.0933 |
| **engram** | **0.7408** | **0.8556** | **0.7044** | **0.0963** |

**NFCorpus** (3,633 docs · 323 queries)

| system | nDCG@10 | Recall@10 | MAP | P@10 |
|---|---|---|---|---|
| bm25 | 0.3062 | 0.1521 | 0.1368 | 0.2180 |
| dense | 0.3174 | 0.1504 | 0.1446 | 0.2316 |
| dense+rerank | 0.3324 | 0.1614 | 0.1518 | 0.2412 |
| **engram** | **0.3411** | **0.1666** | **0.1567** | **0.2446** |

engram beats naive vector RAG clearly (~+10 pts nDCG@10 SciFact / ~+2.4 NFCorpus)
and beats BM25. Its edge over the **strong dense+rerank** pipeline is a positive
*point estimate* (+1.4 / +0.5 pts nDCG@10) **but not statistically significant**
(see the ⚠️ correction in "What this measures" above: 95%CI straddles 0, sign-p
0.27/0.41, and a hybrid(dense+BM25)+rerank control captures nearly all of it — so
the gain is the BM25 channel, not median-proximity/MMR). Treat these rows as
"engram ≥ naive 2-stage", not as a demonstrated architecture quality win.

### Floor stack (MiniLM + ms-marco cross-encoder, CPU)

**SciFact**

| system | nDCG@10 | Recall@10 | MAP | P@10 |
|---|---|---|---|---|
| bm25 | 0.6519 | 0.7740 | 0.6132 | 0.0850 |
| dense | 0.6451 | 0.7833 | 0.6031 | 0.0883 |
| dense+rerank | 0.6936 | 0.8222 | 0.6507 | 0.0923 |
| **engram** | **0.6968** | 0.8172 | **0.6593** | 0.0920 |

**NFCorpus**

| system | nDCG@10 | Recall@10 | MAP | P@10 |
|---|---|---|---|---|
| bm25 | 0.3062 | 0.1521 | 0.1368 | 0.2180 |
| dense | 0.3163 | 0.1550 | 0.1425 | 0.2433 |
| dense+rerank | 0.3389 | 0.1550 | 0.1523 | 0.2424 |
| **engram** | **0.3457** | **0.1583** | **0.1544** | **0.2486** |

Same story at the floor: engram leads nDCG@10 and MAP on both (sweeps NFCorpus).

---

## Benchmark 1c — the learned-sparse channel: a measured lesson in terminal reranking

engram already runs BGE-M3, which emits a SPLADE-style **learned-sparse**
term-weight vector alongside its dense one. An isolated probe
([hybrid_probe.py](hybrid_probe.py)) confirmed the signal is real:

| SciFact, nDCG@10 (no reranker) | score |
|---|---|
| dense only (BGE-M3) | 0.6419 |
| dense + sparse | 0.6815 (**+4.0**) |
| dense + sparse + ColBERT | 0.6998 (+5.8) |

So we wired sparse into engram (opt-in, `SPARSE_ENABLED`): each chunk's sparse
weights are stored and dotted against the query's, folded into the **fused
score**. Then we measured it in the *full* pipeline (`BENCH_SPARSE=1`) across
**four configurations** to find out exactly where a fusion-layer signal survives
the terminal cross-encoder. The answer is precise and product-relevant.

**SciFact, production stack (BGE-M3 + bge-reranker-v2-m3):**

| config | system | nDCG@10 | Recall@10 | MAP | Δ nDCG |
|---|---|---|---|---|---|
| **rerank depth 15** (engram's **default**) | engram | 0.7320 | 0.8262 | 0.6967 | |
| | **engram+sparse** | **0.7428** | **0.8446** | **0.7049** | **+1.1** |
| rerank depth 100 (deep retrieval probe) | engram | 0.7408 | 0.8556 | 0.7044 | |
| | engram+sparse | 0.7376 | 0.8489 | 0.7028 | −0.3 |
| reranker OFF (order = fused score) | engram | 0.6893 | 0.8161 | 0.6493 | |
| | **engram+sparse** | **0.7051** | **0.8393** | **0.6612** | **+1.6** |

**What this proves — sparse helps the *shortlist*, so it wins exactly when the
shortlist is the bottleneck.**

- **Reranker OFF:** sparse lifts engram's own fused ranking **+1.6 nDCG@10 / +2.3
  recall@10**. The signal is genuinely valuable.
- **At engram's default rerank depth (15):** sparse **wins +1.1 nDCG@10 / +1.8
  recall@10** — because the fused score (now including sparse) decides which 15
  candidates the cross-encoder ever sees, and sparse pulls better ones into that
  shallow shortlist. **`engram+sparse @ depth-15 = 0.7428` is the single best
  configuration measured on this dataset.**
- **At depth 100** (used elsewhere in this doc to measure retrieval deeply), the
  shortlist is so deep the reranker sees almost the whole pool, so sparse barely
  changes what it reranks — the gain washes out (−0.3, reordering noise).

**The honest, generalizable takeaways:**

1. **A signal upstream of a strong terminal reranker helps only insofar as it
   changes the shortlist the reranker sees.** It pays off at realistic
   (shallow) rerank depths and is masked by very deep reranking. engram's
   default depth-15 is the realistic case — and there, sparse is a clear win.
2. **Bigger headroom is still on the table:** engram's integration *re-scores*
   the candidate pool but never *expands* it, so it can't yet recover a doc the
   dense channels missed entirely — exactly where the isolated +4.0 lived. Treating
   sparse as a retrieval channel that **contributes candidates** (plan item F3)
   should push the depth-15 win further and is the next step.
3. Sparse stays **opt-in** (it needs a multi-output endpoint — see
   [`deploy/bge-m3`](../deploy/bge-m3)), but on this evidence it is **recommended
   when available**: at the default config a clear win on SciFact and never a
   regression.

**Generalization (NFCorpus, depth-15):** the direction holds but is dataset-
dependent:

| NFCorpus, depth-15 | nDCG@10 | Recall@10 | MAP |
|---|---|---|---|
| engram | 0.3417 | 0.1634 | 0.1356 |
| engram+sparse | **0.3442 (+0.25)** | 0.1624 | **0.1365** |

NFCorpus (medical, many-relevant-per-query, low absolute recall) gives a smaller
nDCG/MAP gain and flat recall — positive but marginal. So the honest summary is:
**sparse is a clear win on SciFact (+1.1 nDCG / +1.8 recall@10) and neutral-to-
slightly-positive on NFCorpus (+0.25 nDCG), at the default rerank depth — and
never a regression on either.**

**Per-channel attribution settles the "should we go further" question (this is
the eval harness measuring itself).** Running the same SciFact config through
engram's `/eval` attribution (`bench/compare.py` now prints it):

```
[engram] gold@10=280  by_channel={content:280, fulltext:271, graph:sequence:13}  unique={content:9}
```

The **dense `content` channel surfaced all 280 gold hits**; fulltext and graph
are overlapping *subsets*, and only 9 gold hits were unique to dense. On SciFact,
dense retrieval already finds everything — so a *retrieval*-side sparse channel
(candidate-expansion, the proposed "F3") has **no headroom here**, and engram
already runs BM25 fulltext for exact terms anyway. **Decision: don't build F3.**
Sparse's value is **re-ranking** the pool (the measured +1.1/+1.8), not finding
new documents; the shipped re-scoring integration is the right one. Exact-term
corpora (codes/IDs) are where a lexical channel earns unique recoveries — and the
attribution is exactly how you'd detect that on *your* corpus rather than guess.

> Why several runs for one feature: the depth-100 number alone looked like a
> negative; the depth-15 (default), reranker-off, and NFCorpus runs reveal a
> clear-to-neutral win whose mechanism is fully pinned down. That is the
> controlled harness doing its job — the right config and the mechanism, not a
> cherry-picked number.

---

## Benchmark 1d — does a *stronger embedder* help? (finding the ceiling)

The embedder is the dominant variable in retrieval, so the obvious lever for a
"clear advantage" is a better one. We swapped BGE-M3 for **Qwen3-Embedding-0.6B**
— a 2026 MTEB leader (~70.7, rivaling NV-Embed-v2 at 7.8B) that is
instruction-aware, so engram's **E1** query/passage prefixes feed it correctly.

> **Caveat (prompt handling) + status.** These Qwen3-Embedding runs predate a
> bench fix: `sentence-transformers` auto-applies Qwen3's default `query` prompt to
> *every* encode, which wrongly prefixes documents and double-prefixes queries on
> top of engram's E1. The fix is in (`embedder.default_prompt_name = None`, so E1
> is the sole instruction source). The clean re-run with a **4B** embedder was
> then **blocked by the 16 GB GPU** (a 4B model + reranker thrashes VRAM → ~2 h),
> so it is **deferred to a higher-memory box** — see [PENDING.md](PENDING.md). The
> **engram-level conclusion — the embedder washes out at the reranker ceiling — is
> robust** to the prompt confound (`engram` and `dense+rerank` share the same
> embeddings, so the architecture Δ cancels it), but the exact *dense* deltas and
> the "negligible at the pipeline" claim are **not yet verified on clean 4B/8B
> numbers** — that's the open item.

| dataset | stage | BGE-M3 | Qwen3-0.6B | Δ |
|---|---|---|---|---|
| **SciFact** | dense (raw embedder) | 0.6415 | **0.6856** | **+4.4** |
| | dense+rerank | 0.7250 | 0.7344 | +0.9 |
| | **engram (full)** | 0.7408 | 0.7409 | **+0.0 (tie)** |
| **NFCorpus** | dense (raw embedder) | **0.3174** | 0.3009 | **−1.65** |
| | dense+rerank | 0.3324 | 0.3379 | +0.55 |
| | **engram (full)** | 0.3411 | 0.3469 | +0.58 |

**Qwen3 is a much stronger *dense* embedder on SciFact (+4.4 nDCG@10) — yet
engram ties (0.7409 vs 0.7408).** On NFCorpus (medical) Qwen3's raw dense is
actually *worse* (−1.65, domain-dependent), yet engram still edges +0.58. The
embedder's large, domain-varying dense differences **collapse to ≈BGE-M3 at the
full pipeline.** This is the same pattern as the sparse and multi-hop findings,
now for the embedder itself.

**The mechanism — and the one lever that can't wash out.** engram's pipeline is a
*robustness equalizer*: it lifts weak BGE-M3 dense by +10 pts and strong Qwen3
dense by +5.5 pts, converging both to the **terminal cross-encoder's ceiling**
(~0.741 on SciFact). Every *upstream* signal — sparse, graph, a +4.4-nDCG
embedder — is re-scored away by `bge-reranker-v2-m3`. So:

- **BGE-M3 stays engram's right default** — robust across domains, and a stronger
  embedder buys nothing at the pipeline level on these benchmarks. (engram's E1
  prefixes still make swapping in an instruction-tuned embedder *correct* for
  anyone who wants to.)
- **The reranker is the ceiling**, and the *only* stage with nothing downstream
  to wash it out. A better reranker is therefore the single highest-leverage
  retrieval lever left — measured next (Benchmark 1e).
- The real "clear advantage" vs alternatives is **not** another upstream
  retrieval tweak (they're capped) but the moats this harness keeps honest:
  judge-free per-corpus eval, the memory write-path, and the agent-tool interface.

---

## Benchmark 1e — the reranker IS the lever (breaking the ceiling)

If the terminal cross-encoder caps everything, then the cross-encoder is the one
stage with nothing downstream to wash it out — so a *better* one lifts the whole
result. We swapped `bge-reranker-v2-m3` for **Qwen3-Reranker-0.6B** (a 2026 SOTA
reranker — Qwen3-Reranker-4B scores 69.76 vs bge's 57.03 on BEIR; the 0.6B is a
drop-in `sentence-transformers` `CrossEncoder`), **keeping the BGE-M3 embedder**
so only the reranker changes.

**SciFact, BGE-M3 embedder, reranker swapped:**

| system | bge-reranker-v2-m3 | **Qwen3-Reranker-0.6B** | Δ |
|---|---|---|---|
| dense+rerank | 0.7250 | 0.7508 | **+2.6** |
| **engram** nDCG@10 | 0.7408 | **0.7723** | **+3.15** |
| engram Recall@10 | 0.8556 | **0.8909** | **+3.5** |
| engram MAP | 0.7044 | **0.7334** | **+2.9** |
| engram gold@10 | 280 | **301** | +21 |

**This breaks the ~0.741 ceiling — the single biggest jump in this whole study
(+3.15 nDCG@10 / +3.5 recall@10), from one drop-in change.** It confirms the
thesis exactly: the reranker *was* the ceiling, and because nothing re-scores
*after* it, a better reranker lifts everything.

> **Rigorous re-check (2026-06-27, engramdb backend, with a `hybrid+rerank`
> control + paired tests):** engram·engramdb + Qwen3-Reranker = **0.7738**, which
> **ties the neo4j 0.7723** above — so the reranker upgrade is backend-agnostic and
> the embedded backend delivers it too (the "+3 nDCG win" is preserved on
> engramdb). But the earlier "engram's fusion+graph+MMR add +2.15 over
> dense+rerank" attribution is **mostly the BM25 fusion, not the graph/MMR**: a
> `hybrid+rerank` baseline reaches **0.7699**, and `engram − hybrid+rerank` is just
> **+0.0038 (n.s.**, sign-p 0.17). `engram − dense+rerank` (+0.0229) clears the
> bootstrap mean-CI [+0.007, +0.043] but its per-query **sign test is n.s.**
> (26 win / 248 tie / 26 loss — the mean is carried by a few large gains, not a
> broad edge). Honest reading: the **reranker** is the real, large, dataset-robust
> lever; engram's distinctive stages do not add a *statistically robust* lift over
> a strong hybrid baseline even at the better reranker.

**It generalizes** — the reranker swap lifts engram on NFCorpus too, by *more*:
0.3411 → **0.3795 nDCG@10 (+3.84)**, recall@10 0.1666 → 0.1801. So the reranker
is a robust, dataset-independent **+3–4 nDCG@10** lever, not a SciFact fluke.

**Why this is the actionable "clear advantage":**

- It's a **drop-in** — engram's reranker is already a configurable endpoint /
  `RERANKERS` strategy; no architecture change.
- **Multilingual preserved** — Qwen3-Reranker covers 100+ languages like
  `bge-reranker-v2-m3`, so it's a strict upgrade, not a trade-off.
- **Small** — 0.6B; and **Qwen3-Reranker-4B** (the +12.7-BEIR model) is the
  bigger swing still on the table.
- The attribution shifts too: with the stronger reranker, fulltext uniquely
  recovers **13** gold hits (vs 0 under bge-reranker) — a better reranker
  *promotes* the lexical channel's unique finds instead of burying them.

**Revised headline of this whole section:** upstream retrieval tweaks (sparse,
graph, embedder) are capped by the reranker on saturated benchmarks — but the
**reranker itself is a real, large, drop-in lever**. The path to a clear
retrieval advantage runs through the reranker (+ the eval/memory moats), not
through more fusion/embedding signals.

---

## Benchmark 2 — Multi-hop retrieval (HotpotQA)

The turf graph-RAG systems (HippoRAG, GraphRAG) compete on: questions whose
answer spans **two supporting passages that must be retrieved together.** Metric
is Recall@k of the supporting passages (HippoRAG's metric). engram runs its
**full graph pipeline** — YAKE keyword extraction (no LLM) builds the
shared-keyword graph; keyword-sibling expansion + GDS PageRank surface a
bridge-linked passage. HotpotQA distractor dev, 500 questions, 4,937 passages.

### Production stack (BGE-M3 + bge-reranker-v2-m3, GPU, n=500)

| system | Recall@2 | Recall@5 | Recall@10 |
|---|---|---|---|
| bm25 | 0.5070 | 0.6610 | 0.8160 |
| dense (naive RAG) | 0.7000 | 0.8570 | 0.9320 |
| dense+rerank | 0.8400 | 0.9370 | 0.9620 |
| hybrid+rerank *(dense+BM25 RRF, no graph)* | 0.8430 | 0.9400 | 0.9650 |
| engram · Neo4j (**PPR**) | 0.8410 | 0.9360 | 0.9640 |
| **engram · engramdb (decay)** | 0.8410 | 0.9370 | 0.9670 |

Paired per-question (sign test + bootstrap 95%CI): engram − dense+rerank
ΔRecall@5 = 0.000, engram − hybrid+rerank = −0.003/−0.004 — **all n.s.** (sign-p
0.16–1.0, all > 0.05; ~490/500 queries tie). With a strong embedder the graph has
nothing left to recover, so it neither helps nor hurts significantly. And
**engramdb (decay, no PPR) ties Neo4j (PPR)** — ~2× faster ingest (122 vs 260 s),
~1.3× faster query — dropping PageRank costs nothing on multi-hop.

**Confirmed on MuSiQue** (harder, less-saturated multi-hop; bge-m3 stack, engramdb,
n=500): dense+rerank R@5 0.5910 / hybrid+rerank 0.5930 / engram 0.5920 — engram −
dense+rerank ΔR@5 +0.001 (n.s., 488/500 tie), engram − hybrid+rerank −0.001 (n.s.).
The graph-is-neutral-with-a-strong-embedder finding holds on a second multi-hop
dataset, so it is not a HotpotQA artifact.

### Floor stack (MiniLM + ms-marco, CPU)

| system | Recall@2 | Recall@5 | Recall@10 |
|---|---|---|---|
| bm25 | 0.5070 | 0.6610 | 0.8160 |
| dense | 0.5640 | 0.7320 | 0.8250 |
| dense+rerank | 0.7010 | 0.8240 | 0.9020 |
| **engram** | **0.7070** | **0.8410** | **0.9210** |

**The key finding — the graph's value depends on the embedder's strength:**

- At the **floor**, engram wins every recall and the margin over dense+rerank
  *grows with k* (+0.6 / +1.7 / +1.9 pts at @2/@5/@10) — the keyword graph +
  PageRank surface the bridge-linked second passage that a weak embedder misses.
- On the **production stack**, engram and dense+rerank are **tied** (±0.2 pts).
  BGE-M3 is strong enough to retrieve the linked passage by dense similarity
  alone (dense R@2 jumps 0.564 → 0.700), so graph expansion is largely redundant.

So engram's graph machinery is a **robustness floor**: it delivers a large lift
when the embedder is weak, and is neutral (never worse) when the embedder is
state-of-the-art. It never *loses* to the standard pipeline; it wins when the
retrieval is hard.

---

## Benchmark 3 — Agent-memory write-path (the unique-capability gate test)

engram's one *structurally-unique* capability is the **write-path**: it learns
from which chunks an agent actually used (`/feedback` + `mark_used`). A stateless
RAG pipeline cannot do this. So we implemented the learning side — an associative
**query→chunk memory boost** (a later query injects, into the rerank shortlist,
chunks that were *used* for ≥`min_sim`-similar past queries) — and tested whether
it improves retrieval. Protocol ([bench/memory_eval.py](memory_eval.py)): split the
test queries into a HISTORY session (record its gold chunks as feedback, with the
history query embeddings) and a held-out TEST set; compare WARM (memory on) vs COLD
(memory off) per test query, paired sign test + bootstrap CI. Production stack,
engramdb backend.

| dataset | subset | cold nDCG@10 | warm nDCG@10 | ΔnDCG@10 | verdict |
|---|---|---|---|---|---|
| NFCorpus | all 130 test | 0.3438 | 0.3431 | −0.0007 | n.s. |
| NFCorpus | memory-applicable (n=10) | 0.4233 | 0.4147 | −0.0086 | n.s. |
| SciFact | all 120 test | 0.7278 | 0.7278 | 0.0000 | n.s. |
| SciFact | memory-applicable (n=3) | 0.7103 | 0.7103 | 0.0000 | n.s. |

**Honest negative — and the reason is structural, not a bug** (the mechanism is
unit-tested and works):

1. **With a strong embedder, base retrieval already finds the gold chunk**, so
   re-injecting "what worked before" is redundant — exactly the graph's pattern.
   On SciFact the warm/cold rankings are *identical* (0/120 changed).
2. **In a pure retrieval benchmark, "used chunks = gold chunks = what base
   retrieval targets"** — so memory can only help where base retrieval *fails*,
   which a SOTA embedder rarely does. Memory has nothing to add that the embedder
   didn't already retrieve.
3. **BEIR queries are diverse one-shots, not the recurring workload agent-memory
   targets** — only 10/130 (NFCorpus) and 3/120 (SciFact) test queries even had a
   ≥0.7-similar past query sharing a gold doc.

**Conclusion for the release gate:** the agent-memory *boost* does **not** deliver
a measurable retrieval-quality edge on standard benchmarks with production models.
Its genuine value is **operational**, not benchmark-nDCG: cross-session persistence
of what worked, personalization, query-drift bridging, and latency/cost savings on
recurring queries — none of which a static IR benchmark can score, and none of
which is a "quality competitors can't match" claim we can *prove* here. It ships as
an **opt-in operational capability** (`MEMORY_BOOST_ENABLED`), honestly, not as a
quality differentiator.

---

## Headline

- **BEIR retrieval, production stack: engram beats naive vector RAG and BM25
  clearly, and is statistically *tied* with the standard dense+rerank and a strong
  hybrid+rerank baseline** (point estimate slightly ahead, but n.s. — 95%CI
  straddles 0, sign-p > 0.05; the edge over naive dense+rerank is mostly the BM25
  channel). engram's distinctive median/MMR/graph stages add no significant
  single-hop nDCG with production models.
- **Multi-hop: engram ties the best baseline** with a SOTA embedder (n.s. vs
  dense+rerank and hybrid+rerank, even with PPR — bge-m3 already retrieves the
  linked passage); the graph delivers a real lift only with a *weak* embedder
  (MiniLM +1.7/+1.9). It's a robustness floor, not a benchmark win.
- **The real, robust quality lever is the reranker** (+3–4 nDCG@10, every system);
  engram's genuine edges are **performance** (embedded engramdb, fastest backend,
  b1 32× memory) and the **operational/agent-memory layer**, not a benchmark crown.

## Caveats (so the numbers are trustworthy)

- The rigorous claim is the **controlled head-to-head** (identical models/data/
  metrics). We do **not** claim direct number-vs-number wins over LightRAG/
  HippoRAG's *published* figures — their setups use different corpora/embedders.
- Multi-hop uses the engram graph built from **YAKE** statistical keywords (no
  LLM). LLM/entity-based keyword extraction would link the graph more precisely
  and could widen engram's multi-hop margin even on a strong embedder.
- Multi-hop is the 500-question HotpotQA distractor-dev subset; BEIR uses the
  full test sets. Differences within ~±0.3 pts are noise at these sample sizes.

## Reproduce

```bash
# CPU floor
docker compose -f bench/docker-compose.yml run --rm --build runner python -m bench.compare
docker compose -f bench/docker-compose.yml run --rm --build multihop

# production stack on GPU (BGE-M3 + bge-reranker-v2-m3) — needs NVIDIA Container Toolkit
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm --build runner python -m bench.compare
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm --build multihop
```

Harnesses: [bench/compare.py](compare.py) (BEIR), [bench/multihop.py](multihop.py)
(HotpotQA). Config: [bench/docker-compose.yml](docker-compose.yml) +
[bench/docker-compose.gpu.yml](docker-compose.gpu.yml).
