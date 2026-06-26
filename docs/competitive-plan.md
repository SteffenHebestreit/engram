# engram competitive plan — decision-ready (post sparse/ColBERT)

> **Shipped since this plan was written (2026-06-26):** **E1** (asymmetric
> query/passage instruction prefixes), **H** (`POST /eval` — judge-free IR metrics
> + bootstrap CIs + per-channel gold-hit attribution; `app/eval.py`), and **M1·C1**
> (`nearest_chunks` dedup primitive, both backends). **F3 (sparse candidate-
> expansion) was DROPPED on evidence:** `/eval` attribution shows the dense
> channel surfaces *all* SciFact gold hits (fulltext/graph are redundant subsets),
> so a sparse *retrieval* channel has no headroom — sparse's value is re-ranking
> (the shipped +1.1 nDCG/+1.8 recall@10), not new recall. Remaining sequence:
> **M1 memory write-path** (C2′ non-destructive dedup → temporal) is the live
> moat; multi-hop tie→win is at risk of the same "graph redundant on a strong
> embedder" finding and is gated on a non-saturated benchmark.

*Status: written after the learned-sparse channel and ColBERT MaxSim reranker
shipped. This supersedes the feature-by-feature framing of
[`competitive-roadmap.md`](competitive-roadmap.md): it is the prioritized,
honest build plan derived from adversarial verification of 16 proposed features
plus a completeness review. Every "win" target below is stated against engram's
**own** harness — no borrowed numbers.*

---

## 1. Executive thesis

engram wins not by out-tuning fusion scores around a frozen embedder — its own
controlled benchmarks prove that graph value shrinks and multi-hop ties as the
embedder strengthens, and the terminal cross-encoder washes out most fusion
tweaks before they reach the agent — but by owning the two layers nothing else
in the category owns end-to-end: a **memory write-path** (semantic dedup,
contradiction/supersession, temporal validity) that turns "graph-augmented RAG"
into defensible **agent long-term memory**, and a **reproducible, judge-free,
per-stage eval harness** over its own provenance that makes every retrieval
claim falsifiable while competitors ship only LLM-judge win-rates. The highest-
leverage near-term moves are the ones that improve the **embedding geometry
itself** (asymmetric query/passage prefixes, per-corpus projection head) and
**scale** (binary/int8 two-stage retrieval) — the layers that actually set
result quality and adoptability — gated behind a **ship-first adaptive router**
so every heavy mode is added without taxing the factoid hot path. Net: engram
becomes the retrieval tool an agent reaches for because it is the only one that
remembers correctly, proves its own quality, and stays fast at memory scale —
not because it has one more fusion knob.

---

## 2. Prioritized feature table

**Legend** — Advantage: **CA** = clear advantage (category-defining or no
competitor parity), **TS** = table-stakes/parity (adoption gate, not a moat),
**INFRA** = enabling plumbing that unlocks other wins, **FIX** = correctness fix
(not customer-facing). LLM = needs the (not-yet-online) local generative LLM.

| # | Feature | Beats whom (honestly) | engram implementation (seam) | Measurable target (engram's harness) | Effort | LLM | Advantage |
|---|---------|----------------------|------------------------------|--------------------------------------|--------|-----|-----------|
| **R0** | **Adaptive query router** (factoid / multi-hop / global) | HippoRAG-2's "no simple-QA regression" — as a *system property*, not per-feature | Deterministic lexical/heuristic classifier at top of `search()`; entity/ID density from the shipped sparse channel; dispatch factoid→today's pipeline byte-identical | Misroute rate: false-positive global/multi-hop **≤2%**; BEIR routes single-pass 100% (nDCG@10 bit-identical) | S | No | INFRA (load-bearing) |
| **E1** | **Asymmetric query/passage instruction prefixes** | Every competitor shipping a frozen embedder; closes a measurable BGE/E5/GTE recall loss engram currently eats | One-line prefix in `embeddings.py`/channel embed call (`query:` vs `passage:` style task instruction) | nDCG@10 strictly up on SciFact **and** NFCorpus vs no-prefix, paired-bootstrap CI excluding 0 | S | No | CA (north-star, nobody proposed it) |
| **M1** | **Memory write-path lifecycle** — semantic dedup + contradiction/supersession + temporal validity | Mem0, Zep/Graphiti, Letta/MemGPT on the consolidation layer they lead with | `ingest.py`: cosine near-dup check over `content_embedding` (same tenant/source); `SUPERSEDES`/`CONTRADICTS` edges + `valid_from/valid_to` chunk props (neo4j props / pgvector JSONB) | Dedup: re-ingesting N paraphrases yields 1 canonical + N-1 linked (not N pool floods). Contradiction recall on a labelled conflict set ≥ target; zero factoid-path change | L | No (dedup/recency); LLM optional for conflict-confirm | CA (category-defining; the memory banner engram doesn't yet earn) |
| **H** | **First-party eval harness + per-stage attribution** (was F7+F9+F8) | RAGAS/TruLens/ARES *only on reproducibility/judge-freedom* — NOT on answer-gen (category error to claim otherwise) | `app/eval/` + `GOLDEN_SETS` registry; retain per-query metric vector (currently summed away); paired-bootstrap CIs (numpy, B=1000); attribute each gold hit to the channel/stage that first surfaced it via existing `SearchResult` provenance | Re-derive sparse channel's ~+4.0 nDCG@10 SciFact lift from the harness itself (proves attribution math); report HotpotQA graph-expansion marginal recall@10 as a concrete number (may be ~0 — print it honestly) | M | No | INFRA → CA (judge-free per-stage attribution competitors can't produce) |
| **E2** | **Per-corpus linear projection head** (fit on golden set) | Any competitor who doesn't own the corpus / a contamination-free labelled set | Fit 1024×1024 (or MRL-truncated) linear map on F16 golden pairs; persist keyed by schema signature; apply in `embeddings.py` before indexing | Held-out (train/test split) nDCG@10 ≥ **+1.0** over no-head, paired-bootstrap CI excluding 0; no SciFact regression | M | LLM (for golden-set gen) | CA (improves geometry, not scores-around-it) |
| **Q1** | **Binary/int8 two-stage retrieval** (binary shortlist → full-precision rescore) | Weaviate / Qdrant / Vespa on scale/cost (invisible in BEIR-small) | Store binary view alongside `content_embedding`; shortlist on Hamming/dot, rescore shortlist with full-precision before fusion+rerank | ~99% recall@k retained vs full-precision on SciFact/NFCorpus; report memory + p50/p95 win at large N | M | No | CA at scale / TS feature-wise |
| **F13** | **Metadata-filtered, multi-tenant, recency-aware retrieval** | Weaviate/Vespa/pgvector/Qdrant — reaches **parity** (admission, not a moat) | `metadata: dict` (reserved `tenant_id`,`date`) on ingest; `filters`/`as_of` through Store protocol; neo4j = honest over-fetch-then-WHERE; pgvector = real indexed JSONB WHERE; recency term in `fused_score` | **0% cross-tenant leakage** invariant (the real claim); filtered Recall@10 within 1pt of exhaustive in-tenant scan (over-fetch guard); unfiltered p50/p95 unchanged | M | No | TS (SaaS adoption gate) |
| **F5** | **Surface dormant community layer into `/search`** (auto-routed global mode) | GraphRAG global / LightRAG high-level path — by construction (no dense+rerank system has it) | Route (via R0) global cues → existing `search_communities()`; return community reports as `SearchResult` with `community_score` provenance; reuse `/communities/search` | Router FP-global ≤2%; BEIR nDCG@10 bit-identical; build an in-repo BenchmarkQED AutoQ subset; report fixed-cost prebuilt reports vs GraphRAG map-reduce LLM cost | M | LLM (for report *quality*; wiring is LLM-free) | CA (capability) — quality gated on reports |
| **F6** | **Iterative multi-hop loop** (decompose → retrieve → reseed graph) | PRISM-style multi-hop systems — *only where single-pass has headroom* | `POST /search/multihop` wrapping `search()` as inner retriever; per-subquery provenance; short-circuit to single `search()` when decomposer returns 1 sub-q | **DROP HotpotQA (saturated)**; on MuSiQue/2Wiki 3-4 hop where single-pass <0.7: ≥**+5 Recall@5** vs dense+rerank; ablate decompose-only vs decompose+reseed | L | **LLM** | CA (new capability) — UNVALIDATED until LLM online |
| **F4** | **Late chunking content channel** | Contextual-Retrieval / Jina late-chunking — *free* variant | Real per-token `last_hidden_state` pooling endpoint in BGE-M3 sidecar (NOT `return_dense`/colbert); macro-window stitching; fix reuse key to `(chunk_text, neighbor_context_hash)` | Multi-chunk corpus (NFCorpus/FiQA) ≥**+1.0** nDCG@10; SciFact non-regressive (≤0.2) | L | No | refine→probe-gated (KILL if probe fails) |
| **F15** | **Contextual chunk prefixing** (doc-situating blurb → channel + contextual BM25) | Anthropic Contextual Retrieval — *reproduce direction, not their numbers* | `contextual` EXTRACTOR + `context` CHANNEL_SOURCE; store on **separate** `context_text` (indexed/embedded, **never** in `c.text` → reranker + agent context stay clean) | Must beat **reweighted summary channel** (not just baseline) by ≥+1.0 nDCG@10 single-hop / +2 recall@2 HotpotQA; factoid p50/p95 ≤+15% | M | **LLM** | refine — parity-plus unless it beats summary channel |
| **F14** | **Layout-aware ingestion** (PARSERS registry, page/bbox citations, table channel) | RAGFlow DeepDoc / Morphik / R2R — **parity** gap-closer | (1) page/bbox on `SearchResult`/`ContextChunk` (low-risk, ship first); (2) `PARSERS` registry, default `passthrough`=identity (text path byte-identical); table units capped to embedder context, excluded from median-proximity outlier penalty | Citations: 100% PDF chunks carry resolvable page (binary). Table parser: ≥+3 recall@10 on FinQA/TAT-QA-derived set; SciFact/NFCorpus nDCG@10 delta <0.003 | L→XL | No | TS (enterprise gate) |
| **FB** | **Implicit-relevance feedback loop** (capture grounded chunks → tune + mine hard negatives) | Any stateless retrieval lib — unique agent-in-the-loop data moat | Optional `mark_used(query_id, chunk_ids)` MCP tool/endpoint; log via `observability.py`; offline batch re-fits fusion weights + exports hard negatives | Self-populating golden set per corpus; learned-weight refit shows held-out nDCG@10 gain (reuses F10 gate) | M | No | CA (compounding flywheel) |
| **F10** | **Fit-on-dev-set fusion weights** (offline fitter ONLY) | engram's own hand-tuned defaults — parity-plus tuning | Coordinate-search existing channel weights; **DROP** the "weighted_sum strategy" (mathematically identical to shipped `dbsf_convex`) | Held-out split, **reranked** nDCG@10 ≥**+0.5** vs defaults, non-overlapping CI; KILL if only shortlist-recall moves | S | No | refine — tuning, not architecture |
| **F1** | **Calibrated graph-vs-dense fusion** | — (internal correctness, NOT a competitive headline) | DBSF-normalize / rank-transform `graph_proximity` onto channel scale before convex add; `FUSIONS` entry `calibrated_dbsf` | **GATE:** SciFact ≥0.741 & NFCorpus ≥0.341 (hard, else KILL); multi-hop Recall@2 > tie with bootstrap CI; reranker-OFF ablation proves mechanism | S | No | FIX (scale-mismatch bug) |
| **F11** | **IDF edge weighting on HAS_KEYWORD** | — (graph hygiene; activates unused `RelationSpec.weight`) | Thread weight into `projection_spec()`; `relationshipWeightProperty` in `gds.pageRank`; summed-IDF sibling strength | Converts production-stack TIE to >+0.3 (beyond ±0.3 noise) w/o SciFact/NFCorpus regression; else internal cleanup only | S | No | FIX — likely no-op on strong embedder |
| **F12** | **Parent-child / auto-merge** (presentation-only) | LlamaIndex auto-merging — **parity** | **NOT** the score-collapsing expander (regresses nDCG, fights MMR). Post-rerank `merged_context` field only; ranking/scores untouched | Ranked-id lists byte-identical on 3 benchmarks (regression guard); Coverage@k lift only on a new long-doc span set | M | No | TS — keep as MCP `get_chunk_context` capability unless long-doc harness built |
| **F2/F3** | **Filtered PPR / keyword-bridge candidate expansion** | HippoRAG-2 multi-hop — *only if it ADDS missed bridges* | F3 reframed to **candidate expansion** in `search.py` (pull HAS_KEYWORD chunks NOT in pool BEFORE fuse), gated by router; F2 = precision-gate keyword bridges, NEXT_CHUNK untouched | Production-stack Recall@2 ≥**+1.0** over current engram on ≥2 multi-hop sets, CI-separated; attribution control vs `graph_proximity_weight=0`; KILL if filtered-PPR gain ≈ unfiltered | M | No | refine — low leverage as originally specced |
| **F16** | **Graph-grounded synthetic golden-set generator** | RAGAS test-gen on contamination-freedom — *only if decircularized* | **Do NOT** sample from engram's own HAS_KEYWORD/NEXT_CHUNK graph (circular). Use orthogonal bridge signal; add **hop-necessity gate** (closed-book LLM judge: neither chunk alone answers, pair does) | On gated set: graph-on minus graph-off Recall@2 ≥+5, CI excluding 0, ≥300 questions. ~0 is a valid honest negative | L | **LLM** | refine — diagnostic, not a self-won benchmark |

---

## 3. Recommended build sequence

The recurring failure mode across the verdicts is identical: a feature
rearranges scores **upstream of a terminal cross-encoder that overwrites them**,
on a **saturated benchmark**, validated with a **borrowed number** and a
**not-yet-online LLM**. The sequence below is ordered to (a) attack the layers
the reranker does *not* overwrite — embedding geometry and the reranker/ranking
itself — (b) prefer wins benchmarkable **without** the local LLM, and (c) ship
the safety harness before anything that could tax the factoid path.

**Phase 0 — Safety + measurement (ship before everything; no LLM)**
1. **R0 adaptive router.** Load-bearing. Every heavy mode (F5/F6/F3/F15) is only
   safe to add behind it. Ship gate = ≤2% misroute, BEIR routes single-pass 100%.
2. **H eval harness + per-stage attribution.** Nothing below is a "win" until
   it is falsifiable on engram's own qrels with a bootstrap CI. H also unlocks
   F10 (learned weights) and E2 (projection head). Re-deriving the sparse
   channel's +4.0 lift from the harness validates the attribution math.

**Phase 1 — THE SINGLE HIGHEST-LEVERAGE NEXT FEATURE (no LLM):**
> ### ➤ E1 — Asymmetric query/passage instruction prefixes.
> This is the recommendation. It is the **only** lever that improves the
> embedding **geometry** (which feeds `content_embedding` → median-proximity →
> MMR → every channel and the candidate set the reranker sees), it is an
> **S-effort one-line** change, it needs **no LLM**, it is provable **today** on
> the existing SciFact/NFCorpus harness, and it directly serves the stated north
> star ("best embeddings") that all 16 features otherwise ignore. engram embeds
> query and passage identically today — a measurable BGE-M3 recall loss it is
> currently eating for free. Highest leverage-to-effort ratio in the entire plan.

3. **E1 prefixes** → then **Q1 binary/int8 two-stage** (no LLM, scale moat,
   provable recall-retention today) → then **F1 + F11** (cheap correctness/
   hygiene fixes, hard no-regression gates) → **F10 offline fusion fitter**
   (now that H exists to fit/test on a split).

**Phase 2 — Toward the multi-hop TIE→WIN goal (no LLM where possible):**
4. **F3 keyword-bridge candidate expansion** (router-gated, ADDS missed bridges
   pre-rerank — the only structurally-correct multi-hop lever that needs no LLM)
   + **F2 bridge precision-gate**. Validate on MuSiQue/2Wiki, **not** saturated
   HotpotQA, with the `graph_proximity_weight=0` attribution control. If the
   graph half is redundant on the strong embedder (RESULTS.md predicts this),
   report the honest negative and pivot multi-hop budget to E2/F6.
5. **F13 metadata/multi-tenant + recency** (table-stakes SaaS gate; no LLM;
   0%-leakage correctness proof) — unblocks paying adoption independent of the
   quality story. Pairs with Q1 (filtered+quantized = the real large-tenant path).

**Phase 3 — Memory-native moat (no LLM for the core):**
6. **M1 write-path lifecycle.** The category-defining gap. Dedup + recency need
   no LLM; contradiction-confirm can use the local LLM later. This is what makes
   the "engram = agent memory" banner true rather than aspirational, and no
   amount of embedder work substitutes for it.
7. **FB feedback loop.** Turns deployment traffic into a self-improving flywheel
   (online fusion adaptation + hard-negative mining) that feeds E2 and F16.

**Phase 4 — Gated on the RTX-4080 local LLM (do NOT claim until it ships):**
8. **E2 projection head** (fit on **F16** golden pairs — closes the loop to the
   north star), **F6 multi-hop loop**, **F15 contextual prefixing** (must beat
   the existing summary channel, not baseline), **F5 global-mode quality**
   (reports), **F16 generator** (decircularized + hop-necessity-gated), **F4
   late chunking** (sidecar probe-gated), **F14 layout/citations** (ship the
   page/bbox citation half early — it needs no LLM — defer the table parser).

**Honest framing rule for every doc/README:** mark **TS** features as parity
that unblocks adoption, **FIX** as correctness, and reserve **CA** claims for
M1, H/attribution, E1/E2, Q1-at-scale, FB, and the capability (not yet quality)
of F5/F6 — and never print a `clear_advantage` line backed only by a borrowed
number or an unvalidated (LLM-offline) feature.

---

## 4. Completeness-critic additions incorporated

The critic surfaced a structural blind spot the 16 features and the adversarial
verdicts all missed: **everything freezes BGE-M3 and rearranges scores around a
terminal reranker, on a saturated benchmark, under a memory banner with no
memory write-path.** The following survived scrutiny and are now first-class:

- **Embedder-as-tunable-surface → E1 (prefixes), E2 (projection head).** Kept:
  these are the only levers touching geometry rather than scores-around-it, and
  RESULTS.md's own finding (embedder is the dominant variable) makes them
  strictly higher-leverage than another fusion tweak. *Swapping to a SOTA-2026
  embedder (Qwen3-Embedding/NV-Embed/Stella)* is noted as the highest single
  lever but deferred to an evaluation task behind H, since it risks shrinking the
  graph's value further — measure before adopting.
- **Memory write-path lifecycle → M1.** Kept as the top strategic correction.
  This is the category gap vs Mem0/Zep/Letta and the reason the "memory" banner
  is currently unearned.
- **Quantization/scale → Q1.** Kept: a real cost/scale moat invisible to
  BEIR-small, essential for *unbounded* agent memory.
- **Implicit-relevance feedback → FB.** Kept: the unique agent-in-the-loop data
  moat; makes H/F16 self-populating and FB's labels are the *real* qrels F9's
  attribution wanted.
- **Router-first → R0.** Kept and **promoted from Tier-5 "later" to Phase 0**.
  The verdicts independently demanded it for F3/F5/F6/F15; it is the shared
  safety harness, not a deferred line item.
- **Reranker is terminal / alternative rerankers.** Noted as the recurring
  wash-out cause; *benchmarking Cohere/Jina/Voyage/listwise rerankers* is added
  as an H-harness evaluation (engram already has the ColBERT MaxSim seam) since
  the reranker is the one stage that sets final order — but no feature should
  claim a fusion-layer win without reporting the post-rerank metric.
- **Deferred deliberately:** LinearRAG-style "delete the recognition LLM" is
  folded into the F2/F3 *LLM-free* multi-hop framing rather than a standalone
  feature; query/embedding/semantic caching is acknowledged as table-stakes
  latency plumbing (an ops task, not a differentiator) and tracked outside this
  competitive plan.
