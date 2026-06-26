# M1 — memory write-path (corrected plan)

engram appends chunks on ingest; it has no memory *write-path lifecycle*. This is
the gap vs dedicated agent-memory systems (Zep/Graphiti, Mem0, Letta) and the
reason the "agent memory" positioning is not yet earned. M1 adds a **no-LLM
deterministic core** plus a **strictly-offline, opt-in LLM enrichment tier**.

Design from a research+design+adversarial-critique workflow (Zep/Graphiti, Mem0,
Letta, Cognee, temporal-KG techniques). **The critique overturned the original
"first slice"** — recorded here so the corrected plan is the one we build.

## Hard constraints (the three guarantees)

1. **The retrieval hot path is touched at most once, additively, and provably.**
   Dedup/supersession work happens on the *write* path. Any read-path filter
   (e.g. validity) must answer the **ANN over-fetch** question first: `queryNodes`
   / pgvector HNSW return exactly *k* by ANN, so post-filtering shrinks the pool
   below *k* — over-fetch must be measured, it is not a free "no-op WHERE".
2. **Both stores stay at parity** (neo4j + pgvector) or degrade explicitly
   (`NotImplementedError`, like `upsert_entities`).
3. **Every component has a target on the judge-free `POST /eval` harness** or a
   small labelled set.

## Corrected component plan

- **C1 — `nearest_chunks(embedding, k, min_sim, exclude_doc_id)` Store method.**
  The dedup candidate primitive. Reuses the **existing content vector index** on
  both backends (the same call `vector_search` makes) → ANN top-k then filter
  `sim >= min_sim`. No new embedding, index, or schema. Parity-tested. **Build
  first; it unblocks everything and touches nothing on the hot path.**

- **C2′ — non-destructive near-duplicate *linking* (NOT merge-and-drop).** The
  critique killed the original C2: reference counting is **per-document**
  (`add_document_source` appends a source to a `Document`; a chunk is owned 1:1
  via scalar `c.doc_id`), so "route the dup's source to the matched chunk's doc"
  corrupts the reference graph and `DETACH DELETE` on re-ingest can delete a
  chunk another document needs. And dropping the incoming chunk **deletes user
  information** — the opposite of the "invalidate, never delete" rule the design
  otherwise follows. Instead: **always store the chunk**, record a
  `near_dup_of` backpointer (neo4j chunk prop / pgvector column) to the most
  similar existing chunk, and **collapse near-dup clusters at search time**
  (keep the best-scored member per cluster) so a paraphrase flood can't crowd
  distinct material out of the candidate pool. Non-destructive (both chunks
  survive → a false link is recoverable, not a factual deletion), never touches
  reference counting, gated by `dedup_enabled`. Measurable: `/eval` flood A/B
  (`tuning:{dedup_enabled}`) — pool stays flat, nDCG@10 within CI of the clean
  baseline; plus a labelled paraphrase/distinct set for *link* precision.
  **Note:** the 0.95 cosine threshold is **embedder-coupled** (BGE-M3 + engram's
  `passage_instruction`), not a portable constant — calibrate it on engram's own
  vectors, and watch the false-merge cases (boilerplate, templated records,
  near-identical-but-different numbers — exactly what the sparse channel protects).

- **C3 — transaction-time validity (`ingested_at`, `valid_to`) + current-valid
  read filter.** Additive props; default `valid_to IS NULL` = current. **Gated on
  answering the ANN over-fetch question** (prototype k-over-fetch, measure
  recall@k on a corpus with ~30% expired). `as_of` defaulting to "now" makes
  search **time-dependent** — reconcile with `/eval` reproducibility (pin `as_of`
  in eval). Do **not** call the no-LLM core "bi-temporal": with `valid_from =
  ingested_at` it is transaction-time only; event-time validity needs the LLM.

- **C4 — recency-wins supersession + `SUPERSEDES` edge** (structured-entity seam;
  `single_valued` predicates in `GraphProfile`). Invalidate-not-delete. NO-LLM
  for declared single-valued relations; the chunk-level supersede band is a
  heuristic that is **not safe without C5** (cosine can't tell supersede from
  elaborate from contradict) — keep it default-off. Decide explicitly whether
  `SUPERSEDES` edges enter the PPR projection (they should **not**, or a stale
  node lifts its own replacement).

- **C5 — LLM-optional contradiction confirmation, offline.** A new
  `app/consolidate.py` out-of-band job (like `scripts/build_communities.py`),
  never on the ingest/search path; default off; degrades to flag-only without the
  LLM. Gated on the RTX-4080 LLM coming online — do not claim until then.

## Build order

C1 → C2′ → (answer ANN over-fetch) → C3 → C4 → C5.

**Shipped:** **C1** (`nearest_chunks`, both backends, raw-cosine parity) and
**C2′** (`dedup_enabled`/`dedup_cosine_threshold`/`dedup_candidate_k`; ingest links
near-dups via `near_dup_of`; `Store.get_near_dup_links` batched read; search
`_collapse_near_dups` keeps the best-retrieved member per cluster;
`SearchResult.near_dup_of` provenance — non-destructive). 163 tests green. **Still
owed:** a `/eval` flood A/B on real BGE-M3 paraphrases to *quantify* C2′'s value
(it overlaps MMR on non-flood corpora; the win is the re-ingest-flood case). Then
C3 — but only after prototyping k-over-fetch and measuring recall@k on a ~30%-
expired corpus (the ANN post-filter shrinks the pool below k).

## Honest framing corrections (from the critique)

- Drop "Mem0 v3 lesson" (unverified provenance) and "bi-temporal" for the no-LLM
  core (it's transaction-time). Letta's dedup is agent-mediated, not automatic
  cosine — don't cite it for a no-LLM threshold.
- The win that's real and shippable today is **non-destructive near-dup linking
  with read-time suppression** + the `nearest_chunks` primitive. Everything
  temporal/contradiction is honestly later and partly LLM-gated.
