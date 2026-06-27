# Multi-tenant / metadata-filtered retrieval — design (needs sign-off before build)

The biggest **adoption** gap (every SaaS RAG needs tenant isolation; no benchmark
measures it). It is **security-sensitive** — a single unfiltered read path leaks
one tenant's documents into another's results — so this is specced for review
*before* implementation rather than rushed in a loop turn. Opt-in throughout:
with no `tenant_id`, behaviour is byte-identical to today.

## The hard requirement: isolation is all-or-nothing

Tenant isolation only holds if **every chunk-surfacing read filters by tenant**.
Missing one = a leak. The complete list (both backends):

1. `vector_search` — the dense channels
2. `fulltext_search` — the BM25 channel
3. `fetch_siblings` — **graph expansion can cross tenants via shared keywords** (a
   `HAS_KEYWORD` sibling may belong to another tenant) — easy to miss, critical
4. `nearest_chunks` — ingest-time dedup must not link across tenants

(Post-filtering the final pool is *not* sufficient: other tenants' chunks would
occupy the per-channel top-k, starving the real tenant's recall. The filter must
be inside each read.)

## Tenant model (decision)

- `tenant_id: str | None` per **document** at ingest, propagated to its chunks
  (`c.tenant_id` / a `tenant_id` column) for read filtering. Default `None` =
  untenanted (current behaviour). *(Open: per-document vs per-source — a source
  can already span documents; per-document is simpler and matches the chunk owner.)*
- `tenant_id` on `SearchRequest` → threaded through the four reads above.

## The ANN over-fetch problem (the real engineering risk)

`db.index.vector.queryNodes(idx, k, emb)` (Neo4j) and pgvector HNSW return the
top-k **before** a tenant filter, so post-filtering yields `< k` for the tenant.
- **Neo4j:** over-fetch `k * OVERFETCH` from the index, then `WHERE c.tenant_id`,
  take k. `OVERFETCH` (e.g. 8) trades recall vs cost; if a tenant is a tiny slice
  of a huge shared corpus, even that under-fills — document it, and consider a
  per-tenant index/label later.
- **pgvector 0.8+:** `SET hnsw.iterative_scan = strict_order` lets the HNSW keep
  scanning until k filtered rows are found — cleaner than over-fetch. Needs a
  btree index on `tenant_id`.

## Storage

- Neo4j: `c.tenant_id` property + an index `FOR (c:Chunk) ON (c.tenant_id)`.
- pgvector: `tenant_id TEXT` column + btree index. (Additive, like `near_dup_of`.)
- Not in the schema signature (doesn't change embedding geometry).

## Metadata filters (phase 2, after tenant_id)

Arbitrary `metadata` filters (date/source/attributes) are clean on **pgvector**
(`JSONB` + `@>` / `->>`), but **awkward on Neo4j** (a nested map isn't a filterable
property; each key would need its own property). So ship **tenant_id first**
(one indexed scalar, symmetric on both backends); treat arbitrary metadata as a
follow-up, pgvector-first, with Neo4j supporting a documented reserved set.

## Acceptance test (the security gate)

- **0% cross-tenant leakage:** ingest docs for tenant A and tenant B; assert
  `search(tenant_id=A)` NEVER returns a B chunk — across all four read paths
  (include a query whose top dense hits AND keyword-siblings are B's, to exercise
  vector + fulltext + sibling filtering).
- **Recall guard:** filtered Recall@10 within ~1 pt of an exhaustive in-tenant
  scan (proves the over-fetch/iterative-scan is deep enough).
- **No-op guard:** with no `tenant_id`, BEIR nDCG@10 + latency unchanged.

## Effort

M–L (every read path × 2 backends + ingest + over-fetch + the isolation test).
Decision points for sign-off: (1) per-document tenant model OK? (2) `OVERFETCH`
factor / require pgvector 0.8 iterative scan? (3) tenant-only now, arbitrary
metadata later? On sign-off it's a careful, well-tested build — not a rush.
