# Customizing engram

engram is built so you can adapt it to a special use case **without forking
the pipeline**. The search/ingest flow is a fixed skeleton that selects swappable
strategies by config key; every strategy ships a default that reproduces the
documented behaviour, so the scoring math never moves until you opt in.

There are two levels:

1. **Config-as-data**: change weights, the channel set, or the graph profile
   with environment variables / JSON. No code.
2. **Code plugins**: register a new strategy (chunker, extractor, fusion,
   expander, proximity, channel source) on its registry. A few lines.

---

## The registry pattern

Each pluggable stage owns a `Registry` ([app/registry.py](../app/registry.py)).
A built-in default registers under a string key; `Settings` selects one by key.

```python
from app.chunking import CHUNKERS

@CHUNKERS.register("semantic")
def semantic_chunker(text, settings):
    # settings is the full Settings object, so read your own knobs from it
    ...
    return ["chunk one", "chunk two"]
```

Then set `CHUNK_STRATEGY=semantic`. Selecting an unregistered key raises a clear
error listing what *is* registered.

Register at import time. To ship strategies in a separate package, expose an
`engram.plugins` entry point whose callable performs the registrations;
`registry.load_entrypoints()` imports them at startup (a broken plugin is
skipped, never fatal).

---

## The seams

| Stage | Registry | Config key | Default | File |
|---|---|---|---|---|
| Chunking | `CHUNKERS` | `CHUNK_STRATEGY` | `fixed` | [app/chunking.py](../app/chunking.py) |
| Metadata extraction | `EXTRACTORS` | `METADATA_EXTRACTOR` | `yake` (no-LLM; `default`=LLM, `none`) | [app/llm.py](../app/llm.py) |
| Channel embed-source | `CHANNEL_SOURCES` | (per channel `source`) | `text`/`summary`/`keywords` | [app/channels.py](../app/channels.py) |
| Score fusion | `FUSIONS` | `FUSION_STRATEGY` | `dbsf_convex` | [app/pipeline.py](../app/pipeline.py) |
| Graph expansion | `EXPANDERS` | `EXPANDER_STRATEGY` | `sequence_keyword` | [app/pipeline.py](../app/pipeline.py) |
| Graph proximity | `PROXIMITIES` | `GRAPH_PROXIMITY_MODE` | `decay` (`ppr` opt-in) | [app/pipeline.py](../app/pipeline.py) |
| Reranker | `RERANKERS` | `RERANKER_STRATEGY` | `http` | [app/rerank.py](../app/rerank.py) |

### Chunker
`(text: str, settings: Settings) -> list[str]`. The default is a
paragraph/sentence-aligned fixed window with overlap.

### Metadata extractor
`async (client, chunk) -> ExtractionResult` exposing at least `.summary` and
`.keywords` (those feed the summary/keywords channels). The default `yake` is
**LLM-free** — statistical keywords + the chunk's lead sentence — so ingest
makes no per-chunk chat call; `default` is the opt-in LLM upgrade (sharper
abstractive gist, one call per fresh chunk). A domain extractor might pull named
entities or code symbols instead.

**A separate small/fast model for the LLM extractor.** Extraction is the
high-volume, low-difficulty LLM call, so when you run `METADATA_EXTRACTOR=default`
it pays to point it at a *small* model independent of the strong one used for
HyDE/contextual/community: set `EXTRACTION_LLM_API_BASE` / `EXTRACTION_LLM_MODEL`
/ `EXTRACTION_LLM_API_KEY` (each falls back to its `LLM_*` counterpart when blank).
Make it reliable + cheap across serving stacks:

- `EXTRACTION_RESPONSE_FORMAT=json_schema` sends the `{keywords,summary}` schema —
  **guaranteed** valid JSON via constrained decoding, and it *suppresses
  chain-of-thought* on servers that gate it (also **required by LM Studio**, which
  rejects `json_object`).
- `EXTRACTION_EXTRA_BODY` merges serving-specific reasoning controls at the request
  top level, stack-agnostically — e.g. `{"chat_template_kwargs":{"enable_thinking":false}}`
  (vLLM Qwen3) or `{"reasoning_effort":"none"}` (Ollama/LM Studio). Thinking models
  (Qwen3, Gemma 4) otherwise waste tokens reasoning on this trivial task.
- `EXTRACTION_MAX_TOKENS` caps the (tiny) output; `EXTRACTION_MIN_CHARS` skips the
  LLM for short chunks (titles/headers), falling back to yake.

Compare candidate extraction models on throughput **and** quality with
[bench/extractor_bench.py](../bench/extractor_bench.py); pick embedders/rerankers
against a live stack with [bench/live_eval.py](../bench/live_eval.py).

### Fusion
`(channel_hits, channels, fulltext_hits, fulltext_weight, settings) -> dict[id, candidate]`.
The default DBSF-normalizes each channel then convex-combines them. An
alternative could implement Reciprocal Rank Fusion.

### Expander
`async (store, seed_ids, settings) -> list[sibling]`. The default asks the store
for `NEXT_CHUNK` siblings both ways and shared-keyword siblings. A custom
expander could walk typed domain relations (see *Special graphs* below).

### Proximity
`async (store, seed_ids, siblings, settings) -> list[float]` parallel to
`siblings`. Built-ins: `decay` (fixed per-hop fade — the **default**) and `ppr`
(graph-activation proximity — Personalized PageRank on the neo4j backend, with a
per-sibling decay fallback when the store reports no proximity capability). PPR is
opt-in: benchmarks showed it matches decay on quality while being far more
expensive at scale (see [engram-db.md](engram-db.md)).

### Reranker
`async (client, query, texts) -> list[float] | None`, one score per text in
input order; return `None` to signal "unavailable" and the pipeline falls back
to the fused score. The default `http` strategy calls a cross-encoder endpoint
(`tei`/`jina` wire formats). Set `RERANKER_ENABLED=false` to skip the
cross-encoder round trip entirely (same fused-score fallback).

---

## Storage backends

The whole data layer sits behind a `Store` protocol ([app/store.py](../app/store.py)),
selected by `STORE_BACKEND` through the `STORES` registry — same pattern as the
pipeline seams, so a third-party backend can register via `engram.plugins`.

| Backend | `STORE_BACKEND` | What you get |
|---|---|---|
| Neo4j | `neo4j` (default) | graph + vector in one store; GDS Personalized-PageRank proximity; structured-entity ingest; community synthesis |
| pgvector | `pgvector` | PostgreSQL + pgvector: vector (HNSW/cosine) + fulltext (`tsvector`) + sequence/keyword siblings |
| Engram-DB | `engramdb` | **Embedded, no server** ([app/store_engramdb.py](../app/store_engramdb.py)): in-process vector + BM25 + **native-adjacency** graph (NEXT_CHUNK / keyword) + decay. Purpose-built from the evaluation — only the parts that pull their weight, none of the overhead. Prototype: brute-force ANN + optional pickle snapshot (`ENGRAMDB_PATH`); **no** PPR/GDS, community synthesis, or structured-entity graph. See [engram-db.md](engram-db.md). |

All share the exact same retrieval pipeline and scoring math; they differ only in
capabilities a given backend can't (or deliberately doesn't) provide:

- **No graph-activation proximity.** pgvector has no GDS PageRank, so
  `graph_proximity` returns `None` and the `ppr` strategy transparently falls
  back to per-hop **decay** (the same path used when the GDS plugin is missing).
  Sequence (`NEXT_CHUNK` → `seq ± hops`) and keyword (`HAS_KEYWORD` → a
  `chunk_keywords` join table) siblings still work.
- **No structured-entity graph.** `POST /graph/entities` and `/graph/relations`
  are neo4j-only; on pgvector they return `501 Not Implemented`.

Run the pgvector backend:

```bash
# bring up Postgres (pgvector image) alongside the API
docker compose --profile pgvector up -d postgres
# point the API at it (in .env, or inline)
STORE_BACKEND=pgvector docker compose up -d --build api
```

`docker compose run --rm tests` starts both Neo4j and Postgres, so the neo4j and
pgvector integration tests both run. A custom backend implements the `Store`
protocol and registers a factory:

```python
from app.store import STORES

@STORES.register("mybackend")
def _make(settings):
    return MyStore(settings)        # implements the Store protocol
```

---

## Channels

A *channel* is one independently-embedded view of a chunk with its own vector
index and fusion weight. The default three (content / summary / keywords) are
built from the `*_CHANNEL_WEIGHT` settings. Override the whole set with the
`VECTOR_CHANNELS` JSON env var:

```json
[
  {"name": "content", "index": "chunk_content_idx", "embedding_prop": "content_embedding", "source": "text",    "weight": 1.0},
  {"name": "title",   "index": "chunk_title_idx",   "embedding_prop": "title_embedding",   "source": "title",   "weight": 0.6}
]
```

- `source` is a `CHANNEL_SOURCES` key: register a new one to embed a field your
  custom extractor produces (e.g. `title`).
- The content channel must keep `content_embedding`: that property is the
  canonical geometry vector used by median-proximity and MMR.
- Changing the channel set changes the **schema signature** (see below); wipe
  and re-ingest, or the guard will stop you.

**Incremental re-ingest.** Re-ingesting a document (same id, edited text)
replaces it, but `REUSE_UNCHANGED_CHUNKS` (default on) makes any chunk whose
text is byte-identical to one in the previous version reuse its stored vectors
and metadata — so only the chunks that actually changed pay for fresh LLM
extraction + embedding. Localized edits cost re-embedding only the chunks they
touch; a full reflow that shifts every chunk boundary still re-embeds everything.

**Cheaper ingest.** The default `METADATA_EXTRACTOR=yake` is already LLM-free at
ingest (0 LLM calls), while keeping the keyword graph and a lead-sentence
summary. To go all the way to the naive baseline — *one* embedding per chunk —
also drop the summary/keywords channels with `SUMMARY_CHANNEL_ENABLED=false` /
`KEYWORDS_CHANNEL_ENABLED=false` (and optionally `METADATA_EXTRACTOR=none`). The
trade-off vs the yake default: no keyword-sibling graph expansion and no summary
in the fulltext index, since both derive from that metadata — so content-only is
opt-in, not the default.

### Contextual Retrieval

`CONTEXTUAL_RETRIEVAL_ENABLED=true` turns on [Anthropic's Contextual
Retrieval](https://www.anthropic.com/news/contextual-retrieval): at ingest, an
LLM writes a short *document-situating context* for each chunk (which entity,
section, time period or topic it belongs to) and prepends it to the chunk
**before embedding**. The content vector then encodes document-level identity
instead of just the bare passage, so near-identical chunks from different
documents stop colliding — a change to the embedding **geometry**, the one layer
a reranker can't overwrite.

It is complementary to engram's `NEXT_CHUNK` expansion, not redundant with it:
the graph recovers *neighbour* context at read time, while this bakes *doc-level
identity* into the vector at index time.

Cost and caveats:
- One extra LLM call per fresh chunk at ingest (reused/unchanged chunks keep the
  context already baked into their stored vector). The whole document is sent as
  the call's prefix — shared across that document's chunks — so providers that
  cache prompt prefixes amortize it; `CONTEXTUAL_MAX_DOC_CHARS` bounds the prefix
  for very large documents.
- Degrades safely: if the LLM is unavailable the chunk is embedded bare, exactly
  as with the feature off — ingest never breaks.
- Changes the stored content vectors, so it is part of the **schema signature**:
  enable it on a fresh store (or wipe + re-ingest), or the guard will stop you.
- **Contextual BM25 too:** the context is also stored and indexed for fulltext
  (Neo4j: indexed alongside `text`/`summary`; pgvector: a separate `context_tsv`),
  so the lexical channel benefits as well — Anthropic's larger reported gain. This
  part is additive and unconditional (the context is empty when the feature is
  off, so a non-contextual store's fulltext results are unchanged).

---

## Special graphs (beyond documents)

engram's graph is normally `Chunk`/`Keyword` over `NEXT_CHUNK`/`HAS_KEYWORD`.
For a domain graph (typed entities with their own relations) there are two
pieces:

### 1. Load the structured graph

```bash
# entities: typed nodes identified by label + key
curl -X POST localhost:8088/graph/entities -H 'Content-Type: application/json' -d '{
  "label": "EbmCode",
  "items": [
    {"key": "03220", "properties": {"text": "Chronikerpauschale"}},
    {"key": "03100", "properties": {"text": "Notfallpauschale"}}
  ]
}'

# relations: typed edges between existing nodes
curl -X POST localhost:8088/graph/relations -H 'Content-Type: application/json' -d '{
  "from_label": "EbmCode", "type": "EXCLUDES_SAME_QUARTAL", "to_label": "EbmCode",
  "items": [{"from_key": "03100", "to_key": "03220"}]
}'
```

Labels and relationship types are sanitized to `[A-Za-z0-9_]`; keys and
properties travel as query parameters.

### 2. Declare them in a graph profile

A `GRAPH_PROFILE` ([app/profiles.py](../app/profiles.py)) lists the node labels
and relationships the GDS projection should span, so Personalized PageRank can
spread activation through your entities:

```json
{
  "name": "mfa",
  "projection_labels": ["Chunk", "EbmCode"],
  "projection_relationships": [
    {"type": "NEXT_CHUNK"}, {"type": "HAS_KEYWORD"},
    {"type": "ABOUT"},
    {"type": "EXCLUDES_SAME_QUARTAL", "sign": -1, "weight": 2.0}
  ]
}
```

`RelationSpec` carries `weight` and `sign` (negative marks a constraint/exclusion
edge) so a custom expander or fusion strategy can treat affinity and exclusion
differently, the foundation for exclusion-aware retrieval over a booking/rule
graph. Link chunks to entities (e.g. `(:Chunk)-[:ABOUT]->(:EbmCode)`) and the
projected graph bridges unstructured text and structured rules.

> Retrieval candidates remain chunk-shaped (the median/MMR/rerank stages assume
> chunk text + `content_embedding`). Entities improve *proximity*; to surface
> entity-linked chunks as candidates, register an expander that returns those
> chunks as siblings.

---

## Recency (temporal decay)

`RECENCY_ENABLED=true` blends an exponential **recency** factor into the final
ranking — the agent-*memory* signal that pure-relevance retrieval ignores. Among
similarly relevant results, newer ones rank higher; what Mem0/Zep/Letta do for
memory and most RAG stacks don't.

- Applied **after reranking**, so it's orthogonal to relevance — the
  cross-encoder can't overwrite it. The final ordering key is
  `(1 - RECENCY_WEIGHT) · normalized_relevance + RECENCY_WEIGHT · recency`.
- `recency = 0.5 ^ (document_age / RECENCY_HALF_LIFE_DAYS)` — 1.0 for a
  just-ingested document, 0.5 at one half-life. Re-ingesting a document refreshes
  its age (its `created_at` resets).
- **No schema or ingest change**: the age is read once per search over the
  candidate pool (reusing each document's `created_at`), like the sparse /
  near-duplicate reads — the hot retrieval queries are untouched.
- Tunable per request (`recency_enabled` / `recency_weight` /
  `recency_half_life_days`); each result carries its `recency_score`.

---

## Per-request tuning

`POST /search` accepts a `tuning` object overriding search-shaping settings for
that call only:

```json
{"query": "...", "tuning": {"hyde_enabled": false, "fusion_strategy": "rrf", "final_top_k": 5}}
```

Only fields in `SEARCH_TUNABLE_FIELDS` ([app/config.py](../app/config.py)) are
accepted: endpoints, credentials, embedding model/dimension and ingest-time
settings are rejected with a 422. Overrides are re-validated and never mutate
the process-wide settings.

### Presets

A `preset` key selects a named bundle of tunable overrides ([app/presets.py](../app/presets.py)):
`cheap` (no HyDE/reranker, decay proximity, shallower channels), `balanced` (the
defaults), `max_quality` (wider recall + diversity + deeper rerank). Explicit
fields override the preset; a process-wide default comes from `SEARCH_PRESET`.

```json
{"query": "...", "tuning": {"preset": "cheap", "final_top_k": 3}}
```

Presets are a thin convenience over the same `tuned()` validation — add your own
to the `PRESETS` dict (every value must be a tunable field, checked at import).

---

## Schema guard

Vector indexes are built for a specific embedding model, dimension and channel
set. Changing any of those silently invalidates them. On startup engram
hashes that config into a **schema signature** stored on an `EngramMeta` node and
compares it:

- `SCHEMA_GUARD_MODE=error` (default): refuse to start on a mismatch.
- `warn`: log and adopt the new signature.
- `off`: silently adopt.

To intentionally change the indexed config, wipe the store
(`docker compose down -v`) and re-ingest, or set the guard to `warn`/`off` for
one start. Non-index-affecting tuning (fusion weights, HyDE, ...) never trips it.

---

## Community synthesis (global themes)

engram's default retrieval is *local* (per query). For corpus-wide
"what are the main themes?" questions there's an opt-in, **offline** GraphRAG-style
layer (Neo4j + GDS only):

```bash
# cluster the chunk graph (Leiden) + write an LLM report per community
docker compose exec api python -m scripts.build_communities        # or --no-reports
# or via the API
curl -X POST 'localhost:8088/communities/rebuild?reports=true'
curl localhost:8088/communities          # list themes with reports + keywords
# global search: rank themes against a question
curl -X POST localhost:8088/communities/search \
  -H 'Content-Type: application/json' -d '{"query": "main themes about X", "top_k": 5}'
```

It clusters the `Chunk`/`Keyword` graph with `gds.leiden`, names each cluster
with a report (reusing the LLM seam; skipped gracefully if the LLM is down), and
persists `(:Community)` nodes + `(:Chunk)-[:IN_COMMUNITY]->(:Community)` in the
same store. Run it after ingest, never on the search path. On the pgvector
backend (no GDS) the rebuild endpoint returns `501`. `COMMUNITY_MIN_SIZE` drops
tiny communities.

---

## Observability

engram keeps observability to stdlib logging (no tracing dependency). Set
`LOG_LEVEL=DEBUG` to surface a one-line diagnostics summary per `/search` —
timing, candidate-pool size, shortlist size, and which degradation fallbacks
(embedding-down, reranker-down/disabled) fired. The degradation paths
themselves log at `WARNING`, so they show up at the default level too.

---

## Keeping the defaults sacred

Every seam's default reproduces the original pipeline, and the test suite pins
the exact scoring math (`tests/test_search_pipeline.py`, `test_scoring.py`).
When adding a strategy, **register a new key** rather than changing a default,
and run `docker compose run --rm tests` (or `pytest`); the pipeline tests will
catch any drift in the golden path.
