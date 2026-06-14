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
| Metadata extraction | `EXTRACTORS` | `METADATA_EXTRACTOR` | `default` | [app/llm.py](../app/llm.py) |
| Channel embed-source | `CHANNEL_SOURCES` | (per channel `source`) | `text`/`summary`/`keywords` | [app/channels.py](../app/channels.py) |
| Score fusion | `FUSIONS` | `FUSION_STRATEGY` | `dbsf_convex` | [app/pipeline.py](../app/pipeline.py) |
| Graph expansion | `EXPANDERS` | `EXPANDER_STRATEGY` | `sequence_keyword` | [app/pipeline.py](../app/pipeline.py) |
| Graph proximity | `PROXIMITIES` | `GRAPH_PROXIMITY_MODE` | `ppr` (→ `decay`) | [app/pipeline.py](../app/pipeline.py) |

### Chunker
`(text: str, settings: Settings) -> list[str]`. The default is a
paragraph/sentence-aligned fixed window with overlap.

### Metadata extractor
`async (client, chunk) -> ExtractionResult` exposing at least `.summary` and
`.keywords` (those feed the summary/keywords channels). A domain extractor might
pull named entities or code symbols instead.

### Fusion
`(channel_hits, channels, fulltext_hits, fulltext_weight, settings) -> dict[id, candidate]`.
The default DBSF-normalizes each channel then convex-combines them. An
alternative could implement Reciprocal Rank Fusion.

### Expander
`async (driver, seed_ids, settings) -> list[sibling]`. The default walks
`NEXT_CHUNK` both ways and shared-keyword siblings. A custom expander could walk
typed domain relations (see *Special graphs* below).

### Proximity
`async (driver, seed_ids, siblings, settings) -> list[float]` parallel to
`siblings`. Built-ins: `ppr` (Personalized PageRank with a per-sibling decay
fallback) and `decay`.

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

## Keeping the defaults sacred

Every seam's default reproduces the original pipeline, and the test suite pins
the exact scoring math (`tests/test_search_pipeline.py`, `test_scoring.py`).
When adding a strategy, **register a new key** rather than changing a default,
and run `docker compose run --rm tests` (or `pytest`); the pipeline tests will
catch any drift in the golden path.
