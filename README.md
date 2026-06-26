<p align="center">
  <img src="logo.svg" alt="engram" width="480">
</p>

**Retrieval-Augmented Generation that doesn't just *search* your documents, it *remembers* them.**

An *engram* is the physical trace a memory leaves in a brain. That's what
ingestion leaves here: not a row in an index, but a structured imprint
(linked, labelled, embedded three ways) that search can later *re-activate*.

The name unpacks into exactly what it is:
**E**dge · **N**ode · **G**enerative · **R**etrieval · **A**ugmented · **M**emory.

- **Edge + Node**, the graph: chunks, documents and keywords wired by `PART_OF`, `NEXT_CHUNK` and `HAS_KEYWORD`.
- **Generative · Retrieval · Augmented**, the RAG, hiding in plain sight inside en**GRA**m.
- **Memory**, the engram itself: a trace you re-activate, not a row you look up.

That's why there's no `RAG` suffix: it's already in the name.

Most RAG systems are a flashlight in a dark warehouse: they point at the one
shelf that looks similar to your question and hope for the best. **engram**
is a team of librarians with a map of the whole building. They know what every
page says, what it *means*, what it's *about*, which page comes before and
after it, and which other books talk about the same things.

One Neo4j database. Four search channels. A knowledge graph. And a scoring
pipeline built from ideas that retrieval research says actually work (HyDE,
DBSF, convex fusion, Personalized PageRank, MMR, median-proximity scoring and
cross-encoder reranking), all in one small, readable FastAPI service.

---

## The big picture

```
        INGESTION  -  reading & remembering
        ─────────────────────────────────────
 document ──► split into pages ──► LLM writes a summary + keywords per page
                                         │
                                         ▼
                       3 embeddings per page (content / summary / keywords)
                                         │
                                         ▼
               ┌─────────────────────────────────────────────┐
               │                  Neo4j                      │
               │   ●──next──►●──next──►●──next──►●  (order)  │
               │   │         │                   │           │
               │   └────┐    └───┐    ┌──────────┘           │
               │        ▼        ▼    ▼                      │
               │         (shared keyword nodes)              │
               └─────────────────────────────────────────────┘

        SEARCH  -  recalling & judging
        ─────────────────────────────────────
 question ─► imagine the ─► 4 librarians ─► fair      ─► ink-drop  ─► crowd
             answer first    in parallel     grading      ripple       check
             (HyDE)          (4 channels)    (DBSF +      (PageRank)   (median)
                                              fusion)        │            │
                                                             ▼            ▼
                              diverse shortlist (MMR) ──► expert judge (reranker)
                                                                  │
                                                                  ▼
                                                 stop at the quality cliff (autocut)
```

---

## How a document is remembered

When you feed a document in, it isn't just stored, it's *studied*:

1. **Pages, not piles.** The text is split into overlapping chunks (~1800
   characters), like tearing a book into pages that slightly overlap so no
   sentence falls into the gap between two pages.

2. **Three name tags per page.** An LLM reads every chunk and writes a
   one-sentence **summary** and a set of **keywords**. Each of the three
   views (full text, summary, keywords) gets its **own embedding**. Why?
   Because a question sometimes matches a page's exact words, sometimes its
   gist, and sometimes just its topic. Three tags, three chances to be found.

3. **A thread through the book.** Chunks are linked in reading order with
   *directional* `NEXT_CHUNK` edges: every page knows which page comes
   **before** it and which comes **after**. Context is never lost.

4. **Colored threads across books.** Keywords become shared graph nodes. Two
   chunks in completely different documents that both talk about "load
   balancing" are physically connected, colored threads running through the
   whole library.

```
   Document A:  ●──►●──►●──►●           ● = chunk
                    │       │
                    ◆ "kubernetes" ──┐   ◆ = shared keyword node
                    │                 │
   Document B:  ●──►●──►●──►●─────────┘
```

---

## How an answer is found

Nine steps, each one earning its place.

### 1. Imagine the answer first (HyDE)

Short questions are terrible search probes: *"alpha decay?"* looks nothing
like a paragraph that answers it. So for short queries, the LLM first **dreams
up a plausible answer**, and we search with *that*. It's like a detective
sketching the suspect before scanning the crowd: you match faces against the
sketch, not against the word "suspect". (LLM down? We silently fall back to
the plain question, search never breaks.)

### 2. Four librarians search in parallel

Your query runs against **four channels at once**:

| Librarian | Looks at | Catches |
|---|---|---|
| Content | full chunk-text embedding | detailed semantic matches |
| Summary | one-sentence-gist embedding | "this page is *about* that" |
| Keywords | topic-label embedding | thematic matches |
| Fulltext (BM25) | the literal words | exact terms, names, error codes |

The fourth matters more than people think: embeddings are great at meaning but
terrible at `ERR_SSL_VERSION_MISMATCH`. The lexical librarian never misses an
exact string.

### 3. Fair grading (DBSF normalization)

Each librarian grades differently: one is generous, one is harsh, one had a
single favourite that dwarfs everything else. Before mixing opinions, each
channel's scores are **normalized against their own distribution** (z-score,
clipped at ±3σ). A single freak outlier can no longer crush everyone else's
grades to the floor: a grading curve that survives the one student who broke
the exam.

### 4. Votes add up (convex fusion)

The four normalized opinions are then **summed** with weights, not max-picked.
A page that the content librarian *and* the summary librarian *and* the
keyword librarian all liked beats a page only one of them liked. **Agreement
is evidence.** (Research agrees: tuned convex score combinations beat
rank-only fusion.)

> **Optional fifth signal — learned-sparse (`SPARSE_ENABLED`, off by default).**
> engram already runs **BGE-M3**, which emits a SPLADE-style *learned-sparse*
> term-weight vector alongside its dense one — and normally throws it away. Turn
> it on and each chunk's sparse weights are stored at ingest and dotted against
> the query's at search time, folding an **exact-term** score (rare entities,
> IDs, numbers the dense vector smooths away) into the fused score. Opt-in, no
> extra index, degrades to a no-op if the endpoint is down.
>
> **Measured honestly:** at engram's **default rerank depth (15)** sparse is a
> **clear win on SciFact — +1.1 nDCG@10 / +1.8 recall@10** (`engram+sparse` 0.7428
> is the best config measured), smaller-but-positive on NFCorpus (+0.25 nDCG),
> never a regression; with the reranker *off* it lifts the fused ranking
> **+1.6 / +2.3**. It only washes out at an artificially **deep** rerank depth
> (100), where the cross-encoder re-scores nearly the whole pool from text — the
> mechanism is that sparse improves the *shortlist*, so it pays off exactly when
> the shortlist is the bottleneck (the realistic case). Still **opt-in** (it needs
> a multi-output endpoint — see [deploy/bge-m3](deploy/bge-m3)) but **recommended
> when available**. Full four-config breakdown + mechanism:
> [bench/RESULTS.md](bench/RESULTS.md) §1c.

### 5. Read the neighbouring pages

The best hits become **seeds**, and the graph wakes up. We walk the
`NEXT_CHUNK` chain up to 3 hops **backwards and forwards** (the answer's
setup often lives on the *previous* page, its conclusion on the *next*), and
follow keyword threads to related chunks in other documents.

> **Why engram barely cares about chunk size or overlap.** Naive RAG obsesses
> over chunking — optimal size, overlap windows, semantic splitting — because a
> flat vector store has *no way to recover context that fell across a boundary*.
> engram does: if an answer spans two chunks, it retrieves **both** via this
> `NEXT_CHUNK` walk. So **overlap is redundant** (the neighbour is already pulled
> in — engram defaults `CHUNK_OVERLAP_CHARS=0`) and **semantic chunking is
> largely unnecessary** (a split passage is stitched back at retrieval). Chunking
> goes from a fragile tuning problem to a robustness property — and
> `GET /chunks/{id}/context` lets the agent widen the window on demand.

### 6. The ink-drop test (Personalized PageRank)

How relevant is a neighbour, really? We drop **ink** on the seed chunks and
let it flow along the graph's edges (Personalized PageRank, the algorithm
behind Google's original ranking, here in its HippoRAG flavour). A chunk fed
by *multiple seeds* over *multiple paths* soaks up more ink than a dead end
three hops from a single seed:

```
     seed ●━━► ◔ ━━► ○                ink spreads and ACCUMULATES:
           ┗━━ ◆ ━━━┓
     seed ●━━━━━━━━► ◕  ◄── drinks from 3 paths → high proximity
           ┗━━━━━━━━━┛
```

Every result carries this **graph proximity** value (1.0 = direct hit, lower
= further away): a built-in "how many edges away is this?" indicator, earned
through actual graph flow instead of a fixed table. No GDS plugin? The system
automatically falls back to a clean per-hop decay. Nothing breaks.

### 7. The crowd check (median proximity)

*Our signature move.* Take every candidate's embedding and compute the
**element-wise median vector**, the center of gravity of everything the
search found. Candidates close to that center get a boost; lonely outliers
that drifted in on a fluke get politely down-weighted. If twenty results talk
about Kubernetes and one talks about cooking, the crowd has spoken.

### 8. The total score

```
total = 0.55 × retrieval   (what the four librarians agreed on)
      + 0.30 × median      (how central to the crowd)
      + 0.15 × proximity   (how much ink reached it)
```

### 9. A diverse panel, then the expert judge

- **MMR shortlist**: we don't send fifteen near-identical neighbouring pages
  to the final round. Each shortlist seat goes to the candidate that is
  *relevant AND different* from those already seated. A panel, not an echo
  chamber.
- **Cross-encoder reranker**: a model that reads your query and each finalist
  *together*, word by word. The slow, careful expert who only judges the
  shortlist.
- **Autocut**: results are returned **until the quality falls off a cliff**.
  Scores go `0.95, 0.93, 0.91, ... 0.30`? We stop before the 0.30. You get five
  great answers, never eight padded ones.

And when a result needs more context, `GET /chunks/{id}/context` instantly
hands you the pages before and after it. The graph already knows the way.

---

## Quickstart

**You need:** Docker (that's it), plus three OpenAI-compatible endpoints on
your network (embeddings, chat LLM, reranker, e.g. TEI / Infinity / vLLM /
Ollama).

```bash
# 1. configuration
cp .env.example .env          # fill in your endpoint URLs

# 2. everything: Neo4j (incl. Graph Data Science) + the API service
docker compose up -d --build
```

Prefer Postgres? Run the pgvector backend instead (graph-lite: no GDS
PageRank, proximity falls back to decay):

```bash
docker compose --profile pgvector up -d postgres
STORE_BACKEND=pgvector docker compose up -d --build api
```

The API is now listening on `localhost:8088`. Endpoints running on the Docker
host itself are reachable from inside the containers as
`http://host.docker.internal:<port>`. Use that instead of `localhost` in
`.env`.

Vector indexes, constraints and the fulltext index are created automatically
on startup. `EMBEDDING_DIM` must match your embedding model; changing it
later requires dropping the three `chunk_*_idx` indexes.

### Talk to it

```bash
# feed it a document
curl -X POST localhost:8088/documents \
  -H "Content-Type: application/json" \
  -d '{"text": "<document text>", "title": "My Doc", "source": "wiki"}'

# ask it something
curl -X POST localhost:8088/search \
  -H "Content-Type: application/json" \
  -d '{"query": "how does the ink-drop test work?"}'
```

**Document identity & reference counting.** Each chunk is tied to its document
by a `doc_id` property and a `PART_OF` edge to a `Document` node. That id is
**stable and recomputable**: pass your own `document_id`, or omit it and the id
is the **SHA-256 of the text**. So you can delete a document later without having
stored the id we returned, and re-ingesting the same content is **deduplicated**
(content-addressed) rather than duplicated: an identical re-ingest just
registers its source, skipping re-embedding.

A **`source` is required** on every ingest, and a document can be pulled in by
several sources (contexts), tracked in a `sources` array. Because every document
is reference-counted this way, it can always be cleaned up, so nothing ever sits
unreferenced leaving chunks orphaned. Deletion comes in two flavours:

| Call | Effect |
|---|---|
| `DELETE /documents/{id}?source=S` | drop source `S`'s reference; the nodes are torn down **only when `S` was the last source** |
| `DELETE /documents/{id}` | **hard remove** now, regardless of how many sources reference it |

Either way the teardown is complete: every chunk, its `PART_OF`/`NEXT_CHUNK`/
`HAS_KEYWORD` edges, and any orphaned keywords. `GET /documents` shows each
document's current `sources`.

| Endpoint | What it does |
|---|---|
| `POST /documents` | ingest: chunk → LLM metadata → 3× embed → graph |
| `GET /documents` | list what's in the library |
| `DELETE /documents/{id}` | forget a document: every chunk, its edges, and orphaned keywords |
| `POST /search` | the full nine-step pipeline above |
| `POST /graph/entities` | load typed domain entity nodes (key + properties) |
| `POST /graph/relations` | load typed relations between entity nodes |
| `POST /communities/rebuild?reports=true` | cluster the corpus into themes + LLM reports (Neo4j+GDS) |
| `GET /communities` | the detected themes with their reports and keywords |
| `POST /communities/search` | rank the themes against a question (global search) |
| `GET /chunks/{id}/context?before=2&after=2` | the surrounding pages of any chunk |
| `GET /health` | is everything alive? |

`POST /search` also accepts a `tuning` object to override search-shaping
settings **per request** (weights, strategies, HyDE, autocut, ...) without a
redeploy, e.g. `{"query": "...", "tuning": {"hyde_enabled": false, "final_top_k": 5}}`.
Only search-shaping fields are accepted; endpoints, credentials and ingest-time
settings are rejected.

Every result is fully transparent. You can see *why* it was chosen:

```json
{
  "chunk_id": "...", "origin": "sibling:sequence:after",
  "channels": ["content", "fulltext"],
  "graph_distance": 1, "graph_proximity": 0.81,
  "retrieval_score": 0.62, "median_score": 0.77,
  "sparse_score": 0.40, "fused_score": 0.71, "rerank_score": 0.94
}
```

(`channels` is the *retrieval* provenance — which channels surfaced the chunk;
the learned-sparse contribution is a re-scoring signal, reported separately as
`sparse_score`.)

### Use it as an agent tool (MCP)

engram is a **retrieval tool**: an agent calls it for the most relevant document
context (with score provenance) and writes the answer itself. Besides the HTTP
API, engram ships a thin **MCP** (Model Context Protocol) adapter so any
MCP-speaking agent can plug it in directly:

```bash
pip install -r requirements-mcp.txt
ENGRAM_API_BASE=http://localhost:8088 python -m app.mcp_server   # stdio
```

It exposes `search`, `get_chunk_context`, `list_documents` and `search_themes`
as MCP tools (a thin proxy over the HTTP API above — same store, same pipeline).
engram does not generate answers; it feeds the calling agent perfect context.

### Prove it on your own corpus (`POST /eval`)

Most RAG tools "evaluate" with an LLM grading its own answers — biased,
irreproducible, and silent about *why*. engram scores **retrieval** against a
golden set on *your* data, judge-free:

```bash
curl -s localhost:8088/eval -H 'content-type: application/json' -d '{
  "cases": [
    {"query_id": "q1", "query": "how do I rotate the signing key?",
     "relevant_document_ids": ["doc-42", "doc-77"]}
  ],
  "k_values": [5, 10]
}'
```

You get standard IR metrics (**nDCG@k / Recall@k / P@k / MAP**) each with a
**bootstrap 95% confidence interval** (reproducible — no LLM in the loop), plus
the part nothing else ships: **per-channel attribution**.

```json
{ "metrics": { "nDCG@10": { "mean": 0.74, "ci95": [0.70, 0.78] } },
  "attribution": { "gold_hits_retrieved": 128,
                   "by_channel": { "content": 124, "fulltext": 103, "graph:keyword": 14 },
                   "unique_to_channel": { "fulltext": 9, "content": 5 } } }
```

`unique_to_channel` is the killer line: **9 relevant documents only the fulltext
(BM25) channel found** — exact-term/rare-entity hits the dense vectors missed
entirely, that you'd lose if you dropped it. (And it cuts the other way: if a
channel's unique count is ~0 on your corpus, it's redundant — engram's own
[benchmark findings](bench/RESULTS.md) used exactly this to show the dense
channel carries SciFact while the rest overlap.) Pass `tuning` to A/B a config on
your corpus and *measure* the difference instead of guessing.

### Memory, not just a vector store (`DEDUP_ENABLED`)

An agent's knowledge arrives messily: the same fact gets re-ingested, paraphrased,
across sources and sessions. A plain vector store accumulates N near-identical
chunks that then crowd distinct material out of every result. engram's memory
write-path (opt-in) catches this on ingest: a fresh chunk that is
≥ `DEDUP_COSINE_THRESHOLD` cosine-similar to an existing one is **linked**
(`near_dup_of`) to it, and search **collapses** near-duplicate clusters to their
best-retrieved member. It's **non-destructive** — the duplicate is still stored
and linked, never dropped, so a wrong link is recoverable, not a deleted fact
(the rule dedicated memory systems like Zep/Graphiti follow). The cosine
threshold is *embedder-coupled* — calibrate it on your own vectors. This is the
first slice of a broader write-path (temporal validity, supersession); the design
and the honest trade-offs are in
[docs/memory-writepath-plan.md](docs/memory-writepath-plan.md).

Handy scripts once `.env` is filled in (all run inside the api container):

- `docker compose exec api python -m scripts.demo`: live end-to-end round
  trip (ingest → search with full score breakdown → cleanup) against your
  network services.
- `docker compose run --rm -v "$PWD:/work" api python -m scripts.ingest_file /work/file.md`:
  ingest a file from disk (mounted into the container as `/work`).
- `docker compose exec api python -m scripts.check_db`: connectivity &
  schema check.

---

## Tuning knobs

Everything is a `.env` variable, with sane defaults and zero code changes:

| Knob | Default | What it turns |
|---|---|---|
| `RETRIEVAL_WEIGHT` / `MEDIAN_WEIGHT` / `GRAPH_PROXIMITY_WEIGHT` | 0.55 / 0.30 / 0.15 | the total-score recipe |
| `CONTENT/SUMMARY/KEYWORDS/FULLTEXT_CHANNEL_WEIGHT` | 1.0 / 0.9 / 0.8 / 0.7 | how loud each librarian speaks |
| `SEQUENCE_MAX_HOPS` | 3 | how many pages before/after to consider |
| `GRAPH_PROXIMITY_MODE` | `ppr` | `ppr` = ink-drop, `decay` = fixed per-hop fade |
| `SEQUENCE_PROXIMITY_DECAY` | 0.7 | per-hop fade in `decay` mode |
| `MMR_LAMBDA` | 0.7 | relevance vs. diversity on the shortlist |
| `AUTOCUT_MIN_GAP` / `AUTOCUT_MIN_KEEP` | 0.25 / 3 | how steep a cliff cuts, floor of kept results |
| `HYDE_ENABLED` / `HYDE_MAX_QUERY_WORDS` | true / 8 | when to imagine the answer first |
| `QUERY_INSTRUCTION` / `PASSAGE_INSTRUCTION` | "" / "" | task prefixes for instruction-tuned embedders (E5/GTE/Qwen3); empty = BGE-M3 default |
| `SPARSE_ENABLED` / `SPARSE_WEIGHT` | false / 0.2 | BGE-M3 learned-sparse exact-term re-scoring (see below) |
| `DEDUP_ENABLED` / `DEDUP_COSINE_THRESHOLD` | false / 0.95 | memory write-path: link + collapse near-duplicate chunks (see below) |
| `TOP_K_PER_INDEX` / `SEED_COUNT` | 12 / 8 | channel depth / how many hits get graph-expanded |
| `RERANK_TOP_K` / `FINAL_TOP_K` | 15 / 8 | shortlist size / answer size |
| `EXTRACTION_CONCURRENCY` | 4 | parallel LLM metadata calls during ingest |
| `EMBEDDING_BATCH_SIZE` / `EMBEDDING_CONCURRENCY` | 64 / 4 | texts per embedding request / requests in flight |

---

## Customization

Tuning knobs change *weights*. When you need to change *behaviour* (a different
chunker, a domain-specific extractor, a new fusion rule, or a whole different
graph), engram exposes **pluggable seams** instead of forks. Every seam ships
a default that reproduces the pipeline above, so nothing moves until you opt in.

| Seam | Swap it for | How |
|---|---|---|
| **Chunker** | semantic / markdown / code-aware splitting | register on `CHUNKERS`, set `CHUNK_STRATEGY` |
| **Metadata extractor** | entities, code symbols, Q/A pairs instead of summary+keywords | register on `EXTRACTORS`, set `METADATA_EXTRACTOR` |
| **Channels** | drop/add/re-weight embedded views (e.g. a "title" channel) | `VECTOR_CHANNELS` JSON, or a custom `CHANNEL_SOURCES` entry |
| **Fusion** | RRF, weighted-max, learned combination | register on `FUSIONS`, set `FUSION_STRATEGY` |
| **Expander** | typed-relation / entity-centric graph walks | register on `EXPANDERS`, set `EXPANDER_STRATEGY` |
| **Proximity** | `ppr` / `decay` (built in) or your own | register on `PROXIMITIES`, set `GRAPH_PROXIMITY_MODE` |
| **Router** | auto-pick a preset per query (`heuristic` built in), or off | register on `ROUTERS`, set `ROUTER_STRATEGY` |
| **Reranker** | a different cross-encoder/provider, the cheap `colbert` late-interaction option, or off | register on `RERANKERS`, set `RERANKER_STRATEGY` / `RERANKER_ENABLED` |
| **Graph profile** | project domain entity nodes/relations into PageRank | `GRAPH_PROFILE` JSON |
| **Store backend** | Neo4j (default) or PostgreSQL + pgvector | register on `STORES`, set `STORE_BACKEND` |

Two capabilities make *special graphs* (not just documents) first-class:

- **Structured-entity ingest**: `POST /graph/entities` and `POST /graph/relations`
  load typed domain nodes and relations (with label/type sanitization) directly
  into the same Neo4j store, no chunk→LLM→embed detour. Link them to chunks
  (e.g. `(:Chunk)-[:ABOUT]->(:Entity)`) and declare those labels/relations in a
  `GRAPH_PROFILE` so Personalized PageRank spreads activation through them.
- **Per-request tuning**: the `tuning` object on `/search` (see above) overlays
  search-shaping settings per call.

Third-party packages can register strategies without touching core by exposing
an `engram.plugins` entry point. A startup **schema guard** (`SCHEMA_GUARD_MODE`)
refuses to serve when the channel set or embedding model no longer match the
indexes already in the store. Full guide: [docs/customization.md](docs/customization.md).

---

## Tests

Tests run in a container too, no local Python needed:

```bash
docker compose run --rm tests
```

This builds the test image, starts Neo4j and Postgres if they aren't already
running, and runs the whole suite. Unit tests cover chunking, all scoring math
(DBSF, MMR, autocut, median-proximity), LLM-output parsing, the store seam and
the full search pipeline with mocked services, including PPR-vs-decay proximity,
HyDE blending and autocut behaviour. Two integration tests run ingest + search
against the live Neo4j and the live pgvector store, each skipping itself
automatically when its database is down. Pass pytest args as usual:
`docker compose run --rm tests pytest -k scoring`.

## Benchmarks 📊

engram is benchmarked **head-to-head** against the standard RAG retrieval
strategies — *same datasets, same embedding model, same reranker, same metrics;
only the architecture changes* — so any difference is the **architecture**, not
the models. (Running LightRAG/HippoRAG directly would confound the result with
their different embedders/LLMs.) Full methodology + every table:
[bench/RESULTS.md](bench/RESULTS.md).

On engram's production stack (**BGE-M3** + **bge-reranker-v2-m3**, on an RTX
4080), it is the **strongest architecture on the board for retrieval quality** —
beating naive single-vector RAG, BM25, *and* the standard dense→rerank pipeline.

**BEIR SciFact** (5,183 docs · 300 queries)

| system | nDCG@10 | Recall@10 | MAP |
|---|---|---|---|
| BM25 | 0.652 | 0.774 | 0.613 |
| dense — *naive vector RAG* | 0.642 | 0.775 | 0.599 |
| dense + rerank — *standard 2-stage* | 0.725 | 0.825 | 0.693 |
| **engram** | **0.741** | **0.856** | **0.704** |

engram wins **every metric on both** BEIR datasets (SciFact + NFCorpus). The line
that matters is `engram` vs `dense+rerank` — *identical models, the only
difference is engram's architecture* (4-channel fusion + graph + median-proximity
+ MMR). That gap — **our contribution, with the models cancelled out** — is
**+1.6 nDCG@10 / +3.1 recall@10** here, and it holds across every model
combination we tried (see [RESULTS.md](bench/RESULTS.md)).

**Two findings that make the advantage actionable** (full study in
[RESULTS.md §1d–1e](bench/RESULTS.md)):

- **The architecture's edge *compounds with a strong reranker*.** Swapping in a
  2026 reranker (Qwen3-Reranker-0.6B — a drop-in, multilingual `CrossEncoder`)
  lifts engram to **0.772 nDCG@10 / 0.891 recall@10**, and the architecture gap
  over `dense+rerank` *grows* to **+2.15 / +4.0** — engram's fusion+graph feed a
  better candidate pool that a strong reranker exploits. The reranker is the one
  stage nothing washes out, so it's the highest-leverage knob — and engram is the
  bigger winner for using it.
- **A stronger *embedder* barely moves the result** (Qwen3-Embedding is +4.4
  dense yet ≈tied at engram's pipeline) — the reranker caps it. So **BGE-M3 stays
  the robust default**; chase the reranker, not the embedder.

On **multi-hop** retrieval (HotpotQA — recall@k of the *linked* supporting
passages, the metric HippoRAG reports), engram **matches or beats** the strongest
baseline. Its keyword-graph + PageRank expansion gives a *large* lift when the
embedder is weak (with MiniLM: **+1.7 / +1.9 pts** Recall@5/@10 over dense+rerank,
+11 pts Recall@5 over naive dense), and is neutral with a SOTA embedder (BGE-M3
already retrieves the linked passage) — so the graph is a **robustness floor that
never costs you**.

```bash
# reproduce on GPU (real stack); drop the gpu override for the CPU floor
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm --build runner python -m bench.compare
docker compose -f bench/docker-compose.yml -f bench/docker-compose.gpu.yml run --rm --build multihop
```

---

## Design principles

- **One store by default, but pluggable.** Neo4j is the graph *and* the vector
  store: no second database to sync, no drift between "what's related" and
  "what's similar". When you'd rather consolidate on Postgres, a `pgvector`
  backend slots in behind the same `Store` seam (graph-lite: vector + fulltext +
  sequence/keyword siblings, proximity via decay instead of GDS PageRank).
- **Bring your own models *and* store.** Embeddings, LLM and reranker are plain
  OpenAI-compatible HTTP endpoints; the backing store is `STORE_BACKEND`
  (`neo4j` or `pgvector`). Local, on-prem, cloud: your call.
- **Degrade, never break.** LLM down? HyDE skips itself. GDS plugin missing (or
  pgvector backend)? Proximity falls back to decay. Reranker down? Results fall
  back to the fused score. Embedding endpoint down? Search degrades to
  fulltext-only (BM25) on the stored embeddings. The store is the only true hard
  dependency in search.
- **Every score is explainable.** Each result tells you how it entered the
  pool, how far it sits in the graph, and what every pipeline stage thought
  of it.

**Stack:** Python 3.12 · FastAPI · Neo4j 5.26 + GDS 2.13 *or* PostgreSQL +
pgvector · httpx · numpy · BGE-M3 embeddings · BGE-reranker-v2-m3. Swap any of them.

---

## License

MIT. See [LICENSE](LICENSE).

---

*Built on ideas from: HyDE (Gao et al.), Distribution-Based Score Fusion,
"An Analysis of Fusion Functions for Hybrid Retrieval" (Bruch et al.),
HippoRAG / Personalized PageRank, Maximal Marginal Relevance (Carbonell &
Goldstein, 1998), and one home-grown idea we're proud of:
**median-proximity scoring**.*
