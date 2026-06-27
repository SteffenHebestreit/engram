"""Pluggable storage backend.

engram's data layer was a flat set of Neo4j/Cypher functions. A `Store` is the
protocol those callers (ingest, search, the pipeline strategies and the API)
talk to instead, so the backing store is swappable: the built-in `neo4j` store
keeps the full graph + vector + GDS personalized-PageRank behaviour, while a
`pgvector` store offers a lighter PostgreSQL alternative.

A store is selected by `Settings.store_backend` through the `STORES` registry,
the same pattern the pipeline stages use; third-party backends can register via
the `engram.plugins` entry-point group. `graph_proximity` is the one optional
capability — a backend without a graph-activation algorithm (e.g. pgvector)
returns None and the pipeline transparently falls back to per-hop decay, reusing
the existing PPR→decay degrade path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

from .registry import Registry

if TYPE_CHECKING:
    from .channels import VectorChannel
    from .config import Settings


@runtime_checkable
class Store(Protocol):
    """The persistence surface the rest of engram depends on.

    Implementations own their own connection (no driver is threaded through the
    callers). Chunk rows and the dict shapes returned by the read methods match
    what the Neo4j backend produced, so the scoring/pipeline code is unchanged.
    """

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        """Open the connection/pool. Called once at startup."""
        ...

    async def init_schema(self) -> None:
        """Create indexes/constraints/tables; idempotent."""
        ...

    async def verify_connectivity(self) -> None:
        """Raise if the store is unreachable (backs GET /health)."""
        ...

    async def close(self) -> None:
        """Release the connection/pool. Called once at shutdown."""
        ...

    # ── documents ────────────────────────────────────────────────────────────
    async def save_document(
        self,
        doc_id: str,
        title: str,
        sources: list[str],
        chunks: list[dict[str, Any]],
    ) -> None: ...

    async def delete_document(self, doc_id: str) -> int | None: ...

    async def get_document(self, doc_id: str) -> dict[str, Any] | None: ...

    async def add_document_source(self, doc_id: str, source: str) -> None: ...

    async def remove_document_source(
        self, doc_id: str, source: str
    ) -> dict[str, Any] | None: ...

    async def list_documents(self) -> list[dict[str, Any]]: ...

    async def fetch_document_chunks(
        self, doc_id: str, embedding_props: list[str]
    ) -> list[dict[str, Any]]:
        """Existing chunks of a document as
        `{text, summary, keywords, embeddings: {prop: vector}}`, for incremental
        re-ingest (reusing the stored vectors/metadata of unchanged chunks)."""
        ...

    # ── retrieval ────────────────────────────────────────────────────────────
    async def vector_search(
        self,
        channel: "VectorChannel",
        embedding: list[float],
        k: int,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]: ...

    async def fulltext_search(
        self, query: str, k: int, tenant_id: str | None = None
    ) -> list[dict[str, Any]]: ...

    async def fetch_siblings(
        self, seed_ids: list[str], keyword_sibling_limit: int, sequence_max_hops: int
    ) -> list[dict[str, Any]]: ...

    async def fetch_context(
        self, chunk_id: str, before: int, after: int
    ) -> list[dict[str, Any]] | None: ...

    async def get_sparse_weights(
        self, chunk_ids: list[str]
    ) -> dict[str, dict[str, float]]:
        """Stored BGE-M3 learned-sparse term-weight maps `{id: {token: weight}}`
        for the given chunks (omitting any ingested without sparse weights).
        Read once per search over the candidate pool — opt-in, so a backend that
        never stored them simply returns `{}`."""
        ...

    async def nearest_chunks(
        self,
        embedding: list[float],
        k: int,
        min_sim: float,
        exclude_doc_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Existing chunks most similar to `embedding` by content-vector cosine,
        for the memory write-path (near-duplicate detection at ingest).

        Returns up to `k` chunks with `sim >= min_sim`, each as
        `{id, doc_id, seq, text, sim}` (highest sim first), excluding chunks of
        `exclude_doc_id`. Reuses the existing content vector index — no new
        index/schema — so it is a read-only primitive that never touches the
        search hot path."""
        ...

    async def get_near_dup_links(self, chunk_ids: list[str]) -> dict[str, str]:
        """`{chunk_id: canonical_chunk_id}` for chunks linked as near-duplicates
        at ingest (memory write-path). Read once per search over the candidate
        pool to collapse near-duplicate clusters; opt-in, so a backend/corpus
        without links simply returns `{}`."""
        ...

    async def get_chunk_recency(self, chunk_ids: list[str]) -> dict[str, float]:
        """`{chunk_id: age_seconds}` — the age of each chunk's document by the
        store's own clock. Read once per search over the candidate pool to blend a
        recency factor into the final ordering (the agent-memory signal). Opt-in,
        so a backend that can't supply ages returns `{}`."""
        ...

    async def record_feedback(
        self,
        query: str,
        used_chunk_ids: list[str],
        query_id: str | None = None,
        query_embedding: list[float] | None = None,
    ) -> int:
        """Record that `used_chunk_ids` were the chunks an agent actually grounded
        its answer on for `query` (implicit-relevance feedback). Persists the
        (query → used-chunk) positives durably. When `query_embedding` is supplied,
        it is stored too, enabling the **agent-memory boost** (`memory_candidates`)
        — the learning side of the write-path that a stateless retriever can't
        produce. Returns how many links were recorded (chunks that exist)."""
        ...

    async def memory_candidates(
        self,
        query_embedding: list[float],
        min_sim: float,
        max_neighbors: int,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Agent-memory recall: given the current query embedding, find recorded
        feedback whose query embedding is ≥ `min_sim` cosine-similar (up to
        `max_neighbors`), and return the chunk hits those similar past queries
        *used* — each a normal chunk hit (id/doc_id/text/summary/keywords/
        content_embedding/tenant_id) plus `memory_score` = the best query-query
        cosine. Lets retrieval surface what worked for similar past queries
        (improves over time) — the learning side of the write-path a stateless
        retriever can't produce. Returns `[]` when the backend doesn't store
        feedback embeddings / has no matching history."""
        return []

    async def graph_proximity(
        self, seed_ids: list[str], candidate_ids: list[str], damping: float
    ) -> dict[str, float] | None:
        """Graph-activation proximity per candidate (HippoRAG-style), or None
        when the backend has no such algorithm — the caller then falls back to
        per-hop decay."""
        ...

    # ── community synthesis (optional capability) ────────────────────────────
    async def detect_communities(
        self, min_size: int
    ) -> list[dict[str, Any]] | None:
        """Cluster the graph into communities, or None when the backend has no
        community-detection algorithm (the caller then reports it unsupported)."""
        ...

    async def save_communities(self, communities: list[dict[str, Any]]) -> int: ...

    async def list_communities(self) -> list[dict[str, Any]]: ...

    async def community_vectors(self) -> list[dict[str, Any]]:
        """Communities carrying a `report_embedding`, for ranked global search
        (empty when unsupported)."""
        ...

    # ── structured-entity ingest (optional capability) ───────────────────────
    async def upsert_entities(
        self, label: str, items: list[dict[str, Any]]
    ) -> int: ...

    async def upsert_relations(
        self,
        from_label: str,
        rel_type: str,
        to_label: str,
        items: list[dict[str, Any]],
    ) -> int: ...


StoreFactory = Callable[["Settings"], Store]

# name -> factory; built-ins register on import, plugins via engram.plugins
STORES: Registry[StoreFactory] = Registry("store")

_builtins_loaded = False


def _ensure_builtin_stores() -> None:
    """Import the built-in store modules so they register their factories.

    Done lazily (and guarded) so importing this module is cheap and so a
    missing optional dependency — e.g. psycopg for the pgvector backend — never
    breaks the default Neo4j path.
    """
    global _builtins_loaded
    if _builtins_loaded:
        return
    from . import store_neo4j  # noqa: F401  (registers "neo4j")

    try:
        from . import store_pgvector  # noqa: F401  (registers "pgvector")
    except Exception:  # pragma: no cover - pgvector deps are optional
        pass

    from . import store_engramdb  # noqa: F401  (registers "engramdb", embedded)

    _builtins_loaded = True


def create_store(settings: "Settings") -> Store:
    """Build the store selected by `settings.store_backend`."""
    _ensure_builtin_stores()
    return STORES.get(settings.store_backend)(settings)
