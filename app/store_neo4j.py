"""Neo4j storage backend.

A thin `Store` facade over the Cypher functions in `app.graph` (each of which
takes the driver as its first argument). Keeping the Cypher as a free-function
library means the existing Neo4j-specific tests keep exercising it directly,
while the rest of engram talks to the backend-agnostic `Store` protocol.

This is the full-feature backend: graph + vector in one store, plus GDS
personalized-PageRank graph proximity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import graph
from .store import STORES

if TYPE_CHECKING:
    from neo4j import AsyncDriver

    from .channels import VectorChannel
    from .config import Settings


class Neo4jStore:
    """Implements `Store` by delegating to the `app.graph` Cypher functions."""

    def __init__(self, driver: "AsyncDriver | None" = None) -> None:
        # the driver may be injected (tests) or created lazily in connect()
        self._driver = driver

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        if self._driver is None:
            self._driver = graph.create_driver()

    async def init_schema(self) -> None:
        await graph.init_schema(self._driver)

    async def verify_connectivity(self) -> None:
        await self._driver.verify_connectivity()

    async def close(self) -> None:
        if self._driver is not None:
            await self._driver.close()

    # ── documents ────────────────────────────────────────────────────────────
    async def save_document(
        self,
        doc_id: str,
        title: str,
        sources: list[str],
        chunks: list[dict[str, Any]],
    ) -> None:
        await graph.save_document(self._driver, doc_id, title, sources, chunks)

    async def delete_document(self, doc_id: str) -> int | None:
        return await graph.delete_document(self._driver, doc_id)

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return await graph.get_document(self._driver, doc_id)

    async def add_document_source(self, doc_id: str, source: str) -> None:
        await graph.add_document_source(self._driver, doc_id, source)

    async def remove_document_source(
        self, doc_id: str, source: str
    ) -> dict[str, Any] | None:
        return await graph.remove_document_source(self._driver, doc_id, source)

    async def list_documents(self) -> list[dict[str, Any]]:
        return await graph.list_documents(self._driver)

    async def fetch_document_chunks(
        self, doc_id: str, embedding_props: list[str]
    ) -> list[dict[str, Any]]:
        return await graph.fetch_document_chunks(self._driver, doc_id, embedding_props)

    # ── retrieval ────────────────────────────────────────────────────────────
    async def vector_search(
        self, channel: "VectorChannel", embedding: list[float], k: int
    ) -> list[dict[str, Any]]:
        return await graph.vector_search(self._driver, channel.index, embedding, k)

    async def fulltext_search(self, query: str, k: int) -> list[dict[str, Any]]:
        return await graph.fulltext_search(self._driver, query, k)

    async def fetch_siblings(
        self, seed_ids: list[str], keyword_sibling_limit: int, sequence_max_hops: int
    ) -> list[dict[str, Any]]:
        return await graph.fetch_siblings(
            self._driver, seed_ids, keyword_sibling_limit, sequence_max_hops
        )

    async def fetch_context(
        self, chunk_id: str, before: int, after: int
    ) -> list[dict[str, Any]] | None:
        return await graph.fetch_context(self._driver, chunk_id, before, after)

    async def get_sparse_weights(
        self, chunk_ids: list[str]
    ) -> dict[str, dict[str, float]]:
        return await graph.get_sparse_weights(self._driver, chunk_ids)

    async def nearest_chunks(
        self,
        embedding: list[float],
        k: int,
        min_sim: float,
        exclude_doc_id: str | None = None,
    ) -> list[dict[str, Any]]:
        return await graph.nearest_chunks(
            self._driver, embedding, k, min_sim, exclude_doc_id
        )

    async def get_near_dup_links(self, chunk_ids: list[str]) -> dict[str, str]:
        return await graph.get_near_dup_links(self._driver, chunk_ids)

    async def graph_proximity(
        self, seed_ids: list[str], candidate_ids: list[str], damping: float
    ) -> dict[str, float] | None:
        return await graph.ppr_proximity(self._driver, seed_ids, candidate_ids, damping)

    # ── community synthesis ──────────────────────────────────────────────────
    async def detect_communities(
        self, min_size: int
    ) -> list[dict[str, Any]] | None:
        return await graph.detect_communities(self._driver, min_size)

    async def save_communities(self, communities: list[dict[str, Any]]) -> int:
        return await graph.save_communities(self._driver, communities)

    async def list_communities(self) -> list[dict[str, Any]]:
        return await graph.list_communities(self._driver)

    async def community_vectors(self) -> list[dict[str, Any]]:
        return await graph.community_vectors(self._driver)

    # ── structured-entity ingest ─────────────────────────────────────────────
    async def upsert_entities(
        self, label: str, items: list[dict[str, Any]]
    ) -> int:
        return await graph.upsert_entities(self._driver, label, items)

    async def upsert_relations(
        self,
        from_label: str,
        rel_type: str,
        to_label: str,
        items: list[dict[str, Any]],
    ) -> int:
        return await graph.upsert_relations(
            self._driver, from_label, rel_type, to_label, items
        )


@STORES.register("neo4j")
def _make_neo4j_store(settings: "Settings") -> Neo4jStore:
    return Neo4jStore()
