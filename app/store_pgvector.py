"""PostgreSQL + pgvector storage backend.

A lighter, graph-lite alternative to the Neo4j backend, for deployments that
already run Postgres. It keeps engram's retrieval model intact in SQL:

  * vector channels   -> one `vector(dim)` column per channel + an HNSW index
  * fulltext (BM25-ish)-> a generated `tsvector` column + GIN index + ts_rank
  * NEXT_CHUNK siblings-> chunks in the same document within ±hops of `seq`
  * HAS_KEYWORD siblings-> a `chunk_keywords` join table (shared keyword counts)

What it does NOT have is a graph-activation algorithm: there is no GDS
personalized PageRank, so `graph_proximity` returns None and the search
pipeline transparently falls back to per-hop decay — exactly the existing
"GDS missing" degrade path. Structured-entity ingest (arbitrary typed nodes +
relations) is Neo4j-only and raises NotImplementedError here.

The dict shapes returned by every read mirror the Neo4j backend, so the
scoring/pipeline code is identical regardless of backend.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

import numpy as np

from . import graph  # reused: schema_signature() is backend-neutral
from .channels import resolve_vector_channels
from .store import STORES

if TYPE_CHECKING:
    from .channels import VectorChannel
    from .config import Settings

log = logging.getLogger(__name__)

_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]")

# columns every candidate-producing read returns, matching the Neo4j backend
# (content_embedding is the canonical geometry vector for median/MMR)
_CHUNK_COLS = "id, doc_id, text, summary, keywords, content_embedding"

_SCHEMA_MISMATCH_MSG = (
    "vector schema signature mismatch: the existing tables were built for a "
    "different embedding model, dimension, or channel set. Re-ingest after "
    "wiping the store, or set SCHEMA_GUARD_MODE=warn/off to override."
)


def _safe_column(value: str) -> str:
    """Whitelist a channel column name (from trusted config) for interpolation."""
    cleaned = _IDENTIFIER_RE.sub("", value or "")
    if not cleaned:
        raise ValueError(f"invalid column name: {value!r}")
    return cleaned


def _sanitize_fulltext_query(query: str) -> str:
    return re.sub(r"[^\w\s]", " ", query).strip()


def _row(record: dict[str, Any]) -> dict[str, Any]:
    """Normalize a fetched row: pgvector columns come back as numpy arrays;
    expose content_embedding as a plain list like the Neo4j backend does."""
    emb = record.get("content_embedding")
    if emb is not None and hasattr(emb, "tolist"):
        record["content_embedding"] = emb.tolist()
    return record


class PgvectorStore:
    """Implements `Store` over PostgreSQL with the pgvector extension."""

    def __init__(self, dsn: str) -> None:
        self._dsn = dsn
        self._pool = None

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        import psycopg
        from pgvector.psycopg import register_vector_async
        from psycopg_pool import AsyncConnectionPool

        # the vector type must exist before any pooled connection registers it,
        # so create the extension up front (init_schema also ensures it)
        async with await psycopg.AsyncConnection.connect(
            self._dsn, autocommit=True
        ) as conn:
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

        async def _configure(conn) -> None:
            await register_vector_async(conn)

        self._pool = AsyncConnectionPool(
            self._dsn, min_size=1, max_size=10, open=False, configure=_configure
        )
        await self._pool.open(wait=True)

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()

    async def verify_connectivity(self) -> None:
        async with self._pool.connection() as conn:
            await conn.execute("SELECT 1")

    async def init_schema(self) -> None:
        from psycopg.rows import dict_row

        settings = self._settings()
        channels = resolve_vector_channels(settings)
        dim = int(settings.embedding_dim)

        async with self._pool.connection() as conn:
            # schema guard: refuse/warn when the embedding model/dim/channel set
            # changed under existing data (same signature logic as Neo4j)
            current_sig = graph.schema_signature(settings)
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                await cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS engram_meta (
                        id TEXT PRIMARY KEY, signature TEXT
                    )
                    """
                )
                await cur.execute(
                    "SELECT signature FROM engram_meta WHERE id = 'schema'"
                )
                row = await cur.fetchone()
            stored_sig = row["signature"] if row else None
            if stored_sig is not None and stored_sig != current_sig:
                if settings.schema_guard_mode == "error":
                    raise RuntimeError(_SCHEMA_MISMATCH_MSG)
                log.warning(_SCHEMA_MISMATCH_MSG)

            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    sources TEXT[] NOT NULL DEFAULT '{}',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    seq INT NOT NULL,
                    text TEXT,
                    summary TEXT,
                    keywords TEXT[] NOT NULL DEFAULT '{}',
                    -- optional BGE-M3 learned-sparse {token: weight} map; folded
                    -- into the search fused score when sparse retrieval is on
                    sparse_weights JSONB,
                    tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('english',
                            coalesce(text, '') || ' ' || coalesce(summary, ''))
                    ) STORED
                )
                """
            )
            # additive for stores created before the sparse column existed
            await conn.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS sparse_weights JSONB"
            )
            # memory write-path: near-duplicate canonical backpointer (M1)
            await conn.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS near_dup_of TEXT"
            )
            # multi-tenancy: per-chunk tenant; every chunk-surfacing read filters
            # on it for 0% cross-tenant leakage (NULL = untenanted)
            await conn.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tenant_id TEXT"
            )
            # Contextual Retrieval: the doc-situating context, plus a separate
            # generated tsvector for contextual BM25. Additive (a generated column
            # can be ADDed and back-computes for existing rows), so non-contextual
            # stores keep an empty context_tsv that never matches — zero change.
            await conn.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context TEXT"
            )
            await conn.execute(
                "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS context_tsv tsvector "
                "GENERATED ALWAYS AS (to_tsvector('english', coalesce(context, ''))) "
                "STORED"
            )
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_keywords (
                    chunk_id TEXT NOT NULL REFERENCES chunks(id) ON DELETE CASCADE,
                    keyword TEXT NOT NULL
                )
                """
            )
            # one vector column + HNSW (cosine) index per active channel
            for ch in channels:
                col = _safe_column(ch.embedding_prop)
                await conn.execute(
                    f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS {col} vector({dim})"
                )
                await conn.execute(
                    f"CREATE INDEX IF NOT EXISTS chunks_{col}_idx "
                    f"ON chunks USING hnsw ({col} vector_cosine_ops)"
                )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_context_tsv_idx "
                "ON chunks USING gin (context_tsv)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_doc_seq_idx ON chunks (doc_id, seq)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id)"
            )
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS chunk_keywords_kw_idx "
                "ON chunk_keywords (keyword)"
            )
            await conn.execute(
                """
                INSERT INTO engram_meta (id, signature) VALUES ('schema', %s)
                ON CONFLICT (id) DO UPDATE SET signature = EXCLUDED.signature
                """,
                (current_sig,),
            )

    # ── documents ────────────────────────────────────────────────────────────
    async def save_document(
        self,
        doc_id: str,
        title: str,
        sources: list[str],
        chunks: list[dict[str, Any]],
    ) -> None:
        from psycopg.types.json import Jsonb

        channels = resolve_vector_channels(self._settings())
        emb_cols = [_safe_column(ch.embedding_prop) for ch in channels]
        cols = [
            "id", "doc_id", "seq", "text", "summary", "keywords",
            "sparse_weights", "near_dup_of", "tenant_id", "context", *emb_cols,
        ]
        placeholders = ", ".join(["%s"] * len(cols))
        col_list = ", ".join(cols)
        updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "id")
        insert_sql = (
            f"INSERT INTO chunks ({col_list}) VALUES ({placeholders}) "
            f"ON CONFLICT (id) DO UPDATE SET {updates}"
        )

        async with self._pool.connection() as conn:
            await conn.execute(
                """
                INSERT INTO documents (id, title, sources) VALUES (%s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                    SET title = EXCLUDED.title, sources = EXCLUDED.sources
                """,
                (doc_id, title, sources),
            )
            chunk_rows = []
            keyword_rows = []
            for ch in chunks:
                embeddings = ch["embeddings"]
                sparse = ch.get("sparse_weights")
                values = [
                    ch["id"],
                    doc_id,
                    ch["seq"],
                    ch["text"],
                    ch["summary"],
                    ch["keywords"],
                    Jsonb(sparse) if sparse else None,
                    ch.get("near_dup_of"),
                    ch.get("tenant_id"),
                    ch.get("context"),
                ]
                for col, channel in zip(emb_cols, channels):
                    vec = embeddings[channel.embedding_prop]
                    values.append(np.asarray(vec, dtype=np.float32))
                chunk_rows.append(values)
                for kw in {k.lower() for k in ch["keywords"]}:
                    keyword_rows.append((ch["id"], kw))

            async with conn.cursor() as cur:
                await cur.executemany(insert_sql, chunk_rows)
                if keyword_rows:
                    await cur.executemany(
                        "INSERT INTO chunk_keywords (chunk_id, keyword) "
                        "VALUES (%s, %s)",
                        keyword_rows,
                    )

    async def delete_document(self, doc_id: str) -> int | None:
        async with self._pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "SELECT count(*) FROM chunks WHERE doc_id = %s", (doc_id,)
                )
                (chunk_count,) = await cur.fetchone()
                # ON DELETE CASCADE clears chunks + chunk_keywords
                await cur.execute(
                    "DELETE FROM documents WHERE id = %s RETURNING id", (doc_id,)
                )
                deleted = await cur.fetchone()
        if deleted is None:
            return None
        return chunk_count

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        from psycopg.rows import dict_row

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT sources FROM documents WHERE id = %s", (doc_id,)
                )
                doc = await cur.fetchone()
                if doc is None:
                    return None
                await cur.execute(
                    "SELECT count(*) AS n FROM chunks WHERE doc_id = %s", (doc_id,)
                )
                chunk_count = (await cur.fetchone())["n"]
                await cur.execute(
                    """
                    SELECT DISTINCT lower(ck.keyword) AS kw
                    FROM chunk_keywords ck JOIN chunks c ON c.id = ck.chunk_id
                    WHERE c.doc_id = %s ORDER BY kw
                    """,
                    (doc_id,),
                )
                keywords = [r["kw"] for r in await cur.fetchall()]
        return {
            "sources": doc["sources"],
            "chunk_count": chunk_count,
            "keywords": keywords,
        }

    async def add_document_source(self, doc_id: str, source: str) -> None:
        async with self._pool.connection() as conn:
            await conn.execute(
                """
                UPDATE documents SET sources = array_append(sources, %s)
                WHERE id = %s AND NOT (%s = ANY(sources))
                """,
                (source, doc_id, source),
            )

    async def remove_document_source(
        self, doc_id: str, source: str
    ) -> dict[str, Any] | None:
        from psycopg.rows import dict_row

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT sources FROM documents WHERE id = %s", (doc_id,)
                )
                doc = await cur.fetchone()
                if doc is None:
                    return None
                was_present = source in doc["sources"]
                remaining = [s for s in doc["sources"] if s != source]
                await cur.execute(
                    "UPDATE documents SET sources = %s WHERE id = %s",
                    (remaining, doc_id),
                )
        # keep the document unless we removed a source it actually had and that
        # was the last one — don't tear down an already-unreferenced document
        if remaining or not was_present:
            return {
                "deleted": False,
                "remaining_sources": remaining,
                "deleted_chunks": 0,
            }
        deleted_chunks = await self.delete_document(doc_id)
        return {
            "deleted": True,
            "remaining_sources": [],
            "deleted_chunks": deleted_chunks,
        }

    async def list_documents(self) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    """
                    SELECT d.id, d.title, d.sources,
                           to_char(d.created_at, 'YYYY-MM-DD"T"HH24:MI:SSOF') AS created_at,
                           count(c.id) AS chunk_count
                    FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
                    GROUP BY d.id, d.title, d.sources, d.created_at
                    ORDER BY d.created_at DESC
                    """
                )
                return list(await cur.fetchall())

    async def fetch_document_chunks(
        self, doc_id: str, embedding_props: list[str]
    ) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        cols = [_safe_column(p) for p in embedding_props]
        select = "text, summary, keywords, sparse_weights" + (
            ", " + ", ".join(cols) if cols else ""
        )
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    f"SELECT {select} FROM chunks WHERE doc_id = %s", (doc_id,)
                )
                rows = await cur.fetchall()
        out = []
        for r in rows:
            embeddings = {}
            for prop, col in zip(embedding_props, cols):
                vec = r.get(col)
                embeddings[prop] = vec.tolist() if hasattr(vec, "tolist") else vec
            out.append(
                {
                    "text": r["text"],
                    "summary": r["summary"],
                    "keywords": r["keywords"],
                    "embeddings": embeddings,
                    # JSONB comes back as a dict already
                    "sparse_weights": r.get("sparse_weights"),
                }
            )
        return out

    async def get_sparse_weights(
        self, chunk_ids: list[str]
    ) -> dict[str, dict[str, float]]:
        from psycopg.rows import dict_row

        if not chunk_ids:
            return {}
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT id, sparse_weights FROM chunks "
                    "WHERE id = ANY(%s) AND sparse_weights IS NOT NULL",
                    (chunk_ids,),
                )
                rows = await cur.fetchall()
        return {
            r["id"]: {str(k): float(v) for k, v in r["sparse_weights"].items()}
            for r in rows
            if r["sparse_weights"]
        }

    async def get_near_dup_links(self, chunk_ids: list[str]) -> dict[str, str]:
        from psycopg.rows import dict_row

        if not chunk_ids:
            return {}
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT id, near_dup_of FROM chunks "
                    "WHERE id = ANY(%s) AND near_dup_of IS NOT NULL",
                    (chunk_ids,),
                )
                rows = await cur.fetchall()
        return {r["id"]: r["near_dup_of"] for r in rows}

    # ── retrieval ────────────────────────────────────────────────────────────
    async def vector_search(
        self,
        channel: "VectorChannel",
        embedding: list[float],
        k: int,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        col = _safe_column(channel.embedding_prop)
        vec = np.asarray(embedding, dtype=np.float32)
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                # cosine distance <=> in [0,2]; similarity = 1 - distance. The
                # tenant filter is applied in-scan (the isolation guarantee). HNSW
                # collects ef_search candidates *before* applying the filter, so a
                # selective tenant in a large corpus can yield a short top-k; raise
                # ef_search (transaction-local) to over-fetch, mirroring the Neo4j
                # backend's ×tenant_overfetch.
                await self._set_tenant_ef_search(cur, k, tenant_id)
                await cur.execute(
                    f"""
                    SELECT {_CHUNK_COLS}, 1 - ({col} <=> %(vec)s) AS score
                    FROM chunks
                    WHERE {col} IS NOT NULL
                      AND (%(tenant)s::text IS NULL OR tenant_id = %(tenant)s::text)
                    ORDER BY {col} <=> %(vec)s LIMIT %(k)s
                    """,
                    {"vec": vec, "tenant": tenant_id, "k": k},
                )
                return [_row(r) for r in await cur.fetchall()]

    async def nearest_chunks(
        self,
        embedding: list[float],
        k: int,
        min_sim: float,
        exclude_doc_id: str | None = None,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        vec = np.asarray(embedding, dtype=np.float32)
        # ANN top-k via the HNSW index, then filter by min_sim (mirrors the Neo4j
        # queryNodes-then-filter path); cosine similarity = 1 - cosine distance.
        # tenant filter keeps dedup within-tenant.
        exclude_clause = "AND doc_id <> %(exclude)s" if exclude_doc_id else ""
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await self._set_tenant_ef_search(cur, k, tenant_id)
                await cur.execute(
                    f"""
                    SELECT id, doc_id, seq, text,
                           1 - (content_embedding <=> %(vec)s) AS sim
                    FROM chunks
                    WHERE content_embedding IS NOT NULL {exclude_clause}
                      AND (%(tenant)s::text IS NULL OR tenant_id = %(tenant)s::text)
                    ORDER BY content_embedding <=> %(vec)s
                    LIMIT %(k)s
                    """,
                    {"vec": vec, "k": k, "exclude": exclude_doc_id, "tenant": tenant_id},
                )
                rows = await cur.fetchall()
        return [r for r in rows if r["sim"] >= min_sim]

    async def _set_tenant_ef_search(
        self, cur: Any, k: int, tenant_id: str | None
    ) -> None:
        """Transaction-local HNSW over-fetch for a tenant-filtered vector read.

        No-op when untenanted. ef_search is clamped to pgvector's [1, 1000] range.
        SET LOCAL only affects the current transaction (the pooled connection's
        `async with` block), so it never leaks to other queries."""
        if tenant_id is None:
            return
        overfetch = max(1, int(self._settings().tenant_overfetch))
        ef = min(1000, max(int(k), int(k) * overfetch))
        # ef_search must be >= LIMIT to return a full k; pgvector ignores values
        # below 1. Identifier is fixed, value is an int we computed — safe.
        await cur.execute(f"SET LOCAL hnsw.ef_search = {ef}")

    async def fulltext_search(
        self, query: str, k: int, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        if not _sanitize_fulltext_query(query):
            return []
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                # Contextual BM25: the doc-situating context is its own generated
                # tsvector (context_tsv), matched + scored alongside the text/
                # summary tsv. Empty for non-contextual chunks, so a store without
                # Contextual Retrieval behaves exactly as before.
                await cur.execute(
                    f"""
                    SELECT {_CHUNK_COLS},
                           ts_rank(tsv, q) + ts_rank(context_tsv, q) AS score
                    FROM chunks, plainto_tsquery('english', %(query)s) q
                    WHERE (tsv @@ q OR context_tsv @@ q)
                      AND (%(tenant)s::text IS NULL OR tenant_id = %(tenant)s::text)
                    ORDER BY score DESC LIMIT %(k)s
                    """,
                    {"query": query, "k": k, "tenant": tenant_id},
                )
                return [_row(r) for r in await cur.fetchall()]

    async def fetch_siblings(
        self, seed_ids: list[str], keyword_sibling_limit: int, sequence_max_hops: int
    ) -> list[dict[str, Any]]:
        from psycopg.rows import dict_row

        hops = max(1, int(sequence_max_hops))
        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                # sequence siblings: same document, within ±hops of the seed seq
                await cur.execute(
                    """
                    SELECT seed.id AS seed_id, sib.id, sib.doc_id, sib.text,
                           sib.summary, sib.keywords, sib.content_embedding,
                           sib.tenant_id,
                           'sequence' AS via,
                           CASE WHEN sib.seq > seed.seq THEN 'after'
                                ELSE 'before' END AS direction,
                           abs(sib.seq - seed.seq) AS distance,
                           1.0::float8 AS strength
                    FROM chunks seed
                    JOIN chunks sib ON sib.doc_id = seed.doc_id
                        AND sib.seq <> seed.seq
                        AND sib.seq BETWEEN seed.seq - %(hops)s AND seed.seq + %(hops)s
                        AND NOT (sib.id = ANY(%(ids)s))
                    WHERE seed.id = ANY(%(ids)s)
                    """,
                    {"hops": hops, "ids": seed_ids},
                )
                seq_sibs = [_row(r) for r in await cur.fetchall()]

                # keyword siblings: chunks sharing keywords, top-N per seed
                await cur.execute(
                    """
                    SELECT seed_id, id, doc_id, text, summary, keywords,
                           content_embedding, tenant_id, 'keyword' AS via,
                           'lateral' AS direction, 1 AS distance, strength
                    FROM (
                        SELECT seed.id AS seed_id, sib.id, sib.doc_id, sib.text,
                               sib.summary, sib.keywords, sib.content_embedding,
                               sib.tenant_id,
                               count(*)::float8 AS strength,
                               row_number() OVER (
                                   PARTITION BY seed.id ORDER BY count(*) DESC
                               ) AS rn
                        FROM chunks seed
                        JOIN chunk_keywords sk ON sk.chunk_id = seed.id
                        JOIN chunk_keywords sks ON sks.keyword = sk.keyword
                            AND sks.chunk_id <> seed.id
                        JOIN chunks sib ON sib.id = sks.chunk_id
                        WHERE seed.id = ANY(%(ids)s)
                            AND NOT (sib.id = ANY(%(ids)s))
                        -- group by the chunk PKs; the other sib.* columns are
                        -- functionally dependent on sib.id (chunks PK), and
                        -- grouping by the vector column itself is not supported
                        GROUP BY seed.id, sib.id
                    ) ranked
                    WHERE rn <= %(kw_limit)s
                    """,
                    {"ids": seed_ids, "kw_limit": keyword_sibling_limit},
                )
                kw_sibs = [_row(r) for r in await cur.fetchall()]
        return seq_sibs + kw_sibs

    async def fetch_context(
        self, chunk_id: str, before: int, after: int
    ) -> list[dict[str, Any]] | None:
        from psycopg.rows import dict_row

        async with self._pool.connection() as conn:
            async with conn.cursor(row_factory=dict_row) as cur:
                await cur.execute(
                    "SELECT doc_id, seq FROM chunks WHERE id = %s", (chunk_id,)
                )
                anchor = await cur.fetchone()
                if anchor is None:
                    return None
                await cur.execute(
                    """
                    SELECT id, doc_id, seq, text, summary, keywords,
                           seq - %(seq)s AS offset
                    FROM chunks
                    WHERE doc_id = %(doc)s
                      AND seq BETWEEN %(seq)s - %(before)s AND %(seq)s + %(after)s
                    ORDER BY offset
                    """,
                    {
                        "doc": anchor["doc_id"],
                        "seq": anchor["seq"],
                        "before": int(before),
                        "after": int(after),
                    },
                )
                return list(await cur.fetchall())

    async def graph_proximity(
        self, seed_ids: list[str], candidate_ids: list[str], damping: float
    ) -> dict[str, float] | None:
        # no graph-activation algorithm on pgvector: signal "fall back to decay"
        return None

    # ── community synthesis (Neo4j+GDS only) ─────────────────────────────────
    async def detect_communities(
        self, min_size: int
    ) -> list[dict[str, Any]] | None:
        # no Leiden/graph-clustering on pgvector → unsupported (caller degrades)
        return None

    async def save_communities(self, communities: list[dict[str, Any]]) -> int:
        raise NotImplementedError("community synthesis requires the neo4j backend")

    async def list_communities(self) -> list[dict[str, Any]]:
        return []

    async def community_vectors(self) -> list[dict[str, Any]]:
        return []

    # ── structured-entity ingest (Neo4j-only) ────────────────────────────────
    async def upsert_entities(self, label: str, items: list[dict[str, Any]]) -> int:
        raise NotImplementedError(
            "structured-entity ingest requires the neo4j backend"
        )

    async def upsert_relations(
        self,
        from_label: str,
        rel_type: str,
        to_label: str,
        items: list[dict[str, Any]],
    ) -> int:
        raise NotImplementedError(
            "structured-entity ingest requires the neo4j backend"
        )

    # ── internals ────────────────────────────────────────────────────────────
    @staticmethod
    def _settings() -> "Settings":
        from .config import get_settings

        return get_settings()


@STORES.register("pgvector")
def _make_pgvector_store(settings: "Settings") -> PgvectorStore:
    return PgvectorStore(settings.postgres_dsn)
