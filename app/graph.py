import hashlib
import json
import logging
import re
from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from .channels import resolve_vector_channels
from .config import get_settings
from .profiles import resolve_profile

log = logging.getLogger(__name__)

FULLTEXT_INDEX = "chunk_fulltext"

# in-memory GDS projection used for personalized PageRank
PPR_GRAPH = "engram_ppr"

# marker node holding the index schema signature (see schema_signature)
SCHEMA_META_ID = "schema"

_SCHEMA_MISMATCH_MSG = (
    "vector index schema signature mismatch: the existing indexes were built "
    "for a different embedding model, dimension, or channel set. Re-ingest "
    "after wiping the store (e.g. docker compose down -v), or set "
    "SCHEMA_GUARD_MODE=warn/off to override."
)

# None = not yet probed; probed once per process, since the plugin set only
# changes with a database restart
_gds_available: bool | None = None


def create_driver() -> AsyncDriver:
    settings = get_settings()
    return AsyncGraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password)
    )


def schema_signature(settings: Any) -> str:
    """Stable hash of the config that the vector indexes were built against.

    Covers the embedding model + dimension and the (index, property) of each
    channel. A change here means existing indexes no longer match how new
    content would be embedded/stored, so serving against them silently returns
    wrong results.
    """
    payload = {
        "embedding_model": settings.embedding_model,
        "embedding_dim": int(settings.embedding_dim),
        "similarity": "cosine",
        "channels": sorted(
            (c.index, c.embedding_prop) for c in resolve_vector_channels(settings)
        ),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


def _schema_guard_decision(
    stored: str | None, current: str, mode: str
) -> str | None:
    """Resolve the startup guard. Returns None when fine (first run or match),
    "warn"/"off" on an overridden mismatch, and raises in "error" mode."""
    if stored is None or stored == current:
        return None
    if mode == "error":
        raise RuntimeError(_SCHEMA_MISMATCH_MSG)
    return "warn" if mode == "warn" else "off"


async def _read_schema_signature(driver: AsyncDriver) -> str | None:
    async with driver.session() as session:
        result = await session.run(
            "MATCH (m:EngramMeta {id: $id}) RETURN m.signature AS sig",
            id=SCHEMA_META_ID,
        )
        record = await result.single(strict=False)
        return record["sig"] if record else None


async def _write_schema_signature(driver: AsyncDriver, signature: str) -> None:
    async with driver.session() as session:
        await session.run(
            "MERGE (m:EngramMeta {id: $id}) "
            "SET m.signature = $sig, m.updated_at = datetime()",
            id=SCHEMA_META_ID,
            sig=signature,
        )


async def init_schema(driver: AsyncDriver) -> None:
    settings = get_settings()

    # guard: refuse (or warn) if the indexes were built for a different
    # embedding model / dimension / channel set before serving stale results
    current_sig = schema_signature(settings)
    stored_sig = await _read_schema_signature(driver)
    if _schema_guard_decision(stored_sig, current_sig, settings.schema_guard_mode):
        log.warning(_SCHEMA_MISMATCH_MSG)

    constraints = [
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
        "CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE",
    ]
    async with driver.session() as session:
        for stmt in constraints:
            await session.run(stmt)
        for channel in resolve_vector_channels(settings):
            # OPTIONS does not accept query parameters; dim is an int from config
            # and index/prop names come from trusted config, not user input
            await session.run(
                f"""
                CREATE VECTOR INDEX {channel.index} IF NOT EXISTS
                FOR (c:Chunk) ON (c.{channel.embedding_prop})
                OPTIONS {{indexConfig: {{
                    `vector.dimensions`: {int(settings.embedding_dim)},
                    `vector.similarity_function`: 'cosine'
                }}}}
                """
            )
        await session.run(
            f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS "
            "FOR (c:Chunk) ON EACH [c.text, c.summary]"
        )

    # record the signature these indexes were built against (first run adopts;
    # warn/off adopt the new one; error mode already raised above on mismatch)
    await _write_schema_signature(driver, current_sig)


async def save_document(
    driver: AsyncDriver,
    doc_id: str,
    title: str,
    sources: list[str],
    chunks: list[dict[str, Any]],
) -> None:
    """Persist a document with its chunks, embeddings, keywords and relations.

    `sources` is the set of references that pulled this document in (a document
    can be contributed by several sources/contexts); the document's nodes are
    only torn down once every source has been removed (see
    `remove_document_source`).

    Each chunk dict needs: id, seq, text, summary, keywords, and an
    `embeddings` map of {embedding_prop: vector} (one entry per active vector
    channel, e.g. content_embedding/summary_embedding/keywords_embedding).

    Relations created:
      (Chunk)-[:PART_OF]->(Document)
      (Chunk)-[:NEXT_CHUNK]->(Chunk)      sequential order within the document
      (Chunk)-[:HAS_KEYWORD]->(Keyword)   shared Keyword nodes connect chunks
                                          across the whole graph
    """
    async with driver.session() as session:
        await session.run(
            """
            MERGE (d:Document {id: $doc_id})
            ON CREATE SET d.created_at = datetime()
            SET d.title = $title, d.sources = $sources
            """,
            doc_id=doc_id,
            title=title,
            sources=sources,
        )
        await session.run(
            """
            MATCH (d:Document {id: $doc_id})
            UNWIND $chunks AS row
            MERGE (c:Chunk {id: row.id})
            SET c.doc_id = $doc_id,
                c.seq = row.seq,
                c.text = row.text,
                c.summary = row.summary,
                c.keywords = row.keywords
            SET c += row.embeddings
            MERGE (c)-[:PART_OF]->(d)
            WITH c, row
            UNWIND row.keywords AS kw
            MERGE (k:Keyword {name: toLower(kw)})
            MERGE (c)-[:HAS_KEYWORD]->(k)
            """,
            doc_id=doc_id,
            chunks=chunks,
        )
        await session.run(
            """
            MATCH (c:Chunk {doc_id: $doc_id})
            WITH c ORDER BY c.seq
            WITH collect(c) AS ordered
            UNWIND range(0, size(ordered) - 2) AS i
            WITH ordered[i] AS a, ordered[i + 1] AS b
            MERGE (a)-[:NEXT_CHUNK]->(b)
            """,
            doc_id=doc_id,
        )
    # the GDS projection (if any) is a snapshot; force a re-project
    await invalidate_ppr_projection(driver)


async def delete_document(driver: AsyncDriver, doc_id: str) -> int | None:
    """Delete a document, its chunks and any keywords left orphaned.

    Returns the number of deleted chunks, or None if the document does not exist.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {id: $id})
            OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d)
            WITH d, collect(c) AS chunks
            DETACH DELETE d
            FOREACH (c IN chunks | DETACH DELETE c)
            RETURN size(chunks) AS chunk_count
            """,
            id=doc_id,
        )
        record = await result.single(strict=False)
        if record is None:
            return None
        await session.run("MATCH (k:Keyword) WHERE NOT (k)<-[:HAS_KEYWORD]-() DELETE k")
    await invalidate_ppr_projection(driver)
    return record["chunk_count"]


async def vector_search(
    driver: AsyncDriver, index_name: str, embedding: list[float], k: int
) -> list[dict[str, Any]]:
    """Query one vector index; returns chunk fields plus the index score [0, 1]."""
    async with driver.session() as session:
        result = await session.run(
            """
            CALL db.index.vector.queryNodes($index_name, $k, $embedding)
            YIELD node, score
            RETURN node.id AS id, node.doc_id AS doc_id, node.text AS text,
                   node.summary AS summary, node.keywords AS keywords,
                   node.content_embedding AS content_embedding, score
            """,
            index_name=index_name,
            k=k,
            embedding=embedding,
        )
        return [dict(record) async for record in result]


async def fulltext_search(
    driver: AsyncDriver, query: str, k: int
) -> list[dict[str, Any]]:
    """BM25-style lexical search over chunk text and summaries.

    Scores are Lucene scores (unbounded); callers must normalize before
    fusing with the [0, 1] vector channel scores.
    """
    sanitized = _sanitize_fulltext_query(query)
    if not sanitized:
        return []
    async with driver.session() as session:
        result = await session.run(
            """
            CALL db.index.fulltext.queryNodes($index, $q, {limit: $k})
            YIELD node, score
            RETURN node.id AS id, node.doc_id AS doc_id, node.text AS text,
                   node.summary AS summary, node.keywords AS keywords,
                   node.content_embedding AS content_embedding, score
            """,
            index=FULLTEXT_INDEX,
            q=sanitized,
            k=k,
        )
        return [dict(record) async for record in result]


def _sanitize_fulltext_query(query: str) -> str:
    """Strip Lucene syntax characters so user input is treated as plain terms."""
    return re.sub(r"[^\w\s]", " ", query).strip()


_IDENTIFIER_RE = re.compile(r"[^A-Za-z0-9_]")


def _sanitize_identifier(value: str, kind: str) -> str:
    """Whitelist a node label / relationship type for safe interpolation.

    Labels and relationship types cannot be query parameters in Cypher, so
    structured-ingest callers' values are reduced to `[A-Za-z0-9_]` before they
    are interpolated. Properties and key values always travel as parameters.
    """
    cleaned = _IDENTIFIER_RE.sub("", value or "")
    if not cleaned:
        raise ValueError(f"invalid {kind}: {value!r}")
    return cleaned


async def upsert_entities(
    driver: AsyncDriver, label: str, items: list[dict[str, Any]]
) -> int:
    """Upsert typed domain nodes into the graph (structured-entity ingest).

    Each item is `{key, properties?}`; nodes are identified by `label` + `key`.
    Lets a domain graph (e.g. typed entities with their own relations) be
    loaded alongside chunks without the chunk -> LLM -> embed pipeline, so the
    graph profile's projection can spread PageRank through it. Returns the
    number of rows written.
    """
    safe_label = _sanitize_identifier(label, "label")
    rows = [
        {"key": str(item["key"]), "properties": item.get("properties") or {}}
        for item in items
    ]
    if not rows:
        return 0
    async with driver.session() as session:
        result = await session.run(
            f"""
            UNWIND $rows AS row
            MERGE (n:`{safe_label}` {{key: row.key}})
            SET n += row.properties, n.updated_at = datetime()
            RETURN count(n) AS n
            """,
            rows=rows,
        )
        record = await result.single()
    await invalidate_ppr_projection(driver)
    return record["n"]


async def upsert_relations(
    driver: AsyncDriver,
    from_label: str,
    rel_type: str,
    to_label: str,
    items: list[dict[str, Any]],
) -> int:
    """Upsert typed relations between existing nodes (structured-entity ingest).

    Each item is `{from_key, to_key, properties?}`; both endpoint nodes must
    already exist (load them with `upsert_entities` first). Returns the number
    of relations written.
    """
    fl = _sanitize_identifier(from_label, "from_label")
    tl = _sanitize_identifier(to_label, "to_label")
    rt = _sanitize_identifier(rel_type, "type")
    rows = [
        {
            "from_key": str(item["from_key"]),
            "to_key": str(item["to_key"]),
            "properties": item.get("properties") or {},
        }
        for item in items
    ]
    if not rows:
        return 0
    async with driver.session() as session:
        result = await session.run(
            f"""
            UNWIND $rows AS row
            MATCH (a:`{fl}` {{key: row.from_key}})
            MATCH (b:`{tl}` {{key: row.to_key}})
            MERGE (a)-[r:`{rt}`]->(b)
            SET r += row.properties, r.updated_at = datetime()
            RETURN count(r) AS n
            """,
            rows=rows,
        )
        record = await result.single()
    await invalidate_ppr_projection(driver)
    return record["n"]


async def document_exists(driver: AsyncDriver, doc_id: str) -> bool:
    """Whether a Document with this id is already in the store."""
    async with driver.session() as session:
        result = await session.run(
            "MATCH (d:Document {id: $id}) RETURN count(d) AS n", id=doc_id
        )
        record = await result.single()
        return record["n"] > 0


async def get_document(driver: AsyncDriver, doc_id: str) -> dict[str, Any] | None:
    """Existing document's sources, chunk count and distinct keywords, or None.

    Lets a content-addressed re-ingest (same text -> same id) skip re-embedding
    identical content and just register the new source.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {id: $id})
            OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d)
            RETURN coalesce(d.sources, []) AS sources, count(c) AS chunk_count,
                   collect(c.keywords) AS keyword_lists
            """,
            id=doc_id,
        )
        record = await result.single(strict=False)
    if record is None:
        return None
    keywords = sorted(
        {kw.lower() for lst in record["keyword_lists"] if lst for kw in lst}
    )
    return {
        "sources": record["sources"],
        "chunk_count": record["chunk_count"],
        "keywords": keywords,
    }


async def add_document_source(driver: AsyncDriver, doc_id: str, source: str) -> None:
    """Register another source as a reference to an existing document (no-op if
    already present)."""
    async with driver.session() as session:
        await session.run(
            """
            MATCH (d:Document {id: $id})
            SET d.sources = CASE
                WHEN $source IN coalesce(d.sources, []) THEN d.sources
                ELSE coalesce(d.sources, []) + $source END
            """,
            id=doc_id,
            source=source,
        )


async def remove_document_source(
    driver: AsyncDriver, doc_id: str, source: str
) -> dict[str, Any] | None:
    """Drop one source's reference. When it was the last one, the document's
    chunks, edges and orphaned keywords are deleted; otherwise the document is
    kept with its remaining sources.

    Returns {deleted, remaining_sources, deleted_chunks} or None if no such
    document.
    """
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document {id: $id})
            WITH d, $source IN coalesce(d.sources, []) AS was_present
            SET d.sources = [s IN coalesce(d.sources, []) WHERE s <> $source]
            RETURN d.sources AS sources, was_present
            """,
            id=doc_id,
            source=source,
        )
        record = await result.single(strict=False)
    if record is None:
        return None
    remaining = record["sources"]
    # keep the document unless we removed a source it actually had and that was
    # the last one — don't tear down an already-unreferenced document
    if remaining or not record["was_present"]:
        return {"deleted": False, "remaining_sources": remaining, "deleted_chunks": 0}
    deleted_chunks = await delete_document(driver, doc_id)
    return {"deleted": True, "remaining_sources": [], "deleted_chunks": deleted_chunks}


async def list_documents(driver: AsyncDriver) -> list[dict[str, Any]]:
    async with driver.session() as session:
        result = await session.run(
            """
            MATCH (d:Document)
            OPTIONAL MATCH (c:Chunk)-[:PART_OF]->(d)
            WITH d, count(c) AS chunk_count
            RETURN d.id AS id, d.title AS title,
                   coalesce(d.sources, []) AS sources,
                   toString(d.created_at) AS created_at, chunk_count
            ORDER BY d.created_at DESC
            """
        )
        return [dict(record) async for record in result]


async def fetch_siblings(
    driver: AsyncDriver,
    chunk_ids: list[str],
    keyword_sibling_limit: int,
    sequence_max_hops: int,
) -> list[dict[str, Any]]:
    """Graph expansion around seed chunks.

    Sequence neighbours are walked along the directional NEXT_CHUNK chain up
    to `sequence_max_hops` in both directions; `distance` is the hop count and
    `direction` says whether the sibling comes 'before' or 'after' the seed in
    the document. Keyword siblings (chunks sharing keywords, strongest first)
    always have distance 1 and direction 'lateral'; `strength` is the number
    of shared keywords (1.0 for sequence siblings). Seeds themselves are
    excluded.
    """
    # variable-length bounds cannot be query parameters; hops is an int from config
    hops = max(1, int(sequence_max_hops))
    async with driver.session() as session:
        result = await session.run(
            f"""
            MATCH (seed:Chunk) WHERE seed.id IN $ids
            CALL (seed) {{
                MATCH p = (seed)-[:NEXT_CHUNK*1..{hops}]->(sib:Chunk)
                WHERE NOT sib.id IN $ids
                RETURN sib, 'sequence' AS via, 'after' AS direction,
                       length(p) AS distance, 1.0 AS strength
                UNION
                MATCH p = (sib:Chunk)-[:NEXT_CHUNK*1..{hops}]->(seed)
                WHERE NOT sib.id IN $ids
                RETURN sib, 'sequence' AS via, 'before' AS direction,
                       length(p) AS distance, 1.0 AS strength
                UNION
                MATCH (seed)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(sib:Chunk)
                WHERE NOT sib.id IN $ids
                WITH sib, count(k) AS shared
                ORDER BY shared DESC
                LIMIT $kw_limit
                RETURN sib, 'keyword' AS via, 'lateral' AS direction,
                       1 AS distance, toFloat(shared) AS strength
            }}
            RETURN seed.id AS seed_id, sib.id AS id, sib.doc_id AS doc_id,
                   sib.text AS text, sib.summary AS summary,
                   sib.keywords AS keywords,
                   sib.content_embedding AS content_embedding,
                   via, direction, distance, strength
            """,
            ids=chunk_ids,
            kw_limit=keyword_sibling_limit,
        )
        return [dict(record) async for record in result]


async def gds_available(driver: AsyncDriver) -> bool:
    """Whether the Graph Data Science plugin is installed (probed once)."""
    global _gds_available
    if _gds_available is None:
        try:
            async with driver.session() as session:
                result = await session.run("RETURN gds.version() AS v")
                await result.single()
            _gds_available = True
        except Exception:
            _gds_available = False
    return _gds_available


async def invalidate_ppr_projection(driver: AsyncDriver) -> None:
    """Drop the in-memory PPR graph so the next search re-projects it.

    Called after ingest/delete; the GDS projection is a snapshot and does not
    see store updates.
    """
    if not await gds_available(driver):
        return
    async with driver.session() as session:
        # YIELD a concrete column: the procedure's full row contains a
        # deprecated `schema` field that triggers a driver warning otherwise
        await session.run(
            "CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName",
            name=PPR_GRAPH,
        )


async def _ensure_ppr_projection(driver: AsyncDriver) -> bool:
    """Project Chunk/Keyword nodes with undirected NEXT_CHUNK/HAS_KEYWORD
    relationships into GDS memory if not already there. False if the store
    has no chunks yet (nothing to project)."""
    async with driver.session() as session:
        result = await session.run(
            "CALL gds.graph.exists($name) YIELD exists RETURN exists",
            name=PPR_GRAPH,
        )
        record = await result.single()
        if record["exists"]:
            return True
        result = await session.run("MATCH (c:Chunk) RETURN count(c) AS n")
        record = await result.single()
        if record["n"] == 0:
            return False
        # node labels + relationship config come from the active graph profile
        # and are passed as parameters (injection-safe); the default profile
        # reproduces the original Chunk/Keyword + NEXT_CHUNK/HAS_KEYWORD graph
        labels, rel_config = resolve_profile(get_settings()).projection_spec()
        await session.run(
            "CALL gds.graph.project($name, $labels, $rel_config)",
            name=PPR_GRAPH,
            labels=labels,
            rel_config=rel_config,
        )
        return True


async def ppr_proximity(
    driver: AsyncDriver,
    seed_ids: list[str],
    candidate_ids: list[str],
    damping: float,
) -> dict[str, float] | None:
    """Graph proximity per candidate via personalized PageRank seeded on the
    direct hits (HippoRAG-style activation spreading).

    Each candidate's PPR score is normalized by the strongest seed's score and
    capped at 1.0, so the value is comparable to the decay-based proximity.
    Chunks reachable from several seeds over several paths accumulate
    activation, which a single best-path decay cannot express.

    Returns None when GDS is unavailable or there is nothing to rank, letting
    the caller fall back to decay-based proximity.
    """
    if not seed_ids or not candidate_ids:
        return None
    if not await gds_available(driver):
        return None
    try:
        if not await _ensure_ppr_projection(driver):
            return None
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (seed:Chunk) WHERE seed.id IN $seed_ids
                WITH collect(seed) AS seeds
                CALL gds.pageRank.stream($graph, {
                    sourceNodes: seeds,
                    dampingFactor: $damping,
                    maxIterations: 20
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS n, score
                WHERE n:Chunk AND (n.id IN $seed_ids OR n.id IN $cand_ids)
                RETURN n.id AS id, score, n.id IN $seed_ids AS is_seed
                """,
                seed_ids=seed_ids,
                cand_ids=candidate_ids,
                graph=PPR_GRAPH,
                damping=damping,
            )
            rows = [dict(record) async for record in result]
    except Exception:
        # projection raced with an ingest drop, plugin misconfigured, ...:
        # degrade to decay-based proximity rather than failing the search
        return None

    max_seed = max((r["score"] for r in rows if r["is_seed"]), default=0.0)
    if max_seed <= 0:
        return None
    return {
        r["id"]: min(1.0, r["score"] / max_seed)
        for r in rows
        if not r["is_seed"]
    }


async def fetch_context(
    driver: AsyncDriver, chunk_id: str, before: int, after: int
) -> list[dict[str, Any]] | None:
    """Sequence window around one chunk via the directional NEXT_CHUNK chain.

    Returns the chunk itself (offset 0) plus up to `before` predecessors
    (negative offsets) and `after` successors (positive offsets) in document
    order, or None if the chunk does not exist.
    """
    branches = ["RETURN c AS n, 0 AS offset"]
    if before > 0:
        branches.append(
            f"MATCH p = (n:Chunk)-[:NEXT_CHUNK*1..{int(before)}]->(c)\n"
            "                RETURN n, -length(p) AS offset"
        )
    if after > 0:
        branches.append(
            f"MATCH p = (c)-[:NEXT_CHUNK*1..{int(after)}]->(n:Chunk)\n"
            "                RETURN n, length(p) AS offset"
        )
    subquery = "\n                UNION\n                ".join(branches)
    async with driver.session() as session:
        result = await session.run(
            f"""
            MATCH (c:Chunk {{id: $id}})
            CALL (c) {{
                {subquery}
            }}
            RETURN n.id AS id, n.doc_id AS doc_id, n.seq AS seq,
                   n.text AS text, n.summary AS summary,
                   n.keywords AS keywords, offset
            ORDER BY offset
            """,
            id=chunk_id,
        )
        rows = [dict(record) async for record in result]
        return rows or None
