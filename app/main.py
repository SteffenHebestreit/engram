from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query

from .config import get_settings
from .ingest import ingest_document
from .community import build_communities, search_communities
from .models import (
    ChunkContextResponse,
    CommunityBuildResponse,
    CommunityInfo,
    CommunitySearchRequest,
    ContextChunk,
    DocumentDeleteResponse,
    DocumentInfo,
    EntityUpsertRequest,
    EvalRequest,
    EvalResponse,
    FeedbackRequest,
    FeedbackResponse,
    GraphWriteResponse,
    IngestRequest,
    IngestResponse,
    RelationUpsertRequest,
    SearchRequest,
    SearchResponse,
)
from .observability import configure_logging
from .registry import load_entrypoints
from .search import search
from .store import create_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    # discover any third-party strategy/store plugins before building the store
    load_entrypoints()
    app.state.store = create_store(settings)
    await app.state.store.connect()
    app.state.http = httpx.AsyncClient(
        timeout=httpx.Timeout(settings.request_timeout),
        limits=httpx.Limits(max_connections=settings.http_max_connections),
    )
    await app.state.store.init_schema()
    yield
    await app.state.http.aclose()
    await app.state.store.close()


app = FastAPI(title="engram", version="0.4.0", lifespan=lifespan)


@app.get("/health")
async def health():
    await app.state.store.verify_connectivity()
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    try:
        doc_id, chunk_count, keywords = await ingest_document(
            app.state.store,
            app.state.http,
            req.text,
            req.title,
            req.source,
            req.document_id,
            req.tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return IngestResponse(document_id=doc_id, chunk_count=chunk_count, keywords=keywords)


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    rows = await app.state.store.list_documents()
    return [DocumentInfo(**row) for row in rows]


@app.delete("/documents/{doc_id}", response_model=DocumentDeleteResponse)
async def delete_document(doc_id: str, source: str | None = Query(None)):
    """Remove a document.

    With `?source=`, drop just that source's reference (reference-counted): the
    nodes are deleted only when it was the last source still holding the
    document. Without `source`, hard-delete the document regardless of how many
    sources reference it.
    """
    if source is None:
        deleted = await app.state.store.delete_document(doc_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="document not found")
        return DocumentDeleteResponse(
            document_id=doc_id, deleted=True, deleted_chunks=deleted
        )

    result = await app.state.store.remove_document_source(doc_id, source)
    if result is None:
        raise HTTPException(status_code=404, detail="document not found")
    return DocumentDeleteResponse(
        document_id=doc_id,
        deleted=result["deleted"],
        deleted_chunks=result["deleted_chunks"],
        remaining_sources=result["remaining_sources"],
    )


@app.post("/graph/entities", response_model=GraphWriteResponse)
async def upsert_entities_endpoint(req: EntityUpsertRequest):
    """Load typed domain entity nodes (key + properties) into the graph."""
    try:
        n = await app.state.store.upsert_entities(
            req.label, [item.model_dump() for item in req.items]
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return GraphWriteResponse(upserted=n)


@app.post("/graph/relations", response_model=GraphWriteResponse)
async def upsert_relations_endpoint(req: RelationUpsertRequest):
    """Load typed relations between existing entity nodes into the graph."""
    try:
        n = await app.state.store.upsert_relations(
            req.from_label,
            req.type,
            req.to_label,
            [item.model_dump() for item in req.items],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return GraphWriteResponse(upserted=n)


@app.post("/communities/rebuild", response_model=CommunityBuildResponse)
async def rebuild_communities(reports: bool = Query(True)):
    """Rebuild the community/theme layer (GraphRAG-style): cluster the chunk
    graph, optionally write LLM reports, and persist it. Offline-ish — run it
    after ingest, not on the search path. Neo4j + GDS only."""
    try:
        result = await build_communities(
            app.state.store, app.state.http, generate_reports=reports
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc))
    return CommunityBuildResponse(**result)


@app.get("/communities", response_model=list[CommunityInfo])
async def list_communities():
    rows = await app.state.store.list_communities()
    return [CommunityInfo(**row) for row in rows]


@app.post("/communities/search", response_model=list[CommunityInfo])
async def search_communities_endpoint(req: CommunitySearchRequest):
    """Global/theme search: rank the community reports against the query."""
    rows = await search_communities(
        app.state.store, app.state.http, req.query, req.top_k or 10
    )
    return [CommunityInfo(**row) for row in rows]


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    try:
        results = await search(
            app.state.store,
            app.state.http,
            req.query,
            req.top_k,
            req.tuning,
            req.tenant_id,
        )
    except ValueError as exc:
        # e.g. a non-tunable field in req.tuning
        raise HTTPException(status_code=422, detail=str(exc))
    return SearchResponse(query=req.query, results=results)


@app.post("/eval", response_model=EvalResponse)
async def eval_endpoint(req: EvalRequest):
    """Judge-free retrieval evaluation against a golden set, on your own corpus.

    Returns standard IR metrics (nDCG@k / Recall@k / P@k / MAP) with bootstrap
    confidence intervals, plus **per-channel attribution**: which retrieval
    channels surfaced each recovered gold hit, and which it found *uniquely*
    (lost without that channel). Reproducible, no LLM judge. Pass `tuning` to
    score a specific config (e.g. sparse on vs off). See app/eval.py.
    """
    from .eval import run_evaluation

    golden = {c.query_id: {d: 1 for d in c.relevant_document_ids} for c in req.cases}
    queries = {c.query_id: c.query for c in req.cases}
    try:
        report = await run_evaluation(
            app.state.store,
            app.state.http,
            golden,
            queries,
            ks=tuple(req.k_values),
            top_k=req.top_k,
            tuning=req.tuning,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return EvalResponse(**report)


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback_endpoint(req: FeedbackRequest):
    """Record which chunks an agent actually grounded its answer on (implicit-
    relevance feedback). engram persists the (query → used-chunk) positives so an
    offline job can mine hard negatives + tune fusion weights — the agent-in-the-
    loop learning signal a stateless retriever can't capture. See app/store.py."""
    recorded = await app.state.store.record_feedback(
        req.query, req.used_chunk_ids, req.query_id
    )
    return FeedbackResponse(recorded=recorded)


@app.get("/chunks/{chunk_id}/context", response_model=ChunkContextResponse)
async def chunk_context(
    chunk_id: str,
    before: int = Query(1, ge=0, le=10),
    after: int = Query(1, ge=0, le=10),
):
    """Neighbouring chunks along the document's NEXT_CHUNK chain, for callers
    that need more surrounding context for a search result."""
    rows = await app.state.store.fetch_context(chunk_id, before, after)
    if rows is None:
        raise HTTPException(status_code=404, detail="chunk not found")
    return ChunkContextResponse(
        chunk_id=chunk_id,
        chunks=[
            ContextChunk(
                chunk_id=row["id"],
                document_id=row["doc_id"],
                seq=row["seq"],
                offset=row["offset"],
                text=row["text"],
                summary=row["summary"] or "",
                keywords=row["keywords"] or [],
            )
            for row in rows
        ],
    )
