from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Query

from . import graph
from .ingest import ingest_document
from .models import (
    ChunkContextResponse,
    ContextChunk,
    DocumentDeleteResponse,
    DocumentInfo,
    EntityUpsertRequest,
    GraphWriteResponse,
    IngestRequest,
    IngestResponse,
    RelationUpsertRequest,
    SearchRequest,
    SearchResponse,
)
from .search import search


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.driver = graph.create_driver()
    app.state.http = httpx.AsyncClient()
    await graph.init_schema(app.state.driver)
    yield
    await app.state.http.aclose()
    await app.state.driver.close()


app = FastAPI(title="engram", version="0.1.2", lifespan=lifespan)


@app.get("/health")
async def health():
    await app.state.driver.verify_connectivity()
    return {"status": "ok"}


@app.post("/documents", response_model=IngestResponse)
async def ingest(req: IngestRequest):
    try:
        doc_id, chunk_count, keywords = await ingest_document(
            app.state.driver,
            app.state.http,
            req.text,
            req.title,
            req.source,
            req.document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return IngestResponse(document_id=doc_id, chunk_count=chunk_count, keywords=keywords)


@app.get("/documents", response_model=list[DocumentInfo])
async def list_documents():
    rows = await graph.list_documents(app.state.driver)
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
        deleted = await graph.delete_document(app.state.driver, doc_id)
        if deleted is None:
            raise HTTPException(status_code=404, detail="document not found")
        return DocumentDeleteResponse(
            document_id=doc_id, deleted=True, deleted_chunks=deleted
        )

    result = await graph.remove_document_source(app.state.driver, doc_id, source)
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
        n = await graph.upsert_entities(
            app.state.driver, req.label, [item.model_dump() for item in req.items]
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return GraphWriteResponse(upserted=n)


@app.post("/graph/relations", response_model=GraphWriteResponse)
async def upsert_relations_endpoint(req: RelationUpsertRequest):
    """Load typed relations between existing entity nodes into the graph."""
    try:
        n = await graph.upsert_relations(
            app.state.driver,
            req.from_label,
            req.type,
            req.to_label,
            [item.model_dump() for item in req.items],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return GraphWriteResponse(upserted=n)


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest):
    try:
        results = await search(
            app.state.driver, app.state.http, req.query, req.top_k, req.tuning
        )
    except ValueError as exc:
        # e.g. a non-tunable field in req.tuning
        raise HTTPException(status_code=422, detail=str(exc))
    return SearchResponse(query=req.query, results=results)


@app.get("/chunks/{chunk_id}/context", response_model=ChunkContextResponse)
async def chunk_context(
    chunk_id: str,
    before: int = Query(1, ge=0, le=10),
    after: int = Query(1, ge=0, le=10),
):
    """Neighbouring chunks along the document's NEXT_CHUNK chain, for callers
    that need more surrounding context for a search result."""
    rows = await graph.fetch_context(app.state.driver, chunk_id, before, after)
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
