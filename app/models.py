from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    title: str = ""
    # required: the source/context referencing this document. Documents are
    # reference-counted by their sources so they can always be cleaned up
    # (DELETE /documents/{id}?source=…) and never leave orphaned chunks.
    source: str = Field(min_length=1)
    # optional stable handle for this document; if omitted the id is the
    # SHA-256 of the text. Re-ingesting with the same id replaces the previous
    # version. Use it (or the content hash) to DELETE /documents/{id} later
    # without having stored the id we returned.
    document_id: str | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    keywords: list[str]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None
    # per-request overrides of search-shaping settings (see
    # SEARCH_TUNABLE_FIELDS); e.g. {"hyde_enabled": false, "final_top_k": 5}
    tuning: dict[str, object] | None = None


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    summary: str
    keywords: list[str]
    # how the chunk entered the candidate pool:
    # "vector", "fulltext" or "sibling:<via>:<before|after|lateral>"
    origin: str
    # edges between this chunk and the nearest direct hit (0 = direct hit);
    # graph_proximity decays with this distance and feeds the fused score
    graph_distance: int
    graph_proximity: float
    retrieval_score: float
    median_score: float
    fused_score: float
    rerank_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class ContextChunk(BaseModel):
    chunk_id: str
    document_id: str
    seq: int
    # position relative to the requested chunk: negative = before, 0 = the
    # chunk itself, positive = after
    offset: int
    text: str
    summary: str
    keywords: list[str]


class ChunkContextResponse(BaseModel):
    chunk_id: str
    chunks: list[ContextChunk]


class DocumentInfo(BaseModel):
    id: str
    title: str | None = None
    # every source/context that references this document (see reference-counted
    # deletion); the nodes live until the last one is removed
    sources: list[str] = []
    created_at: str | None = None
    chunk_count: int


class DocumentDeleteResponse(BaseModel):
    document_id: str
    # True when the document's nodes were actually removed (hard delete, or the
    # last source reference was dropped); False when other sources still hold it
    deleted: bool
    deleted_chunks: int | None = None
    remaining_sources: list[str] = []


class EntityItem(BaseModel):
    key: str = Field(min_length=1)
    properties: dict[str, object] | None = None


class EntityUpsertRequest(BaseModel):
    label: str = Field(min_length=1)
    items: list[EntityItem] = Field(min_length=1)


class RelationItem(BaseModel):
    from_key: str = Field(min_length=1)
    to_key: str = Field(min_length=1)
    properties: dict[str, object] | None = None


class RelationUpsertRequest(BaseModel):
    from_label: str = Field(min_length=1)
    type: str = Field(min_length=1)
    to_label: str = Field(min_length=1)
    items: list[RelationItem] = Field(min_length=1)


class GraphWriteResponse(BaseModel):
    upserted: int
