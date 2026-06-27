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
    # SEARCH_TUNABLE_FIELDS); e.g. {"hyde_enabled": false, "final_top_k": 5}.
    # A "preset" key selects a named bundle (cheap/balanced/max_quality, see
    # app/presets.py); explicit fields here override the preset.
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
    # which retrieval channels surfaced this chunk (e.g. ["content", "fulltext"]
    # or ["graph:keyword"]) — per-stage provenance for the eval harness
    channels: list[str] = []
    # if this chunk is a near-duplicate of an earlier-ingested one (memory
    # write-path), the canonical chunk id it was linked to; null otherwise
    near_dup_of: str | None = None
    # edges between this chunk and the nearest direct hit (0 = direct hit);
    # graph_proximity decays with this distance and feeds the fused score
    graph_distance: int
    graph_proximity: float
    retrieval_score: float
    median_score: float
    # learned-sparse (BGE-M3 lexical) exact-term score in [0, 1]; 0.0 unless
    # sparse retrieval is enabled and this chunk carries stored sparse weights
    sparse_score: float = 0.0
    fused_score: float
    rerank_score: float
    # exponential recency factor in (0, 1] on the chunk's document age (1.0 = new),
    # blended into the final ordering when recency scoring is enabled; 0.0 otherwise
    recency_score: float = 0.0


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class EvalCase(BaseModel):
    query_id: str
    query: str = Field(min_length=1)
    # documents that are relevant to this query (binary relevance / golden set)
    relevant_document_ids: list[str]


class EvalRequest(BaseModel):
    """A golden set to score engram's retrieval against — judge-free, on your
    own corpus (see app/eval.py)."""
    cases: list[EvalCase] = Field(min_length=1)
    k_values: list[int] = [10]
    top_k: int = 50
    # per-request search overrides applied to every case (SEARCH_TUNABLE_FIELDS),
    # so an eval can compare configs (e.g. {"sparse_enabled": true})
    tuning: dict[str, object] | None = None


class MetricValue(BaseModel):
    mean: float
    ci95: list[float]  # [lo, hi] percentile bootstrap 95% interval


class ChannelAttribution(BaseModel):
    # how many relevant docs were retrieved, and which channels surfaced them —
    # `unique_to_channel` is the gold hits a channel found that no other did
    gold_hits_retrieved: int
    by_channel: dict[str, int]
    unique_to_channel: dict[str, int]


class EvalResponse(BaseModel):
    n_queries: int
    metrics: dict[str, MetricValue]
    attribution: ChannelAttribution


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


class CommunityInfo(BaseModel):
    id: str
    # LLM-written report; empty when reports were skipped or the LLM was down
    title: str = ""
    summary: str = ""
    keywords: list[str] = []
    member_count: int
    # cosine relevance to the query, only set by POST /communities/search
    score: float | None = None


class CommunityBuildResponse(BaseModel):
    communities: int


class CommunitySearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None
