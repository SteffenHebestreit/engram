from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from .channels import VectorChannel
from .profiles import GraphProfile


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # storage backend: "neo4j" (graph + vector in one store, with GDS
    # personalized-PageRank proximity) or "pgvector" (PostgreSQL + pgvector — a
    # lighter graph-lite alternative; sequence/keyword siblings still work, but
    # graph proximity degrades to per-hop decay since there is no GDS PPR)
    store_backend: str = "neo4j"

    # graph database (Neo4j backend)
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "engram"

    # PostgreSQL + pgvector backend (used when store_backend == "pgvector")
    postgres_dsn: str = "postgresql://engram:engram@localhost:5432/engram"

    # startup guard when the embedding model/dim or channel set no longer match
    # the indexes already in the store: "error" = refuse to start, "warn" = log
    # and adopt, "off" = silently adopt
    schema_guard_mode: str = "error"

    # root log level (DEBUG surfaces a per-search timing/diagnostics summary)
    log_level: str = "INFO"

    # community synthesis (GraphRAG-style theme layer, neo4j+GDS only): smallest
    # Leiden community kept when building the community layer
    community_min_size: int = 1

    # embedding endpoint (OpenAI-compatible /embeddings)
    embedding_api_base: str = "http://localhost:8080/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

    # asymmetric retrieval instructions for instruction-tuned embedders
    # (E5 / GTE / Qwen3-Embedding / NV-Embed / bge-*-v1.5, ...). These models
    # expect a short task instruction prepended to the QUERY (and sometimes the
    # passage) and measurably lose recall without it — engram embeds both sides
    # identically by default, which is correct for BGE-M3 (the default, needs no
    # instruction) but wrong for those models. Both default to "" = today's exact
    # behavior. Set per your embedder's model card, e.g.
    #   E5:               query="query: "    passage="passage: "
    #   bge-v1.5 (en):    query="Represent this sentence for searching relevant passages: "
    #   Qwen3-Embedding:  query="Instruct: Given a query, retrieve relevant passages\nQuery: "
    # Applied to the dense vector channels only; the BM25/fulltext channel always
    # keeps the raw text. `passage_instruction` changes stored vectors, so it is
    # part of the schema signature (changing it trips the startup guard); the
    # query side is query-time only.
    query_instruction: str = ""
    passage_instruction: str = ""

    # learned-sparse (BGE-M3 lexical) retrieval signal — opt-in. BGE-M3 already
    # produces a sparse term-weight vector alongside its dense one; engram
    # normally discards it. When enabled, chunk + query sparse weights are
    # fetched from a multi-output endpoint and folded into the fused score as an
    # exact-term signal (measured +4 nDCG@10 over dense alone). It re-scores the
    # candidate pool, so no sparse index is needed; the dense path is untouched.
    # Endpoint contract: POST {sparse_api_base}/embed_sparse {"input": [texts]}
    # -> {"data": [{"index": i, "lexical_weights": {token: weight}}]}.
    sparse_enabled: bool = False
    sparse_api_base: str = ""
    sparse_api_key: str = ""
    sparse_model: str = "BAAI/bge-m3"
    # weight of the sparse signal in the fused score (alongside retrieval/median/
    # graph-proximity); only applied when sparse_enabled and weights are present
    sparse_weight: float = 0.2

    # memory write-path: near-duplicate handling (M1). When enabled, ingest links
    # a fresh chunk that is >= dedup_cosine_threshold cosine-similar to an existing
    # chunk in *another* document (the canonical) via `near_dup_of`, and search
    # collapses near-duplicate clusters to their best-scored member so an agent
    # re-ingesting the same knowledge (paraphrased, across sources/sessions) can't
    # flood the candidate pool with redundant chunks. Non-destructive: the
    # duplicate chunk is still stored and linked, never dropped (so a false link
    # is recoverable, not a factual deletion). The threshold is EMBEDDER-COUPLED
    # (BGE-M3 + the passage instruction) — calibrate it on your own vectors;
    # boilerplate/templated text and near-identical-but-different numbers sit high.
    dedup_enabled: bool = False
    dedup_cosine_threshold: float = 0.95
    dedup_candidate_k: int = 5

    # LLM endpoint for metadata extraction (OpenAI-compatible /chat/completions)
    llm_api_base: str = "http://localhost:8000/v1"
    llm_api_key: str = ""
    llm_model: str = "qwen2.5-14b-instruct"
    llm_json_mode: bool = True

    # reranker endpoint
    reranker_api_base: str = "http://localhost:8081"
    reranker_api_key: str = ""
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_format: str = "tei"  # "tei" or "jina"

    # ColBERT late-interaction reranker (select with reranker_strategy="colbert").
    # Scores the MMR shortlist by MaxSim over BGE-M3 multi-vectors — the same
    # model engram already runs. It is NOT a quality lift over the default
    # cross-encoder (bge-reranker-v2-m3 is generally stronger); its win is being
    # ~100x cheaper, so it's the fast late-interaction option (RAGFlow's headline
    # feature) for latency-sensitive or reranker-on-CPU deployments. Needs a
    # multi-vector endpoint; degrades to the fused score when unavailable.
    # Contract: POST {colbert_api_base}/rerank_colbert {"query", "texts": [...]}
    # -> {"data": [{"index": i, "score": float}]} (or a bare [{"index","score"}]).
    colbert_api_base: str = ""
    colbert_api_key: str = ""
    colbert_model: str = "BAAI/bge-m3"
    # reranker stage: disable to skip the cross-encoder round trip entirely
    # (results then fall back to the fused score, exactly like a reranker-down);
    # reranker_strategy selects a registered reranker (see app/rerank.py)
    reranker_enabled: bool = True
    reranker_strategy: str = "http"

    # HTTP client tuning for the embedding/LLM/reranker calls: per-request
    # timeout (seconds), a shorter timeout for the optional HyDE generation, and
    # the connection-pool ceiling on the shared client
    request_timeout: float = 120.0
    hyde_timeout: float = 30.0
    http_max_connections: int = 64

    # chunking
    # which registered Chunker strategy to use (see app/chunking.py); "fixed"
    # is the built-in paragraph/sentence window splitter
    chunk_strategy: str = "fixed"
    chunk_target_chars: int = 1800
    # chunk overlap defaults to 0: engram retrieves a hit's neighbouring chunks
    # via the NEXT_CHUNK graph (sequence expansion), so the seam context an
    # overlap would duplicate is already recovered — overlap is redundant here
    # (unlike naive RAG, which needs it). Raise it if you run without graph
    # expansion or want a small safety margin.
    chunk_overlap_chars: int = 0

    # which registered MetadataExtractor to use (see app/llm.py); "default"
    # produces a one-sentence summary + keywords per chunk
    metadata_extractor: str = "default"

    # ingestion throughput: parallel LLM metadata calls; embedding requests
    # are split into batches (servers cap per-request batch size) with a
    # bounded number in flight
    extraction_concurrency: int = 4
    embedding_batch_size: int = 64
    embedding_concurrency: int = 4
    # incremental re-ingest: when replacing a document, reuse the stored
    # embeddings + metadata of any chunk whose text is byte-identical to one of
    # the document's existing chunks, so only changed chunks pay for fresh LLM
    # extraction + embedding (identical text => identical metadata/vectors)
    reuse_unchanged_chunks: bool = True

    # search tuning
    top_k_per_index: int = 12
    seed_count: int = 8
    keyword_sibling_limit: int = 5
    retrieval_weight: float = 0.55
    median_weight: float = 0.30
    # weight of the graph-proximity indication value (1.0 for direct hits,
    # decaying with edge distance for expanded siblings) in the fused score
    graph_proximity_weight: float = 0.15
    rerank_top_k: int = 15
    final_top_k: int = 8

    # MMR shortlist selection: trade relevance (lambda) against redundancy
    # with already-picked candidates (1 - lambda); 1.0 disables the penalty
    mmr_lambda: float = 0.7

    # autocut: drop final results after the first rerank-score cliff
    autocut_enabled: bool = True
    autocut_min_keep: int = 3
    autocut_min_gap: float = 0.25  # of the min-max normalized score range

    # HyDE: for short queries, embed an LLM-written hypothetical answer
    # blended with the query embedding (lexical channel keeps the raw query)
    hyde_enabled: bool = True
    hyde_max_query_words: int = 8
    hyde_query_weight: float = 0.5  # share of the original query in the blend

    # default search preset applied when a /search request names none (see
    # app/presets.py): "" = no preset, or "cheap" / "balanced" / "max_quality".
    # A per-request `tuning.preset` and explicit tuning fields override it.
    search_preset: str = ""

    # pipeline strategy selection (see app/pipeline.py); defaults reproduce the
    # original inline pipeline
    fusion_strategy: str = "dbsf_convex"
    expander_strategy: str = "sequence_keyword"

    # graph schema profile driving the GDS projection (see app/profiles.py) as
    # JSON; None = the built-in document profile (Chunk/Keyword over
    # NEXT_CHUNK/HAS_KEYWORD). Extend it to project domain entity nodes/
    # relations loaded via the /graph/entities + /graph/relations endpoints
    graph_profile: GraphProfile | None = None

    # graph proximity: "ppr" = personalized PageRank over the chunk/keyword
    # graph via Neo4j GDS (falls back to "decay" when GDS is unavailable),
    # "decay" = fixed per-hop decay
    graph_proximity_mode: str = "ppr"
    ppr_damping: float = 0.85

    # per-channel weights when fusing the default three vector indexes; these
    # feed the built-in channel set (see app/channels.py)
    content_channel_weight: float = 1.0
    summary_channel_weight: float = 0.9
    keywords_channel_weight: float = 0.8
    # disable the summary/keywords vector channels to cut embeddings-per-chunk
    # (content-only => 1 embedding/chunk, the naive-baseline cost). Changing the
    # active channel set changes the schema signature, so flip these on a fresh
    # store / before the first ingest. Ignored when vector_channels is set.
    summary_channel_enabled: bool = True
    keywords_channel_enabled: bool = True
    # full override of the vector channel set as JSON (e.g.
    # VECTOR_CHANNELS='[{"name":"content","index":"chunk_content_idx",
    # "embedding_prop":"content_embedding","source":"text","weight":1.0}]');
    # None = use the built-in three weighted from the settings above
    vector_channels: list[VectorChannel] | None = None
    # weight of the fulltext (BM25-style) channel after max-normalization
    fulltext_channel_weight: float = 0.7

    # graph expansion: sequence neighbours are followed along the directional
    # NEXT_CHUNK chain up to N hops; proximity = decay ** hops, so siblings
    # farther from the seed carry a lower indication value
    sequence_max_hops: int = 3
    sequence_proximity_decay: float = 0.7
    keyword_sibling_base_decay: float = 0.4
    keyword_sibling_shared_bonus: float = 0.1  # per shared keyword, capped at 3

    def tuned(self, overrides: dict) -> "Settings":
        """A copy of these settings with per-request overrides applied.

        Only `SEARCH_TUNABLE_FIELDS` may be overridden; anything else raises so
        a caller cannot retarget an endpoint or change the embedding dimension
        at query time. Values are re-validated/coerced, so JSON ints/strings
        land as the right type.
        """
        if not overrides:
            return self
        bad = set(overrides) - SEARCH_TUNABLE_FIELDS
        if bad:
            raise ValueError(
                f"non-tunable fields: {sorted(bad)}; "
                f"tunable: {sorted(SEARCH_TUNABLE_FIELDS)}"
            )
        return Settings.model_validate({**self.model_dump(), **overrides})


# fields a caller may override per /search request (search-shaping only — never
# endpoints, credentials, embedding dim/model, or ingest-time settings)
SEARCH_TUNABLE_FIELDS: frozenset[str] = frozenset(
    {
        "top_k_per_index",
        "seed_count",
        "keyword_sibling_limit",
        "retrieval_weight",
        "median_weight",
        "graph_proximity_weight",
        "sparse_enabled",
        "sparse_weight",
        "dedup_enabled",
        "rerank_top_k",
        "final_top_k",
        "reranker_enabled",
        "reranker_strategy",
        "mmr_lambda",
        "autocut_enabled",
        "autocut_min_keep",
        "autocut_min_gap",
        "hyde_enabled",
        "hyde_max_query_words",
        "hyde_query_weight",
        "fusion_strategy",
        "expander_strategy",
        "graph_proximity_mode",
        "ppr_damping",
        "content_channel_weight",
        "summary_channel_weight",
        "keywords_channel_weight",
        "fulltext_channel_weight",
        "vector_channels",
        "sequence_max_hops",
        "sequence_proximity_decay",
        "keyword_sibling_base_decay",
        "keyword_sibling_shared_bonus",
    }
)


@lru_cache
def get_settings() -> Settings:
    return Settings()
