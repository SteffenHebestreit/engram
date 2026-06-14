from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

from .channels import VectorChannel
from .profiles import GraphProfile


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # graph database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "engram"

    # startup guard when the embedding model/dim or channel set no longer match
    # the indexes already in the store: "error" = refuse to start, "warn" = log
    # and adopt, "off" = silently adopt
    schema_guard_mode: str = "error"

    # embedding endpoint (OpenAI-compatible /embeddings)
    embedding_api_base: str = "http://localhost:8080/v1"
    embedding_api_key: str = ""
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024

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

    # chunking
    # which registered Chunker strategy to use (see app/chunking.py); "fixed"
    # is the built-in paragraph/sentence window splitter
    chunk_strategy: str = "fixed"
    chunk_target_chars: int = 1800
    chunk_overlap_chars: int = 200

    # which registered MetadataExtractor to use (see app/llm.py); "default"
    # produces a one-sentence summary + keywords per chunk
    metadata_extractor: str = "default"

    # ingestion throughput: parallel LLM metadata calls; embedding requests
    # are split into batches (servers cap per-request batch size) with a
    # bounded number in flight
    extraction_concurrency: int = 4
    embedding_batch_size: int = 64
    embedding_concurrency: int = 4

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
        "rerank_top_k",
        "final_top_k",
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
        "rerank_top_k",
        "final_top_k",
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
