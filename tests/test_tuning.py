import pytest

from app.config import SEARCH_TUNABLE_FIELDS, Settings


def test_tuned_empty_returns_self():
    s = Settings()
    assert s.tuned({}) is s


def test_tuned_applies_and_coerces_without_mutating_original():
    s = Settings()
    t = s.tuned({"final_top_k": 3, "hyde_query_weight": "0.25"})
    assert t.final_top_k == 3
    assert t.hyde_query_weight == 0.25  # JSON string coerced to float
    # original is untouched (per-request copy)
    assert s.final_top_k == 8
    assert s.hyde_query_weight == 0.5


def test_tuned_rejects_non_tunable_field():
    with pytest.raises(ValueError, match="non-tunable.*neo4j_password"):
        Settings().tuned({"neo4j_password": "x"})


def test_sensitive_and_ingest_fields_are_not_tunable():
    for field in (
        "neo4j_password",
        "neo4j_uri",
        "embedding_api_base",
        "llm_api_key",
        "embedding_dim",
        "embedding_model",
        "chunk_strategy",
        "metadata_extractor",
    ):
        assert field not in SEARCH_TUNABLE_FIELDS
