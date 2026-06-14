import pytest

from app import graph
from app.channels import VectorChannel
from app.config import Settings


def test_signature_stable_and_changes_with_index_affecting_config():
    base = Settings()
    sig = graph.schema_signature(base)
    # deterministic
    assert sig == graph.schema_signature(Settings())
    # embedding model / dim are index-affecting
    assert graph.schema_signature(Settings(embedding_model="other")) != sig
    assert graph.schema_signature(Settings(embedding_dim=512)) != sig
    # adding a channel changes the signature
    extra = [
        VectorChannel(
            name="content",
            index="chunk_content_idx",
            embedding_prop="content_embedding",
            source="text",
            weight=1.0,
        ),
        VectorChannel(
            name="title",
            index="chunk_title_idx",
            embedding_prop="title_embedding",
            source="summary",
            weight=0.5,
        ),
    ]
    assert graph.schema_signature(Settings(vector_channels=extra)) != sig


def test_signature_ignores_non_index_affecting_tuning():
    # changing a fusion weight must NOT invalidate the indexes
    assert graph.schema_signature(Settings(retrieval_weight=0.4)) == graph.schema_signature(
        Settings()
    )


def test_guard_decision_first_run_and_match_are_fine():
    assert graph._schema_guard_decision(None, "sig", "error") is None
    assert graph._schema_guard_decision("sig", "sig", "error") is None


def test_guard_decision_error_mode_raises_on_mismatch():
    with pytest.raises(RuntimeError, match="signature mismatch"):
        graph._schema_guard_decision("old", "new", "error")


def test_guard_decision_warn_and_off_override():
    assert graph._schema_guard_decision("old", "new", "warn") == "warn"
    assert graph._schema_guard_decision("old", "new", "off") == "off"
