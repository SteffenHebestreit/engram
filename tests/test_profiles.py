import pytest

from app.config import Settings
from app.profiles import DEFAULT_PROFILE, GraphProfile, RelationSpec, resolve_profile


def test_default_profile_reproduces_document_projection():
    labels, rel_config = DEFAULT_PROFILE.projection_spec()
    assert labels == ["Chunk", "Keyword"]
    assert rel_config == {
        "NEXT_CHUNK": {"orientation": "UNDIRECTED"},
        "HAS_KEYWORD": {"orientation": "UNDIRECTED"},
    }


def test_resolve_profile_defaults_then_override():
    assert resolve_profile(Settings()) is DEFAULT_PROFILE
    custom = GraphProfile(
        name="mfa",
        projection_labels=["Chunk", "EbmCode"],
        projection_relationships=[
            RelationSpec(type="ABOUT", orientation="UNDIRECTED"),
            RelationSpec(type="EXCLUDES_SAME_QUARTAL", sign=-1, weight=2.0),
        ],
    )
    assert resolve_profile(Settings(graph_profile=custom)) is custom


def test_relation_spec_orientation_validation():
    with pytest.raises(ValueError, match="orientation must be one of"):
        RelationSpec(type="X", orientation="SIDEWAYS")
    # accepted case-insensitively, normalized upper
    assert RelationSpec(type="X", orientation="natural").orientation == "NATURAL"


def test_relation_spec_sign_and_weight_carry_through_projection_labels():
    spec = RelationSpec(type="EXCLUDES", sign=-1, weight=2.0)
    assert (spec.sign, spec.weight) == (-1, 2.0)
    profile = GraphProfile(
        name="p", projection_labels=["A"], projection_relationships=[spec]
    )
    labels, rel_config = profile.projection_spec()
    assert labels == ["A"]
    assert rel_config == {"EXCLUDES": {"orientation": "UNDIRECTED"}}
