import pytest

from app.config import SEARCH_TUNABLE_FIELDS, Settings
from app.presets import PRESETS, apply_preset


def test_every_preset_is_tunable_and_resolves():
    # each preset references only tunable fields and produces a valid Settings
    for name, overlay in PRESETS.items():
        assert set(overlay) <= SEARCH_TUNABLE_FIELDS, name
        Settings().tuned(overlay)  # must not raise


def test_balanced_is_a_noop():
    assert apply_preset({"preset": "balanced"}) == {}


def test_explicit_fields_win_over_preset():
    merged = apply_preset({"preset": "cheap", "hyde_enabled": True})
    assert merged["hyde_enabled"] is True          # explicit beats the preset
    assert merged["reranker_enabled"] is False      # still from the preset
    assert "preset" not in merged                   # meta-key stripped


def test_env_default_preset_applies_when_no_key():
    merged = apply_preset({"final_top_k": 3}, default_preset="cheap")
    assert merged["final_top_k"] == 3               # explicit kept
    assert merged["reranker_enabled"] is False      # from the default preset


def test_none_or_empty_tuning_yields_empty():
    assert apply_preset(None) == {}
    assert apply_preset({}) == {}


def test_unknown_preset_raises():
    with pytest.raises(ValueError, match="unknown preset 'nope'"):
        apply_preset({"preset": "nope"})
