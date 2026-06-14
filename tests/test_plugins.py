import pytest

from app.chunking import CHUNKERS, get_chunker
from app.config import Settings
from app.llm import EXTRACTORS, extract_metadata, get_extractor
from app.registry import Registry


def test_registry_register_and_get():
    reg: Registry[int] = Registry("thing")
    reg.register("a", 1)

    @reg.register("b")
    def _b():  # decorator form returns the object unchanged
        return 2

    assert reg.get("a") == 1
    assert reg.get("b") is _b
    assert reg.names() == ["a", "b"]
    assert "a" in reg and "z" not in reg


def test_registry_unknown_key_lists_options():
    reg: Registry[int] = Registry("widget")
    reg.register("only", 1)
    with pytest.raises(KeyError, match="unknown widget 'missing'.*only"):
        reg.get("missing")


def test_default_chunker_matches_fixed_window():
    settings = Settings(chunk_target_chars=1800, chunk_overlap_chars=200)
    chunker = get_chunker("fixed")
    assert chunker("First paragraph.\n\nSecond paragraph.", settings) == [
        "First paragraph.\n\nSecond paragraph."
    ]


def test_custom_chunker_is_selectable():
    @CHUNKERS.register("one-per-line")
    def _lines(text, settings):
        return [ln for ln in text.splitlines() if ln.strip()]

    try:
        chunker = get_chunker("one-per-line")
        assert chunker("a\nb\n\nc", Settings()) == ["a", "b", "c"]
    finally:
        CHUNKERS._items.pop("one-per-line", None)


def test_default_extractor_is_registered():
    assert get_extractor("default") is extract_metadata
    assert "default" in EXTRACTORS
