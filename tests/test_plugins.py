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


async def test_none_extractor_makes_no_llm_call():
    # client=None proves no HTTP request is attempted
    result = await get_extractor("none")(None, "some chunk text")
    assert result.summary == ""
    assert result.keywords == []


async def test_yake_extractor_pulls_keywords_without_an_llm():
    # client=None proves no HTTP request; YAKE is purely statistical
    text = (
        "Apollo 11 was the spaceflight that first landed humans on the Moon. "
        "Neil Armstrong and Buzz Aldrin landed the lunar module Eagle."
    )
    result = await get_extractor("yake")(None, text)
    assert result.keywords  # extracted something
    assert all(isinstance(k, str) for k in result.keywords)
    assert result.summary.startswith("Apollo 11")  # first sentence


class _FakeResp:
    def __init__(self, content: str):
        self._content = content

    def raise_for_status(self):
        pass

    def json(self):
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": {"completion_tokens": 5},
        }


class _FakeClient:
    """Records outbound /chat/completions requests instead of sending them."""

    def __init__(self, content='{"keywords": ["a", "b", "c"], "summary": "A summary."}'):
        self.calls: list[dict] = []
        self._content = content

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResp(self._content)


async def test_extract_metadata_uses_shared_endpoint_by_default(monkeypatch):
    from app import llm as llm_mod

    monkeypatch.setattr(
        llm_mod, "get_settings",
        lambda: Settings(llm_api_base="http://shared/v1", llm_model="big"),
    )
    client = _FakeClient()
    result = await extract_metadata(client, "a reasonably long chunk of text to extract")
    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["url"] == "http://shared/v1/chat/completions"
    assert call["json"]["model"] == "big"
    assert "max_tokens" not in call["json"]  # uncapped by default (byte-identical)
    assert call["json"]["response_format"] == {"type": "json_object"}  # inherits llm_json_mode
    assert result.keywords == ["a", "b", "c"]


async def test_extract_metadata_json_schema_and_extra_body(monkeypatch):
    from app import llm as llm_mod

    monkeypatch.setattr(
        llm_mod, "get_settings",
        lambda: Settings(
            llm_api_base="http://x/v1", llm_model="m",
            extraction_response_format="json_schema",
            extraction_extra_body={"reasoning_effort": "none"},
        ),
    )
    client = _FakeClient()
    await extract_metadata(client, "a reasonably long chunk of text to extract")
    body = client.calls[0]["json"]
    assert body["response_format"]["type"] == "json_schema"  # LM Studio / constrained-decode path
    assert body["response_format"]["json_schema"]["name"] == "extraction"
    assert body["reasoning_effort"] == "none"  # extra_body merged at the top level


async def test_extract_metadata_prefers_separate_extraction_endpoint(monkeypatch):
    from app import llm as llm_mod

    monkeypatch.setattr(
        llm_mod, "get_settings",
        lambda: Settings(
            llm_api_base="http://shared/v1", llm_model="big",
            extraction_llm_api_base="http://small/v1", extraction_llm_model="small",
            extraction_llm_api_key="sk-x", extraction_max_tokens=96,
        ),
    )
    client = _FakeClient()
    await extract_metadata(client, "a reasonably long chunk of text to extract")
    call = client.calls[0]
    assert call["url"] == "http://small/v1/chat/completions"  # extraction endpoint wins
    assert call["json"]["model"] == "small"
    assert call["json"]["max_tokens"] == 96
    assert call["headers"]["Authorization"] == "Bearer sk-x"


async def test_extract_metadata_length_gate_skips_the_llm(monkeypatch):
    from app import llm as llm_mod

    monkeypatch.setattr(
        llm_mod, "get_settings", lambda: Settings(extraction_min_chars=100)
    )
    client = _FakeClient()
    result = await extract_metadata(client, "short header")
    assert client.calls == []  # no LLM round trip for a sub-threshold chunk
    assert result.summary == "short header"  # yake fallback: first sentence
