import httpx
import pytest

from app import mcp_server


class _Resp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload, record):
        self._payload = payload
        self._record = record

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json=None):
        self._record.append(("POST", url, json))
        return _Resp(self._payload)

    async def get(self, url, params=None):
        self._record.append(("GET", url, params))
        return _Resp(self._payload)


def _patch_client(monkeypatch, payload, record):
    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **k: _FakeClient(payload, record))


async def test_proxy_search_posts_to_search(monkeypatch):
    record = []
    _patch_client(monkeypatch, {"results": [{"chunk_id": "x", "rerank_score": 0.9}]}, record)
    out = await mcp_server.proxy_search("hello", top_k=5, preset="cheap")
    assert out == [{"chunk_id": "x", "rerank_score": 0.9}]
    method, url, body = record[0]
    assert method == "POST"
    assert url.endswith("/search")
    assert body == {"query": "hello", "top_k": 5, "tuning": {"preset": "cheap"}}


async def test_proxy_search_omits_optional_fields(monkeypatch):
    record = []
    _patch_client(monkeypatch, {"results": []}, record)
    await mcp_server.proxy_search("q")
    assert record[0][2] == {"query": "q", "top_k": 8}  # no tuning key when no preset


async def test_proxy_chunk_context_gets_context(monkeypatch):
    record = []
    _patch_client(monkeypatch, {"chunk_id": "c", "chunks": [{"chunk_id": "c"}]}, record)
    out = await mcp_server.proxy_chunk_context("c", before=1, after=3)
    assert out == [{"chunk_id": "c"}]
    method, url, params = record[0]
    assert method == "GET"
    assert url.endswith("/chunks/c/context")
    assert params == {"before": 1, "after": 3}


async def test_proxy_search_themes(monkeypatch):
    record = []
    _patch_client(monkeypatch, [{"id": "1", "title": "T"}], record)
    out = await mcp_server.proxy_search_themes("themes?", top_k=3)
    assert out == [{"id": "1", "title": "T"}]
    _, url, body = record[0]
    assert url.endswith("/communities/search")
    assert body == {"query": "themes?", "top_k": 3}


async def test_proxy_mark_used_posts_feedback(monkeypatch):
    record = []
    _patch_client(monkeypatch, {"recorded": 2}, record)
    out = await mcp_server.proxy_mark_used("q", ["a", "b"], query_id="qid")
    assert out == {"recorded": 2}
    method, url, body = record[0]
    assert method == "POST"
    assert url.endswith("/feedback")
    assert body == {"query": "q", "used_chunk_ids": ["a", "b"], "query_id": "qid"}


async def test_proxy_mark_used_omits_query_id(monkeypatch):
    record = []
    _patch_client(monkeypatch, {"recorded": 1}, record)
    await mcp_server.proxy_mark_used("q", ["a"])
    assert record[0][2] == {"query": "q", "used_chunk_ids": ["a"]}  # no query_id key


def test_build_server_registers_tools():
    pytest.importorskip("mcp")  # only when the optional MCP dep is installed
    server = mcp_server.build_server()
    assert server is not None
