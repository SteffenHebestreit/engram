import httpx

from app import rerank as rerank_mod
from app.config import Settings


class _Resp:
    def __init__(self, payload=None, status_error=False):
        self._payload = payload
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise RuntimeError("HTTP 500")

    def json(self):
        return self._payload


class _Client:
    def __init__(self, resp=None, exc=None):
        self._resp = resp
        self._exc = exc
        self.calls = []

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json})
        if self._exc:
            raise self._exc
        return self._resp


async def test_empty_texts_returns_empty_list():
    assert await rerank_mod.rerank(_Client(), "q", []) == []


async def test_tei_format_restores_input_order(monkeypatch):
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: Settings(reranker_format="tei"))
    resp = _Resp(payload=[{"index": 1, "score": 0.3}, {"index": 0, "score": 0.9}])
    client = _Client(resp=resp)
    assert await rerank_mod.rerank(client, "q", ["a", "b"]) == [0.9, 0.3]
    assert client.calls[0]["json"] == {"query": "q", "texts": ["a", "b"]}


async def test_jina_format_parsed(monkeypatch):
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: Settings(reranker_format="jina"))
    resp = _Resp(
        payload={"results": [
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.2},
        ]}
    )
    assert await rerank_mod.rerank(_Client(resp=resp), "q", ["a", "b"]) == [0.8, 0.2]


async def test_connection_error_returns_none(monkeypatch):
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: Settings())
    client = _Client(exc=httpx.ConnectError("reranker down"))
    assert await rerank_mod.rerank(client, "q", ["a"]) is None


async def test_http_error_status_returns_none(monkeypatch):
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: Settings())
    assert await rerank_mod.rerank(_Client(resp=_Resp(status_error=True)), "q", ["a"]) is None


async def test_malformed_payload_returns_none(monkeypatch):
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: Settings())
    # missing "index" -> KeyError during parsing -> caught -> None
    assert await rerank_mod.rerank(_Client(resp=_Resp(payload=[{"score": 0.5}])), "q", ["a"]) is None


async def test_colbert_empty_texts_returns_empty_list():
    assert await rerank_mod.rerank_colbert(_Client(), "q", []) == []


async def test_colbert_without_endpoint_returns_none(monkeypatch):
    monkeypatch.setattr(rerank_mod, "get_settings", lambda: Settings(colbert_api_base=""))
    client = _Client()
    assert await rerank_mod.rerank_colbert(client, "q", ["a"]) is None
    assert client.calls == []  # no request attempted when unconfigured


async def test_colbert_scores_restored_to_input_order(monkeypatch):
    monkeypatch.setattr(
        rerank_mod, "get_settings",
        lambda: Settings(colbert_api_base="http://colbert:9001"),
    )
    resp = _Resp(payload={"data": [
        {"index": 1, "score": 0.4}, {"index": 0, "score": 0.95},
    ]})
    client = _Client(resp=resp)
    assert await rerank_mod.rerank_colbert(client, "q", ["a", "b"]) == [0.95, 0.4]
    assert client.calls[0]["url"].endswith("/rerank_colbert")
    assert client.calls[0]["json"]["texts"] == ["a", "b"]


async def test_colbert_accepts_bare_list_payload(monkeypatch):
    monkeypatch.setattr(
        rerank_mod, "get_settings",
        lambda: Settings(colbert_api_base="http://colbert:9001"),
    )
    resp = _Resp(payload=[{"index": 0, "score": 0.7}, {"index": 1, "score": 0.1}])
    assert await rerank_mod.rerank_colbert(_Client(resp=resp), "q", ["a", "b"]) == [0.7, 0.1]


async def test_colbert_endpoint_down_returns_none(monkeypatch):
    monkeypatch.setattr(
        rerank_mod, "get_settings",
        lambda: Settings(colbert_api_base="http://colbert:9001"),
    )
    client = _Client(exc=httpx.ConnectError("colbert down"))
    assert await rerank_mod.rerank_colbert(client, "q", ["a"]) is None
