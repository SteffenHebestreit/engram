from app import embeddings as emb_mod
from app.config import Settings


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeClient:
    """Records each /embeddings request; embeds every text as [len(text)].

    Items are returned in reverse order to exercise the index-based reorder.
    """

    def __init__(self):
        self.batches: list[list[str]] = []

    async def post(self, url, json=None, headers=None, timeout=None):
        texts = json["input"]
        self.batches.append(list(texts))
        data = [
            {"index": i, "embedding": [float(len(t))]}
            for i, t in enumerate(texts)
        ]
        return FakeResponse({"data": list(reversed(data))})


def _patch_settings(monkeypatch, **overrides):
    settings = Settings(**overrides)
    monkeypatch.setattr(emb_mod, "get_settings", lambda: settings)


async def test_small_input_is_a_single_request(monkeypatch):
    _patch_settings(monkeypatch, embedding_batch_size=64)
    client = FakeClient()
    result = await emb_mod.embed_texts(client, ["a", "bb", "ccc"])
    assert client.batches == [["a", "bb", "ccc"]]
    assert result == [[1.0], [2.0], [3.0]]


async def test_large_input_is_split_and_order_preserved(monkeypatch):
    _patch_settings(monkeypatch, embedding_batch_size=2, embedding_concurrency=2)
    texts = ["x" * n for n in range(1, 8)]  # 7 texts -> batches of 2,2,2,1
    client = FakeClient()
    result = await emb_mod.embed_texts(client, texts)
    assert [len(b) for b in client.batches] == [2, 2, 2, 1]
    assert sorted(t for b in client.batches for t in b) == sorted(texts)
    # output order matches input order despite batching and reversed replies
    assert result == [[float(n)] for n in range(1, 8)]


async def test_empty_input_makes_no_request(monkeypatch):
    _patch_settings(monkeypatch)
    client = FakeClient()
    assert await emb_mod.embed_texts(client, []) == []
    assert client.batches == []


class FakeSparseClient:
    """Returns BGE-M3-style lexical weights, items in reverse to test reorder."""

    def __init__(self, payload=None, raises=None):
        self.calls: list[dict] = []
        self._raises = raises
        self._payload = payload

    async def post(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        if self._raises is not None:
            raise self._raises
        if self._payload is not None:
            return FakeResponse(self._payload)
        data = [
            {"index": i, "lexical_weights": {str(i): 1.0, "99": float(len(t))}}
            for i, t in enumerate(json["input"])
        ]
        return FakeResponse({"data": list(reversed(data))})


async def test_embed_sparse_returns_weights_in_input_order(monkeypatch):
    _patch_settings(monkeypatch, sparse_api_base="http://sparse:9000")
    client = FakeSparseClient()
    out = await emb_mod.embed_sparse_texts(client, ["a", "bbb"])
    assert out == [{"0": 1.0, "99": 1.0}, {"1": 1.0, "99": 3.0}]
    assert client.calls[0]["url"].endswith("/embed_sparse")


async def test_embed_sparse_without_endpoint_returns_none(monkeypatch):
    _patch_settings(monkeypatch, sparse_api_base="")
    client = FakeSparseClient()
    assert await emb_mod.embed_sparse_texts(client, ["a"]) is None
    assert client.calls == []  # no request attempted


async def test_embed_sparse_degrades_to_none_on_failure(monkeypatch):
    _patch_settings(monkeypatch, sparse_api_base="http://sparse:9000")
    client = FakeSparseClient(raises=RuntimeError("endpoint down"))
    assert await emb_mod.embed_sparse_texts(client, ["a"]) is None


async def test_embed_sparse_empty_input_makes_no_request(monkeypatch):
    _patch_settings(monkeypatch, sparse_api_base="http://sparse:9000")
    client = FakeSparseClient()
    assert await emb_mod.embed_sparse_texts(client, []) == []
    assert client.calls == []


async def test_embed_sparse_single_text_helper(monkeypatch):
    _patch_settings(monkeypatch, sparse_api_base="http://sparse:9000")
    client = FakeSparseClient()
    assert await emb_mod.embed_sparse_text(client, "a") == {"0": 1.0, "99": 1.0}
