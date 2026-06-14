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
