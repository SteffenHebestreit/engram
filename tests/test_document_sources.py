import pytest

from app import graph


class _Result:
    def __init__(self, record):
        self._record = record

    async def single(self, strict=True):
        return self._record


class _Session:
    def __init__(self, record, calls):
        self.record = record
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.calls.append((query, params))
        return _Result(self.record)


class _Driver:
    """Fake driver whose every query returns one configured record."""

    def __init__(self, record):
        self.record = record
        self.calls = []

    def session(self):
        return _Session(self.record, self.calls)


@pytest.fixture
def captured_delete(monkeypatch):
    deleted = []

    async def fake_delete(driver, doc_id):
        deleted.append(doc_id)
        return 5

    monkeypatch.setattr(graph, "delete_document", fake_delete)
    return deleted


async def test_remove_source_keeps_document_when_others_remain(captured_delete):
    driver = _Driver({"sources": ["b"], "was_present": True})
    res = await graph.remove_document_source(driver, "doc1", "a")
    assert res == {"deleted": False, "remaining_sources": ["b"], "deleted_chunks": 0}
    assert captured_delete == []  # not torn down


async def test_remove_last_source_deletes_document(captured_delete):
    driver = _Driver({"sources": [], "was_present": True})
    res = await graph.remove_document_source(driver, "doc1", "a")
    assert res == {"deleted": True, "remaining_sources": [], "deleted_chunks": 5}
    assert captured_delete == ["doc1"]


async def test_remove_absent_source_never_deletes(captured_delete):
    # source was not referencing the doc — empty array must NOT trigger a delete
    driver = _Driver({"sources": [], "was_present": False})
    res = await graph.remove_document_source(driver, "doc1", "ghost")
    assert res == {"deleted": False, "remaining_sources": [], "deleted_chunks": 0}
    assert captured_delete == []


async def test_remove_source_unknown_document_returns_none(captured_delete):
    driver = _Driver(None)  # MATCH found nothing
    assert await graph.remove_document_source(driver, "missing", "a") is None
    assert captured_delete == []


async def test_get_document_flattens_and_lowercases_keywords():
    driver = _Driver(
        {
            "sources": ["a"],
            "chunk_count": 2,
            "keyword_lists": [["Alpha", "beta"], ["beta", "Gamma"], None],
        }
    )
    res = await graph.get_document(driver, "doc1")
    assert res == {
        "sources": ["a"],
        "chunk_count": 2,
        "keywords": ["alpha", "beta", "gamma"],
    }


async def test_get_document_missing_returns_none():
    assert await graph.get_document(_Driver(None), "x") is None
