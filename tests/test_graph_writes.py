import pytest

from app import graph


def test_sanitize_identifier_strips_and_rejects_empty():
    assert graph._sanitize_identifier("Ebm-Code!", "label") == "EbmCode"
    assert graph._sanitize_identifier("a.b c", "type") == "abc"
    with pytest.raises(ValueError, match="invalid label"):
        graph._sanitize_identifier("!!!", "label")


class _Result:
    def __init__(self, record):
        self._record = record

    async def single(self):
        return self._record


class _Session:
    def __init__(self, calls):
        self.calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def run(self, query, **params):
        self.calls.append((query, params))
        return _Result({"n": len(params.get("rows", []))})


class _Driver:
    def __init__(self):
        self.calls = []

    def session(self):
        return _Session(self.calls)


@pytest.fixture
def no_invalidate(monkeypatch):
    async def _noop(driver):
        return None

    monkeypatch.setattr(graph, "invalidate_ppr_projection", _noop)


async def test_upsert_entities_builds_cypher_and_params(no_invalidate):
    driver = _Driver()
    n = await graph.upsert_entities(
        driver,
        "Ebm Code",  # space sanitized away
        [{"key": "03000", "properties": {"pt": 186}}, {"key": "03220"}],
    )
    assert n == 2
    query, params = driver.calls[0]
    assert "MERGE (n:`EbmCode` {key: row.key})" in query
    assert params["rows"] == [
        {"key": "03000", "properties": {"pt": 186}},
        {"key": "03220", "properties": {}},  # missing properties default to {}
    ]


async def test_upsert_relations_sanitizes_label_and_type(no_invalidate):
    driver = _Driver()
    n = await graph.upsert_relations(
        driver,
        "EbmCode",
        "EXCLUDES-SAME!QUARTAL",  # punctuation stripped
        "EbmCode",
        [{"from_key": "03100", "to_key": "03220"}],
    )
    assert n == 1
    query, params = driver.calls[0]
    assert "[r:`EXCLUDESSAMEQUARTAL`]" in query
    assert "MATCH (a:`EbmCode` {key: row.from_key})" in query
    assert params["rows"] == [
        {"from_key": "03100", "to_key": "03220", "properties": {}}
    ]


async def test_upsert_empty_items_is_a_noop(no_invalidate):
    driver = _Driver()
    assert await graph.upsert_entities(driver, "X", []) == 0
    assert await graph.upsert_relations(driver, "A", "R", "B", []) == 0
    assert driver.calls == []


async def test_upsert_entities_rejects_unusable_label(no_invalidate):
    with pytest.raises(ValueError, match="invalid label"):
        await graph.upsert_entities(_Driver(), "***", [{"key": "x"}])
