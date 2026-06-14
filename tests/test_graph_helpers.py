from app.graph import _sanitize_fulltext_query, fulltext_search


def test_sanitize_strips_lucene_syntax():
    assert _sanitize_fulltext_query('foo AND (bar:baz)^2 ~"qux"') == "foo AND  bar baz  2   qux"
    assert _sanitize_fulltext_query("plain words stay") == "plain words stay"
    assert _sanitize_fulltext_query("umlauts häßlich okay") == "umlauts häßlich okay"


def test_sanitize_empty_results():
    assert _sanitize_fulltext_query("???") == ""
    assert _sanitize_fulltext_query("  ") == ""


async def test_fulltext_search_skips_db_for_empty_query():
    # driver=None proves no session is opened when nothing remains to search
    assert await fulltext_search(None, "?!*", 5) == []
