from app.llm import _parse_json_object


def test_plain_json():
    assert _parse_json_object('{"keywords": ["a"], "summary": "s"}') == {
        "keywords": ["a"],
        "summary": "s",
    }


def test_json_in_code_fence():
    text = 'Here you go:\n```json\n{"keywords": ["a", "b"], "summary": "s"}\n```'
    assert _parse_json_object(text)["keywords"] == ["a", "b"]


def test_json_with_surrounding_chatter():
    text = 'Sure! {"keywords": ["x"], "summary": "y"} Hope that helps.'
    assert _parse_json_object(text) == {"keywords": ["x"], "summary": "y"}


def test_garbage_returns_empty_dict():
    assert _parse_json_object("not json at all") == {}
