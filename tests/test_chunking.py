from app.chunking import chunk_text


def test_short_text_single_chunk():
    assert chunk_text("Hello world.", 1800, 200) == ["Hello world."]


def test_empty_text():
    assert chunk_text("", 1800, 200) == []
    assert chunk_text("\n\n  \n", 1800, 200) == []


def test_chunks_respect_target_size():
    text = " ".join(f"This is sentence number {i} about some topic." for i in range(400))
    chunks = chunk_text(text, 1800, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 1800 for c in chunks)


def test_paragraphs_kept_together_when_they_fit():
    text = "First paragraph.\n\nSecond paragraph."
    chunks = chunk_text(text, 1800, 200)
    assert len(chunks) == 1
    assert "First paragraph." in chunks[0]
    assert "Second paragraph." in chunks[0]


def test_overlap_between_consecutive_chunks():
    sentences = [f"Sentence {i} is here." for i in range(300)]
    chunks = chunk_text(" ".join(sentences), 1000, 150)
    assert len(chunks) > 2
    # the start of each following chunk repeats the tail of the previous one
    for prev, nxt in zip(chunks, chunks[1:]):
        first_sentence = nxt.split(".")[0] + "."
        assert first_sentence in prev


def test_paragraph_chunks_respect_size_bound():
    # large paragraphs trigger the overlap-prepend path; chunks may exceed
    # the target by at most overlap_chars
    para = " ".join(f"Paragraph sentence {i} with some words." for i in range(40))
    text = "\n\n".join([para] * 4)
    chunks = chunk_text(text, 1800, 200)
    assert len(chunks) > 1
    assert all(len(c) <= 1800 + 200 + 2 for c in chunks)


def test_no_content_lost():
    sentences = [f"Unique sentence {i}." for i in range(100)]
    chunks = chunk_text(" ".join(sentences), 800, 100)
    joined = " ".join(chunks)
    for sentence in sentences:
        assert sentence in joined
