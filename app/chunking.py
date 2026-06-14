from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from .registry import Registry

if TYPE_CHECKING:
    from .config import Settings

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def chunk_text(text: str, target_chars: int, overlap_chars: int) -> list[str]:
    """Split text into chunks near target_chars, preferring paragraph then
    sentence boundaries, with a sentence-aligned overlap between chunks.

    Chunks may exceed target_chars by up to overlap_chars, since the overlap
    tail of the previous chunk is prepended before the next unit is fitted."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= target_chars:
            current = f"{current}\n\n{para}" if current else para
            continue

        # paragraph does not fit; close the current chunk (keeping overlap)
        if current:
            tail = _overlap_tail(current, overlap_chars)
            flush()
            current = tail

        if len(para) <= target_chars:
            current = f"{current}\n\n{para}" if current else para
            continue

        # oversized paragraph: pack sentence by sentence
        for sentence in _SENTENCE_END.split(para):
            if len(current) + len(sentence) + 1 > target_chars and current:
                tail = _overlap_tail(current, overlap_chars)
                flush()
                current = tail
            current = f"{current} {sentence}".strip()

    flush()
    return chunks


class Chunker(Protocol):
    """A chunking strategy: turn a document's text into ordered chunks.

    Receives the whole `Settings` so a strategy can read its own knobs (e.g. a
    semantic chunker reading an embedding threshold) without changing this
    signature.
    """

    def __call__(self, text: str, settings: "Settings") -> list[str]: ...


CHUNKERS: Registry[Chunker] = Registry("chunker")


@CHUNKERS.register("fixed")
def _fixed_chunker(text: str, settings: "Settings") -> list[str]:
    """Default: paragraph/sentence-aligned fixed-size windows with overlap."""
    return chunk_text(text, settings.chunk_target_chars, settings.chunk_overlap_chars)


def get_chunker(name: str) -> Chunker:
    return CHUNKERS.get(name)


def _overlap_tail(text: str, overlap_chars: int) -> str:
    """Last sentences of `text` totalling at most overlap_chars."""
    if overlap_chars <= 0:
        return ""
    sentences = _SENTENCE_END.split(text)
    tail: list[str] = []
    length = 0
    for sentence in reversed(sentences):
        if length + len(sentence) > overlap_chars:
            break
        tail.insert(0, sentence)
        length += len(sentence) + 1
    return " ".join(tail).strip()
