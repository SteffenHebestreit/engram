from __future__ import annotations

import json
import re
from typing import Protocol

import httpx

from .config import get_settings
from .registry import Registry

_EXTRACTION_SYSTEM_PROMPT = """\
You extract metadata from text chunks for a retrieval system.
Reply with JSON only, exactly this shape:
{"keywords": ["...", "..."], "summary": "..."}

Rules:
- keywords: 3 to 8 concise keywords or labels capturing the chunk's topics, \
entities and concepts. Lowercase unless a proper noun.
- summary: exactly ONE sentence summarizing the chunk.
- No markdown, no explanations, JSON only."""


class ExtractionResult(dict):
    @property
    def keywords(self) -> list[str]:
        return self["keywords"]

    @property
    def summary(self) -> str:
        return self["summary"]


class MetadataExtractor(Protocol):
    """A per-chunk metadata strategy.

    Must return an `ExtractionResult` exposing at least `summary` and
    `keywords`; those feed the summary/keywords channels at ingest. A
    domain-specific extractor (entities, code symbols, Q/A pairs) registers
    under its own key and is selected via `Settings.metadata_extractor`.
    """

    async def __call__(
        self, client: httpx.AsyncClient, chunk: str
    ) -> ExtractionResult: ...


EXTRACTORS: Registry[MetadataExtractor] = Registry("metadata_extractor")


def get_extractor(name: str) -> MetadataExtractor:
    return EXTRACTORS.get(name)


@EXTRACTORS.register("default")
async def extract_metadata(client: httpx.AsyncClient, chunk: str) -> ExtractionResult:
    """Ask the LLM for keywords/labels and a one-sentence summary of a chunk."""
    settings = get_settings()

    body: dict = {
        "model": settings.llm_model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
    }
    if settings.llm_json_mode:
        body["response_format"] = {"type": "json_object"}

    headers = {}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    resp = await client.post(
        f"{settings.llm_api_base.rstrip('/')}/chat/completions",
        json=body,
        headers=headers,
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    parsed = _parse_json_object(content)

    keywords = [str(k).strip() for k in parsed.get("keywords", []) if str(k).strip()]
    summary = str(parsed.get("summary", "")).strip()
    if not summary:
        # degrade gracefully: first sentence of the chunk
        summary = re.split(r"(?<=[.!?])\s+", chunk.strip())[0][:300]
    return ExtractionResult(keywords=keywords[:8], summary=summary)


_HYDE_SYSTEM_PROMPT = """\
You write hypothetical answers for a retrieval system (HyDE).
Given a search query, write a short factual passage (2-4 sentences) that a
document answering the query would plausibly contain. Invented specifics are
fine; the text is only embedded to find real documents, never shown to users.
Reply with the passage only - no preamble, no markdown."""


async def generate_hypothetical_answer(
    client: httpx.AsyncClient, query: str
) -> str | None:
    """HyDE: a hypothetical answer passage to embed instead of the bare query.

    Returns None on any failure so retrieval degrades to the plain query
    embedding instead of breaking when the LLM endpoint is unavailable.
    """
    settings = get_settings()

    headers = {}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    try:
        resp = await client.post(
            f"{settings.llm_api_base.rstrip('/')}/chat/completions",
            json={
                "model": settings.llm_model,
                "temperature": 0.3,
                "max_tokens": 220,
                "messages": [
                    {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                    {"role": "user", "content": query},
                ],
            },
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        return None


def _parse_json_object(text: str) -> dict:
    """Parse a JSON object out of an LLM reply, tolerating code fences and chatter."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {}
