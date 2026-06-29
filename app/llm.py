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

# The extraction output as a json_schema response_format. On servers that support
# it (vLLM guided decoding, LM Studio, llama.cpp GBNF) this GUARANTEES schema-valid
# JSON via constrained decoding, and — on servers that gate chain-of-thought behind
# free-form generation — it suppresses the thinking preamble. Required by LM Studio,
# which rejects {"type":"json_object"}. Selected via extraction_response_format.
EXTRACTION_JSON_SCHEMA: dict = {
    "type": "json_schema",
    "json_schema": {
        "name": "extraction",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "array", "items": {"type": "string"}},
                "summary": {"type": "string"},
            },
            "required": ["keywords", "summary"],
            "additionalProperties": False,
        },
    },
}


def _extraction_response_format(settings) -> dict | None:
    """The extraction call's response_format, general across serving stacks: an
    explicit `extraction_response_format` wins, else inherit `llm_json_mode`."""
    mode = (settings.extraction_response_format or "").lower()
    if not mode:
        return {"type": "json_object"} if settings.llm_json_mode else None
    if mode in ("json_schema", "schema"):
        return EXTRACTION_JSON_SCHEMA
    if mode in ("json_object", "json"):
        return {"type": "json_object"}
    return None  # "text" / "none" — rely on the tolerant parser


class ExtractionResult(dict):
    @property
    def keywords(self) -> list[str]:
        return self["keywords"]

    @property
    def summary(self) -> str:
        return self["summary"]

    @property
    def context(self) -> str:
        # Contextual Retrieval: a short document-situating context generated at
        # ingest and prepended to the chunk before embedding. Empty unless
        # contextual retrieval is enabled. Optional, so older results omit it.
        return self.get("context", "")


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


@EXTRACTORS.register("none")
async def extract_nothing(client: httpx.AsyncClient, chunk: str) -> ExtractionResult:
    """No-op extractor: no LLM call, empty summary/keywords.

    The cheap end of the cost/quality trade-off — one embedding per chunk and
    zero LLM round trips. Pair with content-only channels
    (`summary_channel_enabled=false`, `keywords_channel_enabled=false`). Note
    this also drops the summary's fulltext contribution and keyword-sibling
    graph expansion, since both derive from this metadata.
    """
    return ExtractionResult(keywords=[], summary="")


_yake_extractor = None


@EXTRACTORS.register("yake")
async def extract_yake(client: httpx.AsyncClient, chunk: str) -> ExtractionResult:
    """Statistical keyword extraction (YAKE) — no LLM, no network. The default.

    Sits between `default` (LLM) and `none`: it populates the keyword channel
    and the shared-keyword graph (so cross-document HAS_KEYWORD linking and
    keyword-sibling expansion work) and gives the summary channel the chunk's
    first sentence — all without a generative model, so ingest needs no chat
    endpoint and isn't bottlenecked on per-chunk generation. Switch to `default`
    for an LLM-written abstractive summary (an opt-in quality upgrade).
    """
    global _yake_extractor
    if _yake_extractor is None:
        try:
            import yake
        except ImportError as e:  # yake is the default extractor, so required
            raise RuntimeError(
                "The default METADATA_EXTRACTOR='yake' needs the 'yake' package "
                "(pip install yake, or it ships in requirements.txt). Or set "
                "METADATA_EXTRACTOR=default (LLM) / 'none' to avoid it."
            ) from e

        _yake_extractor = yake.KeywordExtractor(lan="en", n=2, top=8)
    keywords = [kw.lower() for kw, _ in _yake_extractor.extract_keywords(chunk)]
    cleaned = chunk.strip()
    summary = re.split(r"(?<=[.!?])\s+", cleaned)[0][:300] if cleaned else ""
    return ExtractionResult(keywords=keywords, summary=summary)


@EXTRACTORS.register("default")
async def extract_metadata(client: httpx.AsyncClient, chunk: str) -> ExtractionResult:
    """Ask the LLM for keywords/labels and a one-sentence summary of a chunk.

    Targets a separate small/fast extraction model when `extraction_llm_*` is
    configured, else the shared `llm_*` endpoint. Extraction is the high-volume,
    low-difficulty LLM call, so it pays to run a small model here (with
    `extraction_max_tokens` capping the tiny output) while HyDE/contextual/
    community keep the stronger model. Chunks below `extraction_min_chars` skip
    the round trip and fall back to the statistical `yake` extractor.
    """
    settings = get_settings()

    # length-gate: short chunks (titles, headers, list items) aren't worth an LLM
    # round trip — statistical keywords + first sentence are good enough.
    if settings.extraction_min_chars and len(chunk.strip()) < settings.extraction_min_chars:
        return await extract_yake(client, chunk)

    # prefer the dedicated extraction endpoint/model, falling back per-field to
    # the shared llm_* settings (all blank => identical to the shared endpoint).
    api_base = settings.extraction_llm_api_base or settings.llm_api_base
    api_key = settings.extraction_llm_api_key or settings.llm_api_key
    model = settings.extraction_llm_model or settings.llm_model

    body: dict = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": chunk},
        ],
    }
    if settings.extraction_max_tokens > 0:
        body["max_tokens"] = settings.extraction_max_tokens
    rf = _extraction_response_format(settings)
    if rf:
        body["response_format"] = rf
    # serving-specific reasoning controls (e.g. {"chat_template_kwargs":
    # {"enable_thinking": false}} on vLLM, {"reasoning_effort": "none"} on
    # Ollama/LM Studio) merged at the top level; empty by default.
    if settings.extraction_extra_body:
        body.update(settings.extraction_extra_body)

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    resp = await client.post(
        f"{api_base.rstrip('/')}/chat/completions",
        json=body,
        headers=headers,
        timeout=settings.request_timeout,
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
            timeout=settings.hyde_timeout,
        )
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"].strip()
        return text or None
    except Exception:
        return None


_CONTEXT_SYSTEM_PROMPT = """\
You situate a text chunk within its source document for a retrieval system
(Contextual Retrieval). Given the whole document and one chunk taken from it,
write a SHORT context (1-2 sentences) that says what the chunk is about and how
it fits the document — name the entities, section, time period or topic a search
would need to find it. This text is prepended to the chunk before embedding, to
disambiguate it from similar passages in other documents.
Reply with the context only: no preamble, no markdown, do not repeat the chunk."""


async def generate_chunk_context(
    client: httpx.AsyncClient, document: str, chunk: str
) -> str:
    """A short document-situating context for a chunk (Anthropic's Contextual
    Retrieval). Prepended to the chunk before embedding so its vector encodes the
    document-level identity a bare chunk lacks.

    Returns "" on any failure (or an empty reply), so ingest degrades to the
    plain chunk embedding instead of breaking when the LLM is unavailable — same
    contract as HyDE / metadata extraction.

    The document is sent first (a long prefix shared by every chunk of the same
    document) so providers that cache prompt prefixes amortize it across the
    document's chunks; the per-document cost is otherwise one LLM call per chunk.
    """
    settings = get_settings()
    doc = document[: settings.contextual_max_doc_chars]
    user = (
        f"<document>\n{doc}\n</document>\n\n"
        f"Here is the chunk to situate within the document:\n"
        f"<chunk>\n{chunk}\n</chunk>"
    )

    headers = {}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"

    try:
        resp = await client.post(
            f"{settings.llm_api_base.rstrip('/')}/chat/completions",
            json={
                "model": settings.llm_model,
                "temperature": 0.2,
                "max_tokens": settings.contextual_max_tokens,
                "messages": [
                    {"role": "system", "content": _CONTEXT_SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
            },
            headers=headers,
            timeout=settings.request_timeout,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


_COMMUNITY_REPORT_SYSTEM_PROMPT = """\
You summarize a thematic cluster of document passages for a knowledge base.
Given the passages' one-sentence summaries and shared keywords, reply with JSON
only, exactly this shape:
{"title": "...", "summary": "..."}

Rules:
- title: a short (3-7 word) descriptive name for the theme.
- summary: 2-4 sentences describing what this cluster of content is about.
- No markdown, no explanations, JSON only."""


async def generate_community_report(
    client: httpx.AsyncClient, summaries: list[str], keywords: list[str]
) -> dict | None:
    """A title + summary for a detected community (GraphRAG-style report).

    Returns {"title", "summary"} or None on any failure, so community building
    degrades to bare structure (keywords only) when the LLM is unavailable.
    """
    settings = get_settings()
    member_summaries = "\n".join(f"- {s}" for s in summaries if s)[:4000]
    user = f"Keywords: {', '.join(keywords)}\n\nPassage summaries:\n{member_summaries}"

    headers = {}
    if settings.llm_api_key:
        headers["Authorization"] = f"Bearer {settings.llm_api_key}"
    body: dict = {
        "model": settings.llm_model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": _COMMUNITY_REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    }
    if settings.llm_json_mode:
        body["response_format"] = {"type": "json_object"}

    try:
        resp = await client.post(
            f"{settings.llm_api_base.rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=settings.request_timeout,
        )
        resp.raise_for_status()
        parsed = _parse_json_object(resp.json()["choices"][0]["message"]["content"])
    except Exception:
        return None
    title = str(parsed.get("title", "")).strip()
    summary = str(parsed.get("summary", "")).strip()
    if not title and not summary:
        return None
    return {"title": title, "summary": summary}


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
