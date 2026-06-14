"""Config-driven retrieval channels.

A channel is one independently-embedded view of a chunk (its content, its
one-sentence summary, its keywords, ...). Each channel owns a vector index and
a fusion weight. The set was hard-coded to three; it is now declarative so a
deployment can drop a channel, re-weight, or add a new one (e.g. a "title" or
"questions" channel produced by a custom metadata extractor) without editing
the pipeline.

The default set is built from the legacy ``*_channel_weight`` settings, so
existing env overrides keep working unchanged and the fusion math (total
channel weight 1.0 + 0.9 + 0.8 = 2.7, +0.7 fulltext = 3.4) is byte-identical.
A deployment can override the whole set with a JSON ``VECTOR_CHANNELS`` env var.

The ``content_embedding`` property is the canonical geometry vector used for
median-proximity and MMR; the content channel must populate it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from pydantic import BaseModel

from .registry import Registry

if TYPE_CHECKING:
    from .config import Settings
    from .llm import ExtractionResult


class VectorChannel(BaseModel):
    """One embedded view of a chunk with its index and fusion weight."""

    name: str
    index: str  # Neo4j vector index name
    embedding_prop: str  # Chunk property the embedding is stored on
    source: str  # registered channel-source key (see CHANNEL_SOURCES)
    weight: float


# How each channel derives the text to embed at ingest from the chunk text and
# its extracted metadata. Registered so a custom extractor can expose new
# fields (e.g. "title", "questions") as embeddable channel sources.
ChannelSource = Callable[[list[str], "list[ExtractionResult]"], list[str]]
CHANNEL_SOURCES: Registry[ChannelSource] = Registry("channel_source")


@CHANNEL_SOURCES.register("text")
def _source_text(chunks: list[str], metadata: "list[ExtractionResult]") -> list[str]:
    return chunks


@CHANNEL_SOURCES.register("summary")
def _source_summary(chunks: list[str], metadata: "list[ExtractionResult]") -> list[str]:
    return [m.summary for m in metadata]


@CHANNEL_SOURCES.register("keywords")
def _source_keywords(
    chunks: list[str], metadata: "list[ExtractionResult]"
) -> list[str]:
    # fall back to the summary when a chunk produced no keywords
    return [
        ", ".join(m.keywords) if m.keywords else m.summary for m in metadata
    ]


def get_channel_source(name: str) -> ChannelSource:
    return CHANNEL_SOURCES.get(name)


def resolve_vector_channels(settings: "Settings") -> list[VectorChannel]:
    """The active channel set: an explicit ``vector_channels`` override, else
    the built-in three weighted from the legacy ``*_channel_weight`` settings."""
    if settings.vector_channels is not None:
        return settings.vector_channels
    return [
        VectorChannel(
            name="content",
            index="chunk_content_idx",
            embedding_prop="content_embedding",
            source="text",
            weight=settings.content_channel_weight,
        ),
        VectorChannel(
            name="summary",
            index="chunk_summary_idx",
            embedding_prop="summary_embedding",
            source="summary",
            weight=settings.summary_channel_weight,
        ),
        VectorChannel(
            name="keywords",
            index="chunk_keywords_idx",
            embedding_prop="keywords_embedding",
            source="keywords",
            weight=settings.keywords_channel_weight,
        ),
    ]
