"""Declarative graph schema profiles.

The graph the PPR projection spans was hard-coded to the document model
(`Chunk`/`Keyword` nodes over `NEXT_CHUNK`/`HAS_KEYWORD` edges). A `GraphProfile`
makes that declarative, so a deployment can extend the projected graph with
domain entity nodes and typed relations (loaded via the structured-entity
ingest endpoints) and let personalized PageRank spread activation through them
— without editing the projection Cypher.

The default profile reproduces the document model exactly, so the projection,
and therefore the proximity math, is unchanged unless a profile is supplied.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, field_validator

if TYPE_CHECKING:
    from .config import Settings

# GDS relationship orientations
_ORIENTATIONS = {"NATURAL", "REVERSE", "UNDIRECTED"}


class RelationSpec(BaseModel):
    """One relationship type included in the projected graph."""

    type: str
    orientation: str = "UNDIRECTED"
    # weight a traversal/proximity strategy may use; sign < 0 marks an
    # exclusion/constraint edge for strategies that understand it
    weight: float = 1.0
    sign: int = 1

    @field_validator("orientation")
    @classmethod
    def _check_orientation(cls, v: str) -> str:
        v = v.upper()
        if v not in _ORIENTATIONS:
            raise ValueError(f"orientation must be one of {sorted(_ORIENTATIONS)}")
        return v


class GraphProfile(BaseModel):
    """Node labels and relationships that make up the projected graph."""

    name: str = "document"
    projection_labels: list[str]
    projection_relationships: list[RelationSpec]

    def projection_spec(self) -> tuple[list[str], dict[str, dict[str, str]]]:
        """`(node_labels, relationship_config)` for `gds.graph.project`."""
        rel_config = {
            spec.type: {"orientation": spec.orientation}
            for spec in self.projection_relationships
        }
        return self.projection_labels, rel_config


# the built-in document profile: identical to the original hard-coded projection
DEFAULT_PROFILE = GraphProfile(
    name="document",
    projection_labels=["Chunk", "Keyword"],
    projection_relationships=[
        RelationSpec(type="NEXT_CHUNK", orientation="UNDIRECTED"),
        RelationSpec(type="HAS_KEYWORD", orientation="UNDIRECTED"),
    ],
)


def resolve_profile(settings: "Settings") -> GraphProfile:
    return settings.graph_profile or DEFAULT_PROFILE
