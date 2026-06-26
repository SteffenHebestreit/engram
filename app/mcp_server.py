"""MCP server — expose engram's retrieval to agents (Model Context Protocol).

engram is a retrieval *tool*: agents (StarlingAI, Claude, ...) call it to pull
the most relevant document context, then write the answer themselves. This is a
thin Model Context Protocol adapter over engram's existing HTTP API, so any
MCP-speaking agent can plug engram in as a drop-in knowledge source. engram's
HTTP API can also be called directly — this is just the agent-tool front door.

Run (stdio):   python -m app.mcp_server
Deps:          pip install -r requirements-mcp.txt    (the `mcp` package)
Config:        ENGRAM_API_BASE (default http://localhost:8088)

The HTTP-proxy helpers below carry the logic and are unit-tested directly; the
`mcp` dependency is only imported when the server is actually run.
"""

from __future__ import annotations

import os

import httpx

ENGRAM_API_BASE = os.environ.get("ENGRAM_API_BASE", "http://localhost:8088").rstrip("/")
_TIMEOUT = float(os.environ.get("ENGRAM_MCP_TIMEOUT", "120"))


async def proxy_search(
    query: str, top_k: int = 8, preset: str | None = None
) -> list[dict]:
    body: dict = {"query": query}
    if top_k:
        body["top_k"] = top_k
    if preset:
        body["tuning"] = {"preset": preset}
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(f"{ENGRAM_API_BASE}/search", json=body)
        resp.raise_for_status()
        return resp.json()["results"]


async def proxy_chunk_context(
    chunk_id: str, before: int = 2, after: int = 2
) -> list[dict]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(
            f"{ENGRAM_API_BASE}/chunks/{chunk_id}/context",
            params={"before": before, "after": after},
        )
        resp.raise_for_status()
        return resp.json()["chunks"]


async def proxy_list_documents() -> list[dict]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(f"{ENGRAM_API_BASE}/documents")
        resp.raise_for_status()
        return resp.json()


async def proxy_search_themes(query: str, top_k: int = 5) -> list[dict]:
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            f"{ENGRAM_API_BASE}/communities/search",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        return resp.json()


def build_server():
    """Construct the FastMCP server (imports `mcp`; install requirements-mcp.txt)."""
    from mcp.server.fastmcp import FastMCP

    server = FastMCP("engram")

    @server.tool()
    async def search(query: str, top_k: int = 8, preset: str | None = None) -> list[dict]:
        """Search the document knowledge base for passages relevant to a query.

        Returns ranked chunks, each with its text, source document id, and score
        provenance (retrieval / median / graph-proximity / rerank scores that
        explain why it was retrieved). Use this to gather grounding context to
        answer a question from the documents. `preset` optionally trades cost vs
        quality: "cheap" | "balanced" | "max_quality".
        """
        return await proxy_search(query, top_k, preset)

    @server.tool()
    async def get_chunk_context(chunk_id: str, before: int = 2, after: int = 2) -> list[dict]:
        """Fetch the passages immediately before and after a chunk (by its id),
        for more surrounding context around a search result."""
        return await proxy_chunk_context(chunk_id, before, after)

    @server.tool()
    async def list_documents() -> list[dict]:
        """List the documents currently in the knowledge base (id, title, sources,
        chunk count)."""
        return await proxy_list_documents()

    @server.tool()
    async def search_themes(query: str, top_k: int = 5) -> list[dict]:
        """For broad, corpus-wide "what are the main themes about X?" questions:
        rank the document collection's theme/community reports against the query
        and return the top theme summaries (requires the community layer to have
        been built)."""
        return await proxy_search_themes(query, top_k)

    return server


def main() -> None:
    build_server().run()  # stdio transport


if __name__ == "__main__":
    main()
