"""End-to-end test against a live Neo4j (docker compose up -d).

Embedding / LLM / reranker services are mocked; the real Cypher in
app.graph is exercised: document + chunk persistence, NEXT_CHUNK chain,
keyword relations, vector index queries and sibling expansion.

Skipped automatically when Neo4j is not reachable.
"""

import hashlib

import numpy as np
import pytest

from app import graph, ingest as ingest_mod, search as search_mod
from app.config import get_settings
from app.llm import ExtractionResult

DIM = get_settings().embedding_dim

# three paragraphs large enough that the chunker emits one chunk each
_PARA = " ".join(f"Topic {{n}} sentence {i} with several filler words." for i in range(35))
DOC_TEXT = "\n\n".join(_PARA.format(n=n) for n in range(3))

CHUNK_KEYWORDS = [["alpha", "graph"], ["beta"], ["alpha", "vector"]]


def fake_embedding(text: str) -> list[float]:
    seed = int.from_bytes(hashlib.md5(text.encode()).digest()[:4], "big")
    vec = np.random.RandomState(seed).normal(size=DIM)
    return (vec / np.linalg.norm(vec)).tolist()


async def _connect_or_skip():
    driver = graph.create_driver()
    try:
        await driver.verify_connectivity()
    except Exception:
        await driver.close()
        pytest.skip("Neo4j not reachable; run: docker compose up -d")
    return driver


async def _cleanup(driver, doc_id: str) -> None:
    deleted = await graph.delete_document(driver, doc_id)
    assert deleted in (None, 3)
    # deleting again reports "not found"
    assert await graph.delete_document(driver, doc_id) is None
    async with driver.session() as session:
        result = await session.run(
            "MATCH (c:Chunk {doc_id: $id}) RETURN count(c) AS n", id=doc_id
        )
        record = await result.single()
    assert record["n"] == 0


async def test_ingest_and_search_end_to_end(monkeypatch):
    driver = await _connect_or_skip()
    await graph.init_schema(driver)

    call_count = {"n": 0}

    async def fake_extract_metadata(client, chunk):
        idx = min(call_count["n"], 2)
        call_count["n"] += 1
        return ExtractionResult(
            keywords=CHUNK_KEYWORDS[idx], summary=f"Summary of part {idx}."
        )

    async def fake_embed_texts(client, texts):
        return [fake_embedding(t) for t in texts]

    # ingest resolves the extractor via the registry seam; patch the resolver
    monkeypatch.setattr(ingest_mod, "get_extractor", lambda name: fake_extract_metadata)
    monkeypatch.setattr(ingest_mod, "embed_texts", fake_embed_texts)

    doc_id, chunk_count, keywords = await ingest_mod.ingest_document(
        driver, None, DOC_TEXT, title="it-doc", source="test"
    )
    try:
        assert chunk_count == 3
        assert set(keywords) == {"alpha", "graph", "beta", "vector"}

        # --- graph structure ---
        async with driver.session() as session:
            result = await session.run(
                """
                MATCH (c:Chunk {doc_id: $id}) WITH c ORDER BY c.seq
                OPTIONAL MATCH (c)-[:NEXT_CHUNK]->(n:Chunk)
                RETURN c.id AS id, c.seq AS seq, c.text AS text, n.id AS next_id
                """,
                id=doc_id,
            )
            rows = [dict(r) async for r in result]
        assert [r["seq"] for r in rows] == [0, 1, 2]
        assert rows[0]["next_id"] == rows[1]["id"]
        assert rows[1]["next_id"] == rows[2]["id"]
        assert rows[2]["next_id"] is None
        chunk_ids = [r["id"] for r in rows]
        chunk_texts = {r["id"]: r["text"] for r in rows}

        # --- vector index lookup finds the exact chunk ---
        hits = await graph.vector_search(
            driver, "chunk_content_idx", fake_embedding(chunk_texts[chunk_ids[0]]), 3
        )
        assert hits[0]["id"] == chunk_ids[0]
        assert hits[0]["score"] > 0.99

        # --- fulltext channel finds the lexically matching chunk ---
        ft_hits = await graph.fulltext_search(driver, "Topic 0", 5)
        assert ft_hits
        assert ft_hits[0]["id"] == chunk_ids[0]

        # --- sibling expansion: directional sequence walk + shared keywords ---
        sibs = await graph.fetch_siblings(driver, [chunk_ids[0]], 5, 3)
        relations = {(s["id"], s["via"]): s for s in sibs}
        # chunk 1 is one NEXT_CHUNK hop after the seed, chunk 2 two hops
        seq1 = relations[(chunk_ids[1], "sequence")]
        assert (seq1["direction"], seq1["distance"]) == ("after", 1)
        seq2 = relations[(chunk_ids[2], "sequence")]
        assert (seq2["direction"], seq2["distance"]) == ("after", 2)
        # chunk 2 also shares the keyword "alpha" with the seed
        kw = relations[(chunk_ids[2], "keyword")]
        assert (kw["direction"], kw["distance"], kw["strength"]) == ("lateral", 1, 1.0)

        # --- full search pipeline on top of the live graph ---
        async def fake_embed_text(client, text):
            return fake_embedding(chunk_texts[chunk_ids[0]])

        async def fake_rerank(client, query, texts):
            return [1.0 / (i + 1) for i in range(len(texts))]  # keep fused order

        monkeypatch.setattr(search_mod, "embed_text", fake_embed_text)
        monkeypatch.setattr(search_mod, "rerank", fake_rerank)

        results = await search_mod.search(driver, None, "what is topic zero?")
        assert results
        assert results[0].chunk_id == chunk_ids[0]
        returned = {r.chunk_id for r in results}
        assert {chunk_ids[1], chunk_ids[2]} <= returned
    finally:
        await _cleanup(driver, doc_id)
        await driver.close()
