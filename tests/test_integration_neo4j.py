"""End-to-end test against a live Neo4j (docker compose up -d).

Embedding / LLM / reranker services are mocked; the real Cypher in
app.graph is exercised: document + chunk persistence, NEXT_CHUNK chain,
keyword relations, vector index queries and sibling expansion.

Skipped automatically when Neo4j is not reachable.
"""

import hashlib

import numpy as np
import pytest

from app import graph, ingest as ingest_mod, rerank as rerank_mod, search as search_mod
from app.community import build_communities
from app.config import get_settings
from app.llm import ExtractionResult
from app.store_neo4j import Neo4jStore

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

    store = Neo4jStore(driver)
    doc_id, chunk_count, keywords = await ingest_mod.ingest_document(
        store, None, DOC_TEXT, title="it-doc", source="test"
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
        monkeypatch.setitem(rerank_mod.RERANKERS._items, "http", fake_rerank)

        results = await search_mod.search(store, None, "what is topic zero?")
        assert results
        assert results[0].chunk_id == chunk_ids[0]
        returned = {r.chunk_id for r in results}
        assert {chunk_ids[1], chunk_ids[2]} <= returned

        # per-channel provenance (feeds the eval harness) flows through real search:
        # the top hit was surfaced by at least one retrieval channel, and every
        # channel label is a known kind
        known = {"content", "summary", "keywords", "fulltext"}
        assert results[0].channels
        for r in results:
            for ch in r.channels:
                assert ch in known or ch.startswith("graph:")

        # end-to-end eval harness over a golden set: topic-0 query -> chunk 0's doc
        from app.eval import run_evaluation

        report = await run_evaluation(
            store, None, {"e1": {doc_id: 1}}, {"e1": "what is topic zero?"}, ks=(5,)
        )
        assert report["n_queries"] == 1
        assert report["metrics"]["Recall@5"]["mean"] == 1.0  # the doc is retrieved
        assert report["attribution"]["gold_hits_retrieved"] >= 1
        assert report["attribution"]["by_channel"]  # at least one channel credited

        # nearest_chunks (memory write-path primitive): a chunk's own content
        # vector finds it back at ~1.0 cosine; excluding its document returns none
        c0_emb = fake_embedding(chunk_texts[chunk_ids[0]])
        near = await store.nearest_chunks(c0_emb, 3, 0.9)
        assert any(n["id"] == chunk_ids[0] and n["sim"] > 0.99 for n in near)
        excluded = await store.nearest_chunks(c0_emb, 3, 0.9, exclude_doc_id=doc_id)
        assert all(n["doc_id"] != doc_id for n in excluded)  # this doc is excluded

        # incremental-reuse read path returns chunks with their stored vectors
        cached = await store.fetch_document_chunks(doc_id, ["content_embedding"])
        assert len(cached) == 3
        assert all(len(c["embeddings"]["content_embedding"]) == DIM for c in cached)

        # community synthesis via Leiden — only when GDS is actually installed
        if await graph.gds_available(driver):
            built = await build_communities(store, None, generate_reports=False)
            comms = await store.list_communities()
            assert built["communities"] == len(comms)
            assert comms
            # every chunk lands in exactly one community
            assert sum(c["member_count"] for c in comms) == 3
            await store.save_communities([])  # clean up the community layer
    finally:
        await _cleanup(driver, doc_id)
        await driver.close()


async def test_sparse_weights_round_trip():
    """BGE-M3 sparse term-weight maps survive save -> get_sparse_weights and the
    reuse read path, stored as a JSON string property (Neo4j has no map type)."""
    driver = await _connect_or_skip()
    await graph.init_schema(driver)
    doc_id = "sparse-it-doc"
    vec = [1.0] + [0.0] * (DIM - 1)
    embeddings = {
        "content_embedding": vec,
        "summary_embedding": vec,
        "keywords_embedding": vec,
    }
    chunks = [
        {
            "id": f"{doc_id}:0", "seq": 0, "text": "first", "summary": "s0",
            "keywords": ["k0"], "embeddings": embeddings,
            "sparse_weights": {"101": 0.5, "202": 1.25},
        },
        {
            "id": f"{doc_id}:1", "seq": 1, "text": "second", "summary": "s1",
            "keywords": ["k1"], "embeddings": embeddings,
            "sparse_weights": None,  # a chunk ingested without sparse weights
        },
    ]
    ids = [f"{doc_id}:0", f"{doc_id}:1"]
    try:
        await graph.save_document(driver, doc_id, "sparse-doc", ["test"], chunks)

        # only the chunk that carried weights is returned, parsed back to floats
        weights = await graph.get_sparse_weights(driver, ids)
        assert weights == {f"{doc_id}:0": {"101": 0.5, "202": 1.25}}

        # the incremental-reuse read path also surfaces the stored sparse map
        cached = await graph.fetch_document_chunks(driver, doc_id, ["content_embedding"])
        by_text = {c["text"]: c for c in cached}
        assert by_text["first"]["sparse_weights"] == {"101": 0.5, "202": 1.25}
        assert by_text["second"]["sparse_weights"] is None
    finally:
        await graph.delete_document(driver, doc_id)
        await driver.close()


async def test_near_dup_links_round_trip():
    """Memory write-path: near_dup_of survives save -> get_near_dup_links."""
    driver = await _connect_or_skip()
    await graph.init_schema(driver)
    doc_id = "neardup-it-doc"
    vec = [1.0] + [0.0] * (DIM - 1)
    embeddings = {
        "content_embedding": vec,
        "summary_embedding": vec,
        "keywords_embedding": vec,
    }
    chunks = [
        {
            "id": f"{doc_id}:0", "seq": 0, "text": "a", "summary": "", "keywords": [],
            "embeddings": embeddings, "near_dup_of": "canonical:7",
        },
        {
            "id": f"{doc_id}:1", "seq": 1, "text": "b", "summary": "", "keywords": [],
            "embeddings": embeddings, "near_dup_of": None,
        },
    ]
    ids = [f"{doc_id}:0", f"{doc_id}:1"]
    try:
        await graph.save_document(driver, doc_id, "t", ["s"], chunks)
        # only the linked chunk is returned
        assert await graph.get_near_dup_links(driver, ids) == {f"{doc_id}:0": "canonical:7"}
    finally:
        await graph.delete_document(driver, doc_id)
        await driver.close()
