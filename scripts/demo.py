"""Live end-to-end demo against the real network services.

Requires a filled-in .env (copy .env.example) and a running Neo4j.
Ingests a sample document, runs two searches, prints the scored results,
then deletes the demo document again.

Usage: python -m scripts.demo
"""

import asyncio
import sys
from pathlib import Path

import httpx

from app import graph
from app.ingest import ingest_document
from app.search import search

SAMPLE_TEXT = """\
Graph databases store data as nodes and relationships instead of tables. \
This makes them a natural fit for highly connected data such as social \
networks, knowledge graphs and recommendation systems. Traversing \
relationships is a first-class operation and does not require joins.

Vector search retrieves documents by semantic similarity. Texts are mapped \
into a high-dimensional embedding space, and nearest-neighbour search finds \
content with similar meaning even when no keywords overlap. Modern systems \
combine vector search with traditional filters for precision.

Retrieval-augmented generation, short RAG, grounds a language model in \
external knowledge. Relevant passages are retrieved first and passed to the \
model as context, which reduces hallucinations and keeps answers current \
without retraining the model.

Rerankers are cross-encoders that score a query and a document together. \
They are slower than embedding similarity but considerably more accurate, \
which is why pipelines typically rerank only a small shortlist of the best \
retrieval candidates as a final quality gate.
"""

QUERIES = [
    "How can I reduce hallucinations of a language model?",
    "Why use a graph database for connected data?",
]


async def main() -> None:
    if not Path(".env").exists():
        sys.exit(
            "No .env found. Copy .env.example to .env and fill in the URLs of "
            "your embedding/LLM/reranker services first."
        )

    driver = graph.create_driver()
    doc_id = None
    try:
        await driver.verify_connectivity()
        await graph.init_schema(driver)

        async with httpx.AsyncClient() as http:
            print("ingesting sample document...")
            doc_id, chunk_count, keywords = await ingest_document(
                driver, http, SAMPLE_TEXT, title="RAG demo", source="demo"
            )
            print(f"  document {doc_id}: {chunk_count} chunks")
            print(f"  extracted keywords: {', '.join(keywords)}\n")

            for query in QUERIES:
                print(f"query: {query}")
                results = await search(driver, http, query, top_k=3)
                for rank, r in enumerate(results, 1):
                    print(
                        f"  {rank}. [{r.origin}] rerank={r.rerank_score:.3f} "
                        f"fused={r.fused_score:.3f} median={r.median_score:.3f}"
                    )
                    print(f"     {r.summary}")
                print()
    finally:
        if doc_id:
            await graph.delete_document(driver, doc_id)
            print(f"cleaned up demo document {doc_id}")
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
