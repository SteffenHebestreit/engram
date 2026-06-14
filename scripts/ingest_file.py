"""Ingest a text/markdown file from disk through the full pipeline.

Requires a filled-in .env and a running Neo4j.

Usage: python -m scripts.ingest_file <path> [--title TITLE] [--source SOURCE]
"""

import argparse
import asyncio
from pathlib import Path

import httpx

from app import graph
from app.ingest import ingest_document


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--title", default="")
    parser.add_argument("--source", default="")
    args = parser.parse_args()

    text = args.path.read_text(encoding="utf-8")
    title = args.title or args.path.stem
    source = args.source or str(args.path)

    driver = graph.create_driver()
    try:
        await driver.verify_connectivity()
        await graph.init_schema(driver)
        async with httpx.AsyncClient() as http:
            doc_id, chunk_count, keywords = await ingest_document(
                driver, http, text, title=title, source=source
            )
        print(f"document_id: {doc_id}")
        print(f"chunks: {chunk_count}")
        print(f"keywords: {', '.join(keywords)}")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
