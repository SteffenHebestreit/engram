"""Build the community/theme layer (GraphRAG-style global synthesis).

Clusters the ingested chunk graph with Leiden, writes an LLM report per
community, and persists the layer. Neo4j + GDS only.

Usage: python -m scripts.build_communities [--no-reports]
"""

import argparse
import asyncio

import httpx

from app.community import build_communities
from app.config import get_settings
from app.store import create_store


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-reports", action="store_true", help="skip LLM community reports"
    )
    args = parser.parse_args()

    store = create_store(get_settings())
    await store.connect()
    try:
        await store.verify_connectivity()
        async with httpx.AsyncClient() as http:
            result = await build_communities(
                store, http, generate_reports=not args.no_reports
            )
        print(f"built {result['communities']} communities")
        for comm in await store.list_communities():
            title = comm["title"] or "(no report)"
            print(f"  [{comm['member_count']:>3}] {title} — {', '.join(comm['keywords'][:5])}")
    except NotImplementedError as exc:
        raise SystemExit(f"unsupported: {exc}")
    finally:
        await store.close()


if __name__ == "__main__":
    asyncio.run(main())
