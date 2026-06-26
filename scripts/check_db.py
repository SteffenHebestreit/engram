"""Verify store connectivity and that schema init succeeds.

For the Neo4j backend it also lists the vector/fulltext indexes and checks the
expected ones are present.

Usage: python -m scripts.check_db
"""

import asyncio

from app import graph
from app.channels import resolve_vector_channels
from app.config import get_settings
from app.store import create_store


async def _check_neo4j_indexes(settings) -> None:
    """Neo4j-specific: list and verify the search indexes exist."""
    driver = graph.create_driver()
    try:
        async with driver.session() as session:
            result = await session.run(
                "SHOW INDEXES YIELD name, type, state "
                "WHERE type IN ['VECTOR', 'FULLTEXT'] "
                "RETURN name, type, state ORDER BY name"
            )
            rows = [dict(r) async for r in result]
        for row in rows:
            print(f"{row['type'].lower()} index: {row['name']} [{row['state']}]")

        expected = {c.index for c in resolve_vector_channels(settings)} | {
            graph.FULLTEXT_INDEX
        }
        missing = expected - {r["name"] for r in rows}
        if missing:
            raise SystemExit(f"MISSING indexes: {missing}")
        print("all search indexes present")
    finally:
        await driver.close()


async def main() -> None:
    settings = get_settings()
    store = create_store(settings)
    await store.connect()
    try:
        await store.verify_connectivity()
        print(f"connectivity: OK (backend: {settings.store_backend})")

        await store.init_schema()
        print("schema init: OK")
    finally:
        await store.close()

    if settings.store_backend == "neo4j":
        await _check_neo4j_indexes(settings)


if __name__ == "__main__":
    asyncio.run(main())
