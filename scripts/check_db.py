"""Verify Neo4j connectivity and that schema init creates the vector indexes.

Usage: python -m scripts.check_db
"""

import asyncio

from app import graph


async def main() -> None:
    driver = graph.create_driver()
    try:
        await driver.verify_connectivity()
        print("connectivity: OK")

        await graph.init_schema(driver)
        print("schema init: OK")

        async with driver.session() as session:
            result = await session.run(
                "SHOW INDEXES YIELD name, type, state "
                "WHERE type IN ['VECTOR', 'FULLTEXT'] "
                "RETURN name, type, state ORDER BY name"
            )
            rows = [dict(r) async for r in result]
        for row in rows:
            print(f"{row['type'].lower()} index: {row['name']} [{row['state']}]")

        expected = set(graph.VECTOR_INDEXES) | {graph.FULLTEXT_INDEX}
        missing = expected - {r["name"] for r in rows}
        if missing:
            raise SystemExit(f"MISSING indexes: {missing}")
        print("all search indexes present")
    finally:
        await driver.close()


if __name__ == "__main__":
    asyncio.run(main())
