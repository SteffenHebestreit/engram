"""Does NEXT_CHUNK recover answer context the chunker split away? (the chunking thesis)

The doc-level benchmarks can't see NEXT_CHUNK — its siblings are same-document, so
they don't change a doc's rank. This measures it at the **chunk** level with a
clean, fuzzy-qrels-free metric: SQuAD answers are exact spans, so "answer covered"
= the answer string appears in the returned chunk texts (substring match).

Setup that makes NEXT_CHUNK matter: each Wikipedia article becomes one document
(its paragraphs concatenated), chunked **small** (`CHUNK_TARGET_CHARS`) so a
question often matches one chunk while its answer sits in the neighbour.
`METADATA_EXTRACTOR=none` so the only expansion is the sequence chain. Compare:
  * expansion ON  : SEED_COUNT>0, SEQUENCE_MAX_HOPS>=1
  * expansion OFF : SEED_COUNT=0
If ON covers more answers than OFF, NEXT_CHUNK recovers split context.

Run (bench runner; set the knobs per arm):
  docker compose -f bench/docker-compose.yml run --rm \
    -e CHUNK_TARGET_CHARS=300 -e SEED_COUNT=8 -e SEQUENCE_MAX_HOPS=2 \
    runner python -m bench.chunk_context
"""

import asyncio
import os

from bench.run_benchmark import install_local_models

N = int(os.environ.get("BENCH_SQUAD_N", "400"))
TOPK = int(os.environ.get("BENCH_COVERAGE_K", "10"))


async def main():
    from datasets import load_dataset

    print(f"loading SQuAD validation (first {N} questions)...", flush=True)
    ds = load_dataset("rajpurkar/squad", split="validation")

    # one document per article (unique paragraphs concatenated); questions carry
    # their gold answer string
    paras_by_title: dict[str, list[str]] = {}
    seen: set[tuple[str, str]] = set()
    questions: list[tuple[str, str, str]] = []
    for ex in ds:
        title, ctx = ex["title"], ex["context"]
        if (title, ctx) not in seen:
            seen.add((title, ctx))
            paras_by_title.setdefault(title, []).append(ctx)
        ans = ex["answers"]["text"]
        if ans:
            questions.append((title, ex["question"], ans[0]))
    questions = questions[:N]
    docs = {t: "\n\n".join(ps) for t, ps in paras_by_title.items()}
    print(
        f"docs(articles)={len(docs)} questions={len(questions)} "
        f"backend={os.environ.get('STORE_BACKEND','neo4j')} "
        f"chunk_chars={os.environ.get('CHUNK_TARGET_CHARS','default')} "
        f"seed_count={os.environ.get('SEED_COUNT','default')}",
        flush=True,
    )

    install_local_models()
    from app.config import get_settings
    from app.ingest import ingest_document
    from app.search import search
    from app.store import create_store

    store = create_store(get_settings())
    await store.connect()
    await store.init_schema()
    for i, (t, text) in enumerate(docs.items()):
        await ingest_document(store, None, text, title=t, source="squad", document_id=t)
        if i % 100 == 0:
            print(f"  ingested {i}/{len(docs)}", flush=True)

    covered = 0
    for j, (_t, q, ans) in enumerate(questions):
        hits = await search(store, None, q, top_k=TOPK)
        blob = "\n".join(h.text for h in hits).lower()
        if ans.lower() in blob:
            covered += 1
        if j % 100 == 0:
            print(f"  queried {j}/{len(questions)}", flush=True)
    await store.close()

    print(
        f"\nanswer-coverage@{TOPK} = {covered/len(questions):.4f} "
        f"({covered}/{len(questions)})",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
