"""Engram-DB — a purpose-built, embedded RAG store (prototype).

One process, no server, no GDS. Implements the engram `Store` protocol with only
the parts the evaluation ([docs/engram-db.md](../docs/engram-db.md)) showed pull
their weight, and leaves out the overhead that didn't:

  * dense vector ANN  → cosine over an in-memory matrix per channel (brute force
                        in this prototype; usearch/HNSW + int8/binary quant is the
                        production swap)
  * lexical (BM25)    → an in-memory inverted index over text + summary + context
                        (so **contextual BM25** comes for free)
  * graph             → **native in-memory adjacency**: NEXT_CHUNK (doc → ordered
                        chunk ids) and HAS_KEYWORD (keyword → chunk ids), traversed
                        directly. The eval showed a SQL keyword self-join is the
                        worst-scaling store op; native adjacency is the fix.
  * proximity         → **decay only** (`graph_proximity` returns None → the
                        pipeline uses per-hop decay). PPR/PageRank is deliberately
                        omitted: it added no measurable quality for ~65 % of the
                        latency at scale.

Supported: multi-tenant isolation, recency, contextual retrieval (+ BM25),
learned-sparse re-scoring, near-duplicate links, implicit feedback. Deliberately
**not** supported (no measured payoff here, all the cost): PPR/GDS, community
synthesis, structured-entity graph — those stay Neo4j-only.

Persistence: an optional pickle snapshot at `path` (in-memory only when `path` is
None). A prototype: brute-force ANN and whole-snapshot persistence are fine at
small/medium scale; the production tier swaps in an ANN index + an incremental
on-disk segment format.
"""

from __future__ import annotations

import math
import pickle
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .store import STORES

if TYPE_CHECKING:
    from .channels import VectorChannel
    from .config import Settings

_TOKEN_RE = re.compile(r"\w+")
_BM25_K1 = 1.5
_BM25_B = 0.75


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class EngramDBStore:
    """Embedded vector + lexical + native-graph store (the engram Store protocol)."""

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path) if path else None
        # documents: id -> {title, sources, created_at}
        self._docs: dict[str, dict[str, Any]] = {}
        # chunks: id -> full chunk record (text/summary/keywords/embeddings/...)
        self._chunks: dict[str, dict[str, Any]] = {}
        # NEXT_CHUNK adjacency: doc_id -> [chunk_id in seq order]
        self._doc_chunks: dict[str, list[str]] = {}
        # HAS_KEYWORD adjacency: lowercased keyword -> set(chunk_id)
        self._keyword_index: dict[str, set[str]] = {}
        # lexical inverted index: token -> {chunk_id: term-frequency}
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_len: dict[str, int] = {}  # chunk_id -> token count
        # implicit-relevance feedback: list of (query, query_id, [chunk_id])
        self._feedback: list[dict[str, Any]] = []

    # ── lifecycle ────────────────────────────────────────────────────────────
    async def connect(self) -> None:
        if self._path and self._path.exists():
            with open(self._path, "rb") as f:
                state = pickle.load(f)
            self.__dict__.update(state)
            self._path = Path(self._path)  # restore after dict update

    async def init_schema(self) -> None:
        # nothing to build: the in-memory indexes are maintained on write
        return None

    async def verify_connectivity(self) -> None:
        return None  # always reachable (in-process)

    async def close(self) -> None:
        if self._path:
            state = {k: v for k, v in self.__dict__.items() if k != "_path"}
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "wb") as f:
                pickle.dump(state, f)

    # ── index maintenance ────────────────────────────────────────────────────
    def _index_chunk(self, rec: dict[str, Any]) -> None:
        cid = rec["id"]
        # lexical: text + summary + context (contextual BM25)
        blob = " ".join(
            x for x in (rec.get("text"), rec.get("summary"), rec.get("context")) if x
        )
        tokens = _tokenize(blob)
        self._doc_len[cid] = len(tokens)
        tf: dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        for t, c in tf.items():
            self._postings.setdefault(t, {})[cid] = c
        # keyword adjacency
        for kw in {k.lower() for k in rec.get("keywords") or []}:
            self._keyword_index.setdefault(kw, set()).add(cid)

    def _deindex_chunk(self, cid: str) -> None:
        rec = self._chunks.get(cid)
        if rec is None:
            return
        blob = " ".join(
            x for x in (rec.get("text"), rec.get("summary"), rec.get("context")) if x
        )
        for t in set(_tokenize(blob)):
            posting = self._postings.get(t)
            if posting:
                posting.pop(cid, None)
                if not posting:
                    self._postings.pop(t, None)
        for kw in {k.lower() for k in rec.get("keywords") or []}:
            s = self._keyword_index.get(kw)
            if s:
                s.discard(cid)
                if not s:
                    self._keyword_index.pop(kw, None)
        self._doc_len.pop(cid, None)

    def _drop_document(self, doc_id: str) -> int:
        n = 0
        for cid in self._doc_chunks.pop(doc_id, []):
            self._deindex_chunk(cid)
            self._chunks.pop(cid, None)
            n += 1
        self._docs.pop(doc_id, None)
        return n

    # ── documents ────────────────────────────────────────────────────────────
    async def save_document(
        self, doc_id: str, title: str, sources: list[str], chunks: list[dict[str, Any]]
    ) -> None:
        # replace any previous version (keeps re-ingest idempotent)
        existing = self._docs.get(doc_id)
        created_at = existing["created_at"] if existing else time.time()
        self._drop_document(doc_id)
        self._docs[doc_id] = {
            "title": title, "sources": list(sources), "created_at": created_at,
        }
        ordered: list[str] = []
        for ch in chunks:
            rec = {
                "id": ch["id"],
                "doc_id": doc_id,
                "seq": ch["seq"],
                "text": ch.get("text"),
                "summary": ch.get("summary") or "",
                "keywords": list(ch.get("keywords") or []),
                "sparse_weights": ch.get("sparse_weights"),
                "near_dup_of": ch.get("near_dup_of"),
                "tenant_id": ch.get("tenant_id"),
                "context": ch.get("context") or "",
                "embeddings": {
                    p: np.asarray(v, dtype=np.float32)
                    for p, v in ch["embeddings"].items()
                },
            }
            self._chunks[rec["id"]] = rec
            self._index_chunk(rec)
            ordered.append(rec["id"])
        ordered.sort(key=lambda c: self._chunks[c]["seq"])
        self._doc_chunks[doc_id] = ordered

    async def delete_document(self, doc_id: str) -> int | None:
        if doc_id not in self._docs:
            return None
        return self._drop_document(doc_id)

    async def get_document(self, doc_id: str) -> dict[str, Any] | None:
        doc = self._docs.get(doc_id)
        if doc is None:
            return None
        cids = self._doc_chunks.get(doc_id, [])
        keywords = sorted(
            {k.lower() for c in cids for k in self._chunks[c].get("keywords") or []}
        )
        return {
            "sources": list(doc["sources"]),
            "chunk_count": len(cids),
            "keywords": keywords,
        }

    async def add_document_source(self, doc_id: str, source: str) -> None:
        doc = self._docs.get(doc_id)
        if doc and source not in doc["sources"]:
            doc["sources"].append(source)

    async def remove_document_source(
        self, doc_id: str, source: str
    ) -> dict[str, Any] | None:
        doc = self._docs.get(doc_id)
        if doc is None:
            return None
        remaining = [s for s in doc["sources"] if s != source]
        if remaining:
            doc["sources"] = remaining
            return {"deleted": False, "remaining_sources": remaining,
                    "deleted_chunks": None}
        deleted_chunks = self._drop_document(doc_id)
        return {"deleted": True, "remaining_sources": [],
                "deleted_chunks": deleted_chunks}

    async def list_documents(self) -> list[dict[str, Any]]:
        out = []
        for doc_id, doc in self._docs.items():
            out.append({
                "id": doc_id,
                "title": doc["title"],
                "sources": list(doc["sources"]),
                "created_at": None,
                "chunk_count": len(self._doc_chunks.get(doc_id, [])),
            })
        out.sort(key=lambda d: self._docs[d["id"]]["created_at"], reverse=True)
        return out

    async def fetch_document_chunks(
        self, doc_id: str, embedding_props: list[str]
    ) -> list[dict[str, Any]]:
        out = []
        for cid in self._doc_chunks.get(doc_id, []):
            rec = self._chunks[cid]
            out.append({
                "text": rec["text"],
                "summary": rec["summary"],
                "keywords": list(rec["keywords"]),
                "embeddings": {
                    p: rec["embeddings"][p].tolist()
                    for p in embedding_props if p in rec["embeddings"]
                },
                "sparse_weights": rec.get("sparse_weights"),
            })
        return out

    # ── retrieval ────────────────────────────────────────────────────────────
    def _tenant_ok(self, rec: dict[str, Any], tenant_id: str | None) -> bool:
        return tenant_id is None or rec.get("tenant_id") == tenant_id

    def _hit(self, rec: dict[str, Any], score: float) -> dict[str, Any]:
        return {
            "id": rec["id"], "doc_id": rec["doc_id"], "text": rec["text"],
            "summary": rec["summary"], "keywords": list(rec["keywords"]),
            "content_embedding": rec["embeddings"]["content_embedding"].tolist(),
            "tenant_id": rec.get("tenant_id"), "score": score,
        }

    async def vector_search(
        self, channel: "VectorChannel", embedding: list[float], k: int,
        tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        prop = channel.embedding_prop
        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scored = []
        for rec in self._chunks.values():
            if not self._tenant_ok(rec, tenant_id):
                continue
            v = rec["embeddings"].get(prop)
            if v is None:
                continue
            denom = (float(np.linalg.norm(v)) or 1.0) * qn
            scored.append((float(np.dot(q, v)) / denom, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._hit(rec, s) for s, rec in scored[:k]]

    def _bm25(self, query: str, tenant_id: str | None) -> dict[str, float]:
        terms = _tokenize(query)
        if not terms or not self._doc_len:
            return {}
        n = len(self._doc_len)
        avgdl = sum(self._doc_len.values()) / n
        scores: dict[str, float] = {}
        for t in set(terms):
            posting = self._postings.get(t)
            if not posting:
                continue
            idf = math.log(1 + (n - len(posting) + 0.5) / (len(posting) + 0.5))
            for cid, tf in posting.items():
                dl = self._doc_len.get(cid, 0)
                denom = tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / avgdl)
                scores[cid] = scores.get(cid, 0.0) + idf * (tf * (_BM25_K1 + 1)) / denom
        return scores

    async def fulltext_search(
        self, query: str, k: int, tenant_id: str | None = None
    ) -> list[dict[str, Any]]:
        scores = self._bm25(query, tenant_id)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        out = []
        for cid, s in ranked:
            rec = self._chunks.get(cid)
            if rec is None or not self._tenant_ok(rec, tenant_id):
                continue
            out.append(self._hit(rec, s))
            if len(out) >= k:
                break
        return out

    async def fetch_siblings(
        self, seed_ids: list[str], keyword_sibling_limit: int, sequence_max_hops: int
    ) -> list[dict[str, Any]]:
        hops = max(1, int(sequence_max_hops))
        seeds = set(seed_ids)
        out: list[dict[str, Any]] = []
        for sid in seed_ids:
            seed = self._chunks.get(sid)
            if seed is None:
                continue
            # NEXT_CHUNK: same document, within ±hops of the seed's seq position
            ordered = self._doc_chunks.get(seed["doc_id"], [])
            try:
                pos = ordered.index(sid)
            except ValueError:
                pos = None
            if pos is not None:
                for off in range(-hops, hops + 1):
                    j = pos + off
                    if off == 0 or j < 0 or j >= len(ordered):
                        continue
                    cid = ordered[j]
                    if cid in seeds:
                        continue
                    sib = self._chunks[cid]
                    out.append(self._sib(seed, sib, "sequence",
                                         "after" if off > 0 else "before",
                                         abs(off), 1.0))
            # HAS_KEYWORD: chunks sharing a keyword, top-N by shared count
            shared: dict[str, int] = {}
            for kw in {k.lower() for k in seed.get("keywords") or []}:
                for cid in self._keyword_index.get(kw, ()):
                    if cid != sid and cid not in seeds:
                        shared[cid] = shared.get(cid, 0) + 1
            for cid, count in sorted(
                shared.items(), key=lambda x: x[1], reverse=True
            )[:keyword_sibling_limit]:
                out.append(self._sib(seed, self._chunks[cid], "keyword",
                                     "lateral", 1, float(count)))
        return out

    def _sib(self, seed, sib, via, direction, distance, strength) -> dict[str, Any]:
        return {
            "seed_id": seed["id"], "id": sib["id"], "doc_id": sib["doc_id"],
            "text": sib["text"], "summary": sib["summary"],
            "keywords": list(sib["keywords"]),
            "content_embedding": sib["embeddings"]["content_embedding"].tolist(),
            "tenant_id": sib.get("tenant_id"), "via": via, "direction": direction,
            "distance": distance, "strength": strength,
        }

    async def fetch_context(
        self, chunk_id: str, before: int, after: int
    ) -> list[dict[str, Any]] | None:
        anchor = self._chunks.get(chunk_id)
        if anchor is None:
            return None
        ordered = self._doc_chunks.get(anchor["doc_id"], [])
        try:
            pos = ordered.index(chunk_id)
        except ValueError:
            return None
        lo, hi = max(0, pos - int(before)), min(len(ordered), pos + int(after) + 1)
        out = []
        for j in range(lo, hi):
            rec = self._chunks[ordered[j]]
            out.append({
                "id": rec["id"], "doc_id": rec["doc_id"], "seq": rec["seq"],
                "text": rec["text"], "summary": rec["summary"],
                "keywords": list(rec["keywords"]), "offset": rec["seq"] - anchor["seq"],
            })
        return out

    async def get_sparse_weights(
        self, chunk_ids: list[str]
    ) -> dict[str, dict[str, float]]:
        out = {}
        for cid in chunk_ids:
            rec = self._chunks.get(cid)
            if rec and rec.get("sparse_weights"):
                out[cid] = {str(k): float(v) for k, v in rec["sparse_weights"].items()}
        return out

    async def nearest_chunks(
        self, embedding: list[float], k: int, min_sim: float,
        exclude_doc_id: str | None = None, tenant_id: str | None = None,
    ) -> list[dict[str, Any]]:
        q = np.asarray(embedding, dtype=np.float32)
        qn = float(np.linalg.norm(q)) or 1.0
        scored = []
        for rec in self._chunks.values():
            if exclude_doc_id and rec["doc_id"] == exclude_doc_id:
                continue
            if not self._tenant_ok(rec, tenant_id):
                continue
            v = rec["embeddings"].get("content_embedding")
            if v is None:
                continue
            sim = float(np.dot(q, v)) / ((float(np.linalg.norm(v)) or 1.0) * qn)
            if sim >= min_sim:
                scored.append((sim, rec))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"id": r["id"], "doc_id": r["doc_id"], "seq": r["seq"],
             "text": r["text"], "sim": s}
            for s, r in scored[:k]
        ]

    async def get_near_dup_links(self, chunk_ids: list[str]) -> dict[str, str]:
        out = {}
        for cid in chunk_ids:
            rec = self._chunks.get(cid)
            if rec and rec.get("near_dup_of"):
                out[cid] = rec["near_dup_of"]
        return out

    async def get_chunk_recency(self, chunk_ids: list[str]) -> dict[str, float]:
        now = time.time()
        out = {}
        for cid in chunk_ids:
            rec = self._chunks.get(cid)
            if rec is None:
                continue
            doc = self._docs.get(rec["doc_id"])
            if doc:
                out[cid] = max(0.0, now - doc["created_at"])
        return out

    async def record_feedback(
        self, query: str, used_chunk_ids: list[str], query_id: str | None = None
    ) -> int:
        valid = [c for c in used_chunk_ids if c in self._chunks]
        if valid:
            self._feedback.append(
                {"query": query, "query_id": query_id, "chunk_ids": valid}
            )
        return len(valid)

    async def graph_proximity(
        self, seed_ids: list[str], candidate_ids: list[str], damping: float
    ) -> dict[str, float] | None:
        # by design: no PPR — the pipeline falls back to per-hop decay (the eval
        # showed PPR adds no quality for the bulk of the latency)
        return None

    # ── optional capabilities: deliberately unsupported ──────────────────────
    async def detect_communities(self, min_size: int) -> list[dict[str, Any]] | None:
        return None

    async def save_communities(self, communities: list[dict[str, Any]]) -> int:
        raise NotImplementedError("community synthesis is neo4j-only")

    async def list_communities(self) -> list[dict[str, Any]]:
        return []

    async def community_vectors(self) -> list[dict[str, Any]]:
        return []

    async def upsert_entities(self, label: str, items: list[dict[str, Any]]) -> int:
        raise NotImplementedError("structured-entity ingest is neo4j-only")

    async def upsert_relations(
        self, from_label: str, rel_type: str, to_label: str,
        items: list[dict[str, Any]],
    ) -> int:
        raise NotImplementedError("structured-entity ingest is neo4j-only")


@STORES.register("engramdb")
def _make_engramdb_store(settings: "Settings") -> EngramDBStore:
    return EngramDBStore(getattr(settings, "engramdb_path", "") or None)
