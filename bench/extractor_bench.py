"""Compare LLM models for engram's metadata-extraction task — speed AND quality.

The `default` metadata extractor makes one /chat/completions call per chunk for
3-8 keywords + a one-sentence summary as JSON (see app/llm.py). When you point it
at a small/fast model (EXTRACTION_LLM_*), you want to know: is it fast enough, and
is it good enough? This harness answers both over the REAL extraction prompt and
parser, across as many models as you can serve, with the no-LLM `yake` path as the
honest baseline.

It measures, per model:
  PERFORMANCE  mean/p50/p95 per-call latency, throughput (chunks/s) at a given
               concurrency, and mean completion tokens — the token count alone
               exposes a Qwen3 "thinking" model (hundreds of tokens for a job that
               needs ~40), so a slow/bloated config is obvious at a glance.
  RELIABILITY  % strictly-valid JSON, % needing the tolerant fallback parser,
               % hard failures (timeout/HTTP/no-JSON), and schema adherence
               (3-8 keywords, exactly one summary sentence).
  QUALITY      judge-free proxies via a local embedder (optional): how close the
               summary and the keywords sit to the chunk they describe (faithful /
               on-topic), compared head-to-head with the yake baseline on the same
               metric. The GOLD quality metric is downstream retrieval nDCG — run
               bench/run_benchmark.py with the chosen extractor for that; these
               proxies are the fast iteration loop for a model bake-off.

Usage:
  # 1) describe the models you've loaded (OpenAI-compatible endpoints):
  cat > models.json <<'JSON'
  [
    {"label": "qwen3-1.7b", "api_base": "http://localhost:8001/v1",
     "model": "Qwen/Qwen3-1.7B-Instruct-2507", "max_tokens": 96, "json_mode": true},
    {"label": "qwen3-0.6b", "api_base": "http://localhost:8002/v1",
     "model": "Qwen/Qwen3-0.6B", "max_tokens": 96, "json_mode": true}
  ]
  JSON

  # 2) run it over a corpus (a .txt / a .jsonl with a "text" field / a dir of
  #    .txt|.md), or omit --corpus to use the built-in sample chunks:
  python -m bench.extractor_bench --models models.json --corpus docs/ \
         --concurrency 8 --max-chunks 100 --out results.json

Per-model fields default to engram's settings: temperature 0.1, the shared
_EXTRACTION_SYSTEM_PROMPT, response_format=json_object when "json_mode" is true.
The yake baseline is always included; pass --no-yake to drop it, --no-quality to
skip the embedder (then only perf + reliability + schema are reported).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path

import httpx

from app.chunking import get_chunker
from app.config import get_settings
from app.llm import (
    EXTRACTION_JSON_SCHEMA,
    _EXTRACTION_SYSTEM_PROMPT,
    _parse_json_object,
    extract_yake,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

def _response_format(cfg: dict):
    """Resolve a model's `response_format`: "json_schema" (guaranteed + reasoning-
    suppressing), "json_object" (OpenAI standard), "text"/"none" (off), a full dict
    passed through verbatim, or — if unset — json_object when `json_mode` is true."""
    rf = cfg.get("response_format")
    if isinstance(rf, dict):
        return rf
    if rf is None:
        return {"type": "json_object"} if cfg.get("json_mode", True) else None
    rf = str(rf).lower()
    if rf in ("json_schema", "schema"):
        return EXTRACTION_JSON_SCHEMA
    if rf in ("json_object", "json"):
        return {"type": "json_object"}
    return None  # text / none / off

# A small, deliberately diverse default corpus (prose / technical / list / code /
# non-English) so the harness runs with no inputs and still stresses JSON
# adherence and topical faithfulness across chunk shapes.
_SAMPLE_CHUNKS = [
    "The mitochondrion is a double-membrane-bound organelle found in most "
    "eukaryotic cells. It generates most of the cell's supply of adenosine "
    "triphosphate (ATP), used as a source of chemical energy. Mitochondria have "
    "their own small genome, inherited maternally in most species.",
    "To configure TLS termination at the load balancer, set ssl_certificate and "
    "ssl_certificate_key in the server block, then redirect port 80 to 443 with a "
    "301. Enable HSTS only after you have verified every subdomain serves HTTPS, "
    "since the max-age directive is sticky in browsers.",
    "Quarterly revenue rose 12% year over year to $4.2B, driven by cloud "
    "services (+28%) while hardware declined 4%. Operating margin expanded 180 "
    "basis points to 31.5%. Management guided full-year growth to the low teens.",
    "Ingredients: 200g flour, 2 eggs, 100ml milk, a pinch of salt. Whisk the eggs "
    "and milk, fold in the sifted flour, rest the batter 30 minutes, then cook "
    "thin on a hot buttered pan until golden on each side.",
    "def binary_search(xs, target):\n    lo, hi = 0, len(xs) - 1\n    while lo <= "
    "hi:\n        mid = (lo + hi) // 2\n        if xs[mid] == target:\n            "
    "return mid\n        if xs[mid] < target:\n            lo = mid + 1\n        "
    "else:\n            hi = mid - 1\n    return -1",
    "La Convention de Ramsar, signée en 1971, est un traité intergouvernemental "
    "qui sert de cadre à la conservation et à l'utilisation rationnelle des zones "
    "humides et de leurs ressources. Elle compte aujourd'hui plus de 170 parties "
    "contractantes.",
    "The reranker is a cross-encoder that scores each (query, passage) pair "
    "jointly, unlike the bi-encoder retriever that embeds them independently. It "
    "is far more accurate but quadratic in candidates, so it runs only on a short "
    "shortlist after first-stage retrieval and fusion.",
    "Symptoms of acute appendicitis classically begin with periumbilical pain that "
    "migrates to the right lower quadrant, accompanied by anorexia, low-grade "
    "fever, and rebound tenderness at McBurney's point. Delay risks perforation.",
]


def load_chunks(corpus: str | None, max_chunks: int) -> list[str]:
    """Chunk the corpus with engram's real chunker (or use the built-in sample)."""
    if not corpus:
        chunks = list(_SAMPLE_CHUNKS)
        return chunks[:max_chunks] if max_chunks else chunks

    settings = get_settings()
    chunker = get_chunker(settings.chunk_strategy)
    path = Path(corpus)
    texts: list[str] = []
    if path.is_dir():
        for p in sorted(path.rglob("*")):
            if p.suffix.lower() in {".txt", ".md"}:
                texts.append(p.read_text(encoding="utf-8", errors="ignore"))
    elif path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                obj = json.loads(line)
                t = obj.get("text") or obj.get("contents") or ""
                if obj.get("title"):
                    t = f"{obj['title']}\n{t}"
                if t.strip():
                    texts.append(t)
    else:
        texts.append(path.read_text(encoding="utf-8", errors="ignore"))

    chunks: list[str] = []
    for t in texts:
        chunks.extend(chunker(t, settings))
        if max_chunks and len(chunks) >= max_chunks:
            break
    return chunks[:max_chunks] if max_chunks else chunks


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * frac


async def _call_model(client: httpx.AsyncClient, cfg: dict, chunk: str) -> dict:
    """One extraction call; returns a per-chunk record (never raises)."""
    # `prompt_suffix` appends a model-specific control token to the system message
    # (e.g. Qwen3's "/no_think"); `extra_body` merges serving controls at the TOP
    # level of the request body (e.g. {"reasoning_effort": "none"} — note there is
    # no nested "extra_body" wire key; that's an OpenAI-SDK-only convenience).
    sys_content = _EXTRACTION_SYSTEM_PROMPT + cfg.get("prompt_suffix", "")
    body: dict = {
        "model": cfg["model"],
        "temperature": cfg.get("temperature", 0.1),
        "messages": [
            {"role": "system", "content": sys_content},
            {"role": "user", "content": chunk},
        ],
    }
    if cfg.get("max_tokens"):
        body["max_tokens"] = cfg["max_tokens"]
    rf = _response_format(cfg)
    if rf:
        body["response_format"] = rf
    if cfg.get("extra_body"):
        body.update(cfg["extra_body"])
    headers = {}
    if cfg.get("api_key"):
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    t0 = time.perf_counter()
    try:
        resp = await client.post(
            f"{cfg['api_base'].rstrip('/')}/chat/completions",
            json=body,
            headers=headers,
            timeout=cfg.get("timeout", 120.0),
        )
        resp.raise_for_status()
        data = resp.json()
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        # a server that splits reasoning out (LM Studio, vLLM reasoning-parser) puts
        # it here; a non-zero count means thinking is still ON (also visible as a
        # high completion-token count, since usage counts reasoning tokens).
        reasoning_chars = len(msg.get("reasoning_content") or msg.get("reasoning") or "")
        latency = time.perf_counter() - t0
    except Exception as e:  # noqa: BLE001 — record the failure, keep going
        return {"ok": False, "error": str(e)[:120], "latency": time.perf_counter() - t0}

    strict = None
    try:
        strict = json.loads(content.strip())
    except json.JSONDecodeError:
        strict = None
    parsed = _parse_json_object(content)
    keywords = [str(k).strip() for k in parsed.get("keywords", []) if str(k).strip()]
    summary = str(parsed.get("summary", "")).strip()
    usage = data.get("usage") or {}
    return {
        "ok": True,
        "latency": latency,
        "strict_json": isinstance(strict, dict) and "keywords" in strict and "summary" in strict,
        "parsed_ok": bool(keywords or summary),
        "keywords": keywords[:8],
        "summary": summary,
        "n_keywords": len(keywords),
        "n_sentences": len([s for s in _SENT_SPLIT.split(summary) if s.strip()]) if summary else 0,
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": reasoning_chars,
    }


async def run_model(cfg: dict, chunks: list[str], concurrency: int) -> list[dict]:
    sem = asyncio.Semaphore(max(1, concurrency))
    limits = httpx.Limits(max_connections=max(8, concurrency * 2))
    async with httpx.AsyncClient(limits=limits) as client:
        async def one(chunk: str) -> dict:
            async with sem:
                return await _call_model(client, cfg, chunk)

        t0 = time.perf_counter()
        records = await asyncio.gather(*(one(c) for c in chunks))
        wall = time.perf_counter() - t0
    for r in records:
        r["_wall"] = wall
    return records


async def run_yake(chunks: list[str]) -> list[dict]:
    """The no-LLM baseline (statistical keywords + lead sentence), same metrics."""
    records = []
    t0 = time.perf_counter()
    for c in chunks:
        s0 = time.perf_counter()
        r = await extract_yake(None, c)
        summary = r.summary
        records.append({
            "ok": True,
            "latency": time.perf_counter() - s0,
            "strict_json": True,  # constructed locally, always valid
            "parsed_ok": True,
            "keywords": list(r.keywords)[:8],
            "summary": summary,
            "n_keywords": len(r.keywords),
            "n_sentences": len([s for s in _SENT_SPLIT.split(summary) if s.strip()]) if summary else 0,
            "completion_tokens": 0,
            "reasoning_chars": 0,
        })
    wall = time.perf_counter() - t0
    for r in records:
        r["_wall"] = wall
    return records


def _load_embedder(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return None
    try:
        return SentenceTransformer(name)
    except Exception:  # noqa: BLE001 — offline / model missing
        return None


class _RemoteEmbedder:
    """Embeds via an OpenAI-compatible /embeddings endpoint, so the quality proxies
    can use the SAME embedder engram retrieves with (e.g. a served bge-m3) instead
    of a local sentence-transformers download. Exposes the .encode() signature the
    proxy code expects."""

    def __init__(self, api_base: str, model: str, api_key: str = ""):
        self._base = api_base.rstrip("/")
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def encode(self, texts, normalize_embeddings: bool = True):
        import numpy as np

        payload = [t if t else " " for t in texts]  # some servers reject empty input
        r = httpx.post(
            f"{self._base}/embeddings",
            headers=self._headers,
            json={"model": self._model, "input": list(payload)},
            timeout=120,
        )
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        arr = np.array([d["embedding"] for d in data], dtype=float)
        if normalize_embeddings:
            arr = arr / np.clip(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12, None)
        return arr


def _cosines(embedder, chunks: list[str], records: list[dict]) -> tuple[float, float]:
    """Mean cosine of (summary, chunk) and (keywords, chunk) for successful rows."""
    import numpy as np

    pairs = [(c, r) for c, r in zip(chunks, records) if r.get("ok")]
    if not pairs:
        return 0.0, 0.0
    src = [c for c, _ in pairs]
    summaries = [r["summary"] or "" for _, r in pairs]
    keywords = [", ".join(r["keywords"]) or "" for _, r in pairs]
    cv = embedder.encode(src, normalize_embeddings=True)
    sv = embedder.encode(summaries, normalize_embeddings=True)
    kv = embedder.encode(keywords, normalize_embeddings=True)
    sum_cos = float(np.mean([float(np.dot(a, b)) for a, b in zip(sv, cv)]))
    kw_cos = float(np.mean([float(np.dot(a, b)) for a, b in zip(kv, cv)]))
    return sum_cos, kw_cos


def aggregate(label: str, records: list[dict], embedder, chunks: list[str]) -> dict:
    ok = [r for r in records if r.get("ok")]
    lat = sorted(r["latency"] for r in ok)
    n = len(records)
    wall = records[0]["_wall"] if records else 0.0
    toks = [r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None]
    schema_ok = [r for r in ok if 3 <= r["n_keywords"] <= 8 and r["n_sentences"] == 1]
    agg = {
        "label": label,
        "n": n,
        "ok_rate": len(ok) / n if n else 0.0,
        "throughput_cps": (n / wall) if wall else 0.0,
        "lat_mean": (sum(lat) / len(lat)) if lat else 0.0,
        "lat_p50": _percentile(lat, 0.50),
        "lat_p95": _percentile(lat, 0.95),
        "strict_json_rate": (sum(1 for r in ok if r.get("strict_json")) / len(ok)) if ok else 0.0,
        "schema_ok_rate": (len(schema_ok) / len(ok)) if ok else 0.0,
        "mean_keywords": (sum(r["n_keywords"] for r in ok) / len(ok)) if ok else 0.0,
        "mean_completion_tokens": (sum(toks) / len(toks)) if toks else None,
        # mean chars of separated reasoning; >0 means a thinking mode is still ON
        # (server split it out of content) — the at-a-glance "did I disable it?" check
        "mean_reasoning_chars": (sum(r.get("reasoning_chars", 0) for r in ok) / len(ok)) if ok else 0.0,
        "summary_cos": None,
        "keywords_cos": None,
    }
    if embedder is not None:
        try:
            agg["summary_cos"], agg["keywords_cos"] = _cosines(embedder, chunks, records)
        except Exception as e:  # noqa: BLE001 — a flaky embedder must not lose the perf data
            agg["quality_error"] = str(e)[:140]
    return agg


def print_table(rows: list[dict]) -> None:
    cols = [
        ("model", "label", "{:<18}"),
        ("ok%", "ok_rate", "{:>5.0%}"),
        ("cps", "throughput_cps", "{:>6.1f}"),
        ("p50 s", "lat_p50", "{:>6.2f}"),
        ("p95 s", "lat_p95", "{:>6.2f}"),
        ("tok", "mean_completion_tokens", "{:>5.0f}"),
        ("rsn", "mean_reasoning_chars", "{:>5.0f}"),
        ("json%", "strict_json_rate", "{:>5.0%}"),
        ("schema%", "schema_ok_rate", "{:>7.0%}"),
        ("kw#", "mean_keywords", "{:>4.1f}"),
        ("sum~chunk", "summary_cos", "{:>9.3f}"),
        ("kw~chunk", "keywords_cos", "{:>8.3f}"),
    ]
    names = [c[0] for c in cols]
    widths = [18, 5, 6, 6, 6, 5, 5, 5, 7, 4, 9, 8]
    print("  ".join(f"{nm:<{w}}" if i == 0 else f"{nm:>{w}}"
                    for i, (nm, w) in enumerate(zip(names, widths))))
    for r in rows:
        cells = []
        for i, (_name, key, fmt) in enumerate(cols):
            v = r.get(key)
            if v is None:
                cells.append(f"{'-':>{widths[i]}}" if i else f"{'-':<{widths[i]}}")
            else:
                cells.append(fmt.format(v))
        print("  ".join(cells))


def load_models(
    models_path: str | None, use_settings: bool, default_key_env: str | None = None
) -> list[dict]:
    models: list[dict] = []
    if models_path:
        models.extend(json.loads(Path(models_path).read_text(encoding="utf-8")))
    if use_settings or not models_path:
        s = get_settings()
        api_base = s.extraction_llm_api_base or s.llm_api_base
        model = s.extraction_llm_model or s.llm_model
        if api_base and model:
            models.append({
                "label": "configured",
                "api_base": api_base,
                "model": model,
                "api_key": s.extraction_llm_api_key or s.llm_api_key,
                "json_mode": s.llm_json_mode,
                "max_tokens": s.extraction_max_tokens or None,
            })
    # resolve each model's key from an env var (keep secrets out of models.json):
    # per-model "api_key_env", else the shared --api-key-env fallback.
    for m in models:
        if not m.get("api_key"):
            env_name = m.get("api_key_env") or default_key_env
            if env_name:
                m["api_key"] = os.environ.get(env_name, "")
    # de-dup labels
    seen: set[str] = set()
    for m in models:
        base = m.get("label", m.get("model", "model"))
        lbl, i = base, 1
        while lbl in seen:
            i += 1
            lbl = f"{base}-{i}"
        m["label"] = lbl
        seen.add(lbl)
    return models


async def main() -> None:
    ap = argparse.ArgumentParser(description="Compare LLM models for engram metadata extraction.")
    ap.add_argument("--models", help="JSON file: list of per-model dicts. Required: api_base, model. "
                    "Optional: label, api_key, api_key_env, max_tokens, temperature, json_mode, "
                    "response_format ('json_object'|'json_schema'|'text' or a full dict), "
                    "extra_body (merged at the TOP level of the request — e.g. "
                    '{"chat_template_kwargs":{"enable_thinking":false}} for vLLM Qwen3, or '
                    '{"reasoning_effort":"none"}), prompt_suffix (e.g. " /no_think").')
    ap.add_argument("--use-settings", action="store_true", help="also test the model from EXTRACTION_LLM_*/LLM_* settings")
    ap.add_argument("--api-key-env", help="env var holding the API key for models that set neither api_key nor api_key_env")
    ap.add_argument("--corpus", help="a .txt / .jsonl (text field) / dir of .txt|.md; omit for the built-in sample")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-chunks", type=int, default=64)
    ap.add_argument("--no-yake", action="store_true", help="drop the no-LLM baseline row")
    ap.add_argument("--no-quality", action="store_true", help="skip the embedder (perf + reliability only)")
    ap.add_argument("--embedder", default="sentence-transformers/all-MiniLM-L6-v2", help="local sentence-transformers model for the quality proxies")
    ap.add_argument("--embedder-api", help="OpenAI-compatible /v1 base for the quality-proxy embedder (e.g. your served bge-m3); overrides --embedder, no local download")
    ap.add_argument("--embedder-model", default="BAAI/bge-m3", help="model id for --embedder-api")
    ap.add_argument("--embedder-key-env", help="env var holding the --embedder-api key")
    ap.add_argument("--out", help="write the full per-model results as JSON here")
    args = ap.parse_args()

    chunks = load_chunks(args.corpus, args.max_chunks)
    models = load_models(args.models, args.use_settings, args.api_key_env)
    if not models and args.no_yake:
        ap.error("no models to test: pass --models, --use-settings, or drop --no-yake")
    print(f"chunks={len(chunks)} models={len(models)}{' +yake' if not args.no_yake else ''} "
          f"concurrency={args.concurrency}", flush=True)

    embedder = None
    if not args.no_quality:
        if args.embedder_api:
            key = os.environ.get(args.embedder_key_env, "") if args.embedder_key_env else ""
            embedder = _RemoteEmbedder(args.embedder_api, args.embedder_model, key)
        else:
            embedder = _load_embedder(args.embedder)
        if embedder is None:
            print("(quality proxies skipped: no local embedder; pass --embedder-api "
                  "to use a served embedder instead)", flush=True)

    rows: list[dict] = []
    full: dict[str, list[dict]] = {}

    if not args.no_yake:
        recs = await run_yake(chunks)
        full["yake"] = recs
        rows.append(aggregate("yake (no-LLM)", recs, embedder, chunks))
        print(f"  ran yake baseline ({len(chunks)} chunks)", flush=True)

    for cfg in models:
        print(f"  running {cfg['label']} ({cfg['model']}) ...", flush=True)
        recs = await run_model(cfg, chunks, args.concurrency)
        full[cfg["label"]] = recs
        rows.append(aggregate(cfg["label"], recs, embedder, chunks))
        fails = [r for r in recs if not r.get("ok")]
        if fails:
            print(f"    {len(fails)}/{len(recs)} calls failed, e.g. {fails[0].get('error')}", flush=True)

    print("\n=== extractor comparison (cps=chunks/sec at the given concurrency; "
          "tok=mean completion tokens; ~chunk=cosine faithfulness proxy) ===")
    print_table(rows)
    qerr = next((r["quality_error"] for r in rows if r.get("quality_error")), None)
    if qerr:
        print(f"\n(quality proxies unavailable: {qerr} — perf/reliability above are unaffected)")
    print("\nNote: sum~chunk/kw~chunk are judge-free PROXIES (higher = more faithful/on-topic). "
          "The gold quality metric is retrieval nDCG -- run bench/run_benchmark.py with the chosen "
          "extractor. A high `tok` with slow p95 is the classic Qwen3 thinking-mode footgun.")

    if args.out:
        Path(args.out).write_text(
            json.dumps({"rows": rows, "records": full}, indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
