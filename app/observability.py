"""Lightweight observability: logging configuration.

engram deliberately keeps a tiny dependency footprint, so observability is
stdlib `logging` rather than a heavyweight tracing stack. `configure_logging`
sets a level + format at startup (honouring `LOG_LEVEL`), and the hot paths emit
a single structured DEBUG summary line per request — turn it on with
`LOG_LEVEL=DEBUG` to see per-search timing, candidate-pool sizes and which
degradation fallbacks fired, without instrumenting every stage.
"""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging. `force=True` so it wins over uvicorn's handlers."""
    resolved = getattr(logging, str(level).upper(), logging.INFO)
    logging.basicConfig(
        level=resolved,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
