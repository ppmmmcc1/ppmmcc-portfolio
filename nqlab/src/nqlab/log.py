"""Logging setup: rich console + optional file."""

from __future__ import annotations

import logging
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: int = logging.INFO, logfile: Path | None = None) -> None:
    global _CONFIGURED
    root = logging.getLogger("nqlab")
    if not _CONFIGURED:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(show_path=False, rich_tracebacks=True)
        except Exception:  # pragma: no cover - rich is a hard dependency
            handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
        root.setLevel(level)
        _CONFIGURED = True
    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(logfile, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"nqlab.{name}")
