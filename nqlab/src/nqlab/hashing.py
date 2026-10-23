"""Stable hashing for run ids, spec hashes and data versions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(obj: Any, n: int = 16) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()[:n]


def file_hash(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def git_commit(root: Path) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        sha = out.stdout.strip() or "unknown"
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=root, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"
