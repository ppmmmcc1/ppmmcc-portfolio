"""The holdout lockbox.

The final ``holdout_months`` of data are moved to ``data/holdout/`` on first ingest and
may only be read inside :func:`holdout_access`, which only ``nqlab holdout`` enters.

Two layers of protection:

1. A process-wide audit hook rejects Python-level ``open``/``listdir``/``scandir`` of
   any path under a registered holdout directory unless access is active.
2. Native readers (polars / pyarrow) bypass audit hooks, so every data read goes through
   ``nqlab.data.store``, which calls :func:`assert_readable` first.

Consumption is recorded permanently in the results DB by the holdout command.
"""

from __future__ import annotations

import os
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_GUARDED: set[str] = set()
_ACCESS = threading.local()
_HOOK_INSTALLED = False
_AUDITED = {"open", "os.listdir", "os.scandir", "glob.glob", "shutil.copyfile",
            "shutil.copytree", "os.rename", "os.remove", "shutil.rmtree"}


class HoldoutAccessError(PermissionError):
    pass


def _norm(p: str | os.PathLike) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(p)))


def _active() -> bool:
    return getattr(_ACCESS, "depth", 0) > 0 or getattr(_ACCESS, "ingest", 0) > 0


def _under_guard(path: str) -> bool:
    n = _norm(path)
    for g in _GUARDED:
        if n == g or n.startswith(g + os.sep):
            return True
    return False


def _hook(event: str, args: tuple) -> None:
    if event not in _AUDITED or not _GUARDED or _active():
        return
    if not args:
        return
    target = args[0]
    if isinstance(target, int) or target is None:
        return
    try:
        if isinstance(target, (str, bytes, os.PathLike)):
            t = os.fsdecode(target)
            if _under_guard(t):
                raise HoldoutAccessError(
                    f"holdout data is locked ({t}). Only `nqlab holdout <family>` may read it."
                )
    except HoldoutAccessError:
        raise
    except Exception:
        return


def register_guard(holdout_dir: Path) -> None:
    """Protect ``holdout_dir`` for the rest of this process."""
    global _HOOK_INSTALLED
    _GUARDED.add(_norm(holdout_dir))
    if not _HOOK_INSTALLED:
        sys.addaudithook(_hook)
        _HOOK_INSTALLED = True


def assert_readable(path: Path) -> None:
    if _under_guard(os.fspath(path)) and not _active():
        raise HoldoutAccessError(
            f"holdout data is locked ({path}). Only `nqlab holdout <family>` may read it."
        )


@contextmanager
def holdout_access(reason: str) -> Iterator[None]:
    """Grant read access to the holdout for the current thread. Only the holdout command
    may call this; ``tests/test_holdout.py`` asserts no other module does."""
    if not reason:
        raise ValueError("holdout access requires a reason")
    _ACCESS.depth = getattr(_ACCESS, "depth", 0) + 1
    try:
        yield
    finally:
        _ACCESS.depth -= 1


@contextmanager
def _ingest_write_access() -> Iterator[None]:
    """Ingest moves data *into* the lockbox; it never reads it back for research."""
    _ACCESS.ingest = getattr(_ACCESS, "ingest", 0) + 1
    try:
        yield
    finally:
        _ACCESS.ingest -= 1
