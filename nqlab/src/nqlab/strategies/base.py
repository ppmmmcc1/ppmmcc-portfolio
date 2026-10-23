"""Strategy interface. Strategies are resolved BY NAME from the registry (also inside
worker processes on Windows spawn); never pass strategy objects or closures to workers."""

from __future__ import annotations

import hashlib
import itertools
from typing import Any

import numpy as np
import polars as pl

from nqlab.engine.bars import EngineBars
from nqlab.engine.orders import EngineParams, Signals
from nqlab.engine.simulate import Sizing


class Strategy:
    name: str = "base"
    description: str = ""
    # default parameter grid: name -> list of values (ordered unless listed in `categorical`)
    param_space: dict[str, list[Any]] = {}
    categorical: tuple[str, ...] = ()
    fixed: dict[str, Any] = {}           # non-grid params with defaults
    start_ct: int = 8 * 60               # engine window start (CT minutes)

    def defaults(self) -> dict[str, Any]:
        d = {k: v[len(v) // 2] for k, v in self.param_space.items()}
        d.update(self.fixed)
        return d

    def grid(self, space: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
        space = space or self.param_space
        keys = list(space)
        return [{**self.fixed, **dict(zip(keys, vals))} for vals in itertools.product(*(space[k] for k in keys))]

    def engine_params(self, p: dict[str, Any], fill_tier: int) -> EngineParams:
        return EngineParams(fill_tier=fill_tier, max_trades_per_day=int(p.get("max_trades", 1)),
                            expiry_bars=int(p.get("expiry_bars", 0)))

    def sizing(self, p: dict[str, Any]) -> Sizing:
        return Sizing(instrument=p.get("instrument", "NQ"), contracts=int(p.get("contracts", 1)),
                      risk_usd=p.get("risk_usd"))

    def signals(self, md, eb: EngineBars, p: dict[str, Any]) -> Signals:  # pragma: no cover
        raise NotImplementedError

    def validate(self, p: dict[str, Any]) -> None:
        return None


# ---- helpers ------------------------------------------------------------------------------
def gather(md, eb: EngineBars, frame: pl.DataFrame, column: str) -> np.ndarray:
    """Feature column (aligned to md.bars) gathered onto engine bars."""
    return frame[column].to_numpy(allow_copy=True).astype(float)[eb.idx]


def session_rng(seed: int, session_date) -> np.random.Generator:
    """Deterministic per (seed, session): truncation or permutation of OTHER sessions never
    changes a session's random draws."""
    h = hashlib.sha256(f"{seed}|{session_date}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def at_minute(eb: EngineBars, minute: int) -> np.ndarray:
    return eb.ct_minute == minute


def session_slices(eb: EngineBars) -> list[tuple[int, int, int]]:
    """(code, start, end) per session; engine bars are sorted by session."""
    codes, starts = np.unique(eb.sess, return_index=True)
    ends = list(starts[1:]) + [eb.n]
    return [(int(c), int(a), int(b)) for c, a, b in zip(codes, starts, ends)]


def first_per_session(mask: np.ndarray, sess: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    seen = set()
    for i in np.nonzero(mask)[0]:
        if sess[i] not in seen:
            seen.add(sess[i])
            out[i] = True
    return out
